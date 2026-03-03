"""WeightSender — reusable sender-side weight transfer.

Manages NCCLEngine lifecycle for one sender GPU.  Given a
``TransferSchedule`` group, creates one NCCL connection per assigned
(replica_id, tp_rank) target and sends weights sequentially through each.

Designed to be used by:
  - ArcticInference benchmark
  - DSS ``TrainingJobEngine`` (via the ``send_weights`` command)
  - Any other training framework that exposes ``(name, tensor)`` iterables

Usage::

    sender = WeightSender(
        group=schedule.groups[my_rank],
        schedule=schedule,
        master_addr="10.0.0.1",
        base_port=29500,
        device=torch.device("cuda", 0),
    )
    # first call creates NCCL connections (blocks until receivers join)
    result = sender.send(weight_iter, direct=True)
    # subsequent calls reuse connections
    result = sender.send(weight_iter, direct=True)
    sender.destroy()
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

import torch

from arctic_inference.server.weight_sync.engine import NCCLEngine
from arctic_inference.server.weight_sync.schedule import TransferGroup, TransferSchedule

logger = logging.getLogger(__name__)


class WeightSender:
    """Sender-side weight transfer for one training GPU.

    Parameters
    ----------
    group : TransferGroup
        The schedule group for this sender (contains ``targets``).
    schedule : TransferSchedule
        Full schedule (used for ``port_for_target``).
    master_addr : str
        NCCL rendezvous address (this sender's IP).
    base_port : int
        Base port; individual target ports are computed from the schedule.
    device : torch.device
        GPU device for this sender.
    bucket_size : int
        Bucket size in bytes (for bucket mode).
    """

    def __init__(
        self,
        group: TransferGroup,
        schedule: TransferSchedule,
        master_addr: str,
        base_port: int,
        device: torch.device,
        bucket_size: int = 256 * 1024 * 1024,
    ) -> None:
        self._group = group
        self._schedule = schedule
        self._master_addr = master_addr
        self._base_port = base_port
        self._device = device
        self._bucket_size = bucket_size
        self._engines: list[NCCLEngine] | None = None

    @property
    def group_id(self) -> int:
        return self._group.group_id

    @property
    def targets(self) -> list[tuple[int, int]]:
        return self._group.targets

    @property
    def ports(self) -> list[int]:
        return [
            self._schedule.port_for_target(r, t, self._base_port)
            for r, t in self._group.targets
        ]

    @property
    def connected(self) -> bool:
        return self._engines is not None

    def connect(self) -> None:
        """Create NCCL connections to all assigned targets.

        Blocks until each receiver joins the rendezvous.
        """
        if self._engines is not None:
            return

        torch.cuda.set_device(self._device)
        engines: list[NCCLEngine] = []
        for r, t in self._group.targets:
            port = self._schedule.port_for_target(r, t, self._base_port)
            logger.info(
                "WeightSender group=%d connecting to R%dT%d port=%d",
                self._group.group_id, r, t, port,
            )
            eng = NCCLEngine(
                master_addr=self._master_addr,
                master_port=port,
                rank=0,
                world_size=2,
                device=self._device,
                bucket_size=self._bucket_size,
            )
            engines.append(eng)
        self._engines = engines

    def send(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        direct: bool = False,
    ) -> dict:
        """Send weights to all assigned targets.

        Parameters
        ----------
        weights : Iterable of (name, tensor) pairs
            Full model weights (un-sharded).  For targets with multiple
            connections, the *same* weights are sent to each sequentially.
        direct : bool
            If True, use per-weight send/recv (no bucket packing).
            Only valid for BF16 TP=1 receivers.

        Returns
        -------
        dict with ``status``, ``elapsed``, ``max_target_elapsed``, etc.
        """
        if self._engines is None:
            self.connect()

        torch.cuda.set_device(self._device)
        start = time.time()
        send_fn = "send_weights_direct" if direct else "send_weights"
        max_elapsed = 0.0
        n_targets = len(self._engines)
        results_per_target: list[dict] = []

        for i, eng in enumerate(self._engines):
            weight_iter = _replayable_iter(weights) if i > 0 else weights
            r = getattr(eng, send_fn)(weight_iter)
            results_per_target.append(r)
            max_elapsed = max(max_elapsed, r.get("elapsed", 0))

        total = time.time() - start
        return {
            "status": "done",
            "group_id": self._group.group_id,
            "n_targets": n_targets,
            "max_target_elapsed": max_elapsed,
            "elapsed": total,
            "targets": results_per_target,
        }

    def destroy(self) -> None:
        """Destroy all NCCL connections."""
        if self._engines is not None:
            for eng in self._engines:
                try:
                    eng.destroy()
                except Exception:
                    pass
            self._engines = None


def _replayable_iter(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Ensure the iterable can be consumed multiple times.

    If *weights* is already a list/tuple, return it.  If it's a
    generator/iterator, materialize it first (only happens on the second
    target — the first target consumes the original).
    """
    if isinstance(weights, (list, tuple)):
        return weights
    return list(weights)
