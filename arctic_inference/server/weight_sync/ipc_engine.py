"""IPCEngine — shared-memory weight transfer for same-GPU (colocated) mode.

When training and inference processes share a physical GPU, NCCL cannot
create a communicator (two processes on the same device — see
https://github.com/NVIDIA/nccl/issues/231).

This engine uses ``/dev/shm`` (POSIX shared memory) as a rendezvous-free
transport:

  1. Sender serializes weights to a file on ``/dev/shm``.
  2. Receiver reads the file back into GPU tensors.

The file path acts as the coordination point — no TCP store, no NCCL
communicator, no IPC handle pickling.  Both processes must be on the same
host (guaranteed in colocated mode).

For small models this adds negligible overhead vs NCCL; for large models
the extra host-memory copy is still faster than a crash.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Iterable

import torch

logger = logging.getLogger(__name__)

_SHM_DIR = Path("/dev/shm")


def shm_path_for_group(group_id: int) -> Path:
    """Deterministic shared-memory path for a weight-sync group."""
    return _SHM_DIR / f"arctic_ws_group_{group_id}.pt"


def shm_ready_path_for_group(group_id: int) -> Path:
    """Sentinel file indicating the weight dump is ready to read."""
    return _SHM_DIR / f"arctic_ws_group_{group_id}.ready"


def shm_done_path_for_group(group_id: int) -> Path:
    """Sentinel file indicating the receiver finished reading."""
    return _SHM_DIR / f"arctic_ws_group_{group_id}.done"


def save_weights_to_shm(
    weights: Iterable[tuple[str, torch.Tensor]],
    group_id: int,
) -> dict:
    """Serialize model weights to shared memory.

    Called by the DeepSpeed training worker in colocated mode instead of
    pushing through NCCL.

    Returns timing/size metrics.
    """
    start = time.time()
    path = shm_path_for_group(group_id)
    ready = shm_ready_path_for_group(group_id)
    done = shm_done_path_for_group(group_id)

    for p in (path, ready, done):
        p.unlink(missing_ok=True)

    weight_list = [(n, t.cpu().contiguous()) for n, t in weights]
    n_params = len(weight_list)

    buf = io.BytesIO()
    torch.save(weight_list, buf)
    data = buf.getvalue()

    path.write_bytes(data)
    ready.touch()

    deadline = time.monotonic() + 300
    while not done.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Receiver did not consume weights within 300s (group {group_id})"
            )
        time.sleep(0.01)

    path.unlink(missing_ok=True)
    ready.unlink(missing_ok=True)
    done.unlink(missing_ok=True)

    elapsed = time.time() - start
    logger.info(
        "save_weights_to_shm: group=%d, %d params, %.1f MB, %.2fs",
        group_id, n_params, len(data) / 1e6, elapsed,
    )
    return {
        "status": "done",
        "params_sent": n_params,
        "bytes": len(data),
        "elapsed": elapsed,
    }


def load_weights_from_shm(
    group_id: int,
    timeout: float = 300,
) -> list[tuple[str, torch.Tensor]]:
    """Load model weights from shared memory.

    Called by the vLLM receiver extension in colocated mode.  Blocks
    until the sender has written the ready sentinel.

    Returns a list of ``(name, tensor)`` pairs with tensors on CPU.
    """
    ready = shm_ready_path_for_group(group_id)
    done = shm_done_path_for_group(group_id)
    path = shm_path_for_group(group_id)

    deadline = time.monotonic() + timeout
    while not ready.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Sender did not produce weights within {timeout}s (group {group_id})"
            )
        time.sleep(0.05)

    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Ready sentinel found but data file missing after {timeout}s "
                f"(group {group_id})"
            )
        time.sleep(0.05)

    data = path.read_bytes()
    weights: list[tuple[str, torch.Tensor]] = torch.load(
        io.BytesIO(data), map_location="cpu", weights_only=True,
    )

    done.touch()

    logger.info(
        "load_weights_from_shm: group=%d, %d params, %.1f MB",
        group_id, len(weights), len(data) / 1e6,
    )
    return weights


def cleanup_shm(group_id: int) -> None:
    """Remove any leftover shared-memory files for a group."""
    for p in (shm_path_for_group(group_id),
              shm_ready_path_for_group(group_id),
              shm_done_path_for_group(group_id)):
        p.unlink(missing_ok=True)
