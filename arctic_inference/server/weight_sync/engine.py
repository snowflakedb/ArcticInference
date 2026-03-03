"""NCCLEngine — pipelined point-to-point weight transfer.

Two transfer modes:

**Bucket mode** (default):
    Weights are packed into fixed-size buckets.  A dedicated ``_nccl_stream``
    overlaps the NCCL transfer of bucket *N* with packing/processing of
    bucket *N±1* on the default stream (true double buffering).

**Direct mode** (BF16 TP=1 only):
    A manifest is sent first, then each weight tensor is transferred
    individually via ``nccl.send``/``recv`` straight into the receiver's
    pre-computed parameter views — zero intermediate buffer, zero copy.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Generator, Iterable

import torch

from arctic_inference.server.weight_sync.utils import stateless_init_nccl

logger = logging.getLogger(__name__)

META_BUF_SIZE = 262_144
DEFAULT_BUCKET_SIZE = 256 * 1024 * 1024


class NCCLEngine:

    def __init__(
        self,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
        device: torch.device,
        bucket_size: int = DEFAULT_BUCKET_SIZE,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.bucket_size = bucket_size
        self.is_sender = rank == 0
        self._port = master_port

        torch.cuda.set_device(device)
        self.nccl = stateless_init_nccl(
            master_addr, master_port, rank, world_size, device,
        )

        self._nccl_stream = torch.cuda.Stream(device=device)

        self._meta_bufs = [
            torch.zeros(META_BUF_SIZE, dtype=torch.uint8, device=device),
            torch.zeros(META_BUF_SIZE, dtype=torch.uint8, device=device),
        ]
        self._data_bufs = [
            torch.zeros(bucket_size, dtype=torch.uint8, device=device),
            torch.zeros(bucket_size, dtype=torch.uint8, device=device),
        ]
        self._send_events = [torch.cuda.Event() for _ in range(2)]
        self._op_count = 0

    # ------------------------------------------------------------------
    # Bucket mode — pipelined double-buffered send / receive
    # ------------------------------------------------------------------

    @torch.no_grad()
    def send_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> dict:
        assert self.is_sender, "Only rank 0 can send"

        start = time.time()
        default = torch.cuda.current_stream(self.device)
        nccl_s = self._nccl_stream

        pack_idx = 0
        prev_idx: int | None = None
        bucket_meta: list[dict] = []
        offset = 0
        n_buckets = 0
        n_params = 0

        for name, weight in weights:
            w = weight.contiguous()
            nbytes = w.nbytes
            if nbytes > self.bucket_size:
                raise ValueError(
                    f"Weight {name} ({nbytes / 1e6:.1f} MB) exceeds "
                    f"bucket_size ({self.bucket_size / 1e6:.1f} MB)."
                )

            if offset + nbytes > self.bucket_size:
                self._flush_bucket_send(pack_idx, bucket_meta, is_last=False,
                                        prev_idx=prev_idx)
                prev_idx = pack_idx
                pack_idx = 1 - pack_idx
                bucket_meta = []
                offset = 0
                n_buckets += 1

            w_bytes = w.view(-1).view(torch.uint8)
            if w_bytes.device != self.device:
                w_bytes = w_bytes.to(self.device, non_blocking=True)
            self._data_bufs[pack_idx][offset:offset + nbytes].copy_(
                w_bytes, non_blocking=True,
            )
            bucket_meta.append({
                "name": name,
                "shape": list(weight.shape),
                "dtype": str(weight.dtype),
                "offset": offset,
                "nbytes": nbytes,
            })
            offset += nbytes
            n_params += 1

        self._flush_bucket_send(pack_idx, bucket_meta, is_last=True,
                                prev_idx=prev_idx)
        n_buckets += 1

        nccl_s.synchronize()
        elapsed = time.time() - start
        logger.info(
            "send_weights: %d params, %d buckets, %.2fs, port=%d",
            n_params, n_buckets, elapsed, self._port,
        )
        return {
            "status": "done",
            "params_sent": n_params,
            "buckets": n_buckets,
            "elapsed": elapsed,
        }

    def _flush_bucket_send(self, idx: int, meta: list[dict], *,
                           is_last: bool, prev_idx: int | None) -> None:
        default = torch.cuda.current_stream(self.device)
        nccl_s = self._nccl_stream

        if prev_idx is not None:
            default.wait_event(self._send_events[prev_idx])

        self._write_meta(idx, meta, is_last=is_last)
        pack_done = torch.cuda.Event()
        pack_done.record(default)

        nccl_s.wait_event(pack_done)
        self._send_pair(idx, nccl_s)
        self._send_events[idx].record(nccl_s)

    @torch.no_grad()
    def receive_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Pipelined receive: prefetch next bucket while caller processes current."""
        assert not self.is_sender, "Only non-zero ranks can receive"

        nccl_s = self._nccl_stream
        buf_idx = 0

        self._recv_pair(buf_idx, nccl_s)
        nccl_s.synchronize()

        while True:
            metadata = self._parse_meta(buf_idx)
            is_last = metadata["is_last"]

            if not is_last:
                self._recv_pair(1 - buf_idx, nccl_s)

            for tm in metadata["tensors"]:
                shape = torch.Size(tm["shape"])
                dtype = getattr(torch, tm["dtype"].replace("torch.", ""))
                tensor = (
                    self._data_bufs[buf_idx][tm["offset"]:tm["offset"] + tm["nbytes"]]
                    .view(dtype)
                    .reshape(shape)
                )
                yield tm["name"], tensor

            if is_last:
                break
            nccl_s.synchronize()
            buf_idx = 1 - buf_idx

    # ------------------------------------------------------------------
    # Direct mode — per-weight send/recv into parameter views
    # ------------------------------------------------------------------

    @torch.no_grad()
    def send_weights_direct(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> dict:
        """Send each weight individually — fully streaming, no materialization.

        Protocol: for each weight, send a meta header (one entry,
        ``is_last=False``) then the raw bytes.  After the last weight,
        send a sentinel meta with ``is_last=True`` and no tensor entries.

        This consumes the *weights* iterable lazily, so FSDP generators
        that call ``full_tensor()`` per parameter work without holding
        more than one full parameter in GPU memory at a time.
        """
        assert self.is_sender, "Only rank 0 can send"

        start = time.time()
        nccl_s = self._nccl_stream
        n_params = 0

        for name, weight in weights:
            w = weight.contiguous()
            w_bytes = w.view(-1).view(torch.uint8)
            if w_bytes.device != self.device:
                w_bytes = w_bytes.to(self.device, non_blocking=True)
                torch.cuda.current_stream(self.device).synchronize()

            entry = {"name": name, "shape": list(w.shape),
                     "dtype": str(w.dtype), "nbytes": w.nbytes}
            self._write_meta(0, [entry], is_last=False)
            self._send_buf(self._meta_bufs[0], nccl_s)
            self._send_buf(w_bytes, nccl_s)
            nccl_s.synchronize()
            n_params += 1

        self._write_meta(0, [], is_last=True)
        self._send_buf(self._meta_bufs[0], nccl_s)
        nccl_s.synchronize()

        elapsed = time.time() - start
        logger.info("send_weights_direct: %d params, %.2fs, port=%d",
                     n_params, elapsed, self._port)
        return {"status": "done", "params_sent": n_params, "elapsed": elapsed}

    @torch.no_grad()
    def receive_weights_direct(
        self,
        param_views: dict[str, torch.Tensor],
    ) -> dict:
        """Receive weights directly into pre-computed parameter views.

        Streaming protocol: the sender transmits a per-weight meta header
        followed by the raw bytes, repeated until a sentinel meta with
        ``is_last=True`` arrives.

        *param_views* maps safetensor key → contiguous parameter tensor.
        When the view's byte size matches, NCCL writes directly into the
        model parameter storage (true zero-copy).
        """
        assert not self.is_sender

        start = time.time()
        nccl_s = self._nccl_stream

        loaded = 0
        orphan = 0

        while True:
            self._recv_buf(self._meta_bufs[0], nccl_s)
            nccl_s.synchronize()
            metadata = self._parse_meta(0)

            if metadata["is_last"]:
                break

            for entry in metadata["tensors"]:
                name = entry["name"]
                nbytes = entry["nbytes"]
                view = param_views.get(name)
                if view is not None and view.nbytes == nbytes:
                    recv_buf = view.view(-1).view(torch.uint8)
                    self._recv_buf(recv_buf, nccl_s)
                else:
                    scratch = self._data_bufs[0][:nbytes]
                    self._recv_buf(scratch, nccl_s)
                    if view is not None:
                        nccl_s.synchronize()
                        view.copy_(
                            scratch.view(
                                getattr(torch, entry["dtype"].replace("torch.", ""))
                            ).reshape(torch.Size(entry["shape"]))
                        )
                    else:
                        orphan += 1
                loaded += 1

            nccl_s.synchronize()

        elapsed = time.time() - start
        logger.info("receive_weights_direct: %d loaded, %d orphan, %.2fs",
                     loaded, orphan, elapsed)
        return {"status": "done", "params_loaded": loaded,
                "orphan": orphan, "elapsed": elapsed}

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _send_pair(self, idx: int, stream: torch.cuda.Stream) -> None:
        self._op_count += 1
        self.nccl.send(self._meta_bufs[idx], dst=1, stream=stream)
        self.nccl.send(self._data_bufs[idx], dst=1, stream=stream)

    def _recv_pair(self, idx: int, stream: torch.cuda.Stream) -> None:
        self._op_count += 1
        self.nccl.recv(self._meta_bufs[idx], src=0, stream=stream)
        self.nccl.recv(self._data_bufs[idx], src=0, stream=stream)

    def _send_buf(self, buf: torch.Tensor, stream: torch.cuda.Stream) -> None:
        self._op_count += 1
        self.nccl.send(buf, dst=1, stream=stream)

    def _recv_buf(self, buf: torch.Tensor, stream: torch.cuda.Stream) -> None:
        self._op_count += 1
        self.nccl.recv(buf, src=0, stream=stream)

    def _write_meta(self, idx: int, entries: list[dict], *,
                    is_last: bool) -> None:
        meta_json = json.dumps(
            {"tensors": entries, "is_last": is_last},
        ).encode("utf-8")
        if len(meta_json) + 8 > META_BUF_SIZE:
            raise RuntimeError(
                f"Metadata ({len(meta_json)} B) exceeds "
                f"META_BUF_SIZE ({META_BUF_SIZE} B)"
            )
        buf = self._meta_bufs[idx]
        buf.zero_()
        len_t = torch.tensor([len(meta_json)], dtype=torch.int64)
        buf[:8].copy_(len_t.view(torch.uint8))
        json_t = torch.frombuffer(bytearray(meta_json), dtype=torch.uint8)
        buf[8:8 + len(meta_json)].copy_(json_t)

    def _parse_meta(self, idx: int) -> dict:
        buf = self._meta_bufs[idx]
        meta_len = buf[:8].cpu().view(torch.int64).item()
        meta_bytes = buf[8:8 + meta_len].cpu().numpy().tobytes()
        return json.loads(meta_bytes)

    def destroy(self) -> None:
        if getattr(self, "nccl", None) is not None:
            del self.nccl
            self.nccl = None
