# lamport_vllm_custom_allreduce.py
from __future__ import annotations
from typing import Optional, List
import os
import torch
import torch.distributed as dist

from vllm.logger import init_logger  # if you don't use vllm's logger, replace with std logging

try:
    import custom_ops  # our compiled extension
except Exception as e:
    custom_ops = None

logger = init_logger(__name__)

def _elem_size(dtype: torch.dtype) -> int:
    if dtype is torch.float16:  return 2
    if dtype is torch.bfloat16: return 2
    if dtype is torch.float32:  return 4
    raise RuntimeError(f"Unsupported dtype: {dtype}")

def _is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )

class LamportCustomAllreduce:
    """Minimal vLLM-like custom allreduce using Lamport oneshot kernel.
       - world sizes: 2 / 4 / 8
       - only kAllReduce (no RMSNorm/quant fusion)
    """
    _SUPPORTED_WORLD_SIZES = [2, 4, 8]

    def __init__(self,
                 group: Optional[dist.ProcessGroup],
                 device: int | str | torch.device,
                 max_elems: int = 8_192 * 1024,   # max elements per call
                 ) -> None:
        self._IS_CAPTURING = False
        self.disabled = True

        if custom_ops is None:
            logger.info("Lamport custom allreduce disabled: extension not available.")
            return

        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        if self.world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.warning("Lamport custom allreduce disabled: unsupported world size %d", self.world_size)
            return

        if isinstance(device, int): device = torch.device(f"cuda:{device}")
        elif isinstance(device, str): device = torch.device(device)
        self.device = device

        # Determine physical device IDs (to enable P2P if needed)
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            device_ids = list(range(torch.cuda.device_count()))
        self.physical_device_id = device_ids[self.device.index]

        # Optional: try to enable peer access from this device to others
        for i, pd in enumerate(device_ids):
            if i == self.device.index: continue
            try:
                with torch.cuda.device(self.device.index):
                    custom_ops.enable_peer_access_to(pd)
            except Exception:
                pass

        # --- Allocate "triple buffer" via CUDA IPC on this rank ---
        self.max_elems = max_elems
        # We'll reserve per-slot bytes = world_size * max_elems * elem_size (dtype-dependent at call time).
        # To make it dtype-agnostic at allocation time, we allocate the worst-case 4 bytes/elt (fp32).
        self.slot_bytes_reserved = self.world_size * self.max_elems * 4
        self.triple_bytes = self.slot_bytes_reserved * 3

        ptr_local_u64, handle_bytes = custom_ops.allocate_shared_buffer_and_handle(self.triple_bytes)

        # Gather handles from all ranks
        handles: List[bytes] = [None] * self.world_size
        dist.all_gather_object(handles, handle_bytes, group=group)

        # Open remote handles (including ours — ok to open self, but not required)
        remote_ptrs: List[int] = [0] * self.world_size
        for i in range(self.world_size):
            if i == self.rank:
                # Use our "owner" pointer for this rank, not an opened mapping (so we can free later)
                remote_ptrs[i] = ptr_local_u64
            else:
                remote_ptrs[i] = custom_ops.open_mem_handle(handles[i])

        # --- Build workspace pointer array (device tensor of uint64 pointers) ---
        # Structure used by kernel:
        #   [2*W + r] -> base of triple buffer of rank r
        #   [3*W]     -> pointer to header int32 array (local)
        # Indices [0 .. 2W-1] and [W .. 2W-1] are unused for oneshot lamport.
        W = self.world_size
        ws_len = 3 * W + 1
        self.workspace_ptrs = torch.empty(ws_len, dtype=torch.int64, device=self.device)
        self.workspace_ptrs.zero_()

        # fill data buffers
        for r in range(W):
            self.workspace_ptrs[2 * W + r] = remote_ptrs[r]

        # header (5 int32 values) lives locally on device
        self.header = torch.zeros(8, dtype=torch.int32, device=self.device)
        self.header[2] = 0  # flag
        self.header[3] = self.slot_bytes_reserved  # comm_size per slot in BYTES
        self.header[4] = 0  # clear_size (elements) -- kernel updates it after each run

        self.workspace_ptrs[3 * W] = self.header.data_ptr()

        # Keep for finalization
        self._owner_ptr_u64 = int(ptr_local_u64)
        self._opened_ptrs_u64 = [int(p) for i, p in enumerate(remote_ptrs) if i != self.rank]

        self.disabled = False

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled: return False
        if inp.device != self.device: return False
        inp_size_bytes = inp.numel() * inp.element_size()
        if inp_size_bytes % 16 != 0: return False
        if not _is_weak_contiguous(inp): return False
        if inp.numel() > self.max_elems:
            logger.debug("Lamport AR: input elements %d > max_elems %d", inp.numel(), self.max_elems)
            return False
        return True

    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        if self.disabled or not self.should_custom_ar(input):
            return None
        # Update header comm_size BYTES if dtype is narrower than 4 bytes (pre-reserved works fine,
        # but keeping it precise is also fine)
        slot_bytes = self.world_size * input.numel() * input.element_size()
        # We *must* keep comm_size == reserved per-slot bytes used to place triple buffers.
        # If you prefer exact per-call stride, also ensure the triple buffer was sized for that (we reserved upper bound).
        # We'll keep the reserved stride which we set in __init__, i.e., self.header[3].

        # Launch
        return custom_ops.lamport_allreduce(
            input,
            self.workspace_ptrs,
            self.world_size,
            self.rank,
            True  # trigger_completion_at_end
        )

    def close(self):
        if self.disabled: return
        # Close opened remote mappings
        for p in self._opened_ptrs_u64:
            try:
                custom_ops.close_mem_handle(p)
            except Exception:
                pass
        # Free our owned allocation
        try:
            custom_ops.device_free(self._owner_ptr_u64)
        except Exception:
            pass
        self.disabled = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

