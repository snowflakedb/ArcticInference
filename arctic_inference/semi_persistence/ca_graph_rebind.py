"""Dense-TP graph reuse: re-establish CustomAllreduce state after reinit so a
surviving decode graph replays without full recapture.

Measured facts (cuda_graph_recap/PROGRESS_savegraph.md, va_landscape + RECAP_PROFILE runs):
  * The decode graph's CA all-reduce kernels read peer DATA pointers at runtime
    from the CA `rank_data` device buffer. `rank_data`'s VA is STABLE across
    destroy_nccl->reinit_nccl (torch caching-allocator reuse), so the baked
    `RankData* _dp` kernel arg stays valid -- only its CONTENTS need refilling.
  * The per-peer signal-pad pointers (`meta_ptrs`) ARE baked by value into the
    kernel nodes and move on reinit -> handled separately (Part B).
  * Re-establishing the ~5800 registered graph buffers costs ~0.1s (vs ~3-5s
    forward-pass recapture).

PART A (this module): refill the new CA's rank_data after reinit by replaying
`register_graph_buffers` with the THIS-rank (handle, offset) recorded at cold
start -- no forward-pass recapture per restore.

Data-acquisition subtlety: `ops.get_graph_buffer_ipc_meta(fa)` returns the
*unregistered delta* and is DRAINED by the cold-start register_graph_buffers, so
reading it post-init yields 0. We therefore record it AS register_graph_buffers
consumes it, by patching the op around one cold-start (re)capture
(install_recorder -> capture_model -> read_recorder). That extra capture is a
one-time cold-start cost; the per-restore path only does the cheap replay.

Everything is wrapped so a failure logs and the caller falls back to recapture
rather than crashing the worker.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("semip.ca.rebind")

_SNAP_ATTR = "_semip_graph_reuse_snapshot"
_RANK_DATA_KEEP_ATTR = "_semip_rank_data_keep"
_E4_INV_ATTR = "_semip_e4_pre_inventory"  # E4 pre-freeze baked-pointer inventory
# Module-global recorder: captures the (handle, offset) that
# register_graph_buffers consumes from get_graph_buffer_ipc_meta.
_recorder: dict = {"active": False, "orig": None, "handle": None, "offset": None}


def enabled() -> bool:
    """Compatibility shim for older callers.

    The active gate now lives in ``recapture_graphs("reuse")`` policy; this
    module only performs work when its explicit preparation/rebind helpers run.
    """
    return True


def _find_ca(worker):
    """Return the worker's CustomAllreduce (ca_comm), or None."""
    try:
        from vllm.distributed import parallel_state as ps
    except Exception as e:  # noqa: BLE001
        log.warning("vllm import failed: %s", e)
        return None
    tp = getattr(ps, "_TP", None)
    comm = getattr(tp, "device_communicator", None) if tp is not None else None
    ca = getattr(comm, "ca_comm", None) if comm is not None else None
    if ca is None or getattr(ca, "disabled", True):
        return None
    return ca


def _rank_data_words(ca, n=24):
    """First n uint64 words of rank_data, as hex, for verification dumps."""
    try:
        import torch
        rd = getattr(ca, "rank_data", None)
        if rd is None:
            return None
        words = rd.view(torch.int64)[:n].cpu().tolist()
        return [hex(w & ((1 << 64) - 1)) for w in words]
    except Exception as e:  # noqa: BLE001
        return [f"err:{type(e).__name__}:{e}"]


def install_recorder() -> bool:
    """Patch ops.get_graph_buffer_ipc_meta to record the (handle, offset) that
    the next register_graph_buffers consumes. Returns True if installed."""
    try:
        from vllm import _custom_ops as ops
    except Exception as e:  # noqa: BLE001
        log.warning("install_recorder: cannot import _custom_ops: %s", e)
        return False
    if _recorder["active"]:
        return True
    orig = ops.get_graph_buffer_ipc_meta

    def _patched(fa):
        handle, offset = orig(fa)
        # Keep the richest (non-empty) capture seen while active.
        if offset:
            _recorder["handle"] = handle
            _recorder["offset"] = offset
        return handle, offset

    _recorder["orig"] = orig
    _recorder["handle"] = None
    _recorder["offset"] = None
    _recorder["active"] = True
    ops.get_graph_buffer_ipc_meta = _patched
    return True


def read_recorder():
    """Return (handle, offset) recorded since install_recorder."""
    return _recorder.get("handle"), _recorder.get("offset")


def restore_recorder() -> None:
    """Undo install_recorder."""
    if not _recorder["active"]:
        return
    try:
        from vllm import _custom_ops as ops
        if _recorder["orig"] is not None:
            ops.get_graph_buffer_ipc_meta = _recorder["orig"]
    except Exception:  # noqa: BLE001
        pass
    _recorder["active"] = False
    _recorder["orig"] = None


_kg = {"active": False, "orig": None}


def install_keepgraph_patch() -> bool:
    """Force torch.cuda.CUDAGraph to retain its topology graph (keep_graph=True)
    so raw_cuda_graph() works for node enumeration/rewrite. Must be installed
    BEFORE the graphs are (re)captured.

    keep_graph appears in BOTH CUDAGraph.__new__(cls, keep_graph=False) and the
    pybind CUDAGraph.__init__. Python constructs via __new__ THEN __init__ with the
    SAME user args -- so for vLLM's `torch.cuda.CUDAGraph()` (no args) __init__ runs
    keep_graph=default-False and is the FINAL word. VERIFIED EMPIRICALLY on this
    torch 2.11 build (see PROGRESS_reuse_B200_v24.md R4): patching __new__ ALONE
    leaves raw_cuda_graph() failing (__init__ resets keep_graph); patching __init__
    makes raw_cuda_graph() succeed. So patch __init__ (this is what the original code
    did; the __new__ detour was a wrong turn). Costs extra host memory for the kept
    topology graphs (fine here)."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if _kg["active"]:
        return True
    orig = torch.cuda.CUDAGraph.__init__

    def _init(self, *a, **k):
        k["keep_graph"] = True
        return orig(self, *a, **k)

    _kg["orig"] = orig
    _kg["active"] = True
    torch.cuda.CUDAGraph.__init__ = _init
    return True


def restore_keepgraph_patch() -> None:
    if not _kg["active"]:
        return
    try:
        import torch
        if _kg["orig"] is not None:
            torch.cuda.CUDAGraph.__init__ = _kg["orig"]
    except Exception:  # noqa: BLE001
        pass
    _kg["active"] = False
    _kg["orig"] = None


_fc = {"active": False, "orig": None}


def install_force_copy_patch() -> bool:
    """Force CustomAllreduce to take the COPY path (`registered=False`) so the
    captured graph bakes only the ONE static `buffer_ptrs[rank]` (in a memcpy
    node) instead of thousands of per-buffer RankData. Install before the
    (re)capture; restore after. The peer table for that one buffer lives in
    rank_data slot 0, which the post-reinit `register_buffer` auto-refreshes."""
    try:
        import torch
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce as _CA)
    except Exception:  # noqa: BLE001
        return False
    if _fc["active"]:
        return True
    orig_ar = _CA.all_reduce
    orig_car = _CA.custom_all_reduce

    def _forced_ar(self, inp, *, out=None, registered=False):
        _fc["ar_calls"] = _fc.get("ar_calls", 0) + 1
        return orig_ar(self, inp, out=out, registered=False)

    def _forced_car(self, input):
        # Mirror custom_all_reduce but FORCE the copy path during capture, so the
        # captured graph bakes slot-0 (the symmetric staging buffer) instead of
        # the registered per-buffer RankData peer slots (which go stale on reinit).
        _fc["car_calls"] = _fc.get("car_calls", 0) + 1
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, registered=False)  # FORCE copy path
            return torch.empty_like(input)
        return self.all_reduce(input, registered=False)

    _fc["orig"] = orig_ar
    _fc["orig_car"] = orig_car
    _fc["active"] = True
    _fc["ar_calls"] = 0
    _fc["car_calls"] = 0
    _CA.all_reduce = _forced_ar
    _CA.custom_all_reduce = _forced_car
    return True


def restore_force_copy_patch() -> None:
    if not _fc["active"]:
        return
    try:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce as _CA)
        if _fc["orig"] is not None:
            _CA.all_reduce = _fc["orig"]
        if _fc.get("orig_car") is not None:
            _CA.custom_all_reduce = _fc["orig_car"]
    except Exception:  # noqa: BLE001
        pass
    _fc["active"] = False
    _fc["orig"] = None
    _fc["orig_car"] = None


_sr = {"active": False, "orig": None}


def install_suppress_register_patch() -> bool:
    """No-op CustomAllreduce.register_graph_buffers for a keep-graph (reuse) capture.

    vLLM 0.24's CUDA-graph capture context ALWAYS calls register_graph_buffers() on exit
    (custom_all_reduce.py), re-opening per-buffer IPC handles. On the reuse-snapshot's SECOND
    capture_model() -- after the cold-start capture already registered once -- that double
    registration faults at custom_all_reduce.cuh:434 'invalid argument' on B200 (and H200; confirmed
    R1 in PROGRESS_reuse_B200_v24.md). With install_force_copy_patch active the kept graph bakes only
    the COPY path (slot-0 staging buffer), so the registered per-buffer peer table is never used at
    replay -- skipping the 2nd registration is safe and avoids the 434. Install alongside
    install_force_copy_patch (around the 2nd capture_model); restore after. The cold-start (1st)
    registration that install_recorder snapshots for rebind is untouched."""
    try:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce as _CA)
    except Exception:  # noqa: BLE001
        return False
    if _sr["active"]:
        return True
    _sr["orig"] = _CA.register_graph_buffers

    def _noop_register(self):
        _sr["skipped"] = _sr.get("skipped", 0) + 1
        return

    _sr["active"] = True
    _sr["skipped"] = 0
    _CA.register_graph_buffers = _noop_register
    return True


def restore_suppress_register_patch() -> None:
    if not _sr["active"]:
        return
    try:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce as _CA)
        if _sr["orig"] is not None:
            _CA.register_graph_buffers = _sr["orig"]
    except Exception:  # noqa: BLE001
        pass
    _sr["active"] = False
    _sr["orig"] = None


# rank_data-reuse patch (ported from v24 commit 66b3bf3): force the reinit
# CustomAllreduce to reuse the preserved cold-start rank_data tensor (same VA)
# instead of a fresh torch.empty, so rank_data_moved=False and the kept-graph _dp
# pointers remain in range. This is the fix for the tp>=2 disk-wake worker death:
# without it the rebuilt CA's rank_data can land at a new VA (allocator free-list
# dependent), dangling the reused graphs' baked all-reduce pointers.
_rd_reuse = {"active": False, "orig": None, "tensor": None, "used": False, "hit": False}


def install_rank_data_reuse_patch(tensor) -> bool:
    """Monkeypatch torch.empty so the NEXT 8MB-uint8-cuda allocation (the reinit
    CustomAllreduce rank_data alloc, custom_all_reduce.py:184-186) returns the
    preserved cold-start rank_data `tensor` instead. One-shot + tightly guarded
    (exactly 8*1024*1024 uint8 on cuda); MUST be paired with restore_rank_data_reuse_patch
    right after the CA is constructed. Returns False if no tensor to reuse.

    Safe because semip runs with VLLM_ALLREDUCE_USE_SYMM_MEM=0 and FlashInfer AR off, so
    within init_worker_distributed_environment the CA rank_data is the only 8MB-uint8-cuda
    torch.empty; the one-shot `used` flag guarantees we intercept it at most once."""
    import torch
    if _rd_reuse["active"]:
        return True
    if tensor is None:
        return False
    orig = torch.empty

    def _patched(*args, **kwargs):
        if (not _rd_reuse["used"]
                and len(args) == 1 and args[0] == 8 * 1024 * 1024
                and kwargs.get("dtype") == torch.uint8
                and str(kwargs.get("device", "")).startswith("cuda")):
            _rd_reuse["used"] = True
            _rd_reuse["hit"] = True
            return _rd_reuse["tensor"]
        return orig(*args, **kwargs)

    _rd_reuse.update(orig=orig, tensor=tensor, used=False, hit=False, active=True)
    torch.empty = _patched
    return True


def restore_rank_data_reuse_patch() -> bool:
    """Undo install_rank_data_reuse_patch. Returns True iff the reuse tensor was
    actually handed out (i.e. the rank_data alloc was intercepted)."""
    if not _rd_reuse["active"]:
        return False
    try:
        import torch
        if _rd_reuse["orig"] is not None:
            torch.empty = _rd_reuse["orig"]
    except Exception:  # noqa: BLE001
        pass
    hit = _rd_reuse["hit"]
    _rd_reuse.update(active=False, orig=None, tensor=None, used=False, hit=False)
    return hit


def store_snapshot(worker, handle, offset) -> dict:
    """Store this rank's recorded registration + old meta_ptrs on the worker so
    they survive the checkpoint and drive the post-reinit rebind."""
    ca = _find_ca(worker)
    snap = {
        "handle": handle,
        "offset": offset,
        "n_offsets": len(offset) if offset else 0,
        "meta_ptrs": list(getattr(ca, "meta_ptrs", []) or []) if ca else [],
        "buffer_ptrs": list(getattr(ca, "buffer_ptrs", []) or []) if ca else [],
        "rank_data": (int(ca.rank_data.data_ptr())
                      if (ca is not None and getattr(ca, "rank_data", None) is not None)
                      else 0),
        "rank": getattr(ca, "rank", None) if ca else None,
        "world_size": getattr(ca, "world_size", None) if ca else None,
    }
    setattr(worker, _SNAP_ATTR, snap)
    # Hold a LIVE ref to the cold-start rank_data tensor so CA teardown at reinit does
    # NOT return its 8MB caching-allocator block to the pool. On reinit the new CA
    # reuses this exact tensor (install_rank_data_reuse_patch), so rank_data's VA is
    # identical across reinit (rank_data_moved=False) and the kept-graph _dp pointers
    # stay in range. Without this, models whose reinit free-list differs from cold-start
    # (observed: 397B tp8 hybrid) get a NEW rank_data VA -> "no CA _dp slots" -> fallback.
    try:
        setattr(worker, _RANK_DATA_KEEP_ATTR,
                ca.rank_data if (ca is not None
                                 and getattr(ca, "rank_data", None) is not None)
                else None)
    except Exception:  # noqa: BLE001
        pass
    result = {
        "ok": True,
        "n_offsets": snap["n_offsets"],
        "meta_ptrs": [hex(x) for x in snap["meta_ptrs"]],
        "rank": snap["rank"],
        "force_copy_ar_calls": _fc.get("ar_calls", 0),
        "force_copy_car_calls": _fc.get("car_calls", 0),
    }
    # Phase-1 SP feasibility probe (Approach 2): dump the NCCL kernel nodes the kept
    # graph bakes, and PERSIST to a file -- so the data survives regardless of log
    # or return-dict propagation across the SP-worker (collective_rpc) boundary.
    # Driven from store_snapshot because it demonstrably runs current code in the
    # SP worker. Read-only (modifies no graph).
    try:
        import json as _json
        import sys as _sys
        sp_dump = dump_sp_nccl_nodes(worker)
        sp_dump["_loaded_files"] = {
            "ca_graph_rebind": getattr(_sys.modules.get("ca_graph_rebind"),
                                       "__file__", None),
            "vllm_child": getattr(_sys.modules.get("vllm_child"),
                                  "__file__", None),
        }
        with open(f"/tmp/sp_graph_dump.{os.getpid()}.json", "w") as _f:
            _json.dump(sp_dump, _f, indent=2, default=str)
        result["sp_graph_dump"] = {k: sp_dump.get(k) for k in
                                   ("n_graphs", "n_kernel_nodes", "n_nccl_nodes",
                                    "nccl_kernels", "top_arg0", "_loaded_files")}
    except Exception as _e:  # noqa: BLE001
        result["sp_graph_dump_error"] = f"{type(_e).__name__}: {_e}"
    # E4 PRE inventory (SEMIP_CA_E4_INVENTORY=1): snapshot every graph-baked size==8
    # device pointer while the kept graph is known-good (this runs after cold capture,
    # before any freeze -- codex resume8 confirmed store_snapshot as the correct
    # last-known-good pre-hook, NOT pre-lock checkpoint_cuda). Stored on the worker for
    # the post-reinit reclassify in rebind_after_reinit.
    if _ca_e4_inventory_enabled():
        result["e4_pre"] = e4_pre_inventory(worker)
    return result


def _replay_register_graph_buffers(ca, my_handle, my_offset) -> dict:
    """Replay register_graph_buffers using the snapshotted local meta, without
    re-running capture. Refills the (stable-VA) rank_data of the freshly
    reinitialized CA. Mirrors CustomAllreduce.register_graph_buffers but uses
    the snapshotted (handle, offset) for this rank.
    """
    import torch.distributed as dist
    from vllm import _custom_ops as ops

    group = ca.group
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    all_data = [[None, None] for _ in range(world_size)]
    all_data[rank] = [my_handle, my_offset]
    ranks = sorted(dist.get_process_group_ranks(group=group))
    for i, r in enumerate(ranks):
        dist.broadcast_object_list(all_data[i], src=r, group=group, device="cpu")
    handles = [d[0] for d in all_data]
    offsets = [d[1] for d in all_data]
    ops.register_graph_buffers(ca._ptr, handles, offsets)
    return {"n_offsets_self": len(my_offset) if my_offset else 0,
            "n_ranks": world_size}


# ---------------------------------------------------------------------------
# PART B: rewrite the baked meta_ptrs in the captured graph execs.
# The per-peer signal-pad pointers (meta_ptrs) are baked by value into the CA
# all-reduce kernel nodes (RankSignals sg, Signal* self_sg) and move on reinit.
# We walk each captured graph, identify CA kernels BY NAME (cuFuncGetName ->
# "cross_device_reduce*") so we never deref a non-CA node's params, then
# value-match old->new meta_ptrs in sg/self_sg and commit with
# cuGraphExecKernelNodeSetParams. No vLLM CustomAllreduce source is touched.
# ---------------------------------------------------------------------------
import ctypes  # noqa: E402

_CU = None
_CU_NODE_TYPE_KERNEL = 0
_CU_NODE_TYPE_MEMCPY = 1
# E8 (resume12 Q2/Q3): the full CUgraphNodeType enum so we can histogram which
# node classes a preserved FULL graph carries -- E4 only inventoried KERNEL(0) +
# MEMCPY(1) params; codex flagged MEMSET(2), MEM_ALLOC(10)/MEM_FREE(11) (graph-
# owned mempool), and GRAPH(4) child graphs (E4 never recursed) as the untested
# node classes that could still bake a VA that moved across reinit.
_CU_NODE_TYPE_MEMSET = 2
_CU_NODE_TYPE_HOST = 3
_CU_NODE_TYPE_GRAPH = 4          # child graph
_CU_NODE_TYPE_EMPTY = 5
_CU_NODE_TYPE_WAIT_EVENT = 6
_CU_NODE_TYPE_EVENT_RECORD = 7
_CU_NODE_TYPE_EXT_SEMAS_SIGNAL = 8
_CU_NODE_TYPE_EXT_SEMAS_WAIT = 9
_CU_NODE_TYPE_MEM_ALLOC = 10
_CU_NODE_TYPE_MEM_FREE = 11
_CU_NODE_TYPE_BATCH_MEM_OP = 12
_CU_NODE_TYPE_CONDITIONAL = 13
_CU_NODE_TYPE_NAMES = {
    0: "kernel", 1: "memcpy", 2: "memset", 3: "host", 4: "child_graph",
    5: "empty", 6: "wait_event", 7: "event_record", 8: "ext_sem_signal",
    9: "ext_sem_wait", 10: "mem_alloc", 11: "mem_free", 12: "batch_mem_op",
    13: "conditional",
}


class _KernelNodeParams(ctypes.Structure):
    # CUDA_KERNEL_NODE_PARAMS_v2 (CUDA 12.x)
    _fields_ = [
        ("func", ctypes.c_void_p),
        ("gridDimX", ctypes.c_uint), ("gridDimY", ctypes.c_uint),
        ("gridDimZ", ctypes.c_uint),
        ("blockDimX", ctypes.c_uint), ("blockDimY", ctypes.c_uint),
        ("blockDimZ", ctypes.c_uint),
        ("sharedMemBytes", ctypes.c_uint),
        ("kernelParams", ctypes.POINTER(ctypes.c_void_p)),
        ("extra", ctypes.POINTER(ctypes.c_void_p)),
        ("kern", ctypes.c_void_p),
        ("ctx", ctypes.c_void_p),
    ]


class _Memcpy3D(ctypes.Structure):
    # CUDA_MEMCPY3D_v2 (driver). We only read/write srcDevice/dstDevice; the rest
    # is here for correct struct SIZE so Get/SetParams round-trip cleanly.
    _fields_ = [
        ("srcXInBytes", ctypes.c_size_t), ("srcY", ctypes.c_size_t),
        ("srcZ", ctypes.c_size_t), ("srcLOD", ctypes.c_size_t),
        ("srcMemoryType", ctypes.c_uint),
        ("srcHost", ctypes.c_void_p),
        ("srcDevice", ctypes.c_uint64),
        ("srcArray", ctypes.c_void_p),
        ("reserved0", ctypes.c_void_p),
        ("srcPitch", ctypes.c_size_t), ("srcHeight", ctypes.c_size_t),
        ("dstXInBytes", ctypes.c_size_t), ("dstY", ctypes.c_size_t),
        ("dstZ", ctypes.c_size_t), ("dstLOD", ctypes.c_size_t),
        ("dstMemoryType", ctypes.c_uint),
        ("dstHost", ctypes.c_void_p),
        ("dstDevice", ctypes.c_uint64),
        ("dstArray", ctypes.c_void_p),
        ("reserved1", ctypes.c_void_p),
        ("dstPitch", ctypes.c_size_t), ("dstHeight", ctypes.c_size_t),
        ("WidthInBytes", ctypes.c_size_t), ("Height", ctypes.c_size_t),
        ("Depth", ctypes.c_size_t),
    ]


class _MemsetNodeParams(ctypes.Structure):
    # CUDA_MEMSET_NODE_PARAMS (driver). E8 reads only `dst` (the baked target VA);
    # the rest is present for correct struct size so GetParams round-trips.
    _fields_ = [
        ("dst", ctypes.c_uint64),
        ("pitch", ctypes.c_size_t),
        ("value", ctypes.c_uint),
        ("elementSize", ctypes.c_uint),
        ("width", ctypes.c_size_t),
        ("height", ctypes.c_size_t),
    ]


def _cu():
    global _CU
    if _CU is not None:
        return _CU
    cu = ctypes.CDLL("libcuda.so.1")
    cu.cuGraphGetNodes.argtypes = [ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.POINTER(ctypes.c_size_t)]
    cu.cuGraphNodeGetType.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    cu.cuGraphKernelNodeGetParams_v2.argtypes = [ctypes.c_void_p,
                                                 ctypes.POINTER(_KernelNodeParams)]
    cu.cuFuncGetName.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_void_p]
    cu.cuGraphExecKernelNodeSetParams_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                     ctypes.POINTER(_KernelNodeParams)]
    cu.cuGraphMemcpyNodeGetParams.argtypes = [ctypes.c_void_p,
                                              ctypes.POINTER(_Memcpy3D)]
    cu.cuGraphExecMemcpyNodeSetParams.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                  ctypes.POINTER(_Memcpy3D),
                                                  ctypes.c_void_p]
    cu.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    # Topology (non-exec) setters: patch the kept cudaGraph_t so graphs that are
    # (re-)instantiated AFTER rebind are born with the new addresses.
    cu.cuGraphKernelNodeSetParams_v2.argtypes = [ctypes.c_void_p,
                                                 ctypes.POINTER(_KernelNodeParams)]
    cu.cuGraphMemcpyNodeSetParams.argtypes = [ctypes.c_void_p,
                                              ctypes.POINTER(_Memcpy3D)]
    # Signal-pad probe (SEMIP_CA_PAD_PROBE / SEMIP_CA_ZERO_SIGNAL): read the local
    # CA Signal header D->H and (optionally) zero it. CUdeviceptr is 64-bit.
    cu.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                   ctypes.c_size_t]
    cu.cuMemsetD8_v2.argtypes = [ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t]
    # E3 (SEMIP_CA_E3_DUMP): classify graph-baked activation/result pointers.
    # cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_MEMORY_TYPE=2) is an independent
    # validity/type check alongside cuMemGetAddressRange_v2 (bound in _cu_ipc).
    cu.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                         ctypes.c_uint64]
    # E4 (SEMIP_CA_E4_INVENTORY): per-function parameter reflection. cuFuncGetParamCount
    # is absent on driver 13000, so iterate cuFuncGetParamInfo(func, i, &off, &sz) until it
    # returns CUDA_ERROR_INVALID_VALUE -> yields both the param count and each (offset,size).
    cu.cuFuncGetParamInfo.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                      ctypes.POINTER(ctypes.c_size_t),
                                      ctypes.POINTER(ctypes.c_size_t)]
    # E5 (SEMIP_CA_E5_REINSTANTIATE): build a FRESH cudaGraphExec_t from the kept,
    # already-rebound cudaGraph_t (cuGraphInstantiateWithFlags), optionally upload
    # it, and launch it in place of the preserved exec (cuGraphLaunch). Tests whether
    # the stale INSTANTIATED exec (device-side launch/scheduling state that survives
    # checkpoint_cuda but is invalidated by reinit) is the IMA source. Additive
    # argtypes only; symbols exist on driver 13000.
    cu.cuGraphInstantiateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                               ctypes.c_void_p, ctypes.c_uint64]
    cu.cuGraphUpload.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cu.cuGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    # E8 (SEMIP_E8_NODE_INV): read the baked target of the non-kernel/non-memcpy
    # node classes E4 skipped. MEMSET.dst and MEM_FREE.dptr are single device VAs
    # we classify against the live map; child graphs are recursed. MEM_ALLOC's
    # struct is nested/complex -> we only COUNT those (presence alone is decisive:
    # a graph-owned mempool node re-reserves VA at instantiate time). Best-effort:
    # some symbols may be absent on this driver -> guarded at call sites.
    for _sym in ("cuGraphMemsetNodeGetParams", "cuGraphMemsetNodeGetParams_v2"):
        try:
            getattr(cu, _sym).argtypes = [ctypes.c_void_p,
                                          ctypes.POINTER(_MemsetNodeParams)]
        except AttributeError:
            pass
    try:
        cu.cuGraphMemFreeNodeGetParams.argtypes = [ctypes.c_void_p,
                                                   ctypes.POINTER(ctypes.c_uint64)]
    except AttributeError:
        pass
    try:
        cu.cuGraphChildGraphNodeGetGraph.argtypes = [ctypes.c_void_p,
                                                     ctypes.POINTER(ctypes.c_void_p)]
    except AttributeError:
        pass
    _CU = cu
    return cu


def _ck(name, code):
    if code != 0:
        raise RuntimeError(f"{name} failed: CUresult={code}")


def _graph_nodes(cu, graph):
    num = ctypes.c_size_t(0)
    _ck("cuGraphGetNodes(count)", cu.cuGraphGetNodes(graph, None, ctypes.byref(num)))
    n = num.value
    if n == 0:
        return []
    arr = (ctypes.c_void_p * n)()
    _ck("cuGraphGetNodes", cu.cuGraphGetNodes(graph, arr, ctypes.byref(num)))
    return [arr[i] for i in range(n)]


def _func_name(cu, func):
    namep = ctypes.c_char_p()
    code = cu.cuFuncGetName(ctypes.byref(namep), func)
    if code != 0 or not namep.value:
        return ""
    try:
        return namep.value.decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""


def _graph_manager_holders(worker):
    """The objects that own a {desc: torch.cuda.CUDAGraph} `graphs` dict, i.e. the
    FULL-cudagraph managers. The dense runner's is the one that matters; a spec-decode
    speculator carries its own and its FULL graphs bake the same CA pointers, so
    include it too. Returns (managers, why) -- `why` explains an empty result so a log
    can tell "V1 runner, nothing to find" apart from "V2 runner, plumbing broken".

    ONLY THE V2 RUNNER HAS ONE. vLLM 0.24 picks the runner per model
    (`gpu_worker.py:355` on `vllm_config.use_v2_model_runner`, an ARCHITECTURE
    ALLOWLIST -- `config/vllm.py:550` DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES =
    {Qwen3ForCausalLM, DeepseekV2ForCausalLM, Qwen2MoeForCausalLM,
    GraniteMoeForCausalLM, LlamaForCausalLM, MistralForCausalLM}). V2
    (`v1/worker/gpu/model_runner.py:466`) always builds a ModelCudaGraphManager; V1
    (`v1/worker/gpu_model_runner.py`) has no decode manager at all -- its only
    `*cudagraph_manager` is `encoder_cudagraph_manager` for multimodal encoders, and
    its FULL graphs go through CUDAGraphWrapper, so source 1 already covers them.

    So `n_managers == 0` is EXPECTED on a V1 model (any Qwen3Moe / Qwen2 / Qwen3_5 /
    GLM / DeepseekV4 / gpt-oss ...) and is a BUG only on a V2 model. Record
    `runner_class` in diag rather than inferring it: an empty result is otherwise
    indistinguishable between the two, which is what made the first read of the
    Qwen3-235B-A22B-tp4 (Qwen3MoeForCausalLM -> V1) validation run wrong."""
    out = []
    mr = getattr(worker, "model_runner", None)
    if worker is None:
        return out, "worker is None (caller did not pass it)"
    if mr is None:
        return out, f"{type(worker).__name__} has no .model_runner"
    # 0.24 names the spec-decode holder `speculator` (model_runner.py:188/193);
    # `drafter` is kept for older/newer spellings, hence both.
    for attr, holder in (("model_runner", mr),
                         ("speculator", getattr(mr, "speculator", None)),
                         ("drafter", getattr(mr, "drafter", None))):
        if holder is None:
            continue
        mgr = getattr(holder, "cudagraph_manager", None)
        if mgr is not None and not any(m is mgr for m in out):
            out.append(mgr)
    if out:
        return out, None
    return out, (f"{type(mr).__name__} has no cudagraph_manager -- expected on the "
                 f"V1 runner, where FULL graphs live in CUDAGraphWrapper (source 1)")


def _find_captured_graphs(worker=None):
    """Collect (cudaGraph_t, cudaGraphExec_t_or_0) for every captured
    torch.cuda.CUDAGraph vLLM 0.24 holds.

    A gc.get_objects() scan returns 0 on 0.24: vLLM gc.freeze()s the worker heap
    after capture (gpu_worker.py freeze_gc_heap; comment literally lists "CUDA
    graphs"), moving the graphs into CPython's permanent generation which
    gc.get_objects() does NOT return. So enumerate the two places vLLM itself keeps
    them, both freeze-immune:

      1. the WRAPPER REGISTRIES -- CUDAGraphWrapper._all_instances (+
         BreakableCUDAGraphWrapper) -> .concrete_cudagraph_entries[*].cudagraph
         (cuda_graph.py). Take ALL wrappers, not just runtime_mode=FULL ones: the TP
         custom-all-reduce kernels live in the piecewise regions too.
      2. the GRAPH MANAGER -- worker.model_runner.cudagraph_manager.graphs, a plain
         {BatchExecutionDescriptor: torch.cuda.CUDAGraph}.

    Source 2 is here because omitting it WAS the reuse-graph bug (resume18, codex
    gpt-5.5 consensus 2026-08-01), on the model runners where it applies. Which runner
    you get is decided per-architecture by vllm/config/vllm.py:522 use_v2_model_runner
    against DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES (vllm.py:68), and that gate is also
    the answer to why reuse "sometimes" worked:

      * arch NOT in that set -> V1 runner (gpu_model_runner.py). With FULL_AND_PIECEWISE
        and breakable cudagraphs off, :5317 wraps the model in
        CUDAGraphWrapper(runtime_mode=FULL), so source 1 already covers the FULL graphs.
        e.g. Qwen3-235B-A22B (Qwen3MoeForCausalLM, 94 layers): n_wrappers=96,
        n_entries=4896=96x51, i.e. 95 piecewise pieces + 1 FULL wrapper. This model never
        faulted on ckpt-wake -- its FULL graphs were always patched.
      * arch IN that set -> V2 runner (gpu/model_runner.py:466). The FULL graphs are bare
        torch.cuda.CUDAGraph objects in mgr.graphs[desc] (gpu/cudagraph_utils.py:316/327)
        replayed via self.graphs[desc].replay() (:373); no CUDAGraphWrapper is ever
        constructed, so they are in NO registry and source 1 misses every one of them.
        e.g. Qwen3-32B (Qwen3ForCausalLM, 64 layers): n_wrappers=65, n_entries=3315=65x51,
        i.e. 65 piecewise pieces and NO FULL wrapper. Across a CA address move those FULL
        graphs kept freed VAs -> async illegal-memory-access on the first FULL decode
        replay. That is the observed fault
        (agent_run_log/ep/wakepath_qwen3-32b-tp4-dense_20260731-203342.log).

    The capture progress-bar label is an independent tell for which runner ran: only V1's
    gpu_model_runner.py:6718 emits the two-part "Capturing CUDA graphs ({}, {})", so
    "(decode, FULL)" means V1 and a bare "(FULL)" means V2.

    The gap also made E4's "pre_valid_post_invalid=0" true but vacuous on V2 models: E4
    used this same enumerator, so it inventoried only the set that had just been patched.

    `worker` is optional so the few diagnostic callers that have no worker still
    work, but passing None silently reverts to the piecewise-only bug -- hence
    diag["src"] records which sources actually contributed.

    Topology needs keep_graph=True (install_keepgraph_patch, on __new__); exec may
    be 0 for a shape captured-but-not-yet-instantiated -- we patch topology for all
    and exec for the instantiated ones (later instantiate() inherits the patched
    topology). Deduped by graph handle as well as by object identity: a handle
    reachable from both sources must be patched once, since applying an old->new
    value map twice can misfire if a new address is also an old key.
    Preserves the (pairs, diag) contract of the old gc-scan version; diag carries
    the manager handle set under the private key "_manager_handles" so callers can
    split their counters by source (pop it before logging -- it is not a metric)."""
    pairs = []
    diag = {"n_wrappers": 0, "n_entries": 0, "n_cudagraph_objs": 0,
            "n_topo_ok": 0, "n_exec_ok": 0, "n_topo_fail": 0,
            "n_uninstantiated": 0, "n_managers": 0, "n_manager_graphs": 0,
            "n_dup_handles": 0, "src": None, "err": None,
            "runner_class": None, "mgr_why": None}
    try:
        import torch
        cls = torch.cuda.CUDAGraph
        wrappers = []
        try:
            from vllm.compilation.cuda_graph import CUDAGraphWrapper
            wrappers += list(CUDAGraphWrapper._all_instances)
        except Exception as ex:  # noqa: BLE001
            diag["err"] = f"import CUDAGraphWrapper: {type(ex).__name__}: {ex}"
        try:
            from vllm.compilation.breakable_cudagraph import (
                BreakableCUDAGraphWrapper)
            wrappers += list(BreakableCUDAGraphWrapper._all_instances)
        except Exception:  # noqa: BLE001
            pass
        diag["n_wrappers"] = len(wrappers)
        diag["src"] = "vllm_wrappers"
        seen = set()
        cgs = []
        for w in wrappers:
            # graphs live in .concrete_cudagraph_entries (CUDAGraphWrapper) or
            # .entries (BreakableCUDAGraphWrapper); duck-type CUDAGraph out of
            # each entry's attrs (robust to attr renames / segment lists).
            for attr in ("concrete_cudagraph_entries", "entries"):
                m = getattr(w, attr, None)
                if not isinstance(m, dict):
                    continue
                for entry in m.values():
                    diag["n_entries"] += 1
                    vals = (list(vars(entry).values())
                            if hasattr(entry, "__dict__") else [])
                    for v in vals:
                        if isinstance(v, cls):
                            if id(v) not in seen:
                                seen.add(id(v)); cgs.append(v)
                        elif isinstance(v, (list, tuple)):
                            for it in v:
                                if isinstance(it, cls) and id(it) not in seen:
                                    seen.add(id(it)); cgs.append(it)
        # Source 2: the FULL graphs, which live in no registry (see docstring).
        # V2 runner only -- on V1 there is no manager and source 1 already has them.
        mgr_ids = set()
        _mr = getattr(worker, "model_runner", None)
        diag["runner_class"] = type(_mr).__name__ if _mr is not None else None
        managers, diag["mgr_why"] = _graph_manager_holders(worker)
        for mgr in managers:
            graphs = getattr(mgr, "graphs", None)
            if not isinstance(graphs, dict):
                diag["mgr_why"] = (f"{type(mgr).__name__}.graphs is "
                                   f"{type(graphs).__name__}, not a dict")
                continue
            diag["n_managers"] += 1
            for g in graphs.values():
                if not isinstance(g, cls) or id(g) in seen:
                    continue
                seen.add(id(g)); cgs.append(g); mgr_ids.add(id(g))
                diag["n_manager_graphs"] += 1
        if diag["n_managers"]:
            diag["src"] = "vllm_wrappers+manager" if wrappers else "manager"
        # Fallback (registries unexpectedly empty): unfreeze the heap, gc scan,
        # then re-freeze -- so we still find graphs if the wrapper path changes.
        if not cgs:
            import gc
            gc.unfreeze()
            try:
                for o in gc.get_objects():
                    try:
                        ok = isinstance(o, cls)
                    except Exception:  # noqa: BLE001
                        ok = False
                    if ok and id(o) not in seen:
                        seen.add(id(o)); cgs.append(o)
            finally:
                gc.freeze()
            diag["src"] = "gc_unfreeze_fallback"
        emitted = set()
        mgr_handles = []
        for o in cgs:
            diag["n_cudagraph_objs"] += 1
            try:
                g = int(o.raw_cuda_graph())
            except Exception as ex:  # noqa: BLE001 - no topology (keep_graph False)
                diag["n_topo_fail"] += 1
                if diag["err"] is None:
                    diag["err"] = f"topo: {type(ex).__name__}: {ex}"
                continue
            if g in emitted:
                # Same graph reached from both sources -- patch it once (see docstring).
                diag["n_dup_handles"] += 1
                continue
            emitted.add(g)
            e = 0
            try:
                e = int(o.raw_cuda_graph_exec())
            except Exception:  # noqa: BLE001 - captured but not yet instantiated
                diag["n_uninstantiated"] += 1
            diag["n_topo_ok"] += 1
            if e:
                diag["n_exec_ok"] += 1
                # E7 (resume11 Q3): remember which graph objects already had a live
                # exec pre-rebind, so a FULL replay hook can tag preserved vs fresh.
                if _reuse_diag_enabled():
                    _PRESERVED_EXEC_IDS.add(id(o))
            if id(o) in mgr_ids:
                mgr_handles.append(g)
            pairs.append((g, e))
        diag["_manager_handles"] = frozenset(mgr_handles)
    except Exception as e:  # noqa: BLE001
        diag["err"] = f"{type(e).__name__}: {e}"
        log.warning("Part B: _find_captured_graphs failed: %s", e)
    log.info("Part B _find_captured_graphs: %s",
             {k: v for k, v in diag.items() if not k.startswith("_")})
    return pairs, diag


def dump_sp_nccl_nodes(worker, max_samples=48, n_words=8):
    """Phase-1 feasibility probe for SP keep-graph (Approach 2): walk the captured
    graphs, find NCCL kernel nodes (the Ulysses all-to-all + any pynccl all-reduce
    that fell back from CA during shift capture) and dump what each bakes into its
    FIRST kernel arg -- so we can judge whether the comm/device state is a small,
    rewritable set (-> build the ncclCommAbort+rebind of Approach 2) or too opaque
    (-> fall back to Approach 3, SP uses recapture).

    READ-ONLY: modifies no graph. We deref only kernelParams[0] (always present
    for a kernel) for a few words; we never walk kp[1..] (unknown NCCL arg count ->
    out-of-bounds). Logs detail to the worker log; returns a compact summary."""
    summary = {"n_graphs": 0, "n_kernel_nodes": 0, "n_nccl_nodes": 0,
               "nccl_kernels": {}, "n_samples": 0, "top_arg0": [], "err": None}
    try:
        cu = _cu()
        pairs, _diag = _find_captured_graphs(worker)
        summary["n_graphs"] = len(pairs)
        arg0_hist = {}
        samples = []
        for gi, (g, _e) in enumerate(pairs):
            if not g:
                continue
            for ni, node in enumerate(_graph_nodes(cu, g)):
                t = ctypes.c_int(-1)
                if cu.cuGraphNodeGetType(node, ctypes.byref(t)) != 0:
                    continue
                if t.value != _CU_NODE_TYPE_KERNEL:
                    continue
                summary["n_kernel_nodes"] += 1
                params = _KernelNodeParams()
                if cu.cuGraphKernelNodeGetParams_v2(node, ctypes.byref(params)) != 0:
                    continue
                name = _func_name(cu, params.func)
                low = name.lower()
                if ("nccl" not in low and "all_to_all" not in low
                        and "alltoall" not in low and "sendrecv" not in low):
                    continue
                summary["n_nccl_nodes"] += 1
                summary["nccl_kernels"][name] = \
                    summary["nccl_kernels"].get(name, 0) + 1
                kp = params.kernelParams
                words = []
                if kp:
                    arg0 = kp[0]               # first kernel-arg storage (host)
                    if arg0:
                        try:
                            buf = (ctypes.c_uint64 * n_words).from_address(arg0)
                            words = [int(buf[i]) for i in range(n_words)]
                        except Exception:      # noqa: BLE001
                            words = []
                if words:
                    arg0_hist[words[0]] = arg0_hist.get(words[0], 0) + 1
                if len(samples) < max_samples:
                    samples.append({"g": gi, "n": ni, "name": name,
                                    "grid": (params.gridDimX, params.gridDimY,
                                             params.gridDimZ),
                                    "arg0_words": [hex(w) for w in words]})
        summary["n_samples"] = len(samples)
        top = sorted(arg0_hist.items(), key=lambda kv: -kv[1])[:16]
        summary["top_arg0"] = [{"addr": hex(a), "count": c} for a, c in top]
        log.info("SP_GRAPH_DUMP: n_graphs=%d n_kernel_nodes=%d n_nccl_nodes=%d "
                 "kernels=%s", summary["n_graphs"], summary["n_kernel_nodes"],
                 summary["n_nccl_nodes"], summary["nccl_kernels"])
        log.info("SP_GRAPH_DUMP: top_arg0(addr:count)=%s", summary["top_arg0"])
        for s in samples:
            log.info("SP_GRAPH_DUMP node g%d n%d %s grid=%s arg0=%s",
                     s["g"], s["n"], s["name"], s["grid"], s["arg0_words"])
    except Exception as ex:  # noqa: BLE001
        summary["err"] = f"{type(ex).__name__}: {ex}"
        log.warning("dump_sp_nccl_nodes failed: %s", ex)
    return summary


def _patch_node(cu, exec_handle, node, meta_map, vfy=None):
    """If `node` is a CA all-reduce kernel, value-match old->new meta_ptrs in its
    sg (kernelParams[1], 8 ptrs) and self_sg (kernelParams[2], 1 ptr) and commit.
    Returns the number of pointer slots rewritten (0 if not a CA node / no match).

    `vfy`: optional counter dict for the transactional topology readback at the end
    of this function (keys k_topo_ok / k_topo_bad / topo_readback_err). None skips it.
    """
    t = ctypes.c_int(-1)
    if cu.cuGraphNodeGetType(node, ctypes.byref(t)) != 0:
        return 0
    if t.value != _CU_NODE_TYPE_KERNEL:
        return 0
    params = _KernelNodeParams()
    if cu.cuGraphKernelNodeGetParams_v2(node, ctypes.byref(params)) != 0:
        return 0
    name = _func_name(cu, params.func)
    if "cross_device_reduce" not in name:
        return 0
    kp = params.kernelParams
    if not kp:
        return 0
    # CA kernel signature: (RankData* _dp, RankSignals sg, Signal* self_sg,
    #                       T* result, int rank, int size) -> 6 args.
    dp_addr = kp[0]
    sg_addr = kp[1]
    self_addr = kp[2]
    if not sg_addr or not self_addr:
        return 0
    rewrites = 0
    # _dp (kernelParams[0]) is a pointer INTO rank_data (the per-buffer RankData
    # slot). Rewrite if rank_data moved across reinit.
    new_dp = None
    if dp_addr:
        dp_cur = int((ctypes.c_uint64 * 1).from_address(dp_addr)[0])
        mapped = meta_map.get(dp_cur, dp_cur)
        if mapped != dp_cur:
            new_dp = mapped
            rewrites += 1
    sg = (ctypes.c_uint64 * 8).from_address(sg_addr)
    self_sg = (ctypes.c_uint64 * 1).from_address(self_addr)
    new_sg_vals = []
    # Snapshot the pre-patch values: `sg` is a view onto the driver's arg storage,
    # so after SetParams below it is no longer a reliable source for "what did this
    # slot hold before?" -- which the readback needs to know which slots to verify.
    sg_before = []
    for i in range(8):
        v = int(sg[i])
        sg_before.append(v)
        nv = meta_map.get(v, v)
        new_sg_vals.append(nv)
        if nv != v:
            rewrites += 1
    sv = int(self_sg[0])
    new_self = meta_map.get(sv, sv)
    if new_self != sv:
        rewrites += 1
    if rewrites == 0:
        return 0
    # Commit with fresh backing buffers (don't mutate driver-owned arg storage).
    new_sg_buf = (ctypes.c_uint64 * 8)(*new_sg_vals)
    new_self_buf = (ctypes.c_uint64 * 1)(new_self)
    dp_slot = kp[0]
    new_dp_buf = None
    if new_dp is not None:
        new_dp_buf = (ctypes.c_uint64 * 1)(new_dp)
        dp_slot = ctypes.cast(new_dp_buf, ctypes.c_void_p)
    new_kp = (ctypes.c_void_p * 6)(
        dp_slot, ctypes.cast(new_sg_buf, ctypes.c_void_p),
        ctypes.cast(new_self_buf, ctypes.c_void_p), kp[3], kp[4], kp[5])
    params.kernelParams = ctypes.cast(new_kp, ctypes.POINTER(ctypes.c_void_p))
    # Topology set (always): so graphs (re-)instantiated AFTER rebind are correct.
    _ck("cuGraphKernelNodeSetParams_v2",
        cu.cuGraphKernelNodeSetParams_v2(node, ctypes.byref(params)))
    # Exec set (only if this graph is already instantiated): fix the live exec.
    if exec_handle:
        _ck("cuGraphExecKernelNodeSetParams_v2",
            cu.cuGraphExecKernelNodeSetParams_v2(exec_handle, node,
                                                 ctypes.byref(params)))
    # Transactional TOPOLOGY readback (codex consensus, E14): prove per node that
    # the commit took, by reading the slot back and comparing it to the value we
    # meant to write. This replaces "does any slot still hold an OLD address?" as
    # the primary confirm-not-lucky counter, because that question is unanswerable
    # on a 2nd-or-later wake: CA reallocates into the VAs the previous wake freed,
    # so addr_map becomes a permutation whose value set overlaps its key set, and a
    # CORRECTLY rewritten slot then holds a value that is also some other rank's old
    # address. (E14 measured exactly that: 4 of 8 rebind events reported tens of
    # thousands of "stale" hits with the rewrite provably complete -- see
    # agent_run_log/e14_v2arch_v24/audit_falsepos_proof.py.)
    #
    # Only the slots actually rewritten are counted, so
    #     k_topo_ok + k_topo_bad == kernel_slots_rewritten
    # is an exact invariant of a healthy wake; unchanged slots need no proof.
    #
    # TOPOLOGY ONLY: there is no cuGraphExecKernelNodeGetParams, so the
    # cuGraphExecKernelNodeSetParams_v2 above stays verified by return code +
    # replay behaviour alone. Hence the field name -- do not read k_topo_ok as
    # evidence about the live exec.
    if vfy is not None:
        back = _KernelNodeParams()
        if cu.cuGraphKernelNodeGetParams_v2(node, ctypes.byref(back)) != 0 \
                or not back.kernelParams:
            vfy["topo_readback_err"] += 1
        else:
            bkp = back.kernelParams
            got_sg = (ctypes.c_uint64 * 8).from_address(bkp[1])
            for i in range(8):
                if new_sg_vals[i] == sg_before[i]:
                    continue
                vfy["k_topo_ok" if int(got_sg[i]) == new_sg_vals[i]
                    else "k_topo_bad"] += 1
            if new_self != sv:
                got_self = int((ctypes.c_uint64 * 1).from_address(bkp[2])[0])
                vfy["k_topo_ok" if got_self == new_self else "k_topo_bad"] += 1
            if new_dp is not None and bkp[0]:
                got_dp = int((ctypes.c_uint64 * 1).from_address(bkp[0])[0])
                vfy["k_topo_ok" if got_dp == new_dp else "k_topo_bad"] += 1
    # keep new_sg_buf/new_self_buf/new_dp_buf/new_kp alive until the calls return
    return rewrites


def _patch_memcpy_node(cu, exec_handle, node, ctx, addr_map, vfy=None):
    """If `node` is a MEMCPY whose src/dst device ptr is a stale (old) CA buffer
    (e.g. the copy-path `buffer_ptrs[rank]`), rewrite it to the new addr and
    commit. Returns the number of device-ptr fields rewritten.

    `vfy`: optional counter dict for the transactional topology readback (keys
    m_topo_ok / m_topo_bad / topo_readback_err). See _patch_node for why readback
    replaced old-address membership as the primary counter."""
    t = ctypes.c_int(-1)
    if cu.cuGraphNodeGetType(node, ctypes.byref(t)) != 0:
        return 0
    if t.value != _CU_NODE_TYPE_MEMCPY:
        return 0
    p = _Memcpy3D()
    if cu.cuGraphMemcpyNodeGetParams(node, ctypes.byref(p)) != 0:
        return 0
    rewrites = 0
    changed = {}
    for field in ("srcDevice", "dstDevice"):
        v = int(getattr(p, field))
        nv = addr_map.get(v)
        if nv is not None and nv != v:
            setattr(p, field, nv)
            changed[field] = nv
            rewrites += 1
    if rewrites:
        # Topology set (always) + exec set (only if instantiated).
        _ck("cuGraphMemcpyNodeSetParams",
            cu.cuGraphMemcpyNodeSetParams(node, ctypes.byref(p)))
        if exec_handle:
            _ck("cuGraphExecMemcpyNodeSetParams",
                cu.cuGraphExecMemcpyNodeSetParams(
                    exec_handle, node, ctypes.byref(p), ctx))
        # Topology-only readback; m_topo_ok + m_topo_bad == memcpy_ptrs_rewritten.
        if vfy is not None:
            back = _Memcpy3D()
            if cu.cuGraphMemcpyNodeGetParams(node, ctypes.byref(back)) != 0:
                vfy["topo_readback_err"] += 1
            else:
                for field, want in changed.items():
                    vfy["m_topo_ok" if int(getattr(back, field)) == want
                        else "m_topo_bad"] += 1
    return rewrites


def _audit_stale_ca_addrs(cu, pairs, addr_map, ca=None) -> dict:
    """Read-only post-rewrite audit of the CA pointer slots baked into `pairs`.

    `bad` must be 0 after rewrite_addrs_in_graphs. It sums only the counters that
    are UNAMBIGUOUS evidence of a graph that will fault on its next replay.

    Why this is not simply "does any slot still hold a key of addr_map" -- which is
    what this function used to ask (and what E14 tripped over): that predicate is a
    staleness test ONLY when addr_map's key set and value set are disjoint. They are
    on the first wake (old = pre-CRIU VAs, new = freshly allocated), but on a 2nd or
    later wake CA reallocates into the VAs the previous wake just freed, so addr_map
    becomes a permutation with heavy key/value overlap and a CORRECTLY rewritten slot
    holds a value that is also some other rank's old address. E14 reported
    hits=39780 on 3 ranks and 26520 on a 4th with the rewrite provably complete; every
    one of those integers is reproduced exactly by the overlap model in
    agent_run_log/e14_v2arch_v24/audit_falsepos_proof.py. Worse than the noise: on
    such a wake the old predicate could not have distinguished a real residual, so it
    would have masked one.

    So the audit now asks two answerable questions instead:

      * POSITIONAL (primary, needs `ca`): a CA kernel is identified by func name --
        address-independent -- and then every slot is compared to the value it OUGHT
        to hold now: sg[i] == ca.meta_ptrs[i] for i < tp, and self_sg ==
        ca.meta_ptrs[ca.rank]. vLLM builds both lists rank-positionally
        (create_shared_buffer fills by enumerate(all_gather_object(...)) and inserts
        its own pointer at i == rank; CustomAllreduce itself indexes
        buffer_ptrs[self.rank]), so position is meaningful. This is strictly stronger
        than membership: it also catches a rank PERMUTATION, which any set-membership
        test passes happily. Reported split as kernel_sg_pos_bad / kernel_self_pos_bad
        so a positional-only failure is legible.
      * DEFINITELY STALE (fallback, no `ca` needed): a slot holding a value in
        `keys - values` -- an old address that is not the correct new value of
        anything -- is provably a freed VA. Slots in `keys & values` are recorded as
        `ambiguous` and NOT counted as bad; that bucket is exactly the old false
        positive, kept visible rather than silently dropped.

    The sg tail (i >= tp) is informational only: nothing here has ever proven
    RankSignals is zero-filled past ngpus, so a nonzero tail is counted
    (`sg_tail_nonzero`) but is `bad` only if it is provably stale (`sg_tail_stale`).
    Making the tail a correctness gate would repeat the mistake this rewrite fixes.

    `legacy_hits` recomputes the OLD predicate purely so runs can be compared against
    pre-fix logs. It is EXPECTED to be nonzero on any 2nd-or-later wake and means
    nothing on its own -- do not gate on it.

    Also counts the node classes the patchers deliberately skip, to answer "does a
    FULL graph bake a CA address somewhere we do not write?" empirically. E14
    answered NO (memset/memfree/memalloc/child all 0 across 8 rebind events on the V2
    manager path), so the patchers stay at KERNEL+MEMCPY; these counters remain as
    the regression tripwire for that decision. Child graphs are recursed one level.
    Never raises."""
    out = {"bad": 0, "kernel_sg_pos_bad": 0, "kernel_self_pos_bad": 0,
           "kernel_definite": 0, "memcpy_definite": 0, "memset_definite": 0,
           "memfree_definite": 0, "sg_tail_stale": 0,
           "kernel_ambiguous": 0, "memcpy_ambiguous": 0, "sg_tail_nonzero": 0,
           "n_ca_kernel": 0, "n_memset": 0, "n_memfree": 0, "n_memalloc": 0,
           "n_child": 0, "legacy_hits": 0, "pos_checked": False, "why": None,
           "err": None}

    values = set(addr_map.values())
    definite = set(addr_map) - values          # provably freed VAs
    ambig = set(addr_map) & values             # indistinguishable by value alone
    out["n_definite"] = len(definite)
    out["n_ambiguous_addrs"] = len(ambig)

    # Expected current values (positional). Guarded: a missing/odd ca must degrade
    # to the definite-stale check, never raise and never fabricate a failure.
    meta: list = []
    rank = None
    try:
        if ca is not None:
            meta = [int(x) for x in (getattr(ca, "meta_ptrs", None) or [])]
            r = getattr(ca, "rank", None)
            rank = int(r) if isinstance(r, int) else None
    except Exception as ex:  # noqa: BLE001
        out["why"] = f"ca read failed: {type(ex).__name__}: {ex}"
    tp = len(meta)
    pos = tp > 0 and rank is not None and 0 <= rank < tp
    out["pos_checked"] = pos
    if not pos and out["why"] is None:
        out["why"] = ("no ca" if ca is None else
                      f"tp={tp} rank={rank} -> positional audit skipped")

    def _cls(v):
        """Classify one slot value; returns 'definite' | 'ambiguous' | None."""
        if v in definite:
            return "definite"
        if v in ambig:
            return "ambiguous"
        return None

    def _walk(g, depth):
        try:
            nodes = _graph_nodes(cu, g)
        except Exception:  # noqa: BLE001
            return
        for node in nodes:
            t = ctypes.c_int(-1)
            if cu.cuGraphNodeGetType(node, ctypes.byref(t)) != 0:
                continue
            tv = t.value
            if tv == _CU_NODE_TYPE_KERNEL:
                params = _KernelNodeParams()
                if cu.cuGraphKernelNodeGetParams_v2(
                        node, ctypes.byref(params)) != 0:
                    continue
                if not params.func:
                    continue
                if "cross_device_reduce" not in _func_name(cu, params.func):
                    continue
                kp = params.kernelParams
                if not kp or not kp[1] or not kp[2]:
                    continue
                out["n_ca_kernel"] += 1
                sg = (ctypes.c_uint64 * 8).from_address(kp[1])
                for i in range(8):
                    v = int(sg[i])
                    out["legacy_hits"] += 1 if v in addr_map else 0
                    c = _cls(v)
                    if c == "definite":
                        out["kernel_definite"] += 1
                    elif c == "ambiguous":
                        out["kernel_ambiguous"] += 1
                    if pos and i < tp:
                        if v != meta[i]:
                            out["kernel_sg_pos_bad"] += 1
                    elif v:
                        out["sg_tail_nonzero"] += 1
                        if c == "definite":
                            out["sg_tail_stale"] += 1
                sv = int((ctypes.c_uint64 * 1).from_address(kp[2])[0])
                out["legacy_hits"] += 1 if sv in addr_map else 0
                c = _cls(sv)
                if c == "definite":
                    out["kernel_definite"] += 1
                elif c == "ambiguous":
                    out["kernel_ambiguous"] += 1
                if pos and sv != meta[rank]:
                    out["kernel_self_pos_bad"] += 1
                if kp[0]:
                    dp = int((ctypes.c_uint64 * 1).from_address(kp[0])[0])
                    out["legacy_hits"] += 1 if dp in addr_map else 0
                    # _dp points INTO rank_data, not at a meta_ptr, so there is no
                    # positional expectation for it -- only "not a freed VA".
                    if _cls(dp) == "definite":
                        out["kernel_definite"] += 1
            elif tv == _CU_NODE_TYPE_MEMCPY:
                p = _Memcpy3D()
                if cu.cuGraphMemcpyNodeGetParams(node, ctypes.byref(p)) != 0:
                    continue
                for f in ("srcDevice", "dstDevice"):
                    v = int(getattr(p, f))
                    out["legacy_hits"] += 1 if v in addr_map else 0
                    c = _cls(v)
                    if c == "definite":
                        out["memcpy_definite"] += 1
                    elif c == "ambiguous":
                        out["memcpy_ambiguous"] += 1
            elif tv == _CU_NODE_TYPE_MEMSET:
                out["n_memset"] += 1
                p = _MemsetNodeParams()
                if _e8_memset_params(cu, node, p):
                    d = int(p.dst)
                    out["legacy_hits"] += 1 if d in addr_map else 0
                    if d in definite:
                        out["memset_definite"] += 1
            elif tv == _CU_NODE_TYPE_MEM_FREE:
                out["n_memfree"] += 1
                dptr = ctypes.c_uint64(0)
                if _e8_memfree_dptr(cu, node, dptr):
                    d = int(dptr.value)
                    out["legacy_hits"] += 1 if d in addr_map else 0
                    if d in definite:
                        out["memfree_definite"] += 1
            elif tv == _CU_NODE_TYPE_MEM_ALLOC:
                # Count only: CUDA_MEM_ALLOC_NODE_PARAMS is deeply nested and a
                # mis-sized struct would let the driver write past our buffer.
                out["n_memalloc"] += 1
            elif tv == _CU_NODE_TYPE_GRAPH:
                out["n_child"] += 1
                if depth == 0:
                    sub = ctypes.c_void_p()
                    if _e8_child_graph(cu, node, sub) and sub.value:
                        _walk(sub.value, depth + 1)

    try:
        for g, _e in pairs:
            if g:
                _walk(g, 0)
        # A positionally-bad slot is usually also a definitely-stale slot, so these
        # buckets can overlap; `bad` is a must-be-zero flag, not a slot count.
        out["bad"] = (out["kernel_sg_pos_bad"] + out["kernel_self_pos_bad"]
                      + out["kernel_definite"] + out["memcpy_definite"]
                      + out["memset_definite"] + out["memfree_definite"]
                      + out["sg_tail_stale"])
    except Exception as ex:  # noqa: BLE001 - an audit must never break the wake
        out["err"] = f"{type(ex).__name__}: {ex}"
    return out


# ---------------------------------------------------------------------------
# CA graph-exec force-recommit (disk-wake stale-exec fix). Root-caused + A/B
# proven in DEBUG_reuse_graph.md: on a reuse DISK-wake, CRIU reproduces every CA
# VA -> addr_map=={} -> rewrite_addrs_in_graphs would early-return WITHOUT calling
# cuGraphExecKernelNodeSetParams for the CA nodes. The pre-CRIU-instantiated
# cuGraphExec then replays stale against the post-restore context and
# intermittently HANGS (1-stage signal barrier) or the worker silently dies
# (2-stage). ckpt-wake never hit this: its addr_map is non-empty -> SetParams
# fires -> the exec is re-committed "for free". A/B (forced reuse, both failers):
# 5/12 failures OFF -> 0/12 ON (p ~= 0.0009).
#
# These helpers DELIBERATELY DUPLICATE the struct handling of
# _patch_node/_patch_memcpy_node (instead of adding a `force` flag) so the vetted
# rewrite path used on the WORKING ckpt-wake (addr_map!={}) stays byte-for-byte
# identical -- all risk is isolated to the disk-wake no-op branch.
# ---------------------------------------------------------------------------
def _recommit_ca_kernel_node(cu, exec_handle, node) -> int:
    """Force the driver to re-accept a CA all-reduce kernel node's params with
    IDENTICAL current values (topology + exec). Mirrors _patch_node's struct
    handling but rewrites nothing. Returns 1 iff this was a CA kernel node and it
    was recommitted, else 0."""
    t = ctypes.c_int(-1)
    if cu.cuGraphNodeGetType(node, ctypes.byref(t)) != 0:
        return 0
    if t.value != _CU_NODE_TYPE_KERNEL:
        return 0
    params = _KernelNodeParams()
    if cu.cuGraphKernelNodeGetParams_v2(node, ctypes.byref(params)) != 0:
        return 0
    name = _func_name(cu, params.func)
    if "cross_device_reduce" not in name:
        return 0
    kp = params.kernelParams
    if not kp:
        return 0
    sg_addr = kp[1]
    self_addr = kp[2]
    if not sg_addr or not self_addr:
        return 0
    # Re-pack sg (8 ptrs) + self_sg (1 ptr) into fresh backing buffers holding the
    # SAME current values; _dp (kp[0]) / result,rank,size (kp[3..5]) pass through
    # by their existing arg-storage pointers (unchanged, exactly as _patch_node
    # does when new_dp is None).
    sg = (ctypes.c_uint64 * 8).from_address(sg_addr)
    self_sg = (ctypes.c_uint64 * 1).from_address(self_addr)
    new_sg_buf = (ctypes.c_uint64 * 8)(*[int(sg[i]) for i in range(8)])
    new_self_buf = (ctypes.c_uint64 * 1)(int(self_sg[0]))
    new_kp = (ctypes.c_void_p * 6)(
        kp[0], ctypes.cast(new_sg_buf, ctypes.c_void_p),
        ctypes.cast(new_self_buf, ctypes.c_void_p), kp[3], kp[4], kp[5])
    params.kernelParams = ctypes.cast(new_kp, ctypes.POINTER(ctypes.c_void_p))
    _ck("cuGraphKernelNodeSetParams_v2(recommit)",
        cu.cuGraphKernelNodeSetParams_v2(node, ctypes.byref(params)))
    if exec_handle:
        _ck("cuGraphExecKernelNodeSetParams_v2(recommit)",
            cu.cuGraphExecKernelNodeSetParams_v2(exec_handle, node,
                                                 ctypes.byref(params)))
    # keep new_sg_buf/new_self_buf/new_kp alive until the calls return
    return 1


def _recommit_ca_memcpy_node(cu, exec_handle, node, ctx, ca_buf_set) -> int:
    """Recommit a memcpy node whose src/dst device ptr is a CURRENT CA buffer_ptr
    (the copy-path scratch buffer) with identical params. ca_buf_set: set of int
    VAs from ca.buffer_ptrs. Returns 1 iff recommitted, else 0."""
    if not ca_buf_set:
        return 0
    t = ctypes.c_int(-1)
    if cu.cuGraphNodeGetType(node, ctypes.byref(t)) != 0:
        return 0
    if t.value != _CU_NODE_TYPE_MEMCPY:
        return 0
    p = _Memcpy3D()
    if cu.cuGraphMemcpyNodeGetParams(node, ctypes.byref(p)) != 0:
        return 0
    if int(p.srcDevice) not in ca_buf_set and int(p.dstDevice) not in ca_buf_set:
        return 0
    _ck("cuGraphMemcpyNodeSetParams(recommit)",
        cu.cuGraphMemcpyNodeSetParams(node, ctypes.byref(p)))
    if exec_handle:
        _ck("cuGraphExecMemcpyNodeSetParams(recommit)",
            cu.cuGraphExecMemcpyNodeSetParams(exec_handle, node, ctypes.byref(p), ctx))
    return 1


def _force_recommit_ca_nodes(worker, out) -> dict:
    """Force-recommit body: walk every captured graph exec and re-commit the CA
    kernel nodes (+ CA-buffer memcpy nodes) with their CURRENT values. Fills
    recommit counters into `out`. Root cause / A-B evidence: see the comment block
    above. Only call site: the addr_map=={} branch of rewrite_addrs_in_graphs."""
    out["forced_recommit"] = True
    try:
        cu = _cu()
        ctx = ctypes.c_void_p()
        cu.cuCtxGetCurrent(ctypes.byref(ctx))
        ca = _find_ca(worker)
        ca_buf_set = set(int(b) for b in (getattr(ca, "buffer_ptrs", []) or [])) if ca else set()
        pairs, gdiag = _find_captured_graphs(worker)
        gdiag.pop("_manager_handles", None)
        out["graph_discovery"] = gdiag
        out["n_graphs"] = len(pairs)
        kn = mn = 0
        for g, e in pairs:
            for node in _graph_nodes(cu, g):
                kn += _recommit_ca_kernel_node(cu, e, node)
                mn += _recommit_ca_memcpy_node(cu, e, node, ctx, ca_buf_set)
        out["kernel_nodes_recommitted"] = kn
        out["memcpy_nodes_recommitted"] = mn
        out["ok"] = True
        log.info("FORCE_RECOMMIT: %d graphs; recommitted kernel=%d memcpy=%d "
                 "(identical values, addr_map empty)", len(pairs), kn, mn)
    except Exception as ex:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(ex).__name__}: {ex}"
        log.warning("force_recommit failed: %s", out["error"])
    return out


def rewrite_addrs_in_graphs(worker, addr_map) -> dict:
    """Rewrite every baked old->new address across all captured graph execs:
    meta_ptrs in CA kernel nodes (sg/self_sg) and CA buffer_ptrs in memcpy nodes
    (the copy path's scratch buffer). addr_map: {old_int: new_int}."""
    out: dict = {"step": "rewrite_addrs"}
    addr_map = {int(o): int(n) for o, n in addr_map.items() if int(o) != int(n)}
    out["n_map"] = len(addr_map)
    out["addr_map_sample"] = {hex(k): hex(v) for k, v in list(addr_map.items())[:6]}
    if not addr_map:
        # Disk-wake: CRIU reproduced every CA VA -> addr_map=={}. Re-committing
        # the CA graph-exec node params (identical values) forces the driver to
        # re-accept the pre-CRIU exec against the post-restore context; without it
        # the stale exec intermittently hangs/dies on the first real forward
        # (DEBUG_reuse_graph.md root cause, A/B 5/12 -> 0/12). The recommit is
        # unconditional on this branch -- it is the only correct behaviour here,
        # and a no-op early-return is NOT a safe alternative. Only this no-op
        # branch is affected; it is reached only via rebind_after_reinit (ckpt/disk
        # wake), never on sleep-wake. Cost: one cuGraphExecKernelNodeSetParams per
        # CA node at wake (us-scale).
        return _force_recommit_ca_nodes(worker, out)
    try:
        cu = _cu()
        ctx = ctypes.c_void_p()
        cu.cuCtxGetCurrent(ctypes.byref(ctx))
        pairs, gdiag = _find_captured_graphs(worker)
        mgr_handles = gdiag.pop("_manager_handles", frozenset())
        out["graph_discovery"] = gdiag
        out["n_graphs"] = len(pairs)
        kn = kr = mn = mr = 0
        mkn = mmn = 0   # manager-source split: proves the FULL graphs got patched
        # Transactional readback counters, filled in-place by the patchers. Invariants
        # on a healthy wake: k_topo_ok == kernel_slots_rewritten and
        # m_topo_ok == memcpy_ptrs_rewritten, with the *_bad/err counters 0.
        vfy = {"k_topo_ok": 0, "k_topo_bad": 0, "m_topo_ok": 0, "m_topo_bad": 0,
               "topo_readback_err": 0}
        for g, e in pairs:
            is_mgr = g in mgr_handles
            for node in _graph_nodes(cu, g):
                r = _patch_node(cu, e, node, addr_map, vfy)
                if r:
                    kn += 1
                    kr += r
                    if is_mgr:
                        mkn += 1
                r2 = _patch_memcpy_node(cu, e, node, ctx, addr_map, vfy)
                if r2:
                    mn += 1
                    mr += r2
                    if is_mgr:
                        mmn += 1
        out["kernel_nodes_patched"] = kn
        out["kernel_slots_rewritten"] = kr
        out["memcpy_nodes_patched"] = mn
        out["memcpy_ptrs_rewritten"] = mr
        # Split by source so a regression to the piecewise-only enumerator is visible
        # as a number rather than as an IMA three minutes later (resume18): on a
        # ckpt-wake with FULL graphs present, manager_kernel_nodes_patched == 0 means
        # the FULL graphs were not reached and the next decode replay will fault.
        out["manager_kernel_nodes_patched"] = mkn
        out["manager_memcpy_nodes_patched"] = mmn
        # Primary confirm-not-lucky counters. topo_readback proves the commits took
        # (per node, immune to addr_map key/value overlap); stale_ca_audit answers
        # the semantic question against the CA object's CURRENT rank-ordered ptrs.
        vfy["k_topo_expected"] = kr
        vfy["m_topo_expected"] = mr
        vfy["ok"] = (vfy["k_topo_bad"] == 0 and vfy["m_topo_bad"] == 0
                     and vfy["topo_readback_err"] == 0
                     and vfy["k_topo_ok"] == kr and vfy["m_topo_ok"] == mr)
        out["topo_readback"] = vfy
        try:
            ca_for_audit = _find_ca(worker)
        except Exception as ex:  # noqa: BLE001 - the audit is optional, the wake is not
            ca_for_audit = None
            log.warning("audit: _find_ca failed: %s", ex)
        out["stale_ca_audit"] = _audit_stale_ca_addrs(cu, pairs, addr_map,
                                                     ca_for_audit)
        out["ok"] = True
        log.info("rewrite_addrs_in_graphs: %d graphs (%d from manager); kernel %d "
                 "nodes/%d slots (manager %d); memcpy %d nodes/%d ptrs (manager %d); "
                 "topo_readback ok=%s(%d/%d ok, %d bad, %d err); audit bad=%s "
                 "(pos_checked=%s ambiguous=%d legacy_hits=%d)",
                 len(pairs), len(mgr_handles), kn, kr, mkn, mn, mr, mmn,
                 vfy["ok"], vfy["k_topo_ok"] + vfy["m_topo_ok"], kr + mr,
                 vfy["k_topo_bad"] + vfy["m_topo_bad"], vfy["topo_readback_err"],
                 out["stale_ca_audit"].get("bad"),
                 out["stale_ca_audit"].get("pos_checked"),
                 (out["stale_ca_audit"].get("kernel_ambiguous", 0)
                  + out["stale_ca_audit"].get("memcpy_ambiguous", 0)),
                 out["stale_ca_audit"].get("legacy_hits"))
    except Exception as ex:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(ex).__name__}: {ex}"
        log.warning("rewrite_addrs_in_graphs failed: %s", out["error"])
    return out


# ---------------------------------------------------------------------------
# PATH 1: in-place rank_data peer-pointer refresh (keeps CA's registered fast
# path). The captured CA kernels bake `_dp = rank_data_base + slot*sizeof(RankData)`
# and read peer buffer pointers from RankData.ptrs[peer] at replay. rank_data's
# VA is stable across destroy/reinit, but its CONTENTS go stale: destroy_nccl
# unmaps the old peer IPC mappings and the new CA only re-registers slot 0. So
# we refresh every registered slot in place: read its own local ptr (ptrs[rank],
# stable VA, still valid), re-derive a FRESH IPC handle for it, exchange across
# ranks, open the peer's fresh handle, and write the live peer ptr into
# ptrs[peer]. In place (slot-aligned, no append) + fresh handles -> fixes the two
# bugs that sank the original Part A (slot-misalignment + stale handles).
# ---------------------------------------------------------------------------

_CU_IPC_HANDLE_SIZE = 64
_CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1
_RANKDATA_PTRS = 8           # struct RankData { void* ptrs[8]; } -> 8 i64 / slot


class _CUipcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ubyte * _CU_IPC_HANDLE_SIZE)]


def _cu_ipc(cu):
    """Lazily add IPC + address-range bindings to the cached libcuda handle."""
    if getattr(cu, "_semip_ipc_ready", False):
        return cu
    cu.cuMemGetAddressRange_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64),
                                           ctypes.POINTER(ctypes.c_size_t),
                                           ctypes.c_uint64]
    cu.cuIpcGetMemHandle.argtypes = [ctypes.POINTER(_CUipcMemHandle), ctypes.c_uint64]
    cu.cuIpcOpenMemHandle_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64),
                                         _CUipcMemHandle, ctypes.c_uint]
    cu._semip_ipc_ready = True
    return cu


def _collect_dp_slots(cu, base_rd, nbytes, worker=None):
    """Walk captured graphs; return sorted distinct RankData slot indices that
    the CA kernels' baked `_dp` (kernelParams[0]) reference, within rank_data.

    Pass `worker` so the FULL graphs are included: a slot referenced only by a FULL
    graph would otherwise never be refreshed by Path 1 (same root cause as the
    unpatched-FULL-graph bug, resume18)."""
    slots = set()
    stride = _RANKDATA_PTRS * 8
    lo, hi = base_rd, base_rd + nbytes
    pairs, _diag = _find_captured_graphs(worker)
    for g, _e in pairs:
        for node in _graph_nodes(cu, g):
            t = ctypes.c_int(-1)
            if (cu.cuGraphNodeGetType(node, ctypes.byref(t)) != 0
                    or t.value != _CU_NODE_TYPE_KERNEL):
                continue
            params = _KernelNodeParams()
            if cu.cuGraphKernelNodeGetParams_v2(node, ctypes.byref(params)) != 0:
                continue
            if "cross_device_reduce" not in _func_name(cu, params.func):
                continue
            kp = params.kernelParams
            if not kp or not kp[0]:
                continue
            dp = int((ctypes.c_uint64 * 1).from_address(kp[0])[0])
            if lo <= dp < hi and (dp - base_rd) % stride == 0:
                slots.add((dp - base_rd) // stride)
    return sorted(slots)


def refresh_rank_data_peers(worker) -> dict:
    """Path 1: re-establish the stale peer pointers in the registered RankData
    slots after reinit (rank_data VA stable; only its peer IPC contents are
    stale). For each slot the kept graph references: read ptrs[rank] (own local
    buffer, stable), derive a FRESH IPC handle, exchange cross-rank, open the
    peer handle, write the live peer ptr into ptrs[peer] -- in place."""
    out: dict = {"step": "refresh_rank_data_peers"}
    try:
        import torch
        import torch.distributed as dist
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"import: {e}"}
    ca = _find_ca(worker)
    if ca is None or getattr(ca, "rank_data", None) is None:
        out["skipped"] = "no ca/rank_data"
        return out
    rd = ca.rank_data
    rank = int(ca.rank)
    world_size = int(ca.world_size)
    group = ca.group
    base_rd = int(rd.data_ptr())
    nbytes = rd.numel() * rd.element_size()
    mask = (1 << 64) - 1
    try:
        cu = _cu_ipc(_cu())
        torch.cuda.synchronize()
        slots = _collect_dp_slots(cu, base_rd, nbytes, worker)
        out["n_slots"] = len(slots)
        if not slots:
            out["skipped"] = "no CA _dp slots in rank_data range"
            return out
        n_words = (max(slots) + 1) * _RANKDATA_PTRS
        view = rd.view(torch.int64)
        words = view[:n_words].cpu().tolist()
        # 1. FRESH (handle, offset) for OUR own buffer at each slot (dedupe / alloc)
        my, hcache, n_own_fail = {}, {}, 0
        out["base_rd"] = hex(base_rd)
        out["words_head"] = [hex(int(w) & mask) for w in words[0:16]]
        out["slots_head"] = list(slots[0:6])
        sample = []
        for idx, s in enumerate(slots):
            own = int(words[s * _RANKDATA_PTRS + rank]) & mask
            if own == 0:
                if idx < 6:
                    sample.append(f"slot{s}: own=0x0")
                continue
            b, sz = ctypes.c_uint64(0), ctypes.c_size_t(0)
            rc1 = cu.cuMemGetAddressRange_v2(ctypes.byref(b), ctypes.byref(sz),
                                             ctypes.c_uint64(own))
            if rc1 != 0:
                n_own_fail += 1
                if idx < 6:
                    sample.append(f"slot{s}: own={hex(own)} addrrange_rc={rc1}")
                continue
            own_base = int(b.value)
            hb = hcache.get(own_base)
            if hb is None:
                h = _CUipcMemHandle()
                rc2 = cu.cuIpcGetMemHandle(ctypes.byref(h), ctypes.c_uint64(own_base))
                if rc2 != 0:
                    n_own_fail += 1
                    if idx < 6:
                        sample.append(f"slot{s}: own={hex(own)} base={hex(own_base)} "
                                      f"ipc_rc={rc2}")
                    continue
                hb = bytes(h.reserved)
                hcache[own_base] = hb
            if idx < 6:
                sample.append(f"slot{s}: own={hex(own)} base={hex(own_base)} OK")
            my[s] = (hb, own - own_base)
        out["sample"] = sample
        out["n_my"] = len(my)
        out["n_own_fail"] = n_own_fail
        # 2. exchange each rank's {slot: (handle, offset)} (mirror CA's broadcast)
        ranks = sorted(dist.get_process_group_ranks(group=group))
        gathered = [[None] for _ in range(world_size)]
        gathered[rank] = [my]
        for i, r in enumerate(ranks):
            dist.broadcast_object_list(gathered[i], src=r, group=group, device="cpu")
        # 3. open peers + write ptrs[peer] in place
        opened, n_written, n_open_fail = {}, 0, 0
        for s in slots:
            for j in range(world_size):
                if j == rank:
                    continue
                peer_map = gathered[j][0] or {}
                ent = peer_map.get(s)
                if not ent:
                    continue
                hb, offset = ent
                pbase = opened.get(hb)
                if pbase is None:
                    h = _CUipcMemHandle()
                    ctypes.memmove(h.reserved, hb, _CU_IPC_HANDLE_SIZE)
                    pd = ctypes.c_uint64(0)
                    if cu.cuIpcOpenMemHandle_v2(
                            ctypes.byref(pd), h,
                            _CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS) != 0:
                        n_open_fail += 1
                        continue
                    pbase = int(pd.value)
                    opened[hb] = pbase
                words[s * _RANKDATA_PTRS + j] = pbase + offset
                n_written += 1
        # 4. write the refreshed slots back to device
        view[:n_words].copy_(torch.tensor(words, dtype=torch.int64))
        torch.cuda.synchronize()
        out.update(n_written=n_written, n_opened=len(opened), n_open_fail=n_open_fail)
        out["ok"] = (n_written > 0 and n_own_fail == 0 and n_open_fail == 0)
        log.info("refresh_rank_data_peers: slots=%d my=%d own_fail=%d written=%d "
                 "opened=%d open_fail=%d", len(slots), len(my), n_own_fail,
                 n_written, len(opened), n_open_fail)
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"
        log.warning("refresh_rank_data_peers failed: %s", out["error"])
    return out


def refresh_copy_path_slots(worker) -> dict:
    """Copy-path keep-graph fix (no IPC, no peer derivation). With force-copy, the
    captured CA kernels reduce the ONE symmetric staging buffer (buffer_ptrs[rank]),
    but the C++ still registers it per-call during capture, baking `_dp` into slots
    6043..N. The new CA's register_buffer only refreshes slot-0 (the staging), so the
    captured slots are stale. Since they ALL reference the same staging buffer, just
    replicate slot-0's (refreshed) RankData into every used slot."""
    out: dict = {"step": "refresh_copy_path_slots"}
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"import: {e}"}
    ca = _find_ca(worker)
    if ca is None or getattr(ca, "rank_data", None) is None:
        out["skipped"] = "no ca/rank_data"
        return out
    rd = ca.rank_data
    base_rd = int(rd.data_ptr())
    nbytes = rd.numel() * rd.element_size()
    mask = (1 << 64) - 1
    try:
        cu = _cu()
        torch.cuda.synchronize()
        slots = _collect_dp_slots(cu, base_rd, nbytes, worker)
        out["n_slots"] = len(slots)
        if not slots:
            out["skipped"] = "no CA _dp slots"
            return out
        view = rd.view(torch.int64)
        slot0 = view[0:_RANKDATA_PTRS].clone()      # the refreshed staging RankData
        # E3 tail-zero (codex resume7, gated SEMIP_CA_REFRESH_FIRST_WORLD=1): slot0's
        # uninitialized tail ptrs[world..7] is per-rank garbage that the loop below
        # would replicate into every used slot (0->garbage vs cold-start ZERO). When
        # gated, zero the clone's tail so each slot matches the cold-start RankData.
        # Robust across repeated rebinds (explicit zero, not 'leave as-is').
        if _ca_refresh_first_world_enabled():
            w = int(getattr(ca, "world_size", 0) or 0)
            if w <= 0:
                w = len(getattr(ca, "buffer_ptrs", []) or []) or _RANKDATA_PTRS
            w = max(0, min(w, _RANKDATA_PTRS))
            if 0 < w < _RANKDATA_PTRS:
                slot0[w:_RANKDATA_PTRS] = 0      # zero tail -> match cold-start RankData
            out["refresh_first_world"] = w
        out["slot0"] = [hex(int(x) & mask) for x in slot0.tolist()]
        # sample a used slot BEFORE (to confirm it was stale / differs from slot0)
        s_first = slots[0]
        out["used_before"] = [hex(int(x) & mask)
                              for x in view[s_first * _RANKDATA_PTRS:
                                            s_first * _RANKDATA_PTRS + _RANKDATA_PTRS].tolist()]
        for s in slots:
            view[s * _RANKDATA_PTRS:(s + 1) * _RANKDATA_PTRS] = slot0
        torch.cuda.synchronize()
        out["n_refreshed"] = len(slots)
        out["ok"] = True
        log.info("refresh_copy_path_slots: replicated slot0=%s into %d slots "
                 "(used_before=%s)", out["slot0"], len(slots), out["used_before"])
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"
        log.warning("refresh_copy_path_slots failed: %s", out["error"])
    return out


def _ca_pad_probe_enabled() -> bool:
    """Diagnostic dump of the CA one-shot Signal-pad state after reinit (default
    OFF). Enable with SEMIP_CA_PAD_PROBE=1. See probe_ca_signal_pads and the
    reuse-graph flakiness investigation."""
    return os.environ.get("SEMIP_CA_PAD_PROBE", "") == "1"


def _ca_zero_signal_enabled() -> bool:
    """Zero THIS rank's CA Signal header (meta_ptrs[rank], meta_size() bytes) after
    reinit, before the warmup replay (default OFF; enable SEMIP_CA_ZERO_SIGNAL=1).

    Candidate fix for the reuse one-shot-barrier deadlock/death: the CA one-shot
    barrier flag is a per-block MONOTONIC device counter (`_flag`) that is read,
    incremented, and waited on with EQUALITY (`while ld_volatile(slot) != flag`),
    and it is NEVER reset. A fresh post-reinit Signal that is zero-initialized is
    the correct start state (0 -> 1 uniformly across ranks). A fresh Signal holding
    garbage that DIFFERS ACROSS RANKS makes each rank derive a different `flag` ->
    the equality waits never converge (one-shot spin-deadlock) or a half-written
    slot is read (illegal access / silent worker death). Zeroing is a harmless
    no-op if the allocator already zeroed the buffer. Each rank zeros ONLY its own
    local Signal, then a group barrier guarantees all ranks finish before ANY
    captured graph replays."""
    return os.environ.get("SEMIP_CA_ZERO_SIGNAL", "") == "1"


def _ca_e3_dump_enabled() -> bool:
    """E3 fault-localization dump (default OFF; enable SEMIP_CA_E3_DUMP=1).

    Read-only classification, run inside probe_ca_signal_pads AFTER rewrite +
    refresh and BEFORE the warmup replay, of the graph-baked pointers the pad
    probe does NOT inspect: kp[3] (CA all-reduce result / output activation), the
    copy-path memcpy's non-CA-buffer endpoint (input activation), and the RankData
    tail ptrs[world..7]. The pad probe validated every Signal / sg / self_sg /
    rank_data slot as correct on a run that still IMA'd, so the invalid pointer is
    one of these. Classifies each unique pointer with cuMemGetAddressRange_v2 +
    cuPointerGetAttribute to name whether the IMA is a moved/unmapped activation
    VA, a peer-buffer mismatch, or the tail garbage. NEVER mutates; never raises.
    Consensus: codex resume6 (agent_run_log/codex_reuse_resume6*)."""
    return os.environ.get("SEMIP_CA_E3_DUMP", "") == "1"


def _ca_refresh_first_world_enabled() -> bool:
    """E3 tail-zero confirmation (default OFF; enable SEMIP_CA_REFRESH_FIRST_WORLD=1).

    E3 proved every CA-internal pointer valid (result / memcpy activation endpoints
    / peers all in-range, attr_rc=0). The SOLE rebind-introduced anomaly is that
    refresh_copy_path_slots replicates slot0's uninitialized tail ptrs[world..7]
    (per-rank garbage) into all used _dp slots, flipping the tail from the cold-start
    ZERO to garbage. When this gate is set, the refresh copies only the real peer
    words ptrs[0..world-1] and EXPLICITLY zeros the tail [world..8] to match the
    cold-start RankData (explicit zero, not 'leave as-is', so repeated rebinds cannot
    inherit a prior rebind's garbage tail). If this clears the ckpt-wake IMA the tail
    was causal; if it persists, CA metadata is excluded and we move to E4 (pre/post
    non-CA baked-pointer inventory). Consensus: codex resume7
    (agent_run_log/codex_reuse_resume7*)."""
    return os.environ.get("SEMIP_CA_REFRESH_FIRST_WORLD", "") == "1"


def _ca_e4_inventory_enabled() -> bool:
    """E4 pre/post baked-pointer inventory (default OFF; enable SEMIP_CA_E4_INVENTORY=1).

    E3 + tail-zero conclusively excluded ALL CA-internal pointers (result / memcpy
    endpoints / peers / tail) as the immediate IMA source, yet the ckpt-wake IMA
    persists. E4 tests the surviving hypothesis: a NON-CA graph kernel bakes a static
    activation/output VA that reinit (checkpoint_cuda + reinit_nccl) reallocates, and
    NOTHING rebinds it (the rebind touches only CA meta/buffer/rank_data). Sleep-wake
    replays the same baked VAs and never IMAs -> the delta is exactly what reinit does
    to non-CA baked pointers. This snapshots EVERY graph-baked size==8 device pointer
    pre-freeze (cold capture, known-good) via cuFuncGetParamInfo reflection, then
    post-reinit re-reads each at the same (graph,node,arg) and buckets it: a non-CA arg
    that was valid pre and is now unmapped (pre_valid_post_invalid) or now maps to a
    different allocation (value_same_range_changed) is the stale baked VA we are hunting.
    Read-only; never mutates a graph. Consensus: codex resume8
    (agent_run_log/codex_reuse_resume8*)."""
    return os.environ.get("SEMIP_CA_E4_INVENTORY", "") == "1"


def _ca_e5_reinstantiate_enabled() -> bool:
    """E5a re-instantiate-vs-preserved exec discriminator (default OFF; enable
    SEMIP_CA_E5_REINSTANTIATE=1).

    E4 proved EVERY baked pointer arg valid post-reinit (0 stale/dangling), topology
    stable (node_mismatch=0), CA structures valid (E3) -- yet the ckpt-wake reuse
    replay still IMAs. E1 proved a FRESH full recapture on the same path replays
    clean. So the fresh-vs-preserved delta is NOT a pointer value / topology / CA: it
    is something the PRESERVED, previously-INSTANTIATED cudaGraphExec_t carries that a
    fresh capture+instantiate regenerates and that a host-side param inventory cannot
    see -- H_exec: stale instantiated-exec device-side launch/scheduling/dependency
    state (incl. stream/event waits captured against pre-checkpoint objects reinit
    replaced) that survives checkpoint_cuda(cuCheckpointProcessLock) but is invalidated
    by the reinit_nccl frees/reallocs.

    When set, rebind_after_reinit (after the normal CA rewrite/refresh -- so the kept
    cudaGraph_t's CA nodes are ALREADY topology-patched, since ckpt-wake addr_map is
    non-empty) calls cuGraphInstantiateWithFlags on each kept cudaGraph_t to build a
    FRESH exec, then installs a torch.cuda.CUDAGraph.replay monkeypatch that launches
    the FRESH exec (cuGraphLaunch on the current stream) in place of the preserved one.
    If the IMA vanishes -> H_exec confirmed and re-instantiate (NOT re-capture -> no P2
    pool balloon) is the fix; if it persists -> H_exec dead -> E5b full-node scalar
    differential. Consensus: codex resume9/9b (agent_run_log/codex_reuse_resume9*)."""
    return os.environ.get("SEMIP_CA_E5_REINSTANTIATE", "") == "1"


def _e8_node_inventory_enabled() -> bool:
    """E8 node-class inventory (default OFF; enable SEMIP_E8_NODE_INV=1).

    E7 pinned the fault to a FRESH lazy re-instantiate + FULL-graph replay of a
    preserved-but-rebound cudaGraph_t (H_topology): 66 FULL replays, all
    preserved_exec=False, 0 piecewise, synchronous IMA on run_fullgraph ->
    self.graphs[desc].replay() under CUDA_LAUNCH_BLOCKING=1. E4 already proved every
    KERNEL-node arg + MEMCPY endpoint valid post-reinit. So the stale VA lives in a
    node class E4 never inventoried. When set, rebind_after_reinit walks every
    captured FULL/PIECEWISE graph and histograms node types (cuGraphNodeGetType),
    then classifies the baked target of the classes E4 skipped -- MEMSET.dst,
    MEM_FREE.dptr (single device VAs -> _e4_classify vs the live map), COUNTS
    MEM_ALLOC nodes (graph-owned cudaMallocAsync mempool; presence alone is decisive
    -- instantiate re-reserves VA), and recurses one level into CHILD_GRAPH nodes.
    Paired with the run_fullgraph desc-marker (install_reuse_diag_counters) this names
    both the faulting descriptor and the node class carrying the stale pointer.
    Consensus: codex resume12 (agent_run_log/codex_reuse_resume12*)."""
    return os.environ.get("SEMIP_E8_NODE_INV", "") == "1"


def _e9_h4_enabled() -> bool:
    """E9 H4 discriminator (default OFF; enable SEMIP_E9_H4=1).

    E8 was a DECISIVE NEGATIVE: across all 3315 full-graph objects the only node
    types present are kernel/memcpy/memset, and every baked pointer of every class
    validates as mapped + in-range post-reinit. So there is NO enumerable node
    carrying an unmapped stale VA -- option (b) targeted-node-repair has no target.
    Two mechanisms survive E8, both invisible to a graph-node inventory (codex
    resume13 ranks H4 > H1):
      (H4) level-2 GPU-resident indirection -- a baked pointer is valid, but its
           CONTENTS (block_table / slot_mapping / seq_lens -> paged-KV index) land
           OOB on the REAL decode inputs (fits warmup-OK / first-real-decode-fault).
      (H1) decode-only CA all-reduce peer-buffer read via IPC.
    When set, the run_fullgraph diag hook, immediately BEFORE the replay of the
    smallest decode desc (nt=1,nr=1,utc=1) -- the E7/E8 faulting descriptor --
    reads the eager-written index tensors off the live GPUModelRunner
    (input_buffers.seq_lens, block_tables.input_block_tables / .slot_mappings /
    .num_blocks) and bounds-checks them against the KV-cache capacity
    (kv_cache_config.num_blocks x block_size). Read-only, crash-safe marker written
    BEFORE the (possibly faulting) replay, wrapped in try/except -- never perturbs
    the replay path. Also logs the low-level gid on every full replay (nearly free)
    to settle whether the faulting graph succeeded in warmup (input-dependent, H4)
    or faulted on first instantiate. Consensus: codex resume13 Q3
    (agent_run_log/codex_reuse_resume13*)."""
    return os.environ.get("SEMIP_E9_H4", "") == "1"


def _e11_valdiff_enabled() -> bool:
    """E11 preserved-vs-fresh baked-value diff (default OFF; enable SEMIP_E11_VALDIFF=1).

    codex resume15 Q3 consensus. E4/E8's baked-pointer inventory read ONLY size==8
    kernel args (the `sz != 8` filter at _e4_read_kernel_ptrs), so every by-value
    struct arg -- prime suspect the 128-byte CUtensorMap/TMA descriptor, which bakes a
    global-memory base address at capture time -- was SKIPPED. That voids the
    "exhaustion" claim: the mapped+in-range verdict only ever covered 8-byte pointers.
    E11 bypasses the mapping/aliasing blind spot entirely by diffing the preserved-
    rebound decode graph's baked kernel-arg VALUES (ALL sizes, raw bytes) against a
    KNOWN-GOOD in-process FRESH recapture of the same descriptor. Any arg where
    preserved != fresh is a baked resource the rebind did not rewrite; that names the
    channel (which arg of which kernel) and the targeted repair. Runs IN-PROCESS (VAs
    are process-local) inside the collective reuse path so the fresh capture_model() is
    collective across ranks. Read-only diff; the fresh recapture replaces the preserved
    graphs as a side effect (a decode after E11 uses the fresh graphs -> no IMA). Run at
    reduced gpu_memory_utilization (E1v5: fresh recapture-FULL is zero-IMA + fits at
    gmu=0.35; at 0.9 it OOMs -- P2). Consensus: codex resume15
    (agent_run_log/codex_reuse_resume15*)."""
    return os.environ.get("SEMIP_E11_VALDIFF", "") == "1"


def probe_ca_signal_pads(worker, do_zero: bool = False) -> dict:
    """Env-gated CA one-shot Signal-pad diagnostic (+ optional local zero).

    Runs inside rebind_after_reinit AFTER rewrite + refresh_copy_path_slots and
    BEFORE the warmup replay (worker collective_rpc; no CA collective in flight,
    so a local memset is safe). Measures whether the fresh post-reinit LOCAL Signal
    (meta_ptrs[rank], first meta_size() bytes) is zero-initialized, validates that
    the graph-baked sg/self_sg pointers still match the current CA, checks that the
    refreshed rank_data _dp slots equal slot-0, and (if do_zero) cudaMemsets this
    rank's Signal to 0 then group-barriers so every rank starts the reused one-shot
    barrier from _flag=0. Entirely best-effort; never raises into the caller."""
    out: dict = {"step": "probe_ca_signal_pads", "do_zero": bool(do_zero)}
    try:
        import torch
        from vllm import _custom_ops as ops
        from vllm.distributed import parallel_state as ps
    except Exception as e:  # noqa: BLE001
        return {"step": "probe_ca_signal_pads", "ok": False,
                "error": f"import: {e}"}
    try:
        ca = _find_ca(worker)
        if ca is None:
            out["skipped"] = "no ca"
            return out
        rank = int(getattr(ca, "rank", -1))
        meta_ptrs = [int(x) for x in (getattr(ca, "meta_ptrs", []) or [])]
        world = len(meta_ptrs)
        out["rank"] = rank
        out["world"] = world
        if rank < 0 or rank >= world:
            out["skipped"] = f"bad rank {rank} / {world} meta_ptrs"
            return out
        local = meta_ptrs[rank]
        out["local_meta_ptr"] = hex(local)
        try:
            meta_size = int(ops.meta_size())
        except Exception as e:  # noqa: BLE001
            out["meta_size_err"] = f"{type(e).__name__}: {e}"
            meta_size = 0
        out["meta_size"] = meta_size
        cu = _cu()

        # --- dump: nonzero-u32 count over the Signal header (layout-agnostic and
        # decisive). Heads at the standard Signal layout (kMaxBlocks=36,
        # kMaxRanks=16): start@word0, end@word576, _flag@word1152 -- best-effort
        # color only; a version with different maxima shifts these but not the
        # total nonzero count. ---
        def _dump(ptr, n):
            if n <= 0:
                return None
            nb = (n // 4) * 4
            host = (ctypes.c_uint32 * (nb // 4))()
            code = cu.cuMemcpyDtoH_v2(ctypes.cast(host, ctypes.c_void_p),
                                      ctypes.c_uint64(ptr), ctypes.c_size_t(nb))
            if code != 0:
                return {"err": f"cuMemcpyDtoH_v2 CUresult={code}"}
            words = [int(host[i]) for i in range(nb // 4)]
            nz = [(i, w) for i, w in enumerate(words) if w != 0]

            def _head(word_off, k):
                if word_off >= len(words):
                    return None
                return [hex(words[j])
                        for j in range(word_off, min(word_off + k, len(words)))]

            return {"n_u32": len(words), "nonzero_u32": len(nz),
                    "first_nonzero": [(i, hex(w)) for i, w in nz[:16]],
                    "start0_head": _head(0, max(world, 1)),
                    "end0_head": _head(576, max(world, 1)),
                    "flag_head": _head(1152, 8)}

        out["pre"] = _dump(local, meta_size)

        # --- graph-baked signal-pointer validation: for each cross_device_reduce
        # node assert self_sg == meta_ptrs[rank] and every nonzero sg[i] is a
        # current meta_ptr. Closes the "over-attribute failures to dirty flags"
        # hole before we act on the pad dump. ---
        meta_set = set(meta_ptrs)
        n_nodes = n_self_ok = n_self_bad = n_sg_bad = 0
        bad_samples: list = []
        try:
            pairs, _gd = _find_captured_graphs(worker)
            for g, _e in pairs:
                for node in _graph_nodes(cu, g):
                    t = ctypes.c_int(-1)
                    if cu.cuGraphNodeGetType(node, ctypes.byref(t)) != 0:
                        continue
                    if t.value != _CU_NODE_TYPE_KERNEL:
                        continue
                    params = _KernelNodeParams()
                    if cu.cuGraphKernelNodeGetParams_v2(
                            node, ctypes.byref(params)) != 0:
                        continue
                    if "cross_device_reduce" not in _func_name(cu, params.func):
                        continue
                    kp = params.kernelParams
                    if not kp or not kp[1] or not kp[2]:
                        continue
                    n_nodes += 1
                    sg = (ctypes.c_uint64 * 8).from_address(kp[1])
                    self_sg = (ctypes.c_uint64 * 1).from_address(kp[2])
                    sv = int(self_sg[0])
                    if sv == local:
                        n_self_ok += 1
                    else:
                        n_self_bad += 1
                        if len(bad_samples) < 4:
                            bad_samples.append({"self_sg": hex(sv),
                                                "expected": hex(local)})
                    for i in range(8):
                        v = int(sg[i])
                        if v and v not in meta_set:
                            n_sg_bad += 1
                            break
            out["n_ca_nodes"] = n_nodes
            out["n_self_ok"] = n_self_ok
            out["n_self_mismatch"] = n_self_bad
            out["n_sg_mismatch"] = n_sg_bad
            out["graph_signal_ptr_ok"] = (n_self_bad == 0 and n_sg_bad == 0)
            if bad_samples:
                out["self_bad_samples"] = bad_samples
        except Exception as e:  # noqa: BLE001
            out["ptr_validate_err"] = f"{type(e).__name__}: {e}"

        # --- rank_data slot-equality: used _dp slots must equal slot-0 after
        # refresh_copy_path_slots (proves the copy actually took). ---
        try:
            rd = getattr(ca, "rank_data", None)
            if rd is not None:
                base_rd = int(rd.data_ptr())
                nbytes = rd.numel() * rd.element_size()
                slots = _collect_dp_slots(cu, base_rd, nbytes, worker)
                view = rd.view(torch.int64)
                slot0 = view[0:_RANKDATA_PTRS]
                eq = True
                for s in slots:
                    seg = view[s * _RANKDATA_PTRS:(s + 1) * _RANKDATA_PTRS]
                    if not bool(torch.equal(seg, slot0)):
                        eq = False
                        break
                out["rankdata_slots_equal_slot0"] = eq
                out["rankdata_n_slots"] = len(slots)
        except Exception as e:  # noqa: BLE001
            out["slot_eq_err"] = f"{type(e).__name__}: {e}"

        # --- E3 (SEMIP_CA_E3_DUMP): read-only classification of the graph-baked
        # pointers the pad probe does NOT inspect -- kp[3] (result / output
        # activation), the copy-path memcpy's non-CA-buffer endpoint (input
        # activation), and the RankData tail ptrs[world..7]. Names whether the IMA
        # is a moved/unmapped activation VA (cuMemGetAddressRange_v2 fails), a peer
        # mismatch, or the tail garbage. Dedupes pointer queries; treats an invalid
        # query as DATA (never raises). Consensus: codex resume6. ---
        if _ca_e3_dump_enabled():
            e3: dict = {"sampled_kernels": [], "sampled_memcpys": []}
            try:
                cu2 = _cu_ipc(cu)
                mask = (1 << 64) - 1
                buf_ptrs = [int(x) & mask
                            for x in (getattr(ca, "buffer_ptrs", []) or [])]
                buf_set = set(buf_ptrs)
                rd = getattr(ca, "rank_data", None)
                rd_base = int(rd.data_ptr()) if rd is not None else 0
                rd_view = rd.view(torch.int64) if rd is not None else None
                sw = _RANKDATA_PTRS          # 8 int64 words / RankData slot
                sb = sw * 8                  # 64 bytes / slot
                # CA range table: allocation base -> label (base-membership).
                ca_bases: dict = {}
                for i, m in enumerate(meta_ptrs):
                    ca_bases[int(m) & mask] = f"ca_meta[{i}]"
                for i, b in enumerate(buf_ptrs):
                    ca_bases[b] = f"ca_buffer[{i}]"
                if rd_base:
                    ca_bases[rd_base & mask] = "ca_rank_data"
                pcache: dict = {}

                def _classify(p):
                    p = int(p) & mask
                    if p in pcache:
                        return pcache[p]
                    base = ctypes.c_uint64(0)
                    size = ctypes.c_size_t(0)
                    rc = cu2.cuMemGetAddressRange_v2(
                        ctypes.byref(base), ctypes.byref(size),
                        ctypes.c_uint64(p))
                    mt = ctypes.c_int(-1)
                    rc2 = cu2.cuPointerGetAttribute(
                        ctypes.byref(mt), 2, ctypes.c_uint64(p))  # 2=MEMORY_TYPE
                    valid = (rc == 0)
                    b = int(base.value) & mask
                    d = {"ptr": hex(p), "valid": valid, "range_rc": rc,
                         "base": (hex(b) if valid else None),
                         "size": (int(size.value) if valid else None),
                         "memtype": (mt.value if rc2 == 0 else None),
                         "attr_rc": rc2,
                         "region": ("INVALID" if not valid
                                    else ca_bases.get(b, "other_device"))}
                    pcache[p] = d
                    return d

                nk = nm = 0
                n_result_invalid = n_result_other = 0
                n_tail_nonzero = n_peer_mismatch = 0
                n_memcpy_other_invalid = 0
                CAP = 24
                pairs2, _gd2 = _find_captured_graphs(worker)
                for g, _e2 in pairs2:
                    for node in _graph_nodes(cu2, g):
                        t = ctypes.c_int(-1)
                        if cu2.cuGraphNodeGetType(node, ctypes.byref(t)) != 0:
                            continue
                        if t.value == _CU_NODE_TYPE_KERNEL:
                            params = _KernelNodeParams()
                            if cu2.cuGraphKernelNodeGetParams_v2(
                                    node, ctypes.byref(params)) != 0:
                                continue
                            if ("cross_device_reduce"
                                    not in _func_name(cu2, params.func)):
                                continue
                            kp = params.kernelParams
                            if not kp or not kp[0] or not kp[3]:
                                continue
                            nk += 1
                            dp = int((ctypes.c_uint64 * 1).from_address(kp[0])[0])
                            res = int((ctypes.c_uint64 * 1).from_address(kp[3])[0])
                            rk = (int((ctypes.c_int * 1).from_address(kp[4])[0])
                                  if kp[4] else None)
                            sz = (int((ctypes.c_int * 1).from_address(kp[5])[0])
                                  if kp[5] else None)
                            slot = (((dp - rd_base) // sb)
                                    if (rd_base and dp >= rd_base
                                        and (dp - rd_base) % sb == 0) else -1)
                            tail_hex = None
                            peers_ok = None
                            if (rd_view is not None and 0 <= slot
                                    and (slot + 1) * sw <= rd_view.numel()):
                                seg = [int(x) & mask for x in
                                       rd_view[slot * sw:
                                               (slot + 1) * sw].tolist()]
                                tail = seg[world:sw]
                                tail_hex = [hex(v) for v in tail]
                                if any(v != 0 for v in tail):
                                    n_tail_nonzero += 1
                                peers_ok = all(v in buf_set for v in seg[0:world])
                                if not peers_ok:
                                    n_peer_mismatch += 1
                            rcls = _classify(res)
                            if rcls["region"] == "INVALID":
                                n_result_invalid += 1
                            elif rcls["region"] == "other_device":
                                n_result_other += 1
                            if len(e3["sampled_kernels"]) < CAP:
                                e3["sampled_kernels"].append(
                                    {"dp": hex(dp), "slot": slot,
                                     "kp4_rank": rk, "kp5_size": sz,
                                     "tail": tail_hex, "peers_ok": peers_ok,
                                     "result": rcls})
                        elif t.value == _CU_NODE_TYPE_MEMCPY:
                            p = _Memcpy3D()
                            if cu2.cuGraphMemcpyNodeGetParams(
                                    node, ctypes.byref(p)) != 0:
                                continue
                            src = int(p.srcDevice) & mask
                            dst = int(p.dstDevice) & mask
                            if src not in buf_set and dst not in buf_set:
                                continue
                            nm += 1
                            in_src = src in buf_set
                            ocls = _classify(dst if in_src else src)
                            if ocls["region"] == "INVALID":
                                n_memcpy_other_invalid += 1
                            if len(e3["sampled_memcpys"]) < CAP:
                                e3["sampled_memcpys"].append(
                                    {"src": hex(src), "dst": hex(dst),
                                     "buf_endpoint": ("src" if in_src else "dst"),
                                     "other": ocls})
                n_attr_invalid = sum(1 for d in pcache.values()
                                     if d["attr_rc"] != 0)
                n_outside = sum(1 for d in pcache.values()
                                if d["region"] == "other_device")
                e3["counters"] = {
                    "n_ca_kernels": nk, "n_ca_memcpys": nm,
                    "n_result_invalid": n_result_invalid,
                    "n_result_other_device": n_result_other,
                    "n_memcpy_other_invalid": n_memcpy_other_invalid,
                    "n_tail_nonzero": n_tail_nonzero,
                    "n_rankdata_peer_mismatch": n_peer_mismatch,
                    "n_attr_invalid": n_attr_invalid,
                    "n_outside_ca_ranges": n_outside,
                    "n_unique_ptrs": len(pcache)}
                out["e3"] = e3
                log.info(
                    "CA_E3 rank=%s kernels=%s memcpys=%s result_invalid=%s "
                    "result_other=%s memcpy_other_invalid=%s tail_nonzero=%s "
                    "peer_mismatch=%s attr_invalid=%s outside_ca=%s uniq=%s | "
                    "sample_result=%s sample_memcpy_other=%s",
                    rank, nk, nm, n_result_invalid, n_result_other,
                    n_memcpy_other_invalid, n_tail_nonzero, n_peer_mismatch,
                    n_attr_invalid, n_outside, len(pcache),
                    (e3["sampled_kernels"][0]["result"]
                     if e3["sampled_kernels"] else None),
                    (e3["sampled_memcpys"][0]["other"]
                     if e3["sampled_memcpys"] else None))
            except Exception as _e3e:  # noqa: BLE001
                out["e3_err"] = f"{type(_e3e).__name__}: {_e3e}"
                log.warning("CA_E3 dump failed: %s", out["e3_err"])

        # --- candidate fix: zero THIS rank's local Signal, sync, group-barrier ---
        if do_zero and meta_size > 0:
            code = cu.cuMemsetD8_v2(ctypes.c_uint64(local), ctypes.c_ubyte(0),
                                    ctypes.c_size_t(meta_size))
            out["zero_memset_code"] = code
            torch.cuda.synchronize()
            barr = "none"
            try:
                tpg = getattr(ps, "_TP", None)
                cpu_grp = getattr(tpg, "cpu_group", None) if tpg else None
                if cpu_grp is not None:
                    torch.distributed.barrier(group=cpu_grp)
                    barr = "cpu_group"
                elif tpg is not None and hasattr(tpg, "barrier"):
                    tpg.barrier()
                    barr = "tp.barrier"
            except Exception as e:  # noqa: BLE001
                barr = f"err:{type(e).__name__}:{e}"
            out["zero_barrier"] = barr
            out["post"] = _dump(local, meta_size)

        log.info("CA_PAD_PROBE rank=%s meta_size=%s pre_nonzero_u32=%s "
                 "graph_ptr_ok=%s self_mismatch=%s sg_mismatch=%s "
                 "slots_eq_slot0=%s do_zero=%s post_nonzero_u32=%s barrier=%s",
                 rank, meta_size, (out.get("pre") or {}).get("nonzero_u32"),
                 out.get("graph_signal_ptr_ok"), out.get("n_self_mismatch"),
                 out.get("n_sg_mismatch"), out.get("rankdata_slots_equal_slot0"),
                 do_zero, (out.get("post") or {}).get("nonzero_u32"),
                 out.get("zero_barrier"))
        out["ok"] = True
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"
        log.warning("probe_ca_signal_pads failed: %s", out["error"])
    return out


# ---------------------------------------------------------------------------
# E4: pre/post baked-pointer inventory (codex resume8 consensus). See
# _ca_e4_inventory_enabled for the hypothesis. Additive, env-gated, read-only.
# ---------------------------------------------------------------------------
# CU_LAUNCH_PARAM sentinels for the `extra` packed-arg convention (cuLaunchKernel).
_CU_LAUNCH_PARAM_END = 0
_CU_LAUNCH_PARAM_BUFFER_POINTER = 1
_CU_LAUNCH_PARAM_BUFFER_SIZE = 2

_E4_PARAM_LAYOUT_CACHE: dict = {}  # int(CUfunction) -> [(offset, size), ...]


def _func_param_layout(cu, func):
    """[(offset, size), ...] for CUfunction `func` via cuFuncGetParamInfo, iterating
    paramIndex until CUDA_ERROR_INVALID_VALUE (the terminator IS the count on driver
    13000, which lacks cuFuncGetParamCount). Cached per func handle (layout is
    per-function -> O(unique funcs), not O(nodes)). [] if func NULL / first query fails."""
    key = int(func) if func else 0
    if key == 0:
        return []
    cached = _E4_PARAM_LAYOUT_CACHE.get(key)
    if cached is not None:
        return cached
    layout = []
    off = ctypes.c_size_t(0)
    sz = ctypes.c_size_t(0)
    i = 0
    while i < 512:  # hard cap; real kernels have far fewer params
        rc = cu.cuFuncGetParamInfo(ctypes.c_void_p(key), ctypes.c_size_t(i),
                                   ctypes.byref(off), ctypes.byref(sz))
        if rc != 0:
            break
        layout.append((int(off.value), int(sz.value)))
        i += 1
    _E4_PARAM_LAYOUT_CACHE[key] = layout
    return layout


def _e4_read_kernel_ptrs(cu, params, layout):
    """(list_of_(arg_idx, value), n_extra_unreadable) for each size==8 param of a
    kernel node. Handles BOTH arg conventions:
      - kernelParams != NULL: kernelParams[i] is the ADDRESS of arg-i storage -> deref.
      - kernelParams == NULL, extra != NULL: args packed in one buffer at layout
        offsets (CU_LAUNCH_PARAM_BUFFER_POINTER/SIZE). Parse extra; on failure or
        offset+size > buffer_size, skip and flag n_extra_unreadable (codex point 4)."""
    out = []
    kp = params.kernelParams
    if kp:
        for i, (_off, sz) in enumerate(layout):
            if sz != 8:
                continue
            addr = kp[i]
            if not addr:
                continue
            try:
                val = int((ctypes.c_uint64 * 1).from_address(int(addr))[0])
            except Exception:  # noqa: BLE001
                continue
            out.append((i, val))
        return out, 0
    extra = params.extra
    if not extra:
        return out, 0
    buf_base = 0
    buf_size = None
    try:
        j = 0
        while j < 64:
            key = extra[j]
            if not key or int(key) == _CU_LAUNCH_PARAM_END:
                break
            k = int(key)
            nxt = extra[j + 1]
            if k == _CU_LAUNCH_PARAM_BUFFER_POINTER:
                buf_base = int(nxt) if nxt else 0  # entry IS the packed-arg buffer ptr
            elif k == _CU_LAUNCH_PARAM_BUFFER_SIZE:
                buf_size = (int((ctypes.c_size_t * 1).from_address(int(nxt))[0])
                            if nxt else None)  # entry is &size_t
            j += 2
    except Exception:  # noqa: BLE001
        return out, 1
    if not buf_base:
        return out, 1
    n_unreadable = 0
    for i, (off, sz) in enumerate(layout):
        if sz != 8:
            continue
        if buf_size is not None and off + sz > buf_size:
            n_unreadable = 1
            continue
        try:
            val = int((ctypes.c_uint64 * 1).from_address(buf_base + off)[0])
        except Exception:  # noqa: BLE001
            n_unreadable = 1
            continue
        out.append((i, val))
    return out, n_unreadable


def _e4_walk(cu, worker=None):
    """Walk every captured graph; return (records, counters). records: dicts
    {gi, ni, kind, func, arg, val} for each size==8 kernel arg + each memcpy src/dst
    that read as a nonzero value. NO classification here (pre/post classify against
    their own live driver state). (gi, ni) are enumeration ordinals -- stable across
    checkpoint/restore because the same process holds the same graph objects; func
    is the guard against any residual (gi, ni) drift (-> node_mismatch).

    Pass `worker`: without it this walks only the piecewise graphs, which is why
    E4's original "pre_valid_post_invalid=0" was true and vacuous -- it inventoried
    exactly the set the rebind had just patched and never saw a FULL graph."""
    records = []
    n_kernel_nodes = n_memcpy_nodes = 0
    n_unreadable = n_extra_unreadable = n_no_layout = 0
    fname_cache: dict = {}
    pairs, _diag = _find_captured_graphs(worker)
    for gi, (g, _e) in enumerate(pairs):
        try:
            nodes = _graph_nodes(cu, g)
        except Exception:  # noqa: BLE001
            continue
        for ni, node in enumerate(nodes):
            t = ctypes.c_int(-1)
            if cu.cuGraphNodeGetType(node, ctypes.byref(t)) != 0:
                continue
            if t.value == _CU_NODE_TYPE_KERNEL:
                params = _KernelNodeParams()
                if cu.cuGraphKernelNodeGetParams_v2(
                        node, ctypes.byref(params)) != 0:
                    n_unreadable += 1
                    continue
                if not params.func:  # contextless kernel node (kern set, func NULL)
                    n_unreadable += 1
                    continue
                fkey = int(params.func)
                fn = fname_cache.get(fkey)
                if fn is None:
                    fn = _func_name(cu, params.func)
                    fname_cache[fkey] = fn
                layout = _func_param_layout(cu, params.func)
                if not layout:
                    n_no_layout += 1
                    continue
                n_kernel_nodes += 1
                ptrs, ne = _e4_read_kernel_ptrs(cu, params, layout)
                n_extra_unreadable += ne
                for (ai, val) in ptrs:
                    if val:
                        records.append({"gi": gi, "ni": ni, "kind": "kernel",
                                        "func": fn, "arg": ai, "val": val})
            elif t.value == _CU_NODE_TYPE_MEMCPY:
                p = _Memcpy3D()
                if cu.cuGraphMemcpyNodeGetParams(node, ctypes.byref(p)) != 0:
                    n_unreadable += 1
                    continue
                n_memcpy_nodes += 1
                src = int(p.srcDevice)
                dst = int(p.dstDevice)
                if src:
                    records.append({"gi": gi, "ni": ni, "kind": "memcpy",
                                    "func": "<memcpy>", "arg": -1, "val": src})
                if dst:
                    records.append({"gi": gi, "ni": ni, "kind": "memcpy",
                                    "func": "<memcpy>", "arg": -2, "val": dst})
    counters = {"n_kernel_nodes": n_kernel_nodes, "n_memcpy_nodes": n_memcpy_nodes,
                "n_unreadable": n_unreadable, "n_extra_unreadable": n_extra_unreadable,
                "n_no_layout": n_no_layout, "n_records": len(records),
                "n_unique_funcs": len(fname_cache)}
    return records, counters


def _e4_classify(cu, p, cache):
    """Classify a device VA -> {valid, base, size, memtype}. cuMemGetAddressRange_v2
    is primary truth (valid == mapped); cuPointerGetAttribute(MEMORY_TYPE) is a
    secondary check. Cached by value."""
    mask = (1 << 64) - 1
    p = int(p) & mask
    if p in cache:
        return cache[p]
    base = ctypes.c_uint64(0)
    size = ctypes.c_size_t(0)
    rc = cu.cuMemGetAddressRange_v2(ctypes.byref(base), ctypes.byref(size),
                                    ctypes.c_uint64(p))
    mt = ctypes.c_int(-1)
    rc2 = cu.cuPointerGetAttribute(ctypes.byref(mt), 2, ctypes.c_uint64(p))
    d = {"valid": (rc == 0),
         "base": (int(base.value) & mask if rc == 0 else None),
         "size": (int(size.value) if rc == 0 else None),
         "memtype": (mt.value if rc2 == 0 else None),
         "range_rc": rc, "attr_rc": rc2}
    cache[p] = d
    return d


def _e4_torch_segments():
    """Sorted [(base, end), ...] from torch.cuda.memory_snapshot(). LABEL set only
    (a valid ptr outside these is 'outside_current_snapshot', not a fault -- non-torch
    driver allocations exist). [] on any failure."""
    try:
        import torch
        segs = []
        for s in torch.cuda.memory_snapshot():
            a = int(s.get("address", 0) or 0)
            n = int(s.get("total_size", 0) or 0)
            if a and n:
                segs.append((a, a + n))
        segs.sort()
        return segs
    except Exception:  # noqa: BLE001
        return []


def _e4_ca_ranges(cu, ca):
    """Sorted [(base, end), ...] for the NEW CA meta/buffer/rank_data allocations so
    CA-owned pointers get a label instead of 'outside'. Best-effort."""
    ranges = []
    if ca is None:
        return ranges
    ptrs = [int(x) for x in (getattr(ca, "meta_ptrs", []) or [])]
    ptrs += [int(x) for x in (getattr(ca, "buffer_ptrs", []) or [])]
    rd = getattr(ca, "rank_data", None)
    if rd is not None:
        try:
            ptrs.append(int(rd.data_ptr()))
        except Exception:  # noqa: BLE001
            pass
    for p in ptrs:
        base = ctypes.c_uint64(0)
        size = ctypes.c_size_t(0)
        if cu.cuMemGetAddressRange_v2(ctypes.byref(base), ctypes.byref(size),
                                      ctypes.c_uint64(p)) == 0:
            ranges.append((int(base.value), int(base.value) + int(size.value)))
    ranges.sort()
    return ranges


def _e4_in_ranges(base, los, ranges):
    """True if `base` falls within any labeled live range (torch segment or CA).
    los is the pre-extracted sorted list of range starts for bisect."""
    import bisect
    if not ranges:
        return False
    i = bisect.bisect_right(los, base) - 1
    return 0 <= i < len(ranges) and ranges[i][0] <= base < ranges[i][1]


def e4_pre_inventory(worker) -> dict:
    """E4 PRE hook (store_snapshot, cold capture, pre-freeze). Snapshot every
    graph-baked size==8 device pointer while the kept graph is KNOWN-GOOD, keeping
    only those classifying as VALID device pointers. Stores a compact tuple list on
    the worker for the post-reinit reclassify. Read-only; never raises to the caller.
    Consensus: codex resume8."""
    out = {"step": "e4_pre_inventory"}
    try:
        import torch
        torch.cuda.synchronize()  # surface any pending async error before the inventory
        cu = _cu_ipc(_cu())
        records, counters = _e4_walk(cu, worker)
        cache: dict = {}
        inv = []
        n_valid = n_invalid = 0
        for r in records:
            c = _e4_classify(cu, r["val"], cache)
            if not c["valid"]:
                n_invalid += 1
                continue
            n_valid += 1
            inv.append((r["gi"], r["ni"], r["kind"], r["func"], r["arg"],
                        r["val"], c["base"], c["size"]))
        setattr(worker, _E4_INV_ATTR, inv)
        out["ok"] = True
        out["counters"] = counters
        out["n_pre_valid"] = n_valid
        out["n_pre_invalid_skipped"] = n_invalid
        out["n_unique_ptrs"] = len(cache)
        log.info("CA_E4_PRE stored=%s valid=%s invalid_skipped=%s uniq_ptrs=%s "
                 "kernels=%s memcpys=%s uniq_funcs=%s no_layout=%s unreadable=%s "
                 "extra_unreadable=%s",
                 len(inv), n_valid, n_invalid, len(cache),
                 counters["n_kernel_nodes"], counters["n_memcpy_nodes"],
                 counters["n_unique_funcs"], counters["n_no_layout"],
                 counters["n_unreadable"], counters["n_extra_unreadable"])
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"
        log.warning("e4_pre_inventory failed: %s", out["error"])
    return out


def e4_post_inventory(worker) -> dict:
    """E4 POST hook (rebind_after_reinit, post-reinit, after CA rewrite/refresh,
    before warmup). Re-read each pre-snapshotted baked pointer at its (gi, ni, arg)
    and bucket it. The rebind rewrites ONLY CA (cross_device_reduce) node params +
    CA-buffer memcpy endpoints, so every OTHER baked pointer keeps its cold value; a
    non-CA arg that was valid pre and is now unmapped (pre_valid_post_invalid) or now
    maps to a different allocation (value_same_range_changed) is the stale baked VA
    we are hunting. Read-only; never raises. Consensus: codex resume8."""
    out = {"step": "e4_post_inventory"}
    try:
        import torch
        inv = getattr(worker, _E4_INV_ATTR, None)
        if not inv:
            out["skipped"] = "no pre-inventory (e4_pre_inventory did not run?)"
            log.warning("e4_post_inventory: %s", out["skipped"])
            return out
        torch.cuda.synchronize()
        cu = _cu_ipc(_cu())
        cur_records, cur_counters = _e4_walk(cu, worker)
        cur = {}
        for r in cur_records:
            cur[(r["gi"], r["ni"], r["arg"])] = (r["func"], r["val"])
        # Label set (secondary only): torch segments + new CA ranges, merged + sorted.
        ca = _find_ca(worker)
        label_ranges = _e4_torch_segments() + _e4_ca_ranges(cu, ca)
        label_ranges.sort()
        label_los = [r[0] for r in label_ranges]
        cache: dict = {}
        buckets = {k: 0 for k in (
            "pre_valid_post_invalid", "value_same_range_changed", "value_changed",
            "still_valid_same_range", "outside_current_snapshot", "ca_control",
            "memcpy_pre_valid_post_invalid", "node_mismatch", "unreadable")}
        hist: dict = {}     # bucket -> {func: count}
        samples: dict = {}  # bucket -> [few dicts]
        SAMP = 8
        for (gi, ni, kind, func, arg, pre_val, pre_base, pre_size) in inv:
            is_ca = ("cross_device_reduce" in func)
            key = (gi, ni, arg)
            if key not in cur:
                b = "node_mismatch"
            else:
                cur_func, cur_val = cur[key]
                if cur_func != func:
                    b = "node_mismatch"
                elif is_ca:
                    b = "ca_control"
                elif cur_val != pre_val:
                    b = "value_changed"
                else:
                    c = _e4_classify(cu, cur_val, cache)
                    if not c["valid"]:
                        b = ("memcpy_pre_valid_post_invalid" if kind == "memcpy"
                             else "pre_valid_post_invalid")
                    elif c["base"] == pre_base and c["size"] == pre_size:
                        b = ("still_valid_same_range"
                             if _e4_in_ranges(pre_base, label_los, label_ranges)
                             else "outside_current_snapshot")
                    else:
                        b = "value_same_range_changed"
            buckets[b] += 1
            hb = hist.setdefault(b, {})
            hb[func] = hb.get(func, 0) + 1
            sl = samples.setdefault(b, [])
            if len(sl) < SAMP:
                sl.append({"gi": gi, "ni": ni, "kind": kind, "func": func,
                           "arg": arg, "pre_val": hex(pre_val),
                           "pre_base": (hex(pre_base) if pre_base else None),
                           "pre_size": pre_size})
        out["ok"] = True
        out["n_pre_entries"] = len(inv)
        out["buckets"] = buckets
        out["hist"] = {b: dict(sorted(h.items(), key=lambda kv: -kv[1])[:12])
                       for b, h in hist.items()}
        out["samples"] = samples
        out["cur_counters"] = cur_counters
        log.info("CA_E4_POST pre_entries=%s | pre_valid_post_invalid=%s "
                 "value_same_range_changed=%s memcpy_pre_valid_post_invalid=%s "
                 "outside_snapshot=%s node_mismatch=%s value_changed=%s "
                 "still_valid=%s ca_control=%s unreadable=%s",
                 len(inv), buckets["pre_valid_post_invalid"],
                 buckets["value_same_range_changed"],
                 buckets["memcpy_pre_valid_post_invalid"],
                 buckets["outside_current_snapshot"], buckets["node_mismatch"],
                 buckets["value_changed"], buckets["still_valid_same_range"],
                 buckets["ca_control"], buckets["unreadable"])
        # Detail the smoking-gun buckets: per-func histogram + samples.
        for b in ("pre_valid_post_invalid", "value_same_range_changed",
                  "memcpy_pre_valid_post_invalid"):
            if buckets[b]:
                log.info("CA_E4_POST bucket=%s funcs=%s samples=%s",
                         b, out["hist"].get(b), out["samples"].get(b))
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"
        log.warning("e4_post_inventory failed: %s", out["error"])
    return out


# ---------------------------------------------------------------------------
# E11 (SEMIP_E11_VALDIFF): preserved-vs-fresh baked-value diff (codex resume15 Q3).
# Extends the E4 inventory beyond the size==8 filter -- reads EVERY kernel arg's raw
# bytes (incl. 128-byte CUtensorMap/TMA descriptors) -- and diffs the preserved-rebound
# graphs against an in-process FRESH recapture of the same descriptors. A KNOWN-GOOD
# fresh capture is the comparand (not a mapping predicate), so it sidesteps the aliasing
# blind spot that let E4/E8 pass a stale VA aliasing a re-reserved identical segment.
# ---------------------------------------------------------------------------
def _e11_read_all_args(cu, params, layout):
    """(list_of_(arg_idx, offset, size, raw_bytes), n_unreadable) for EVERY param of a
    kernel node -- E11 extends _e4_read_kernel_ptrs past the size==8 filter so by-value
    struct args (the 128-byte CUtensorMap/TMA descriptor, and any other non-8 arg) are
    captured as raw bytes. Same two arg conventions:
      - kernelParams != NULL: kernelParams[i] is the ADDRESS of arg-i storage -> read
        `size` bytes from it.
      - kernelParams == NULL, extra != NULL: args packed in one buffer at layout
        offsets (CU_LAUNCH_PARAM_BUFFER_POINTER/SIZE) -> read `size` bytes at off."""
    out = []
    n_unreadable = 0
    kp = params.kernelParams
    if kp:
        for i, (off, sz) in enumerate(layout):
            if sz <= 0 or sz > 4096:  # sanity bound; real kernel args are tiny
                continue
            addr = kp[i]
            if not addr:
                continue
            try:
                raw = bytes((ctypes.c_ubyte * sz).from_address(int(addr)))
            except Exception:  # noqa: BLE001
                n_unreadable += 1
                continue
            out.append((i, off, sz, raw))
        return out, n_unreadable
    extra = params.extra
    if not extra:
        return out, n_unreadable
    buf_base = 0
    buf_size = None
    try:
        j = 0
        while j < 64:
            key = extra[j]
            if not key or int(key) == _CU_LAUNCH_PARAM_END:
                break
            k = int(key)
            nxt = extra[j + 1]
            if k == _CU_LAUNCH_PARAM_BUFFER_POINTER:
                buf_base = int(nxt) if nxt else 0
            elif k == _CU_LAUNCH_PARAM_BUFFER_SIZE:
                buf_size = (int((ctypes.c_size_t * 1).from_address(int(nxt))[0])
                            if nxt else None)
            j += 2
    except Exception:  # noqa: BLE001
        return out, n_unreadable + 1
    if not buf_base:
        return out, n_unreadable + 1
    for i, (off, sz) in enumerate(layout):
        if sz <= 0 or sz > 4096:
            continue
        if buf_size is not None and off + sz > buf_size:
            n_unreadable += 1
            continue
        try:
            raw = bytes((ctypes.c_ubyte * sz).from_address(buf_base + off))
        except Exception:  # noqa: BLE001
            n_unreadable += 1
            continue
        out.append((i, off, sz, raw))
    return out, n_unreadable


def _e11_desc_by_handle(worker) -> dict:
    """{raw cudaGraph_t handle -> (num_tokens, num_reqs, uniform_token_count)} from the
    dense model runner's ModelCudaGraphManager (self.graphs keyed by
    BatchExecutionDescriptor). Lets the walk tag the faulting decode graph (the smallest
    desc, nt==nr==1) and its siblings. Empty if the runner has no such manager."""
    desc_by_handle = {}
    try:
        mr = getattr(worker, "model_runner", None)
        mgr = getattr(mr, "cudagraph_manager", None)
        graphs = getattr(mgr, "graphs", None)
        if isinstance(graphs, dict):
            for desc, g in graphs.items():
                try:
                    h = int(g.raw_cuda_graph())
                except Exception:  # noqa: BLE001
                    continue
                desc_by_handle[h] = (getattr(desc, "num_tokens", None),
                                     getattr(desc, "num_reqs", None),
                                     getattr(desc, "uniform_token_count", None))
    except Exception:  # noqa: BLE001
        pass
    return desc_by_handle


def _e11_walk_one(cu, handle, fname_cache) -> tuple:
    """Walk one cudaGraph_t handle -> (sig_tuple, node_recs). sig = ordered
    (node_type, func); node_recs carry the FULL baked kernel-arg inventory (ALL sizes,
    raw bytes). Read-only. Returns (None, None) if node enumeration fails."""
    try:
        nodes = _graph_nodes(cu, handle)
    except Exception:  # noqa: BLE001
        return None, None
    node_recs = []
    sig = []
    for ni, node in enumerate(nodes):
        t = ctypes.c_int(-1)
        if cu.cuGraphNodeGetType(node, ctypes.byref(t)) != 0:
            continue
        if t.value == _CU_NODE_TYPE_KERNEL:
            params = _KernelNodeParams()
            if cu.cuGraphKernelNodeGetParams_v2(
                    node, ctypes.byref(params)) != 0:
                continue
            if not params.func:
                continue
            fkey = int(params.func)
            fn = fname_cache.get(fkey)
            if fn is None:
                fn = _func_name(cu, params.func)
                fname_cache[fkey] = fn
            sig.append((0, fn))
            layout = _func_param_layout(cu, params.func)
            if not layout:
                continue
            args, _nu = _e11_read_all_args(cu, params, layout)
            node_recs.append({"ni": ni, "kind": "kernel", "func": fn,
                              "args": args})
        elif t.value == _CU_NODE_TYPE_MEMCPY:
            p = _Memcpy3D()
            if cu.cuGraphMemcpyNodeGetParams(node, ctypes.byref(p)) != 0:
                continue
            sig.append((1, "<memcpy>"))
            args = [(-1, 0, 8, int(p.srcDevice).to_bytes(8, "little")),
                    (-2, 0, 8, int(p.dstDevice).to_bytes(8, "little"))]
            node_recs.append({"ni": ni, "kind": "memcpy", "func": "<memcpy>",
                              "args": args})
        else:
            sig.append((int(t.value), ""))
    return tuple(sig), node_recs


def _e11_walk(worker) -> list:
    """E11: dump the FULL baked kernel-arg inventory (ALL sizes, raw bytes) of every
    captured graph, tagged with the vLLM decode descriptor when known. Returns a list of
    per-graph dicts:
      {gi, handle, desc, sig, nodes: [{ni, kind, func, args: [(ai, off, sz, raw)]}]}

    PRIMARY SOURCE = the dense runner's ModelCudaGraphManager (mr.cudagraph_manager.graphs,
    desc->CUDAGraph). This is where the FAULTING FULL decode graph desc=(1,1,1) lives (the
    E9 rfg hook replays self.graphs[desc]); it carries the desc natively; and crucially
    capture_model() REPOPULATES it on the fresh side. The earlier version enumerated
    _find_captured_graphs() (the CUDAGraphWrapper piecewise registry) instead -- but
    _semip_cleargraph EMPTIES those wrapper entries and the manager-driven recapture does
    NOT refill them, so the fresh dump came back EMPTY -> matched pairs=0 and the (1,1,1)
    graph was never even walked (NO_FAULTING_DESC_MATCHED). We now walk the manager
    directly. Fall back to the wrapper scan (desc=None, sig-matched) only if no manager
    graphs are present (non-dense / piecewise-only models). Read-only."""
    cu = _cu()
    fname_cache: dict = {}
    graphs_out = []
    seen_handles = set()

    # Primary: the ModelCudaGraphManager desc->CUDAGraph mapping.
    mgr_pairs = []  # (handle, desc)
    try:
        mr = getattr(worker, "model_runner", None)
        mgr = getattr(mr, "cudagraph_manager", None)
        graphs = getattr(mgr, "graphs", None)
        if isinstance(graphs, dict):
            for desc, g in graphs.items():
                try:
                    h = int(g.raw_cuda_graph())
                except Exception:  # noqa: BLE001
                    continue
                d = (getattr(desc, "num_tokens", None),
                     getattr(desc, "num_reqs", None),
                     getattr(desc, "uniform_token_count", None))
                mgr_pairs.append((h, d))
    except Exception:  # noqa: BLE001
        pass
    for gi, (h, d) in enumerate(mgr_pairs):
        if not h or h in seen_handles:
            continue
        sig, node_recs = _e11_walk_one(cu, h, fname_cache)
        if sig is None:
            continue
        seen_handles.add(h)
        graphs_out.append({"gi": gi, "handle": h, "desc": d,
                           "sig": sig, "nodes": node_recs})

    # Fallback: wrapper/piecewise graphs (desc=None) only if the manager gave nothing.
    if not graphs_out:
        pairs, _diag = _find_captured_graphs()
        for gi, (g, _e) in enumerate(pairs):
            if not g or int(g) in seen_handles:
                continue
            sig, node_recs = _e11_walk_one(cu, int(g), fname_cache)
            if sig is None:
                continue
            seen_handles.add(int(g))
            graphs_out.append({"gi": 1000 + gi, "handle": int(g), "desc": None,
                               "sig": sig, "nodes": node_recs})
    return graphs_out


def _e11_match_graphs(pres: list, fresh: list) -> list:
    """Pair preserved graphs to fresh graphs: by decode desc first (strong -- pins the
    faulting (1,1,1) preserved to fresh (1,1,1)), then by structural signature for the
    rest. Returns [(pres_g, fresh_g, how), ...] for matched pairs only."""
    from collections import deque
    fresh_by_desc: dict = {}
    fresh_by_sig: dict = {}
    for fg in fresh:
        if fg["desc"] is not None:
            fresh_by_desc.setdefault(fg["desc"], deque()).append(fg)
        fresh_by_sig.setdefault(fg["sig"], deque()).append(fg)
    used = set()          # id() of fresh graphs already paired
    matched_pres = set()  # id() of preserved graphs already paired
    matches = []
    # desc pass first so a desc match is never consumed by a sig match.
    for pg in pres:
        if pg["desc"] is None:
            continue
        q = fresh_by_desc.get(pg["desc"])
        while q and id(q[0]) in used:
            q.popleft()
        if q:
            fg = q.popleft()
            used.add(id(fg))
            matched_pres.add(id(pg))
            matches.append((pg, fg, "desc"))
    for pg in pres:
        if id(pg) in matched_pres:
            continue
        q = fresh_by_sig.get(pg["sig"])
        while q and id(q[0]) in used:
            q.popleft()
        if q:
            fg = q.popleft()
            used.add(id(fg))
            matches.append((pg, fg, "sig"))
    return matches


def _e11_diff_pair(cu, cache, pg, fg) -> dict:
    """Diff one matched (preserved, fresh) graph pair. Match nodes by ordered func-name
    queue (sig already matched -> lists align; func queue absorbs any residual reorder),
    args by arg index. Returns per-graph diff stats; 8-byte diffs classified (mapped?
    same base?) as annotation, non-8 struct diffs (TMA/CUtensorMap) recorded raw."""
    from collections import defaultdict, deque
    res = {"desc": pg["desc"], "how": None,
           "n_nodes_pres": len(pg["nodes"]), "n_nodes_fresh": len(fg["nodes"]),
           "n_args": 0, "n_equal": 0,
           "n_diff8": 0, "n_diff8_pres_invalid": 0, "n_diff8_same_base": 0,
           "n_diff8_both_valid_diff_base": 0,
           "n_diff_other": 0, "node_mismatch": 0,
           "diff8": [], "diff_other": [],
           # (func, arg) keys that differed -- used for the cross-graph uniqueness
           # control (a benign graph-pool scratch diff recurs across every graph using
           # that kernel; a stale persistent-buffer diff is unique to the faulting graph).
           "diff_keys": set(), "diff_keys_other": set()}
    fq = defaultdict(deque)
    for nrec in fg["nodes"]:
        fq[nrec["func"]].append(nrec)
    for pnode in pg["nodes"]:
        q = fq.get(pnode["func"])
        if not q:
            res["node_mismatch"] += 1
            continue
        fnode = q.popleft()
        fa = {a[0]: a for a in fnode["args"]}
        for (ai, off, sz, raw) in pnode["args"]:
            fresh_arg = fa.get(ai)
            if fresh_arg is None:
                continue
            res["n_args"] += 1
            fraw = fresh_arg[3]
            if raw == fraw:
                res["n_equal"] += 1
                continue
            if sz == 8:
                pv = int.from_bytes(raw, "little")
                fv = int.from_bytes(fraw, "little")
                pc = _e4_classify(cu, pv, cache)
                fc = _e4_classify(cu, fv, cache)
                res["n_diff8"] += 1
                if not pc["valid"]:
                    res["n_diff8_pres_invalid"] += 1
                elif fc["valid"] and pc["base"] == fc["base"]:
                    res["n_diff8_same_base"] += 1
                else:
                    res["n_diff8_both_valid_diff_base"] += 1
                res["diff8"].append(
                    (pnode["func"], pnode["ni"], ai, pv, fv,
                     bool(pc["valid"]), bool(fc["valid"])))
                res["diff_keys"].add((pnode["func"], ai))
            else:
                res["n_diff_other"] += 1
                # Annotation only (codex: do not interpret TMA layout): scan the struct's
                # 8-byte words and classify any that maps as a device VA -- a stale global
                # address baked inside a TMA descriptor would surface here.
                words = []
                for w in range(0, sz - 7, 8):
                    pv = int.from_bytes(raw[w:w + 8], "little")
                    fv = int.from_bytes(fraw[w:w + 8], "little")
                    if pv == fv:
                        continue
                    pc = _e4_classify(cu, pv, cache)
                    fc = _e4_classify(cu, fv, cache)
                    words.append((w, pv, fv, bool(pc["valid"]), bool(fc["valid"])))
                res["diff_other"].append(
                    (pnode["func"], pnode["ni"], ai, sz,
                     raw.hex(), fraw.hex(), words))
                res["diff_keys_other"].add((pnode["func"], ai))
    return res


def _e11_diff_and_report(pres: list, fresh: list) -> dict:
    """E11 top-level diff + report (codex resume15 Q3). Matches graphs (desc then sig),
    diffs each matched pair, aggregates buckets, and focuses the detailed dump on the
    FAULTING decode graph desc=(1,1,1) with sibling manager graphs as controls -- a
    stale baked resource unique to the decode graph shows a diff there that the siblings
    (using the same kernels) do not. Writes a crash-safe summary marker + verbose per-
    graph detail to the worker log. Read-only. Returns a compact summary dict."""
    cu = _cu_ipc(_cu())
    cache: dict = {}
    matches = _e11_match_graphs(pres, fresh)
    per_graph = []
    tot = {"n_pairs": len(matches), "n_pres": len(pres), "n_fresh": len(fresh),
           "n_args": 0, "n_equal": 0, "n_diff8": 0, "n_diff8_pres_invalid": 0,
           "n_diff8_same_base": 0, "n_diff8_both_valid_diff_base": 0,
           "n_diff_other": 0, "node_mismatch": 0, "n_desc_matched": 0}
    faulting = None
    for (pg, fg, how) in matches:
        r = _e11_diff_pair(cu, cache, pg, fg)
        r["how"] = how
        per_graph.append(r)
        if how == "desc":
            tot["n_desc_matched"] += 1
        for k in ("n_args", "n_equal", "n_diff8", "n_diff8_pres_invalid",
                  "n_diff8_same_base", "n_diff8_both_valid_diff_base",
                  "n_diff_other", "node_mismatch"):
            tot[k] += r[k]
        d = r["desc"]
        if (d is not None and isinstance(d[0], int) and isinstance(d[1], int)
                and d[0] == 1 and d[1] == 1):
            faulting = r

    # Cross-graph uniqueness control (codex resume15 Q3, THE decisive lens): a benign
    # graph-pool scratch diff at (func, arg) recurs across EVERY graph that uses that
    # kernel (fresh recapture uses a new mempool, so all scratch VAs move); a stale
    # persistent-buffer diff that the rebind failed to rewrite is UNIQUE to the faulting
    # decode graph. Count, per differing (func, arg) key, how many graphs show it.
    from collections import Counter
    key8_graphcount: Counter = Counter()
    keyoth_graphcount: Counter = Counter()
    for r in per_graph:
        for k in r["diff_keys"]:
            key8_graphcount[k] += 1
        for k in r["diff_keys_other"]:
            keyoth_graphcount[k] += 1

    # crash-safe headline marker. n_pres/n_fresh + the desc-tagged counts are FIRST so a
    # pairs=0 result is immediately diagnosable (empty side vs match failure).
    n_pres_desc = sum(1 for g in pres if g["desc"] is not None)
    n_fresh_desc = sum(1 for g in fresh if g["desc"] is not None)
    _reuse_diag_marker(
        "e11", "pres=%d(desc=%d) fresh=%d(desc=%d) pairs=%d desc_matched=%d args=%d "
        "equal=%d diff8=%d (pres_invalid=%d same_base=%d both_valid_diff_base=%d) "
        "diff_other=%d node_mismatch=%d"
        % (tot["n_pres"], n_pres_desc, tot["n_fresh"], n_fresh_desc,
           tot["n_pairs"], tot["n_desc_matched"], tot["n_args"], tot["n_equal"],
           tot["n_diff8"], tot["n_diff8_pres_invalid"], tot["n_diff8_same_base"],
           tot["n_diff8_both_valid_diff_base"], tot["n_diff_other"],
           tot["node_mismatch"]))
    log.info("E11 VALDIFF TOTALS pres_desc=%d fresh_desc=%d %s",
             n_pres_desc, n_fresh_desc, tot)

    # faulting-graph detail (the smoking gun, if any)
    faulting_uniq8: list = []
    faulting_uniqoth: list = []
    if faulting is not None:
        # keys that differ in the faulting graph AND in NO other matched graph -> the
        # prime suspects (a stale persistent rebind unique to decode, not pool scratch).
        faulting_uniq8 = sorted(
            k for k in faulting["diff_keys"] if key8_graphcount[k] == 1)
        faulting_uniqoth = sorted(
            k for k in faulting["diff_keys_other"] if keyoth_graphcount[k] == 1)
        _reuse_diag_marker(
            "e11fault", "desc=%s args=%d equal=%d diff8=%d "
            "(pres_invalid=%d same_base=%d both_valid_diff_base=%d) diff_other=%d "
            "node_mismatch=%d UNIQUE(diff8=%s diff_other=%s)"
            % (faulting["desc"], faulting["n_args"], faulting["n_equal"],
               faulting["n_diff8"], faulting["n_diff8_pres_invalid"],
               faulting["n_diff8_same_base"],
               faulting["n_diff8_both_valid_diff_base"],
               faulting["n_diff_other"], faulting["node_mismatch"],
               faulting_uniq8, faulting_uniqoth))
        log.info("E11 FAULT UNIQUE diff8=%s diff_other=%s (keys differing ONLY in the "
                 "faulting graph -- prime stale-rebind suspects)",
                 faulting_uniq8, faulting_uniqoth)
        log.info("E11 FAULTING desc=%s stats={args:%d equal:%d diff8:%d "
                 "pres_invalid:%d same_base:%d both_valid_diff_base:%d diff_other:%d "
                 "node_mismatch:%d}", faulting["desc"], faulting["n_args"],
                 faulting["n_equal"], faulting["n_diff8"],
                 faulting["n_diff8_pres_invalid"], faulting["n_diff8_same_base"],
                 faulting["n_diff8_both_valid_diff_base"], faulting["n_diff_other"],
                 faulting["node_mismatch"])
        for (func, ni, ai, pv, fv, pvld, fvld) in faulting["diff8"][:64]:
            log.info("E11 FAULT diff8 func=%s ni=%d arg=%d pres=0x%x(valid=%s) "
                     "fresh=0x%x(valid=%s)", func, ni, ai, pv, pvld, fv, fvld)
        for (func, ni, ai, sz, phex, fhex, words) in faulting["diff_other"][:32]:
            log.info("E11 FAULT diff_other func=%s ni=%d arg=%d sz=%d words=%s "
                     "pres=%s fresh=%s", func, ni, ai, sz, words, phex, fhex)
    else:
        _reuse_diag_marker("e11fault", "NO_FAULTING_DESC_MATCHED "
                           "(no matched graph with desc nt==nr==1)")

    # sibling manager graphs (desc != None) as controls
    for r in per_graph:
        if r["desc"] is not None and r is not faulting:
            log.info("E11 SIBLING desc=%s how=%s args=%d diff8=%d "
                     "(pres_invalid=%d both_valid_diff_base=%d) diff_other=%d",
                     r["desc"], r["how"], r["n_args"], r["n_diff8"],
                     r["n_diff8_pres_invalid"], r["n_diff8_both_valid_diff_base"],
                     r["n_diff_other"])

    return {"ok": True, "totals": tot,
            "faulting": (None if faulting is None else {
                **{k: faulting[k] for k in (
                    "desc", "n_args", "n_equal", "n_diff8",
                    "n_diff8_pres_invalid", "n_diff8_same_base",
                    "n_diff8_both_valid_diff_base", "n_diff_other",
                    "node_mismatch")},
                "unique_diff8": [list(k) for k in faulting_uniq8],
                "unique_diff_other": [list(k) for k in faulting_uniqoth]})}


# ---------------------------------------------------------------------------
# E8 (SEMIP_E8_NODE_INV): node-class inventory over every captured graph. E4 proved
# KERNEL args + MEMCPY endpoints valid post-reinit; E7 pinned the fault to a fresh
# FULL-graph re-instantiate (H_topology). So the stale VA is in a node class E4 never
# walked. This histograms node types and classifies the baked target of the skipped
# classes (MEMSET.dst, MEM_FREE.dptr) + counts MEM_ALLOC (graph-owned mempool) and
# recurses one level into CHILD_GRAPH. Read-only; never raises. codex resume12.
# ---------------------------------------------------------------------------
def _e8_memset_params(cu, node, params) -> bool:
    """Fill _MemsetNodeParams for a MEMSET node. driver 13000 exposes only the base
    cuGraphMemsetNodeGetParams (no _v2), so the 6-field v1 struct is exact. Returns
    True on rc==0."""
    for sym in ("cuGraphMemsetNodeGetParams", "cuGraphMemsetNodeGetParams_v2"):
        fn = getattr(cu, sym, None)
        if fn is None:
            continue
        try:
            if fn(node, ctypes.byref(params)) == 0:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _e8_memfree_dptr(cu, node, dptr) -> bool:
    fn = getattr(cu, "cuGraphMemFreeNodeGetParams", None)
    if fn is None:
        return False
    try:
        return fn(node, ctypes.byref(dptr)) == 0
    except Exception:  # noqa: BLE001
        return False


def _e8_child_graph(cu, node, sub) -> bool:
    fn = getattr(cu, "cuGraphChildGraphNodeGetGraph", None)
    if fn is None:
        return False
    try:
        return fn(node, ctypes.byref(sub)) == 0
    except Exception:  # noqa: BLE001
        return False


def _e8_collect_graphs_tagged(worker=None):
    """(kind, cudaGraph_t_int) for every captured torch.cuda.CUDAGraph, tagged by
    where it came from: 'wrapper_full' (CUDAGraphWrapper), 'piecewise'
    (BreakableCUDAGraphWrapper), or 'manager_full' (the runner's
    cudagraph_manager.graphs). Returns (pairs, diag).

    The third kind is the one that matters and it needs `worker`. The wrapper kinds
    used to be labelled 'full'/'piecewise' by wrapper class, but with breakable
    cudagraphs enabled the real FULL graphs sit in NO wrapper -- they are bare
    torch.cuda.CUDAGraph objects in the manager dict (resume18). So the old 'full'
    bucket was mislabelled and E8's histogram never covered a FULL graph, which is
    exactly the gap to close before calling the CA rewrite complete for them."""
    pairs = []
    diag = {"n_wrappers": 0, "n_full_wrappers": 0, "n_pw_wrappers": 0,
            "n_manager_graphs": 0, "n_graphs": 0, "err": None}
    try:
        import torch
        cls = torch.cuda.CUDAGraph
        tagged = []
        try:
            from vllm.compilation.cuda_graph import CUDAGraphWrapper
            tagged += [("wrapper_full", w) for w in CUDAGraphWrapper._all_instances]
        except Exception as ex:  # noqa: BLE001
            diag["err"] = f"import CUDAGraphWrapper: {type(ex).__name__}: {ex}"
        try:
            from vllm.compilation.breakable_cudagraph import (
                BreakableCUDAGraphWrapper)
            tagged += [("piecewise", w)
                       for w in BreakableCUDAGraphWrapper._all_instances]
        except Exception:  # noqa: BLE001
            pass
        diag["n_wrappers"] = len(tagged)
        diag["n_full_wrappers"] = sum(1 for k, _ in tagged if k == "wrapper_full")
        diag["n_pw_wrappers"] = sum(1 for k, _ in tagged if k == "piecewise")
        seen = set()
        for kind, w in tagged:
            for attr in ("concrete_cudagraph_entries", "entries"):
                m = getattr(w, attr, None)
                if not isinstance(m, dict):
                    continue
                for entry in m.values():
                    vals = (list(vars(entry).values())
                            if hasattr(entry, "__dict__") else [])
                    objs = []
                    for v in vals:
                        if isinstance(v, cls):
                            objs.append(v)
                        elif isinstance(v, (list, tuple)):
                            objs += [it for it in v if isinstance(it, cls)]
                    for o in objs:
                        if id(o) in seen:
                            continue
                        seen.add(id(o))
                        try:
                            g = int(o.raw_cuda_graph())
                        except Exception:  # noqa: BLE001 - no topology kept
                            continue
                        if g:
                            pairs.append((kind, g))
        # The FULL graphs: in no registry, so reachable only via the manager.
        # `_graph_manager_holders` returns (managers, why); `why` is only useful
        # to the main enumerator's diag, so drop it here.
        for mgr in _graph_manager_holders(worker)[0]:
            graphs = getattr(mgr, "graphs", None)
            if not isinstance(graphs, dict):
                continue
            for o in graphs.values():
                if not isinstance(o, cls) or id(o) in seen:
                    continue
                seen.add(id(o))
                try:
                    g = int(o.raw_cuda_graph())
                except Exception:  # noqa: BLE001
                    continue
                if g:
                    diag["n_manager_graphs"] += 1
                    pairs.append(("manager_full", g))
        diag["n_graphs"] = len(pairs)
    except Exception as e:  # noqa: BLE001
        diag["err"] = f"{type(e).__name__}: {e}"
    return pairs, diag


def e8_node_inventory(worker) -> dict:
    """E8 POST hook (rebind_after_reinit, post-reinit, after CA rewrite/refresh,
    before warmup). Histogram node types across EVERY captured graph (split
    full-vs-piecewise), then classify the baked target of the node classes E4 skipped:
    MEMSET.dst and MEM_FREE.dptr against the live VA map (via _e4_classify), COUNT
    MEM_ALLOC nodes (graph-owned cudaMallocAsync mempool -- presence alone is decisive:
    instantiate re-reserves VA), and recurse ONE level into CHILD_GRAPH nodes. Goal:
    name the node class carrying the stale VA that survives the KERNEL/MEMCPY-only E4
    rebind. Read-only; never raises. Consensus: codex resume12."""
    out = {"step": "e8_node_inventory"}
    try:
        import torch
        torch.cuda.synchronize()
        cu = _cu_ipc(_cu())
        pairs, cdiag = _e8_collect_graphs_tagged(worker)
        out["collect"] = cdiag
        # Human-readable inside-live-alloc label set (same basis as E4 POST): torch
        # segments + the NEW CA ranges. A valid ptr outside these is 'outside' (a
        # non-torch/non-CA driver allocation), NOT a fault -- only invalid==unmapped is.
        ca = _find_ca(worker)
        label_ranges = _e4_torch_segments() + _e4_ca_ranges(cu, ca)
        label_ranges.sort()
        label_los = [r[0] for r in label_ranges]
        cache: dict = {}
        hist: dict = {}     # kind -> {type_name: count}
        memset = {"n": 0, "invalid": 0, "outside": 0, "samples": []}
        memfree = {"n": 0, "invalid": 0, "outside": 0, "samples": []}
        memalloc = {"n": 0}
        child = {"n": 0, "recursed": 0}
        SAMP = 12

        def _classify_ptr(p):
            c = _e4_classify(cu, p, cache)
            inside = bool(c["valid"]
                          and _e4_in_ranges(c["base"], label_los, label_ranges))
            return c, inside

        def _walk(kind, g, depth):
            kh = hist.setdefault(kind, {})
            try:
                nodes = _graph_nodes(cu, g)
            except Exception:  # noqa: BLE001
                return
            for node in nodes:
                t = ctypes.c_int(-1)
                if cu.cuGraphNodeGetType(node, ctypes.byref(t)) != 0:
                    continue
                tv = t.value
                tn = _CU_NODE_TYPE_NAMES.get(tv, "type%d" % tv)
                kh[tn] = kh.get(tn, 0) + 1
                if tv == _CU_NODE_TYPE_MEMSET:
                    p = _MemsetNodeParams()
                    if _e8_memset_params(cu, node, p):
                        memset["n"] += 1
                        dst = int(p.dst)
                        if dst:
                            c, inside = _classify_ptr(dst)
                            if not c["valid"]:
                                memset["invalid"] += 1
                            elif not inside:
                                memset["outside"] += 1
                            if (len(memset["samples"]) < SAMP
                                    and (not c["valid"] or not inside)):
                                memset["samples"].append(
                                    {"kind": kind, "dst": hex(dst),
                                     "valid": c["valid"],
                                     "base": (hex(c["base"]) if c["base"]
                                              else None),
                                     "size": c["size"], "inside": inside})
                elif tv == _CU_NODE_TYPE_MEM_FREE:
                    dptr = ctypes.c_uint64(0)
                    if _e8_memfree_dptr(cu, node, dptr):
                        memfree["n"] += 1
                        p = int(dptr.value)
                        if p:
                            c, inside = _classify_ptr(p)
                            if not c["valid"]:
                                memfree["invalid"] += 1
                            elif not inside:
                                memfree["outside"] += 1
                            if (len(memfree["samples"]) < SAMP
                                    and (not c["valid"] or not inside)):
                                memfree["samples"].append(
                                    {"kind": kind, "dptr": hex(p),
                                     "valid": c["valid"], "inside": inside})
                elif tv == _CU_NODE_TYPE_MEM_ALLOC:
                    # COUNT only: CUDA_MEM_ALLOC_NODE_PARAMS is deeply nested; a
                    # mis-sized struct passed to the getter would let the driver write
                    # past our buffer inside the frozen rebind. Count is the decisive
                    # signal (mempool world -> fix likely recapture-FULL). Reflect dptr
                    # in a follow-up only if this is >0.
                    memalloc["n"] += 1
                elif tv == _CU_NODE_TYPE_GRAPH and depth == 0:
                    child["n"] += 1
                    sub = ctypes.c_void_p(0)
                    if _e8_child_graph(cu, node, sub) and sub.value:
                        child["recursed"] += 1
                        _walk(kind + "/child", int(sub.value), depth + 1)

        for kind, g in pairs:
            _walk(kind, g, 0)

        out["ok"] = True
        out["node_hist"] = {k: dict(sorted(v.items(), key=lambda kv: -kv[1]))
                            for k, v in hist.items()}
        out["memset"] = memset
        out["mem_free"] = memfree
        out["mem_alloc"] = memalloc
        out["child_graph"] = child
        log.info("E8_NODE_INV graphs=%s (full_wrappers=%s pw_wrappers=%s) "
                 "node_hist=%s",
                 cdiag.get("n_graphs"), cdiag.get("n_full_wrappers"),
                 cdiag.get("n_pw_wrappers"), out["node_hist"])
        log.info("E8_NODE_INV memset(n=%s invalid=%s outside=%s) "
                 "mem_free(n=%s invalid=%s outside=%s) mem_alloc(n=%s) "
                 "child_graph(n=%s recursed=%s)",
                 memset["n"], memset["invalid"], memset["outside"],
                 memfree["n"], memfree["invalid"], memfree["outside"],
                 memalloc["n"], child["n"], child["recursed"])
        if memset["samples"]:
            log.info("E8_NODE_INV memset SUSPECTS: %s", memset["samples"])
        if memfree["samples"]:
            log.info("E8_NODE_INV mem_free SUSPECTS: %s", memfree["samples"])
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"
        log.warning("e8_node_inventory failed: %s", out["error"])
    return out


# ---------------------------------------------------------------------------
# E5a (SEMIP_CA_E5_REINSTANTIATE): re-instantiate the kept cudaGraph_t into a FRESH
# cudaGraphExec_t and launch THAT in place of the preserved (possibly stale) exec.
# See _ca_e5_reinstantiate_enabled. The working reuse path is byte-untouched when the
# gate is off; all state lives in _E5_STATE and is only populated by the gated call.
# ---------------------------------------------------------------------------
_E5_STATE = {"map": {}, "orig_replay": None, "cu": None, "keep": [],
             "n_launched": 0, "n_fallthrough": 0}


def _e5_collect_graph_objs():
    """Return the torch.cuda.CUDAGraph OBJECTS vLLM 0.24 holds (not the (g,e) int pairs
    _find_captured_graphs returns), because E5a must intercept each object's .replay()
    by identity. Mirrors _find_captured_graphs' freeze-immune wrapper enumeration --
    DELIBERATELY duplicated so all E5 risk stays isolated to the gated path. Returns
    (objs, diag)."""
    objs = []
    diag = {"n_wrappers": 0, "n_entries": 0, "n_objs": 0, "src": None, "err": None}
    try:
        import torch
        cls = torch.cuda.CUDAGraph
        wrappers = []
        try:
            from vllm.compilation.cuda_graph import CUDAGraphWrapper
            wrappers += list(CUDAGraphWrapper._all_instances)
        except Exception as ex:  # noqa: BLE001
            diag["err"] = f"import CUDAGraphWrapper: {type(ex).__name__}: {ex}"
        try:
            from vllm.compilation.breakable_cudagraph import (
                BreakableCUDAGraphWrapper)
            wrappers += list(BreakableCUDAGraphWrapper._all_instances)
        except Exception:  # noqa: BLE001
            pass
        diag["n_wrappers"] = len(wrappers)
        diag["src"] = "vllm_wrappers"
        seen = set()
        for w in wrappers:
            for attr in ("concrete_cudagraph_entries", "entries"):
                m = getattr(w, attr, None)
                if not isinstance(m, dict):
                    continue
                for entry in m.values():
                    diag["n_entries"] += 1
                    vals = (list(vars(entry).values())
                            if hasattr(entry, "__dict__") else [])
                    for v in vals:
                        if isinstance(v, cls):
                            if id(v) not in seen:
                                seen.add(id(v)); objs.append(v)
                        elif isinstance(v, (list, tuple)):
                            for it in v:
                                if isinstance(it, cls) and id(it) not in seen:
                                    seen.add(id(it)); objs.append(it)
        diag["n_objs"] = len(objs)
    except Exception as e:  # noqa: BLE001
        diag["err"] = f"{type(e).__name__}: {e}"
    return objs, diag


def _e5_install_replay_patch(cu):
    """Install a one-time class-level torch.cuda.CUDAGraph.replay override: for any
    object whose id() is in _E5_STATE['map'], launch the FRESH exec via cuGraphLaunch
    on the CURRENT stream (matching vLLM's single-stream capture/replay) instead of the
    preserved exec. Objects not in the map -- and any synchronous launch error -- fall
    through to the original replay, so a graph we did not re-instantiate degrades to
    normal reuse rather than crashing. (The IMA under test is ASYNC: cuGraphLaunch
    returns 0 and the fault surfaces at the next sync, exactly as with the preserved
    exec -- so H_exec-wrong still reproduces, it does not silently fall back.)"""
    import torch
    if _E5_STATE["orig_replay"] is not None:
        return  # already installed this process
    orig = torch.cuda.CUDAGraph.replay
    _E5_STATE["orig_replay"] = orig

    def _replay(self):
        ne = _E5_STATE["map"].get(id(self))
        if ne:
            try:
                s = torch.cuda.current_stream().cuda_stream
                rc = cu.cuGraphLaunch(ctypes.c_void_p(ne), ctypes.c_void_p(s))
                if rc == 0:
                    n = _E5_STATE["n_launched"] = _E5_STATE["n_launched"] + 1
                    if n == 1:
                        log.info("E5 FIRST FRESH LAUNCH ok (id=%s exec=0x%x) -- "
                                 "patch engaged, replay routed to fresh exec",
                                 id(self), ne)
                    elif n % 512 == 0:
                        log.info("E5 fresh launches so far: %d", n)
                    return
                log.warning("E5 cuGraphLaunch(fresh) rc=%s -> fallback", rc)
            except Exception as ex:  # noqa: BLE001
                log.warning("E5 replay override failed: %s -> fallback", ex)
        else:
            nf = _E5_STATE["n_fallthrough"] = _E5_STATE["n_fallthrough"] + 1
            if nf == 1:
                log.info("E5 replay FALLTHROUGH (unmapped id=%s) -- this graph "
                         "replays via the PRESERVED exec (not covered by E5a)",
                         id(self))
        return orig(self)

    torch.cuda.CUDAGraph.replay = _replay


def e5_reinstantiate_and_swap(worker) -> dict:
    """Build a FRESH cudaGraphExec_t from each kept (already-rebound) cudaGraph_t and
    route replay to it. Read-mostly: creates new execs + swaps the LAUNCHED handle for
    the gated graphs; never destroys/mutates the preserved execs or the graph topology.
    Runs AFTER the normal CA rewrite/refresh (so the kept cudaGraph_t CA nodes are
    already topology-patched -- ckpt-wake addr_map is non-empty) and BEFORE the warmup
    replay. Returns counters."""
    out = {"step": "e5_reinstantiate", "gate": "SEMIP_CA_E5_REINSTANTIATE"}
    try:
        import torch
        cu = _cu()
        _E5_STATE["cu"] = cu
        objs, diag = _e5_collect_graph_objs()
        out["discovery"] = diag
        n_ok = n_topo_fail = n_uninst = n_inst_fail = 0
        errs = []
        for o in objs:
            try:
                g = int(o.raw_cuda_graph())
            except Exception:  # noqa: BLE001 - no kept topology (keep_graph False)
                n_topo_fail += 1
                continue
            try:
                _ = int(o.raw_cuda_graph_exec())  # skip not-yet-instantiated shapes
            except Exception:  # noqa: BLE001
                n_uninst += 1
                continue
            new_exec = ctypes.c_void_p()
            rc = cu.cuGraphInstantiateWithFlags(ctypes.byref(new_exec),
                                                ctypes.c_void_p(g),
                                                ctypes.c_uint64(0))
            if rc != 0 or not new_exec.value:
                n_inst_fail += 1
                if len(errs) < 6:
                    errs.append(f"instantiate rc={rc}")
                continue
            try:  # optional pre-upload; the first launch would upload anyway
                st = torch.cuda.current_stream().cuda_stream
                cu.cuGraphUpload(new_exec, ctypes.c_void_p(st))
            except Exception:  # noqa: BLE001
                pass
            _E5_STATE["map"][id(o)] = int(new_exec.value)
            _E5_STATE["keep"].append((o, new_exec))  # strong refs (obj + handle)
            n_ok += 1
        out["n_objs"] = len(objs)
        out["n_reinstantiated"] = n_ok
        out["n_topo_fail"] = n_topo_fail
        out["n_uninstantiated"] = n_uninst
        out["n_instantiate_fail"] = n_inst_fail
        if errs:
            out["errors"] = errs
        if n_ok:
            _e5_install_replay_patch(cu)
            out["replay_patched"] = True
        out["ok"] = True
        log.info("E5 REINSTANTIATE: objs=%d reinstantiated=%d topo_fail=%d "
                 "uninst=%d inst_fail=%d patched=%s", len(objs), n_ok, n_topo_fail,
                 n_uninst, n_inst_fail, out.get("replay_patched", False))
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"
        log.warning("e5_reinstantiate_and_swap failed: %s", out["error"])
    return out


# ---------------------------------------------------------------------------
# E6 (SEMIP_REUSE_DIAG): pure-diagnostic replay counters. codex resume10
# consensus: E5a's class-level torch.cuda.CUDAGraph.replay patch NEVER fired
# during the faulting post-rebind warmup waves because the PIECEWISE path
# launches segments via bound methods captured at cold-start capture_end()
# (vllm/compilation/breakable_cudagraph.py:190 -- BreakableCUDAGraphCapture
# .segments is a list of bound CUDAGraph.replay callables, consumed by
# BreakableCUDAGraphCapture.replay() at :212 via entry.capture.replay() at
# :423). A post-restore class reassignment cannot intercept an already-bound
# method. E6 counts, WITHOUT changing behavior, how many piecewise replays
# (patch the dynamically-dispatched BreakableCUDAGraphCapture.replay method)
# and FULL replays (count-only patch of torch.cuda.CUDAGraph.replay) run before
# the IMA -> names the failing surface. The working reuse path is byte-untouched
# when the gate is off; all state lives in _REUSE_DIAG_STATE.
# ---------------------------------------------------------------------------
_REUSE_DIAG_STATE = {"pw_calls": 0, "pw_seg_calls": 0, "full_calls": 0,
                     "rfg_calls": 0, "pw_patched": False, "full_patched": False,
                     "rfg_patched": False}

# E8 (resume12 Q3): descriptor keys (num_tokens, num_reqs, uniform_token_count)
# already seen by run_fullgraph, so each marker can flag first-vs-repeat replay for
# that desc -- disambiguates C-(i) one bad static desc from C-(ii) progressive
# corruption. Behavior-neutral; only read/written in the run_fullgraph diag patch.
_SEEN_DESCS: set = set()

# E7 (resume11 Q3): id() of every torch.cuda.CUDAGraph that already had a live
# cudaGraphExec at rebind time (the "130 preserved execs"). Populated in
# _find_captured_graphs when reuse-diag is on. A FULL replay whose id() is NOT in
# this set is a FRESH lazy-instantiate of one of the 3185 uninstantiated shapes ->
# distinguishes H_exec (stale preserved exec) from H_topology (fresh exec built
# from a preserved-but-rebound graph). Behavior-neutral; only read for logging.
_PRESERVED_EXEC_IDS = set()

# E9 (resume13 Q3): handle to the live GPUModelRunner, stashed in
# rebind_after_reinit (same worker process) so the run_fullgraph diag hook -- which
# only receives `self`=CudaGraphManager, not the runner -- can reach the eager-written
# index tensors (input_buffers.seq_lens, block_tables.*) and the KV-cache config to
# bounds-check the H4 level-2 indirection just before the faulting replay. Set only
# when SEMIP_E9_H4=1; otherwise stays None (hook is a no-op). Read-only use.
_E9_RUNNER = None
# One-shot latch: only dump the FIRST time we see the faulting decode desc, so we
# don't add a per-step host<->device sync to every subsequent replay.
_E9_DUMPED: set = set()


def _reuse_diag_enabled() -> bool:
    return os.environ.get("SEMIP_REUSE_DIAG", "") == "1"


def _reuse_diag_marker(kind: str, detail: str = "") -> None:
    """E7: crash-safe (os.write + os.fsync) one-line marker so the LAST replay
    surface executed before an async IMA / abrupt terminate survives even when the
    worker's buffered stdout is lost. Written BEFORE the wrapped replay launches, so
    under CUDA_LAUNCH_BLOCKING=1 the final line names the faulting surface. One file
    per worker pid. Best-effort; never raises into the replay path."""
    if not _reuse_diag_enabled():
        return
    try:
        path = "/semiplog/save/reuse_diag_%d.marker" % os.getpid()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
        try:
            os.write(fd, ("%s %s\n" % (kind, detail)).encode())
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:  # noqa: BLE001
        pass


def _e9_h4_dump(key, seq: int) -> None:
    """E9 (resume13 Q3): H4 level-2-indirection bounds-check. Immediately BEFORE the
    faulting FULL replay of the smallest decode desc, read the eager-written decode
    index tensors off the live GPUModelRunner and bounds-check them against the
    KV-cache capacity:
      * block_table  = block_tables.input_block_tables[g][req]  (what the FULL graph
                       actually reads; each entry must be a valid physical block id in
                       [0, kv_cache_config.num_blocks));
      * slot_mapping = block_tables.slot_mappings[g][:num_tokens] (write slot per token;
                       must be in [0, num_blocks * block_size));
      * seq_lens     = input_buffers.seq_lens[req]              (must be sane / > 0);
      * used         = block_tables.num_blocks.np[g, req]       (host UVA -> no sync).
    E8 proved every baked NODE POINTER is mapped+in-range, so if any of these CONTENTS
    are OOB it names H4 as the fault (a valid pointer indexing a stale/OOB location).
    All tensors read here are small, post-reinit-valid torch/UVA buffers -- NOT the
    paged KV the graph faults on -- and are read while the CUDA context is still clean
    (before the replay), so this dump never itself faults. Crash-safe marker survives
    the subsequent IMA. Read-only. Caller wraps in try/except; we also self-guard."""
    runner = _E9_RUNNER
    if runner is None:
        _reuse_diag_marker("e9h4", "seq=%d skip=no_runner" % seq)
        return
    try:
        kvc = getattr(runner, "kv_cache_config", None)
        total_blocks = int(getattr(kvc, "num_blocks", -1)) if kvc is not None else -1
        bt = getattr(runner, "block_tables", None)
        ib = getattr(runner, "input_buffers", None)
        if bt is None or ib is None:
            _reuse_diag_marker("e9h4", "seq=%d skip=no_bt_or_ib bt=%s ib=%s"
                               % (seq, bt is not None, ib is not None))
            return
        ngroups = getattr(bt, "num_kv_cache_groups", 1)
        bsz = getattr(bt, "block_sizes", None)
        block_size = int(bsz[0]) if bsz else -1
        total_slots = (total_blocks * block_size
                       if (total_blocks > 0 and block_size > 0) else -1)
        # req-0 used-block count -- host UVA numpy view, no device sync.
        try:
            used = int(bt.num_blocks.np[0, 0])
        except Exception:  # noqa: BLE001
            used = -1
        # bounded window of the physical block ids the FULL graph reads for req 0.
        row = bt.input_block_tables[0][0]
        width = int(row.shape[0])
        win = used if (0 < used <= width) else min(width, 128)
        win = max(win, 8)
        ids = row[:win].detach().to("cpu")
        bmin = int(ids.min().item()); bmax = int(ids.max().item())
        b_oob = (int(((ids < 0) | (ids >= total_blocks)).sum().item())
                 if total_blocks > 0 else -1)
        # slot mapping for the decode token(s) of this desc (group 0).
        ntok = key[0] if (isinstance(key[0], int) and key[0] > 0) else 1
        sm = bt.slot_mappings[0][:max(ntok, 1)].detach().to("cpu")
        smin = int(sm.min().item()); smax = int(sm.max().item())
        s_oob = (int(((sm < 0) | (sm >= total_slots)).sum().item())
                 if total_slots > 0 else -1)
        # seq_lens for req 0.
        try:
            seq_len0 = int(ib.seq_lens[:1].detach().to("cpu")[0].item())
        except Exception:  # noqa: BLE001
            seq_len0 = -1
        _reuse_diag_marker(
            "e9h4",
            "seq=%d desc=(nt=%s,nr=%s,utc=%s) ngroups=%s total_blocks=%d "
            "block_size=%d total_slots=%d used=%d win=%d blk[min=%d,max=%d,oob=%d] "
            "slot[min=%d,max=%d,oob=%d] seq_len0=%d"
            % (seq, key[0], key[1], key[2], ngroups, total_blocks, block_size,
               total_slots, used, win, bmin, bmax, b_oob, smin, smax, s_oob,
               seq_len0))
        log.info("REUSE_DIAG E9 H4 desc=%s total_blocks=%d block_size=%d used=%d "
                 "blk[%d..%d oob=%d] slot[%d..%d oob=%d] seq_len0=%d",
                 key, total_blocks, block_size, used, bmin, bmax, b_oob,
                 smin, smax, s_oob, seq_len0)
    except Exception as ex:  # noqa: BLE001 - never perturb the replay path
        _reuse_diag_marker("e9h4", "seq=%d error=%r" % (seq, ex))


def install_reuse_diag_counters() -> dict:
    """Install count-only patches on the PIECEWISE and FULL replay entry points so
    we can see, from the worker log, which replay kind runs during the post-rebind
    warmup waves (and thus which one carries the IMA). Behavior-preserving: every
    patched method still calls the original. Idempotent per process."""
    import torch
    st = _REUSE_DIAG_STATE
    # PIECEWISE: BreakableCUDAGraphCapture.replay is a plain class method, invoked
    # via entry.capture.replay() (dynamic lookup) -> a class patch IS honored here,
    # unlike the bound-method segments it iterates internally.
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
        if not st["pw_patched"]:
            _orig_pw = BreakableCUDAGraphCapture.replay

            def _diag_pw_replay(self):
                n = st["pw_calls"] = st["pw_calls"] + 1
                try:
                    nseg = len(self.segments)
                    st["pw_seg_calls"] += nseg
                except Exception:  # noqa: BLE001
                    nseg = -1
                # E7: crash-safe marker BEFORE the launch -> last line names the
                # faulting surface under CUDA_LAUNCH_BLOCKING=1.
                _reuse_diag_marker("piecewise", "seq=%d segments=%d" % (n, nseg))
                if n == 1:
                    log.info("REUSE_DIAG FIRST piecewise replay (segments=%s) -- "
                             "PIECEWISE replay IS on the warmup path", nseg)
                elif n % 256 == 0:
                    log.info("REUSE_DIAG piecewise replays so far: %d (seg_calls=%d)",
                             n, st["pw_seg_calls"])
                return _orig_pw(self)

            BreakableCUDAGraphCapture.replay = _diag_pw_replay
            st["pw_patched"] = True
    except Exception as ex:  # noqa: BLE001
        log.warning("REUSE_DIAG piecewise patch failed: %s", ex)
    # FULL: entry.cudagraph.replay() (cuda_graph.py:360) is a dynamic lookup on the
    # torch CUDAGraph instance -> a class patch IS honored. Skip if E5a already owns
    # this attr (its n_launched/n_fallthrough already count FULL replays).
    try:
        if not st["full_patched"] and not _ca_e5_reinstantiate_enabled():
            _orig_full = torch.cuda.CUDAGraph.replay

            def _diag_full_replay(self):
                n = st["full_calls"] = st["full_calls"] + 1
                # E7 (Q3): was this exec preserved pre-rebind (H_exec) or freshly
                # lazy-instantiated post-reinit (H_topology)?
                preserved = id(self) in _PRESERVED_EXEC_IDS
                # E9 (resume13 Q3): log id(self) on EVERY low-level full replay.
                # self.graphs[desc] IS this torch.cuda.CUDAGraph, so id(self) here
                # matches the rfg marker's gid -- letting us see whether the faulting
                # decode graph (its gid from the rfg line) already replayed OK during
                # warmup (input-dependent fault, H4) or faulted on first instantiate.
                _reuse_diag_marker(
                    "full", "seq=%d preserved_exec=%s gid=0x%x"
                    % (n, preserved, id(self)))
                if n == 1:
                    log.info("REUSE_DIAG FIRST full-graph replay -- FULL replay IS "
                             "on the warmup path (preserved_exec=%s)", preserved)
                return _orig_full(self)

            torch.cuda.CUDAGraph.replay = _diag_full_replay
            st["full_patched"] = True
    except Exception as ex:  # noqa: BLE001
        log.warning("REUSE_DIAG full patch failed: %s", ex)
    # E8 (resume12 Q3): the torch.cuda.CUDAGraph.replay hook above sees only `self`,
    # not the batch descriptor. Patch the LEVEL ABOVE it -- CudaGraphManager.run_fullgraph
    # (base class; the 0.24 subclass calls super()) -- to write a crash-safe marker
    # naming the exact `desc` (num_tokens, num_reqs, uniform_token_count), the graph
    # object id + whether its exec was preserved pre-rebind, and first-vs-repeat for
    # that desc. Under CUDA_LAUNCH_BLOCKING=1 the LAST "rfg" marker names the faulting
    # descriptor, so we can tell one bad static desc (C-i) from progressive corruption
    # (C-ii) and map the 66 FULL replays to distinct shapes. Marker BEFORE super().
    try:
        from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
        if not st["rfg_patched"]:
            _orig_rfg = CudaGraphManager.run_fullgraph

            def _diag_run_fullgraph(self, desc):
                n = st["rfg_calls"] = st["rfg_calls"] + 1
                try:
                    key = (getattr(desc, "num_tokens", None),
                           getattr(desc, "num_reqs", None),
                           getattr(desc, "uniform_token_count", None))
                    g = None
                    try:
                        g = self.graphs.get(desc)
                    except Exception:  # noqa: BLE001
                        g = None
                    gid = id(g) if g is not None else 0
                    preserved = gid in _PRESERVED_EXEC_IDS
                    first = key not in _SEEN_DESCS
                    _SEEN_DESCS.add(key)
                    _reuse_diag_marker(
                        "rfg", "seq=%d desc=(nt=%s,nr=%s,utc=%s) gid=0x%x "
                        "preserved_exec=%s first=%s"
                        % (n, key[0], key[1], key[2], gid, preserved, first))
                    if first:
                        log.info("REUSE_DIAG rfg FIRST replay of desc=%s "
                                 "(gid=0x%x preserved_exec=%s seq=%d)",
                                 key, gid, preserved, n)
                    # E9 (resume13 Q3): H4 bounds-check on the FIRST replay of each
                    # small decode-shaped desc (nt==nr, <=8) -- captures the E7/E8
                    # faulting (1,1,1) -- dumped BEFORE the (possibly faulting) replay.
                    # Latched per key so we add the host<->device sync at most once per
                    # desc, never on the steady-state replay path.
                    if (_e9_h4_enabled()
                            and isinstance(key[0], int) and isinstance(key[1], int)
                            and key[0] == key[1] and 0 < key[0] <= 8
                            and key not in _E9_DUMPED):
                        _E9_DUMPED.add(key)
                        _e9_h4_dump(key, n)
                except Exception:  # noqa: BLE001 - never perturb the replay path
                    pass
                return _orig_rfg(self, desc)

            CudaGraphManager.run_fullgraph = _diag_run_fullgraph
            st["rfg_patched"] = True
    except Exception as ex:  # noqa: BLE001
        log.warning("REUSE_DIAG run_fullgraph patch failed: %s", ex)
    return {"pw_patched": st["pw_patched"], "full_patched": st["full_patched"],
            "rfg_patched": st["rfg_patched"]}


def rebind_after_reinit(worker) -> dict:
    """Post-reinit hook (Path 2 / force-copy). The captured graph bakes only
    meta_ptrs (CA kernel nodes) and the static buffer_ptrs[rank] (copy memcpy
    node); the per-buffer RankData is rank_data slot 0, which the new CA's
    register_buffer already refilled at the same (stable) VA. So we just rewrite
    meta_ptrs + buffer_ptrs old->new across the captured graphs."""
    if not enabled():
        return {"enabled": False}
    out: dict = {"enabled": True, "step": "rebind_after_reinit"}
    snap = getattr(worker, _SNAP_ATTR, None)
    if snap is None:
        out["skipped"] = "no snapshot (snapshot step did not run?)"
        log.warning("rebind_after_reinit: %s", out["skipped"])
        return out
    ca = _find_ca(worker)
    if ca is None:
        out["skipped"] = "no ca_comm post-reinit"
        log.warning("rebind_after_reinit: %s", out["skipped"])
        return out
    old_meta = list(snap.get("meta_ptrs", []))
    new_meta = list(getattr(ca, "meta_ptrs", []) or [])
    old_buf = list(snap.get("buffer_ptrs", []))
    new_buf = list(getattr(ca, "buffer_ptrs", []) or [])
    out["meta_old"] = [hex(x) for x in old_meta]
    out["meta_new"] = [hex(x) for x in new_meta]
    out["buf_old"] = [hex(x) for x in old_buf]
    out["buf_new"] = [hex(x) for x in new_buf]
    old_rd = int(snap.get("rank_data", 0) or 0)
    new_rd = (int(ca.rank_data.data_ptr())
              if getattr(ca, "rank_data", None) is not None else 0)
    out["rank_data_old"] = hex(old_rd)
    out["rank_data_new"] = hex(new_rd)
    out["rank_data_moved"] = (old_rd != new_rd)
    # DIAG: distribution of the CA kernels' baked _dp slot offsets in the KEPT
    # graph. offset 0 == copy path (slot-0 staging); >0 == registered peer slots.
    # Tells us whether force-copy actually took during the kept-graph capture
    # (copy_only=True is the goal for the copy-path approach).
    try:
        _slots = _collect_dp_slots(
            _cu(), new_rd,
            int(ca.rank_data.numel() * ca.rank_data.element_size()), worker)
        out["dp_slots_distinct"] = len(_slots)
        out["dp_slot_min"] = (min(_slots) if _slots else None)
        out["dp_slot_max"] = (max(_slots) if _slots else None)
        out["dp_copy_only"] = (_slots == [0])
        log.info("CA _dp slot dist: distinct=%d min=%s max=%s copy_only=%s",
                 len(_slots), out.get("dp_slot_min"), out.get("dp_slot_max"),
                 out.get("dp_copy_only"))
    except Exception as _de:  # noqa: BLE001
        out["dp_diag_err"] = f"{type(_de).__name__}: {_de}"
    addr_map = {}
    for o, n in zip(old_meta, new_meta):
        addr_map[int(o)] = int(n)
    for o, n in zip(old_buf, new_buf):
        addr_map[int(o)] = int(n)
    if old_rd and new_rd and old_rd != new_rd:
        addr_map[old_rd] = new_rd
    try:
        out["rewrite"] = rewrite_addrs_in_graphs(worker, addr_map)
        out["ok"] = bool(out["rewrite"].get("ok"))
        # Copy-path (now unconditional): all captured CA kernels reduce the
        # symmetric staging buffer; replicate the refreshed slot-0 into the
        # stale used slots (no IPC / peer derivation needed).
        out["refresh_copy"] = refresh_copy_path_slots(worker)
        out["ok"] = out["ok"] and bool(out["refresh_copy"].get("ok"))
        log.info("rebind_after_reinit: rank=%s rewrite=%s refresh=%s",
                 getattr(ca, "rank", "?"), out["rewrite"],
                 out.get("refresh_copy"))
        # E4 POST reclassify (SEMIP_CA_E4_INVENTORY=1): re-read the pre-snapshotted
        # baked pointers now (post-reinit, after CA rewrite/refresh, BEFORE warmup)
        # and bucket each. Runs before probe_ca_signal_pads so it captures the stale-
        # pointer state even if the probe's leader sync surfaces the async IMA first.
        # Read-only; e4_post_inventory never raises, so it cannot flip out["ok"].
        if _ca_e4_inventory_enabled():
            out["e4"] = e4_post_inventory(worker)
        # E8 node-class inventory (SEMIP_E8_NODE_INV=1): histogram node types over
        # every captured graph + classify the MEMSET/MEM_FREE/MEM_ALLOC/CHILD_GRAPH
        # classes E4 never walked. Same window as E4 POST (post-reinit, after CA
        # rewrite/refresh, before warmup). Read-only; never flips out["ok"]. resume12.
        if _e8_node_inventory_enabled():
            out["e8"] = e8_node_inventory(worker)
        # CA one-shot Signal-pad diagnostic + optional local zero (env-gated,
        # default OFF). Investigating why reused one-shot all-reduce graphs
        # sometimes deadlock / silently die after restore: measures whether the
        # fresh Signal is zero-initialized, validates the graph-baked sg/self_sg
        # ptrs, and can zero this rank's Signal before the warmup replay. See
        # probe_ca_signal_pads / _ca_zero_signal_enabled.
        if (_ca_pad_probe_enabled() or _ca_zero_signal_enabled()
                or _ca_e3_dump_enabled()):
            out["ca_pad_probe"] = probe_ca_signal_pads(
                worker, do_zero=_ca_zero_signal_enabled())
        # E5a (SEMIP_CA_E5_REINSTANTIATE=1): re-instantiate the kept, now-rebound
        # cudaGraph_t into a FRESH exec and route replay to it (H_exec discriminator).
        # Runs LAST in the rebind body -> after rewrite/refresh/E4/probe and BEFORE the
        # warmup replay that follows rebind_after_reinit, so the fresh exec is the one
        # launched. Read-mostly; never flips out["ok"].
        if _ca_e5_reinstantiate_enabled():
            out["e5"] = e5_reinstantiate_and_swap(worker)
        # E6 (SEMIP_REUSE_DIAG=1): install count-only piecewise/FULL replay
        # counters BEFORE the post-rebind warmup waves so the worker log names
        # which replay kind runs (and carries the IMA). Behavior-preserving.
        if _reuse_diag_enabled():
            out["reuse_diag"] = install_reuse_diag_counters()
        # E9 (resume13 Q3): stash the live GPUModelRunner so the run_fullgraph diag
        # hook (which only receives self=CudaGraphManager) can reach the eager-written
        # decode index tensors + KV-cache config for the H4 bounds-check. Same worker
        # process; read-only handle. Only when SEMIP_E9_H4=1.
        if _e9_h4_enabled():
            global _E9_RUNNER
            _E9_RUNNER = getattr(worker, "model_runner", None)
            out["e9_runner"] = _E9_RUNNER is not None
        # FIX (stale-snapshot drift): keep the snapshot in sync with what the
        # graphs NOW bake. Each rebind rewrites the baked meta/buffer ptrs
        # old->new; leaving snap at the cold-start values means the NEXT rebind
        # (e.g. the disk-wake after this ckpt-wake) builds addr_map from stale keys
        # that no longer match the graphs' baked (previous-rebind) values -> 0
        # patched -> signals NOT rewritten. That is benign only while reinit
        # reproduces the same VAs; when it does not, the stale signal ptrs crash
        # the worker (the intermittent disk-wake death). Syncing snap here makes
        # every subsequent rebind map baked->current correctly regardless of reinit
        # determinism.
        #
        # PER-CATEGORY, not one coupled guard: meta_ptrs (rewritten in KERNEL
        # sg/self_sg nodes -> kernel_nodes_patched) and buffer_ptrs (rewritten in
        # MEMCPY nodes -> memcpy_nodes_patched) are INDEPENDENT create_shared_buffer
        # allocations that can move independently (only rank_data is VA-pinned).
        # Gating both on kernel_nodes_patched would skip the buffer sync whenever
        # the buffer moves but meta reproduces its VA -> the same drift, on the copy
        # path. Each field is synced ONLY when (a) that category actually moved
        # (new != old) and (b) its own rewrite counter fired; the len/non-empty
        # guard stops a degenerate reinit (empty ca.meta_ptrs while kernel patches
        # came from a _dp match) from clobbering snap with [] and latching the crash.
        rw = out.get("rewrite") or {}
        if out["ok"]:
            synced = []
            if (new_meta and len(new_meta) == len(old_meta) and new_meta != old_meta
                    and int(rw.get("kernel_nodes_patched", 0)) > 0):
                snap["meta_ptrs"] = list(new_meta)
                synced.append("meta")
            if (new_buf and len(new_buf) == len(old_buf) and new_buf != old_buf
                    and int(rw.get("memcpy_nodes_patched", 0)) > 0):
                snap["buffer_ptrs"] = list(new_buf)
                synced.append("buffer")
            if (new_rd and new_rd != old_rd
                    and int(rw.get("kernel_nodes_patched", 0)) > 0):
                snap["rank_data"] = new_rd
                synced.append("rank_data")
            if synced:
                setattr(worker, _SNAP_ATTR, snap)
                out["snapshot_synced"] = synced
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"
        log.warning("rebind_after_reinit failed: %s", out["error"])
    return out
