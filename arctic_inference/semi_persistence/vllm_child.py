"""vLLM child process loop.

Spawned by the worker process.  Owns CUDA and vLLM.
Reads (cmd, kwargs) from a pipe, puts results on result_queue.

Init loads real weights (load_format=auto) so that vLLM runs
process_weights_after_loading and produces its internal kernel format
(Marlin-packed for GPTQ, cutlass layout for FP8, plain tensors for
BF16).

The generate path drives LLMEngine directly via add_request() + step()
instead of using the blocking LLM.generate().  This allows the child
to accept new generate requests (and other commands) while the engine
is actively decoding, enabling concurrent request handling without
asyncio or extra threads.

Attach allocates CPU memory sized to model.named_parameters().  Stage
snapshots the post-processed GPU parameters into that buffer.
plan_restore_weights walks the param index once and caches a chunk plan
(chunk_lo, chunk_hi, members) bounded by max_buffer_bytes.
restore_weights then loops over the cached plan: per chunk, copy a
slice of host memory into a single reused GPU staging buffer and
scatter into model parameters by name.  If no plan is cached,
restore_weights falls back to a single-chunk path.

All of that state lives on each vLLM worker (``worker._semip_*``) and
runs there via ``collective_rpc``, not in this process.  At TP>1 the
callable is cloudpickled into every worker subprocess, so a buffer held
here would be copied by value per worker and its writes discarded; each
rank also owns a different shard of the parameters.  The same code path
serves TP=1, where the single worker is in-process.
"""
import ctypes, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor


def _drop_caps_for_portable_image():
    """Zero this process's Linux capabilities so its CRIU image restores on
    nodes whose capability bounding set can't grant them.

    Must run in the spawned vLLM child BEFORE ``import torch`` -- torch/vLLM
    spawn many background threads (cuda, jemalloc, hf-xet, gloo, ...) and
    CRIU records credentials *per thread*.  A thread inherits the creating
    thread's capabilities, so dropping here (before any are created) yields
    an image where every task records an empty cap set; restore_creds()
    (capset) then trivially succeeds instead of failing with EPERM on a
    node lacking CAP_SYS_ADMIN etc.  (Dropping later, e.g. in
    prepare_criu_dump, would only affect the main thread and leave the
    others with full caps.)

    Gated by the internal _SEMIP_CHILD_DROP_CAPS signal (set transiently by
    the worker only across this child's spawn -- NOT a user-facing flag; the
    user-facing switch is SEMIP_UNPRIVILEGED) so ONLY the child drops -- the
    parent worker keeps its caps to run ``sudo criu``.  Safe for inference: host
    pinning uses RLIMIT_MEMLOCK (unlimited on GPU nodes), not CAP_IPC_LOCK,
    and all sockets bind to unprivileged ports.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_CAPBSET_DROP = 24
        PR_CAP_AMBIENT = 47
        PR_CAP_AMBIENT_CLEAR_ALL = 4
        PR_SET_DUMPABLE = 4
        # Drop the bounding set first -- it needs CAP_SETPCAP, which the
        # capset() below removes.  EINVAL past the last valid cap is ignored.
        for cap in range(64):
            libc.prctl(PR_CAPBSET_DROP, cap, 0, 0, 0)
        libc.prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0)

        class _CapHeader(ctypes.Structure):
            _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

        class _CapData(ctypes.Structure):
            _fields_ = [("effective", ctypes.c_uint32),
                        ("permitted", ctypes.c_uint32),
                        ("inheritable", ctypes.c_uint32)]

        _LINUX_CAPABILITY_VERSION_3 = 0x20080522
        hdr = _CapHeader(_LINUX_CAPABILITY_VERSION_3, 0)
        data = (_CapData * 2)()  # zero-initialized -> clears eff/prm/inh
        rc = libc.capset(ctypes.byref(hdr), ctypes.byref(data))
        # Keep the process dumpable so `sudo criu` seizes it cleanly after
        # the credential change (capset can reset dumpable to suid_dumpable).
        libc.prctl(PR_SET_DUMPABLE, 1, 0, 0, 0)
        print(f"[semip] dropped capabilities for portable image "
              f"(capset rc={rc})", flush=True)
    except Exception as e:  # never block startup on a cap-drop failure
        print(f"[semip] cap-drop failed (continuing): {e}", flush=True)


# Internal signal (leading underscore): the worker sets _SEMIP_CHILD_DROP_CAPS
# only in this spawned child's environment when running unprivileged.  It is
# deliberately NOT the user-facing SEMIP_UNPRIVILEGED flag -- the worker itself
# imports this module and must keep its caps to run `sudo criu`.
if os.environ.get("_SEMIP_CHILD_DROP_CAPS") == "1":
    _drop_caps_for_portable_image()

import torch

import semip_logging

# Ensure this package dir is importable in the child + its TP worker
# subprocesses (worker_cls="_semip_worker.SemipGPUWorker" and ca_graph_rebind
# are resolved by bare module name).
_semip_here = os.path.dirname(os.path.abspath(__file__))
if _semip_here not in sys.path:
    sys.path.insert(0, _semip_here)

try:
    import ca_graph_rebind  # dense-TP graph reuse support
    _CA_REBIND_AVAILABLE = True
except Exception as _ca_e:  # best-effort: reuse degrades to full recapture
    _CA_REBIND_AVAILABLE = False
    ca_graph_rebind = None
    print(f"[semip-ca-rebind] unavailable: {_ca_e}", flush=True)


def _truncate_for_display(value, limit=200):
    """Truncate strings (or strings inside a list/tuple) to ``limit`` chars,
    appending ``...(<n> chars)`` when the original exceeds ``limit``.
    """
    if isinstance(value, str):
        if len(value) > limit:
            return f"{value[:limit]}...({len(value)} chars)"
        return value
    if isinstance(value, (list, tuple)):
        out = [_truncate_for_display(v, limit) for v in value]
        return out if isinstance(value, list) else tuple(out)
    return value


# Shard size / parallelism for save_weights + load_weights disk I/O.
_WEIGHTS_SHARD_BYTES = 2 * 2**30   # 2 GiB per shard
_WEIGHTS_IO_WORKERS = 8            # thread pool size for shard I/O

_cudart = ctypes.CDLL("libcudart.so")
_cudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
_cudart.cudaHostUnregister.restype = ctypes.c_int
_cudart.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
_cudart.cudaHostRegister.restype = ctypes.c_int


def _unpin_buffer(buf):
    ret = _cudart.cudaHostUnregister(ctypes.c_void_p(buf.data_ptr()))
    if ret != 0:
        raise RuntimeError(f"cudaHostUnregister failed with cudaError={ret}")


def _repin_buffer(buf):
    ret = _cudart.cudaHostRegister(
        ctypes.c_void_p(buf.data_ptr()),
        ctypes.c_size_t(buf.numel() * buf.element_size()),
        ctypes.c_uint(0),
    )
    if ret != 0:
        raise RuntimeError(f"cudaHostRegister failed with cudaError={ret}")


# ---------------------------------------------------------------------------
# Worker-local semi-persistence primitives (TP-safe).
#
# These run inside each vLLM worker via ``llm.collective_rpc``.  At TP>1 the
# callable is cloudpickled and executed in every worker process, so the
# staging buffer must live on the worker object (``worker._semip_*``), not in
# the vllm_child process -- a captured closure over a vllm_child-local buffer
# would be copied by value per worker and its writes discarded.  The same code
# path works at TP=1 (a single worker, in-process).  ``worker`` is whatever
# ``collective_rpc`` passes (the WorkerWrapperBase driver at TP=1, the real
# Worker at TP>1); attribute reads forward to the underlying Worker either way,
# and attribute writes persist across calls because the object is stable.
#
# Param names are namespaced ``main:p:`` / ``drafter:p:`` to match the layout
# built at attach so stage/restore dispatch each entry to the right tensor.
# ---------------------------------------------------------------------------
def _semip_layout(worker):
    layout = []
    mr = worker.model_runner
    for name, p in mr.model.named_parameters():
        d = p.data
        layout.append((f"main:p:{name}", d.nbytes, d.dtype, tuple(d.shape)))
    drafter = getattr(mr, "drafter", None)
    dm = getattr(drafter, "model", None) if drafter is not None else None
    if dm is not None:
        for name, p in dm.named_parameters():
            d = p.data
            layout.append((f"drafter:p:{name}", d.nbytes, d.dtype, tuple(d.shape)))
    return layout


def _semip_param_tensors(worker):
    mr = worker.model_runner
    t = {f"main:p:{n}": p.data for n, p in mr.model.named_parameters()}
    drafter = getattr(mr, "drafter", None)
    dm = getattr(drafter, "model", None) if drafter is not None else None
    if dm is not None:
        for n, p in dm.named_parameters():
            t[f"drafter:p:{n}"] = p.data
    return t


def _semip_attach(worker):
    layout = _semip_layout(worker)
    total_size = sum(nb for _, nb, _, _ in layout)
    index = {}
    offset = 0
    for name, nbytes, dtype, shape in layout:
        index[name] = (offset, nbytes, dtype, shape)
        offset += nbytes
    worker._semip_buf = torch.empty(total_size, dtype=torch.uint8)
    worker._semip_index = index
    worker._semip_chunk_plan = None
    worker._semip_chunk_size = None
    worker._semip_pinned = False
    return (total_size, len(layout))


def _semip_stage(worker):
    buf = worker._semip_buf
    index = worker._semip_index
    sources = _semip_param_tensors(worker)
    for name, (offset, nbytes, dtype, shape) in index.items():
        src = sources[name].contiguous().reshape(-1).view(torch.uint8)
        buf[offset:offset + nbytes].copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    return buf.numel()


def _semip_unpin(worker):
    # Idempotent: the attach buffer starts unpinned, so a cudaHostUnregister on
    # an unregistered (or already-unpinned) buffer would hard-error.
    buf = getattr(worker, "_semip_buf", None)
    if buf is None or not getattr(worker, "_semip_pinned", False):
        return 0
    _unpin_buffer(buf)
    worker._semip_pinned = False
    return buf.numel()


def _semip_repin(worker):
    # Idempotent: a double cudaHostRegister would hard-error.
    buf = getattr(worker, "_semip_buf", None)
    if buf is None:
        return 0
    if getattr(worker, "_semip_pinned", False):
        return buf.numel()
    _repin_buffer(buf)
    worker._semip_pinned = True
    return buf.numel()


def _semip_plan_load_weights(worker, max_buffer_bytes=None):
    buf = worker._semip_buf
    index = worker._semip_index
    total_bytes = buf.numel()
    cs = total_bytes if max_buffer_bytes is None else min(int(max_buffer_bytes), total_bytes)
    plan = []
    cur = []
    cur_lo = 0
    for name, (off, nbytes, dtype, shape) in index.items():
        if nbytes > cs:
            raise RuntimeError(f"param {name} ({nbytes}B) exceeds chunk_size ({cs}B)")
        if cur and (off + nbytes - cur_lo) > cs:
            cur_hi = cur[-1][1] + cur[-1][2]
            plan.append((cur_lo, cur_hi, cur))
            cur = []
            cur_lo = off
        cur.append((name, off, nbytes, dtype, shape))
    if cur:
        cur_hi = cur[-1][1] + cur[-1][2]
        plan.append((cur_lo, cur_hi, cur))
    worker._semip_chunk_plan = plan
    worker._semip_chunk_size = cs
    return {"bytes": total_bytes, "n_chunks": len(plan), "chunk_size": cs}


def _semip_restore_weights(worker):
    buf = worker._semip_buf
    index = worker._semip_index
    targets = _semip_param_tensors(worker)
    total_bytes = buf.numel()
    plan = worker._semip_chunk_plan
    cs = worker._semip_chunk_size
    if plan is None:
        plan = [(0, total_bytes,
                 [(n, o, nb, dt, sh) for n, (o, nb, dt, sh) in index.items()])]
        cs = total_bytes
    torch.cuda.synchronize()
    gpu_buf = torch.empty(cs, dtype=torch.uint8, device=worker.device)
    loaded = 0
    for chunk_lo, chunk_hi, members in plan:
        n = chunk_hi - chunk_lo
        gpu_buf[:n].copy_(buf[chunk_lo:chunk_hi], non_blocking=True)
        torch.cuda.synchronize()
        for name, off, nbytes, dtype, shape in members:
            start = off - chunk_lo
            src = gpu_buf[start:start + nbytes].view(dtype).reshape(shape)
            targets[name].copy_(src)
            loaded += 1
        torch.cuda.synchronize()
    gpu_buf.storage().resize_(0)
    del gpu_buf
    # Return the staging buffer to the CUDA *driver*, not just to torch's
    # caching allocator.  resize_(0)/free only hands the ~cs-byte block back
    # to torch's pool (cudaMalloc arena); it stays reserved from the driver's
    # point of view.  The next step, wake_up(["kv_cache"]), maps the KV cache
    # via vLLM's cumem allocator (cuMemCreate/cuMemMap), which allocates from
    # driver-free memory -- so a torch-cached staging block (up to the full
    # 50+ GiB weight size at chunk_size == total) starves it and the KV map
    # OOMs.  empty_cache() releases torch's cached blocks so cumem can map.
    torch.cuda.empty_cache()
    return {"bytes": total_bytes, "loaded": loaded,
            "n_chunks": len(plan), "chunk_size": cs}


def _semip_detach(worker):
    buf = getattr(worker, "_semip_buf", None)
    total = buf.numel() if buf is not None else 0
    if buf is not None and getattr(worker, "_semip_pinned", False):
        _unpin_buffer(buf)
    worker._semip_buf = None
    worker._semip_index = None
    worker._semip_chunk_plan = None
    worker._semip_chunk_size = None
    worker._semip_pinned = False
    return total


def _semip_save_weights(worker, weights_dir, shard_bytes=None, io_workers=None):
    buf = worker._semip_buf
    index = worker._semip_index
    shard_bytes = int(shard_bytes or _WEIGHTS_SHARD_BYTES)
    workers = int(io_workers or _WEIGHTS_IO_WORKERS)
    # TP1 keeps the flat weights/ layout; TP>1 fans out per-rank shards into
    # weights/rank{R}/, since each rank holds a different slice of the params.
    rank_dir = (weights_dir if _semip_tp_size(worker) <= 1
                else os.path.join(weights_dir, f"rank{worker.rank}"))
    if os.path.exists(rank_dir):
        try:
            shutil.rmtree(rank_dir)
        except PermissionError:
            subprocess.run(["sudo", "rm", "-rf", rank_dir], check=True)
    os.makedirs(rank_dir, exist_ok=True)

    total = buf.numel()
    mv = memoryview(buf.numpy())
    ranges = []
    lo = 0
    i = 0
    while lo < total:
        hi = min(lo + shard_bytes, total)
        ranges.append((i, lo, hi))
        lo = hi
        i += 1

    def _write_shard(i, lo, hi):
        fd = os.open(os.path.join(rank_dir, f"shard_{i:04d}.bin"),
                     os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            pos = lo
            while pos < hi:
                pos += os.write(fd, mv[pos:hi])
            os.fsync(fd)
        finally:
            os.close(fd)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(ranges)))) as ex:
        for fu in [ex.submit(_write_shard, *r) for r in ranges]:
            fu.result()

    manifest = {
        "total_bytes": total,
        "n_params": len(index),
        "shard_bytes": shard_bytes,
        "shards": [{"name": f"shard_{i:04d}.bin", "offset": lo, "nbytes": hi - lo}
                   for (i, lo, hi) in ranges],
        "layout": [[name, off, nbytes, str(dtype), list(shape)]
                   for name, (off, nbytes, dtype, shape) in index.items()],
    }
    with open(os.path.join(rank_dir, "weights_meta.json"), "w") as f:
        json.dump(manifest, f)
    return total


def _semip_load_weights(worker, weights_dir, io_workers=None):
    buf = worker._semip_buf
    index = worker._semip_index
    workers = int(io_workers or _WEIGHTS_IO_WORKERS)
    rank_dir = (weights_dir if _semip_tp_size(worker) <= 1
                else os.path.join(weights_dir, f"rank{worker.rank}"))
    meta_path = os.path.join(rank_dir, "weights_meta.json")
    with open(meta_path) as f:
        manifest = json.load(f)

    total = buf.numel()
    if int(manifest["total_bytes"]) != total:
        raise RuntimeError(
            f"weights size mismatch: manifest {manifest['total_bytes']}B but "
            f"attached buffer {total}B (config/model changed?)")
    cur_layout = [[name, off, nbytes]
                  for name, (off, nbytes, dtype, shape) in index.items()]
    man_layout = [[row[0], row[1], row[2]] for row in manifest.get("layout", [])]
    if man_layout and man_layout != cur_layout:
        raise RuntimeError("weights layout mismatch between manifest and "
                           "attached model (param order/sizes differ)")

    mv = memoryview(buf.numpy())
    shards = manifest["shards"]

    def _read_shard(s):
        lo = int(s["offset"])
        n = int(s["nbytes"])
        dst = mv[lo:lo + n]
        got = 0
        with open(os.path.join(rank_dir, s["name"]), "rb", buffering=0) as f:
            while got < n:
                r = f.readinto(dst[got:])
                if r == 0:
                    break
                got += r
        if got != n:
            raise RuntimeError(f"{s['name']}: read {got} != {n}")

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(shards)))) as ex:
        for fu in [ex.submit(_read_shard, s) for s in shards]:
            fu.result()
    return total


# ---------------------------------------------------------------------------
# TP>1 NCCL teardown / reinit around CRIU (worker-local, via collective_rpc).
#
# CRIU cannot restore live NCCL communicators or CustomAllreduce IPC handles,
# so they are torn down before checkpoint and rebuilt after restore.  For
# graph reuse the teardown must be graph-preserving (unilateral ncclCommAbort,
# not the collective ncclCommDestroy that a live captured graph would deadlock);
# for MoE/EP the abort must also be concurrent (see _nccl_abort_comms_concurrent).
# ---------------------------------------------------------------------------
GRAPH_MODE_REUSE = "reuse"
GRAPH_MODE_FULL = "full"
_MISSING = object()
_LIBNCCL = None


def _semip_config_value(config, key, default=None):
    """Read ``key`` from a vllm config object or dict, falling back to its
    nested ``parallel_config``."""
    if config is None:
        return default
    if isinstance(config, dict):
        if key in config:
            return config[key]
        pc = config.get("parallel_config")
    else:
        if hasattr(config, key):
            return getattr(config, key)
        pc = getattr(config, "parallel_config", None)
    if pc is not None:
        if isinstance(pc, dict):
            if key in pc:
                return pc[key]
        elif hasattr(pc, key):
            return getattr(pc, key)
    return default


def _semip_tp_size(worker):
    return int(_semip_config_value(
        getattr(worker, "vllm_config", None), "tensor_parallel_size", 1) or 1)


def _semip_ep_enabled(config):
    return bool(_semip_config_value(config, "enable_expert_parallel", False))


def _is_arctic_parallel_worker(worker):
    # Ulysses / shift SP is out of scope for this port (dense TP + EP only).
    return False


def _clear_fd_backed_nccl_env():
    """Drop restored NCCL/OFI env that points at process-local fds (closed
    before CRIU save), so the next NCCL init regenerates them."""
    native_keys = (
        "NCCL_TOPO_FILE", "NCCL_TUNER_PLUGIN", "NCCL_NETDEVS_POLICY",
        "NCCL_NET_FORCE_FLUSH", "NCCL_NVLS_CHUNKSIZE",
        "NCCL_NVLSTREE_MAX_CHUNKSIZE", "NCCL_P2P_NET_CHUNKSIZE",
        "FI_EFA_FORK_SAFE",
    )
    for key in native_keys:
        os.environ.pop(key, None)
        os.unsetenv(key)
    for key, value in list(os.environ.items()):
        if key == "NCCL_TOPO_FILE" or (
                key.startswith("NCCL_") and value.startswith("/proc/self/fd/")):
            os.environ.pop(key, None)
            os.unsetenv(key)


def _nccl_abort_comm(comm):
    """ncclCommAbort(comm) via ctypes -- UNILATERAL and non-blocking, unlike the
    collective ncclCommDestroy which deadlocks when a live captured graph or an
    overlapping MoE/EP topology pins the comm."""
    global _LIBNCCL
    if _LIBNCCL is None:
        _LIBNCCL = ctypes.CDLL("libnccl.so.2")
        _LIBNCCL.ncclCommAbort.restype = ctypes.c_int
        _LIBNCCL.ncclCommAbort.argtypes = [ctypes.c_void_p]
    cp = comm if isinstance(comm, ctypes.c_void_p) else ctypes.c_void_p(int(comm))
    return _LIBNCCL.ncclCommAbort(cp)


def _nccl_abort_comms_concurrent(targets, timeout=60.0, after_fire=None):
    """Abort a set of pynccl comms CONCURRENTLY (one thread each).  Sequential
    abort deadlocks on DeepSeek-style overlapping world/tp/dp/ep comms: the
    shared per-rank proxy only drains once every comm's abort flag is set.  The
    ``after_fire`` hook sets the torch ProcessGroupNCCL abort flags while the
    pynccl aborts are in flight, so all flags are set together."""
    import threading
    import time as _t
    results = {}

    def _worker(nm, ptr):
        try:
            results[nm] = ("ok", _nccl_abort_comm(ptr))
        except BaseException as e:  # noqa: BLE001
            results[nm] = ("err", f"{type(e).__name__}: {e}")

    threads = []
    for nm, ptr in targets:
        th = threading.Thread(target=_worker, args=(nm, ptr), daemon=True)
        th.start()
        threads.append(th)
    if after_fire is not None:
        try:
            after_fire()
        except BaseException:  # noqa: BLE001
            pass
    deadline = _t.monotonic() + timeout
    for th in threads:
        th.join(max(0.0, deadline - _t.monotonic()))
    return results


def _abort_torch_process_groups():
    """Abort (non-blocking) every torch NCCL process group so the subsequent
    clean destroy does not deadlock on a comm a live graph still pins."""
    import torch
    from torch.distributed import distributed_c10d as c
    try:
        world = getattr(c, "_world", None)
        pg_map = dict(getattr(world, "pg_map", {}) or {})
        for pg in list(pg_map.keys()):
            try:
                be = pg._get_backend(torch.device("cuda"))
            except Exception:  # noqa: BLE001
                be = None
            fn = getattr(be, "abort", None) or getattr(pg, "abort", None)
            if fn is not None:
                fn()
    except Exception:  # noqa: BLE001
        pass


def _semip_destroy_fi_ar_workspace():
    """Free the FlashInfer fused allreduce+RMSNorm workspace before checkpoint
    (full mode only).  Defensive no-op when it was never created."""
    try:
        from vllm.distributed.device_communicators import (
            flashinfer_all_reduce as _fiar)
    except Exception:
        return
    try:
        _fiar.destroy_fi_ar_workspace()
    except Exception:  # noqa: BLE001
        pass


def _mark_rst_on_close(fd_int):
    """Set SO_LINGER(1,0) on an inet TCP socket so its eventual close sends RST
    (skips TIME_WAIT) -> avoids restore-time EADDRINUSE on the rebind. AF_UNIX
    and non-stream sockets are left untouched. Operates on a *dup* so we never
    close the worker's live fd here -- the option lives on the shared socket, so
    the original fd RSTs when it is finally closed by teardown/CRIU-kill."""
    import socket, struct
    try:
        dup = os.dup(fd_int)
    except OSError:
        return None
    try:
        s = socket.socket(fileno=dup)   # Linux auto-detects family/type/proto
    except OSError:
        os.close(dup)
        return None
    try:
        if (s.family in (socket.AF_INET, socket.AF_INET6)
                and s.type == socket.SOCK_STREAM):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                         struct.pack("ii", 1, 0))
            try:
                return (fd_int, s.family.name, s.getsockname())
            except OSError:
                return (fd_int, s.family.name, None)
    finally:
        s.close()   # closes the dup only; fd_int stays open for CRIU
    return None


def _mark_inet_sockets_rst(where):
    """Scan this worker's fds and SO_LINGER(1,0) every inet TCP socket, so the
    next close (distributed teardown / destructive CRIU dump) sends RST and the
    tuple skips TIME_WAIT -> restore rebinds the rendezvous port immediately with
    no cooldown. AF_UNIX / non-stream sockets are left alone. Returns marked
    (fd, family, addr) tuples."""
    pid = os.getpid()
    marked = []
    try:
        fd_names = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return []
    for fd_name in fd_names:
        try:
            fd_int = int(fd_name)
            if fd_int <= 2:
                continue
            link = os.readlink(f"/proc/{pid}/fd/{fd_name}")
        except (OSError, ValueError):
            continue
        if not link.startswith("socket:"):
            continue
        r = _mark_rst_on_close(fd_int)
        if r is not None:
            marked.append(r)
    if marked:
        print(f"[rst-on-dump] {where}: SO_LINGER(1,0) on {len(marked)} "
              f"inet TCP socket(s): {marked}", flush=True)
    return marked


def _destroy_nccl(worker, graph_mode=None):
    """Tear down NCCL process groups before checkpoint.  No-op at TP1."""
    import torch.distributed as dist

    graph_mode = graph_mode or GRAPH_MODE_REUSE
    if _semip_tp_size(worker) <= 1:
        return None

    preserve_graph = graph_mode == GRAPH_MODE_REUSE
    # Abort (unilateral) rather than collective-destroy when a live graph pins
    # the comm (reuse) or the MoE/EP topology would deadlock a collective destroy.
    abort_nccl = preserve_graph or _semip_ep_enabled(
        getattr(worker, "vllm_config", None))

    def _close_custom_allreduce_ipc_handles(ca):
        if ca is None or getattr(ca, "disabled", True):
            return
        rank = ca.rank
        for pointers in (getattr(ca, "meta_ptrs", []),
                         getattr(ca, "buffer_ptrs", [])):
            for i, ptr in enumerate(pointers):
                if i == rank or not ptr:
                    continue
                ret = _cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))
                if ret != 0:
                    raise RuntimeError(
                        f"cudaIpcCloseMemHandle failed with cudaError={ret}")

    if not dist.is_initialized():
        return {}
    from vllm.distributed import parallel_state as ps
    from vllm.distributed.parallel_state import (
        destroy_model_parallel, destroy_distributed_environment)

    # RST-on-close: mark the rendezvous inet TCP sockets now, while they are
    # still open, so the teardown below closes them with RST (skips TIME_WAIT).
    # Otherwise the destructive dump leaves the tuple in TIME_WAIT and the
    # restore rebind fails with EADDRINUSE (sk-inet.c: Address already in use).
    # Runs per-worker (this is a collective_rpc target), so it hits each rank's
    # rendezvous socket.
    _mark_inet_sockets_rst("destroy_nccl")

    # Free the FlashInfer AR+RMS multicast workspace before NCCL teardown (full
    # only; reuse preserves graphs and thus the workspace they reference).
    if not preserve_graph:
        _semip_destroy_fi_ar_workspace()

    seen_pynccl_ids = set()
    seen_ca_ids = set()
    abort_targets = []
    abort_pynccls = []
    # Every group, not just tp/world: MoE adds ep (and may add dp/dcp/pcp/eplb);
    # leaving any undestroyed leaks NCCL/NVLS handles across the checkpoint.
    for name, ref in list(getattr(ps, "_groups", {}).items()):
        group = ref() if callable(ref) else ref
        if group is None:
            continue
        comm = getattr(group, "device_communicator", None)
        if comm is None:
            continue
        pynccl = getattr(comm, "pynccl_comm", None)
        if pynccl is not None and getattr(pynccl, "comm", None) is not None:
            if id(pynccl) not in seen_pynccl_ids:
                seen_pynccl_ids.add(id(pynccl))
                if abort_nccl:
                    cp = pynccl.comm
                    cp = (int(cp.value) if isinstance(cp, ctypes.c_void_p)
                          else int(cp))
                    abort_targets.append((name, cp))
                    abort_pynccls.append(pynccl)
                else:
                    pynccl.nccl.ncclCommDestroy(pynccl.comm)
                    pynccl.comm = None
        ca = getattr(comm, "ca_comm", None)
        if ca is not None and id(ca) not in seen_ca_ids:
            seen_ca_ids.add(id(ca))
            _close_custom_allreduce_ipc_handles(ca)
            ca.close()
            comm.ca_comm = None

    _torch_abort_state = {"done": False}

    def _do_torch_pg_abort():
        _abort_torch_process_groups()
        _torch_abort_state["done"] = True

    if abort_targets:
        _nccl_abort_comms_concurrent(
            abort_targets,
            after_fire=(_do_torch_pg_abort if not preserve_graph else None))
        for pynccl in abort_pynccls:
            pynccl.comm = None

    if abort_nccl and not _torch_abort_state["done"]:
        _do_torch_pg_abort()

    try:
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:  # noqa: BLE001
        # Aborted comms make the clean destroy raise; parallel_state is reset
        # enough for reinit to rebuild.  Only re-raise on the clean path.
        if not abort_nccl:
            raise
    _clear_fd_backed_nccl_env()
    return {"graph_mode": graph_mode}


def _force_dist_uninitialized_for_restore():
    """Force torch.distributed + parallel_state to a clean uninitialized state so
    the next init rebuilds a FRESH process group / TCP store / NCCL comms.  After
    an abort-based teardown the CRIU image can carry is_initialized()==True with
    dead PGs, which makes post-restore init skip init_process_group and deadlock."""
    from vllm.distributed import parallel_state as ps
    try:
        import torch.distributed as dist  # noqa: F401
        from torch.distributed import distributed_c10d as c10d
        try:
            c10d._update_default_pg(None)
        except BaseException:  # noqa: BLE001
            c10d.GroupMember.WORLD = None
        w = getattr(c10d, "_world", None)
        if w is not None:
            for _attr in ("comms", "pg_map", "pg_names", "pg_group_ranks",
                          "pg_backend_config", "pg_to_tag", "tags_to_pg",
                          "pg_coalesce_state"):
                try:
                    getattr(w, _attr).clear()
                except BaseException:  # noqa: BLE001
                    pass
            try:
                w.group_count = 0
            except BaseException:  # noqa: BLE001
                pass
        try:
            c10d._unregister_all_process_groups()
        except BaseException:  # noqa: BLE001
            pass
    except BaseException:  # noqa: BLE001
        pass
    for _g in ("_WORLD", "_INNER_DP_WORLD", "_NODE_COUNT", "_TP", "_PP", "_DP",
               "_DCP", "_PCP", "_EP", "_EPLB", "_SP", "_SP_TP"):
        try:
            if hasattr(ps, _g):
                setattr(ps, _g, None)
        except BaseException:  # noqa: BLE001
            pass


def _reinit_nccl(worker, port):
    """Re-initialize NCCL after restore on a fresh TCP port, then rebind the
    canonical tp:0/world:0 (+ ep:0/dp:0 for MoE) group slots the captured graphs
    look up."""
    import traceback
    import weakref
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state as ps
    from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
    try:
        _clear_fd_backed_nccl_env()
        _force_dist_uninitialized_for_restore()
        # NVLS / SymmMem exchange fds that do not survive CRIU -> keep them off.
        os.environ["NCCL_NVLS_ENABLE"] = "0"
        os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
        _rd_keep = getattr(worker, "_semip_rank_data_keep", None)
        with set_current_vllm_config(worker.vllm_config):
            # Reuse the preserved cold-start rank_data tensor (same VA) so the
            # kept-graph CA _dp pointers stay valid.
            _rd_patched = bool(
                _CA_REBIND_AVAILABLE and ca_graph_rebind is not None
                and _rd_keep is not None
                and ca_graph_rebind.install_rank_data_reuse_patch(_rd_keep))
            try:
                init_worker_distributed_environment(
                    worker.vllm_config,
                    worker.rank,
                    distributed_init_method=f"tcp://127.0.0.1:{port}",
                    local_rank=worker.local_rank,
                    backend="nccl",
                )
            finally:
                if _rd_patched:
                    ca_graph_rebind.restore_rank_data_reuse_patch()
        new_tp = ps.get_tp_group()
        new_world = ps.get_world_group()
        if hasattr(ps, "_groups"):
            ps._groups["tp:0"] = weakref.ref(new_tp)
            ps._groups["world:0"] = weakref.ref(new_world)
            if _semip_ep_enabled(getattr(worker, "vllm_config", None)):
                for _grp_name, _getter in (("ep:0", "get_ep_group"),
                                           ("dp:0", "get_dp_group")):
                    try:
                        _grp = getattr(ps, _getter)()
                    except Exception:  # noqa: BLE001
                        _grp = None
                    if _grp is not None:
                        ps._groups[_grp_name] = weakref.ref(_grp)
        return {"ok": True, "rank": getattr(worker, "rank", "?")}
    except BaseException as e:  # noqa: BLE001
        return {"ok": False, "rank": getattr(worker, "rank", "?"),
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}


def _prepare_worker_dump(worker):
    """Clean up per-worker FDs / io_uring / IB verbs mappings for CRIU dump.
    Invoked from the parent's prepare_criu_dump handler via collective_rpc so
    each TP worker subprocess sheds state CRIU cannot serialize."""
    pid = os.getpid()
    closed_fds = []
    unmapped = []
    devnull = os.open(os.devnull, os.O_RDWR)
    for std_fd in (1, 2):
        os.dup2(devnull, std_fd)
    os.close(devnull)
    close_anon = ("infinibandevent", "io_uring")
    close_shm = ("/dev/shm/psm_",)
    keep_prefixes = ("/dev/nvidia", "/dev/shm", "anon_inode:", "socket:", "pipe:")
    for fd_name in sorted(os.listdir(f"/proc/{pid}/fd"), key=int):
        try:
            fd_int = int(fd_name)
            if fd_int <= 2:
                continue
            link = os.readlink(f"/proc/{pid}/fd/{fd_name}")
            if link.startswith("anon_inode:"):
                if any(bad in link for bad in close_anon):
                    os.close(fd_int)
                    closed_fds.append(fd_int)
                continue
            if any(link.startswith(p) for p in close_shm):
                os.close(fd_int)
                closed_fds.append(fd_int)
                continue
            if any(link.startswith(p) for p in keep_prefixes):
                continue
            os.close(fd_int)
            closed_fds.append(fd_int)
        except (OSError, ValueError):
            pass
    libc = ctypes.CDLL("libc.so.6")
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            if (("io_uring" in line) or ("/dev/infiniband/" in line)
                    or ("uverbs" in line)):
                start_s, end_s = line.split()[0].split("-")
                start = int(start_s, 16)
                length = int(end_s, 16) - start
                libc.munmap(ctypes.c_void_p(start), ctypes.c_size_t(length))
                unmapped.append(f"0x{start:x}")

    # PSM (IB/EFA) shm: do NOT munmap -- the mapping may still be referenced by
    # a live provider thread even after NCCL teardown, so munmap SIGSEGVs the
    # worker.  Instead unlink the /dev/shm/psm_* file while keeping the mapping;
    # the inode stays alive so the worker is unaffected, and CRIU then captures
    # the mapping as anonymous memory (no file to reopen on restore).
    import glob as _glob
    removed_psm = []
    for _pf in _glob.glob("/dev/shm/psm_*") + _glob.glob("/dev/shm/psm2_*"):
        try:
            os.remove(_pf)
            removed_psm.append(_pf)
        except OSError:
            pass
    return {"closed_fds": closed_fds, "unmapped": unmapped,
            "removed_psm": removed_psm}


# ---------------------------------------------------------------------------
# CUDA graph handling around CRIU (worker-local, via collective_rpc).
#
# Default is graph REUSE: cold-start graphs are preserved in the CRIU image and
# their stale CustomAllreduce addresses are rewritten after reinit by
# ca_graph_rebind (fast).  graph_mode="full" is an explicit fallback that drops
# the graphs (cleargraph) and rebuilds them with capture_model() (slow).
# ---------------------------------------------------------------------------
def _semip_prepare_graph_reuse_snapshot(worker):
    """Cold-start hook: record the CA meta/buffer/rank_data snapshot the
    post-reinit rebind rewrites against.  No-op at TP1 or without ca_graph_rebind.
    The cold capture was already forced onto the CA copy path + keep_graph by
    SemipGPUWorker.compile_or_warm_up_model, so this only records state."""
    if _semip_tp_size(worker) <= 1:
        return {"enabled": False, "reason": "tp1"}
    if not _CA_REBIND_AVAILABLE or ca_graph_rebind is None:
        return {"available": False}
    try:
        result = ca_graph_rebind.store_snapshot(worker, None, None)
        result["graph_mode"] = GRAPH_MODE_REUSE
        return result
    except Exception as e:  # noqa: BLE001
        return {"available": True, "ok": False, "error": f"{type(e).__name__}: {e}"}


def _semip_cleargraph(worker, graph_mode=None):
    """Drop captured cudagraphs so the next capture_model() runs fresh.  No-op in
    reuse mode (graphs are preserved).  In full mode: destroy exec handles, call
    each wrapper's clear_all_graphs(), refresh the shared graph pool (else
    recapture mints private per-size pools and OOMs), reset the MoE aux stream,
    and reset the V2 ModelCudaGraphManager."""
    import gc
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm.platforms import current_platform

    graph_mode = graph_mode or GRAPH_MODE_REUSE
    if graph_mode == GRAPH_MODE_REUSE:
        return {"graph_mode": graph_mode, "skipped": "reuse_preserves_graph"}

    set_cudagraph_capturing_enabled(False)
    try:
        mr = getattr(worker, "model_runner", None)
        mgr = getattr(mr, "cudagraph_manager", None)
        wrapper_classes = {}

        def _consider(cls):
            if cls is not None and hasattr(cls, "_all_instances") \
                    and hasattr(cls, "clear_all_graphs"):
                wrapper_classes[id(cls)] = cls

        for _mod, _name in (
                ("vllm.compilation.cuda_graph", "CUDAGraphWrapper"),
                ("vllm.compilation.breakable_cudagraph", "BreakableCUDAGraphWrapper")):
            try:
                _m = __import__(_mod, fromlist=[_name])
                _consider(getattr(_m, _name, None))
            except Exception:  # noqa: BLE001
                pass
        _mdl = getattr(mr, "model", None)
        _consider(type(_mdl) if _mdl is not None else None)
        _inner = getattr(_mdl, "cudagraph_wrapper", None) if _mdl is not None else None
        _consider(type(_inner) if _inner is not None else None)
        _bcr = getattr(mgr, "breakable_cg_runner", None)
        _consider(type(_bcr) if _bcr is not None else None)
        if len(wrapper_classes) < 2:
            for o in gc.get_objects():
                t = type(o)
                try:
                    if hasattr(t, "_all_instances") and hasattr(t, "clear_all_graphs"):
                        wrapper_classes[id(t)] = t
                except Exception:  # noqa: BLE001
                    pass

        def _entries_of(inst):
            for attr in ("concrete_cudagraph_entries", "entries", "cudagraphs"):
                d = getattr(inst, attr, None)
                if isinstance(d, dict):
                    return d
            return {}

        def _reset_graph(obj):
            if obj is not None and hasattr(obj, "reset"):
                try:
                    obj.reset()
                except Exception:  # noqa: BLE001
                    pass

        for cls in wrapper_classes.values():
            for i in list(getattr(cls, "_all_instances", []) or []):
                for e in list(_entries_of(i).values()):
                    _reset_graph(getattr(e, "cudagraph", None))
                    cap = getattr(e, "capture", None)
                    for seg in list(getattr(cap, "segments", []) or []):
                        _reset_graph(getattr(seg, "__self__", None))

        for cls in wrapper_classes.values():
            try:
                cls.clear_all_graphs()
            except Exception:  # noqa: BLE001
                for i in list(getattr(cls, "_all_instances", []) or []):
                    try:
                        i.clear_graphs()
                    except Exception:  # noqa: BLE001
                        try:
                            _entries_of(i).clear()
                        except Exception:  # noqa: BLE001
                            pass

        for cand in (_mdl, getattr(_mdl, "cudagraph_wrapper", None)
                     if _mdl is not None else None):
            if (cand is not None and not hasattr(type(cand), "_all_instances")
                    and hasattr(cand, "cudagraphs") and hasattr(cand, "clear_graphs")):
                try:
                    cand.clear_graphs()
                except Exception:  # noqa: BLE001
                    pass

        fresh_pool = None
        try:
            type(current_platform)._global_graph_pool = None
            fresh_pool = current_platform.get_global_graph_pool()
        except Exception:  # noqa: BLE001
            pass
        if fresh_pool is not None:
            for cls in wrapper_classes.values():
                for i in list(getattr(cls, "_all_instances", []) or []):
                    if hasattr(i, "graph_pool"):
                        try:
                            i.graph_pool = fresh_pool
                        except Exception:  # noqa: BLE001
                            pass

        try:
            import vllm.utils.torch_utils as _vtu
            if getattr(_vtu, "_aux_stream", None) is not None:
                _vtu._aux_stream = None
        except Exception:  # noqa: BLE001
            pass

        if mgr is not None:
            try:
                for _g in list(getattr(mgr, "graphs", {}).values()):
                    _reset_graph(_g)
                if hasattr(mgr, "graphs"):
                    mgr.graphs.clear()
                if hasattr(mgr, "_graphs_captured"):
                    mgr._graphs_captured = False
                if getattr(mgr, "pool", None) is not None and fresh_pool is not None:
                    mgr.pool = fresh_pool
            except Exception:  # noqa: BLE001
                pass

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    finally:
        set_cudagraph_capturing_enabled(True)
    return {"graph_mode": graph_mode}


def _semip_reuse_graphs(worker):
    """Repair preserved CUDA graphs against the post-restore runtime by rewriting
    the moved CustomAllreduce addresses (ca_graph_rebind).  No capture_model()."""
    if _semip_tp_size(worker) <= 1:
        torch.cuda.synchronize()
        return {"ok": True, "recaptured": False, "graph_mode": GRAPH_MODE_REUSE,
                "skipped": "tp1_graph_reuse"}
    if not _CA_REBIND_AVAILABLE or ca_graph_rebind is None:
        return {"ok": False, "graph_mode": GRAPH_MODE_REUSE,
                "error": "ca_graph_rebind unavailable"}
    ca_rebind = ca_graph_rebind.rebind_after_reinit(worker)
    torch.cuda.synchronize()
    return {"ok": bool(ca_rebind.get("ok")), "recaptured": False,
            "graph_mode": GRAPH_MODE_REUSE, "ca_rebind": ca_rebind}


def _semip_connect_ep_channels_before_capture(worker):
    """Connect every MoE/EP NCCL channel on the default stream before a full
    capture_model().  Without this the ep all_gather/reduce_scatter lazily
    connects on the graph-capture stream and recapture hangs.  EP-only."""
    if not _semip_ep_enabled(getattr(worker, "vllm_config", None)):
        return {"skipped": "not_ep"}
    mr = getattr(worker, "model_runner", None)
    dummy = getattr(mr, "_dummy_run", None)
    if mr is None or dummy is None:
        return {"skipped": "no_dummy_run"}
    from vllm.config import CUDAGraphMode
    none_mode = CUDAGraphMode.NONE
    try:
        cc = (getattr(mr, "compilation_config", None)
              or getattr(getattr(worker, "vllm_config", None),
                         "compilation_config", None))
        sizes = getattr(cc, "cudagraph_capture_sizes", None) or [32]
        max_sz = int(max(sizes))
    except Exception:  # noqa: BLE001
        max_sz = 32
    diag = {}
    for name, ntok, uniform in (("mixed", max_sz, False), ("decode", 1, True)):
        try:
            dummy(ntok, cudagraph_runtime_mode=none_mode, uniform_decode=uniform,
                  skip_eplb=True, remove_lora=False)
            torch.cuda.synchronize()
            diag[name] = "ok"
        except Exception as e:  # noqa: BLE001
            diag[name] = f"err: {type(e).__name__}: {e}"
    return diag


def _semip_recapture_graphs(worker, graph_mode=None):
    """Repair (reuse) or rebuild (full) CUDA graphs against the live post-restore
    state."""
    graph_mode = graph_mode or GRAPH_MODE_REUSE
    if graph_mode == GRAPH_MODE_REUSE:
        return _semip_reuse_graphs(worker)
    mr = worker.model_runner
    cleargraph = _semip_cleargraph(worker, GRAPH_MODE_FULL)
    _semip_connect_ep_channels_before_capture(worker)
    torch.cuda.synchronize()
    mr.capture_model()
    torch.cuda.synchronize()
    return {"ok": True, "recaptured": True, "graph_mode": graph_mode,
            "cleargraph": cleargraph, "rank": getattr(worker, "rank", "?")}


def vllm_child_loop(pipe_conn, instance_id, gpus, model_dir=None):
    """Runs in a spawned child process: owns CUDA and vLLM.

    ``gpus`` is the physical GPU list for this instance (a single-element
    list at TP=1).  The main loop has two modes:
    - **Idle**: blocks on pipe_conn.recv() (zero CPU).
    - **Active** (engine has unfinished requests): alternates between
      engine.step() and non-blocking pipe_conn.poll() so new generate
      requests can be submitted mid-decode.
    """
    if isinstance(gpus, int):
        gpus = [gpus]
    gpus = list(gpus)
    rank = gpus[0]
    if len(gpus) > 1:
        # TP>1: keep ALL GPUs visible so tensor parallelism can span the
        # group; each vLLM worker is placed on its physical GPU by
        # SemipGPUWorker.init_device via SEMIP_GPU_MAP.  Any inherited
        # CUDA_VISIBLE_DEVICES mask must be cleared first, else the physical
        # indices in SEMIP_GPU_MAP disagree with the visible-device namespace
        # (and it would confuse the cuda-checkpoint physical-GPU addressing).
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["SEMIP_GPU_MAP"] = ",".join(str(g) for g in gpus)
    else:
        # TP1: unchanged single-GPU behavior -- pin to the one GPU so it is
        # visible as device 0.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["USE_LIBUV"] = "0"

    # Bind vLLM's internal rendezvous (the torch.distributed TCPStore) to
    # loopback instead of the node's routable IP.  get_ip() otherwise picks
    # the routable interface address, which CRIU bakes into the image as the
    # listening socket's bound address; restoring on a different node then
    # fails at bind() with EADDRNOTAVAIL.  127.0.0.1 exists identically on
    # every node, so loopback makes images node-portable.
    os.environ["VLLM_HOST_IP"] = "127.0.0.1"
    # VLLM_HOST_IP only steers vLLM's own rendezvous.  The collective
    # libraries pick their transport interface independently and default to
    # the routable NIC: gloo keeps a persistent listening socket for the life
    # of the process and NCCL opens a bootstrap listener, both of which get
    # baked into the image bound to the capture node's IP.  Pin them to
    # loopback so every internal socket binds to 127.0.0.1.  An instance is
    # single-node even at TP>1 -- the TP group's ranks are local, so their
    # NCCL bootstrap reaches across loopback fine and the data path rides
    # NVLink/P2P rather than these sockets.
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"

    # JIT/compile caches (Triton, vLLM torch.compile, torch inductor,
    # FlashInfer) all produce .so's that get dlopen()'d into the process.
    # CRIU records those mappings by absolute path and requires the files
    # to exist at restore; their defaults live in node-local dirs
    # ($HOME/.triton, ~/.cache/vllm, /tmp/torchinductor_*), so on another
    # node they are absent and restore fails with "Can't open file ...".
    #
    # When the Instance supplies a model_dir, the cache lives at
    # <model_dir>/compilation -- embedded next to the CRIU image so the
    # compile cache is isolated per model and travels with the image as one
    # unit.  This makes restore-on-another-node require model_dir to exist
    # at the same absolute path on that node (copy the whole model_dir over
    # first).  Absent a model_dir, the caches keep their defaults, which is
    # correct for same-node restore.
    #
    # Must be set before vLLM (and FlashInfer, whose env module resolves
    # these at import) is imported.
    if model_dir:
        _compile_root = os.path.join(model_dir, "compilation")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(_compile_root, "triton")
        os.environ["VLLM_CACHE_ROOT"] = os.path.join(_compile_root, "vllm")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
            _compile_root, "inductor")
        os.environ["FLASHINFER_WORKSPACE_BASE"] = os.path.join(
            _compile_root, "flashinfer")
        for _cache_dir in (os.environ["TRITON_CACHE_DIR"],
                           os.environ["VLLM_CACHE_ROOT"],
                           os.environ["TORCHINDUCTOR_CACHE_DIR"],
                           os.environ["FLASHINFER_WORKSPACE_BASE"]):
            os.makedirs(_cache_dir, exist_ok=True)

    semip_logging.init_process()
    log = semip_logging.child(instance_id, rank)
    # First-run path: route this process's stdout/stderr to the shared
    # per-instance log file from the very first byte.  CRIU later dumps
    # fd 1/2 as regular-file references to this path.
    #
    # Across runs the path baked into the image can be stale: if this
    # model was instance N when it was dumped, the restored child will
    # re-open /tmp/instN.log even when the orchestrator has now placed
    # it at a different instance_id (so its output silently leaks into
    # another instance's log).  To fix that, the worker sends a
    # ``rebind_log`` command immediately after CRIU restore (handled in
    # ``_handle_command`` below), which dup2s fd 1/2 onto the file
    # matching the *current* instance_id and rebuilds the log adapter.
    _child_log_path = semip_logging.redirect_stdio_to_instance_file(
        instance_id)

    # Detach from the controlling terminal before anything is captured.
    # fd 1/2 are already the per-instance log file (above); fd 0 is still
    # the interactive shell's pts, inherited down the spawn chain.  A pts
    # on fd 0 makes CRIU dump the process as a --shell-job tied to an
    # external terminal/session, which cannot be reattached when the tree
    # is restored inside a private PID namespace (criu tty.c: TIOCSPGRP
    # fails because the namespaced pgid can't own the host terminal).
    # Point fd 0 at /dev/null and start a fresh session so the captured
    # tree owns its own session and holds no controlling terminal.
    try:
        _devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(_devnull, 0)
        os.close(_devnull)
    except OSError:
        pass
    try:
        os.setsid()
    except OSError:
        # Already a session/group leader (rare for a spawned child); the
        # fd-0 redirect above is the part that matters for CRIU.
        pass

    # TP1 pins its single GPU via CUDA_VISIBLE_DEVICES, so it is device 0
    # here.  At TP>1 all GPUs stay visible, so address the group's first
    # physical GPU directly.
    torch.cuda.set_device(0 if len(gpus) == 1 else gpus[0])

    llm = None
    engine = None
    # The staging buffer, param index and chunk plan live on each vLLM worker
    # (``worker._semip_*``), not here -- see the worker-local primitives above.
    # A buffer held in this process could not be written by the workers at
    # TP>1, where collective_rpc cloudpickles the callable into subprocesses.

    _active_reqs = {}     # req_id -> {"t0", "engine_ids", "finished"}
    _engine_to_req = {}   # engine_request_id -> req_id
    _next_engine_id = 0
    _deferred_cmds = []   # non-generate commands received during drain

    # Pause/resume state.  `_paused` gates the engine.step() call in
    # the main loop, and is also the single switch that routes
    # generate-while-paused submits: `_submit_generate` parks them
    # in `_saved_requests` (skipping the engine entirely) iff
    # `_paused` is True.  `_saved_requests` is populated by both
    # `pause` (which snapshots active requests and aborts them in
    # the engine) and post-pause `_submit_generate` (which
    # synthesises a never-stepped record), then drained on `resume`
    # via `engine.add_request` for each entry.  All of this lives
    # in plain Python state so CRIU dumps and restores it for free
    # across cuda_checkpoint/cuda_restore cycles, and keeping the
    # engine untouched while paused makes the path robust to any
    # pipe interleaving of pause / sleep / cuda_checkpoint /
    # generate that the orchestrator's "Walking down past `up`
    # while paused" rule permits.
    #
    # `_dormant` is a separate, *defensive* flag that brackets the
    # span where the vLLM engine is unsafe to mutate because
    # `llm.sleep(level=2)` discarded its KV cache and possibly
    # `cuda-checkpoint` froze the entire CUDA context.  Set True at
    # the bottom of the `sleep` handler, False at the bottom of the
    # `wake_up_kv_cache` handler.  `_submit_generate` checks it
    # BEFORE `_paused` and, when True, sends back a `generate_done`
    # ack carrying a `RuntimeError("generate against dormant
    # engine")` instead of touching the engine -- so any race that
    # slips past the orchestrator's Phase-2 eviction sentinel
    # (`Orchestrator._evict_for_phase2`) surfaces as a loud, fail-
    # fast future exception in `_on_generate_done` instead of a
    # silent hang inside `engine.step()` on a torn-down executor.
    # Defense in depth: with the sentinel intact this branch is
    # unreachable in normal operation; the historical record of the
    # `_engine_dormant` / `_paused` unification (commit `ad74086`)
    # is in `orchestrator_DESIGN.md` "Eviction-mid-generate
    # dormant-engine wedge".
    _paused = False
    _dormant = False
    _saved_requests = []

    def _alloc_engine_id():
        nonlocal _next_engine_id
        eid = f"req-{_next_engine_id}"
        _next_engine_id += 1
        return eid

    def _submit_generate(req_id, prompts, sampling_params_dict):
        if _dormant and not _paused:
            # Defense-in-depth fail-fast: the orchestrator should
            # never enqueue a generate cmd onto an engine that has
            # been put to sleep without a corresponding pause (the
            # Phase-2 eviction sentinel in
            # ``Orchestrator._evict_for_phase2`` gates this).  If
            # that gate ever has a hole, abort with a loud error
            # ack instead of silently hanging inside
            # ``engine.step()`` on a torn-down executor.  Routes
            # through the demuxer's standard error path
            # (``error is not None`` on the result tuple), which
            # latches the error, decrements ``_pending_count``
            # cleanly, and surfaces to ``Orchestrator.
            # _on_generate_done`` as the ``error`` arg so the
            # in-flight ``done_event.set()`` happens with
            # ``q_rec["state"]="error"``.
            err = RuntimeError(
                f"generate req_id={req_id} arrived against dormant "
                f"engine (sleep without prior pause); orchestrator "
                f"sentinel breach -- see orchestrator_DESIGN.md "
                f"'Eviction-mid-generate dormant-engine wedge'")
            log.error(
                "  _dormant fail-fast: rejecting req_id=%s "
                "(prompts=%s)  -- %s",
                req_id, _truncate_for_display(list(prompts)), err)
            pipe_conn.send((
                "generate_done", 0.0, err, {"req_id": req_id}))
            return
        if _paused:
            # Single rule: while paused, the vLLM engine sees no
            # scheduler mutations from this child.  Park the request
            # in `_saved_requests` and let the next `resume` reload
            # it; `pause` already did the same for whatever was
            # in-flight at pause-time, so on resume the deferred
            # entries and the pause-snapshotted entries flow back
            # into the engine through one code path.
            #
            # This keeps the engine untouched for the entire dormant
            # span -- `llm.sleep` discards cumem-allocated KV blocks
            # and `cuda-checkpoint` (during `cuda_checkpoint`)
            # freezes the CUDA context, so any `engine.add_request`
            # / `engine.abort_request` call inside that window would
            # either enqueue into a scheduler that can never `step`
            # or block on a torn-down executor.  It is also
            # order-independent w.r.t. pipe interleavings of
            # generate/sleep/checkpoint/etc. while paused.
            #
            # `prompt_token_ids: []` (not None) matches the shape
            # `_snapshot_active_into_saved` produces for an empty
            # per-eid state via its `list(... or [])` clause, so
            # the resume branch's `len(prompt_tids)` works and its
            # `if prompt_tids:` test falls through to the
            # `elif i < len(prompts_orig)` re-prefill branch.
            _saved_requests.append({
                "req_id": req_id,
                "t0": time.perf_counter(),
                "first_token_ts": None,
                "prompts": list(prompts),
                "sampling_params": dict(sampling_params_dict),
                "eids": [{"prompt_token_ids": [],
                          "output_token_ids": [],
                          "output_text": ""}
                         for _ in prompts],
            })
            log.info("  submitted req_id=%s  prompts=%s  "
                     "(deferred to _saved_requests; paused)",
                     req_id, _truncate_for_display(list(prompts)))
            return

        from vllm import SamplingParams
        sp = SamplingParams(**sampling_params_dict)
        engine_ids = []
        for prompt in prompts:
            eid = _alloc_engine_id()
            engine.add_request(eid, prompt, sp)
            _engine_to_req[eid] = req_id
            engine_ids.append(eid)
        # `per_eid` tracks the latest cumulative engine output per
        # sub-request so that `pause` can snapshot the current state
        # without poking engine internals.  Updated in
        # `_process_step_outputs` on every step.
        per_eid = {eid: {"prompt_token_ids": None,
                         "output_token_ids": [],
                         "output_text": ""} for eid in engine_ids}
        _active_reqs[req_id] = {
            "t0": time.perf_counter(),
            "engine_ids": engine_ids,
            "finished": {},
            "prompts": list(prompts),
            "first_token_ts": None,
            "sampling_params": dict(sampling_params_dict),
            "per_eid": per_eid,
        }
        log.info("  submitted req_id=%s  prompts=%s",
                 req_id, _truncate_for_display(list(prompts)))

    def _process_step_outputs(step_outputs):
        for output in step_outputs:
            eid = output.request_id
            req_id = _engine_to_req.get(eid)
            if req_id is None:
                continue
            entry = _active_reqs.get(req_id)
            if entry is None:
                continue

            # First-token detection: stamp on the first step that
            # produced any decoded tokens for any sub-request of this
            # req_id.  output_kind defaults to CUMULATIVE so token_ids
            # is the running total -- non-empty iff at least one token
            # has been generated.
            if entry["first_token_ts"] is None and any(
                    o.token_ids for o in output.outputs):
                entry["first_token_ts"] = time.perf_counter()

            # Per-eid cumulative snapshot used by `pause`.  This must
            # happen on every step (not just the finishing one)
            # because pause can be invoked mid-decode.  We track only
            # the n=1 case (outputs[0]).
            per_eid_state = entry.get("per_eid", {}).get(eid)
            if per_eid_state is not None:
                if (per_eid_state["prompt_token_ids"] is None
                        and output.prompt_token_ids):
                    per_eid_state["prompt_token_ids"] = list(
                        output.prompt_token_ids)
                if output.outputs:
                    per_eid_state["output_token_ids"] = list(
                        output.outputs[0].token_ids)
                    per_eid_state["output_text"] = output.outputs[0].text

            if not output.finished:
                continue
            _engine_to_req.pop(eid, None)
            entry["finished"][eid] = output

            if len(entry["finished"]) == len(entry["engine_ids"]):
                ordered = [entry["finished"][e] for e in entry["engine_ids"]]

                # If this entry was resumed via `resume`, fold
                # pre-pause output back into the reported view
                # so the caller sees seamless continuation.
                pre_completion = entry.get("pre_pause_completion")
                pre_text = entry.get("pre_pause_text")
                orig_prompt_tokens = entry.get("original_prompt_tokens")

                if pre_completion is not None:
                    eid_index = {e: i for i, e in enumerate(entry["engine_ids"])}
                    outputs = [
                        [pre_text[eid_index[r.request_id]] + o.text
                         for o in r.outputs]
                        for r in ordered]
                    completion_tokens = sum(
                        len(o.token_ids)
                        + pre_completion[eid_index[r.request_id]]
                        for r in ordered for o in r.outputs)
                    prompt_tokens = sum(orig_prompt_tokens)
                else:
                    outputs = [[o.text for o in r.outputs] for r in ordered]
                    prompt_tokens = sum(
                        len(r.prompt_token_ids) for r in ordered)
                    completion_tokens = sum(
                        len(o.token_ids) for r in ordered for o in r.outputs)
                cached_tokens = sum(
                    (r.num_cached_tokens or 0) for r in ordered)
                finish_reasons = sorted({
                    o.finish_reason for r in ordered for o in r.outputs
                    if o.finish_reason is not None
                })
                info = {
                    "req_id": req_id,
                    "outputs": outputs,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "num_cached_tokens": cached_tokens,
                    "finish_reasons": finish_reasons,
                }

                t_done = time.perf_counter()
                elapsed = t_done - entry["t0"]
                first_token_ts = entry["first_token_ts"]
                ttft = (first_token_ts - entry["t0"]
                        if first_token_ts is not None else None)
                decode_time = (t_done - first_token_ts
                               if first_token_ts is not None else None)
                tpot_ms = (decode_time * 1000.0 / (completion_tokens - 1)
                           if (decode_time is not None
                               and completion_tokens > 1) else None)
                gen_tput = (completion_tokens / elapsed
                            if elapsed > 0 else 0.0)

                info["ttft_s"] = ttft
                info["decode_s"] = decode_time
                info["tpot_ms"] = tpot_ms
                info["gen_tput_tok_s"] = gen_tput

                prompts = entry.get("prompts")
                del _active_reqs[req_id]
                log.info(
                    "<<< generate req_id=%s OK (%.3fs)  "
                    "prompt_tokens=%s  completion_tokens=%s  "
                    "cached_tokens=%s  finish=%s  "
                    "prompt=%s  output=%s",
                    req_id, elapsed,
                    prompt_tokens, completion_tokens,
                    cached_tokens, finish_reasons,
                    _truncate_for_display(prompts),
                    _truncate_for_display(outputs),
                )
                log.info(
                    "    perf  ttft=%s  decode=%s  tpot=%s  "
                    "gen_tput=%.1f tok/s",
                    f"{ttft * 1000:.1f}ms" if ttft is not None else "n/a",
                    (f"{decode_time:.3f}s"
                     if decode_time is not None else "n/a"),
                    (f"{tpot_ms:.2f}ms"
                     if tpot_ms is not None else "n/a"),
                    gen_tput,
                )
                pipe_conn.send(("generate_done", elapsed, None, info))

    def _drain_engine():
        while engine is not None and engine.has_unfinished_requests():
            _process_step_outputs(engine.step())

    def _snapshot_active_into_saved() -> tuple[int, int]:
        """Snapshot every active sub-request into ``_saved_requests`` and
        abort it in the engine.  Mirrors the pause-time snapshot path,
        factored out so callers other than ``pause`` (e.g. a ``sleep``
        arriving on a paused engine that picked up a generate after
        pause) can preserve those requests for the next ``resume``
        instead of silently draining them to completion.

        Returns ``(saved_count, aborted_eid_count)``.  Safe to call when
        ``_active_reqs`` is empty (returns ``(0, 0)``).  Caller is
        responsible for any state flag updates (``_paused``) and for
        emitting the appropriate log line; this helper only touches the
        ledgers.
        """
        saved = []
        for req_id, entry in list(_active_reqs.items()):
            sp_dict = entry.get("sampling_params") or {}
            n_branch = sp_dict.get("n", 1)
            if n_branch != 1:
                raise RuntimeError(
                    f"snapshot with n={n_branch} not supported "
                    "(n=1 only)")
            eids_data = []
            for eid in entry["engine_ids"]:
                per_eid_state = entry["per_eid"].get(eid, {})
                eids_data.append({
                    "prompt_token_ids": list(
                        per_eid_state.get("prompt_token_ids") or []),
                    "output_token_ids": list(
                        per_eid_state.get("output_token_ids") or []),
                    "output_text":
                        per_eid_state.get("output_text", ""),
                })
            saved.append({
                "req_id": req_id,
                "t0": entry["t0"],
                "first_token_ts": entry["first_token_ts"],
                "prompts": list(entry.get("prompts") or []),
                "sampling_params": dict(sp_dict),
                "eids": eids_data,
            })

        all_eids = [eid
                    for entry in _active_reqs.values()
                    for eid in entry["engine_ids"]]
        if all_eids:
            try:
                engine.abort_request(all_eids)
            except Exception as _e:
                log.warning("  snapshot: abort_request failed: %s", _e)

        _saved_requests.extend(saved)
        _active_reqs.clear()
        _engine_to_req.clear()
        return len(saved), len(all_eids)

    def _handle_command(cmd, kwargs):
        nonlocal llm, engine
        nonlocal _paused, _dormant
        nonlocal log, _child_log_path

        error = None
        info = {}

        try:
            if cmd == "init":
                vllm_config = dict(kwargs["vllm_config"])
                vllm_config["enable_sleep_mode"] = True

                # TP>1 only: steer the collective path onto shapes that survive
                # a checkpoint.  Fused allreduce+RMS and NVLS/symmetric-memory
                # allreduce keep state CRIU cannot serialize, and the MoE
                # shared-experts side stream complicates graph capture.
                _tp = int(vllm_config.get("tensor_parallel_size", 1) or 1)
                if _tp >= 2:
                    _cc = vllm_config.get("compilation_config")
                    _cc = dict(_cc) if isinstance(_cc, dict) else {}
                    _pc = dict(_cc.get("pass_config") or {})
                    _pc["fuse_allreduce_rms"] = False
                    _cc["pass_config"] = _pc
                    vllm_config["compilation_config"] = _cc
                    os.environ["NCCL_NVLS_ENABLE"] = "0"
                    os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
                    os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"

                # Per-model env vars: vllm_config["_env"] is a reserved
                # mapping applied to os.environ before vLLM is imported,
                # so flags vLLM reads at import time take effect.  The
                # trio set at the top of vllm_child_loop is reserved
                # (CUDA isolation + in-process EngineCore + libuv off);
                # silently drop any attempt to override it from _env.
                _RESERVED_ENV = {
                    "CUDA_VISIBLE_DEVICES",
                    "VLLM_ENABLE_V1_MULTIPROCESSING",
                    "USE_LIBUV",
                    # Loopback pinning: repointing these at a routable NIC
                    # bakes the capture node's IP into the image and breaks
                    # restore elsewhere with EADDRNOTAVAIL.
                    "VLLM_HOST_IP",
                    "NCCL_SOCKET_IFNAME",
                    "GLOO_SOCKET_IFNAME",
                    "SEMIP_GPU_MAP",
                    # Compile-cache roots: CRIU bakes the resulting .so
                    # paths into the image, so a user override here would
                    # make the image unrestorable.
                    "TRITON_CACHE_DIR",
                    "VLLM_CACHE_ROOT",
                    "TORCHINDUCTOR_CACHE_DIR",
                    "FLASHINFER_WORKSPACE_BASE",
                }
                for k, v in (vllm_config.pop("_env", None) or {}).items():
                    if k in _RESERVED_ENV:
                        log.warning(
                            "ignoring reserved env key in _env: %s", k)
                        continue
                    os.environ[k] = str(v)

                # Force vLLM plugins (e.g. arctic_inference) to load
                # *before* `LLM(**vllm_config)` so plugin-installed
                # EngineArgs fields like `ulysses_sequence_parallel_size`
                # are present when EngineArgs is instantiated from
                # vllm_config.  vLLM normally loads plugins itself
                # during LLMEngine construction, but that fires too
                # late for plugins that extend the EngineArgs dataclass.
                # NB: `vllm.plugins` is a submodule, not auto-attached
                # to the `vllm` package on `import vllm` -- use the
                # `from vllm.plugins import ...` form.
                from vllm.plugins import load_general_plugins
                load_general_plugins()
                from vllm import LLM
                llm = LLM(**vllm_config)
                engine = llm.llm_engine
                info["pid"] = os.getpid()

                # Opt this worker out of arctic_inference's level-2
                # sleep/wake fast paths.  We restore main and drafter
                # params from a host-side pinned buffer ourselves
                # (stage / restore_weights), so:
                #   - skip the disk reload of the main model on wake_up
                #   - skip the per-sleep CPU snapshot of drafter
                #     ``named_parameters()`` (drafter ``named_buffers()``
                #     are still snapshotted; sub-MB)
                # Default arctic behavior is preserved for other users
                # because the flags are read via ``getattr(..., False)``.
                #
                # Note: ``GPUModelRunnerPatch.reload_weights`` is not
                # gated -- semi-persistence never calls
                # ``model_runner.reload_weights`` (the patched
                # ``Worker.wake_up`` reaches the unpatched original via
                # ``GPUModelRunnerPatch._orig_reload_weights``), so the
                # drafter-load augmentation in that path is never hit
                # from this child.
                def _enable_semi_persistence_flags(self):
                    # ``self`` here is the ``WorkerWrapperBase`` driver
                    # (see ``UniProcExecutor.collective_rpc`` ->
                    # ``run_method(self.driver_worker, ...)``).  Plain
                    # ``self.X = ...`` writes onto the wrapper; the
                    # wrapper only forwards ``__getattr__`` to
                    # ``self.worker``, so arctic's patched
                    # ``Worker.wake_up`` (where ``self`` is the real
                    # ``Worker``) would never see these flags and would
                    # fall back to the disk reload.  Write through to
                    # ``self.worker`` so the gating actually fires.
                    self.worker._skip_main_reload_on_wake = True
                    self.worker._skip_drafter_param_snapshot = True

                llm.collective_rpc(_enable_semi_persistence_flags)

                # Record the cold-start CustomAllreduce snapshot that the
                # post-reinit graph rebind rewrites against (TP>=2, reuse).
                if _tp >= 2:
                    llm.collective_rpc(_semip_prepare_graph_reuse_snapshot)

            elif cmd == "attach":
                if llm is None:
                    raise RuntimeError("attach requires init first")
                # Each worker allocates a plain (unpinned) CPU buffer sized to
                # its own shard and records the param layout on itself; pinning
                # is the explicit repin step.  Per-worker byte counts are
                # reported so the instance can size the restore chunk budget by
                # the largest worker shard (not the TP-aggregate sum).
                results = llm.collective_rpc(_semip_attach)
                worker_bytes = [r[0] for r in results]
                total_size = sum(worker_bytes)
                info["pinned_cpu_bytes"] = total_size
                info["pinned_bytes_per_worker"] = worker_bytes
                info["max_pinned_bytes_per_worker"] = max(worker_bytes, default=0)
                log.info("  attached %.2f GiB across %d worker(s)",
                         total_size / 2**30, len(results))

            elif cmd == "attach_pinned":
                raise RuntimeError(
                    "attach_pinned is not implemented for the worker-local "
                    "staging path; use attach -> repin instead")

            elif cmd == "detach":
                if llm is not None:
                    results = llm.collective_rpc(_semip_detach)
                    total = sum(results)
                    log.info("  freed %.2f GiB pinned memory", total / 2**30)

            elif cmd == "unpin":
                if llm is None:
                    raise RuntimeError("unpin requires attach first")
                results = llm.collective_rpc(_semip_unpin)
                log.info("  unpinned %.2f GiB", sum(results) / 2**30)

            elif cmd == "repin":
                if llm is None:
                    raise RuntimeError("repin requires attach first")
                results = llm.collective_rpc(_semip_repin)
                log.info("  repinned %.2f GiB", sum(results) / 2**30)

            elif cmd == "sleep":
                # Invariant: while `_paused` is True, `_active_reqs`
                # is empty -- `pause` snapshot-and-aborts whatever
                # was in flight at pause time and post-pause submits
                # go straight to `_saved_requests` via
                # `_submit_generate`, never touching the engine.
                # `_drain_engine` therefore has no scheduled work to
                # step through here.  If we are not paused, the
                # drain just runs the engine to completion as in a
                # normal cold sleep.
                _drain_engine()
                llm.sleep(level=2)
                torch.cuda.synchronize(0)
                torch.cuda.empty_cache()
                # Set _dormant AFTER llm.sleep so the fail-fast in
                # _submit_generate only fires once the engine is
                # actually torn down.  The flag is a defensive net
                # against the eviction-mid-generate wedge described
                # in orchestrator_DESIGN.md; the orchestrator's
                # Phase-2 sentinel is the primary gate.
                _dormant = True

            elif cmd == "stage":
                if llm is None:
                    raise RuntimeError("stage requires attach first")
                results = llm.collective_rpc(_semip_stage)
                total_bytes = sum(results)
                info["bytes"] = total_bytes
                log.info("  staged %.2f GiB across %d worker(s)",
                         total_bytes / 2**30, len(results))

            elif cmd == "wake_up_weights":
                llm.wake_up(tags=["weights"])

            elif cmd == "plan_restore_weights":
                if llm is None:
                    raise RuntimeError("plan_restore_weights requires init first")
                mb = kwargs.get("max_buffer_bytes")
                results = llm.collective_rpc(_semip_plan_load_weights, args=(mb,))
                worker_bytes = [r["bytes"] for r in results]
                chunk_bytes = [r["chunk_size"] for r in results]
                n_chunks = [r["n_chunks"] for r in results]
                info["bytes"] = sum(worker_bytes)
                info["pinned_bytes_per_worker"] = worker_bytes
                info["max_pinned_bytes_per_worker"] = max(worker_bytes, default=0)
                info["chunk_bytes_per_worker"] = chunk_bytes
                info["max_chunk_bytes_per_worker"] = max(chunk_bytes, default=0)
                info["n_chunks_per_worker"] = n_chunks
                info["n_chunks"] = max(n_chunks, default=0)
                info["chunk_size"] = max(chunk_bytes, default=0)
                log.info("  planned <= %d chunk(s) per worker (total %.2f GiB)",
                         info["n_chunks"], info["bytes"] / 2**30)

            elif cmd == "restore_weights":
                if llm is None:
                    raise RuntimeError("restore_weights requires init first")
                results = llm.collective_rpc(_semip_restore_weights)
                total_bytes = sum(r["bytes"] for r in results)
                total_loaded = sum(r["loaded"] for r in results)
                info["bytes"] = total_bytes
                info["n_chunks"] = max(r["n_chunks"] for r in results)
                log.info("  loaded %d params in <= %d chunk(s) (total %.2f GiB)",
                         total_loaded, info["n_chunks"], total_bytes / 2**30)

            elif cmd == "wake_up_kv_cache":
                llm.wake_up(tags=["kv_cache"])
                # Clear _dormant AFTER wake_up so the engine is
                # actually back up before _submit_generate stops
                # short-circuiting.  Pairs with the set in the
                # `sleep` handler.
                _dormant = False

            elif cmd == "pause":
                if engine is None:
                    raise RuntimeError("pause requires init first")

                was_paused = _paused
                _paused = True

                # Snapshot every active sub-request and abort it in
                # the engine so the upcoming `unpin` / `sleep` /
                # `cuda_checkpoint` runs against an empty scheduler.
                # Pending `generate_done` messages are deferred until
                # `resume` re-adds the requests via prefill.
                saved_count, aborted_count = _snapshot_active_into_saved()

                info["paused"] = True
                info["was_paused"] = was_paused
                info["saved"] = saved_count
                log.info("  pause: saved %d req_id(s) "
                         "(%d sub-requests aborted, was_paused=%s)",
                         saved_count, aborted_count, was_paused)

            elif cmd == "resume":
                if engine is None:
                    raise RuntimeError("resume requires init first")

                was_paused = _paused

                from vllm import SamplingParams
                from vllm.inputs import TokensPrompt

                restored = 0
                synthesized = 0
                for record in _saved_requests:
                    req_id = record["req_id"]
                    sp_dict = dict(record["sampling_params"] or {})
                    n_branch = sp_dict.get("n", 1)
                    if n_branch != 1:
                        raise RuntimeError(
                            f"resume with n={n_branch} not supported "
                            "(n=1 only)")
                    original_max = sp_dict.get("max_tokens")
                    eids_data = record["eids"]
                    prompts_orig = record["prompts"]

                    new_engine_ids = []
                    pre_pause_completion = []
                    pre_pause_text = []
                    original_prompt_tokens = []
                    all_finished_outputs = []

                    for i, eid_data in enumerate(eids_data):
                        prompt_tids = eid_data["prompt_token_ids"]
                        output_tids = eid_data["output_token_ids"]
                        output_text = eid_data["output_text"]
                        original_prompt_tokens.append(len(prompt_tids))
                        all_finished_outputs.append(output_text)

                        remaining = (original_max - len(output_tids)
                                     if original_max is not None else None)
                        if (remaining is not None and remaining <= 0):
                            # Already at max_tokens pre-pause; skip
                            # re-submission and synthesize the result.
                            continue

                        if prompt_tids:
                            full_token_ids = prompt_tids + list(output_tids)
                            prompt_obj = TokensPrompt(
                                prompt_token_ids=full_token_ids)
                        elif i < len(prompts_orig):
                            prompt_obj = prompts_orig[i]
                        else:
                            log.warning(
                                "  resume: req_id=%s eid#%d has no "
                                "prompt_token_ids and no original "
                                "prompt; skipping", req_id, i)
                            continue

                        sp_kwargs = dict(sp_dict)
                        if remaining is not None:
                            sp_kwargs["max_tokens"] = remaining
                        sp = SamplingParams(**sp_kwargs)

                        new_eid = _alloc_engine_id()
                        engine.add_request(new_eid, prompt_obj, sp)
                        _engine_to_req[new_eid] = req_id
                        new_engine_ids.append(new_eid)
                        pre_pause_completion.append(len(output_tids))
                        pre_pause_text.append(output_text)

                    if not new_engine_ids:
                        # Every branch was already finished pre-pause;
                        # emit a synthetic generate_done so the
                        # original waiter unblocks.
                        completion_tokens = sum(
                            len(d["output_token_ids"]) for d in eids_data)
                        prompt_tokens = sum(original_prompt_tokens)
                        synth_info = {
                            "req_id": req_id,
                            "outputs": [[t] for t in all_finished_outputs],
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "num_cached_tokens": 0,
                            "finish_reasons": ["length"],
                            "ttft_s": None,
                            "decode_s": None,
                            "tpot_ms": None,
                            "gen_tput_tok_s": 0.0,
                        }
                        pipe_conn.send(("generate_done", 0.0, None, synth_info))
                        synthesized += 1
                        log.info("  resume: req_id=%s synthesized "
                                 "(all branches at max_tokens pre-pause)",
                                 req_id)
                        continue

                    new_per_eid = {
                        new_eid: {"prompt_token_ids": None,
                                   "output_token_ids": [],
                                   "output_text": ""}
                        for new_eid in new_engine_ids
                    }
                    _active_reqs[req_id] = {
                        "t0": record["t0"],
                        "engine_ids": new_engine_ids,
                        "finished": {},
                        "prompts": list(prompts_orig),
                        "first_token_ts": record["first_token_ts"],
                        "sampling_params": dict(sp_dict),
                        "per_eid": new_per_eid,
                        "pre_pause_completion": pre_pause_completion,
                        "pre_pause_text": pre_pause_text,
                        "original_prompt_tokens": original_prompt_tokens,
                    }
                    restored += 1

                _saved_requests.clear()
                _paused = False

                info["paused"] = False
                info["was_paused"] = was_paused
                info["restored"] = restored
                info["synthesized"] = synthesized
                log.info("  resume: restored=%d synthesized=%d "
                         "(was_paused=%s)",
                         restored, synthesized, was_paused)

            elif cmd == "get_pipe_fd":
                info["pipe_fd"] = pipe_conn.fileno()

            elif cmd == "prepare_criu_dump":
                _drain_engine()

                # TP>1: each worker is a separate subprocess with its own FDs /
                # io_uring rings / IB verbs mappings that CRIU cannot serialize.
                # Clean them up before the tree is dumped.  Skipped at TP1, where
                # the "worker" is this same (driver) process -- running it here
                # would close this process's own FDs (incl. the worker pipe).
                if llm is not None and len(gpus) > 1:
                    try:
                        wr = llm.collective_rpc(_prepare_worker_dump)
                        info["worker_closed_fds"] = [r["closed_fds"] for r in wr]
                        info["worker_unmapped"] = [r["unmapped"] for r in wr]
                    except Exception as _e:
                        log.warning(
                            "  prepare_criu_dump: worker dump prep error: %s", _e)

                closed_fds = []
                unmapped = []
                destroyed_pg = False
                pid = os.getpid()

                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        dist.destroy_process_group()
                        destroyed_pg = True
                except Exception as _e:
                    log.warning("  prepare_criu_dump: dist teardown error: %s", _e)

                if destroyed_pg:
                    store_names = ("pt_tcpstore", "pt_nccl_watchdg",
                                   "pt_nccl_heartbt")
                    for _attempt in range(50):
                        alive = []
                        for tid_name in os.listdir(f"/proc/{pid}/task"):
                            try:
                                comm = open(
                                    f"/proc/{pid}/task/{tid_name}/comm"
                                ).read().strip()
                                if any(comm.startswith(s)
                                       for s in store_names):
                                    alive.append(f"{tid_name}({comm})")
                            except (OSError, ValueError):
                                pass
                        if not alive:
                            log.info("  prepare_criu_dump: store threads "
                                     "exited after %d polls", _attempt)
                            break
                        time.sleep(0.05)
                    else:
                        log.warning("  prepare_criu_dump: store threads "
                                    "still alive: %s", alive)

                pipe_fd = kwargs.get("pipe_fd", -1)
                # stdout/stderr were already pointed at the per-instance
                # log file by redirect_stdio_to_instance_file at process
                # startup, so CRIU dumps them as regular files pointing
                # to that path -- restoring re-opens the same path and
                # the restored child keeps logging there.

                keep_prefixes = ("/dev/nvidia", "/dev/shm", "anon_inode:",
                                 "socket:", "pipe:")
                for fd_name in sorted(os.listdir(f"/proc/{pid}/fd"),
                                      key=int):
                    try:
                        fd_int = int(fd_name)
                        if fd_int == pipe_fd or fd_int <= 2:
                            continue
                        link = os.readlink(f"/proc/{pid}/fd/{fd_name}")
                        if any(link.startswith(p) for p in keep_prefixes):
                            continue
                        os.close(fd_int)
                        closed_fds.append(fd_int)
                    except (OSError, ValueError):
                        pass

                libc = ctypes.CDLL("libc.so.6")
                with open(f"/proc/{pid}/maps") as f:
                    for line in f:
                        if "io_uring" in line:
                            addr_range = line.split()[0]
                            start_s, end_s = addr_range.split("-")
                            start = int(start_s, 16)
                            length = int(end_s, 16) - start
                            libc.munmap(ctypes.c_void_p(start),
                                        ctypes.c_size_t(length))
                            unmapped.append(f"0x{start:x}")

                import glob as _criu_glob
                for _sem in _criu_glob.glob("/dev/shm/sem.*"):
                    try:
                        os.remove(_sem)
                    except OSError:
                        pass

                remaining_threads = []
                for tid_name in os.listdir(f"/proc/{pid}/task"):
                    try:
                        comm = open(
                            f"/proc/{pid}/task/{tid_name}/comm"
                        ).read().strip()
                        if comm != "python":
                            remaining_threads.append(f"{tid_name}({comm})")
                    except (OSError, ValueError):
                        pass
                info["closed_fds"] = closed_fds
                info["unmapped"] = unmapped
                info["destroyed_pg"] = destroyed_pg
                info["remaining_threads"] = remaining_threads
                log.info("  prepare_criu_dump: fds=%s, unmapped=%s, "
                         "destroyed_pg=%s, remaining_threads=%s",
                         closed_fds, unmapped, destroyed_pg,
                         remaining_threads)

            elif cmd == "rebind_log":
                # Sent by the worker right after CRIU restore when the
                # current instance_id differs from the one baked into the
                # image: re-dup2 stdout/stderr onto /tmp/inst{new_id}.log
                # and rebuild the log adapter so subsequent records carry
                # the correct i{N} scope.
                new_id = kwargs["instance_id"]
                _child_log_path = semip_logging.redirect_stdio_to_instance_file(
                    new_id)
                log = semip_logging.child(new_id, rank)
                info["instance_id"] = new_id
                info["path"] = _child_log_path

            elif cmd == "destroy_nccl":
                if llm is None:
                    raise RuntimeError("destroy_nccl requires init first")
                graph_mode = kwargs.get("graph_mode")
                results = llm.collective_rpc(_destroy_nccl, args=(graph_mode,))
                info["graph_mode"] = graph_mode
                if all(r is None for r in results):
                    log.info("  destroy_nccl(%s): TP1 no-op", graph_mode)
                else:
                    log.info("  NCCL destroyed across %d workers (graph_mode=%s)",
                             len(results), graph_mode)

            elif cmd == "reinit_nccl":
                if llm is None:
                    raise RuntimeError("reinit_nccl requires init first")
                from vllm.utils.network_utils import get_open_port
                port = get_open_port()
                results = llm.collective_rpc(_reinit_nccl, args=(port,))
                failures = [r for r in results
                            if isinstance(r, dict) and not r.get("ok", True)]
                if failures:
                    raise RuntimeError(f"NCCL reinit failed on workers: {failures}")
                log.info("  NCCL re-initialized on port %d across %d workers",
                         port, len(results))

            elif cmd == "cleargraph":
                if llm is None:
                    raise RuntimeError("cleargraph requires init/load first")
                graph_mode = kwargs.get("graph_mode")
                results = llm.collective_rpc(_semip_cleargraph, args=(graph_mode,))
                info["graph_mode"] = graph_mode
                log.info("  cleargraph(%s) across %d worker(s)",
                         graph_mode, len(results))

            elif cmd == "recapture_graphs":
                if llm is None:
                    raise RuntimeError("recapture_graphs requires init/load first")
                graph_mode = kwargs.get("graph_mode") or GRAPH_MODE_REUSE
                results = llm.collective_rpc(
                    _semip_recapture_graphs, args=(graph_mode,))
                info["graph_mode"] = graph_mode
                failures = [r for r in results
                            if not (isinstance(r, dict) and r.get("ok", False))]
                if failures:
                    raise RuntimeError(
                        f"semip recapture_graphs({graph_mode}) failed: {failures}")
                log.info("  recapture_graphs(%s) across %d worker(s)",
                         graph_mode, len(results))
                # Reuse preserves graph topology but restore drops most
                # cudaGraphExec handles; re-instantiate them in lockstep across
                # ranks before live traffic by driving a few co-scheduled batches
                # at increasing concurrency.
                if graph_mode == GRAPH_MODE_REUSE:
                    if engine is None:
                        engine = llm.llm_engine
                    from vllm import SamplingParams
                    for nreq in (1, 2, 4, 8, 16):
                        toklen = max(1, min(64 // nreq, 20))
                        for _ in range(nreq):
                            engine.add_request(
                                _alloc_engine_id(),
                                {"prompt_token_ids": [0] * toklen},
                                SamplingParams(max_tokens=2, ignore_eos=True))
                        while engine.has_unfinished_requests():
                            engine.step()

            elif cmd == "save_weights":
                if llm is None:
                    raise RuntimeError("save_weights requires attach+stage first")
                results = llm.collective_rpc(
                    _semip_save_weights,
                    args=(kwargs["weights_dir"], kwargs.get("shard_bytes"),
                          kwargs.get("io_workers")))
                total = sum(results)
                info["weights_dir"] = kwargs["weights_dir"]
                info["bytes"] = total
                log.info("  saved weights: %.2f GiB across %d worker(s)",
                         total / 2**30, len(results))

            elif cmd == "load_weights":
                if llm is None:
                    raise RuntimeError("load_weights requires attach first")
                results = llm.collective_rpc(
                    _semip_load_weights,
                    args=(kwargs["weights_dir"], kwargs.get("io_workers")))
                total = sum(results)
                info["bytes"] = total
                log.info("  loaded weights: %.2f GiB across %d worker(s)",
                         total / 2**30, len(results))

            else:
                error = f"unknown command: {cmd}"

        except Exception as e:
            import traceback
            traceback.print_exc()
            error = f"{type(e).__name__}: {e}"

        return error, info

    # -- Main loop --------------------------------------------------------------

    # After a CRIU restore the original stdout/stderr fds are stale and the
    # first write raises OSError.  Redirect to a per-rank log file (instead
    # of /dev/null) so that any traceback/log from a restored child is
    # still captured for post-mortem debugging.
    _stdout_fixed = False

    # The current in-flight command, captured outside the per-iteration
    # scope so the fatal-error reporter below can blame the right cmd.
    cmd = None

    try:
        while True:
            if engine is None and llm is not None:
                engine = llm.llm_engine

            has_active = (engine is not None
                          and engine.has_unfinished_requests()
                          and not _paused)

            if has_active:
                _process_step_outputs(engine.step())
                if not pipe_conn.poll(0):
                    continue

            if _deferred_cmds:
                cmd, kwargs = _deferred_cmds.pop(0)
            else:
                try:
                    cmd, kwargs = pipe_conn.recv()
                except EOFError:
                    break

            if not _stdout_fixed:
                try:
                    sys.stdout.write("")
                    sys.stdout.flush()
                except OSError:
                    _logfp = open(_child_log_path, "a", buffering=1)
                    sys.stdout = _logfp
                    sys.stderr = _logfp
                    _stdout_fixed = True
                    semip_logging.rebind_stdout()
                    log.info("stdout/stderr redirected to %s after CRIU restore",
                             _child_log_path)

            if cmd == "exit":
                _drain_engine()
                log.info("exit")
                pipe_conn.send("exit_ack")
                break

            log.info(">>> %s", cmd)

            if cmd == "generate":
                req_id = kwargs.get("req_id")
                if req_id is None:
                    req_id = f"auto-{_next_engine_id}"
                try:
                    _submit_generate(req_id, kwargs["prompts"],
                                     kwargs["sampling_params"])
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    pipe_conn.send(("generate_done", 0.0,
                                    f"{type(e).__name__}: {e}",
                                    {"req_id": req_id}))
                # Drain any additional generate commands already on the pipe
                # so they get added to the engine before the first step().
                while pipe_conn.poll(0):
                    try:
                        cmd2, kwargs2 = pipe_conn.recv()
                    except EOFError:
                        break
                    if cmd2 == "generate":
                        rid2 = kwargs2.get("req_id",
                                           f"auto-{_next_engine_id}")
                        try:
                            _submit_generate(rid2, kwargs2["prompts"],
                                             kwargs2["sampling_params"])
                        except Exception as e2:
                            import traceback
                            traceback.print_exc()
                            pipe_conn.send(("generate_done", 0.0,
                                            f"{type(e2).__name__}: {e2}",
                                            {"req_id": rid2}))
                    else:
                        log.info(">>> %s (deferred)", cmd2)
                        _deferred_cmds.append((cmd2, kwargs2))
                continue

            t0 = time.perf_counter()
            error, info = _handle_command(cmd, kwargs)
            elapsed = time.perf_counter() - t0
            status = "OK" if error is None else "FAILED"
            log.info("<<< %s %s (%.3fs)", cmd, status, elapsed)
            pipe_conn.send((cmd, elapsed, error, info))

    except BaseException as _fatal:
        # Last-resort reporter: any unhandled exception in the main loop
        # (including KeyboardInterrupt, SystemExit) gets a final error
        # frame on the pipe so the worker can attribute the failure to a
        # specific cmd instead of just seeing "child pipe broken".  Both
        # the traceback and the offending cmd are logged to the per-rank
        # log file via log so the post-mortem survives a CRIU restore.
        import traceback as _tb
        _trace = _tb.format_exc()
        log.error("FATAL in main loop (cmd=%s): %s: %s",
                  cmd, type(_fatal).__name__, _fatal)
        log.error("%s", _trace)
        try:
            pipe_conn.send((
                cmd if cmd is not None else "__fatal__",
                0.0,
                f"FATAL {type(_fatal).__name__}: {_fatal}",
                {"traceback": _trace, "cmd": cmd},
            ))
        except Exception:
            pass
        raise
