"""Standalone Instance for a vLLM engine.

Each Instance is a GPU-agnostic handle.  The GPU is specified at
init(gpu) time, which also spawns the worker process.  All primitives
are non-blocking and return self for chaining.

Instances can be saved to disk via CRIU after CUDA checkpoint, then
restored (possibly on a different GPU):

    inst.unpin().sleep().cuda_checkpoint().criu_dump(filename="/data-fast/ckpt/m").wait()
    inst.cuda_restore(gpu=2).wake_up_weights().repin() \
        .plan_restore_weights().restore_weights().wake_up_kv_cache().wait()

On a later run, criu_restore() restores from the on-disk image:

    inst = Instance(vllm_config)
    inst.criu_restore("/data-fast/ckpt/m").plan_restore_weights().wait()
    inst.cuda_restore(gpu=0).wake_up_weights().repin() \
        .restore_weights().wake_up_kv_cache().wait()

Passing ``model_dir`` instead lets the image path be implicit; it then
defaults to ``<model_dir>/image`` and the filename argument can be
omitted:

    inst = Instance(vllm_config, "/data-fast/image-cache/qwen")
    inst.unpin().sleep().cuda_checkpoint().criu_dump().wait()
"""
from __future__ import annotations

import json
import os
import threading
import time
import weakref

import pynvml
import torch.multiprocessing as mp

import semip_logging
from demuxer import Demuxer
from worker import worker_loop

_spawn_ctx = mp.get_context("spawn")

_next_instance_id = 0
_id_lock = threading.Lock()


def _alloc_instance_id():
    global _next_instance_id
    with _id_lock:
        _next_instance_id += 1
        return _next_instance_id - 1


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


class Instance:

    _all: weakref.WeakValueDictionary[int, "Instance"] = weakref.WeakValueDictionary()

    def __init__(self, vllm_config: dict, model_dir: str | None = None):
        self.gpu = None
        self.vllm_config = vllm_config
        # Optional per-model directory.  When set, the image lives at
        # ``<model_dir>/image`` and ``criu_dump`` / ``criu_restore`` can be
        # called without a filename.  When unset, callers pass explicit
        # paths (the orchestrator does).
        self.model_dir = model_dir
        self.instance_id = _alloc_instance_id()
        Instance._all[self.instance_id] = self
        self.log = semip_logging.instance(self.instance_id, self.gpu)
        # Per-instance file gets a fresh start at construction time so
        # any later instance.N / worker.N / child.N records land in a
        # clean file.  Worker subprocesses do NOT re-truncate (they'd
        # erase parent-side records that arrived before they spawned).
        _log_path = semip_logging.truncate_instance_file(self.instance_id)
        semip_logging.attach_instance_file(self.instance_id)
        # Breadcrumb on the terminal (via the orch logger so it isn't
        # swallowed by the per-instance file route) so users know where
        # to tail.
        semip_logging.orch().info(
            "instance %d created  model=%s  log=%s",
            self.instance_id,
            vllm_config.get("model", "?"),
            _log_path,
        )

        self.pid = None
        self.state = "created"
        self.pinned_cpu_bytes = 0
        self.total_gpu_bytes = 0
        self._image_dir = None
        self._weights_dir = None
        # Multi-GPU (tensor-parallel) state.  TP is wired from the vLLM config
        # (``tensor_parallel_size``, default 1) at construction, not inferred
        # from the GPU count at ``init`` -- so ``n_gpus`` is authoritative here
        # and ``gpus`` (the physical GPU list) is only a placement argument
        # validated against it.  ``max_pinned_bytes_per_worker`` is the largest
        # per-worker staging shard (used to size the restore chunk budget at
        # TP>1, where the aggregate ``pinned_cpu_bytes`` overstates the per-GPU
        # budget).
        self.gpus = None
        self.n_gpus = int(vllm_config.get("tensor_parallel_size", 1) or 1)
        self.max_pinned_bytes_per_worker = 0

        self._cmd_queue = None
        self._result_queue = None
        self._completed_counter = None
        self._worker = None
        self._next_req_id = 0
        self.last_generate_result = None
        self.last_prompt_tokens = None
        self.last_completion_tokens = None
        self.generate_results = {}  # req_id -> {prompts, outputs, prompt_tokens, completion_tokens, ttft_s, tpot_ms}
        self._pending_prompts = {}  # req_id -> prompts (popped on completion)

        # The demuxer is the sole consumer of ``_result_queue``; it is
        # (re)created whenever queues are (re)created via _ensure_queues.
        # Listeners registered before the demuxer exists are buffered and
        # installed at ``_ensure_queues`` time so the orchestrator can
        # call ``add_cmd_listener`` regardless of init ordering.
        self._demuxer: Demuxer | None = None
        self._deferred_listeners: list[tuple[str | None, object]] = []

    def _ensure_queues(self):
        """Create mp queues/counter on demand, right before spawning a worker."""
        if self._cmd_queue is None:
            self._cmd_queue = _spawn_ctx.Queue()
            self._result_queue = _spawn_ctx.Queue()
            self._completed_counter = _spawn_ctx.Value('i', 0)
            self._demuxer = Demuxer(
                instance_id=self.instance_id,
                result_queue=self._result_queue,
                log=self.log,
                apply_result_cb=self._demuxer_apply_result,
                summarise_cb=self._summarise_for_log,
            )
            for cmd, cb in self._deferred_listeners:
                self._demuxer.add_listener(cmd, cb)
            self._demuxer.start()

    def __repr__(self):
        parts = [f"id={self.instance_id}", f"gpu={self.gpu}",
                 f"pid={self.pid}",
                 f"pinned_cpu={self.pinned_cpu_bytes / 2**30:.2f} GiB"]
        return f"Instance({', '.join(parts)})"

    # -- Internal helpers -------------------------------------------------------

    def _close_queues(self):
        """Stop the demuxer and deterministically close mp queues.

        Stopping the demuxer first lets the consumer thread exit before
        we close the queue under it.  ``Demuxer.stop`` is a no-op when
        invoked from the demuxer thread itself (e.g. when teardown's
        ``_apply_result`` calls into ``_reset``), so this is safe from
        every caller.
        """
        if self._demuxer is not None:
            self._demuxer.stop()
            self._demuxer = None
        for q in (self._cmd_queue, self._result_queue):
            if q is not None:
                try:
                    q.close()
                    q.join_thread()
                except Exception:
                    pass
        self._cmd_queue = None
        self._result_queue = None
        self._completed_counter = None

    def _reset(self):
        """Reset instance to created state after teardown completes."""
        if self._worker is not None:
            self._worker.join(timeout=10)
            if self._worker.is_alive():
                self.log.warning("worker still alive after join, force-killing")
                try:
                    import signal
                    os.kill(self._worker.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._worker.join(timeout=5)
            self._worker = None
        self._close_queues()
        self.state = "created"
        self.gpu = None
        self.gpus = None
        self.log.set_gpu(None)
        self.pid = None
        self.pinned_cpu_bytes = 0
        self.max_pinned_bytes_per_worker = 0

    def _send(self, cmd, **kwargs):
        self._cmd_queue.put((cmd, kwargs))
        self._demuxer.notify_send(cmd)
        return self

    def _resolve_image_dir(self, filename=None):
        """Pick the image directory: explicit arg, then model_dir, then last used."""
        if filename is not None:
            return filename
        if self.model_dir is not None:
            return os.path.join(self.model_dir, "image")
        return self._image_dir

    def _resolve_weights_dir(self, weights_dir=None):
        """Pick the weights directory: explicit arg, then model_dir, then
        a ``weights`` sibling of the image directory.

        The sibling fallback keeps ``save_weights`` usable for callers that
        pass explicit image paths (the orchestrator) rather than a model_dir.
        """
        if weights_dir is not None:
            return weights_dir
        if self.model_dir is not None:
            return os.path.join(self.model_dir, "weights")
        if self._image_dir is not None:
            return os.path.join(os.path.dirname(self._image_dir.rstrip("/")),
                                "weights")
        return None

    @property
    def _pending_count(self) -> int:
        """Number of cmds in flight; readers include the dashboard.

        Returns 0 when the demuxer hasn't been created yet (e.g. before
        ``init`` or after ``teardown``); callers use this for display
        only and don't expect transient counts to persist across
        worker lifecycle.
        """
        if self._demuxer is None:
            return 0
        return self._demuxer.pending_count

    @property
    def _pending_cmds(self) -> list[str]:
        """FIFO snapshot of currently-pending cmds.  Used by the dashboard."""
        if self._demuxer is None:
            return []
        return self._demuxer.pending_cmds

    def add_cmd_listener(self, cmd: str | None, callback) -> "Instance":
        """Register *callback* to fire when the demuxer processes *cmd*.

        ``cmd=None`` registers a catch-all that fires for every cmd.
        Listeners persist across the current worker lifecycle; if the
        demuxer hasn't been created yet (Instance fresh from
        ``__init__``), the registration is buffered and applied when
        ``_ensure_queues`` brings the demuxer up.
        """
        if self._demuxer is not None:
            self._demuxer.add_listener(cmd, callback)
        else:
            self._deferred_listeners.append((cmd, callback))
        return self

    def remove_cmd_listener(self, cmd: str | None, callback) -> "Instance":
        if self._demuxer is not None:
            self._demuxer.remove_listener(cmd, callback)
        else:
            try:
                self._deferred_listeners.remove((cmd, callback))
            except ValueError:
                pass
        return self

    # -- Primitives (non-blocking, return self) --------------------------------

    def init(self, gpus=None, gpu=None):
        """Cold start the engine on the given physical GPU(s).

        ``gpus`` is placement only and must have exactly
        ``tensor_parallel_size`` entries; TP size itself comes from the vLLM
        config.  A scalar ``gpu=`` (or positional ``init(0)``) still works
        for TP=1.
        """
        if gpus is None:
            gpus = [gpu] if gpu is not None else []
        elif isinstance(gpus, int):
            gpus = [gpus]
        gpus = list(gpus)
        if not gpus:
            raise RuntimeError("init requires a gpu/gpus")
        # TP is wired from the vLLM config, not inferred from the GPU count.
        tp = self.n_gpus
        if len(gpus) != tp:
            raise RuntimeError(
                f"init: {len(gpus)} gpu(s) given ({gpus}) but "
                f"tensor_parallel_size={tp}; the gpu list must have exactly "
                f"tensor_parallel_size entries")
        self.gpus = gpus
        gpu = gpus[0]
        self.gpu = gpu
        self.log.set_gpu(gpu)
        # Snapshot the GPU's physical capacity once per instance lifetime.
        # Safe under the orchestrator contract that init always takes a
        # full L1 slot (no shared tenants), so .total matches what vLLM
        # uses internally for its gpu_memory_utilization budget.  Every GPU
        # in a TP group has the same capacity, so gpus[0] is representative.
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        self.total_gpu_bytes = int(pynvml.nvmlDeviceGetMemoryInfo(h).total)
        self._log(f"init(gpus={gpus})")
        self._ensure_queues()
        # model_dir threads through to the vLLM child so it points its
        # compile cache at <model_dir>/compilation (embedding the JIT/compile
        # artifacts next to the image, whose recorded mmap paths must resolve
        # at restore).
        self._worker = _spawn_ctx.Process(
            target=worker_loop,
            args=(self.instance_id, list(gpus), self._cmd_queue,
                  self._result_queue, self._completed_counter, self.model_dir),
        )
        self._worker.start()
        # For TP>1: place the group on these physical GPUs via a custom worker
        # (all GPUs visible + SEMIP_GPU_MAP).  tensor_parallel_size already
        # comes from the user's vllm_config; we only inject the worker class.
        # TP1 keeps the vanilla single-GPU path untouched.
        vllm_config = dict(self.vllm_config)
        if tp > 1:
            vllm_config.setdefault("worker_cls", "_semip_worker.SemipGPUWorker")
        return self._send("init", vllm_config=vllm_config)

    def attach(self):
        self._log("attach")
        return self._send("attach")

    def attach_pinned(self):
        """Not supported on the worker-local staging path; the child raises.

        Use ``attach()`` followed by ``repin()`` instead.
        """
        self._log("attach_pinned")
        return self._send("attach_pinned")

    def detach(self):
        self._log("detach")
        return self._send("detach")

    def unpin(self):
        self._log("unpin")
        return self._send("unpin")

    def repin(self):
        self._log("repin")
        return self._send("repin")

    def sleep(self):
        self._log("sleep")
        return self._send("sleep")

    def pause(self):
        """Freeze the engine and snapshot in-flight requests.

        Sets the child's ``_paused`` flag and, in the same step,
        captures every active sub-request's
        ``(prompt_token_ids, output_token_ids_so_far,
        sampling_params, t0, first_token_ts)`` into a child-local
        list, then ``engine.abort_request(eids)`` so subsequent
        ``unpin()`` / ``sleep()`` / ``cuda_checkpoint()`` are safe.
        Pending ``generate_done`` messages are deferred until
        ``resume()`` re-adds the requests via prefill and drives them
        to completion.

        Idempotent (can be called when no requests are active).
        """
        self._log("pause")
        return self._send("pause")

    def resume(self):
        """Re-add saved requests via prefill and unfreeze the engine.

        Pairs with ``pause()``.  Each saved sub-request is re-added
        with ``prompt = original_prompt + output_so_far`` and
        ``max_tokens`` reduced by the number of pre-pause output
        tokens, so the original ``req_id`` continues seamlessly.
        Then clears ``_paused`` so the child's main loop resumes
        calling ``engine.step()``.

        Re-prefill is bit-exact for greedy (``temperature=0``) only;
        stochastic decode trajectories will diverge across the pause
        because per-request RNG state is not captured.
        """
        self._log("resume")
        return self._send("resume")

    def cuda_checkpoint(self):
        self._log("cuda_checkpoint")
        # TP>1: drop/preserve graphs then tear down NCCL (CRIU cannot restore
        # live communicators / CustomAllreduce IPC) before the CUDA checkpoint.
        # graph_mode="reuse" preserves the captured graphs; destroy_nccl uses
        # the graph-preserving unilateral-abort teardown.  Both are no-ops at
        # TP=1, but the gate keeps the single-GPU path free of extra commands.
        if self.n_gpus > 1:
            self._send("cleargraph", graph_mode="reuse")
            self._send("destroy_nccl", graph_mode="reuse")
        return self._send("cuda_checkpoint")

    def reinit_nccl(self):
        """Rebuild NCCL / the torch process group after a CRIU restore.

        Must run immediately after ``cuda_restore`` and before any
        collective (attach, weight restore, graph replay).  No-op at TP=1.
        """
        self._log("reinit_nccl")
        return self._send("reinit_nccl")

    def destroy_nccl(self, graph_mode="reuse"):
        """Tear down NCCL and CustomAllreduce IPC.  No-op at TP=1."""
        self._log(f"destroy_nccl(graph_mode={graph_mode})")
        return self._send("destroy_nccl", graph_mode=graph_mode)

    def cleargraph(self, graph_mode="reuse"):
        """Drop CUDA-graph exec handles.  ``reuse`` preserves them."""
        self._log(f"cleargraph(graph_mode={graph_mode})")
        return self._send("cleargraph", graph_mode=graph_mode)

    def recapture_graphs(self, graph_mode="reuse"):
        """Rebind (``reuse``) or recapture (``full``) the decode graphs.

        Run after ``wake_up_kv_cache``.  No-op at TP=1.
        """
        self._log(f"recapture_graphs(graph_mode={graph_mode})")
        return self._send("recapture_graphs", graph_mode=graph_mode)

    def criu_dump(self, filename: str | None = None):
        """CRIU-dump the child process tree to disk (destructive).

        Must be called after cuda_checkpoint() (GPU resources released).
        The child process is killed after a successful dump.  The
        on-disk image is later restored via criu_restore().

        If filename is None, uses ``<model_dir>/image``.
        """
        filename = self._resolve_image_dir(filename)
        if filename is None:
            raise RuntimeError(
                "criu_dump() requires a filename or a model_dir")
        self._log(f"criu_dump({filename})")
        self._image_dir = filename
        return self._send(
            "criu_dump", filename=filename,
            meta_extra={"vllm_config":      self.vllm_config,
                        "model_dir":        self.model_dir,
                        "total_gpu_bytes":  self.total_gpu_bytes,
                        "pinned_cpu_bytes": self.pinned_cpu_bytes,
                        "n_gpus":           self.n_gpus,
                        "max_pinned_bytes_per_worker":
                            self.max_pinned_bytes_per_worker})

    def criu_restore(self, filename: str | None = None):
        """Restore a live process from a CRIU image on disk.

        If filename is None, uses ``<model_dir>/image`` when a model_dir
        was given, else the image from the last criu_dump().
        Validates that the image's vllm_config matches this instance's
        config (raises RuntimeError on mismatch).  Spawns a new worker
        and CRIU-restores the child.  After criu_restore completes the
        instance is in 'checkpointed' state, ready for cuda_restore(gpu).
        """
        filename = self._resolve_image_dir(filename)
        if filename is None:
            raise RuntimeError(
                "criu_restore() requires a filename, a model_dir, or a "
                "prior criu_dump()")
        meta_path = os.path.join(filename, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            saved_config = meta.get("vllm_config")
            if saved_config is not None and saved_config != self.vllm_config:
                raise RuntimeError(
                    f"vllm_config mismatch: instance has {self.vllm_config} "
                    f"but image at {filename} was saved with {saved_config}")
            # CRIU records the compile-cache .so/cubin mmaps by absolute
            # path, so an image is bound to the model_dir it was dumped
            # with; restoring under a different one leaves those mappings
            # unresolvable ("Can't open file ...").
            saved_model_dir = meta.get("model_dir")
            if saved_model_dir and saved_model_dir != self.model_dir:
                raise RuntimeError(
                    f"model_dir mismatch: instance has {self.model_dir} but "
                    f"image at {filename} was dumped with {saved_model_dir}; "
                    f"the image bakes absolute compile-cache paths, so it "
                    f"must be restored under the same model_dir")
            # Hydrate budget inputs from meta.json; the child holds the
            # real pinned buffer that survived CRIU.  Old images without
            # ``total_gpu_bytes`` degrade to single-chunk behavior in
            # plan_restore_weights().  Fall back to the legacy
            # ``pinned_bytes`` key for one release.
            self.total_gpu_bytes = int(meta.get("total_gpu_bytes", 0))
            self.pinned_cpu_bytes = int(meta.get(
                "pinned_cpu_bytes", meta.get("pinned_bytes", 0)))
            # TP size is authoritative from this instance's vllm_config
            # (already equal to the image's, per the mismatch check above);
            # meta["n_gpus"] is only a fallback for images predating
            # config-wired TP.  Placement and the per-worker budget are
            # hydrated from the image, with a legacy rank -> [rank] shim.
            self.n_gpus = int(
                self.vllm_config.get("tensor_parallel_size",
                                     meta.get("n_gpus", 1)) or 1)
            self.max_pinned_bytes_per_worker = int(
                meta.get("max_pinned_bytes_per_worker", 0))
            _meta_gpus = meta.get("gpus") or (
                [meta["rank"]] if "rank" in meta else None)
            if _meta_gpus:
                self.gpus = list(_meta_gpus)
        self._log(f"criu_restore({filename})")
        self._image_dir = filename
        self._close_queues()
        self._ensure_queues()
        self._worker = _spawn_ctx.Process(
            target=worker_loop,
            args=(self.instance_id, list(self.gpus) if self.gpus else [0],
                  self._cmd_queue, self._result_queue,
                  self._completed_counter, self.model_dir),
        )
        self._worker.start()
        return self._send("criu_restore", filename=filename)

    def cuda_restore(self, gpu=None, gpus=None):
        """Restore the checkpointed CUDA state onto the given physical GPU(s).

        With neither argument, falls back to the placement hydrated from
        ``meta.json``, so a restore can re-place the group on a different
        physical GPU set.  The count is validated against
        ``tensor_parallel_size``; TP size cannot change across a restore.
        """
        if gpus is None:
            gpus = [gpu] if gpu is not None else list(self.gpus or [])
        elif isinstance(gpus, int):
            gpus = [gpus]
        gpus = list(gpus)
        if not gpus:
            raise RuntimeError(
                "cuda_restore requires a gpu/gpus (none given and no "
                "placement recorded in the image)")
        if len(gpus) != self.n_gpus:
            raise RuntimeError(
                f"cuda_restore: {len(gpus)} gpu(s) given ({gpus}) but "
                f"tensor_parallel_size={self.n_gpus}")
        self.gpus = gpus
        self._log(f"cuda_restore(gpus={gpus})")
        return self._send("cuda_restore", gpus=gpus)

    def stage(self):
        self._log("stage")
        return self._send("stage")

    def wake_up_weights(self):
        self._log("wake_up_weights")
        return self._send("wake_up_weights")

    def wake_up_kv_cache(self):
        self._log("wake_up_kv_cache")
        return self._send("wake_up_kv_cache")

    def plan_restore_weights(self, max_buffer_bytes=None):
        """Precompute the chunk plan that the next restore_weights() will consume.

        If ``max_buffer_bytes`` is given, it is passed through verbatim as
        the staging-buffer cap.  Use this to force small chunks when
        restoring an image whose checkpointed ``restore_weights`` does not
        release the staging buffer back to the CUDA driver before the KV
        cache is mapped (older images lack the ``torch.cuda.empty_cache()``
        fix): a small staging buffer stays negligible even if it lingers in
        torch's caching allocator, leaving room for ``wake_up_kv_cache``.

        Otherwise self-computes the staging budget from instance state
        populated by ``init`` (cold start) or by ``criu_restore`` reading
        meta.json (restore):

            allotment = self.total_gpu_bytes * gpu_memory_utilization
            budget    = min(self.pinned_cpu_bytes,
                            allotment - self.pinned_cpu_bytes)

        The formula is self-validating: if ``budget`` ends up smaller
        than the largest single parameter, the child's plan walk raises
        with a precise ``param X exceeds chunk_size`` message.

        If ``self.total_gpu_bytes`` or ``self.pinned_cpu_bytes`` is
        missing (legacy meta.json, or not yet attached), passes
        ``max_buffer_bytes=None`` to the child, yielding the single-chunk
        fallback (today's behavior).
        """
        if max_buffer_bytes is not None:
            mb = int(max_buffer_bytes)
        else:
            # At TP>1 the chunk budget is per-GPU, so use the largest
            # per-worker staging shard, not the TP-aggregate
            # ``pinned_cpu_bytes`` (which would overstate it ~N-fold and
            # shrink chunks needlessly, or spuriously fail the
            # "param exceeds chunk_size" check).
            pinned = self.max_pinned_bytes_per_worker or self.pinned_cpu_bytes
            if self.total_gpu_bytes <= 0 or pinned <= 0:
                mb = None
            else:
                util = self.vllm_config["gpu_memory_utilization"]
                allotment = int(self.total_gpu_bytes * util)
                mb = int(0.9 * min(pinned, allotment - pinned))
        self._log(f"plan_restore_weights(max_buffer_bytes={mb})")
        return self._send("plan_restore_weights", max_buffer_bytes=mb)

    def restore_weights(self):
        """Copy staged weights from pinned CPU into model parameters.

        Pure execution against the chunk plan cached by a prior
        ``plan_restore_weights()``.  For each chunk, the worker copies
        a slice of the pinned buffer to a single reused GPU staging
        buffer (PCIe H2D) and then scatters into
        ``model.named_parameters()`` in place.  If no plan is cached
        (paths that skip ``plan_restore_weights``), falls back to a
        single-chunk path identical to the prior unbounded behavior.

        Requires a prior ``attach() -> ... -> stage()`` to have populated
        the pinned buffer, and ``wake_up_weights()`` to have allocated
        the destination parameter tensors.
        """
        self._log("restore_weights")
        return self._send("restore_weights")

    def save_weights(self, shard_bytes=None, io_workers=None,
                     weights_dir=None):
        """Write the staged pinned buffer to <model_dir>/weights/ as shards.

        Call after ``stage()`` (buffer populated) and before ``detach()``,
        so ``criu_dump()`` runs against a detached (tiny) process image.
        ``shard_bytes`` / ``io_workers`` override the child defaults
        (``None`` leaves the child default in place).
        """
        weights_dir = self._resolve_weights_dir(weights_dir)
        if weights_dir is None:
            raise RuntimeError(
                "save_weights() requires a weights_dir, a model_dir, or a "
                "prior criu_dump()")
        self._weights_dir = weights_dir
        self._log(f"save_weights({weights_dir})")
        return self._send("save_weights", weights_dir=weights_dir,
                          shard_bytes=shard_bytes, io_workers=io_workers)

    def load_weights(self, io_workers=None, weights_dir=None):
        """Read <model_dir>/weights/ shards back into the pinned buffer.

        Requires a prior ``attach()`` on the restore side (rebuilds the
        index and allocates the buffer).  Feeds ``restore_weights()``.
        """
        weights_dir = self._resolve_weights_dir(weights_dir)
        if weights_dir is None:
            raise RuntimeError(
                "load_weights() requires a weights_dir, a model_dir, or a "
                "prior criu_restore()")
        self._weights_dir = weights_dir
        self._log(f"load_weights({weights_dir})")
        return self._send("load_weights", weights_dir=weights_dir,
                          io_workers=io_workers)

    def generate(self, prompts, sampling_params):
        self._log(f"generate({len(prompts)} prompts)")
        req_id = f"inst{self.instance_id}-{self._next_req_id}"
        self._next_req_id += 1
        self.last_req_id = req_id
        self._pending_prompts[req_id] = prompts
        return self._send("generate", req_id=req_id, prompts=prompts,
                           sampling_params=sampling_params)

    def teardown(self):
        self._log("teardown")
        return self._send("teardown")

    def remove(self):
        """Deregister this instance from the class-level registry.

        Non-blocking and non-destructive: does not touch the worker
        process or pending commands.  Returns the ``Instance`` class so
        a subsequent chained ``status()`` resolves to the classmethod
        view (the deregistered instance no longer appears there).
        """
        self._log("remove")
        Instance._all.pop(self.instance_id, None)
        return Instance

    # -- Synchronization -------------------------------------------------------

    def _apply_result(self, cmd: str, info: dict) -> None:
        """Update local state after a successful command completion."""
        if cmd == "init":
            self.pid = info.get("pid")
            self.state = "alive"
        elif cmd == "attach":
            self.pinned_cpu_bytes = info.get(
                "pinned_cpu_bytes", self.pinned_cpu_bytes)
            self.max_pinned_bytes_per_worker = info.get(
                "max_pinned_bytes_per_worker",
                self.max_pinned_bytes_per_worker)
        elif cmd == "plan_restore_weights":
            self.max_pinned_bytes_per_worker = info.get(
                "max_pinned_bytes_per_worker",
                self.max_pinned_bytes_per_worker)
        elif cmd == "detach":
            self.pinned_cpu_bytes = 0
            self.max_pinned_bytes_per_worker = 0
        elif cmd == "cuda_checkpoint":
            self.gpu = None
            self.log.set_gpu(None)
            self.state = "checkpointed"
        elif cmd == "criu_dump":
            self._image_dir = info.get("image_dir", self._image_dir)
            self.state = "checkpointed"
        elif cmd == "criu_restore":
            self.pid = info.get("pid")
            # Placement echo from the child; n_gpus stays config-authoritative.
            _g = info.get("gpus")
            if _g:
                self.gpus = list(_g)
            self.gpu = None
            self.log.set_gpu(None)
            self.state = "checkpointed"
        elif cmd == "cuda_restore":
            gpus = info.get("gpus")
            if gpus:
                self.gpus = list(gpus)
                self.gpu = gpus[0]
            else:
                self.gpu = info.get("gpu", self.gpu)
            self.log.set_gpu(self.gpu)
            self.state = "alive"
        elif cmd == "generate":
            self.last_generate_result = info.get("outputs")
            self.last_prompt_tokens = info.get("prompt_tokens")
            self.last_completion_tokens = info.get("completion_tokens")
            req_id = info.get("req_id")
            if req_id is not None:
                self.generate_results[req_id] = {
                    "prompts": info.get("prompts"),
                    "outputs": info.get("outputs"),
                    "prompt_tokens": info.get("prompt_tokens"),
                    "completion_tokens": info.get("completion_tokens"),
                    "ttft_s": info.get("ttft_s"),
                    "tpot_ms": info.get("tpot_ms"),
                }
        elif cmd == "teardown":
            self._reset()

    def _demuxer_apply_result(self, cmd: str, info: dict) -> None:
        """Demuxer apply-result callback.

        Pops the matching ``_pending_prompts`` entry for generate cmds
        (so ``_apply_result`` can read ``info["prompts"]``) and then
        delegates to ``_apply_result``.  Runs on the demuxer thread.
        """
        if cmd == "generate":
            rid = info.get("req_id")
            info["prompts"] = self._pending_prompts.pop(rid, None)
        self._apply_result(cmd, info)

    @staticmethod
    def _summarise_for_log(cmd: str, info: dict):
        """Demuxer log-summary callback.

        Compacts the noisy generate ``info`` dict into a token summary
        (prompt/output text is already logged by ``vllm_child.py`` at
        generate completion; we keep the instance log to a token
        summary so it isn't duplicated).
        """
        if cmd == "generate":
            return {
                "req_id": info.get("req_id"),
                "prompt_tokens": info.get("prompt_tokens"),
                "completion_tokens": info.get("completion_tokens"),
            }
        return info

    def wait(self):
        """Block until all pending commands complete for this instance.

        Re-raises ``RuntimeError`` (and clears the latch) on the first
        command that failed in this batch.  Implemented as a thin
        condvar wait on the demuxer's ``_pending_count`` -- safe under
        any number of concurrent ``wait()`` callers because the
        demuxer is the sole consumer of ``_result_queue``.
        """
        if self._demuxer is None:
            return self
        self._log(f"wait ({self._pending_count} pending)")
        self._demuxer.wait_idle()
        return self

    # -- Status ----------------------------------------------------------------

    def status(self=None):
        """Print all instances grouped by GPU, with CPU/GPU memory footprints.

        Non-blocking: per-instance state is kept fresh in real time by
        each instance's demuxer (which always drains
        ``_result_queue``), so no explicit sync step is needed before
        rendering.  GPU memory is read via NVML so no CUDA context is
        initialized in the caller.

        Returns ``self`` when called on an instance (so it can be chained
        with other primitives), and the ``Instance`` class when called as
        ``Instance.status()``.
        """
        from collections import defaultdict

        cls = Instance
        instances = list(cls._all.values())

        by_gpu: dict[int, list["Instance"]] = defaultdict(list)
        unassigned: list["Instance"] = []
        for inst in instances:
            if inst.gpu is None:
                unassigned.append(inst)
            else:
                by_gpu[inst.gpu].append(inst)

        gpu_mem: dict[int, tuple[int, int]] = {}
        pid_gpu_bytes: dict[int, int] = {}
        num_gpus = 0
        try:
            pynvml.nvmlInit()
            num_gpus = pynvml.nvmlDeviceGetCount()
            _NVML_NA = 0xFFFFFFFFFFFFFFFF
            for g in range(num_gpus):
                h = pynvml.nvmlDeviceGetHandleByIndex(g)
                m = pynvml.nvmlDeviceGetMemoryInfo(h)
                gpu_mem[g] = (int(m.used), int(m.total))
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                except Exception:
                    procs = []
                for p in procs:
                    used = getattr(p, "usedGpuMemory", None)
                    if used is None or used == _NVML_NA:
                        continue
                    pid_gpu_bytes[p.pid] = pid_gpu_bytes.get(p.pid, 0) + int(used)
        except Exception:
            pass

        bar = "=" * 80
        print(f"\n{bar}", flush=True)
        print(f"  Instance Status  [{time.strftime('%H:%M:%S')}]"
              f"  ({len(instances)} instance(s))", flush=True)
        print(bar, flush=True)

        for gpu in sorted({*by_gpu.keys(), *range(num_gpus)}):
            if gpu in gpu_mem:
                used, total = gpu_mem[gpu]
                free = total - used
                print(f"  GPU {gpu}:  {used / 2**30:.2f} / {total / 2**30:.2f} GiB used  "
                      f"({free / 2**30:.2f} GiB free)", flush=True)
            else:
                print(f"  GPU {gpu}:", flush=True)
            for inst in by_gpu.get(gpu, []):
                cls._print_instance(inst, pid_gpu_bytes)

        if unassigned:
            print(f"  Unassigned:", flush=True)
            for inst in unassigned:
                cls._print_instance(inst, pid_gpu_bytes)

        print(f"{bar}\n", flush=True)
        return self if self is not None else cls

    # Chainable alias used by the scripts; reads better mid-chain than
    # ``status`` and works both bound and unbound, as ``status`` does.
    print_status = status

    @staticmethod
    def _print_instance(inst, pid_gpu_bytes):
        model = inst.vllm_config.get("model", "?")
        if isinstance(model, str):
            model = model.split("/")[-1]
        pinned_gib = inst.pinned_cpu_bytes / 2**30
        gpu_gib = pid_gpu_bytes.get(inst.pid, 0) / 2**30 if inst.pid else 0.0
        pending = inst._pending_cmds or []
        marker = "*" if inst.state == "alive" else " "
        print(f"    [{marker}] inst{inst.instance_id:<3} "
              f"{inst.state:<14} {model:<40} "
              f"pinned_cpu={pinned_gib:5.2f} GiB  "
              f"gpu_mem={gpu_gib:5.2f} GiB  "
              f"pid={inst.pid}  "
              f"pending={pending}", flush=True)

    # -- Logging ---------------------------------------------------------------

    def _log(self, cmd):
        self.log.info("enqueue %s pending=%s", cmd, self._pending_cmds)
