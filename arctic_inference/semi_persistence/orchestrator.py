"""Lightweight orchestrator for managing multiple vLLM instances by model ID.

State machine with a single ordered ladder:

    saved  <-->  checkpoint  <-->  sleep  <-->  up  -->  running (transient)

move(model_id, target) walks up or down the ladder.
generate() auto-transitions to 'up', then up -> running -> up.
remove() auto-transitions to 'saved', then deletes.
"""
from __future__ import annotations

import collections
import json
import math
import os
import shutil
import subprocess
import threading
import time

import pynvml
from contextlib import contextmanager
from concurrent.futures import Future
from http.server import HTTPServer
from typing import Any

import semip_logging
from abstract import OrchestratorBase
from instance import Instance
from pipeline import (
    EvictForPeerOp,
    GenerateOp,
    Interrupted,
    ModelPipeline,
    MoveOp,
    PauseOp,
    PendingRequest,
    RegisterOp,
    RemoveOp,
    ResumeOp,
)
from slots import Slot, Slots
from state_server import init_t0, start_state_server

log = semip_logging.orch()


def _discover_gpu_ids() -> list[int]:
    """Return GPU device indices via NVML without initializing CUDA."""
    pynvml.nvmlInit()
    return list(range(pynvml.nvmlDeviceGetCount()))


_STATES = ["saved", "checkpoint", "sleep", "up"]


class _CmdAck(threading.Event):
    """Ack slot used by ``Orchestrator._send_cmd_with_ack``.

    A ``threading.Event`` (so callers can block on completion) with an
    extra ``error`` attribute that the demuxer's catch-all listener
    copies from the worker ack BEFORE setting the Event.  The waiter
    inspects ``error`` after ``wait()`` returns and re-raises if the
    child reported a failure -- the previous design dropped the error
    on the floor and the caller mistook a failed cmd (e.g. a CUDA OOM
    on ``wake_up_kv_cache``) for success, then marched the state
    machine forward into a wedged engine.
    """

    __slots__ = ("error",)

    def __init__(self) -> None:
        super().__init__()
        self.error: object | None = None


class WorkerCmdFailed(RuntimeError):
    """Raised by ``_send_cmd_with_ack`` when the child process reports
    a failure for a non-generate command (e.g. ``wake_up_kv_cache``
    OOM, ``cuda_restore`` failure, ``repin`` failure).

    Distinct from a generic ``RuntimeError`` so the pipeline can
    distinguish "engine state is now inconsistent, fail every
    in-flight generate for this model" from "guard check rejected
    this op, engine is fine".
    """

    def __init__(self, model_id: str, cmd: str, error: object) -> None:
        super().__init__(f"{model_id!r}: {cmd} failed in worker: {error}")
        self.model_id = model_id
        self.cmd = cmd
        self.worker_error = error


def _pick_level(gpu_memory_utilization: float) -> int:
    """Pick the smallest slot *level* (largest GPU fraction) that satisfies *util*.

    A level-L slot covers ``1 / 2**(L-1)`` of a GPU.  We pick the largest
    such fraction that still fits the model's ``gpu_memory_utilization``,
    which is the smallest L with ``1 / 2**(L-1) >= util``:

        util in (0.5,   1.0] -> L1   (whole GPU)
        util in (0.25,  0.5] -> L2   (half GPU)
        util in (0.125, 0.25]-> L3   (quarter GPU)
        util in (0.0625,0.125]->L4   (eighth GPU)
        ...

    Boundary points fall into the lower-L (larger-fraction) bucket, e.g.
    ``util == 0.5`` -> ``L1``.  Degenerate / non-positive ``util`` returns
    ``L1`` (whole GPU).
    """
    if gpu_memory_utilization <= 0.0 or gpu_memory_utilization >= 1.0:
        return 1
    return max(1, int(math.ceil(math.log2(1.0 / gpu_memory_utilization))))


class Orchestrator(OrchestratorBase):
    """Static registry that maps human-readable model IDs to Instances.

    States (ordered ladder):
        saved      -- image on disk, no process, no GPU
        checkpoint -- image on disk + live CRIU process in memory, no GPU
        sleep      -- CUDA context restored on a GPU; slot held until
                      `running -> up` releases it, or `sleep -> checkpoint`
                      tears the context down
        up         -- image on disk + live process + GPU held + weights on GPU
        running    -- transient sub-state of 'up' during generate();
                      always holds a slot, therefore unevictable
    """

    _registry: dict[str, Any] = {}
    _gpu_futures: dict[int, Future] = {}
    _gpu_ids: list[int] = []
    _image_cache: str | None = None
    _state_server: HTTPServer | None = None

    _request_log: list[dict] = []
    _request_counter: int = 0
    _request_lock: threading.Lock = threading.Lock()

    _generate_futures: list[Future] = []
    _inflight: dict[str, list] = {}  # model_id -> [(req_id, req_record, Event)]

    # FIFO of pending ack slots per (model_id, cmd).  Each
    # ``_send_cmd_with_ack`` call appends a fresh ``_CmdAck`` under
    # ``_cmd_ack_lock`` and then sends the cmd; the demuxer's
    # catch-all listener pops the head slot for that cmd (also under
    # the lock), copies the ack's ``error`` field onto it, and sets
    # the underlying Event.  ``_send_cmd_with_ack`` then re-raises if
    # the worker reported a failure -- previously the ``error`` field
    # was silently dropped here, so a CUDA OOM or restore failure on
    # the child looked like success to the orchestrator and the
    # caller marched the state forward into a wedged engine (see e.g.
    # the ``wake_up_kv_cache`` OOM -> infinite ``generate`` hang on
    # model 10 / 2026-05-17).
    #
    # The FIFO discipline tolerates concurrent ``_send_cmd_with_ack``
    # calls for the same cmd (e.g. a peer-eviction sleep racing with
    # a self-evacuation sleep on the same model) -- both slots get
    # installed and both get signalled in send order, where a
    # single-slot dict would have lost one ack and hung the loser.
    _cmd_ack_events: dict[tuple[str, str], "collections.deque[_CmdAck]"] = {}
    _cmd_ack_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Explicit per-model pipeline (see pipeline_DESIGN.md, pipeline.py).
    #
    # One ``ModelPipeline`` per registered model_id holds the FIFO queue
    # + worker thread that serialises every operation on that model.
    # Created lazily in ``register(...)`` and during ``init(...)``'s
    # image-cache scan; torn down in ``remove(...)`` and on hard-reset.
    # ------------------------------------------------------------------
    _pipelines: dict[str, "ModelPipeline"] = {}


    @staticmethod
    def _set_state(model_id: str, state: str) -> None:
        """Set a model's state and emit a one-line transition log.

        The log line includes the wall-clock time the model spent in
        ``prev_state`` (computed from ``state_since``), giving per-step
        timing without an extra summary line.
        """
        now = time.perf_counter()
        entry = Orchestrator._registry[model_id]
        prev_state = entry.get("state")
        prev_since = entry.get("state_since")
        entry["state"] = state
        entry["state_since"] = now
        # Mirror this transition into any in-flight request bound to
        # *model_id* so the dashboard can render a precise per-request
        # state timeline.  Deduped against the trailing entry.
        try:
            with Orchestrator._request_lock:
                for rec in Orchestrator._request_log:
                    if (rec.get("model_id") == model_id
                            and rec.get("t_done") is None):
                        rlog = rec.setdefault("state_log", [])
                        if not rlog or rlog[-1][1] != state:
                            rlog.append((now, state))
        except Exception:
            # State tracking is observational; never let it block a
            # real state transition.
            pass
        if prev_state != state:
            if prev_since is not None:
                log.info("%s: %s (%.1fs) -> %s",
                         model_id, prev_state, now - prev_since, state)
            else:
                log.info("%s: %s -> %s", model_id, prev_state, state)

    _timing = threading.local()

    @staticmethod
    @contextmanager
    def _locks_ordered(*model_ids: str):
        """Acquire per-model locks in sorted *model_id* order (deadlock-safe for pairs)."""
        ids = sorted(set(model_ids))
        if not ids:
            yield
            return
        if len(ids) == 1:
            with Orchestrator._registry[ids[0]]["_lock"]:
                yield
            return
        if len(ids) == 2:
            with Orchestrator._registry[ids[0]]["_lock"]:
                with Orchestrator._registry[ids[1]]["_lock"]:
                    yield
            return
        raise RuntimeError("at most two distinct model locks per _locks_ordered call")

    @staticmethod
    def _image_dir_for(model_id: str) -> str:
        """Return the image-cache directory for a given model_id."""
        safe_name = model_id.replace("/", "--")
        return os.path.join(Orchestrator._image_cache, safe_name)

    @staticmethod
    def _make_pipeline(model_id: str) -> "ModelPipeline":
        """Construct (and stash) a ``ModelPipeline`` for *model_id*.

        Used by ``register(...)`` (after the registry entry is in place)
        and by ``init(...)``'s image-cache scan (so discovered
        ``saved`` models also have a pipeline ready for the first user
        op).  Idempotent: returns the existing pipeline if one is
        already registered.
        """
        existing = Orchestrator._pipelines.get(model_id)
        if existing is not None:
            return existing
        pipe = ModelPipeline(
            model_id=model_id,
            entry=Orchestrator._registry[model_id],
            slots=Slots,
            orch=Orchestrator,
            pipelines=Orchestrator._pipelines,
        )
        Orchestrator._pipelines[model_id] = pipe
        return pipe

    @staticmethod
    def _shutdown_all_pipelines(*, timeout: float | None = 5.0) -> None:
        """Drain + join every ``ModelPipeline``.  Called on hard-reset.

        Without this, repeated ``init(...)`` (typical in tests) would
        accumulate zombie worker threads.
        """
        pipes = list(Orchestrator._pipelines.values())
        for pipe in pipes:
            try:
                pipe.shutdown(drain=True, timeout=timeout)
            except Exception:
                log.exception("error shutting down pipeline %s", pipe.model_id)
        Orchestrator._pipelines.clear()

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------

    @staticmethod
    def init(image_cache: str = "/data-fast/image-cache",
             gpus: list[int] | None = None,
             dashboard_port: int = 8157) -> None:
        """Discover GPUs, scan image cache, and populate registry.

        Each subdirectory of *image_cache* that contains a ``meta.json``
        is registered in **saved** state (image on disk, no live process).

        *dashboard_port* starts an HTTP server for the dashboard on that
        port.  Set to 0 to disable.
        """
        semip_logging.init_process()
        # Hard-reset path: drain + join any existing pipelines from a
        # previous init so this call doesn't leak worker threads.  Done
        # before clearing _registry so the workers see a stable entry
        # while shutting down.
        if Orchestrator._pipelines:
            Orchestrator._shutdown_all_pipelines()
        Orchestrator._gpu_ids = gpus if gpus is not None else _discover_gpu_ids()
        if Slots._inited:
            # Hard-reset: bypass leak assertions (tests may re-init).
            with Slots._cv:
                Slots._pools.clear()
                Slots._live.clear()
                Slots._last_used.clear()
                Slots._waiters.clear()
                Slots._draining.clear()
                Slots._inited = False
        Slots.init(Orchestrator._gpu_ids)
        Orchestrator._image_cache = image_cache
        os.makedirs(image_cache, exist_ok=True)

        import glob as _glob
        for _pattern in [
            "/tmp/torchinductor_root/**/*.ghost",
            "/tmp/__triton_launcher.*.ghost",
            "/dev/shm/*.ghost",
            "/dev/shm/link_remap.*",
        ]:
            for _f in _glob.glob(_pattern, recursive=True):
                try:
                    os.remove(_f)
                except OSError:
                    subprocess.run(["sudo", "rm", "-f", _f], capture_output=True)

        Orchestrator._registry = {}
        Orchestrator._gpu_futures = {}
        Orchestrator._generate_futures = []
        Orchestrator._inflight = {}
        Orchestrator._cmd_ack_events = {}
        with Orchestrator._request_lock:
            Orchestrator._request_log = []
            Orchestrator._request_counter = 0

        for entry_name in sorted(os.listdir(image_cache)):
            image_dir = os.path.join(image_cache, entry_name)
            meta_path = os.path.join(image_dir, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            vllm_config = meta.get("vllm_config", {})
            model_id = entry_name
            run_level = _pick_level(vllm_config["gpu_memory_utilization"])
            Orchestrator._registry[model_id] = {
                "state": "saved",
                "instance": None,
                "gpu": None,
                "slot": None,
                "level": run_level,
                "vllm_config": vllm_config,
                "image_dir": image_dir,
                "pinned_cpu_bytes": meta.get(
                    "pinned_cpu_bytes", meta.get("pinned_bytes", 0)),
                "total_gpu_bytes": meta.get("total_gpu_bytes", 0),
                "paused": False,
                "paused_since": None,
                "_lock": threading.RLock(),
                # Synchronises ``_send_cmd_with_ack`` (called from the
                # pipeline worker) with the demuxer's
                # ``_on_generate_done`` slot-release decision.  RLock
                # because ``_send_cmd_with_ack`` may be called nested
                # from inside an Op body that already holds it.
                "_gen_lock": threading.RLock(),
            }
            pinned = Orchestrator._registry[model_id]["pinned_cpu_bytes"]
            log.info("discovered %s  model=%s  pinned=%.1f GiB  image=%s",
                     model_id, vllm_config.get('model', '?'),
                     pinned / 2**30, image_dir)
            Orchestrator._make_pipeline(model_id)

        if Orchestrator._state_server is not None:
            try:
                Orchestrator._state_server.shutdown()
            except Exception:
                pass
            Orchestrator._state_server = None

        if dashboard_port:
            Orchestrator._state_server = start_state_server(dashboard_port)
            log.info("dashboard server on port %s", dashboard_port)

        n = len(Orchestrator._registry)
        log.info("init  image_cache=%s  gpus=%s  discovered %d saved models",
                 image_cache, Orchestrator._gpu_ids, n)

        # Anchor the dashboard's relative-time clock here so per-state
        # ages render even before the first generate (phases that only
        # register / move never call generate, but still need timers).
        init_t0()

    # ------------------------------------------------------------------
    # register
    # ------------------------------------------------------------------

    @staticmethod
    def register(model_id: str, vllm_config: dict) -> None:
        """Cold-start a new model, save its image, and register it.

        *vllm_config* must be a dict (e.g. ``{"model": "Qwen/Qwen3-32B"}``).
        Caller-side string shorthand normalisation lives in
        :class:`client.OrchestratorClient`; the in-process API is
        deliberately strict so the registry never holds two
        non-canonical variants of the same config.

        The dict is **stored as-is** -- no defaults are injected and the
        caller's object is not mutated.  This keeps client-side dedup
        (matching ``vllm_config`` against ``/state``) honest.  Required
        fields (currently: ``gpu_memory_utilization``) must be supplied
        by the caller; downstream consumers raise ``KeyError`` if a
        required field is missing.

        The dump is destructive — the child process is killed after the
        image is written.  The model ends up in **saved** state (image on
        disk, no live process).
        """
        if model_id in Orchestrator._registry:
            log.info("register  model_id=%s  already registered – skipping",
                     model_id)
            return
        log.info("register  model_id=%s  model=%s  (received)",
                 model_id, vllm_config.get('model', '?'))

        # Create the registry entry inline FIRST so the pipeline worker
        # can access ``_registry[mid]`` from the moment it starts
        # running RegisterOp.  RegisterOp reconciles the entry to its
        # working shape (live instance, gpu, state="init", etc.)
        # inside ``execute``.
        Orchestrator._registry[model_id] = {
            "state": "saved",
            "instance": None,
            "gpu": None,
            "slot": None,
            "level": _pick_level(vllm_config["gpu_memory_utilization"]),
            "vllm_config": vllm_config,
            "image_dir": Orchestrator._image_dir_for(model_id),
            "pinned_cpu_bytes": 0,
            "total_gpu_bytes": 0,
            "paused": False,
            "paused_since": None,
            "state_since": time.perf_counter(),
            "_lock": threading.RLock(),
            "_gen_lock": threading.RLock(),
        }
        pipe = Orchestrator._make_pipeline(model_id)
        pipe.submit(RegisterOp(model_id=model_id, vllm_config=vllm_config))

    # ------------------------------------------------------------------
    # registry introspection
    # ------------------------------------------------------------------

    @staticmethod
    def models() -> list[str]:
        """Return the list of currently registered model_ids.

        Useful for client-side fan-out loops::

            for mid in Orchestrator.models():
                Orchestrator.wait(mid)
        """
        return list(Orchestrator._registry)

    # ------------------------------------------------------------------
    # GPU pool management (add / sub)
    # ------------------------------------------------------------------

    @staticmethod
    def add(gpu: int) -> None:
        """Add *gpu* to the pool.  Synchronous, fast (pure bookkeeping).

        Raises ``ValueError`` if *gpu* is not visible to NVML.  Allowing
        a non-existent index into the pool is unrecoverable: the slot
        allocator will eventually hand it out, ``cuda_restore(gpu=N)``
        will crash inside ``_build_restore_args`` (out-of-range UUID
        lookup), and the dependent ``repin`` already pipelined to the
        vLLM child will deadlock the instance.  Fail loudly here
        instead.

        Idempotent: calling ``add`` for a GPU already in the pool logs a
        warning and returns without re-adding.  Wakes any FIFO waiters
        that may now be satisfiable on the new GPU.
        """
        try:
            pynvml.nvmlInit()
            visible = list(range(pynvml.nvmlDeviceGetCount()))
        except Exception:
            visible = None
        if visible is not None and gpu not in visible:
            raise ValueError(
                f"add: GPU {gpu} not visible to NVML (visible={visible}); "
                f"refusing to add a non-existent device to the pool")

        added = Slots.add(gpu)
        if not added:
            log.warning("add: GPU %d already in pool", gpu)
            return
        if gpu not in Orchestrator._gpu_ids:
            Orchestrator._gpu_ids.append(gpu)
        log.info("add: GPU %d joined pool", gpu)

    @staticmethod
    def sub(gpu: int) -> None:
        """Mark *gpu* draining and submit a non-blocking drain to the pool.

        Phase 1 (synchronous, under ``Slots._cv``): flag *gpu* in
        :attr:`Slots._draining` so :func:`Slots._try_allocate` stops
        handing out slots on it.  Any in-flight FIFO waiters get a
        chance to retry against non-draining GPUs.

        Phase 2 (background, on ``_pool``): :meth:`_sub_sync`
        repeatedly snapshots residents and issues
        ``move(mid, "checkpoint")`` until no slot or registry entry
        references the GPU, then pops it from the pool.

        Returns immediately; the drain future is stored in
        :attr:`_gpu_futures` and is awaitable via :meth:`wait_gpu`.
        """
        with Slots._cv:
            if not Slots._inited:
                log.warning("sub: Slots not initialised")
                return
            if gpu not in Slots._last_used:
                log.warning("sub: GPU %d not in pool", gpu)
                return
            if gpu in Slots._draining:
                log.warning("sub: GPU %d already draining", gpu)
                return
            Slots._draining.add(gpu)
            Slots._cv.notify_all()
            log.info("sub: GPU %d marked draining", gpu)

        # Run the drain loop on a daemon thread (was on ``_pool``;
        # the pool is gone in pipeline-only mode).  ``_gpu_futures``
        # keeps a Future-shaped handle so ``wait_gpu`` continues to
        # work without changes.
        gpu_fut: Future = Future()

        def _drain(_gpu: int = gpu, _fut: Future = gpu_fut) -> None:
            try:
                Orchestrator._sub_sync(_gpu)
                _fut.set_result(None)
            except BaseException as exc:  # noqa: BLE001
                _fut.set_exception(exc)

        threading.Thread(
            target=_drain,
            name=f"sub[gpu={gpu}]",
            daemon=True,
        ).start()
        Orchestrator._gpu_futures[gpu] = gpu_fut

    @staticmethod
    def _sub_sync(gpu: int) -> None:
        """Snapshot-await drain loop.

        Each pass: under ``Slots._cv``, find every model with either
        ``entry["gpu"] == gpu`` or ``entry["slot"].gpu_id == gpu`` (the
        double criterion catches a model whose ``_step_up`` wrote one
        of the two fields but not yet the other).  If no residents
        remain and no orphan slot is in :attr:`Slots._live` for *gpu*,
        reap the GPU.

        Move issuance is **at most once per resident per drain**.  We
        track per-mid move futures locally; a model is only reissued
        if it has fully drained (popped from ``submitted``) and a new
        resident appears (e.g. someone migrated back onto *gpu*).
        Between passes we park on ``Slots._cv`` with a timeout so a
        slot deallocate / state flip / migration wakes us instead of
        the loop busy-spinning.

        Why this matters: the previous implementation called
        ``Orchestrator.move(mid, "checkpoint")`` every iteration and
        awaited the new future inline.  ``move()`` overwrites
        ``_futures[mid]`` on each call, and the new ``_move_sync``
        chains on the prior (failed) future via ``prev.result()``,
        re-raising the prior exception without re-reading the world.
        With Fix 1 (move waits on ``_inflight`` events) the genuine
        race is gone, but the at-most-once discipline below still
        prevents that re-poisoning shape from re-emerging in any
        future variant of ``_move_sync``.

        The orphan-slot branch (slot in ``_live`` with no registry
        owner) keeps its 50ms ``cv.wait``: it covers the microsecond
        window between ``Slots.allocate`` returning and
        ``entry["slot"] = slot`` being written under the model's own
        lock.
        """
        log.info("sub: GPU %d drain starting", gpu)
        submitted: dict[str, Future] = {}  # mid -> in-flight move future
        while True:
            with Slots._cv:
                residents: set[str] = set()
                for mid, e in Orchestrator._registry.items():
                    if e.get("gpu") == gpu:
                        residents.add(mid)
                        continue
                    s = e.get("slot")
                    if s is not None and s.gpu_id == gpu:
                        residents.add(mid)
                orphan = (not residents
                          and any(s.gpu_id == gpu for s in Slots._live))
                if orphan:
                    log.info("sub: GPU %d orphan slot in flight; "
                             "waiting briefly", gpu)
                    Slots._cv.wait(timeout=0.05)
                    continue
                if not residents and not submitted:
                    # Lock released by the ``with`` exit below.
                    # ``Slots.pop`` re-acquires ``_cv`` internally;
                    # calling it while holding ``_cv`` would deadlock
                    # (non-reentrant lock backing the Condition).
                    # Reaping outside ``_cv`` is safe because
                    # ``Slots._draining`` is still set, so no new
                    # slot can land on *gpu* between this emptiness
                    # check and the pop -- ``_try_allocate`` skips
                    # draining GPUs unconditionally.
                    break

            # Issue one move per resident we haven't already issued
            # one for.  Skip mids whose previous move future is still
            # pending (we're still waiting on it).
            new_residents = sorted(residents - submitted.keys())
            if new_residents:
                log.info("sub: GPU %d draining residents %s",
                         gpu, new_residents)
                for mid in new_residents:
                    pipe = Orchestrator._pipelines.get(mid)
                    if pipe is None:
                        continue
                    f = pipe.submit(MoveOp(target="checkpoint"))
                    submitted[mid] = f

            # Park on cv until something changes (slot deallocate,
            # state flip, residency change) or the timeout fires.
            # When ``submitted`` is empty but residents remain (rare;
            # e.g. a model in ``saved`` state still showing
            # ``entry["gpu"] == gpu`` for some reason), we still wait
            # on the cv so the next pass can re-evaluate.
            with Slots._cv:
                Slots._cv.wait(timeout=0.5)

            # Reap completed move futures, log failures, but DO NOT
            # resubmit -- the next pass's registry scan picks up any
            # mid that's still resident and we'll only reissue if it
            # has been popped here (i.e. its prior move completed).
            for mid in list(submitted.keys()):
                f = submitted[mid]
                if not f.done():
                    continue
                try:
                    f.result()
                except Exception as exc:
                    log.warning("sub: GPU %d resident %s move failed: %s",
                                gpu, mid, exc)
                submitted.pop(mid, None)

        Slots.pop(gpu)
        if gpu in Orchestrator._gpu_ids:
            Orchestrator._gpu_ids.remove(gpu)
        log.info("sub: GPU %d removed from pool", gpu)

    @staticmethod
    def wait_gpu(gpu: int) -> None:
        """Block until the pending :meth:`sub` for *gpu* completes.

        No-op if no drain is in flight.  Does NOT participate in
        :meth:`wait` (jobs vs GPUs are separate concerns).
        """
        log.info("wait_gpu  gpu=%d", gpu)
        t0 = time.perf_counter()
        fut = Orchestrator._gpu_futures.get(gpu)
        if fut is not None:
            fut.result()
        elapsed = time.perf_counter() - t0
        log.info("wait_gpu done  (%.1fs)", elapsed)

    # ------------------------------------------------------------------
    # move  (walk the state ladder)
    # ------------------------------------------------------------------

    @staticmethod
    def move(model_id: str, target: str, target_gpu: int | None = None) -> None:
        """Move *model_id* to *target* state by walking the ladder.

        Valid targets: ``"saved"``, ``"checkpoint"``, ``"sleep"``, ``"up"``.
        Submits the transition to the thread pool (non-blocking).

        *target_gpu* (optional) is only valid when *target* is ``"sleep"``
        and pins the model onto that specific GPU.  This is a **slotless
        sleep** flavour: no slot is allocated and any slot the model
        currently holds is released.  The model parks on ``target_gpu``
        without competing for slot resources; on the next ``sleep -> up``
        it goes through the standard tier-A/B/C acquisition logic (which
        may stay on ``target_gpu`` or migrate elsewhere depending on what
        is free).
        """
        if target not in _STATES:
            raise ValueError(f"invalid target state '{target}'; "
                             f"must be one of {_STATES}")
        if target_gpu is not None and target != "sleep":
            raise ValueError("target_gpu may only be specified when target is 'sleep'")
        if target_gpu is not None and target_gpu in Slots._draining:
            raise ValueError(
                f"GPU {target_gpu} is draining; cannot move {model_id!r} onto it")
        entry = Orchestrator._registry.get(model_id)
        if entry is None:
            log.warning("move(%r, %r) skipped – model not registered",
                        model_id, target)
            return
        # Paused models must not reach ``saved``: tearing the live
        # process down would orphan every per-request Future that
        # ``_generate_sync`` is parked on (the saved sub-requests can
        # only complete via ``resume``).  Reject explicitly so the
        # caller learns to ``resume`` first.
        if target == "saved" and entry.get("paused"):
            log.warning("%s: move(target='saved') refused while paused; "
                        "call resume() (and let in-flight generates "
                        "drain) before moving to 'saved'", model_id)
            return
        if target_gpu is not None:
            log.info("move  model_id=%s  target=%s  target_gpu=%s",
                     model_id, target, target_gpu)
        else:
            log.info("move  model_id=%s  target=%s", model_id, target)

        pipe = Orchestrator._pipelines.get(model_id)
        if pipe is None:
            # Init's image-cache scan should have created one; fall
            # back defensively (e.g. a register raced).
            pipe = Orchestrator._make_pipeline(model_id)
        pipe.submit(MoveOp(target=target, target_gpu=target_gpu))

    @staticmethod
    def _acquire_slot_for_running(model_id: str) -> None:
        """Reacquire a slot for a slotless ``up`` model that is about to
        publish ``running`` (or ``resume``).

        Tier A: ``Slots.try_allocate(level, gpu=home_gpu)`` -- if the
        home GPU is free right now, claim the slot in place; weights
        stay in HBM.

        Fallback: retreat to ``sleep`` (frees HBM), then climb back
        via the standard ``sleep -> up`` Tier-A/B/C path which handles
        migration, blocking FIFO, and Phase-2 HBM eviction.  Before
        retreating we re-check the model's published state under its
        own lock: if a peer's Phase-2 eviction has *already* stepped
        us down (state != "up"), we skip the redundant own-side
        ``_step_down`` and just climb back from wherever the peer
        left us.  This is best-effort (the lock can't be held across
        the cmd send without inverting against the demuxer), but
        even when both sides race the FIFO ack registry tolerates
        the duplicate sleep cmd, so the worst case is benign churn
        instead of an orphan ack.

        On return, ``entry["slot"]`` is set and ``entry["state"]`` is
        ``"up"`` (slotted).  Caller is responsible for the final
        ``_set_state`` flip to ``running`` (or whatever sub-state).
        """
        entry = Orchestrator._registry[model_id]
        if entry.get("slot") is not None:
            return
        level = entry["level"]
        home_gpu = entry["gpu"]
        slot = Slots.try_allocate(level=level, gpu=home_gpu)
        if slot is not None:
            with Orchestrator._locks_ordered(model_id):
                entry["slot"] = slot
            log.info("%s: claimed slot on GPU %s", model_id, home_gpu)
            return
        with Orchestrator._locks_ordered(model_id):
            need_retreat = (entry.get("state") == "up"
                            and entry.get("slot") is None)
        if need_retreat:
            log.info("%s: no slot on GPU %s, retreating to sleep",
                     model_id, home_gpu)
            Orchestrator._step_down(model_id, "up", "sleep")
        else:
            log.info("%s: peer already stepped down (state=%s); "
                     "climbing back without own-side retreat",
                     model_id, entry.get("state"))
        Orchestrator._step_up(model_id, "sleep", "up")

    @staticmethod
    def _step_up(model_id: str, from_state: str, to_state: str,
                 *, target_gpu: int | None = None,
                 announce_state: str | None = None) -> None:
        """Execute one upward step on the ladder.

        *announce_state* overrides the published state when the step
        completes.  Used by generate to atomically go sleep -> running
        (skipping the observable 'up' window).
        """
        entry = Orchestrator._registry[model_id]

        if from_state == "saved" and to_state == "checkpoint":
            with Orchestrator._locks_ordered(model_id):
                inst = Instance(entry["vllm_config"])
                # Install the orchestrator's per-model demuxer
                # listeners before any cmd is sent so their callbacks
                # are live for the very first ack.  The demuxer is
                # created lazily in _ensure_queues (driven by
                # criu_restore below); add_cmd_listener buffers
                # registrations until then.
                Orchestrator._install_listeners(model_id, inst)
                # load() reads meta.json and hydrates total_gpu_bytes /
                # pinned_cpu_bytes on the instance; plan_restore_weights
                # uses those to build the chunk plan in the child once.
                inst.criu_restore(entry["image_dir"]).plan_restore_weights().wait()
                # Mirror onto the registry entry to keep registry-as-truth
                # even though Instance.criu_restore already set self.* from meta.
                inst.pinned_cpu_bytes = entry.get("pinned_cpu_bytes", 0)
                inst.total_gpu_bytes = entry.get("total_gpu_bytes", 0)
                entry["instance"] = inst
                Orchestrator._set_state(model_id, "checkpoint")
            return

        if from_state == "checkpoint" and to_state == "sleep":
            if target_gpu is None:
                # Publish a transient "wait" state only if Slots.allocate
                # actually has to block; otherwise the model races
                # checkpoint -> sleep without ever observably entering
                # "wait", which keeps zero-duration ``wait``/``ckpt``
                # entries out of every request's state_log.
                published_wait = False

                def _on_block() -> None:
                    nonlocal published_wait
                    published_wait = True
                    Orchestrator._set_state(model_id, "wait")

                t_wait = time.perf_counter()
                slot = Slots.allocate(level=entry["level"],
                                      on_block=_on_block)
                gpu = slot.gpu_id
                Orchestrator._timing.gpu_wait_s = time.perf_counter() - t_wait
                if published_wait:
                    Orchestrator._set_state(model_id, "checkpoint")
            else:
                # Slotless-sleep flavour: user pinned a target GPU, so
                # park the model there without consuming a slot.
                slot = None
                gpu = target_gpu
                Orchestrator._timing.gpu_wait_s = 0.0
            with Orchestrator._locks_ordered(model_id):
                entry["slot"] = slot
                entry["gpu"] = gpu
                tag = "" if slot is not None else " (slotless)"
                log.info("%s: placed on GPU %s%s", model_id, gpu, tag)
            # _send_cmd_with_ack drops the model lock around sends
            # so it doesn't invert against the demuxer's generate
            # listener, which takes gen_lock first and then the
            # model lock.
            Orchestrator._send_cmd_with_ack(model_id, "cuda_restore", gpu)
            Orchestrator._send_cmd_with_ack(model_id, "repin")
            with Orchestrator._locks_ordered(model_id):
                Orchestrator._set_state(model_id, "sleep")
            return

        if from_state == "sleep" and to_state == "up":
            inst = entry["instance"]
            level = entry["level"]
            home_gpu = entry["gpu"]

            # Phase 1: ensure we hold a slot.
            if entry["slot"] is None:
                # Tier A: home GPU is free right now.
                slot = Slots.try_allocate(level=level, gpu=home_gpu)
                if slot is None:
                    # Tier B: any GPU free right now (possibly elsewhere).
                    slot = Slots.try_allocate(level=level)
                if slot is not None:
                    # Commit ownership before any (potentially slow)
                    # migration so the dashboard sees the slot held
                    # throughout instead of a slotless ``sleep``
                    # interlude on the wrong GPU.
                    with Orchestrator._locks_ordered(model_id):
                        entry["slot"] = slot
                    if slot.gpu_id != home_gpu:
                        t_mig = time.perf_counter()
                        log.info("%s: migrating GPU %s -> GPU %s",
                                 model_id, home_gpu, slot.gpu_id)
                        # Mirror the brief checkpoint pass-through in
                        # the published state so dashboards see the
                        # sleep -> checkpoint -> sleep transition.
                        # Sends go through ``_send_cmd_with_ack`` (not
                        # ``inst.wait()``) so multiple concurrent
                        # senders -- e.g. a paused-resume on this
                        # model overlapping with a peer's eviction --
                        # each get their own FIFO ack Event.  Lock
                        # only around state mutations to avoid
                        # model_lock -> gen_lock inversion against
                        # the demuxer's generate listener.
                        Orchestrator._send_cmd_with_ack(model_id, "unpin")
                        Orchestrator._send_cmd_with_ack(model_id, "cuda_checkpoint")
                        with Orchestrator._locks_ordered(model_id):
                            entry["gpu"] = None
                            Orchestrator._set_state(model_id, "checkpoint")
                        Orchestrator._send_cmd_with_ack(
                            model_id, "cuda_restore", slot.gpu_id)
                        Orchestrator._send_cmd_with_ack(model_id, "repin")
                        with Orchestrator._locks_ordered(model_id):
                            entry["gpu"] = slot.gpu_id
                            Orchestrator._set_state(model_id, "sleep")
                        Orchestrator._timing.migrate_s = (
                            time.perf_counter() - t_mig)
                else:
                    # Tier C: nothing free -> retreat and FIFO.
                    log.info("%s: waiting for slot ...", model_id)
                    t_wait = time.perf_counter()
                    Orchestrator._step_down(model_id, "sleep", "checkpoint")
                    Orchestrator._step_up(model_id, "checkpoint", "sleep")
                    Orchestrator._timing.gpu_wait_s = (
                        time.perf_counter() - t_wait)
                    slot = entry["slot"]
                entry["gpu"] = slot.gpu_id
                home_gpu = slot.gpu_id

            # Phase 2: free enough HBM on home_gpu for this model's
            # wake-up by evicting the oldest slotless `up` incumbents,
            # but no more than necessary.
            #
            # HBM accounting (each level-L share == 1 / 2**(L-1) of the
            # GPU):
            #
            #   slotted_others = sum of slot shares for every OTHER
            #                    slotted model on home_gpu, regardless
            #                    of state.  We use the slot share, not
            #                    the live state, because slot allocation
            #                    is what serialises wake-ups: a slotted
            #                    `sleep` model whose Phase 3 is already
            #                    in flight on the worker (queued behind
            #                    ours) will be HBM-resident by the time
            #                    our Phase 3 finishes, even though its
            #                    registry state still reads "sleep".
            #   new_share      = the slot we just took for this model;
            #                    it will be resident at end of Phase 3.
            #   slack          = 1.0 - slotted_others - new_share
            #                  = HBM the buddy allocator hasn't handed
            #                    out, available for slotless squatters.
            #
            # The buddy allocator already guarantees the sum of all
            # slot shares on home_gpu is <= 1.0, so slack is >= 0.
            # Slotless `up` squatters consume HBM but no slot, so they
            # must fit inside `slack`.  Evict oldest-first until the
            # remaining slotless share fits.
            #
            # Example (the case originally raised): an L2 wakes up on
            # a GPU with two slotless L3 squatters (each 0.25), one
            # slotless L2 squatter (0.5), and no other slotted models.
            # new_share=0.5, slotted_others=0, slack=0.5.  Slotless
            # total = 1.0; eviction stops as soon as the remaining
            # slotless share is <= 0.5.  If the two L3s are oldest,
            # both get evicted (frees 0.5) and the L2 squatter stays.
            # Use ``slot.gpu_id`` as the source of truth for slotted
            # residency on ``home_gpu`` -- it matches ``Slots._live``
            # exactly.  ``entry["gpu"]`` is *not* reliable here: during a
            # peer's migration its slot flips to the new GPU at line 624
            # well before ``entry["gpu"]`` is updated (None during the
            # checkpoint pass-through, then the new GPU after restore).
            # A scan keyed on ``entry["gpu"] == home_gpu`` would therefore
            # double-count peers whose slot has already moved away,
            # driving ``slack`` negative and either tripping the assert
            # below or starving Phase 3 of room.  Slotless residents
            # don't migrate their slot (they have none), so falling back
            # to ``entry["gpu"]`` for them is correct.
            new_share = 1.0 / (1 << (level - 1))
            slotted_others = 0.0
            slotless_cands: list[tuple[float, str, float]] = []
            for mid, e in Orchestrator._registry.items():
                if mid == model_id:
                    continue
                s = e.get("slot")
                if s is not None:
                    if s.gpu_id != home_gpu:
                        continue
                    slotted_others += 1.0 / (1 << (s.level - 1))
                elif (e.get("gpu") == home_gpu
                        and e.get("state") == "up"):
                    e_share = 1.0 / (1 << (e["level"] - 1))
                    slotless_cands.append(
                        (e.get("state_since", 0.0), mid, e_share))
            slack = 1.0 - slotted_others - new_share
            assert slack >= -1e-9, (
                f"HBM over-subscribed on GPU {home_gpu}: "
                f"slotted_others={slotted_others}, new_share={new_share}")
            slotless_cands.sort()
            remaining = sum(share for _, _, share in slotless_cands)
            for _, incumbent, share in slotless_cands:
                if remaining <= slack + 1e-9:
                    break
                # Re-validate the candidate under its own lock before
                # touching its instance.  The Phase 2 scan above is
                # lock-free, so an incumbent we picked may have
                # self-evacuated in the meantime (its own thread retreated
                # from `up` to acquire a slot, leaving the child process
                # CRIU-checkpointed).  Queuing a `sleep` on a checkpointed
                # child raises "child pipe broken".
                #
                # Lock ordering: hold ``_lock`` only for the
                # validation, then release it before calling
                # ``_step_down``.  Holding ``_lock`` across
                # ``_step_down`` deadlocks against the incumbent's
                # own thread when it is parked in
                # ``_send_cmd_with_ack`` while still holding the
                # outer ``gen_lock`` (typical pattern: the
                # incumbent's ``_generate_sync`` is in
                # ``_acquire_slot_for_running`` retreating to sleep).
                # In that scenario T1 holds ``gen_lock`` and waits
                # for ``_lock``; we'd hold ``_lock`` and our
                # ``_step_down`` would wait for ``gen_lock`` -- AB-BA.
                # The race window opened by releasing ``_lock`` early
                # is handled on the ``_step_down`` side: vllm sleep is
                # idempotent, the FIFO ack registry tolerates
                # duplicates by design (see ``_send_cmd_with_ack``),
                # and ``_step_down(up, sleep)`` reconciles the
                # registry to the post-sleep worker state
                # unconditionally (releases any slot, flips to
                # ``sleep``).  That covers both well-behaved peers
                # (incumbent self-evacuated to ``sleep`` between our
                # check and our send -- our ``_set_state`` is the
                # no-op same-state path) and the pathological case
                # where the incumbent raced all the way back up to
                # ``running`` while our sleep ack was queued behind a
                # generate on the worker -- we still drop the slot
                # the late ``_on_generate_done`` could not release
                # (its ``_pending_count == 0`` guard saw our sleep as
                # still in flight) and reconcile state to ``sleep``
                # so the publish view matches the asleep engine.
                # Either branch decrements ``remaining``: if we
                # evicted, we freed ``share``; if the incumbent
                # self-evacuated, it already freed ``share`` on its
                # own.
                inc_entry = Orchestrator._registry[incumbent]
                with Orchestrator._locks_ordered(incumbent):
                    evict_now = (inc_entry.get("slot") is None
                                 and inc_entry.get("state") == "up"
                                 and inc_entry.get("gpu") == home_gpu)
                if evict_now:
                    log.info("%s: evicting %s from GPU %s",
                             model_id, incumbent, home_gpu)
                    # Cross-pipeline submit: the acquirer's pipeline
                    # worker (currently running this ``_step_up``
                    # inside its ``MoveOp`` / ``GenerateOp.execute``)
                    # submits onto the incumbent's pipeline and
                    # blocks on the result.  ``submit_to_peer_and_wait``
                    # sets ``_waiting_on`` so the cycle detector sees
                    # the edge.  See pipeline_DESIGN.md section 4.3
                    # and ``EvictForPeerOp``.
                    acquirer_pipe = Orchestrator._pipelines.get(model_id)
                    incumbent_pipe = Orchestrator._pipelines.get(incumbent)
                    if (acquirer_pipe is not None
                            and incumbent_pipe is not None):
                        acquirer_pipe.submit_to_peer_and_wait(
                            incumbent_pipe,
                            EvictForPeerOp(acquirer_id=model_id),
                        )
                    else:
                        # No pipeline registered for one of the
                        # parties (e.g. incumbent in mid-tear-down).
                        # Fall back to a direct ``_step_down`` -- the
                        # post-sleep reconcile is idempotent.
                        Orchestrator._step_down(incumbent, "up", "sleep")
                remaining -= share

            # Phase 2 diagnostic: re-scan HBM residency on home_gpu and
            # warn if any slotless ``up`` peer still squats beyond the
            # acquirer's ``slack``.  The eviction loop's
            # ``remaining -= share`` is unconditional (it pre-deducts
            # the share regardless of whether the cross-pipeline
            # ``EvictForPeerOp`` actually sent ``sleep``), so a
            # soft-failed eviction would otherwise be invisible until
            # ``wake_up_kv_cache`` OOMs with CUresult=2 -- the exact
            # shape of the "Paused-peer silent OOM" hang (06:37
            # 2026-05-18, six waiting requests).  That specific bug is
            # fixed at the EvictForPeerOp layer (paused-skip removed),
            # but the diagnostic stays as cheap insurance against any
            # future soft-fail path: if this warning fires, the next
            # log line is almost certainly the OOM.
            residual: list[tuple[str, float]] = []
            for _mid, _e in Orchestrator._registry.items():
                if _mid == model_id:
                    continue
                if _e.get("slot") is not None:
                    continue
                if _e.get("gpu") != home_gpu:
                    continue
                if _e.get("state") != "up":
                    continue
                residual.append((_mid, 1.0 / (1 << (_e["level"] - 1))))
            residual_share = sum(s for _, s in residual)
            if residual_share > slack + 1e-9:
                log.warning(
                    "%s: Phase-2 eviction left residual HBM on GPU %s "
                    "(residual=%.2f > slack=%.2f); next wake_up_kv_cache "
                    "may OOM. Residual slotless-up peers: %s",
                    model_id, home_gpu, residual_share, slack,
                    ", ".join(
                        f"{m}(paused={Orchestrator._registry[m].get('paused', False)},"
                        f"share={s:.2f})"
                        for m, s in residual),
                )

            # Phase 3: weights to HBM, announce up/running.
            # Use _send_cmd_with_ack -- when a paused model is
            # resuming, its deferred generate cmd is still pending
            # in the demuxer, so ``inst.wait()`` would not return
            # until the generate eventually drained on a future
            # resume.  Lock only around the state flip to avoid
            # model_lock -> gen_lock inversion against the demuxer's
            # generate listener.
            Orchestrator._send_cmd_with_ack(model_id, "wake_up_weights")
            Orchestrator._send_cmd_with_ack(model_id, "restore_weights")
            Orchestrator._send_cmd_with_ack(model_id, "wake_up_kv_cache")
            with Orchestrator._locks_ordered(model_id):
                Orchestrator._set_state(model_id, announce_state or "up")
            return

        raise AssertionError(
            f"_step_up unexpected transition {from_state} -> {to_state}")

    @staticmethod
    def _step_down(model_id: str, from_state: str, to_state: str) -> None:
        """Execute one downward step on the ladder."""
        entry = Orchestrator._registry[model_id]

        # NOTE: model lock is taken only around registry mutations
        # (slot/gpu/state).  The cmd sends use ``_send_cmd_with_ack``
        # which internally takes ``gen_lock`` -- holding model_lock
        # across that would invert the demuxer's generate listener,
        # which takes gen_lock -> model_lock.
        if from_state == "up" and to_state == "sleep":
            # Use _send_cmd_with_ack so multiple concurrent senders
            # (e.g. peer-eviction sleep racing with a self-evacuation
            # sleep on the same model) each get their own FIFO Event;
            # and so paused models -- whose deferred generate cmd
            # keeps ``_pending_count`` from reaching zero -- can
            # still synchronise on the specific cmd's ack.
            Orchestrator._send_cmd_with_ack(model_id, "sleep")
            # Reconcile the registry to the worker on the *post-sleep*
            # side, not the caller's expected pre-sleep side.  The
            # worker has just executed ``llm.sleep(level=2)`` so the
            # engine is asleep and any HBM the slot represented is
            # released, regardless of which state the registry reads
            # right now.
            #
            # Race window that motivates the unconditional release:
            # an evictor's gate passes when the model is slotless+up,
            # but its ``_send_cmd_with_ack("sleep")`` parks on the ack
            # while a fresh ``_generate_sync`` Phase 2 enqueues a
            # ``generate`` ahead of the still-pending ``sleep`` on the
            # worker.  The worker drains the generate, then runs the
            # late sleep.  By the time the ack lands here, the model
            # is ``running`` with a slot that ``_on_generate_done``
            # already declined to free (its ``_pending_count == 0``
            # guard saw the queued sleep as still in flight).
            # Without this branch the slot leaks and the registry
            # publishes ``running`` over a sleeping engine.
            #
            # Legitimate callers (``_acquire_slot_for_running``
            # retreat, Phase-2 eviction, ``_move_sync`` down-walk)
            # all enter with ``slot is None`` or expect the slot to be
            # released, so the unconditional deallocate is a no-op for
            # them.
            with Orchestrator._locks_ordered(model_id):
                if entry.get("slot") is not None:
                    Slots.deallocate(entry["slot"])
                    entry["slot"] = None
                Orchestrator._set_state(model_id, "sleep")
            return

        if from_state == "sleep" and to_state == "checkpoint":
            Orchestrator._send_cmd_with_ack(model_id, "unpin")
            Orchestrator._send_cmd_with_ack(model_id, "cuda_checkpoint")
            with Orchestrator._locks_ordered(model_id):
                if entry.get("slot") is not None:
                    Slots.deallocate(entry["slot"])
                    entry["slot"] = None
                entry["gpu"] = None
                Orchestrator._set_state(model_id, "checkpoint")
            return

        if from_state == "checkpoint" and to_state == "saved":
            # Paused models are blocked from reaching this step by
            # ``move()``/``remove()``.  ``inst.wait()`` is now a
            # condvar-based idle wait on the demuxer's
            # ``_pending_count``, so it's safe under any number of
            # concurrent waiters; the demuxer itself drains and
            # applies the teardown ack (which calls into
            # ``Instance._reset``) before notifying the wait.
            with Orchestrator._locks_ordered(model_id):
                entry["instance"].teardown().wait().remove()
                entry["instance"] = None
                Orchestrator._set_state(model_id, "saved")
            return

            raise AssertionError(
                f"_step_down unexpected transition {from_state} -> {to_state}")

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    @staticmethod
    def submit_generate(model_id: str, prompts: list[str] | str,
                        sampling_params: dict | int | None = None
                        ) -> tuple[int | None, Future | None]:
        """Submit a non-blocking generate; returns ``(req_id, future)``.

        Companion to :meth:`generate` for callers that need a stable
        request id (e.g. the HTTP control plane) without racing on
        ``_request_counter``.  Returns ``(None, None)`` when the model is
        not registered, mirroring :meth:`generate`'s warn-and-skip
        semantics.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(sampling_params, int):
            sampling_params = {"max_tokens": sampling_params, "ignore_eos": True}
        if sampling_params is None:
            sampling_params = {}
        entry = Orchestrator._registry.get(model_id)
        if entry is None:
            log.warning("model %r is not registered, skipping generate",
                        model_id)
            return None, None
        log.info("generate  model_id=%s", model_id)

        with Orchestrator._request_lock:
            ent = Orchestrator._registry[model_id]
            start_state = ent.get("state")
            req_id = Orchestrator._request_counter
            Orchestrator._request_counter += 1
            t_submit = time.perf_counter()
            req_record = {
                "req_id": req_id,
                "model_id": model_id,
                "state": "waiting",
                "start_state": start_state,
                "t_submit": t_submit,
                "t_gen_start": None,
                "t_done": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "ttft_s": None,
                "tpot_ms": None,
                # Timeline of (t_perf, model_state) pairs observed
                # while this request is in flight.  Seeded with the
                # model's state at submit; _set_state appends more.
                "state_log": [(t_submit, start_state)] if start_state else [],
                # Cumulative seconds this request has spent suspended
                # because its model was paused; ``t_pause_started`` is
                # the timestamp of the current pause window (None when
                # not paused).  ``state_server`` subtracts both from
                # the reported ``gen_s`` so the visible counter freezes
                # while paused and continues from where it left off
                # after resume.
                "paused_s": 0.0,
                "t_pause_started": None,
                # One-shot marker: True iff this request was submitted
                # while its model was already paused (orchestrator-side
                # ``entry["paused"]``).  Immutable for the request's
                # lifetime so ``state_server`` can keep classifying R
                # as a pause-piggyback follower (collapsing the bracket
                # to ``wait #L+1``) even after ``_resume_sync`` clears
                # the transient ``t_pause_started`` stamp -- matching
                # how climb-piggyback persists past the lead's
                # ``t_gen_start``.
                "paused_at_submit": bool(ent.get("paused")),
            }
            Orchestrator._request_log.append(req_record)

        # Submit the GenerateOp and return a Future that resolves with
        # the user-visible result list.
        #
        # GenerateOp.execute returns a PendingRequest as soon as
        # Phase 1+2 are done; a helper daemon thread does the Phase-3
        # done_event wait + result collection so the pipeline worker
        # is freed for subsequent ops on this model (a PauseOp from a
        # racing pause(), the next GenerateOp, an EvictForPeerOp from
        # a peer's wake-up).
        pipe = Orchestrator._pipelines.get(model_id)
        if pipe is None:
            pipe = Orchestrator._make_pipeline(model_id)
        op_fut = pipe.submit(
            GenerateOp(
                prompts=prompts,
                sampling_params=sampling_params,
                q_rec=req_record,
            )
        )
        user_fut: Future = Future()

        def _wait_and_collect(_op_fut: Future = op_fut,
                              _user_fut: Future = user_fut,
                              _mid: str = model_id,
                              _q_rec: dict = req_record) -> None:
            try:
                pending: PendingRequest = _op_fut.result()
            except BaseException as exc:  # noqa: BLE001
                # Mark the dashboard record as errored.  Without this
                # the q_rec stays at ``state="waiting"`` / ``t_done=None``
                # forever and the dashboard's ``wait_s`` counter ticks
                # up indefinitely -- a Phase-1 failure (e.g.
                # ``WorkerCmdFailed`` from ``cuda_restore`` OOM while
                # walking checkpoint->sleep) never reaches the Phase-2
                # ``q_rec["state"] = "generating"`` assignment and is
                # not in ``_inflight``, so ``_on_generate_done`` /
                # ``_fail_all_inflight`` never see it.  The user's
                # future already carries *exc* (set just below); this
                # write is the matching dashboard-side ack so the
                # observer sees the request terminate.
                #
                # Take ``_request_lock`` to match ``_set_state``'s
                # ``rec.get("t_done")`` skip-guard: an interleaving
                # state-transition iteration either sees a still-live
                # rec (and appends a final state_log entry -- harmless
                # tail) or sees ``t_done`` set (and skips).  No torn
                # read.
                with Orchestrator._request_lock:
                    if _q_rec.get("t_done") is None:
                        _q_rec["state"] = "error"
                        _q_rec["t_done"] = time.perf_counter()
                        _q_rec["error"] = repr(exc)
                _user_fut.set_exception(exc)
                return
            # Phase 3 (user-thread wait): survives a pause window
            # because PauseOp does not destroy the engine request --
            # the engine resumes computing tokens after a future
            # ResumeOp.
            pending.done_event.wait()
            result: list = []
            if (pending.inst is not None
                    and pending.q_rec.get("state") == "done"
                    and pending.rid):
                gen_result = pending.inst.generate_results.pop(
                    pending.rid, None)
                if gen_result:
                    result = gen_result.get("outputs") or []
            _user_fut.set_result(result)

        # Phase 3 must NOT run on the pipeline worker (it would block
        # subsequent ops on this model for the duration of the
        # engine's token generation).  A dedicated daemon thread does
        # the wait so the worker stays available.
        threading.Thread(
            target=_wait_and_collect,
            name=f"generate-wait[{model_id}:{req_id}]",
            daemon=True,
        ).start()
        Orchestrator._generate_futures.append(user_fut)
        return req_id, user_fut

    @staticmethod
    def generate(model_id: str, prompts: list[str] | str,
                 sampling_params: dict | int | None = None) -> Future:
        """Submit a non-blocking generate.  Returns a Future[list].

        *sampling_params* can be a dict, or an int shorthand for
        ``{"max_tokens": N}``.

        Automatically moves the model to **up** if needed, runs inference,
        and leaves the model in **running** state until a waiter thread
        detects all in-flight requests are done, then transitions to **up**.
        """
        _, fut = Orchestrator.submit_generate(model_id, prompts, sampling_params)
        return fut

    # ------------------------------------------------------------------
    # demuxer listeners (per-model, installed at saved->checkpoint)
    # ------------------------------------------------------------------

    @staticmethod
    def _install_listeners(model_id: str, inst: Instance) -> None:
        """Install the orchestrator's demuxer listeners on *inst*.

        Two listeners cover everything the legacy
        ``_start_generate_waiter`` did:

        * ``cmd="generate"``: per-request done-event resolution and
          slot release on the trailing edge of inflight.
        * ``cmd=None`` (catch-all): pop the FIFO head Event for any
          non-generate cmd so any parked ``_send_cmd_with_ack`` caller
          wakes up.

        Listener callbacks fire on the demuxer thread, after
        ``_apply_result`` has updated Instance state and after
        ``_pending_count`` has been decremented (see
        ``demuxer.Demuxer._handle``).  Closures capture *model_id* so
        the listeners survive ``entry["instance"]`` rotation across
        teardown / re-load cycles.
        """
        inst.add_cmd_listener(
            "generate",
            lambda cmd, e, err, info: Orchestrator._on_generate_done(
                model_id, cmd, e, err, info),
        )
        inst.add_cmd_listener(
            None,
            lambda cmd, e, err, info: Orchestrator._on_cmd_ack(
                model_id, cmd, e, err, info),
        )

    @staticmethod
    def _on_cmd_ack(model_id: str, cmd: str, elapsed: float,
                    error: object | None, info: dict) -> None:
        """Catch-all demuxer listener: pop the FIFO head ack-slot for
        *cmd*, copy the worker's ``error`` payload onto it, then set
        the Event so any parked ``_send_cmd_with_ack`` caller wakes
        up.  The caller re-raises if ``error`` is non-None -- this is
        what turns a child-side failure (CUDA OOM, restore failure,
        ...) into a surfaced exception instead of a silent hang.

        Ignores ``cmd == "generate"`` because generate cmds are sent
        directly via ``inst.generate(...)`` (not through
        ``_send_cmd_with_ack``) and have no FIFO slot paired with
        them.  When no slot is queued (e.g. for cmds sent on the
        cold-start chain in ``_register_sync`` before any listener
        was installed, or for a chained ``inst.teardown().wait()``
        that bypasses ``_send_cmd_with_ack``) this is a benign no-op.
        """
        if cmd == "generate":
            return
        with Orchestrator._cmd_ack_lock:
            q = Orchestrator._cmd_ack_events.get((model_id, cmd))
            ev = q.popleft() if q else None
        if ev is not None:
            ev.error = error
            ev.set()

    @staticmethod
    def _on_generate_done(model_id: str, cmd: str, elapsed: float,
                          error: object | None, info: dict) -> None:
        """Demuxer listener for ``cmd="generate"``.

        Resolves the matching inflight entry (by ``req_id``), copies
        token counts onto the request record, and -- when no more
        generates are inflight or pending -- releases the slot and
        flips the published state ``running -> up``.  Mirrors the
        slot-release / state-flip behaviour of the legacy
        ``_start_generate_waiter`` exit branches.

        ``_pending_count`` is read here under ``gen_lock``; both
        ``_send_cmd_with_ack`` and ``_generate_sync`` Phase 2 enqueue
        new sends under the same lock, so the listener cannot race
        them and prematurely release a slot that a peer is about to
        repopulate.
        """
        entry = Orchestrator._registry.get(model_id)
        if entry is None:
            return
        inst = entry.get("instance")
        if inst is None:
            return
        gen_lock = entry["_gen_lock"]

        completed_rid = info.get("req_id")
        with gen_lock:
            inflight = Orchestrator._inflight.get(model_id, [])
            matched = None
            if completed_rid is not None:
                for i, (rid, q_rec, done_event) in enumerate(inflight):
                    if rid == completed_rid:
                        matched = (i, rid, q_rec, done_event)
                        break
            if matched is None and inflight:
                matched = (0, inflight[0][0], inflight[0][1],
                           inflight[0][2])

            if matched:
                i, rid, q_rec, done_event = matched
                inflight.pop(i)
                t_done = time.perf_counter()
                if error is not None:
                    # Worker-side fail-fast (typically the
                    # ``_dormant`` defense-in-depth branch in
                    # ``vllm_child._submit_generate``).  Mark the
                    # request as errored so the dashboard surfaces
                    # it and ``_generate_sync`` Phase 3's
                    # ``done_event.wait()`` returns immediately;
                    # the slot release / state flip below still
                    # runs so the model doesn't leak the slot on
                    # the failed generate.
                    q_rec["state"] = "error"
                    q_rec["t_done"] = t_done
                    q_rec["error"] = repr(error)
                    log.error("%s: generate FAILED  (%.1fs)  %s",
                              model_id, elapsed, error)
                else:
                    q_rec["state"] = "done"
                    q_rec["t_done"] = t_done

                    gen_result = inst.generate_results.get(rid)
                    if gen_result is not None:
                        q_rec["prompt_tokens"] = gen_result["prompt_tokens"]
                        q_rec["completion_tokens"] = gen_result["completion_tokens"]
                        q_rec["ttft_s"] = gen_result.get("ttft_s")
                        q_rec["tpot_ms"] = gen_result.get("tpot_ms")
                    else:
                        q_rec["prompt_tokens"] = inst.last_prompt_tokens
                        q_rec["completion_tokens"] = inst.last_completion_tokens
                        q_rec["ttft_s"] = info.get("ttft_s")
                        q_rec["tpot_ms"] = info.get("tpot_ms")

                    log.info("%s: generate done  (%.1fs)",
                             model_id, elapsed)
                done_event.set()

            if not inflight and inst._pending_count == 0:
                # Take the model's _lock so the slot release + state
                # flip is atomic from the perspective of other models'
                # Phase 2 scans (which would otherwise observe a torn
                # ``slot=None, state=running`` snapshot and drop this
                # model from HBM accounting).  Lock order matches the
                # rest of the orchestrator: gen_lock -> _lock.
                with Orchestrator._locks_ordered(model_id):
                    if entry.get("slot") is not None:
                        Slots.deallocate(entry["slot"])
                        entry["slot"] = None
                    if entry.get("state") == "running":
                        Orchestrator._set_state(model_id, "up")

    @staticmethod
    def _fail_all_inflight(model_id: str, exc: BaseException) -> None:
        """Fail every still-parked generate for *model_id* with *exc*.

        Called from the pipeline worker when an op raises
        :class:`WorkerCmdFailed` -- the engine is wedged from the
        child-side cmd failure, the user's
        ``PendingRequest.done_event`` would otherwise never fire, and
        the calling thread (in :meth:`generate`) would hang forever.

        Releases the slot iff this was the last in-flight generate
        and the model is currently in ``running`` state, mirroring
        the slot-release / state-flip tail of
        :meth:`_on_generate_done` so the failed model doesn't leak
        the slot.
        """
        entry = Orchestrator._registry.get(model_id)
        if entry is None:
            return
        gen_lock = entry["_gen_lock"]
        t_done = time.perf_counter()
        with gen_lock:
            inflight = Orchestrator._inflight.get(model_id, [])
            drained = list(inflight)
            inflight.clear()
            for _rid, q_rec, _done in drained:
                if q_rec.get("t_done") is None:
                    q_rec["state"] = "error"
                    q_rec["t_done"] = t_done
                    q_rec["error"] = repr(exc)
                    # Close out any open pause window on the errored
                    # rec: accumulate the live slice into ``paused_s``
                    # and clear ``t_pause_started``.  Otherwise
                    # ``state_server`` keeps adding ``now -
                    # t_pause_started`` to the reported ``paused_s``
                    # forever -- the rec can never be resumed
                    # (it's errored), so no ``ResumeOp`` will close
                    # the window for us.  Observed shape: req 421 /
                    # model 16 at 22:33:13 on 2026-05-19 reporting
                    # ``paused_s`` growing past its actual lifetime.
                    t_pause_started = q_rec.get("t_pause_started")
                    if t_pause_started is not None:
                        q_rec["paused_s"] = (
                            (q_rec.get("paused_s") or 0.0)
                            + max(0.0, t_done - t_pause_started))
                        q_rec["t_pause_started"] = None
            inst = entry.get("instance")
            release_slot = (
                not inflight
                and inst is not None
                and getattr(inst, "_pending_count", 0) == 0
            )
        for _rid, _q_rec, done_event in drained:
            done_event.set()
        if release_slot:
            with Orchestrator._locks_ordered(model_id):
                if entry.get("slot") is not None:
                    Slots.deallocate(entry["slot"])
                    entry["slot"] = None
                if entry.get("state") == "running":
                    Orchestrator._set_state(model_id, "up")
        if drained:
            log.error("%s: failed %d in-flight generate(s) (%s)",
                      model_id, len(drained), exc)

    # ------------------------------------------------------------------
    # pause / resume
    # ------------------------------------------------------------------

    @staticmethod
    def _send_cmd_with_ack(model_id: str, cmd: str, *args, **kwargs) -> None:
        """Send a non-``generate`` command to the child and wait for
        that specific cmd's ack.  Raises ``RuntimeError`` if the
        child reported a failure -- the caller MUST NOT march the
        state machine forward on a failed cmd.

        Mechanism: install a fresh ``_CmdAck`` at the tail of a
        per-``(model_id, cmd)`` FIFO under ``_cmd_ack_lock``, send
        the cmd under ``gen_lock``, then wait on the slot.  The
        demuxer's catch-all listener pops the head of the same FIFO
        for each non-generate ack, stamps the error on it, and sets
        it.

        FIFO (rather than a single dict slot) is what tolerates
        concurrent senders on the same cmd: e.g. a peer's
        Phase-2 eviction sleep racing with the same model's own
        ``_acquire_slot_for_running`` retreat each install their own
        slot and each get woken in send order, where a single-slot
        dict would have lost the first one when the second arrived.
        """
        entry = Orchestrator._registry[model_id]
        inst = entry["instance"]
        gen_lock = entry["_gen_lock"]
        method = getattr(inst, cmd)

        ev = _CmdAck()
        with gen_lock:
            with Orchestrator._cmd_ack_lock:
                q = Orchestrator._cmd_ack_events.setdefault(
                    (model_id, cmd), collections.deque())
                q.append(ev)
            # Enqueue the command under gen_lock so the demuxer's
            # generate listener (also under gen_lock when checking
            # ``_pending_count`` for slot-release) sees the bumped
            # count and doesn't release the slot between our send
            # and the ack arrival.
            method(*args, **kwargs)

        ev.wait()
        if ev.error is not None:
            raise WorkerCmdFailed(model_id, cmd, ev.error)

    @staticmethod
    def pause(model_id: str) -> None:
        """Pause an actively-generating model.

        Pre-condition: ``state == "running"`` and not already paused;
        otherwise a no-op (logs and returns).

        Effect: snapshots in-flight requests in the child and aborts
        them in the engine (``Instance.pause``), then deallocates the
        slot and flips the published state ``running -> up`` (slotless),
        with ``entry["paused"] = True``.  Pending ``generate_done``
        messages for the in-flight requests are deferred until
        :meth:`resume` re-prefills them.

        Non-blocking, but **deliberately unchained**: ``pause`` is an
        interrupt, not a successor.  It does not wait on any prior
        op's completion -- it submits ``PauseOp`` at the head of the
        pipeline FIFO and trips ``InterruptFlag`` so an in-flight op
        bails at its next yield-point.  See ``PauseOp`` docstring in
        ``pipeline.py`` for the full sequence.
        """
        entry = Orchestrator._registry.get(model_id)
        if entry is None:
            log.warning("pause(%r) skipped -- model not registered", model_id)
            return
        log.info("%s: pause received", model_id)

        pipe = Orchestrator._pipelines.get(model_id)
        if pipe is None:
            log.warning("%s: no pipeline registered; pause skipped", model_id)
            return
        # Pre-checks: skip the submit when this pause would be a
        # no-op (already paused, not running).
        if entry.get("paused"):
            log.info("%s: already paused, skipping", model_id)
            return
        if entry.get("state") != "running":
            log.info("%s: not running (state=%s), pause is a no-op",
                     model_id, entry.get("state"))
            return
        # Trip flag synchronously, then submit_front.  Order matters:
        # setting the flag before queueing PauseOp gives any in-flight
        # op a chance to bail at its next yield-point BEFORE PauseOp
        # dequeues, so the worker reaches PauseOp promptly even if the
        # in-flight op has a long tail.
        pipe.interrupt_now("pause")
        pipe.submit_front(PauseOp())

    @staticmethod
    def resume(model_id: str) -> None:
        """Resume a paused model.  Generate-shaped: no-op when there
        is nothing to drive, otherwise behaves like ``generate``,
        walking the ladder up first if the user parked the paused
        model below ``up``.

        Pre-condition: ``entry["paused"]`` is True and ``state`` in
        ``("up", "sleep", "checkpoint")``; otherwise a no-op (logs
        and returns).  ``running`` and ``saved`` cannot coexist with
        ``paused=True`` -- pause sets state=up, and ``move(saved)``
        explicitly refuses while paused -- so reaching the no-op
        branch for those means an invariant violation; logged as a
        warning.

        Behaviour, by case:

        * **Nothing to drive** (``_inflight[mid]`` is empty):
          observably a no-op.  Clears ``entry["paused"]`` and
          leaves the model wherever the user parked it (no walk-up
          cost when there is nothing to drive).  No slot
          acquisition, no worker cmd.

        * **Something to drive** (``_inflight[mid]`` has either
          saved-from-pause subreqs, queued-during-pause new
          generates, or both):

          1. Reuse :meth:`_generate_sync` Phase 1's exact entry
             point -- ``_move_sync(model_id, "up",
             announce_state="running")`` -- to walk the ladder
             (when the user parked the paused model below ``up``)
             *and* acquire the running slot via the same Tier
             A/B/C path.  Critically, this inherits the
             ``_move_sync`` self-heal: when ``_acquire_slot_for_running``
             Tier A succeeds but a peer's concurrent Phase-2
             eviction steals the slot back in the same instant
             (the original wedge -- see commit history), the
             post-acquire state re-check observes ``state="sleep"``
             and falls through to ``_step_up(sleep, up,
             announce_state="running")``, which re-acquires a
             slot via the standard sleep -> up path.  Without this
             re-check the model would be left publishing ``running``
             over a slotless, sleeping engine.

             ``paused=True`` is preserved across the walk (only
             this method clears it), so saved subreqs and queued
             requests in ``_inflight`` ride the walk untouched.

          2. Send the worker ``resume`` cmd to re-prefill the
             saved sub-requests and unfreeze the engine, then
             clear ``entry["paused"]``.  State is already
             ``running`` from step 1 so we don't flip it again;
             the brief ``state=running, paused=True`` window
             (bounded by the ``resume`` cmd round-trip) is a
             non-issue for readers: ``_step_up`` Phase 2 eviction
             only targets ``state=="up"`` slotless squatters,
             ``pause``/``resume``/``move`` all serialise via
             ``_futures[mid]``, and ``_generate_sync``'s paused
             fast-path tolerates either ordering.

          The original :meth:`generate` futures resolve as the
          engine drives the now-running requests to completion.

        ``resume`` is a successor: it depends on the world being in
        ``paused`` state with a settled ``_inflight`` ledger.  The
        pipeline FIFO orders this naturally -- ``ResumeOp`` lands
        behind any prior ``PauseOp`` / ``MoveOp`` on the same
        pipeline.  See ``ResumeOp`` docstring in ``pipeline.py``.
        """
        entry = Orchestrator._registry.get(model_id)
        if entry is None:
            log.warning("resume(%r) skipped -- model not registered", model_id)
            return
        log.info("%s: resume received", model_id)

        pipe = Orchestrator._pipelines.get(model_id)
        if pipe is None:
            log.warning("%s: no pipeline registered; resume skipped", model_id)
            return
        pipe.submit(ResumeOp())

    # ------------------------------------------------------------------
    # remove
    # ------------------------------------------------------------------

    @staticmethod
    def remove(model_id: str) -> None:
        """Delete a model's image and remove it from the registry.

        Auto-transitions to **saved** if needed.  For multi-model
        teardown, fan out via :meth:`models`::

            for mid in Orchestrator.models():
                Orchestrator.remove(mid)
        """
        entry = Orchestrator._registry.get(model_id)
        if entry is None:
            log.warning("%s: not registered, skipping remove", model_id)
            return
        # ``remove`` walks down to ``saved``, which would tear down the
        # live process and orphan every per-request future parked at
        # ``done_event.wait()``.  Refuse on paused models for the same
        # reason ``move(target='saved')`` does -- caller must
        # ``resume`` and let the in-flight generates drain first.
        if entry.get("paused"):
            log.warning("%s: remove refused while paused; call resume() "
                        "(and let in-flight generates drain) before "
                        "removing", model_id)
            return
        log.info("%s: remove received", model_id)

        pipe = Orchestrator._pipelines.get(model_id)
        if pipe is None:
            log.warning("%s: no pipeline registered; remove is a no-op",
                        model_id)
            return
        op_fut = pipe.submit(RemoveOp())
        # Schedule pipeline tear-down on a background thread so the
        # user-visible ``remove`` call stays non-blocking.  After
        # RemoveOp resolves, drain + join the worker + pop from
        # ``_pipelines`` so a subsequent ``register(mid)`` for the
        # same model_id starts fresh.
        def _teardown(_op_fut: Future = op_fut,
                      _pipe: "ModelPipeline" = pipe,
                      _mid: str = model_id) -> None:
            try:
                _op_fut.result()
            except BaseException as exc:  # noqa: BLE001
                log.exception("%s: RemoveOp failed: %s", _mid, exc)
            # Drain any stray ops queued after RemoveOp (there
            # shouldn't be any -- but be defensive), then stop the
            # worker.
            _pipe.shutdown(drain=True, timeout=10.0)
            Orchestrator._pipelines.pop(_mid, None)

        threading.Thread(
            target=_teardown,
            name=f"remove-teardown[{model_id}]",
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # wait
    # ------------------------------------------------------------------

    @staticmethod
    def wait(model_id: str | None = None) -> None:
        """Block until pending pipeline ops + in-flight generates complete.

        With *model_id*, drains that model's pipeline (move / register /
        remove / pause / resume ops queued behind any other op) and
        waits for any in-flight generate user-futures for that model.

        Without arguments, drains every pipeline and every generate
        future.  Models registered *after* this call starts are not
        awaited; call again to drain a fresh wave.
        """
        if model_id is not None:
            log.info("wait  model_id=%s", model_id)
            t0 = time.perf_counter()
            pipe = Orchestrator._pipelines.get(model_id)
            if pipe is not None:
                pipe.drain()
            # Also wait for any in-flight generate user-futures (the
            # daemon-thread Phase 3 wait + result-collection path).
            for gen_fut in list(Orchestrator._generate_futures):
                if not gen_fut.done():
                    gen_fut.result()
            elapsed = time.perf_counter() - t0
            log.info("wait done  (%.1fs)", elapsed)
            return

        model_ids = list(Orchestrator._registry)
        log.info("wait  all  models=%d", len(model_ids))
        t0 = time.perf_counter()
        for mid in model_ids:
            pipe = Orchestrator._pipelines.get(mid)
            if pipe is not None:
                pipe.drain()
        for gen_fut in list(Orchestrator._generate_futures):
            if not gen_fut.done():
                gen_fut.result()
        elapsed = time.perf_counter() - t0
        log.info("wait done  (%.1fs)", elapsed)

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    @staticmethod
    def status() -> None:
        """Log GPUs and registered models with their states."""
        if not Orchestrator._gpu_ids:
            log.info("Orchestrator not initialized. Call Orchestrator.init() first.")
            return

        log.info("status: image_cache=%s", Orchestrator._image_cache)

        try:
            pynvml.nvmlInit()
            log.info("status: GPUs (%d):", len(Orchestrator._gpu_ids))
            for idx in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(idx)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                m = pynvml.nvmlDeviceGetMemoryInfo(h)
                used = m.used / (1 << 30)
                total = m.total / (1 << 30)
                free = total - used
                log.info("status:   GPU %d: %s  %.1f / %.1f GiB used  "
                         "(%.1f GiB free)",
                         idx, name, used, total, free)
        except Exception:
            log.info("status: GPUs: %s", Orchestrator._gpu_ids)

        if not Orchestrator._registry:
            log.info("status: no models registered")
            return

        pid_gpu_mib: dict[int, int] = {}
        try:
            pynvml.nvmlInit()
            _NVML_NOT_AVAILABLE = 0xFFFFFFFFFFFFFFFF
            for idx in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(idx)
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                except Exception:
                    continue
                for p in procs:
                    used = getattr(p, "usedGpuMemory", None)
                    if used is None or used == _NVML_NOT_AVAILABLE:
                        continue
                    pid_gpu_mib[p.pid] = pid_gpu_mib.get(p.pid, 0) + int(used // (1 << 20))
        except Exception:
            pass

        sorted_models = sorted(Orchestrator._registry.items(),
                               key=lambda item: item[1].get("pinned_cpu_bytes", 0),
                               reverse=True)
        max_id = max(len(mid) for mid in Orchestrator._registry)
        log.info("status: models (%d):", len(Orchestrator._registry))
        for model_id, entry in sorted_models:
            state = entry["state"]
            inst = entry["instance"]
            gpu = entry.get("gpu")
            gpu_str = f"  gpu={gpu}" if gpu is not None else ""
            pinned = entry.get("pinned_cpu_bytes", 0)
            pinned_str = f"  pinned_cpu={pinned / 2**30:.1f} GiB" if pinned and state != "saved" else ""
            gpu_mem_str = ""
            if inst is not None and inst.pid and inst.pid in pid_gpu_mib:
                gpu_mem_str = f"  memory={pid_gpu_mib[inst.pid] / 1024:.1f} GiB"
            saved_str = ""
            if state == "saved":
                saved_str = f"  image={entry.get('image_dir', '?')}"
            log.info("status:   %s  [%s]%s%s%s%s",
                     model_id.ljust(max_id), state,
                     gpu_str, pinned_str, gpu_mem_str, saved_str)
