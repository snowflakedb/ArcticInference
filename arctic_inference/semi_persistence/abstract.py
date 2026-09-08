"""Abstract interface for a standalone vLLM instance.

Each Instance is a GPU-agnostic handle for a vLLM engine.  The GPU is
specified at init(gpu) time.  All primitives are non-blocking (except
wait) and return Self for chaining.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from slots import Slot

__all__ = [
    "OrchestratorBase",
    "InstanceBase",
    "OrchestratorClientBase",
    "SlotsBase",
]


class OrchestratorBase(ABC):
    """Abstract interface for an orchestrator that manages named model instances.

    State ladder:  saved <-> checkpoint <-> sleep <-> up -> running (transient)
    """

    @staticmethod
    @abstractmethod
    def init(image_cache: str = "/data-fast/image-cache",
             gpus: list[int] | None = None) -> None:
        """Discover GPUs, scan image cache, and populate registry."""

    @staticmethod
    @abstractmethod
    def models() -> list[str]:
        """Return the list of currently registered model_ids.

        Provided so callers can fan out per-model operations themselves;
        the orchestrator deliberately offers no built-in ``*_all``
        helpers (that concern lives one layer up, in
        :class:`OrchestratorClientBase`).
        """

    @staticmethod
    @abstractmethod
    def register(model_id: str, vllm_config: dict) -> None:
        """Cold-start a new model.  Ends in checkpoint state.

        *vllm_config* must be a dict; string shorthand
        (``"Qwen/Qwen3-32B"`` -> ``{"model": "Qwen/Qwen3-32B"}``) is
        a caller-side concern, handled by
        :class:`OrchestratorClientBase`, not the in-process API.

        Reserved keys in *vllm_config*:

        * ``_env`` -- optional ``dict[str, str]`` mapping environment
          variable names to values.  Popped in the vLLM child before
          ``LLM(**vllm_config)`` and applied to ``os.environ`` before
          ``from vllm import LLM``, so flags vLLM reads at import time
          take effect.  The trio
          ``CUDA_VISIBLE_DEVICES`` / ``VLLM_ENABLE_V1_MULTIPROCESSING``
          / ``USE_LIBUV`` is reserved by the child loop and silently
          dropped if present.  Persisted in ``meta.json`` alongside
          the rest of *vllm_config*, so it participates in
          client-side dedup and survives orch reboots.  CRIU-restored
          children inherit the dump-time env directly; ``_env`` is
          re-applied only on cold-start paths.
        """

    @staticmethod
    @abstractmethod
    def move(model_id: str, target: str,
             target_gpu: int | None = None) -> None:
        """Walk the state ladder to *target* (saved/checkpoint/sleep/up)."""

    @staticmethod
    @abstractmethod
    def generate(model_id: str, prompts: list[str] | str,
                 sampling_params: dict | int | None = None) -> object:
        """Run inference.  Auto-transitions to up if needed."""

    @staticmethod
    @abstractmethod
    def wait(model_id: str | None = None) -> None:
        """Block until pending operations complete.

        With *model_id*, waits only for that model's pending futures.
        Without arguments, acts as a concurrent barrier across every
        registered model: snapshots the set of pending futures and
        waits on each, so total wall time is ``max`` of the per-model
        waits (not the sum).  Models registered *after* the
        no-argument barrier starts are not awaited.
        """

    @staticmethod
    @abstractmethod
    def remove(model_id: str) -> None:
        """Auto-transition to saved, delete image, de-register *model_id*."""

    @staticmethod
    @abstractmethod
    def pause(model_id: str) -> None:
        """Pause an actively-generating model (``running -> up`` slotless).

        No-op unless *model_id* is currently in ``running`` state.
        Releases the slot, snapshots in-flight requests, and sets a
        paused flag on the registry entry.  Pending generate futures
        resolve only after :meth:`resume`.
        """

    @staticmethod
    @abstractmethod
    def resume(model_id: str) -> None:
        """Resume a paused model (``up -> running``).

        No-op unless *model_id* is in ``up`` state and paused.
        Re-acquires a slot via the same Tier A/B/C path as
        :meth:`generate`, re-prefills the saved sub-requests, and
        clears the paused flag.
        """

    @staticmethod
    @abstractmethod
    def status() -> None:
        """Print GPU view and registered models with states."""

    @staticmethod
    @abstractmethod
    def add(gpu: int) -> None:
        """Add *gpu* to the pool.

        Synchronous bookkeeping; new placements may immediately land on
        *gpu*.  Implementations should warn (not raise) if *gpu* is not
        visible to the platform's GPU enumeration so the call is usable
        in setups where device indices may differ from the live view.
        """

    @staticmethod
    @abstractmethod
    def sub(gpu: int) -> None:
        """Drain *gpu* from the pool (non-blocking).

        Marks *gpu* as draining so no new placements land on it, and
        submits a background drain that walks every resident to
        ``checkpoint`` (releasing slot + GPU residency) before removing
        *gpu* from the pool.  Awaitable via :meth:`wait_gpu`.
        """

    @staticmethod
    @abstractmethod
    def wait_gpu(gpu: int) -> None:
        """Block until the pending :meth:`sub` for *gpu* completes."""


class InstanceBase(ABC):
    """Abstract interface for a vLLM instance.

    Usage:
        instance = Instance(vllm_config)
        instance.init(gpu=0).attach().sleep().cuda_checkpoint().wait()
    """

    @abstractmethod
    def init(self, gpu: int) -> InstanceBase:
        """Cold start a model with random weights on the given GPU."""

    @abstractmethod
    def wait(self) -> InstanceBase:
        """Block until all pending commands complete for this instance.

        Raises RuntimeError on the first command that failed.
        """

    @abstractmethod
    def sleep(self) -> InstanceBase:
        """Free GPU memory for weights and KV cache (vLLM sleep level=2)."""

    @abstractmethod
    def cuda_checkpoint(self) -> InstanceBase:
        """Save CUDA state to CPU. Instance becomes stateless (gpu=None)."""

    @abstractmethod
    def criu_dump(self, filename: str | None = None) -> InstanceBase:
        """CRIU-dump the child process tree to disk (non-destructive).

        Uses --leave-running so the child stays alive after the dump.
        Writes meta.json with vllm_config and CRIU metadata.
        """

    @abstractmethod
    def criu_restore(self, filename: str | None = None) -> InstanceBase:
        """Restore a live process from a CRIU image on disk.

        Validates that the image's vllm_config matches this instance.
        Spawns a new worker and CRIU-restores the child.
        """

    @abstractmethod
    def cuda_restore(self, gpu: int) -> InstanceBase:
        """Restore checkpointed CUDA state onto the specified GPU."""

    @abstractmethod
    def attach(self) -> InstanceBase:
        """Allocate CPU memory for staging weights, on each vLLM worker.

        One buffer per worker, sized to that rank's parameters; pinning is
        the separate repin() step.
        """

    @abstractmethod
    def attach_pinned(self) -> InstanceBase:
        """Not supported; raises RuntimeError.  Use attach() -> repin().

        It allocated a permanently-pinned buffer (torch pin_memory=True)
        before staging moved onto the workers.
        """

    @abstractmethod
    def detach(self) -> InstanceBase:
        """Free pinned CPU memory."""

    @abstractmethod
    def stage(self, model_path: str | None = None) -> InstanceBase:
        """Load model weights from disk into pinned CPU buffer.

        Requires a prior attach().
        """

    @abstractmethod
    def wake_up_weights(self) -> InstanceBase:
        """Re-allocate weight tensors on GPU."""

    @abstractmethod
    def wake_up_kv_cache(self) -> InstanceBase:
        """Re-allocate KV cache on GPU."""

    @abstractmethod
    def plan_restore_weights(self) -> InstanceBase:
        """Precompute the chunk plan that the next ``restore_weights()`` consumes.

        Self-computes the staging budget from instance state populated
        by ``init`` (cold start) or by ``load`` reading ``meta.json``
        (restore):

            allotment = total_gpu_bytes * gpu_memory_utilization
            budget    = min(pinned_cpu_bytes, allotment - pinned_cpu_bytes)

        Splits the parameter index into chunks of ``<= budget`` bytes
        and caches the result on the worker.  Subsequent
        ``restore_weights()`` calls execute the cached plan.
        """

    @abstractmethod
    def restore_weights(self) -> InstanceBase:
        """Copy staged weights from pinned CPU into model parameters.

        Pure execution against the chunk plan cached by a prior
        ``plan_restore_weights()``.  For each chunk the worker copies
        a slice of the pinned buffer into a single reused GPU staging
        buffer (PCIe H2D) and then scatters into
        ``model.named_parameters()`` in place.  When no plan is cached,
        falls back to a single-chunk path identical to the pre-chunk
        behavior.  The staging buffer is freed before returning.
        """

    @abstractmethod
    def teardown(self) -> InstanceBase:
        """Tear down this instance and its worker. Resets to created state."""

    @abstractmethod
    def remove(self) -> type[InstanceBase]:
        """Deregister from the class-level instance registry.

        Non-blocking and non-destructive: does not affect the worker
        process or pending commands.  Returns the class so a chained
        ``status()`` call resolves to the classmethod view.
        """


class OrchestratorClientBase(ABC):
    """Abstract interface for a remote (HTTP) orchestrator client.

    The client layers a *jobs directory* over the orchestrator's
    single-model API: callers register/use *job_ids*, the client
    transparently maps them to orchestrator-side model_ids and
    deduplicates by ``vllm_config``.  Per-job methods (``generate``,
    ``wait``, ``remove``, ``pause``, ``resume``) accept ``job_id=None``
    to fan out client-side over every registered job; ``move`` is the
    exception and exposes a separate ``move_all`` for fan-out because
    its ``target`` arg is also required.
    """

    @classmethod
    @abstractmethod
    def init(cls, session: str | None = None, *,
             base_url: str = ...,
             timeout_s: float = 600.0) -> None:
        """Bind to *base_url* and print models available on the orchestrator.

        Concrete implementations should default *base_url* to match
        :mod:`orch_server`'s default port so a no-arg ``init()`` works
        out of the box for local development.

        Resets the local jobs directory (server-side state is left
        intact) and prints the available models for interactive use.
        Does not return anything; use :meth:`status` for a
        programmatic view (also via print).

        If *session* is provided, the JSON file at that path is
        treated as the on-disk mirror of the in-memory job
        directory: any existing entries are replayed via
        :meth:`register` (so :meth:`vllm_config` dedup against
        current server-side models applies), and every subsequent
        :meth:`register` / :meth:`remove` rewrites the file
        atomically.  Calling :meth:`init` without *session* (or with
        a different *session*) detaches from any previously-bound
        file.  One owner per file; no cross-process locking is
        performed.
        """

    @classmethod
    @abstractmethod
    def register(cls, job_id: str, vllm_config: dict | str) -> None:
        """Register *job_id* and bind it to a model on the orchestrator.

        Two forms for the second argument:

        * ``dict`` -- a vllm config; reuses an existing orchestrator
          model whose ``vllm_config`` matches, otherwise registers a
          fresh model server-side.
        * ``str`` -- an explicit orchestrator-side ``model_id``; binds
          *job_id* directly to that model.  Raises if the model is not
          registered on the orchestrator (no implicit registration).

        Reserved keys inside the dict form:

        * ``_env`` -- optional ``dict[str, str]`` of env vars applied
          to ``os.environ`` in the vLLM child before
          ``from vllm import LLM``, so import-time flags
          (e.g. ``VLLM_USE_DEEP_GEMM``) take effect.  Participates in
          ``vllm_config`` equality checks: two registrations with the
          same model but different ``_env`` deduplicate to *different*
          backing models.  The reserved trio
          ``CUDA_VISIBLE_DEVICES`` / ``VLLM_ENABLE_V1_MULTIPROCESSING``
          / ``USE_LIBUV`` is silently dropped.
        """

    @classmethod
    @abstractmethod
    def jobs(cls) -> None:
        """Print one row per registered job: model, state, gpu, flags.

        Focused view -- the per-job table only.  See :meth:`status`
        for the same table plus requests and pause-record footer.
        """

    @classmethod
    @abstractmethod
    def requests(cls) -> None:
        """Print one row per in-flight or completed request on the server.

        Focused view -- the per-request table only.  See
        :meth:`status` for the same table plus jobs and pause-record
        footer.
        """

    @classmethod
    @abstractmethod
    def paused(cls) -> None:
        """Print every paused job/model and the GPU each is paused on.

        Focused view of the orchestrator's ``paused`` flag, joined
        with the client-side ``_paused_gpu`` records so the caller
        can see which paused jobs will be picked up by a matching
        ``resume(N)`` (and which were paused via the explicit single-
        job form instead).
        """

    @classmethod
    @abstractmethod
    def model_of(cls, job_id: str) -> None:
        """Print the orchestrator model_id bound to *job_id*."""

    @classmethod
    @abstractmethod
    def move(cls, job_id: str, target: str,
             target_gpu: int | None = None) -> None:
        """Walk the model bound to *job_id* to *target* state."""

    @classmethod
    @abstractmethod
    def move_all(cls, target: str,
                 target_gpu: int | None = None) -> None:
        """Client-side fan-out of :meth:`move` over every registered job.

        :meth:`move` is the only per-job method that does not absorb
        its fan-out form via ``job_id=None``; *target* is also
        required, so a separate ``move_all`` keeps the positional
        ``move(job_id, target)`` signature ergonomic.
        """

    @classmethod
    @abstractmethod
    def generate(cls, job_id: str | int | None = None,
                 prompts: list[str] | str = ...,
                 sampling_params: dict | int | None = ...) -> None:
        """Submit a non-blocking generate against *job_id*.

        ``job_id=None`` (or omitted) fans out over every registered
        job.  Implementations should default *prompts* and
        *sampling_params* so a no-arg ``generate()`` is a usable
        smoke test.

        ``int`` in the *job_id* slot is a shorthand: ``generate(N)``
        means "fan out over every job with ``max_tokens=N``".
        Implementations should reject the shorthand when combined
        with explicit *prompts* / *sampling_params* args (it would
        otherwise silently override one of them).
        """

    @classmethod
    @abstractmethod
    def wait(cls, job_id: str | None = None) -> None:
        """Block until the bound model's pending futures complete.

        ``job_id=None`` (or omitted) acts as a client-side barrier
        across every registered job.
        """

    @classmethod
    @abstractmethod
    def remove(cls, job_id: str | None = None) -> None:
        """Unbind *job_id* from its backing model.

        Pure local operation; server-side models are never touched.
        Use a separate, explicit call against the orchestrator if you
        want to actually delete a model.  If a session file is bound
        (see the ``session`` arg of :meth:`init`), the file is updated
        to drop *job_id*.

        ``job_id=None`` (or omitted) drops every binding (and rewrites
        the bound session file as empty, if one is bound).
        """

    @classmethod
    @abstractmethod
    def pause(cls, job_id: str | int | None = None) -> None:
        """Pause the model bound to *job_id* (``running -> up`` slotless).

        ``job_id=None`` (or omitted) fans out over every registered
        job.  ``int`` in the *job_id* slot is a shorthand:
        ``pause(N)`` pauses every registered job whose backing model
        is currently resident on GPU *N*, and records the pairing so
        a later :meth:`resume` (``int``) can undo exactly that set
        without re-querying the server.
        """

    @classmethod
    @abstractmethod
    def resume(cls, job_id: str | int | None = None) -> None:
        """Resume the model bound to *job_id* (``up -> running``).

        ``job_id=None`` (or omitted) fans out over every registered
        job.  ``int`` in the *job_id* slot is a shorthand:
        ``resume(N)`` resumes every job that a previous
        ``pause(N)`` recorded against GPU *N* (pure client-side
        bookkeeping; no extra server round-trip).  Symmetric to
        :meth:`pause` (``int``).
        """

    @classmethod
    @abstractmethod
    def status(cls) -> None:
        """Pretty-print a job-centric view of the full server state.

        Composes :meth:`jobs` and :meth:`requests` plus a pause-record
        footer, all from a single ``GET /state`` so the view is
        internally consistent.
        """

    @classmethod
    @abstractmethod
    def add(cls, gpu: int) -> None:
        """Add *gpu* to the orchestrator's pool (synchronous on the server)."""

    @classmethod
    @abstractmethod
    def sub(cls, gpu: int) -> None:
        """Drain *gpu* from the orchestrator's pool (non-blocking)."""


class SlotsBase(ABC):
    """Abstract interface for a buddy-allocator over fractional GPU slots.

    A level-L slot covers ``1 / 2**(L-1)`` of a GPU.  Implementations
    are expected to be process-wide singletons exposing class-level
    methods (no instances are constructed).
    """

    @classmethod
    @abstractmethod
    def init(cls, gpu_ids: list[int]) -> None:
        """Initialise the pool with one whole-GPU root slot per id."""

    @classmethod
    @abstractmethod
    def remove(cls) -> None:
        """Tear down the pool.  Asserts no slots are still live."""

    @classmethod
    @abstractmethod
    def allocate(cls, level: int, gpu: int | None = None,
                 on_block: Callable[[], None] | None = None) -> "Slot":
        """Block until a level-*level* slot is available, then return it.

        If *gpu* is ``None``, the coldest GPU that can satisfy the
        request is chosen.  Strict head-of-line FIFO among waiters.
        *on_block*, if given, fires at most once when the call has to
        actually wait.
        """

    @classmethod
    @abstractmethod
    def try_allocate(cls, level: int,
                     gpu: int | None = None) -> "Slot | None":
        """Non-blocking allocate.  Returns ``None`` if not satisfiable now.

        Bypasses the FIFO waiter queue; intended for opportunistic
        fast-path acquisition.
        """

    @classmethod
    @abstractmethod
    def deallocate(cls, slot: "Slot") -> None:
        """Release *slot* and coalesce buddies upward."""

    @classmethod
    @abstractmethod
    def add(cls, gpu: int) -> bool:
        """Add *gpu* to the pool with a single whole-GPU root slot.

        Returns ``True`` if the GPU was added, ``False`` if it was
        already in the pool (idempotent).
        """

    @classmethod
    @abstractmethod
    def pop(cls, gpu: int) -> None:
        """Remove a fully-idle *gpu* from the pool.

        Must only be called when no live slot or non-root-free pool
        entry references *gpu*; the caller is responsible for marking
        the GPU draining and ensuring residents have moved off first.
        """

    @classmethod
    @abstractmethod
    def status(cls) -> None:
        """Print a tree-style view of every GPU's slot state."""
