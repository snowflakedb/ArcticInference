from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from arctic_inference.server.config import ModelConfig
from arctic_inference.server.scheduler import (
    RoutingFn,
    Scheduler,
    prefix_affinity_routing,
)
from arctic_inference.server.worker import InferenceWorker

logger = logging.getLogger("arctic_inference.server")


def ensure_ray() -> int:
    """Initialize Ray and return the total number of GPUs in the cluster."""
    ray.init(ignore_reinit_error=True, log_to_driver=True)
    nodes = [n for n in ray.nodes() if n["Alive"]]
    if not nodes:
        raise RuntimeError("No alive Ray nodes")
    total = sum(
        int(n["Resources"].get("GPU", torch.cuda.device_count()))
        for n in nodes
    )
    if total == 0:
        raise RuntimeError("No GPUs available in the Ray cluster")
    logger.info(f"{total} GPUs available across {len(nodes)} node(s)")
    return total


class ReplicaPool:
    """Manages a set of worker replicas for a single model.

    Owns the workers and an internal :class:`Scheduler` that handles
    request routing and concurrency control.

    Args:
        worker_cls: Ray actor class for inference workers.
    """

    def __init__(
        self,
        worker_cls=InferenceWorker,
        routing_fn: RoutingFn | None = None,
        enable_prefix_hash: bool | None = None,
    ) -> None:
        self._worker_cls = worker_cls
        _opt_out = os.environ.get(
            "ARCTIC_PREFIX_ROUTING_DISABLED", ""
        ).lower() in ("1", "true", "yes")
        if _opt_out:
            self._routing_fn = None
            self._enable_prefix_hash = False
        elif enable_prefix_hash is None and routing_fn is None:
            self._routing_fn = prefix_affinity_routing
            self._enable_prefix_hash = True
        else:
            self._routing_fn = routing_fn
            self._enable_prefix_hash = bool(enable_prefix_hash)
        self._config: ModelConfig | None = None
        self._model_id: str | None = None
        self._ray_num_gpus: float | int | None = None
        self._placement_groups: list[PlacementGroup] | None = None
        self._bundle_indices: list[int] | None = None
        self._workers: list[ray.actor.ActorHandle] = []
        self._scheduler: Scheduler | None = None
        self._lock = asyncio.Lock()
        self._stop_monitoring = False
        self._health_task: asyncio.Task | None = None
        # Background scale_up task scheduled by Driver._rebalance_up. Tracked
        # so shutdown() can cancel it before worker init finishes — otherwise
        # the task races with shutdown and tries to add_worker on a None
        # scheduler.
        self._scale_task: asyncio.Task | None = None
        self._updating_workers: set[int] = set()
        self._cached_weights_info: list[dict] | None = None
        self._cached_spec_weights_info: list[dict] | None = None
        self._sleeping = False

    @property
    def config(self) -> ModelConfig:
        if self._config is None:
            raise RuntimeError("ReplicaPool not initialized")
        return self._config

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def tp_size(self) -> int:
        return self.config.tensor_parallel_size

    @property
    def num_replicas(self) -> int:
        return len(self._workers)

    def _worker_ray_options(self, replica_idx: int | None = None) -> dict[str, Any]:
        """Build ``ray.remote().options()`` kwargs for a worker actor."""
        opts: dict[str, Any] = {"max_concurrency": 2048}
        # `placement_groups` may be length 1 (shared PG for every replica, the
        # legacy single-PG behavior) or length `num_replicas` (one PG per
        # replica, used to STRICT_PACK each TP group on a single node).  Modulo
        # indexing covers both.
        pg: PlacementGroup | None = None
        if self._placement_groups and replica_idx is not None:
            pg = self._placement_groups[replica_idx % len(self._placement_groups)]

        if (
            pg is not None
            and self._bundle_indices is not None
            and replica_idx is not None
            and replica_idx < len(self._bundle_indices)
        ):
            if self.tp_size > 1:
                opts["num_gpus"] = 0
            else:
                opts["num_gpus"] = self._ray_num_gpus
            # For TP>1 the env-var path (see `initialize`) uses
            # `bundle_indices[i] * tp + t` as the local bundles for the
            # vLLM TP workers.  The actor itself reserves num_gpus=0, so it
            # only needs to land on a valid bundle of `pg`; pinning it to the
            # start of the TP group keeps the actor on the same node as its
            # children.
            if self.tp_size > 1:
                actor_bundle = self._bundle_indices[replica_idx] * self.tp_size
            else:
                actor_bundle = self._bundle_indices[replica_idx]
            opts["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=actor_bundle,
            )
        else:
            opts["num_gpus"] = self._ray_num_gpus
        return opts

    def _check_model_id(self, model_id: str | None) -> None:
        if model_id is not None and self._model_id is not None and model_id != self._model_id:
            raise ValueError(
                f"model_id mismatch: got {model_id!r}, "
                f"expected {self._model_id!r}"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _make_scheduler(self, workers: list[ray.actor.ActorHandle]) -> Scheduler:
        kwargs: dict[str, Any] = {"workers": workers, "initial_concurrency": 64}
        if self._routing_fn is not None:
            kwargs["routing_fn"] = self._routing_fn
        if self._enable_prefix_hash:
            kwargs["enable_prefix_hash"] = True
        return Scheduler(**kwargs)

    async def initialize(
        self,
        config: ModelConfig,
        model_id: str | None = None,
        num_replicas: int | None = None,
        ray_num_gpus: float | int | None = None,
        placement_groups: list[PlacementGroup] | None = None,
        bundle_indices: list[int] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> int:
        """Set configuration, create worker actors, start scheduler.

        Args:
            config: Model configuration (defines TP size, model name, etc.).
            model_id: Ignored in single-model mode.
            num_replicas: Number of replicas. If ``None``, uses all
                available GPUs (``total_gpus // tensor_parallel_size``).
            ray_num_gpus: ``num_gpus`` passed to ``ray.remote().options()``.
                Defaults to ``tensor_parallel_size``. Use a fractional
                value to colocate workers on the same physical GPUs.
            placement_groups: Optional Ray placement groups for colocated
                scheduling.  Two layouts are supported:

                * length 1: ``placement_groups[0]`` is shared by every
                  replica (legacy single-PG behavior).
                * length ``num_replicas``: each replica uses its own PG,
                  intended for per-replica STRICT_PACK groups that pin a
                  whole TP group onto a single physical node.

                Must be paired with *bundle_indices* of length ``num_replicas``.
            bundle_indices: Per-replica bundle indices into each replica's
                resolved PG.  For TP>1 this acts as the TP-group index within
                that PG, so the vLLM TP workers occupy local bundles
                ``[bundle_indices[r]*tp .. *tp+tp-1]``.
            extra_env: Extra environment variables passed to workers on init.
                Used for TP>1 colocated mode (VLLM_RAY_PER_WORKER_GPUS, etc).

        Returns the number of workers created.
        """
        if self._config is not None:
            raise RuntimeError("Already initialized. Call shutdown() first.")

        self._config = config
        self._model_id = model_id
        self._extra_env = extra_env

        if (
            ray_num_gpus is not None
            and self.tp_size > 1
            and not placement_groups
        ):
            raise ValueError(
                f"ray_num_gpus={ray_num_gpus} is not supported with "
                f"tensor_parallel_size={self.tp_size}. "
                f"Fractional GPU colocation requires a placement group for TP>1."
            )

        self._ray_num_gpus = ray_num_gpus if ray_num_gpus is not None else self.tp_size
        self._placement_groups = placement_groups
        self._bundle_indices = bundle_indices

        if num_replicas is None:
            total_gpus = ensure_ray()
            num_replicas = total_gpus // self.tp_size
            if num_replicas == 0:
                raise RuntimeError(
                    f"Not enough GPUs: TP={self.tp_size} needs at least "
                    f"{self.tp_size} GPUs but only {total_gpus} available"
                )

        if bundle_indices is not None and len(bundle_indices) != num_replicas:
            raise ValueError(
                f"bundle_indices length ({len(bundle_indices)}) must match "
                f"num_replicas ({num_replicas})"
            )
        if placement_groups is not None and len(placement_groups) not in (1, num_replicas):
            raise ValueError(
                f"placement_groups length ({len(placement_groups)}) must be "
                f"either 1 (shared) or num_replicas ({num_replicas}) (per-replica)"
            )

        n = num_replicas
        logger.info(f"Creating {n} workers (TP={self.tp_size}, ray_num_gpus={self._ray_num_gpus})")

        self._workers = []
        for i in range(n):
            opts = self._worker_ray_options(replica_idx=i)
            self._workers.append(self._worker_cls.options(**opts).remote())

        engine_kwargs = self._config.to_engine_kwargs()
        base_env = self._config.extra_env or {}
        if self._extra_env:
            base_env.update(self._extra_env)

        init_tasks = []
        for i, w in enumerate(self._workers):
            env = dict(base_env)
            if (self.tp_size > 1
                    and self._bundle_indices is not None
                    and "VLLM_RAY_PER_WORKER_GPUS" in env):
                rid = self._bundle_indices[i]
                env["VLLM_RAY_BUNDLE_INDICES"] = ",".join(
                    str(rid * self.tp_size + t) for t in range(self.tp_size))
            # Ensure we start from offset base ports so that vllm workers do not
            # collide when attempting to obtain free ports. vLLM increments ports
            # until a free one is found, but this can lead to a race condition when
            # spinning up many replicas at the same time.
            base, stride = 30000, 100
            env["VLLM_PORT"] = str(base + i*stride)
            init_tasks.append(w.initialize.remote(engine_kwargs, env or None))
        await asyncio.gather(*init_tasks)

        self._scheduler = self._make_scheduler(self._workers)

        self._stop_monitoring = False
        # self._health_task = asyncio.create_task(self._monitor_health())

        logger.info(f"ReplicaPool ready: {n} workers")
        return n

    async def shutdown(self, model_id: str | None = None) -> None:
        self._check_model_id(model_id)
        self._stop_monitoring = True

        # Cancel any background scale_up before tearing down the scheduler;
        # otherwise the in-flight scale_up will finish loading a worker and
        # then try to add_worker on a None scheduler.
        if self._scale_task and not self._scale_task.done():
            self._scale_task.cancel()
            try:
                await self._scale_task
            except (asyncio.CancelledError, Exception):
                pass
        self._scale_task = None

        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        if self._scheduler:
            await self._scheduler.shutdown()
            self._scheduler = None

        for w in self._workers:
            try:
                ray.get(w.shutdown.remote(), timeout=10)
            except Exception:
                pass
        self._workers.clear()
        self._config = None
        self._model_id = None
        self._ray_num_gpus = None
        self._placement_groups = None
        self._bundle_indices = None
        self._cached_weights_info = None
        self._cached_spec_weights_info = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompts: str | list[int] | list[str | list[int]],
        sampling_params: dict[str, Any] | None = None,
        model_id: str | None = None,
        routing_key: str | list[str | None] | None = None,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        """Generate completions for one or more prompts.

        Args:
            prompts: A single prompt (``str`` or token-id ``list[int]``) or a
                batch of prompts (``list[str | list[int]]``).
            sampling_params: vLLM sampling parameters (dict or SamplingParams).
            model_id: Ignored in single-model mode.
            routing_key: Optional per-batch affinity key. A single string
                applies to every prompt in the batch; a list must match the
                batch size element-wise. Hashed by the scheduler and used in
                place of the prompt hash to pin same-keyed requests to the
                same worker (e.g. all turns of one multi-turn rollout).
            strict: When True, enforce hard pinning to the keyed worker even
                under load, instead of ringing to the next worker on overload.
        """
        self._check_model_id(model_id)
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")
        if self._sleeping:
            raise RuntimeError("Model is sleeping; call /wake_up first")
        params = sampling_params or {}

        def _single_key() -> str | None:
            if routing_key is None or isinstance(routing_key, str):
                return routing_key
            if len(routing_key) != 1:
                raise ValueError(
                    f"Got {len(routing_key)} routing keys for a single prompt"
                )
            return routing_key[0]

        if isinstance(prompts, str):
            return [await self._scheduler.submit(
                prompts, params, routing_key=_single_key(), strict=strict,
            )]
        if prompts and isinstance(prompts[0], int):
            return [await self._scheduler.submit(
                prompts, params, routing_key=_single_key(), strict=strict,
            )]
        return await self._scheduler.submit_batch(
            prompts, params, routing_key=routing_key, strict=strict,
        )

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    async def scale_down(self, target_count: int) -> None:
        """Remove workers from the end until *target_count* remain."""
        if target_count < 0:
            raise ValueError(f"target_count must be >= 0, got {target_count}")
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")

        async with self._lock:
            while len(self._workers) > target_count:
                idx = len(self._workers) - 1
                self._scheduler.mark_worker_unavailable(idx)
                await self._scheduler.drain_worker(idx)
                worker = self._workers.pop()
                try:
                    ray.kill(worker)
                except Exception:
                    pass
                self._scheduler.remove_last_worker()
                logger.info(f"Scaled down: removed worker {idx}")

    async def scale_up(self, target_count: int) -> None:
        """Add workers until *target_count* are running.

        Safe to cancel mid-flight: any half-built actor is killed and the
        pool's worker list is left consistent.
        """
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")

        async with self._lock:
            engine_kwargs = self._config.to_engine_kwargs()
            extra_env = self._config.extra_env or None

            while len(self._workers) < target_count:
                idx = len(self._workers)
                opts = self._worker_ray_options(replica_idx=idx)
                worker = self._worker_cls.options(**opts).remote()
                try:
                    await worker.initialize.remote(engine_kwargs, extra_env)
                except asyncio.CancelledError:
                    try:
                        ray.kill(worker)
                    except Exception:
                        pass
                    raise
                except Exception:
                    try:
                        ray.kill(worker)
                    except Exception:
                        pass
                    raise

                # The pool may have been torn down while we awaited the
                # (slow) worker init. Don't touch the now-None scheduler;
                # discard the freshly-built worker.
                if self._scheduler is None:
                    try:
                        ray.kill(worker)
                    except Exception:
                        pass
                    logger.info(
                        "scale_up aborted: pool was shut down during worker init"
                    )
                    return


                self._workers.append(worker)
                self._scheduler.add_worker(worker)
                logger.info(f"Scaled up: added worker {idx}")

    # ------------------------------------------------------------------
    # Sleep / Wake
    # ------------------------------------------------------------------

    @property
    def sleeping(self) -> bool:
        return self._sleeping

    async def sleep(self, model_id: str | None = None, level: int = 1,
                    offload_weights: bool = False) -> dict[str, Any]:
        """Drain requests, then free GPU memory on every worker."""
        self._check_model_id(model_id)
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")
        if self._sleeping:
            return {"status": "already_sleeping", "level": level}

        async with self._lock:
            self._scheduler.pause()
            await self._scheduler.drain()

            results = await asyncio.gather(
                *[w.sleep.remote(level=level, offload_weights=offload_weights)
                  for w in self._workers],
                return_exceptions=True,
            )
            self._sleeping = True

            per_worker = [
                {"status": "error", "message": str(r)} if isinstance(r, Exception) else r
                for r in results
            ]
            logger.info("ReplicaPool sleeping (%d workers, level=%d, offload_weights=%s)",
                        len(self._workers), level, offload_weights)
            return {"status": "sleeping", "level": level, "workers": per_worker}

    @staticmethod
    def _wake_finalize(tags: list[str] | None) -> bool:
        """True when generate() may run after this wake (KV cache restored)."""
        if tags is None:
            return True
        return "kv_cache" in tags

    @staticmethod
    def _check_wake_worker_results(results: list, finalize: bool) -> None:
        errors = [r for r in results if isinstance(r, Exception)]
        if not errors:
            return
        msg = f"wake_up failed on {len(errors)} worker(s): {errors[0]}"
        if finalize:
            raise RuntimeError(msg)
        logger.error(msg)

    async def wake_up(self, model_id: str | None = None, tags: list[str] | None = None,
                      restore_weights: bool = False,
                      finalize: bool | None = None) -> dict[str, Any]:
        """Restore GPU memory on workers.

        For colocated weight sync, use ``tags=['weights']`` first (keeps the pool
        asleep and the scheduler paused), load weights, then ``tags=['kv_cache']``
        to allocate KV and resume scheduling.
        """
        self._check_model_id(model_id)
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")
        if finalize is None:
            finalize = self._wake_finalize(tags)
        if not self._sleeping:
            return {"status": "already_ready"}

        async with self._lock:
            results = await asyncio.gather(
                *[w.wake_up.remote(tags=tags, restore_weights=restore_weights)
                  for w in self._workers],
                return_exceptions=True,
            )
            self._check_wake_worker_results(results, finalize=finalize)

            per_worker = [
                {"status": "error", "message": str(r)} if isinstance(r, Exception) else r
                for r in results
            ]
            if finalize:
                self._sleeping = False
                self._scheduler.resume()
                logger.info(
                    "ReplicaPool awake (%d workers, tags=%s)",
                    len(self._workers),
                    tags,
                )
                return {"status": "ready", "tags": tags, "workers": per_worker}

            logger.info(
                "ReplicaPool weights stage (%d workers, tags=%s); KV wake pending",
                len(self._workers),
                tags,
            )
            return {"status": "weights_ready", "tags": tags, "workers": per_worker}

    async def reset_prefix_cache(self, model_id: str | None = None) -> dict[str, Any]:
        """Reset the prefix cache on all workers."""
        self._check_model_id(model_id)
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")
        async with self._lock:
            results = await asyncio.gather(*[w.reset_prefix_cache.remote() for w in self._workers], return_exceptions=True)
            # Clear group->worker affinity so each rollout starts with a fresh,
            # balanced assignment (no stale group pins / round-robin offset).
            self._scheduler.reset_affinity()
            logger.info("ReplicaPool reset prefix cache (%d workers)", len(self._workers))
            return {"status": "prefix_cache_reset"}

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_status(self) -> dict[str, Any]:
        if self._config is None:
            return {"status": "not_initialized"}
        states = await asyncio.gather(*[w.get_state.remote() for w in self._workers])
        return {
            "model": self._config.model,
            "num_replicas": len(self._workers),
            "sleeping": self._sleeping,
            "replica_states": list(states),
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def drain_metrics(self, model_id: str | None = None) -> dict[str, Any]:
        """Drain per-replica snapshots and per-request records.

        See :meth:`Scheduler.drain_metrics` for the payload shape; the
        ``model`` field is added at this level so consumers tagging metrics
        by ``model_id`` don't have to look it up separately.
        """
        self._check_model_id(model_id)
        if self._scheduler is None:
            return {
                "model": self._config.model if self._config else None,
                "drained_at": time.time(),
                "requests": [],
                "replicas": [],
            }
        payload = await self._scheduler.drain_metrics()
        payload["model"] = self._config.model if self._config else None
        payload["drained_at"] = time.time()
        return payload

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    async def compute_weight_norm(self, model_id: str | None = None) -> dict[str, Any]:
        """Global L2 norm of the live model weights on replica 0.

        All replicas hold identical weights, so one replica suffices. Used by
        tests to confirm a weight sync produced the expected weights.
        """
        self._check_model_id(model_id)
        if not self._workers:
            raise RuntimeError("ReplicaPool not initialized")
        return await self._workers[0].compute_weight_norm.remote()

    def get_weights_info(self, model_id: str | None = None) -> list[dict]:
        self._check_model_id(model_id)
        if self._config is None:
            raise RuntimeError("ReplicaPool not initialized")
        if self._cached_weights_info is None:
            from arctic_inference.server.weight_sync import build_weights_info
            infos = build_weights_info(self._config.model)
            self._cached_weights_info = [wi.to_dict() for wi in infos]
        return self._cached_weights_info

    async def sync_weights(
        self,
        groups: list[dict[str, Any]] | None = None,
        bucket_size: int = 256 * 1024 * 1024,
        strategy: str = "hotswap",
        engine_only: bool = False,
        direct_mode: bool = False,
        reverse: bool = False,
        model_id: str | None = None,
        master_addr: str | None = None,
        master_port: int | None = None,
        world_size: int | None = None,
    ) -> dict[str, Any]:
        """Receive weights from sender(s) and load into all replicas.

        Accepts either ``groups`` or legacy flat fields (``master_addr``,
        ``master_port``, ``world_size``).  The *strategy* controls how
        in-flight requests are handled:

          - **drain**: pause scheduler, wait for in-flight to finish, sync, resume.
          - **skip**: mark workers unavailable, cancel in-flight, sync, re-enable.
          - **hotswap**: sync while serving continues.
        """
        self._check_model_id(model_id)
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")

        if groups is None:
            if master_addr is None or master_port is None:
                raise ValueError(
                    "Provide either 'groups' or legacy flat fields "
                    "(master_addr, master_port, world_size)"
                )
            n = self.num_replicas
            tp = self.tp_size
            groups = [{
                "group_id": 0,
                "master_addr": master_addr,
                "master_port": master_port,
                "world_size": world_size or (1 + n * tp),
                "replica_ids": list(range(n)),
            }]

        async with self._lock:
            t0 = time.time()
            n = len(self._workers)

            if strategy == "drain":
                self._scheduler.pause()
                await self._scheduler.drain()
            elif strategy == "skip":
                for i in range(n):
                    self._scheduler.mark_worker_unavailable(i)
                for i in range(n):
                    self._scheduler.cancel_worker_inflight(i)
            elif strategy != "hotswap":
                raise ValueError(f"Unknown strategy: {strategy!r}. Use: drain, skip, hotswap")

            tp = self.tp_size
            replica_to_group: dict[int, dict[str, Any]] = {}
            for g in groups:
                base_port = g["master_port"]
                for rid in g["replica_ids"]:
                    replica_to_group[rid] = {
                        "master_addr": g["master_addr"],
                        "master_port": base_port + rid * tp,
                        "world_size": 2,
                        "rank_offset": 1,
                    }

            self._updating_workers = set(range(n))

            tasks = []
            for i, worker in enumerate(self._workers):
                gcfg = replica_to_group.get(i)
                if gcfg is None:
                    raise RuntimeError(
                        f"Replica {i} not assigned to any group. "
                        f"Groups cover replicas: {[g['replica_ids'] for g in groups]}"
                    )
                tasks.append(
                    worker.sync_weights.remote(
                        gcfg["master_addr"], gcfg["master_port"],
                        gcfg["rank_offset"], gcfg["world_size"],
                        bucket_size, engine_only, direct_mode, reverse,
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)
            self._updating_workers.clear()

            if strategy == "drain":
                self._scheduler.resume()
            elif strategy == "skip":
                for i in range(n):
                    self._scheduler.mark_worker_available(i)

            per_worker = [
                {"status": "error", "message": str(r)} if isinstance(r, Exception) else r
                for r in results
            ]
            elapsed = time.time() - t0
            logger.info(f"Weight sync: {n} workers, {len(groups)} group(s) in {elapsed:.2f}s")

            failed = [
                (i, w) for i, w in enumerate(per_worker)
                if not isinstance(w, dict) or w.get("status") != "done"
            ]
            if failed:
                raise RuntimeError(
                    f"Weight sync failed on {len(failed)}/{n} worker(s): "
                    + "; ".join(
                        f"worker[{i}]={w.get('message', w) if isinstance(w, dict) else w}"
                        for i, w in failed
                    )
                )

            return {
                "elapsed": elapsed,
                "num_groups": len(groups),
                "strategy": strategy,
                "strategy_elapsed": elapsed,
                "workers": per_worker,
            }

    async def sync_weights_ipc(
        self,
        group_id: int,
        replica_id: int,
        strategy: str = "hotswap",
    ) -> dict[str, Any]:
        """Receive weights via shared memory (colocated mode).

        Used when training and inference share a physical GPU, making
        NCCL communicator creation impossible.  The training worker writes
        weights to ``/dev/shm`` and the vLLM worker reads them back.
        """
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")

        async with self._lock:
            t0 = time.time()
            n = len(self._workers)

            if strategy == "drain":
                self._scheduler.pause()
                await self._scheduler.drain()
            elif strategy == "skip":
                for i in range(n):
                    self._scheduler.mark_worker_unavailable(i)
                for i in range(n):
                    self._scheduler.cancel_worker_inflight(i)

            worker = self._workers[replica_id]
            self._updating_workers = {replica_id}
            result = await worker.sync_weights_ipc.remote(group_id)
            self._updating_workers.clear()

            if strategy == "drain":
                self._scheduler.resume()
            elif strategy == "skip":
                for i in range(n):
                    self._scheduler.mark_worker_available(i)

            elapsed = time.time() - t0
            logger.info(
                f"Weight sync (IPC): replica {replica_id}, "
                f"group {group_id} in {elapsed:.2f}s"
            )
            return {
                "elapsed": elapsed,
                "strategy": strategy,
                "replica_id": replica_id,
                "worker_result": result if not isinstance(result, Exception) else str(result),
            }

    def get_spec_weights_info(self, model_id: str | None = None) -> list[dict]:
        """Return weight metadata for the spec (drafter) model."""
        self._check_model_id(model_id)
        if self._config is None:
            raise RuntimeError("ReplicaPool not initialized")
        spec_model = getattr(self._config, "speculative_model", None)
        if not spec_model:
            raise RuntimeError("No speculative_model configured")
        if self._cached_spec_weights_info is None:
            from arctic_inference.server.weight_sync import build_weights_info
            infos = build_weights_info(spec_model)
            self._cached_spec_weights_info = [wi.to_dict() for wi in infos]
        return self._cached_spec_weights_info

    async def sync_spec_weights(
        self,
        groups: list[dict[str, Any]] | None = None,
        bucket_size: int = 256 * 1024 * 1024,
        strategy: str = "hotswap",
        engine_only: bool = False,
        reverse: bool = False,
        model_id: str | None = None,
        master_addr: str | None = None,
        master_port: int | None = None,
        world_size: int | None = None,
    ) -> dict[str, Any]:
        """Receive spec (drafter) weights from sender(s) and load into all replicas.

        Same semantics as :meth:`sync_weights` but targets the drafter model.
        """
        self._check_model_id(model_id)
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")

        if groups is None:
            if master_addr is None or master_port is None:
                raise ValueError(
                    "Provide either 'groups' or legacy flat fields "
                    "(master_addr, master_port, world_size)"
                )
            n = self.num_replicas
            tp = self.tp_size
            groups = [{
                "group_id": 0,
                "master_addr": master_addr,
                "master_port": master_port,
                "world_size": world_size or (1 + n * tp),
                "replica_ids": list(range(n)),
            }]

        async with self._lock:
            t0 = time.time()
            n = len(self._workers)

            if strategy == "drain":
                self._scheduler.pause()
                await self._scheduler.drain()
            elif strategy == "skip":
                for i in range(n):
                    self._scheduler.mark_worker_unavailable(i)
                for i in range(n):
                    self._scheduler.cancel_worker_inflight(i)
            elif strategy != "hotswap":
                raise ValueError(
                    f"Unknown strategy: {strategy!r}. "
                    "Use: drain, skip, hotswap"
                )

            tp = self.tp_size
            replica_to_group: dict[int, dict[str, Any]] = {}
            for g in groups:
                base_port = g["master_port"]
                for rid in g["replica_ids"]:
                    replica_to_group[rid] = {
                        "master_addr": g["master_addr"],
                        "master_port": base_port + rid * tp,
                        "world_size": 2,
                        "rank_offset": 1,
                    }

            tasks = []
            for i, worker in enumerate(self._workers):
                gcfg = replica_to_group.get(i)
                if gcfg is None:
                    raise RuntimeError(
                        f"Replica {i} not assigned to any group. "
                        f"Groups cover replicas: "
                        f"{[g['replica_ids'] for g in groups]}"
                    )
                tasks.append(
                    worker.sync_spec_weights.remote(
                        gcfg["master_addr"], gcfg["master_port"],
                        gcfg["rank_offset"], gcfg["world_size"],
                        bucket_size, engine_only, reverse,
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            if strategy == "drain":
                self._scheduler.resume()
            elif strategy == "skip":
                for i in range(n):
                    self._scheduler.mark_worker_available(i)

            per_worker = [
                {"status": "error", "message": str(r)}
                if isinstance(r, Exception) else r
                for r in results
            ]
            elapsed = time.time() - t0
            logger.info(
                "Spec weight sync: %d workers, %d group(s) in %.2fs",
                n, len(groups), elapsed,
            )
            return {
                "elapsed": elapsed,
                "num_groups": len(groups),
                "strategy": strategy,
                "strategy_elapsed": elapsed,
                "workers": per_worker,
            }

    async def close_weight_sync(self, model_id: str | None = None) -> dict[str, Any]:
        self._check_model_id(model_id)
        async with self._lock:
            results = await asyncio.gather(*[w.close_weight_sync.remote() for w in self._workers])
            return {"status": "ok", "workers": list(results)}

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    async def _monitor_health(self) -> None:
        loop = asyncio.get_event_loop()
        while not self._stop_monitoring:
            await asyncio.sleep(30)
            for i, w in enumerate(self._workers):
                if i in self._updating_workers:
                    continue
                if self._scheduler is not None and not self._scheduler.is_worker_available(i):
                    continue
                try:
                    ref = w.is_healthy.remote()
                    healthy = await loop.run_in_executor(
                        None, lambda r=ref: ray.get(r, timeout=120)
                    )
                except Exception:
                    healthy = False
                if not healthy:
                    logger.warning(f"Worker {i} unhealthy, attempting restart")
                    await self._restart_worker(i)

    async def _restart_worker(self, idx: int) -> None:
        old = self._workers[idx]
        if self._scheduler is not None:
            self._scheduler.mark_worker_unavailable(idx)

        try:
            ray.kill(old)
        except Exception:
            pass

        opts = self._worker_ray_options(replica_idx=idx)
        new_worker = self._worker_cls.options(**opts).remote()

        engine_kwargs = self._config.to_engine_kwargs()
        extra_env = self._config.extra_env or None
        await new_worker.initialize.remote(engine_kwargs, extra_env)

        self._workers[idx] = new_worker

        if self._scheduler is not None:
            self._scheduler.update_worker_handle(idx, new_worker)
            self._scheduler.mark_worker_available(idx)

        logger.info(f"Worker {idx} restarted successfully")
