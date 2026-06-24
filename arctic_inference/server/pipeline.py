"""High-level dataset processing on top of ReplicaPool."""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, TYPE_CHECKING

import tqdm

if TYPE_CHECKING:
    from arctic_inference.server.replica_pool import ReplicaPool

logger = logging.getLogger("arctic_inference.server")


class LLMHandle:
    """User-facing async interface for LLM calls within a pipeline task.

    Wraps a ReplicaPool with retry logic and a simple generate() API.
    """

    def __init__(self, pool: ReplicaPool) -> None:
        self._pool = pool

    async def generate(self, prompt: str, retries: int = 3, **kwargs: Any) -> str:
        return await self._call(prompt, "text", retries, kwargs)

    async def generate_token_ids(self, prompt_token_ids: list[int], retries: int = 3, **kwargs: Any) -> list[int]:
        return await self._call(prompt_token_ids, "token_ids", retries, kwargs)

    async def _call(self, prompt: str | list[int], key: str, retries: int, params: dict[str, Any]) -> Any:
        for attempt in range(retries + 1):
            try:
                results = await self._pool.generate([prompt], params)
                return results[0][key]
            except Exception as e:
                if attempt == retries:
                    raise
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{retries + 1}): {e}")
                await asyncio.sleep(1)


class Pipeline:
    """Process a dataset through a user-defined async task with managed concurrency.

    Supports one or more models via ReplicaPool instances.

    Usage (single model):
        pipeline = Pipeline(pool)

        @pipeline.task
        async def process(sample, llm):
            return await llm.generate(prompt=sample["text"], max_tokens=100)

        results = await pipeline.run(dataset)

    Usage (multiple models):
        pipeline = Pipeline({"base": pool_a, "ft": pool_b})

        @pipeline.task
        async def process(sample, llms):
            base = await llms["base"].generate(prompt=sample["text"])
            ft = await llms["ft"].generate(prompt=sample["text"])
            return {"base": base, "ft": ft}

        results = await pipeline.run(dataset)
    """

    def __init__(self, pools: ReplicaPool | dict[str, ReplicaPool]) -> None:
        if isinstance(pools, dict):
            if not pools:
                raise ValueError("pools must not be empty")
            self._handles: LLMHandle | dict[str, LLMHandle] = {
                name: LLMHandle(pool)
                for name, pool in pools.items()
            }
        else:
            self._handles = LLMHandle(pools)
        self._fn = None

    def task(self, func):
        if not inspect.iscoroutinefunction(func):
            raise ValueError("Pipeline task must be async")
        params = list(inspect.signature(func).parameters)
        if len(params) != 2:
            raise ValueError(f"Pipeline task must accept exactly 2 parameters (sample, llm/llms), got {params}")
        self._fn = func
        return func

    async def run(
        self,
        dataset,
        max_retries: int | None = None,
        show_progress: bool = True,
    ) -> list[Any]:
        if self._fn is None:
            raise RuntimeError("No task registered — use @pipeline.task")

        handles = self._handles
        total = len(dataset)
        results: list[Any] = [None] * total

        pbar = tqdm.tqdm(total=total, desc="Processing", disable=not show_progress)
        task_map: dict[asyncio.Task, int] = {}
        retry_counts: dict[int, int] = {}
        tasks: list[asyncio.Task] = []
        idx_iter = iter(range(total))
        done_submitting = False

        try:
            while not done_submitting or tasks:
                while len(tasks) < 1024 and not done_submitting:
                    try:
                        idx = next(idx_iter)
                    except StopIteration:
                        done_submitting = True
                        break
                    t = asyncio.create_task(self._fn(dataset[idx], handles))
                    task_map[t] = idx
                    tasks.append(t)

                if not tasks:
                    break

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=1.0)
                tasks = list(pending)

                for t in done:
                    idx = task_map.pop(t)
                    try:
                        results[idx] = await t
                        pbar.update(1)
                    except Exception as e:
                        retry_counts[idx] = retry_counts.get(idx, 0) + 1
                        attempt = retry_counts[idx]

                        if max_retries is not None and attempt > max_retries:
                            logger.error(f"Task {idx} failed after {max_retries} retries: {e}")
                            results[idx] = {"__error__": str(e)}
                            pbar.update(1)
                            continue

                        logger.warning(f"Task {idx} failed (retry {attempt}): {e}")
                        rt = asyncio.create_task(self._fn(dataset[idx], handles))
                        task_map[rt] = idx
                        tasks.append(rt)
        finally:
            pbar.close()
            for t in tasks:
                if not t.done():
                    t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        return results
