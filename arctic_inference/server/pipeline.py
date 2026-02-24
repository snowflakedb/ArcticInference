"""High-level dataset processing on top of Driver/Scheduler."""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

import tqdm
from arctic_inference.server.scheduler import Scheduler

logger = logging.getLogger("arctic_inference.server")


class LLMHandle:
    """User-facing async interface for LLM calls within a pipeline task.

    Wraps the Scheduler with retry logic and a simple generate() API.
    """

    def __init__(self, scheduler: Scheduler) -> None:
        self._scheduler = scheduler

    async def generate(self, prompt: str, retries: int = 3, **kwargs: Any) -> str:
        """Generate text from a prompt. kwargs become vLLM SamplingParams."""
        return await self._call(prompt, "text", retries, kwargs)

    async def generate_token_ids(self, prompt_token_ids: list[int], retries: int = 3, **kwargs: Any) -> list[int]:
        """Generate from token IDs, returns token IDs."""
        return await self._call(prompt_token_ids, "token_ids", retries, kwargs)

    async def _call(self, prompt: str | list[int], key: str, retries: int, params: dict[str, Any]) -> Any:
        for attempt in range(retries + 1):
            try:
                future = self._scheduler.submit(prompt, params)
                result = await future
                return result[key]
            except Exception as e:
                if attempt == retries:
                    raise
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{retries + 1}): {e}")
                await asyncio.sleep(1)


class Pipeline:
    """Process a dataset through a user-defined async task with managed concurrency.

    Usage:
        driver = Driver()
        await driver.init(config)

        pipeline = Pipeline(driver)

        @pipeline.task
        async def process(sample, llm):
            return await llm.generate(prompt=sample["text"], max_tokens=100)

        results = await pipeline.run(dataset)
    """

    def __init__(self, driver: "Driver") -> None:  # noqa: F821
        self.driver = driver
        self._fn = None

    def task(self, func):
        """Decorator to register the per-sample processing function.

        Must be ``async def fn(sample, llm) -> result``.
        """
        if not inspect.iscoroutinefunction(func):
            raise ValueError("Pipeline task must be async")
        params = list(inspect.signature(func).parameters)
        if len(params) != 2:
            raise ValueError(f"Pipeline task must accept exactly 2 parameters (sample, llm), got {params}")
        self._fn = func
        return func

    async def run(
        self,
        dataset,
        max_retries: int | None = None,
        show_progress: bool = True,
    ) -> list[Any]:
        """Process *dataset* through the registered task.

        Returns a list of results in dataset order.  Failed tasks (after
        exhausting retries) are stored as ``{"__error__": "<message>"}``.
        """
        if self._fn is None:
            raise RuntimeError("No task registered — use @pipeline.task")
        if self.driver.scheduler is None:
            raise RuntimeError("Driver not initialized — call driver.init() first")

        llm = LLMHandle(self.driver.scheduler)
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
                concurrency_cap = self.driver.scheduler.total_concurrency_limit
                while len(tasks) < concurrency_cap and not done_submitting:
                    try:
                        idx = next(idx_iter)
                    except StopIteration:
                        done_submitting = True
                        break
                    t = asyncio.create_task(self._fn(dataset[idx], llm))
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
                        rt = asyncio.create_task(self._fn(dataset[idx], llm))
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
