#!/usr/bin/env python3
"""Make _monitor_health truly robust by using thread pool for Ray calls."""
path = "arctic_inference/server/replica_pool.py"
with open(path, "r") as f:
    content = f.read()

old = """\
    async def _monitor_health(self) -> None:
        while not self._stop_monitoring:
            await asyncio.sleep(30)
            for i, w in enumerate(self._workers):
                if i in self._updating_workers:
                    continue
                if self._scheduler is not None and not self._scheduler.is_worker_available(i):
                    continue
                try:
                    healthy = await asyncio.wait_for(w.is_healthy.remote(), timeout=120)
                except Exception:
                    healthy = False
                if not healthy:
                    logger.warning(f"Worker {i} unhealthy, attempting restart")
                    await self._restart_worker(i)"""

new = """\
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
                    await self._restart_worker(i)"""

assert old in content, "Could not find patched _monitor_health"
content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
print("Patched: health check now uses thread pool executor (event-loop independent)")
