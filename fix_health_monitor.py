#!/usr/bin/env python3
"""Patch _monitor_health to be robust against event loop starvation."""
path = "arctic_inference/server/replica_pool.py"
with open(path, "r") as f:
    content = f.read()

old = """\
    async def _monitor_health(self) -> None:
        while not self._stop_monitoring:
            await asyncio.sleep(10)
            for i, w in enumerate(self._workers):
                if i in self._updating_workers:
                    continue
                if self._scheduler is not None and not self._scheduler.is_worker_available(i):
                    continue
                try:
                    healthy = await asyncio.wait_for(w.is_healthy.remote(), timeout=30)
                except Exception:
                    healthy = False
                if not healthy:
                    logger.warning(f"Worker {i} unhealthy, attempting restart")
                    await self._restart_worker(i)"""

new = """\
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

assert old in content, "Could not find old _monitor_health"
content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
print("Patched: health check interval 10s→30s, timeout 30s→120s")
