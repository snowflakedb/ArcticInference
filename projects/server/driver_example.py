"""Multi-model example: dynamic GPU rebalancing with continuous generation.

1. Load model-a (gets all GPUs), start sending prompts in a loop.
2. Load model-b — model-a scales down, model-b comes up.
3. Both models serve concurrently.
4. Shut down model-a — model-b claims the freed GPUs and scales up.
5. Shut down model-b.

A dashboard line prints periodically showing replicas, throughput, and
total tokens served per model.
"""

import asyncio
from arctic_inference.server import Driver, ModelConfig

CONFIG = ModelConfig(
    model="Qwen/Qwen3-30B-A3B",
    tensor_parallel_size=1,
    quantization="fp8",
    max_model_len=4096,
)

PROMPTS = [
    "Explain the theory of general relativity in detail.",
    "Write a short story about a robot learning to paint.",
    "Describe how a compiler transforms source code into machine code.",
    "What are the major differences between TCP and UDP? Explain thoroughly.",
    "Explain the process of photosynthesis step by step.",
    "Write a detailed comparison of Python and Rust.",
    "Describe the history of the internet from ARPANET to today.",
    "Explain how neural networks learn, including backpropagation.",
]

SAMPLING_PARAMS = {"max_tokens": 4096, "temperature": 0.7}
CONCURRENCY = 64


class Stats:
    def __init__(self):
        self.total_tokens = 0
        self.total_requests = 0

    def record(self, tokens: int, requests: int):
        self.total_tokens += tokens
        self.total_requests += requests


# model_id -> Stats
all_stats: dict[str, Stats] = {}


async def _request_loop(driver: Driver, model_id: str, stop: asyncio.Event, stats: Stats, idx: int):
    """Single worker: keep submitting one prompt at a time until stopped."""
    n = len(PROMPTS)
    while not stop.is_set():
        try:
            results = await driver.sample(
                model_id=model_id,
                prompts=[PROMPTS[idx % n]],
                sampling_params=SAMPLING_PARAMS,
            )
            stats.record(len(results[0]["token_ids"]), 1)
        except (KeyError, RuntimeError):
            break


async def generate_loop(driver: Driver, model_id: str, stop: asyncio.Event):
    """Maintain CONCURRENCY in-flight requests until *stop* is set."""
    stats = all_stats.setdefault(model_id, Stats())
    tasks = [
        asyncio.create_task(_request_loop(driver, model_id, stop, stats, i))
        for i in range(CONCURRENCY)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)


W = 88

async def print_dashboard(driver: Driver, stop: asyncio.Event, interval: float = 10):
    """Periodically print a one-line-per-model dashboard."""
    while not stop.is_set():
        await asyncio.sleep(interval)
        status = await driver.status()
        lines = []
        for mid, info in status["models"].items():
            ready = sum(1 for s in info["replica_states"] if s == "ready")
            stats = all_stats.get(mid)
            n = info["num_replicas"]
            if stats:
                lines.append(
                    f"  {mid:12s}  replicas {ready}/{n}  │  "
                    f"total_tok {stats.total_tokens:>9,d}  │  "
                    f"reqs {stats.total_requests:>6,d}"
                )
            else:
                lines.append(f"  {mid:12s}  replicas {ready}/{n}  │  (waiting for traffic)")
        gpus = f"GPUs {status['allocated_gpus']}/{status['total_gpus']}"
        print(f"\n{'─'*W}")
        print(f"  {gpus}")
        for line in lines:
            print(line)
        print(f"{'─'*W}")


async def wait_ready(driver: Driver, model_id: str):
    """Poll until all replicas for *model_id* report 'ready'."""
    while True:
        status = await driver.status()
        info = status["models"].get(model_id)
        if info and all(s == "ready" for s in info["replica_states"]):
            return info["num_replicas"]
        await asyncio.sleep(2)


async def wait_scale_up(driver: Driver, model_id: str, prev_replicas: int):
    """Poll until *model_id* has more replicas than *prev_replicas* and all ready."""
    while True:
        status = await driver.status()
        info = status["models"][model_id]
        n = info["num_replicas"]
        ready = sum(1 for s in info["replica_states"] if s == "ready")
        if n > prev_replicas and ready == n:
            return n
        await asyncio.sleep(2)


async def main():
    driver = Driver()
    dash_stop = asyncio.Event()

    # ---- 1. Load model-a (gets all GPUs) ----
    print("=== 1. Init model-a (gets all GPUs) ===")
    result = await driver.init(CONFIG, model_id="model-a")
    n = await wait_ready(driver, "model-a")
    print(f"  model-a: {n} replicas ready")

    # Start background dashboard.
    dash_task = asyncio.create_task(print_dashboard(driver, dash_stop))

    # ---- 2. Start continuous generation on model-a ----
    print("\n=== 2. Start generating on model-a ===")
    stop_a = asyncio.Event()
    task_a = asyncio.create_task(generate_loop(driver, "model-a", stop_a))

    # Let model-a serve for a bit.
    await asyncio.sleep(30)

    # ---- 3. Load model-b (model-a scales down) ----
    print("\n=== 3. Init model-b (model-a scales down) ===")
    await driver.init(CONFIG, model_id="model-b")
    n = await wait_ready(driver, "model-b")
    print(f"  model-b: {n} replicas ready")

    # ---- 4. Start continuous generation on model-b ----
    print("\n=== 4. Start generating on model-b ===")
    stop_b = asyncio.Event()
    task_b = asyncio.create_task(generate_loop(driver, "model-b", stop_b))

    # Both models serve concurrently for a while.
    await asyncio.sleep(30)

    # ---- 5. Shutdown model-a → model-b reclaims GPUs ----
    print("\n=== 5. Shutdown model-a ===")
    stop_a.set()
    await task_a
    prev = (await driver.status())["models"]["model-b"]["num_replicas"]
    await driver.shutdown_model("model-a")
    del all_stats["model-a"]
    print("  model-a shut down, model-b scaling up in background...")

    n = await wait_scale_up(driver, "model-b", prev)
    print(f"  model-b scaled up: {prev} -> {n} replicas")

    # Let model-b serve with more replicas for a while.
    await asyncio.sleep(30)

    # ---- 6. Shutdown model-b ----
    print("\n=== 6. Shutdown model-b ===")
    stop_b.set()
    await task_b
    dash_stop.set()
    await dash_task
    await driver.shutdown_model("model-b")
    print("  Done.")


if __name__ == "__main__":
    asyncio.run(main())
