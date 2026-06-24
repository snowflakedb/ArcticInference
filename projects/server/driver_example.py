"""Multi-model example: load two models, generate, sleep/wake, generate again.

1. Load model-a and model-b.
2. Generate on both models.
3. Sleep both models (frees GPU memory).
4. Wake both models.
5. Generate again on both models.
6. Shut down.
"""

import asyncio

from arctic_inference.server import ModelConfig
from arctic_inference.server.multi_model import Driver

CONFIG = ModelConfig(
    model="Qwen/Qwen3-30B-A3B",
    tensor_parallel_size=1,
    quantization="fp8",
    max_model_len=4096,
)

PROMPTS = ["Explain photosynthesis.", "What is general relativity?"]
SAMPLING_PARAMS = {"max_tokens": 256, "temperature": 0.7}


async def generate_and_print(driver: Driver, model_id: str):
    results = await driver.generate(PROMPTS, SAMPLING_PARAMS, model_id=model_id)
    for i, r in enumerate(results):
        print(f"  [{model_id} #{i}] {len(r['token_ids'])} tokens: {r['text'][:80]}...")


async def main():
    driver = Driver()

    # 1. Load both models
    print("=== 1. Init models ===")
    await driver.initialize(CONFIG, model_id="model-a")
    await driver.initialize(CONFIG, model_id="model-b")
    status = await driver.get_status()
    for mid, info in status["models"].items():
        print(f"  {mid}: {info['num_replicas']} replicas")

    # 2. Generate on both
    print("\n=== 2. Generate ===")
    await generate_and_print(driver, "model-a")
    await generate_and_print(driver, "model-b")

    # 3. Sleep both models
    print("\n=== 3. Sleep ===")
    await driver.sleep("model-a")
    await driver.sleep("model-b")
    print("  Both models sleeping")

    # 4. Wake both models
    print("\n=== 4. Wake ===")
    await driver.wake_up("model-a")
    await driver.wake_up("model-b")
    print("  Both models awake")

    # 5. Generate again
    print("\n=== 5. Generate again ===")
    await generate_and_print(driver, "model-a")
    await generate_and_print(driver, "model-b")

    # 6. Shutdown
    print("\n=== 6. Shutdown ===")
    await driver.shutdown()
    print("  Done.")


if __name__ == "__main__":
    asyncio.run(main())
