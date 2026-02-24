"""Example: using the Driver Python API directly."""

import asyncio

from arctic_inference.server import Driver, ModelConfig


async def main():
    config = ModelConfig(
        model="Qwen/Qwen3-30B-A3B",
        tensor_parallel_size=1,
        quantization="fp8",
        max_model_len=4096,
    )

    driver = Driver()

    # Initialize workers
    result = await driver.init(config)
    print(f"Init: {result}")

    # Check status
    status = await driver.status()
    print(f"Status: {status}")

    # Generate from text prompts
    results = await driver.sample(
        prompts=["What is the capital of France?"],
        sampling_params={"max_tokens": 64, "temperature": 0.7},
    )
    for r in results:
        print(f"Response: {r['text']}")

    # Generate from token IDs
    results = await driver.sample(
        prompt_token_ids=[[1, 2, 3, 4, 5]],
        sampling_params={"max_tokens": 32},
    )
    for r in results:
        print(f"Token IDs: {r['token_ids']}")

    # Shutdown
    await driver.shutdown()
    print("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
