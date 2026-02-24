"""Example: batch dataset processing with the Pipeline API."""

import asyncio

from arctic_inference.server import Driver, ModelConfig, Pipeline


async def main():
    config = ModelConfig(
        model="Qwen/Qwen3-30B-A3B",
        tensor_parallel_size=1,
        quantization="fp8",
        max_model_len=4096,
    )

    driver = Driver()
    await driver.init(config)

    pipeline = Pipeline(driver)

    @pipeline.task
    async def summarize(sample, llm):
        prompt = f"Summarize the following in one sentence:\n\n{sample['text']}"
        return await llm.generate(prompt, max_tokens=128, temperature=0.3)

    dataset = [
        {"text": "Ray is a framework for scaling Python applications. It provides simple APIs for distributed computing and is used for ML training, serving, and data processing."},
        {"text": "vLLM is a high-throughput serving engine for large language models. It uses PagedAttention to efficiently manage GPU memory for KV caches."},
        {"text": "FastAPI is a modern web framework for building APIs with Python. It leverages type hints for automatic validation and documentation."},
    ]

    results = await pipeline.run(dataset, max_retries=2)

    for i, result in enumerate(results):
        print(f"[{i}] {result}")

    await driver.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
