"""Example: multi-model batch processing with the Pipeline API."""

import asyncio

from arctic_inference.server import ModelConfig, Pipeline, ReplicaPool, ensure_ray

CONFIG = ModelConfig(
    model="Qwen/Qwen3-30B-A3B",
    tensor_parallel_size=1,
    quantization="fp8",
    max_model_len=4096,
)


async def main():
    total_gpus = ensure_ray()
    gpus_per_model = total_gpus // 2

    pool_a = ReplicaPool()
    await pool_a.initialize(CONFIG, model_id="model-a", num_replicas=gpus_per_model)

    pool_b = ReplicaPool()
    await pool_b.initialize(CONFIG, model_id="model-b", num_replicas=gpus_per_model)

    pipeline = Pipeline({"a": pool_a, "b": pool_b})

    @pipeline.task
    async def compare(sample, llms):
        result_a = await llms["a"].generate(sample["text"], max_tokens=128, temperature=0.3)
        result_b = await llms["b"].generate(sample["text"], max_tokens=128, temperature=0.3)
        return {"model_a": result_a, "model_b": result_b}

    dataset = [
        {"text": "Summarize: Ray is a framework for scaling Python applications."},
        {"text": "Summarize: vLLM uses PagedAttention for efficient KV cache management."},
        {"text": "Summarize: FastAPI leverages type hints for automatic validation."},
    ]

    results = await pipeline.run(dataset, max_retries=2)

    for i, result in enumerate(results):
        print(f"[{i}] A: {result['model_a'][:60]}...")
        print(f"    B: {result['model_b'][:60]}...")

    await pool_a.shutdown()
    await pool_b.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
