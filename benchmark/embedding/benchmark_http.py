#!/usr/bin/env python3

import argparse
import asyncio
import time
import sys
import aiohttp
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import gen_random_num


class HTTPEmbeddingBenchmark:
    """Benchmark for vLLM HTTP embedding endpoints."""

    def __init__(
        self,
        server_address: str,
        batch_sizes: List[int],
        num_requests: int,
        concurrency: int,
        prompt_length: int,
        distribution: "str",
        model_name: str,
        backend: str,
    ):
        self.server_address = server_address
        self.batch_sizes = batch_sizes
        self.num_requests = num_requests
        self.concurrency = concurrency
        self.prompt_length = prompt_length
        self.model_name = model_name
        self.backend = backend
        self.distribution = distribution
        self.prompts = self._generate_prompts(
            prompt_length, max(batch_sizes) * num_requests, distribution
        )
        if self.backend == "openai" or self.backend == "vllm":
            self.api_url = f"{server_address}/v1/embeddings"
        elif self.backend == "TEI":
            self.api_url = f"{server_address}/embed"
        else:
            raise ValueError(f"Invalid backend: {self.backend}")

    def _generate_prompts(
        self, length: int, count: int, distribution: "str"
    ) -> List[str]:
        """Generate random prompts of specified length."""

        # Generate prompt lengths according to the specified distribution
        prompt_lengths = gen_random_num(length, count, distribution)
        
        # Create prompts by repeating "hello " the specified number of times for each length
        prompts = [
            "hello " * (prompt_length - 4)
            for prompt_length in prompt_lengths
        ]

        return prompts

    async def _warmup(self, session: aiohttp.ClientSession) -> bool:
        """Run warmup requests to ensure the server is ready."""
        print("\nRunning warmup requests...")
        warmup_prompts = self._generate_prompts(self.prompt_length, 20, "fixed")
        
        for i in range(20):
            try:
                if self.backend == "openai" or self.backend == "vllm":
                    request_body = {
                        "input": [warmup_prompts[i]],
                        "model": self.model_name,
                    }
                elif self.backend == "TEI":
                    request_body = {
                        "inputs": [warmup_prompts[i]],
                    }

                async with session.post(
                    self.api_url,
                    json=request_body,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Warmup request {i} failed: {response.status}, {error_text}")
                        return False
            except Exception as e:
                print(f"Warmup request {i} failed: {e}")
                return False
        
        print("Warmup completed successfully")
        return True

    async def _embed_batch(
        self, session: aiohttp.ClientSession, batch_size: int, prompts: List[str]
    ) -> Tuple[float, float]:
        """Make a single embedding request with the given batch size."""
        start_time = time.time()

        # Prepare request body for OpenAI-style embeddings endpoint
        if self.backend == "openai" or self.backend == "vllm":
            request_body = {
                "input": prompts,
                "model": self.model_name,
            }
        elif self.backend == "TEI":
            request_body = {
                "inputs": prompts,
            }

        try:
            async with session.post(
                self.api_url,
                json=request_body,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error: {response.status}, {error_text}")
                    return 0, 0

                result = await response.json()

                if self.backend == "openai" or self.backend == "vllm":
                    # Check if the response contains the expected data
                    if "data" not in result or len(result["data"]) != batch_size:
                        print(
                            f"Expected {batch_size} embeddings, got {len(result.get('data', []))}"
                        )
                        return 0, 0
                elif self.backend == "TEI":
                    # Check if the response contains the expected data
                    if len(result) != batch_size:
                        print(f"Expected {batch_size} embeddings, got {len(result)}")
                        return 0, 0

                elapsed = time.time() - start_time
                # Estimate token count +2 because the tokenizer will add BOS and EOS to the prompt
                total_tokens = sum(len(prompt.split()) + 2 for prompt in prompts)

                return elapsed, total_tokens
        except Exception as e:
            print(f"Exception in request: {e}")
            return 0, 0

    async def _run_concurrent_requests(
        self, batch_size: int, session: aiohttp.ClientSession
    ) -> Tuple[float, float, float]:
        """Run concurrent embedding requests and measure throughput."""
        prompt_index = 0
        tasks = []

        for i in range(self.num_requests):
            curr_batch_size = gen_random_num(batch_size, 1, self.distribution)
            prompts = self.prompts[prompt_index : prompt_index + curr_batch_size]
            prompt_index += curr_batch_size

            tasks.append(self._embed_batch(session, curr_batch_size, prompts))

        # Run requests with limited concurrency
        semaphore = asyncio.Semaphore(self.concurrency)

        async def bounded_embed_batch(coro):
            async with semaphore:
                return await coro

        # Start timer for overall throughput measurement
        start_time = time.time()
        results = await asyncio.gather(*[bounded_embed_batch(task) for task in tasks])
        total_time = time.time() - start_time

        # Filter out failed requests and sum up tokens
        valid_results = [
            (elapsed, tokens) for elapsed, tokens in results if elapsed > 0
        ]
        if not valid_results:
            return 0, 0, 0

        # Calculate average latency and throughput
        total_elapsed = sum(elapsed for elapsed, _ in valid_results)
        avg_latency = total_elapsed / len(valid_results)
        total_tokens = sum(tokens for _, tokens in valid_results)
        throughput = total_tokens / total_time

        return avg_latency, throughput, len(valid_results) / self.num_requests * 100

    async def run_benchmark(self):
        """Run the benchmark with different batch sizes."""
        print(f"\nConnecting to server: {self.server_address}")
        print(f"Using backend: {self.backend}")
        print(f"Model: {self.model_name}")

        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Run warmup requests
            if not await self._warmup(session):
                print("Warmup failed, exiting benchmark")
                return

            # Run benchmark for each batch size
            print("\nRESULTS:")
            print("| Batch Size | Seq Length | Avg Latency (s) | Throughput (K tokens/s) | Success Rate |")
            print("|------------|------------|-----------------|------------------------|---------------|")

            for batch_size in self.batch_sizes:
                try:
                    (
                        avg_latency,
                        throughput,
                        success_rate,
                    ) = await self._run_concurrent_requests(batch_size, session)
                    if avg_latency > 0:
                        print(
                            f"| {batch_size:^10} | {self.prompt_length:^10} | {avg_latency:^15.4f} | {throughput / 1000:^22.2f} | {success_rate:^12.2f}% |"
                        )
                    else:
                        print(
                            f"| {batch_size:^10} | {self.prompt_length:^10} | {'N/A':^15} | {'N/A':^22} | {0:^12.2f}% |"
                        )
                except Exception as e:
                    print(f"| {batch_size:^10} | {self.prompt_length:^10} | Error: {str(e):<11} | {'N/A':^22} | {'N/A':^12} |")


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM HTTP embedding endpoints"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8000",
        help="Server address including protocol (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Snowflake/snowflake-arctic-embed-m-v1.5",
        help="Model name to use for embedding (default: Snowflake/snowflake-arctic-embed-m-v1.5)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,16,64",
        help="Comma-separated list of batch sizes to test",
    )
    parser.add_argument(
        "--requests", type=int, default=10, help="Number of requests per batch size"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum number of concurrent requests",
    )
    parser.add_argument(
        "--prompt-length", type=int, default=128, help="Length of prompts to encode"
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="fixed",
        choices=["uniform", "normal", "fixed"],
        help="Distribution of batch sizes (default: fixed)",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["openai", "vllm", "TEI"],
        help="Backend to use for embedding (default: vllm). Note that vllm is the same as openai.",
    )

    args = parser.parse_args()
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
    if not args.server.startswith("http"):
        args.server = f"http://{args.server}"

    # Run benchmark
    benchmark = HTTPEmbeddingBenchmark(
        server_address=args.server,
        batch_sizes=batch_sizes,
        num_requests=args.requests,
        concurrency=args.concurrency,
        prompt_length=args.prompt_length,
        model_name=args.model,
        distribution=args.distribution,
        backend=args.backend,
    )

    print("\nStarting benchmark...")
    print(f"Server: {args.server}")
    print(f"Model: {args.model}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Requests per batch size: {args.requests}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Prompt length: {args.prompt_length}")
    print(f"Distribution: {args.distribution}")
    print(f"Backend: {args.backend}")

    await benchmark.run_benchmark()
    print("\nBenchmark completed.")


if __name__ == "__main__":
    asyncio.run(main())
