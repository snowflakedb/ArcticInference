#!/usr/bin/env python3

import argparse
import asyncio
import numpy as np
import time
import uuid
import grpc
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
import arctic_inference.embedding.proto.python.inference_pb2 as inference_pb2
import arctic_inference.embedding.proto.python.inference_pb2_grpc as inference_pb2_grpc


def gen_random_num(param: int, count: int, distribution: str) -> np.ndarray:
    """Generate random values based on the specified distribution.

    Args:
        param: The maximum value for the distribution parameter (max length or batch size)
        count: Number of values to generate
        distribution: Type of distribution - "uniform", "normal", or "fixed"

    Returns:
        A single integer if count=1, otherwise a numpy array of integers

    Raises:
        ValueError: If an invalid distribution is specified
    """
    if distribution == "uniform":
        # Generate values following uniform distribution between 1 and param
        values = np.random.randint(1, param + 1, count)
    elif distribution == "normal":
        # Generate values following normal distribution with mean at param/2
        # Standard deviation is param/4, clipped to range [1, param]
        values = np.random.normal(param / 2, param / 4, count)
        values = np.clip(values, 1, param).astype(int)
    elif distribution == "fixed":
        # Generate constant values equal to param
        values = np.full(count, param)
    else:
        raise ValueError(f"Invalid distribution: {distribution}")

    # Return a single value if count=1, otherwise the full array
    if count == 1:
        return values[0]
    return values


class EncodeBenchmark:
    """Benchmark for the Encode RPC method of embedding models.

    This class provides functionality to benchmark the performance of the Encode
    method with different batch sizes, distributions, and concurrency levels.
    """

    def __init__(
        self,
        server_address: str,
        batch_sizes: List[int],
        num_requests: int,
        concurrency: int,
        prompt_length: int,
        model_name: str,
        distribution: str,
    ):
        """Initialize the benchmark with the given parameters.

        Args:
            server_address: Address of the gRPC server to benchmark
            batch_sizes: List of batch sizes to test
            num_requests: Number of requests to send for each batch size
            concurrency: Maximum number of concurrent requests
            prompt_length: Maximum length of generated prompts
            model_name: Name of the model to use for encoding
            distribution: Distribution type for prompt lengths and batch sizes
        """
        self.server_address = server_address
        self.batch_sizes = batch_sizes
        self.num_requests = num_requests
        self.concurrency = concurrency
        self.prompt_length = prompt_length
        self.model_name = model_name
        self.distribution = distribution

        # Pre-generate all prompts needed for the benchmark
        self.prompts = self._generate_prompts(
            prompt_length, max(batch_sizes) * num_requests, distribution
        )

        # Create gRPC channel with appropriate buffer sizes
        self.channel = grpc.aio.insecure_channel(
            self.server_address,
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ],
        )
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)

    def _generate_prompts(
        self, length: int, count: int, distribution: str
    ) -> List[str]:
        """Generate random prompts with lengths following the specified distribution.

        Args:
            length: Maximum length of prompts
            count: Number of prompts to generate
            distribution: Distribution type for prompt lengths

        Returns:
            List of generated prompts
        """
        # Generate prompt lengths according to the specified distribution
        prompt_lengths = gen_random_num(length, count, distribution)

        # Create prompts by repeating "hello " the specified number of times for each length
        prompts = ["hello " * (prompt_length - 4) for prompt_length in prompt_lengths]

        return prompts

    async def wait_for_server_ready(self):
        while True:
            try:
                # Get server info
                info_response = await self.stub.GetReplicaInfo(
                    inference_pb2.ReplicaInfoRequest()
                )

                if info_response.n_healthy_replicas < info_response.n_replicas:
                    print(
                        "Waiting for replicas to be ready... {} / {}".format(
                            info_response.n_healthy_replicas, info_response.n_replicas
                        ),
                        end="\r",
                    )
                    await asyncio.sleep(2)
                else:
                    break
            except Exception as e:
                print(f"Failed to check server health: {e}")
                return False

    async def _warmup(self):
        """Run warmup requests to ensure the server is ready."""
        print("\nRunning warmup requests...")
        warmup_prompts = self._generate_prompts(self.prompt_length, 20, "fixed")

        for i in range(20):
            try:
                request_id = f"warmup-{i}-{uuid.uuid4()}"
                request = inference_pb2.EncodeRequest(
                    request_id=request_id,
                    n_prompts=1,
                    prompts=[warmup_prompts[i]],
                    priority=0,
                    model_name=self.model_name,
                )
                await self.stub.Encode(request)
            except Exception as e:
                print(f"Warmup request {i} failed: {e}")
                return False

        print("Warmup completed successfully")
        return True

    async def _encode_batch(
        self, batch_size: int, request_id: str, prompts: List[str]
    ) -> Tuple[float, int]:
        """Make a single encode request with the given batch size.

        Args:
            batch_size: Batch size for this request
            request_id: Unique ID for this request
            prompts: List of prompts to encode

        Returns:
            Tuple of (elapsed time, total tokens) if successful, (0, 0) if failed
        """
        start_time = time.time()

        # Prepare the request
        request = inference_pb2.EncodeRequest(
            request_id=request_id,
            n_prompts=batch_size,
            prompts=prompts,
            priority=0,
            model_name=self.model_name,
        )

        try:
            # Send the request and wait for response
            response = await self.stub.Encode(request)
            if response.error:
                print(f"Error in request {request_id}: {response.error}")
                return 0, 0

            # Verify response
            if len(response.embedding_bytes_fp32) != batch_size:
                print(
                    f"Expected {batch_size} embeddings, got {len(response.embedding_bytes_fp32)}"
                )
                return 0, 0

            # Calculate statistics
            elapsed = time.time() - start_time
            # Estimate token count, +2 because tokenizers typically add BOS and EOS tokens
            total_tokens = sum(len(prompt.split()) + 2 for prompt in prompts)

            return elapsed, total_tokens
        except Exception as e:
            print(f"Exception in request {request_id}: {e}")
            return 0, 0

    async def _run_concurrent_requests(
        self,
        batch_size: int,
        distribution: str,
    ) -> Tuple[float, float, float]:
        """Run concurrent encode requests and measure performance metrics.

        Args:
            batch_size: Batch size to use for requests
            distribution: Distribution for actual batch sizes

        Returns:
            Tuple of (average latency, throughput, success rate %)
        """
        prompt_index = 0
        tasks = []

        # Create tasks for all requests with appropriate batch sizes
        for i in range(self.num_requests):
            request_id = f"bench-{batch_size}-{i}-{uuid.uuid4()}"

            # Determine actual batch size based on distribution
            curr_batch_size = gen_random_num(batch_size, 1, distribution)

            # Get the prompts for this batch
            prompts = self.prompts[prompt_index : prompt_index + curr_batch_size]
            prompt_index += curr_batch_size

            # Create a task for this request
            tasks.append(self._encode_batch(curr_batch_size, request_id, prompts))

        # Set up concurrency control with semaphore
        semaphore = asyncio.Semaphore(self.concurrency)

        async def bounded_encode_batch(coro):
            """Wrapper to limit concurrency with semaphore."""
            async with semaphore:
                return await coro

        # Start timer for overall throughput measurement
        start_time = time.time()

        # Run all tasks concurrently with concurrency limit
        results = await asyncio.gather(*[bounded_encode_batch(task) for task in tasks])
        total_time = time.time() - start_time

        # Filter out failed requests
        valid_results = [
            (elapsed, tokens) for elapsed, tokens in results if elapsed > 0
        ]
        if not valid_results:
            return 0, 0, 0

        # Calculate performance metrics
        total_elapsed = sum(elapsed for elapsed, _ in valid_results)
        avg_latency = total_elapsed / len(valid_results)
        total_tokens = sum(tokens for _, tokens in valid_results)
        throughput = total_tokens / total_time
        success_rate = len(valid_results) / self.num_requests * 100

        return avg_latency, throughput, success_rate

    async def run_benchmark(self):
        """Run the benchmark with all configured batch sizes and collect results."""
        print(f"\nConnecting to server: {self.server_address}")

        # Run warmup requests
        if not await self._warmup():
            print("Warmup failed, exiting benchmark")
            return

        # Print table header for results
        print("\nRESULTS:")
        print(
            "| Batch Size | Seq Length | Avg Latency (s) | Throughput (K tokens/s) | Success Rate |"
        )
        print(
            "|------------|------------|-----------------|------------------------|---------------|"
        )

        # Run benchmark for each batch size
        for batch_size in self.batch_sizes:
            try:
                # Run the benchmark for this batch size
                (
                    avg_latency,
                    throughput,
                    success_rate,
                ) = await self._run_concurrent_requests(batch_size, self.distribution)

                # Print results
                if avg_latency > 0:
                    print(
                        f"| {batch_size:^10} | {self.prompt_length:^10} | {avg_latency:^15.4f} | {throughput / 1000:^22.2f} | {success_rate:^12.2f}% |"
                    )
                else:
                    print(
                        f"| {batch_size:^10} | {self.prompt_length:^10} | {'N/A':^15} | {'N/A':^22} | {0:^12.2f}% |"
                    )
            except Exception as e:
                print(
                    f"| {batch_size:^10} | {self.prompt_length:^10} | Error: {str(e):<11} | {'N/A':^22} | {'N/A':^12} |"
                )

        # Clean up
        await self.channel.close()


async def main():
    """Main entry point for the benchmark script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Benchmark the gRPC Encode method")
    parser.add_argument(
        "--server", type=str, default="localhost:50050", help="Server address"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Snowflake/snowflake-arctic-embed-m-v1.5",
        help="Model name",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,16,64",
        help="Comma-separated list of batch sizes to test",
    )
    parser.add_argument(
        "--requests", type=int, default=1024, help="Number of batches to run"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Maximum number of concurrent requests",
    )
    parser.add_argument(
        "--prompt-length", type=int, default=508, help="Length of prompts to encode"
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="fixed",
        choices=["uniform", "normal", "fixed"],
        help="Distribution of prompts to encode",
    )

    args = parser.parse_args()
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]

    # Validate model-specific constraints
    if args.model == "Snowflake/snowflake-arctic-embed-m-v1.5":
        if args.prompt_length > 512:
            print(
                "Prompt length must be less than 512 for Snowflake/snowflake-arctic-embed-m-v1.5 "
                "because the model has a max sequence length of 512"
            )
            return

    # Initialize and run benchmark
    benchmark = EncodeBenchmark(
        server_address=args.server,
        batch_sizes=batch_sizes,
        num_requests=args.requests,
        concurrency=args.concurrency,
        prompt_length=args.prompt_length,
        model_name=args.model,
        distribution=args.distribution,
    )

    await benchmark.wait_for_server_ready()

    # Print benchmark configuration
    print("\nStarting benchmark...")
    print(f"Server: {args.server}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Requests per batch size: {args.requests}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Prompt length: {args.prompt_length}")
    print(f"Distribution: {args.distribution}")

    # Run the benchmark
    await benchmark.run_benchmark()
    print("\nBenchmark completed.")


if __name__ == "__main__":
    asyncio.run(main())
