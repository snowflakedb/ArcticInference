# Benchmarking Tools

This section covers two benchmarking tools available for ArcticInference.

## gRPC Benchmark Tool

The benchmark tool (`benchmark.py`) allows you to test the throughput and latency of the embedding functionality of the ArcticInference gRPC server.

### Usage

```bash
python benchmark.py [options]
```

#### Options

- `--server ADDRESS`: Server address in the format `host:port` (default: localhost:50050)
- `--model MODEL_NAME`: Model name to use for encoding (default: Snowflake/snowflake-arctic-embed-m-v1.5)
- `--batch-sizes SIZES`: Comma-separated list of batch sizes to test (default: 1,4,16,64)
- `--requests N`: Number of requests per batch size (default: 1024)
- `--concurrency N`: Maximum number of concurrent requests (default: 16)
- `--prompt-length N`: Length of prompts to encode (default: 512)
- `--distribution TYPE`: Distribution type for prompt lengths and batch sizes. Options are "fixed", "uniform" or "normal". Note that this applies to both prompt lengths and batch sizes. The default is fixed, which uses a fixed prompt length and batch size. "normal" uses a normal distribution with mean = prompt lengths (or batch sizes) / 2 and standard deviation = prompt lengths (or batch sizes) / 4. "uniform" uses a uniform distribution of [1, prompt lengths (or batch sizes)].

#### Example

```bash
# Test with default settings
python benchmark.py

# Connect to a specific server and test various batch sizes and concurrency
python benchmark.py --server 0.0.0.0:50050 --batch-sizes 1,4,16,64 --concurrency 64

# Specify a different model
python benchmark.py --model "Snowflake/snowflake-arctic-embed-m-v1.5"

# The command we use for benchmarking on H200
python benchmark.py --model "Snowflake/snowflake-arctic-embed-m-v1.5" \
    --server localhost:50050 \
    --batch-sizes 1,16,64 \
    --requests 10240 \
    --concurrency 1024 \
    --prompt-length 50

```

### Output

The benchmark will:
1. Check if the server is healthy
2. Display server info (model name, ready status)
3. Run benchmarks with different batch sizes
4. Report results showing average latency, throughput in tokens/sec, and success rate

#### Example Output

```
Server health: True
Server message: Service is healthy
Server model: llama2-7b
Ready: True

Starting benchmark...
Server: localhost:50050
Batch sizes: [1, 2, 4, 8, 16, 32]
Requests per batch size: 10
Concurrency: 4
Prompt length: 128

Connecting to server: localhost:50050

RESULTS:
Batch Size | Avg Latency (s) |  Throughput (K tokens/s)  |  Success Rate  
---------------------------------------------------------------------------
    1      |     0.0657      |          633.40           |     100.00     %
    4      |     0.0818      |          2010.16          |     100.00     %
    16     |     0.3172      |          2077.76          |     100.00     %
    64     |     1.2048      |          2189.42          |     100.00     %

Benchmark completed.
```

## HTTP Benchmark Tool

The HTTP benchmark tool (`benchmark_http.py`) allows you to test the throughput and latency of vLLM HTTP embedding endpoints that follow the OpenAI-compatible API format.

### Usage

```bash
python benchmark_http.py [options]
```

#### Options

- `--server ADDRESS`: Server address including protocol (default: http://localhost:8000)
- `--model MODEL_NAME`: Model name to use for embedding (default: Snowflake/snowflake-arctic-embed-m-v1.5)
- `--endpoint ENDPOINT`: API endpoint for embeddings (default: v1/embeddings)
- `--batch-sizes SIZES`: Comma-separated list of batch sizes to test (default: 1,4,16,64)
- `--requests N`: Number of requests per batch size (default: 1024)
- `--concurrency N`: Maximum number of concurrent requests (default: 16)
- `--prompt-length N`: Length of prompts to encode (default: 512)
- `--distribution TYPE`: Distribution type for prompt lengths ("uniform" or "normal", default is fixed)

#### Example

```bash
# Test with default settings
python benchmark_http.py

# Connect to a specific server
python benchmark_http.py --server http://localhost:8000

# Use uniform distribution for prompt lengths
python benchmark_http.py --distribution fixed

# Use normal distribution for prompt lengths
python benchmark_http.py --distribution normal

```

### Comparing arctic_inference vs vLLM Performance

To compare the performance of arctic_inference and vLLM:
First, start the vLLM server and the arctic_inference server:

```bash
# Start the vLLM server
vllm serve Snowflake/snowflake-arctic-embed-m-v1.5

# Start the arctic_inference server
python -m arctic_inference.embedding.replica_manager --model Snowflake/snowflake-arctic-embed-m-v1.5 --port 50050
```

```bash
# Run HTTP benchmark for vLLM
python benchmark_http.py --server http://localhost:8000 --batch-sizes 1,8,64 --requests 1280 --concurrency 256 --prompt-length 508 --distribution normal

# Run gRPC benchmark for arctic_inference
python benchmark.py --server localhost:50050 --batch-sizes 1,8,64 --requests 1280 --concurrency 256 --prompt-length 508 --distribution normal
```

Both benchmarks output comparable metrics for direct performance comparison.



## Notes on Performance Testing

- For optimal results, run benchmarks from a machine with sufficient resources and proximity to the server
- To identify performance bottlenecks, try varying batch sizes and concurrency settings
- Larger batch sizes often increase throughput but at the cost of higher latency
- Monitor server resource usage during benchmarks to identify bottlenecks 


