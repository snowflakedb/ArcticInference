# Arctic Inference gRPC Server

This directory contains a gRPC server and client implementation for the vLLM AsyncLLMEngine.

## Installation

First, ensure you have the required dependencies:

```bash
pip install grpcio grpcio-tools protobuf vllm
```

## Generating gRPC Code

Before using the server or client, you need to generate the gRPC code from the proto file. There are two ways to do this:

```bash
python arctic_inference/grpc/generate_proto.py
```

This will generate the following files:
- `inference_pb2.py`: Contains message classes
- `inference_pb2_grpc.py`: Contains server and client classes

## Running the Replica Server

To run the gRPC Replica server:

```bash
python -m arctic_inference.grpc.replica --model <MODEL_NAME> [--host <HOST>] [--port <PORT>] [--workers <WORKERS>] [--disable-log-request]
# e.g., 
python -m arctic_inference.grpc.replica --model BAAI/bge-base-en-v1.5 --host 127.0.0.1 --port 50000
```

Options:
- `--model`: Model name or path (required)
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 50051)
- `--workers`: Number of gRPC workers (default: 16)
- `--disable-log-request`: Disable logging of request (default: False)

You can also pass more vLLM engine arguments to the replica server.

# Replica Manager

The replica manager (`replica_manager.py`) enables horizontal scaling of vLLM inference by managing multiple replica instances and load balancing requests between them.

## Features

- **Multiple Model Replicas**: Launch and manage multiple vLLM model replicas
- **Load Balancing**: Distribute requests across replicas using various strategies:
  - Round Robin
  - Random
  - Least Loaded (based on active requests)
- **Health Monitoring**: Continuously monitor replica health and readiness
- **Automatic Recovery**: Retry requests if replicas are unavailable
- **Unified API**: Present a single API endpoint that handles distribution internally

## Usage

```bash
python -m arctic_inference.grpc.replica_manager [options]
```

### Options

- `--model MODEL_NAME`: Model name or path (required)
- `--host HOST`: Host to bind the manager to (default: 0.0.0.0)
- `--port PORT`: Port to bind the manager to (default: 50050)
- `--replica-host HOST`: Host for replicas to bind to (default: 127.0.0.1)
- `--start-port PORT`: Starting port number for replicas (default: 50100)
- `--num-replicas N`: Number of replicas to launch (default: 2)
- `--num-gpus N`: Number of GPUs available (default: 1)
- `--gpu-assignment STRATEGY`: GPU assignment strategy, either "dedicated" or "shared" (default: dedicated)
- `--tensor-parallel-size N`: Tensor parallel size for each replica (default: 1)
- `--gpu-memory-utilization FLOAT`: GPU memory utilization for each replica (default: 0.9)
- `--load-balancing POLICY`: Load balancing policy: "round_robin", "random", or "least_loaded" (default: round_robin)

### Example

```bash
# Start a manager with 3 replicas using a round-robin load balancing policy
python -m arctic_inference.grpc.replica_manager --model BAAI/bge-base-en-v1.5 --num-replicas 4

# Use a least-loaded policy for better handling of varying request complexities
python -m arctic_inference.grpc.replica_manager --model BAAI/bge-base-en-v1.5 --num-replicas 4 --load-balancing least_loaded
```

## Using the Client

To use the client:

```bash
python -m arctic_inference.grpc.client --prompt "Your prompt here" [--host <HOST>] [--port <PORT>] [--temperature <TEMP>] [--top-p <TOP_P>] [--top-k <TOP_K>] [--max-tokens <MAX_TOKENS>] [--stream] [--lora-name <LORA_NAME>]
```

Options:
- `--prompt`: Prompt to generate completions for (required)
- `--host`: Server host (default: localhost)
- `--port`: Server port (default: 50051)
- `--temperature`: Temperature for sampling (default: 0.7)
- `--top-p`: Top-p value for sampling (default: 0.95)
- `--top-k`: Top-k value for sampling (default: 50)
- `--max-tokens`: Maximum number of tokens to generate (default: 512)
- `--stream`: Stream the results (flag)
- `--lora-name`: LoRA adapter name

## Troubleshooting

### Import Errors

If you encounter import errors related to `inference_pb2` or `inference_pb2_grpc`, make sure you've generated the gRPC code first using one of the methods above.


