#!/usr/bin/env python3

import asyncio
import argparse
import logging
import uuid
import sys
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, Optional, List

import grpc
from grpc import aio
import numpy as np
# Import the generated gRPC modules
# Make sure to add the parent directory to sys.path if needed
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    import arctic_inference.embedding.proto.python.inference_pb2 as inference_pb2
    import arctic_inference.embedding.proto.python.inference_pb2_grpc as inference_pb2_grpc
except ImportError:
    print("Error: Could not import gRPC modules. Make sure to run generate_proto.py first.")
    print("Run: python arctic_inference/grpc/generate_proto.py")
    sys.exit(1)

logger = logging.getLogger(__name__)

class InferenceClient:
    """Client for the InferenceService gRPC service."""
    
    def __init__(self, host: str = "localhost", port: int = 50051):
        """Initialize the client.
        
        Args:
            host: The host of the gRPC server.
            port: The port of the gRPC server.
        """
        self.host = host
        self.port = port
        self.channel = None
        self.stub = None
    
    async def connect(self):
        """Connect to the gRPC server."""
        if self.channel is None:
            address = f"{self.host}:{self.port}"
            logger.info(f"Connecting to gRPC server at {address}")
            
            # Create a channel with increased message size limits
            self.channel = aio.insecure_channel(
                address,
                options=[
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ]
            )
            self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)
    
    async def close(self):
        """Close the connection to the gRPC server."""
        if self.channel is not None:
            await self.channel.close()
            self.channel = None
            self.stub = None
    
    
    async def embed(
        self,
        prompts: List[str],
        model_name: str,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Embed a given prompt.
        
        Args:
            prompt: The prompt to embed.
            model_name: The name of the model to use.
            
        Returns:
            An async generator that yields completion responses.
        """
        if self.stub is None:
            await self.connect()
        
        # Create the request
        request_id = request_id or str(uuid.uuid4())
        
        # Create the request
        request = inference_pb2.EncodeRequest(
            request_id=request_id,
            n_prompts=len(prompts),
            prompts=prompts,
            model_name=model_name,
        )
    
        embedding_response = await self.stub.Encode(request)

        if embedding_response.error:
            raise Exception(embedding_response.error)
        
        n_prompts = len(prompts)
        embedding_dim = embedding_response.embedding_dim

        if embedding_response.n_prompts != n_prompts:
            raise Exception(f"Expected {n_prompts} embeddings, got {embedding_response.n_prompts}")

        if len(embedding_response.embedding_bytes_fp32) != n_prompts:
            raise Exception(f"Expected {n_prompts * embedding_dim * 4} bytes, got {len(embedding_response.embedding_bytes_fp32)}")
        
        embeddings = []
        for i in range(n_prompts):
            embedding = np.frombuffer(embedding_response.embedding_bytes_fp32[i], dtype=np.float32).reshape(embedding_dim)
            embeddings.append(embedding)
            print(f"embedding {i} shape: {embedding.shape}")

        return embeddings

    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        stream: bool = True,
        request_id: Optional[str] = None,
        include_logprobs: bool = False,
        lora_name: Optional[str] = None,
        priority: int = 0,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate completions for a given prompt.
        
        Args:
            prompt: The prompt to generate completions for.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            top_k: The top-k value to use for sampling.
            max_tokens: The maximum number of tokens to generate.
            stop: A list of stop sequences.
            stream: Whether to stream the results.
            request_id: The request ID to use.
            include_logprobs: Whether to include log probabilities.
            lora_name: The name of the LoRA adapter to use.
            priority: The priority of the request.
            
        Returns:
            An async generator that yields completion responses.
        """
        if self.stub is None:
            await self.connect()
        
        # Create the request
        request_id = request_id or str(uuid.uuid4())
        
        # Create sampling params
        sampling_params = inference_pb2.SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop or [],
            include_logprobs=include_logprobs,
            stream=stream,
        )
        
        # Create the request
        request = inference_pb2.GenerateRequest(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            priority=priority,
        )
        
        # Add LoRA request if specified
        if lora_name:
            request.lora_request.lora_name = lora_name
        
        try:
            # Stream the results
            async for response in self.stub.Generate(request):
                yield {
                    "request_id": response.request_id,
                    "text": response.text,
                    "finished": response.finished,
                    "token_ids": list(response.token_ids),
                    "generated_tokens": response.generated_tokens,
                }
        except grpc.RpcError as e:
            logger.error(f"RPC error: {e}")
            raise
    
    async def abort(self, request_id: str) -> Dict[str, Any]:
        """Abort an ongoing generation.
        
        Args:
            request_id: The ID of the request to abort.
            
        Returns:
            A dictionary with the result of the abort operation.
        """
        if self.stub is None:
            await self.connect()
        
        request = inference_pb2.AbortRequest(request_id=request_id)
        
        try:
            response = await self.stub.Abort(request)
            return {
                "success": response.success,
                "message": response.message,
            }
        except grpc.RpcError as e:
            logger.error(f"RPC error: {e}")
            raise
    
    async def get_replica_info(self):
        """Get replica information.
        
        Returns:
            A dictionary with model information.
        """
        if self.stub is None:
            await self.connect()
        
        request = inference_pb2.ReplicaInfoRequest()
        
        try:
            response = await self.stub.GetReplicaInfo(request)

            return response
        
        except grpc.RpcError as e:
            logger.error(f"RPC error: {e}")
            raise
    
    async def health_check(self) -> inference_pb2.HealthCheckResponse:
        """Perform a health check.
        
        Returns:
            A dictionary with the health status.
        """
        if self.stub is None:
            await self.connect()
        
        request = inference_pb2.HealthCheckRequest()
        
        try:
            response = await self.stub.HealthCheck(request)
            return response
        except grpc.RpcError as e:
            logger.error(f"RPC error: {e}")
            raise


async def main():
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="gRPC client for vLLM inference")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=50050, help="Server port")
    parser.add_argument("--model", type=str, default="Snowflake/snowflake-arctic-embed-m-v1.5", help="Model name")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate completions for")
    
    
    
    args = parser.parse_args()
    
    # Create the client
    client = InferenceClient(host=args.host, port=args.port)
    
    try:
        # Check server health
        logger.info("Checking server health...")
        health = await client.health_check()
        logger.info(f"Health check result: {health}")
        
        if not health.healthy:
            logger.error(f"Server is not healthy: {health.message}")
            return
        
        # Get replica info
        logger.info("Getting replica info...")
        replica_info = await client.get_replica_info()
        logger.info(f"Replica info: {replica_info}")

        # Embed text
        logger.info("Embedding text...")
        embeddings = await client.embed(prompts=["Hello, world!"] * 16, model_name="Snowflake/snowflake-arctic-embed-m-v1.5")
        
    finally:
        # Close the client
        await client.close()


if __name__ == "__main__":
    asyncio.run(main()) 