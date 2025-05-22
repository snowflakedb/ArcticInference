#!/usr/bin/env python3
"""
gRPC server for running one model on a given GPU. It can be used directly
if only one model is needed. More commonly, it is used together with a load
balancer (manager.py) to serve multiple models on one GPU concurrently. This
is useful for load balancing and improving throughput.

"""

import os
import asyncio
import logging
import torch
import numpy as np
import uuid
import sys
from concurrent import futures
from typing import Optional
from argparse import ArgumentParser, Namespace

import grpc
from grpc import aio
from grpc import ServicerContext


from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.inputs import TokensPrompt
from vllm.pooling_params import PoolingParams

# Ensure we're using vLLM v0 for embedding support
# assert os.environ.get("VLLM_USE_V1", "0") == "0", (
#     "embedding is only supported for vLLM v0"
# )
os.environ["VLLM_USE_V1"] = "0"


# Import the generated protobuf code
try:
    import arctic_inference.grpc.proto.python.inference_pb2 as inference_pb2
    import arctic_inference.grpc.proto.python.inference_pb2_grpc as inference_pb2_grpc
except ImportError:
    print(
        "Error: Could not import gRPC modules. Make sure to run generate_proto.py first."
    )
    print("Run: python arctic_inference/grpc/generate_proto.py")
    sys.exit(1)

# Configure logger
logger = logging.getLogger("arctic_inference.grpc.replica")


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """Implementation of the InferenceService gRPC service.

    This class handles incoming gRPC requests for inference operations including
    text embedding and generation.
    """

    def __init__(self, engine_args: AsyncEngineArgs):
        """Initialize the servicer with AsyncLLMEngine arguments.

        Args:
            engine_args: Configuration for the AsyncLLMEngine.
        """
        self.engine_args = engine_args
        self.engine = None
        self.tokenizer = None
        self.model_config = None
        self.model_name = None
        self.max_model_len = None
        self._active_requests = {}  # Tracks active generation requests
        self.ready = False

    async def start(self):
        """Initialize and start the LLM engine.

        This method creates the AsyncLLMEngine instance and initializes
        the tokenizer and model configuration.
        """
        # Create the engine from the provided arguments
        self.engine = AsyncLLMEngine.from_engine_args(
            self.engine_args, usage_context=UsageContext.API_SERVER
        )
        self.engine.log_requests = False
        self.engine.start_background_loop()

        # Get tokenizer and model configuration
        self.tokenizer = await self.engine.get_tokenizer()
        self.model_config = await self.engine.get_model_config()
        self.model_name = self.model_config.model
        self.max_model_len = self.model_config.max_model_len

        self.ready = True

    async def stop(self):
        """Shut down the LLM engine."""
        await self.engine.shutdown()

    async def _encode_one_prompt(
        self,
        prompt: TokensPrompt,
        pooling_params: PoolingParams,
        request_id: str,
        priority: int,
    ) -> Optional[torch.Tensor]:
        """Encode a single prompt into an embedding.

        Args:
            prompt: The tokenized prompt to encode.
            pooling_params: Parameters for embedding pooling.
            request_id: Unique identifier for this request.
            priority: Priority level of this request.

        Returns:
            Tensor containing the embedding, or None if encoding failed.
        """
        result = None
        async for encode_result in self.engine.encode(
            prompt=prompt,
            pooling_params=pooling_params,
            request_id=request_id,
            priority=priority,
        ):
            result = encode_result.outputs.data

        return result

    async def Encode(
        self, request: inference_pb2.EncodeRequest, context: ServicerContext
    ) -> inference_pb2.EncodeResponse:
        """Handle a request to encode prompts into embeddings.

        Args:
            request: The client request containing prompts to encode.
            context: gRPC service context.

        Returns:
            Response containing the generated embeddings or an error message.
        """
        # Check if the model is ready
        if not self.ready:
            return inference_pb2.EncodeResponse(
                request_id=request.request_id,
                error="Model not ready",
            )

        # Ensure we have a request ID
        request_id = request.request_id or str(uuid.uuid4())
        pooling_params = PoolingParams()

        # Validate the request contains the required number of prompts
        n_prompts = request.n_prompts
        if not hasattr(request, "n_prompts"):
            return inference_pb2.EncodeResponse(
                request_id=request_id,
                error="n_prompts is required",
            )

        # Validate the model name if specified
        if hasattr(request, "model_name") and request.model_name != self.model_name:
            return inference_pb2.EncodeResponse(
                request_id=request_id,
                error=f"Expected model name {self.model_name}, got {request.model_name}",
            )

        logger.info(
            f"Received encoding request: {request_id} having {n_prompts} prompts for {self.model_name}"
        )

        # Process pre-tokenized prompts or raw text prompts
        if hasattr(request, "token_id_bytes_i32") and request.token_id_bytes_i32:
            logging.info("Using tokenized prompts for encoding")
            if len(request.token_id_bytes_i32) != n_prompts:
                return inference_pb2.EncodeResponse(
                    request_id=request_id,
                    error=f"Expected {n_prompts} prompts, got {len(request.token_id_bytes_i32)}",
                )

            # Convert binary token IDs to TokensPrompt objects
            tokenized_prompts = []
            for i in range(n_prompts):
                token_ids = np.frombuffer(request.token_id_bytes_i32[i], dtype=np.int32)
                # TODO(juncheng): need to patch vllm to support tensor input
                tokenized_prompts.append(
                    TokensPrompt(prompt_token_ids=token_ids.tolist())
                )
        else:
            # Check if the correct number of text prompts is provided
            if len(request.prompts) != n_prompts:
                return inference_pb2.EncodeResponse(
                    request_id=request_id,
                    error=f"Expected {n_prompts} prompts, got {len(request.prompts)}",
                )

            # Tokenize the prompts before sending to vLLM
            # This enables batch tokenization which is more efficient than
            # vLLM's per-prompt tokenization
            tokens = self.tokenizer(
                [text_prompt for text_prompt in request.prompts],
                return_tensors=None,
                padding=False,
                return_token_type_ids=False,
                return_attention_mask=False,
                truncation=True,
            ).input_ids
            tokenized_prompts = [
                TokensPrompt(prompt_token_ids=token_ids) for token_ids in tokens
            ]

        # Create encoding tasks for all prompts
        tasks = [
            self._encode_one_prompt(
                tokenized_prompt,
                pooling_params,
                f"{request_id}:{i}",
                request.priority,
            )
            for i, tokenized_prompt in enumerate(tokenized_prompts)
        ]

        # Process all tasks concurrently
        embedding_tensors = await asyncio.gather(*tasks)
        if any(tensor is None for tensor in embedding_tensors):
            return inference_pb2.EncodeResponse(
                request_id=request_id,
                error="No encoding result",
            )

        # Get embedding dimension from the first result
        embedding_dim = embedding_tensors[0].shape[0]

        # Convert embeddings to bytes for the response
        embedding_bytes_fp32 = [
            tensor.numpy().astype(dtype="<f4").tobytes() for tensor in embedding_tensors
        ]

        # Return the response with embeddings
        return inference_pb2.EncodeResponse(
            request_id=request_id,
            n_prompts=n_prompts,
            embedding_dim=embedding_dim,
            embedding_bytes_fp32=embedding_bytes_fp32,
        )

    # TODO: Implement the _convert_sampling_params and _convert_lora_request methods
    # that are referenced in the Generate method

    async def Generate(
        self, request: inference_pb2.GenerateRequest, context: ServicerContext
    ) -> inference_pb2.GenerateResponse:
        """Handle a request to generate text completions.

        Args:
            request: The client request containing prompts and generation parameters.
            context: gRPC service context.

        Returns:
            Iterator of responses with generated tokens.
        """
        # Check if the model is ready
        if not self.ready:
            yield inference_pb2.GenerateResponse(
                request_id=request.request_id,
                error="Model not ready",
            )
            return

        # Ensure we have a request ID
        request_id = request.request_id or str(uuid.uuid4())
        logger.info(f"Received generation request: {request_id}")

        # Convert proto SamplingParams to vLLM SamplingParams
        sampling_params = self._convert_sampling_params(request)

        # Convert proto LoRARequest to vLLM LoRARequest if provided
        lora_request = None
        if hasattr(request, "lora_request") and request.HasField("lora_request"):
            lora_request = self._convert_lora_request(request.lora_request)

        # Track the request for potential abortion
        self._active_requests[request_id] = context

        try:
            # Generate completions
            async for output in self.engine.generate(
                prompt=request.prompt[0] if request.prompt else "",
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
                priority=request.priority,
            ):
                # Convert the output to a proto response
                response = self._convert_output_to_response(output)
                yield response

                # Check if the client has cancelled the request
                if await context.cancelled():
                    logger.info(f"Request {request_id} was cancelled by client")
                    await self.engine.abort(request_id)
                    break

        except Exception as e:
            logger.exception(f"Error processing request {request_id}: {e}")
            yield inference_pb2.GenerateResponse(
                request_id=request_id,
                text="",
                finished=True,
                token_ids=[],
                error=f"Error processing request: {str(e)}",
            )
        finally:
            # Remove the request from active requests
            self._active_requests.pop(request_id, None)

    async def Abort(
        self, request: inference_pb2.AbortRequest, context: ServicerContext
    ) -> inference_pb2.AbortResponse:
        """Abort an ongoing generation request.

        Args:
            request: The abort request containing the request ID to abort.
            context: gRPC service context.

        Returns:
            Response indicating whether the abort was successful.
        """
        request_id = request.request_id
        logger.info(f"Received abort request for: {request_id}")

        try:
            await self.engine.abort(request_id)
            return inference_pb2.AbortResponse(
                success=True, message=f"Request {request_id} aborted successfully"
            )
        except Exception as e:
            logger.exception(f"Error aborting request {request_id}: {e}")
            return inference_pb2.AbortResponse(
                success=False, message=f"Error aborting request: {str(e)}"
            )

    async def GetReplicaInfo(
        self, request: inference_pb2.ReplicaInfoRequest, context: ServicerContext
    ) -> inference_pb2.ReplicaInfoResponse:
        """Get information about this replica including model and configuration.

        Args:
            request: The info request.
            context: gRPC service context.

        Returns:
            Response containing model and configuration information.
        """
        try:
            # Gather all configuration information
            model_config = await self.engine.get_model_config()
            parallel_config = await self.engine.get_parallel_config()
            decoding_config = await self.engine.get_decoding_config()
            scheduler_config = await self.engine.get_scheduler_config()
            lora_config = await self.engine.get_lora_config()

            replica_info = inference_pb2.SingleReplicaInfoResponse(
                model_name=str(model_config.model),
                task=str(model_config.task),
                dtype=str(model_config.dtype),
                ready=self.ready,
                parallel_config=str(parallel_config),
                decoding_config=str(decoding_config),
                scheduler_config=str(scheduler_config),
                lora_config=str(lora_config),
            )

            return inference_pb2.ReplicaInfoResponse(
                replica_infos=[replica_info, replica_info],
                n_replicas=2,
                n_healthy_replicas=2,
                message="",
            )
        except Exception as e:
            logger.exception(f"Error getting model info: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error getting model info: {str(e)}")
            return inference_pb2.ReplicaInfoResponse(
                replica_infos=[],
                n_replicas=1,
                n_healthy_replicas=0,
                message=f"Error getting model info: {str(e)}",
            )

    async def HealthCheck(
        self, request: inference_pb2.HealthCheckRequest, context: ServicerContext
    ) -> inference_pb2.HealthCheckResponse:
        """Check if the service is healthy.

        Args:
            request: The health check request.
            context: gRPC service context.

        Returns:
            Response indicating whether the service is healthy.
        """
        try:
            await self.engine.check_health()
            return inference_pb2.HealthCheckResponse(healthy=self.ready)
        except Exception as e:
            logger.exception(f"Health check failed: {e}")
            return inference_pb2.HealthCheckResponse(
                healthy=False, message=f"Service is unhealthy: {str(e)}"
            )

    def _convert_output_to_response(
        self, output: RequestOutput
    ) -> inference_pb2.GenerateResponse:
        """Convert a vLLM RequestOutput to a proto GenerateResponse.

        Args:
            output: The vLLM output to convert.

        Returns:
            Converted gRPC response.
        """
        response = inference_pb2.GenerateResponse(
            request_id=output.request_id,
            text=output.text,
            finished=output.finished,
            token_ids=output.token_ids,
            generated_tokens=len(output.token_ids) if output.token_ids else 0,
        )

        return response


class InferenceServer:
    """gRPC server for the InferenceService.

    This class manages the lifecycle of the gRPC server and the InferenceServicer.
    """

    def __init__(
        self,
        args: Namespace,
    ):
        """Initialize the server with command line arguments.

        Args:
            args: Command line arguments containing server configuration.
        """
        self.args = args
        self.engine_args = AsyncEngineArgs.from_cli_args(self.args)
        self.server = None

        self.host = args.host
        self.port = args.port
        self.workers = args.workers
        self.ready = False

    async def start(self):
        """Start the gRPC server and initialize the servicer.

        This method configures and starts the gRPC server, then waits for
        termination signals.
        """
        # Create the gRPC server with appropriate concurrency and message size limits
        self.server = aio.server(
            futures.ThreadPoolExecutor(max_workers=self.workers),
            options=[
                ("grpc.max_message_length", 200 * 1024 * 1024),
                ("grpc.max_send_message_length", 200 * 1024 * 1024),
                ("grpc.max_receive_message_length", 200 * 1024 * 1024),
            ],
        )

        # TODO(juncheng): set up metrics

        # Create and start the servicer
        self.servicer = InferenceServicer(self.engine_args)
        await self.servicer.start()

        # Register the servicer with the server
        inference_pb2_grpc.add_InferenceServiceServicer_to_server(
            self.servicer, self.server
        )

        # Start the server
        address = f"{self.host}:{self.port}"
        self.server.add_insecure_port(address)

        logger.info(f"Starting gRPC replica on {address}")

        # Set up exception handling
        # loop = asyncio.get_running_loop()
        # loop.set_exception_handler(exception_handler)

        await self.server.start()
        logger.info("arctic_inference gRPC replica started")

        try:
            # Wait for replica termination
            await self.server.wait_for_termination()
        except asyncio.CancelledError:
            # Handle task cancellation
            print("Server task cancelled.")
        except KeyboardInterrupt:
            # Handle Ctrl+C
            logger.info("KeyboardInterrupt detected. Shutting down server...")
        finally:
            # Ensure server is stopped
            await self.stop()

    async def stop(self):
        """Stop the replica and clean up resources."""
        if self.server:
            logger.info("Stopping replica")
            await self.server.stop(0)
            await self.servicer.stop()
            self.server = None
            self.servicer = None

        # Clean up PyTorch distributed process group if initialized
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


async def serve(args: Namespace) -> None:
    """Main entry point to start the replica.

    Args:
        args: Command line arguments.
    """
    logger.info("args: %s", args)

    server = InferenceServer(args)
    await server.start()


if __name__ == "__main__":
    # Configure logging
    from vllm import logger as vllm_logger  # type: ignore

    logging.basicConfig(
        format=vllm_logger._FORMAT, datefmt=vllm_logger._DATE_FORMAT, level=logging.INFO
    )

    # Parse command line arguments
    parser = ArgumentParser(description="gRPC replica for vLLM inference")
    parser = AsyncEngineArgs.add_cli_args(parser)

    # Replica-specific arguments
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--port", type=int, default=50050, help="Port to bind to for inference"
    )
    parser.add_argument(
        "--metrics-port", type=int, default=80001, help="Port to bind to for metrics"
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Number of gRPC workers"
    )

    # Default model
    parser.set_defaults(model="BAAI/bge-base-en-v1.5")

    args = parser.parse_args()

    # Start the replica
    asyncio.run(serve(args))
