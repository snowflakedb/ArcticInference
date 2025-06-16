#!/usr/bin/env python3
"""Replica manager and load-inger that launches multiple replicas on a **single GPU**
(technically the same CUDA device) and forwards client requests to them.

## Architecture

The replica manager consists of:

1. **Replica Manager**: Coordinates replica lifecycle and request routing
2. **Replica State**: Tracks health and status of each replica
3. **Load inger**: Selects replicas based on the configured policy
4. **gRPC Server**: Provides the same API as a single replica but handles distribution

When a request comes in:
1. The replica manager selects a replica based on the load balancing policy
2. The request is forwarded to the selected replica
3. The response is returned to the client
4. If a replica fails, the request is retried on a different replica


Usage example:

```bash
python -m arctic_inference.embedding.replica_manager \
  --port 60050                         \
  --num-replicas 2                     \
  --model "Snowflake/snowflake-arctic-embed-m-v1.5"        
```

The replica manager will listen on ``--port`` (60050) and spawn two replicas on
60051, and 60052.


"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import subprocess
import sys
import time
import uuid
from argparse import Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import grpc
import grpc.aio

os.environ["VLLM_PLUGINS"] = ""

# vLLM imports – *lazy* import inference protos to avoid import-time failures.
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser

# Ensure repo root is on the path so ``python -m`` works from anywhere.
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import arctic_inference.embedding.proto.python.inference_pb2 as inference_pb2
import arctic_inference.embedding.proto.python.inference_pb2_grpc as inference_pb2_grpc

import contextlib

logger = logging.getLogger("arctic_inference.embedding.manage4")


class LoadingerType(str, Enum):
    """Different policies for selecting the next replica."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"


@dataclass
class ReplicaInfo:
    """Book-keeping information for a single replica."""

    id: str
    port: int
    process: subprocess.Popen
    channel: grpc.aio.Channel
    stub: inference_pb2_grpc.InferenceServiceStub
    healthy: bool = False
    current_load: int = 0  # active requests (best-effort)
    total_load: int = 0  # total number of requests (best-effort)
    # current_load_tokens: int = 0  # active tokens (best-effort)
    # total_load_tokens: int = 0  # total number of tokens (best-effort)
    last_checked: int = 0

    async def close(self) -> None:
        if self.channel:
            await self.channel.close()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(5)
            except subprocess.TimeoutExpired:
                self.process.kill()


class ReplicaManager:
    """Maintains a pool of replicas and provides selection + health logic."""

    def __init__(
        self,
        args: Namespace,
        args_list: List[str],
        base_port: int,
        num_replicas: int,
        lb: LoadingerType = LoadingerType.ROUND_ROBIN,
        health_interval: float = 5.0,
    ) -> None:
        self.args = args
        self.args_list = args_list
        self.base_port = base_port
        self.num_replicas = num_replicas
        self.lb = lb
        self.health_interval = health_interval
        self.ready = False

        self._replicas: Dict[str, ReplicaInfo] = {}
        self._rr_index = -1  # will become 0 after first selection
        self._health_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        # configurable startup timeout (seconds) for each replica to become healthy
        self.startup_timeout: int = getattr(args, "startup_timeout", 300)

    # ---------------------------------------------------------------------
    # Public lifecycle helpers
    # ---------------------------------------------------------------------
    async def start(self) -> None:
        """Launch all replicas concurrently and start background health loop."""
        ports = [self.base_port + 1 + i for i in range(self.num_replicas)]

        # check if the ports are available
        for p in ports:
            if not self._check_port_available(p):
                raise RuntimeError(f"Port {p} is already in use")

        results = await asyncio.gather(
            *(self._launch_replica_process(self.args.host, p) for p in ports)
        )
        for replica in results:
            if replica:
                self._replicas[replica.id] = replica

        if not self._replicas:
            raise RuntimeError("Failed to start ANY replicas: aborting manager.")

        self._health_task = asyncio.create_task(self._health_loop())
        self.ready = True

    async def stop(self) -> None:
        if self._health_task:
            self._health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_task
        self.ready = False
        await asyncio.gather(
            *(r.close() for r in list(self._replicas.values())), return_exceptions=True
        )

    # ------------------------------------------------------------------
    # Request routing helpers
    # ------------------------------------------------------------------
    async def route_request(self, method_name: str, request, context):
        """Route request to an appropriate replica and forward the response."""
        replica = await self._select_replica()
        if not replica:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("No healthy replicas")
            return getattr(inference_pb2, self._response_type_name(method_name))()
        # Track load roughly.
        replica.current_load += request.n_prompts
        replica.total_load += request.n_prompts

        try:
            stub_method = getattr(replica.stub, method_name)
            response = await stub_method(request, timeout=self.args.forward_timeout)
            return response
        except grpc.RpcError as exc:
            logger.warning("RPC Error from replica %s - %s", replica.id, exc)
            # TODO(juncheng): consider exporting some metrics and launching a new replica
            await self._mark_unhealthy(replica)
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Replica failure: " + str(exc))
            return getattr(inference_pb2, self._response_type_name(method_name))()
        finally:
            replica.current_load = max(replica.current_load - 1, 0)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    def _check_port_available(self, port: int) -> bool:
        """Check if a port is available.
        return True if the port is available, False otherwise.
        """
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) != 0

    def _build_replica_cmd(self, host: str, port: int) -> List[str]:
        cmd: List[str] = [
            sys.executable,
            "-m",
            "arctic_inference.embedding.replica",
            "--host",
            host,
            "--port",
            str(port),
        ]

        skip_current_arg = False
        for arg in self.args_list:
            if arg.startswith("-"):
                skip_current_arg = False
            if skip_current_arg:
                continue
            if arg in [
                "--host",
                "--port",
                "--num-replicas",
                "--load-balancing",
                "--health-interval",
                "--forward-timeout",
                "--startup-timeout",
            ]:
                skip_current_arg = True
                continue

            cmd.append(arg)

        return cmd

    async def _launch_replica_process(
        self, host: str, port: int
    ) -> Optional[ReplicaInfo]:
        """Start a replica process and wait until it reports healthy."""
        cmd = self._build_replica_cmd(host, port)
        time.sleep(2)
        logger.info("Starting replica on port %d: %s", port, " ".join(cmd))

        # Use line-buffered output so we can stream logs and avoid deadlocks.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,  # capture stdout
            stderr=subprocess.STDOUT,  # redirect stderr to same pipe
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )

        # Stream logs asynchronously to avoid blocking on full pipe buffers.
        asyncio.create_task(self._stream_subprocess_output(proc, f"replica-{port}"))

        # Create gRPC stub.
        channel = grpc.aio.insecure_channel(f"localhost:{port}")
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        replica = ReplicaInfo(
            id=str(uuid.uuid4()), port=port, process=proc, channel=channel, stub=stub
        )

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if await self._check_health(replica):
                replica.healthy = True
                logger.info("Replica %s healthy on port %d", replica.id, port)
                return replica
            # Early exit if process already died.
            if proc.poll() is not None:
                logger.error(
                    "Replica process on port %d exited prematurely with code %s",
                    port,
                    proc.returncode,
                )
                break
            await asyncio.sleep(2)

        logger.error("Replica on port %d failed to become healthy within timeout", port)
        await replica.close()
        return None

    async def _stream_subprocess_output(self, proc: subprocess.Popen, prefix: str):
        """Continuously read a subprocess' combined stdout/stderr and log lines."""
        if not proc.stdout:
            return
        loop = asyncio.get_running_loop()
        try:
            while True:
                line = await loop.run_in_executor(None, proc.stdout.readline)
                if not line:
                    break
                print(line.rstrip())
        except Exception as exc:
            logger.warning("Log stream for %s terminated: %s", prefix, exc)

    async def _health_loop(self):
        while True:
            await asyncio.sleep(self.health_interval)
            await asyncio.gather(
                *(self._periodic_health(r) for r in self._replicas.values())
            )

    async def _periodic_health(self, replica: ReplicaInfo):
        healthy = await self._check_health(replica)
        if healthy and not replica.healthy:
            logger.info("Replica %s recovered", replica.id)
        if not healthy and replica.healthy:
            logger.warning("Replica %s became UNHEALTHY", replica.id)
        replica.healthy = healthy
        replica.last_checked = time.time()

    async def _check_health(self, replica: ReplicaInfo) -> bool:
        try:
            resp = await replica.stub.HealthCheck(
                inference_pb2.HealthCheckRequest(), timeout=2
            )
            return resp.healthy
        except Exception:
            return False

    async def _mark_unhealthy(self, replica: ReplicaInfo):
        replica.healthy = False

    async def _select_replica(self) -> Optional[ReplicaInfo]:
        """Return a healthy replica based on the chosen LB policy."""
        healthy = [r for r in self._replicas.values() if r.healthy]
        if not healthy:
            return None
        if self.lb == LoadingerType.ROUND_ROBIN:
            self._rr_index = (self._rr_index + 1) % len(healthy)
            return healthy[self._rr_index]
        if self.lb == LoadingerType.LEAST_LOADED:
            return min(healthy, key=lambda r: r.current_load)
        if self.lb == LoadingerType.RANDOM:
            return random.choice(healthy)
        # Fallback:
        return healthy[0]

    @staticmethod
    def _response_type_name(method_name: str) -> str:
        # Encode -> EncodeResponse, Generate -> GenerateResponse, etc.
        return method_name + "Response"


class ManagerServicer(inference_pb2_grpc.InferenceServiceServicer):
    """Thin wrapper that forwards every call to ``ReplicaManager``."""

    def __init__(self, replica_manager: ReplicaManager):
        self.replica_manager = replica_manager

    # ----------------------------------------------------------------------------
    # Forwarded RPC methods – keep only Encode / Generate for brevity; add others
    # if you rely on them in production.
    # ----------------------------------------------------------------------------
    async def Encode(self, request, context) -> inference_pb2.EncodeResponse:  # type: ignore
        return await self.replica_manager.route_request("Encode", request, context)

    async def HealthCheck(
        self, request, context
    ) -> inference_pb2.HealthCheckResponse:  # pass‐through to manager
        return inference_pb2.HealthCheckResponse(healthy=self.replica_manager.ready)

    async def GetReplicaInfo(
        self, request, context
    ) -> inference_pb2.ReplicaInfoResponse:  # type: ignore
        """query each replica's info
        and return the info of all replicas
        """
        replica_info_list = []
        n_healthy_replicas = 0
        for replica in self.replica_manager._replicas.values():
            try:
                r = await replica.stub.GetReplicaInfo(
                    request, timeout=self.replica_manager.args.forward_timeout
                )
                replica_info_list.append(r.replica_infos[0])
                n_healthy_replicas += 1
            except grpc.RpcError as e:
                # Mark replica unhealthy and fall through to default response.
                logger.warning(
                    f"grpc Error: {e} getting info from replica {replica.id} mark unhealthy"
                )
                await self.replica_manager._mark_unhealthy(replica)
            except (AttributeError, IndexError) as e:
                # Handle case where replica_infos is missing or empty
                logger.warning(
                    f"Failed to get replica info from {replica.id}: {e}, mark unhealthy"
                )
                await self.replica_manager._mark_unhealthy(replica)

        # Fallback minimal info when no replica is healthy.
        return inference_pb2.ReplicaInfoResponse(
            replica_infos=replica_info_list,
            n_replicas=self.replica_manager.num_replicas,
            n_healthy_replicas=n_healthy_replicas,
            message="",
        )


async def serve(args_list: List[str]):
    from concurrent.futures import ThreadPoolExecutor

    args = parser.parse_args(args_list)

    lm = ReplicaManager(
        args=args,
        args_list=args_list,
        base_port=args.port,
        num_replicas=args.num_replicas,
        lb=LoadingerType(args.load_balancing),
        health_interval=args.health_interval,
    )

    grpc_server = grpc.aio.server(
        ThreadPoolExecutor(max_workers=args.workers),
        options=[
            ("grpc.max_message_length", 200 * 1024 * 1024),
            ("grpc.max_send_message_length", 200 * 1024 * 1024),
            ("grpc.max_receive_message_length", 200 * 1024 * 1024),
        ],
    )
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        ManagerServicer(lm), grpc_server
    )
    listen_addr = f"{args.host}:{args.port}"
    grpc_server.add_insecure_port(listen_addr)

    logger.info("Manager server listening on %s", listen_addr)
    await grpc_server.start()
    await lm.start()

    try:
        await grpc_server.wait_for_termination()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Received shutdown signal")
    finally:
        await lm.stop()
        await grpc_server.stop(0)


if __name__ == "__main__":
    from vllm import logger as vllm_logger  # type: ignore

    logging.basicConfig(
        format=vllm_logger._FORMAT, datefmt=vllm_logger._DATE_FORMAT, level=logging.INFO
    )

    parser = FlexibleArgumentParser(
        description="Replica manager that manages and load balances between multiple replicas on a single GPU"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument(
        "--port",
        type=int,
        default=60050,
        help="Port for the manager, replicas will be on consecutive ports",
    )
    parser.add_argument(
        "--num-replicas", type=int, default=2, help="Number of replicas to launch"
    )
    parser.add_argument(
        "--load-balancing",
        default="round_robin",
        choices=[e.value for e in LoadingerType],
        help="Load balancing strategy",
    )
    parser.add_argument(
        "--health-interval", type=int, default=2, help="Seconds between health checks"
    )
    parser.add_argument(
        "--forward-timeout",
        type=int,
        default=20,
        help="Timeout (s) for requests forwarded to replicas",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=60,
        help="Seconds to wait for each replica to become healthy",
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Number of gRPC workers"
    )

    # Propagate vLLM engine args so we can pass them through to each replica.
    parser = AsyncEngineArgs.add_cli_args(parser)

    parser.set_defaults(
        host="0.0.0.0",
        port=50050,
        num_replicas=4,
        health_interval=1,
        forward_timeout=120,
        startup_timeout=120,
        load_balancing="round_robin",
        model="Snowflake/snowflake-arctic-embed-m-v1.5",
    )

    asyncio.run(serve(sys.argv[1:]))
