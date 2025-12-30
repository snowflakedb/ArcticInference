# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import weakref
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any

import torch
import vllm.distributed.parallel_state as parallel_state
import vllm.envs as envs
from vllm.attention.layer import Attention
from vllm.config import ModelConfig, ParallelConfig, CUDAGraphMode
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed.parallel_state import (init_model_parallel_group,
                                             get_world_group,
                                             destroy_model_parallel,
                                             destroy_distributed_environment)
from vllm.v1.executor.multiproc_executor import (
    set_multiprocessing_worker_envs)
from vllm.utils import get_distributed_init_method, get_open_port, get_loopback_ip
from vllm.v1.executor.abstract import FailureCallback
from vllm.v1.executor.multiproc_executor import (MultiprocExecutor, WorkerProc,
                                                 UnreadyWorkerProcHandle)
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.forward_context import BatchDescriptor


from arctic_inference.patching import ArcticPatch


def apply_shift_parallel_patches():
    UlyssesModelConfig.apply_patch()
    UlyssesParallelState.apply_patch()
    UlyssesWorkerProc.apply_patch()
    UlyssesMultiprocExecutor.apply_patch()
    UlyssesAttention.apply_patch()
    UlyssesCudagraphDispatcher.apply_patch()


class UlyssesModelConfig(ArcticPatch[ModelConfig]):

    _orig_get_num_kv_heads = ModelConfig.get_num_kv_heads
    _orig_get_num_attention_heads = ModelConfig.get_num_attention_heads

    def get_num_kv_heads(self: ModelConfig,
                         parallel_config: ParallelConfig) -> int:
        num_kv_heads = self._orig_get_num_kv_heads(parallel_config)
        sp_size = parallel_config.ulysses_sequence_parallel_size
        return max(1, num_kv_heads // sp_size)

    def get_num_attention_heads(self: ModelConfig,
                                parallel_config: ParallelConfig) -> int:
        num_heads = self._orig_get_num_attention_heads(parallel_config)
        sp_size = parallel_config.ulysses_sequence_parallel_size
        return max(1, num_heads // sp_size)

    def get_layers_start_end_indices(
            self, parallel_config: "ParallelConfig") -> tuple[int, int]:
        from vllm.distributed.utils import get_pp_indices
        if (self.hf_text_config.model_type == "deepseek_mtp"
                or self.hf_config.model_type == "mimo_mtp"
                or self.hf_config.model_type == "glm4_moe_mtp"):
            total_num_hidden_layers = getattr(self.hf_text_config,
                                              "num_nextn_predict_layers", 0)
        else:
            total_num_hidden_layers = getattr(self.hf_text_config,
                                              "num_hidden_layers", 0)
        # the layout order is: DP x PP x SP x TP
        pp_rank = (parallel_config.rank //
                   (parallel_config.tensor_parallel_size *
                    parallel_config.ulysses_sequence_parallel_size)
                   ) % parallel_config.pipeline_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
        return start, end


class UlyssesParallelState(ArcticPatch[parallel_state]):

    _SP = None
    _SP_TP = None

    def initialize_model_parallel(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        decode_context_model_parallel_size: Optional[int] = 1,
        backend: Optional[str] = None,
    ) -> None:
        
        from vllm.distributed.parallel_state import _DP, _EP, _PP, _TP
        # Get world size and rank. Ensure some consistencies.
        assert torch.distributed.is_initialized()
        world_size: int = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        backend = backend or torch.distributed.get_backend(
            get_world_group().device_group)

        data_parallel_size = 1
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        if config is not None:
            data_parallel_size = config.parallel_config.data_parallel_size

        sequence_parallel_size = \
            config.parallel_config.ulysses_sequence_parallel_size

        all_ranks = torch.arange(world_size).reshape(
            -1, data_parallel_size, pipeline_model_parallel_size,
            sequence_parallel_size, tensor_model_parallel_size)  # noqa

        # Build the tensor model-parallel groups.
        assert _TP is None, ("tensor model parallel group is already initialized")
        group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        TP_group_ranks = group_ranks
        # message queue broadcaster is only used in tensor model parallel group
        _TP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        use_message_queue_broadcaster=True,
                                        group_name="tp")

        # Build the pipeline model-parallel groups.
        assert _PP is None, (
            "pipeline model parallel group is already initialized")
        group_ranks = all_ranks.transpose(2, 4).reshape(
            -1, pipeline_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        PP_group_ranks = group_ranks
        _PP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="pp")

        assert _DP is None, ("data parallel group is already initialized")
        group_ranks = all_ranks.transpose(1,
                                          4).reshape(-1,
                                                     data_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        DP_group_ranks = group_ranks
        _DP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="dp")

        assert _EP is None, ("expert parallel group is already initialized")
        group_ranks = all_ranks.transpose(1, 3).reshape(
            -1, data_parallel_size * tensor_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        EP_group_ranks = group_ranks
        _EP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="ep")

        # Build the sequence parallel groups.
        assert parallel_state._SP is None, (
            "sequence parallel group is already initialized")
        group_ranks = all_ranks.transpose(3, 4).reshape(
            -1, sequence_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        SP_group_ranks = group_ranks
        _SP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="sp")

        # Build full-TP groups for ShiftParallel
        shift_parallel_size = (tensor_model_parallel_size *
                               sequence_parallel_size)
        assert parallel_state._SP_TP is None, (
            "full-TP group is already initialized")
        # transpose(3, 4) for obtaining the correct attn head order
        group_ranks = all_ranks.transpose(3, 4).reshape(
            -1, shift_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        SP_TP_group_ranks = group_ranks
        _SP_TP = init_model_parallel_group(group_ranks,
                                           get_world_group().local_rank,
                                           backend,
                                           group_name="sp_tp")

        parallel_state.logger.info(
            "rank %s in world size %s is assigned as DP rank %s, PP rank %s, "
            "TP rank %s, EP rank %s, SP rank %s, SP_TP rank %s", rank,
            world_size, _DP.rank_in_group, _PP.rank_in_group,
            _TP.rank_in_group, _EP.rank_in_group, _SP.rank_in_group,
            _SP_TP.rank_in_group)

        parallel_state._TP = _TP
        parallel_state._PP = _PP
        parallel_state._SP = _SP
        parallel_state._SP_TP = _SP_TP
        parallel_state._DP = _DP
        parallel_state._EP = _EP

        # check if SP requires kv replication
        num_kv_heads = config.model_config._orig_get_num_kv_heads(config.parallel_config)

        if get_world_group().local_rank == 0:
            parallel_state.logger.info(
                    f"UlyssesParallelState initialized:\n"
                    f"  PP {_PP.world_size} ranks {PP_group_ranks}\n"
                    f"  TP {_TP.world_size} ranks {TP_group_ranks}\n"
                    f"  SP {_SP.world_size} ranks {SP_group_ranks}\n"
                    f"  DP {_DP.world_size} ranks {DP_group_ranks}\n"
                    f"  EP {_EP.world_size} ranks {EP_group_ranks}\n"
                    f"  SP_TP {_SP_TP.world_size} ranks {SP_TP_group_ranks}")
            if num_kv_heads < sequence_parallel_size:
                parallel_state.logger.info(
                    f"  KV cache is replicated by factor {sequence_parallel_size // num_kv_heads}\n")

    @contextmanager
    def graph_capture(device: torch.device):
        """
        `graph_capture` is a context manager which should surround the code that
        is capturing the CUDA graph. Its main purpose is to ensure that the
        some operations will be run after the graph is captured, before the graph
        is replayed. It returns a `GraphCaptureContext` object which contains the
        necessary data for the graph capture. Currently, it only contains the
        stream that the graph capture is running on. This stream is set to the
        current CUDA stream when the context manager is entered and reset to the
        default stream when the context manager is exited. This is to ensure that
        the graph capture is running on a separate stream from the default stream,
        in order to explicitly distinguish the kernels to capture
        from other kernels possibly launched on background in the default stream.
        """
        from vllm.distributed.parallel_state import GraphCaptureContext
        context = GraphCaptureContext(torch.cuda.Stream(device=device))
        with parallel_state._TP.graph_capture(context), parallel_state._PP.graph_capture(
                context), parallel_state._SP_TP.graph_capture(context):
            yield context


class UlyssesWorkerProc(ArcticPatch[WorkerProc]):

    def destroy_model_parallel(self):
        from vllm.distributed.parallel_state import _SP, _SP_TP
        if _SP:
            _SP.destroy()
        _SP = None
        if _SP_TP:
            _SP_TP.destroy()
        _SP_TP = None

    def shutdown(self):
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        # destroy Ulysses communicators here
        self.destroy_model_parallel()
        destroy_distributed_environment()


class UlyssesMultiprocExecutor(ArcticPatch[MultiprocExecutor]):

    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: Optional[FailureCallback] = None
        self.io_thread_pool: Optional[ThreadPoolExecutor] = None

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        pp_parallel_size = self.parallel_config.pipeline_parallel_size
        sp_parallel_size = self.parallel_config.ulysses_sequence_parallel_size
        assert (self.world_size ==
                tensor_parallel_size * pp_parallel_size * sp_parallel_size), (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}) x pipeline"
            f"_parallel_size ({pp_parallel_size}) x ulysses_sequence_parallel"
            f"_size ({sp_parallel_size}).")

        # Set multiprocessing envs that are common to V0 and V1
        set_multiprocessing_worker_envs()

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # get_loopback_ip() for communication.
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port())

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        self.rpc_broadcast_mq = MessageQueue(self.world_size,
                                             self.world_size,
                                             max_chunk_bytes=max_chunk_bytes)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Create workers
        unready_workers: list[UnreadyWorkerProcHandle] = []
        from vllm.utils import get_mp_context
        context = get_mp_context()
        shared_worker_lock = context.Lock()
        success = False
        try:
            for rank in range(self.world_size):
                unready_workers.append(
                    WorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=rank,
                        rank=rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                        shared_worker_lock=shared_worker_lock,
                    ))

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.
            self.rpc_broadcast_mq.wait_until_ready()
            for w in self.workers:
                w.worker_response_mq.wait_until_ready()

            self.start_worker_monitor()
            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                self._ensure_worker_termination(
                    [uw.proc for uw in unready_workers])

        # For pipeline parallel, we use a thread pool for asynchronous
        # execute_model.
        if self.max_concurrent_batches > 1:
            # Note: must use only 1 IO thread to keep dequeue sequence
            # from the response queue
            # _async_aggregate_workers_output also assumes a single IO thread
            self.io_thread_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mp_exec_io")

        self.output_rank = self._get_output_rank()
        self.has_connector = self.vllm_config.kv_transfer_config is not None
        self.kv_output_aggregator = KVOutputAggregator(
            self.parallel_config.world_size)


class UlyssesAttention(ArcticPatch[Attention]):

    _orig_init = Attention.__init__
    _orig_forward = Attention.forward

    def __init__(self, num_heads, *args, **kwargs):
        from .model_runner import is_shift_parallel_mode
        self.sp_size = parallel_state._SP.world_size
        self.sp_device_group = parallel_state._SP.device_group
        if not is_shift_parallel_mode():
            num_heads //= self.sp_size
            num_kv_heads = kwargs["num_kv_heads"]
            self.is_kv_replicated = True if num_kv_heads < self.sp_size else False
            if self.is_kv_replicated:
                self.replication_factor = self.sp_size // num_kv_heads
                num_kv_heads = 1
            else:
                num_kv_heads //= self.sp_size
            kwargs["num_kv_heads"] = num_kv_heads
        return self._orig_init(num_heads, *args, **kwargs)

    def forward(self, query, key, value, **kwargs):
        from .model_runner import is_shift_parallel_mode
        if self.sp_size == 1 or is_shift_parallel_mode():
            return self._orig_forward(query, key, value, **kwargs)

        # prepare
        q = query.view(-1, self.sp_size, self.num_heads * self.head_size)
        if self.is_kv_replicated:
            k = key.view(-1, self.sp_size // self.replication_factor, self.head_size).repeat_interleave(self.replication_factor, dim=1)
            v = value.view(-1, self.sp_size // self.replication_factor, self.head_size).repeat_interleave(self.replication_factor, dim=1)
        else:
            k = key.view(-1, self.sp_size, self.num_kv_heads * self.head_size)
            v = value.view(-1, self.sp_size, self.num_kv_heads * self.head_size)

        # pack
        qkv = torch.cat((q, k, v), dim=-1).transpose(0, 1).reshape(
            -1, (self.num_heads + 2 * self.num_kv_heads) * self.head_size)
        
        # Ulysses all-to-all 1/2
        qkv_ = torch.empty_like(qkv)
        torch.distributed.all_to_all_single(qkv_, qkv, group=self.sp_device_group)

        # unpack
        q_, k_, v_ = qkv_.split([
            self.num_heads * self.head_size, 
            self.num_kv_heads * self.head_size, 
            self.num_kv_heads * self.head_size
        ], dim=-1)

        # original attention
        c_ = self._orig_forward(q_, k_, v_, **kwargs)

        # Ulysses all-to-all 2/2
        c = torch.empty_like(c_)
        torch.distributed.all_to_all_single(c, c_, group=self.sp_device_group)
        output = (c.view(self.sp_size, -1, self.num_heads * self.head_size)
                  .transpose(0, 1)
                  .reshape(-1, self.num_heads * self.sp_size * self.head_size))
        
        return output


class UlyssesCudagraphDispatcher(ArcticPatch[CudagraphDispatcher]):

    _orig_initialize_cudagraph_keys = CudagraphDispatcher.initialize_cudagraph_keys

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode,
                                  uniform_decode_query_len: int):

        self._orig_initialize_cudagraph_keys(cudagraph_mode, uniform_decode_query_len)

        # Ulysses specific keys for mixed prefill/decode mode
        if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
            sp_size = parallel_state._SP.world_size
            for bs in self.compilation_config.cudagraph_capture_sizes:
                self.add_cudagraph_key(
                    cudagraph_mode.mixed_mode(),
                    BatchDescriptor(num_tokens=bs * sp_size, uniform_decode=False))

        # Ulyssses specific keys for full decode mode
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL \
            and cudagraph_mode.separate_routine():
            max_num_tokens = uniform_decode_query_len * \
                self.vllm_config.scheduler_config.max_num_seqs
            cudagraph_capture_sizes_for_decode = [
                x for x in self.compilation_config.cudagraph_capture_sizes
                if x <= max_num_tokens and x >= uniform_decode_query_len
            ]
            for bs in cudagraph_capture_sizes_for_decode:
                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    BatchDescriptor(num_tokens=bs * sp_size, uniform_decode=True))
        self.keys_initialized = True



