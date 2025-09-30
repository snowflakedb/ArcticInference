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
from typing import Optional, Callable

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
from vllm.executor.multiproc_worker_utils import (
    set_multiprocessing_worker_envs)
from vllm.utils import get_distributed_init_method, get_open_port, get_loopback_ip
from vllm.v1.executor.abstract import FailureCallback
from vllm.v1.executor.multiproc_executor import (MultiprocExecutor, WorkerProc,
                                                 UnreadyWorkerProcHandle)
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.forward_context import BatchDescriptor
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig, FusedMoEConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod, Fp8Config
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention


from arctic_inference.patching import ArcticPatch


def apply_shift_parallel_patches():
    UlyssesModelConfig.apply_patch()
    UlyssesParallelState.apply_patch()
    UlyssesWorkerProc.apply_patch()
    UlyssesMultiprocExecutor.apply_patch()
    UlyssesAttention.apply_patch()
    UlyssesCudagraphDispatcher.apply_patch()
    UlyssesFusedMoEParallelConfig.apply_patch()
    UlyssesFp8MoEMethod_dense.apply_patch()
    UlyssesDeepseekV2MLAAttention.apply_patch()


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
    _SP_AA = None
    _SP_AG = None
    # Rationale for SP_AA and SP_AG groups:
    # When num_kv_heads > SP, the kv heads are distributed and replicated as in TP.
    # To implement the logic, the distributed kv heads are exchanged with a local
    # all-to-all within SP_AA group followed by an local all-gather within SP_AG
    # group. The SP_AA and SP_AG groups partitions the SP group into two orthogonal
    # sub-groups and will not be initialized if max(1, num_kv_heads / TP) < SP.
    # See the figure in PR #126 https://github.com/snowflakedb/ArcticInference/pull/126

    def initialize_model_parallel(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
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
        if num_kv_heads < sequence_parallel_size:

            # divide SP group into two orthogonal sub-groups:
            sp_aa_size = num_kv_heads
            sp_ag_size = sequence_parallel_size // num_kv_heads
            all_ranks_ = torch.arange(world_size).reshape(
            -1, data_parallel_size, pipeline_model_parallel_size,
            sp_aa_size, sp_ag_size, tensor_model_parallel_size)

            group_ranks = all_ranks_.transpose(3, 5).reshape(
                -1, sp_aa_size).unbind(0)
            group_ranks = [x.tolist() for x in group_ranks]
            SP_AA_group_ranks = group_ranks
            # SP_AA group is used for all-to-all communication of kv heads
            _SP_AA = init_model_parallel_group(group_ranks,
                                            get_world_group().local_rank,
                                            backend,
                                            group_name="sp_aa")
            
            group_ranks = all_ranks_.transpose(4, 5).reshape(
                -1, sp_ag_size).unbind(0)
            group_ranks = [x.tolist() for x in group_ranks]
            SP_AG_group_ranks = group_ranks
            # SP_AG group is used for all-gather communication of kv heads
            _SP_AG = init_model_parallel_group(group_ranks,
                                            get_world_group().local_rank,
                                            backend,
                                            group_name="sp_ag")

            parallel_state._SP_AA = _SP_AA
            parallel_state._SP_AG = _SP_AG

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
                    f"  SP_AA {parallel_state._SP_AA.world_size} ranks {SP_AA_group_ranks}\n"
                    f"  SP_AG {parallel_state._SP_AG.world_size} ranks {SP_AG_group_ranks}\n")

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
        from vllm.distributed.parallel_state import _SP, _SP_TP, _SP_AA, _SP_AG
        if _SP:
            _SP.destroy()
        _SP = None
        if _SP_TP:
            _SP_TP.destroy()
        _SP_TP = None
        if _SP_AA:
            _SP_AA.destroy()
        _SP_AA = None
        if _SP_AG:
            _SP_AG.destroy()
        _SP_AG = None

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
        set_multiprocessing_worker_envs(self.parallel_config)

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
                num_kv_heads = 1
                assert parallel_state._SP_AA is not None and parallel_state._SP_AG is not None, (
                    "UlyssesAttention requires SP_AA and SP_AG groups to be initialized.")
                self.sp_aa_device_group = parallel_state._SP_AA.device_group
                self.sp_ag_device_group = parallel_state._SP_AG.device_group
                self.sp_aa_size = parallel_state._SP_AA.world_size
                self.sp_ag_size = parallel_state._SP_AG.world_size
                # this reorders the all-gathered sequence
                self.order = [j * self.sp_aa_size + i 
                              for i in range(self.sp_aa_size) 
                              for j in range(self.sp_ag_size)]
            else:
                num_kv_heads //= self.sp_size
            kwargs["num_kv_heads"] = num_kv_heads

            self._orig_init(num_heads, *args, **kwargs)

            # if torch.distributed.get_rank() == 0:
            #     print(f"UlyssesAttention: num_heads {num_heads}, num_kv_heads {num_kv_heads}, is_kv_replicated {self.is_kv_replicated}, sp_size {self.sp_size}")
            #     print(f"self.use_mla {self.use_mla}")

        return

    def forward_mla(self, q, kv_c_normed, k_pe, output_shape):

        assert output_shape is not None

        q_head_size = q.shape[2]
        # Transpose query
        q = q.view(-1, self.sp_size, self.num_heads, q_head_size).transpose(0, 1).contiguous()
        q_ = torch.empty_like(q)
        torch.distributed.all_to_all_single(q_, q, group=self.sp_device_group)
        q_ = q_.reshape(-1, self.num_heads, q_head_size)

        # all-gather kv_c_normed
        kv_c_normed_ = torch.empty((kv_c_normed.shape[0] * self.sp_size, kv_c_normed.shape[1]), dtype=kv_c_normed.dtype, device=kv_c_normed.device)
        torch.distributed.all_gather_into_tensor(kv_c_normed_, kv_c_normed, group=self.sp_device_group)

        # all-gather k_pe
        k_pe_ = torch.empty((k_pe.shape[0] * self.sp_size, k_pe.shape[1], k_pe.shape[2]), dtype=k_pe.dtype, device=k_pe.device)
        torch.distributed.all_gather_into_tensor(k_pe_, k_pe, group=self.sp_device_group)

        # original attention
        c_ = self._orig_forward(q_, kv_c_normed_, k_pe_, output_shape=(output_shape[0] * self.sp_size,
                                                                       output_shape[1] // self.sp_size))

        # Ulysses all-to-all
        c = torch.empty_like(c_)
        torch.distributed.all_to_all_single(c, c_, group=self.sp_device_group)
        c = (c.view(self.sp_size, -1)
             .transpose(0, 1)
             .reshape(output_shape))

        return c # torch.randn(output_shape, dtype=q.dtype, device=q.device)

    def forward(self, query, key, value, **kwargs):
        from .model_runner import is_shift_parallel_mode
        if self.sp_size == 1 or is_shift_parallel_mode():
            return self._orig_forward(query, key, value, **kwargs)

        if self.use_mla: 
            # return self.forward_mla(query, key, value, kwargs["output_shape"])
            output_shape = kwargs.get("output_shape", None)
            c_ = self.forward_mla(query, key, value, (output_shape[0] * self.sp_size,
                                                     output_shape[1] // self.sp_size))
            # Ulysses all-to-all
            c = torch.empty_like(c_)
            torch.distributed.all_to_all_single(c, c_, group=self.sp_device_group)
            return (c.view(self.sp_size, -1)
                .transpose(0, 1)
                .reshape(output_shape))

        if self.is_kv_replicated:
            # Ulysses all-to-all 1/2 (query)
            q = query.view(-1,
                           self.sp_size, self.num_heads * self.head_size).transpose(
                               0, 1).reshape(-1,
                                             self.num_heads * self.head_size)
            q_ = torch.empty_like(q)
            torch.distributed.all_to_all_single(q_, q, group=self.sp_device_group)
            # Ulysses pack (key, value)
            kv = torch.cat((key.view(-1, self.sp_aa_size, self.num_kv_heads * self.head_size),
                            value.view(-1, self.sp_aa_size, self.num_kv_heads * self.head_size)),
                           dim=-1).transpose(0, 1).reshape(
                               -1, 2 * self.num_kv_heads * self.head_size)
            # Ulysses all-to-all (key, value)
            kv_part = torch.empty_like(kv)
            torch.distributed.all_to_all_single(kv_part, kv, group=self.sp_aa_device_group)
            # Ulysses all-gather (key, value)
            kv_ = torch.empty(q_.shape[0],
                              2 * self.num_kv_heads * self.head_size,
                              dtype=query.dtype,
                              device=query.device)
            torch.distributed.all_gather_into_tensor(kv_,
                                                     kv_part,
                                                     group=self.sp_ag_device_group)
            # reorder
            kv_chunk = kv_.chunk(self.sp_size)
            kv_ordered = torch.cat([kv_chunk[i] for i in self.order])
            # unpack (key, value)
            k_, v_ = kv_ordered.split([self.num_kv_heads * self.head_size] * 2, dim=-1)
        else:
            # pack
            qkv = (torch.cat(
                (query.view(-1, self.sp_size, self.num_heads * self.head_size),
                key.view(-1, self.sp_size, self.num_kv_heads * self.head_size),
                value.view(-1, self.sp_size, self.num_kv_heads * self.head_size)),
                dim=-1)
                .transpose(0, 1)
                .reshape(-1, (self.num_heads + 2 * self.num_kv_heads) * self.head_size))
            # Ulysses all-to-all 1/2
            qkv_ = torch.empty_like(qkv)
            torch.distributed.all_to_all_single(qkv_, qkv, group=self.sp_device_group)
            # unpack
            q_, k_, v_ = qkv_.split([
                self.num_heads * self.head_size, self.num_kv_heads *
                self.head_size, self.num_kv_heads * self.head_size
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

class UlyssesDeepseekV2MLAAttention(ArcticPatch[DeepseekV2MLAAttention]):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None

        if self.q_lora_rank is not None:
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            qkv_lora_ = torch.empty((qkv_lora.shape[0] * self.sp_size, qkv_lora.shape[1]), dtype=qkv_lora.dtype, device=qkv_lora.device)
            torch.distributed.all_gather_into_tensor(qkv_lora_, qkv_lora, group=self.sp_device_group)
            qkv_lora = qkv_lora_
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert False, "not implemented"
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim],
                                   dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim:], k_pe)

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0],
                          self.num_local_heads * self.v_head_dim))
        return self.o_proj(attn_out)[0]


class UlyssesCudagraphDispatcher(ArcticPatch[CudagraphDispatcher]):

    _orig_initialize_cudagraph_keys = CudagraphDispatcher.initialize_cudagraph_keys

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode,
                                  uniform_decode_query_len: int):

        self._orig_initialize_cudagraph_keys(cudagraph_mode, uniform_decode_query_len)

        # Ulysses specific keys
        if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
            sp_size = parallel_state._SP.world_size
            for bs in self.compilation_config.cudagraph_capture_sizes:
                self.add_cudagraph_key(
                    cudagraph_mode.mixed_mode(),
                    BatchDescriptor(num_tokens=bs * sp_size, uniform_decode=False))

class UlyssesFusedMoEParallelConfig(ArcticPatch[FusedMoEParallelConfig]):

    _orig_make = FusedMoEParallelConfig.make

    def make(tp_size_: int, dp_size_: int,
             vllm_parallel_config: ParallelConfig) -> "FusedMoEParallelConfig":

        _tp_size = parallel_state._TP.world_size
        _tp_rank = parallel_state._TP.rank_in_group
        _sp_size = parallel_state._SP.world_size
        _sp_rank = parallel_state._SP.rank_in_group

        from .model_runner import is_shift_parallel_mode
        # ep logic
        use_ep = True if (vllm_parallel_config.enable_expert_parallel and
                          not is_shift_parallel_mode()) else False
        # ep is not significant if use_ep == False
        return FusedMoEParallelConfig(tp_size=_tp_size,
                                      tp_rank=_tp_rank,
                                      dp_size=1,
                                      dp_rank=0,
                                      ep_size=_sp_size,
                                      ep_rank=_sp_rank,
                                      use_ep=use_ep)

class UlyssesFp8MoEMethod_dense(ArcticPatch[Fp8MoEMethod]):

    _orig_init = Fp8MoEMethod.__init__

    def __init__(self, quant_config: Fp8Config, moe: FusedMoEConfig):
        self._orig_init(quant_config, moe)
        self.use_ep = moe.moe_parallel_config.use_ep

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        assert not enable_eplb

        topk_weights, topk_ids = FusedMoE.select_experts(
                hidden_states=x,
                router_logits=router_logits,
                use_grouped_topk=use_grouped_topk,
                top_k=top_k,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
                indices_type=self.topk_indices_dtype,
                enable_eplb=enable_eplb,
                expert_map=expert_map,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

        # dispatch
        if self.use_ep:
            sp_size = parallel_state._SP.world_size
            sp_group = parallel_state._SP.device_group
            # convert to uint8
            merge_buff = torch.cat([x.view(torch.uint8), topk_weights.view(torch.uint8), topk_ids.view(torch.uint8)], dim=1)
            merge = torch.empty((merge_buff.shape[0] * sp_size, merge_buff.shape[1]), dtype=merge_buff.dtype, device=merge_buff.device)
            # all-gather
            torch.distributed.all_gather_into_tensor(merge, merge_buff, group=sp_group)
            # split
            output_tokens, output_weights, output_ids = merge.split([x.shape[1] * x.element_size(), 
                                                                     topk_weights.shape[1] * topk_weights.element_size(), 
                                                                     topk_ids.shape[1] * topk_ids.element_size()], dim=1)
            # convert to original dtype
            output_tokens = output_tokens.view(x.dtype).contiguous()
            output_weights = output_weights.view(topk_weights.dtype).contiguous()
            output_ids = output_ids.view(topk_ids.dtype).contiguous()
        else:
            output_tokens, output_weights, output_ids = x, topk_weights, topk_ids

        # call experts on GPU
        from vllm.model_executor.layers.fused_moe import fused_experts
        out_expert = fused_experts(
                hidden_states=output_tokens, # x
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=output_weights, # topk_weights
                topk_ids=output_ids, # topk_ids
                inplace=True,
                activation=activation,
                global_num_experts=global_num_experts,
                apply_router_weight_on_input=apply_router_weight_on_input,
                expert_map=expert_map,
                w1_scale=(layer.w13_weight_scale_inv
                          if self.block_quant else layer.w13_weight_scale),
                w2_scale=(layer.w2_weight_scale_inv
                          if self.block_quant else layer.w2_weight_scale),
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                use_fp8_w8a8=True,
                block_shape=self.quant_config.weight_block_size,
                allow_deep_gemm=self.allow_deep_gemm,
                allow_cutlass_block_scaled_grouped_gemm=(
                    self.allow_cutlass_block_scaled_grouped_gemm))

        # combine
        if self.use_ep:
            output = torch.empty_like(x)
            torch.distributed.reduce_scatter_tensor(output, out_expert, group=sp_group)
        else:
            return out_expert

        return output

class UlyssesFp8MoEMethod_sparse(ArcticPatch[Fp8MoEMethod]):

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        assert not enable_eplb

        if torch.distributed.get_rank() == 0:
            print(f"before select_experts x {x.shape}")

        topk_weights, topk_ids = FusedMoE.select_experts(
                hidden_states=x,
                router_logits=router_logits,
                use_grouped_topk=use_grouped_topk,
                top_k=top_k,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
                indices_type=self.topk_indices_dtype,
                enable_eplb=enable_eplb,
                expert_map=expert_map,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

        # dispatch
        if expert_map is None:
            # do it only if EP is on
            output_tokens = x
            output_weights = topk_weights
            output_ids = topk_ids
        else:
            # print
            from vllm.distributed import get_world_group
            torch.cuda.synchronize()
            get_world_group().barrier()
            for i in range(get_world_group().world_size):
                if torch.distributed.get_rank() == i:
                    # print(f"mode_align_block_size topk_ids {topk_ids}, block_size {block_size}, num_experts, {num_experts}, expert_map {expert_map}, pad_sorted_ids {pad_sorted_ids}")
                    print(f"before fused_experts topk_ids {topk_ids} expert_map {expert_map}")
                get_world_group().barrier()

            # find sparse mapping on CPU
            from vllm.distributed.parallel_state import _SP
            sp_size = _SP.world_size
            num_tokens_per_gpu = [0] * sp_size
            gather_ids = [[] for _ in range(sp_size)]
            num_local_experts = expert_map.numel() // sp_size
            for i in range(topk_ids.shape[0]):
                for j in range(topk_ids.shape[1]):
                    expert_id = topk_ids[i][j]
                    gpu_id = expert_id // num_local_experts
                    if num_tokens_per_gpu[gpu_id] == 0:
                        num_tokens_per_gpu[gpu_id] += 1
                        gather_ids[gpu_id].append(i)
                    elif gather_ids[gpu_id][num_tokens_per_gpu[gpu_id]-1] < i:
                        num_tokens_per_gpu[gpu_id] += 1
                        gather_ids[gpu_id].append(i)

            # sparse replication on GPU
            dispatch_index = torch.tensor([i for sub in gather_ids for i in sub], device=x.device)
            input_tokens = torch.index_select(x, 0, dispatch_index)
            input_ids = torch.index_select(topk_ids, 0, dispatch_index)
            input_weights = torch.index_select(topk_weights, 0, dispatch_index)

            # exchange input/output split sizes on CPU
            input_split = torch.tensor(num_tokens_per_gpu)
            output_split = torch.empty_like(input_split)
            torch.distributed.all_to_all_single(output_split, input_split, group=_SP.cpu_group)
            num_input_token = input_split.sum()
            num_output_token = output_split.sum()

            # print
            from vllm.distributed import get_world_group
            torch.cuda.synchronize()
            get_world_group().barrier()
            for i in range(get_world_group().world_size):
                if torch.distributed.get_rank() == i:
                    print(f"input_split {input_split} {num_input_token} output_split {output_split} {num_output_token}, dispatch_index {dispatch_index}")
                get_world_group().barrier()

            # communications on GPU (TODO: fuse)
            output_tokens = torch.empty((num_output_token, x.shape[1]), dtype=x.dtype, device=x.device)
            output_ids = torch.empty((num_output_token, topk_ids.shape[1]), dtype=topk_ids.dtype, device=topk_ids.device)
            output_weights = torch.empty((num_output_token, topk_weights.shape[1]), dtype=topk_weights.dtype, device=topk_weights.device)
            torch.distributed.all_to_all_single(output_tokens, input_tokens, output_split.tolist(), input_split.tolist(), group=_SP.device_group)
            torch.distributed.all_to_all_single(output_ids, input_ids, output_split.tolist(), input_split.tolist(), group=_SP.device_group)
            torch.distributed.all_to_all_single(output_weights, input_weights, output_split.tolist(), input_split.tolist(), group=_SP.device_group)

            # print
            from vllm.distributed import get_world_group
            torch.cuda.synchronize()
            get_world_group().barrier()
            for i in range(get_world_group().world_size):
                if torch.distributed.get_rank() == i:
                    print(f"num_input_token {num_input_token}, num_output_token {num_output_token}, input_tokens {input_tokens.shape} output_tokens {output_tokens.shape}")
                    print(f"output_ids {output_ids}")
                get_world_group().barrier()

        # call experts on GPU
        from vllm.model_executor.layers.fused_moe import fused_experts
        out_expert = fused_experts(
                hidden_states=output_tokens,
                # hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=output_weights,
                # topk_weights=topk_weights,
                topk_ids=output_ids,
                # topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                global_num_experts=global_num_experts,
                apply_router_weight_on_input=apply_router_weight_on_input,
                expert_map=expert_map,
                w1_scale=(layer.w13_weight_scale_inv
                          if self.block_quant else layer.w13_weight_scale),
                w2_scale=(layer.w2_weight_scale_inv
                          if self.block_quant else layer.w2_weight_scale),
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                use_fp8_w8a8=True,
                block_shape=self.quant_config.weight_block_size,
                allow_deep_gemm=self.allow_deep_gemm,
                allow_cutlass_block_scaled_grouped_gemm=(
                    self.allow_cutlass_block_scaled_grouped_gemm))

        # print
        from vllm.distributed import get_world_group
        torch.cuda.synchronize()
        get_world_group().barrier()
        for i in range(get_world_group().world_size):
            if torch.distributed.get_rank() == i:
                print(f"after fused_experts out_expert {out_expert.shape}")
            get_world_group().barrier()

        # combine
        if expert_map is None:
            # reuse output_tokens
            output_tokens = out_expert
        else:
            # communication on GPU
            torch.distributed.all_to_all_single(input_tokens, out_expert, input_split.tolist(), output_split.tolist(), group=_SP.device_group)
            # sparse reduction on GPU
            ext_dispatch_index = dispatch_index.unsqueeze(1).repeat(1, input_tokens.shape[1])
            output_tokens = torch.zeros_like(x).scatter_add(0, ext_dispatch_index, input_tokens)

            # print
            from vllm.distributed import get_world_group
            torch.cuda.synchronize()
            get_world_group().barrier()
            for i in range(get_world_group().world_size):
                if torch.distributed.get_rank() == i:
                    print(f"after combine input_tokens {input_tokens.shape} output_tokens {output_tokens.shape}")
                get_world_group().barrier()

            import traceback
            if torch.distributed.get_rank() == 0:
                traceback.print_stack()

        return output_tokens

# import vllm.model_executor.layers.fused_moe.fused_moe
# class UlyssesFusedMoE(ArcticPatch[ArcticPatch[vllm.model_executor.layers.fused_moe.fused_moe]]):

#     _orig_fused_experts_impl = fused_moe.fused_experts_impl

#     @static_method
#     def fused_experts_impl(*args, **kwargs):
#         print(f"fused_experts_impl")
#         return self._orig_fused_expert_impl(*args, **kwargs)
