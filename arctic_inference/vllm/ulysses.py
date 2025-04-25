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

import psutil
import signal
import weakref
from typing import List, Optional

import torch
import vllm.distributed.parallel_state as parallel_state
import vllm.v1.executor.multiproc_executor
from vllm.attention.layer import Attention
from vllm.config import ModelConfig, ParallelConfig
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed.parallel_state import (init_model_parallel_group,
                                             get_world_group)
from vllm.executor.multiproc_worker_utils import (
    set_multiprocessing_worker_envs)
from vllm.utils import get_distributed_init_method, get_open_port
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.v1.executor.multiproc_executor import (MultiprocExecutor, WorkerProc,
                                                 WorkerProcHandle)

from arctic_inference.patching import ArcticPatch


def apply_ulysses_patches():
    UlyssesModelConfigPatch.apply_patch()
    UlyssesParallelStatePatch.apply_patch()
    UlyssesMultiprocExecutorPatch.apply_patch()
    UlyssesAttentionPatch.apply_patch()
    UlyssesFlashAttentionImplPatch.apply_patch()


class UlyssesModelConfigPatch(ArcticPatch[ModelConfig]):

    def get_num_kv_heads(self: ModelConfig,
                         parallel_config: ParallelConfig) -> int:
        """Returns the number of KV heads per GPU."""
        if self.use_mla:
            # When using MLA during decode it becomes MQA
            return 1
        total_num_kv_heads = self.get_total_num_kv_heads()
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(
            1, total_num_kv_heads // (parallel_config.tensor_parallel_size *
                                      parallel_config.sequence_parallel_size))

    def get_num_attention_heads(self: ModelConfig,
                                parallel_config: ParallelConfig) -> int:
        num_heads = getattr(self.hf_text_config, "num_attention_heads", 0)
        return num_heads // (parallel_config.tensor_parallel_size *
                             parallel_config.sequence_parallel_size)
    
    def get_layers_start_end_indices(self: ModelConfig,
                                     parallel_config: ParallelConfig,
                                     ) -> tuple[int, int]:
        from vllm.distributed.utils import get_pp_indices
        if self.hf_text_config.model_type == "deepseek_mtp":
            total_num_hidden_layers = getattr(self.hf_text_config,
                                              "num_nextn_predict_layers", 0)
        else:
            total_num_hidden_layers = getattr(self.hf_text_config,
                                              "num_hidden_layers", 0)
        # the layout order is: DP x PP x SP x TP
        pp_rank = (parallel_config.rank //
                   (parallel_config.tensor_parallel_size *
                    parallel_config.sequence_parallel_size)
                   ) % parallel_config.pipeline_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
        return start, end


class UlyssesParallelStatePatch(ArcticPatch[parallel_state]):

    _SP = None

    @staticmethod
    def initialize_model_parallel(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        backend: Optional[str] = None,
    ) -> None:
        """
        Initialize model parallel groups.

        Arguments:
            tensor_model_parallel_size: number of GPUs used for tensor model
                parallelism.
            pipeline_model_parallel_size: number of GPUs used for pipeline model
                parallelism.

        Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
        use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
        the model pipeline. The present function will
        create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
            4 tensor model-parallel groups:
                [g0, g1], [g2, g3], [g4, g5], [g6, g7]
            2 pipeline model-parallel groups:
                [g0, g2, g4, g6], [g1, g3, g5, g7]
        Note that for efficiency, the caller should make sure adjacent ranks
        are on the same DGX box. For example if we are using 2 DGX-1 boxes
        with a total of 16 GPUs, rank 0 to 7 belong to the first box and
        ranks 8 to 15 belong to the second box.
        """
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
            config.parallel_config.sequence_parallel_size

        # the layout order is: ExternalDP x DP x PP x SP x TP
        # ExternalDP is the data parallel group that is not part of the model,
        # every dp rank can generate independently (in verl integration).
        # DP is the data parallel group that is part of the model,
        # all the ranks in the same DP group should generate simultaneously,
        # i.e. the `generate` call in the same DP group should be called together,
        # otherwise it will cause deadlock.
        # to get group_ranks for each dimension, transpose that dimension to the
        # last dimension, then reshape to 2D, then unbind the last dimension
        all_ranks = torch.arange(world_size).reshape(
            -1, data_parallel_size, pipeline_model_parallel_size,
            sequence_parallel_size, tensor_model_parallel_size)  # noqa

        # Build the tensor model-parallel groups.
        from vllm.distributed.parallel_state import _TP
        assert _TP is None, ("tensor model parallel group is already initialized")
        group_ranks = []
        for i in range(world_size // tensor_model_parallel_size):
            ranks = list(
                range(i * tensor_model_parallel_size,
                    (i + 1) * tensor_model_parallel_size))
            group_ranks.append(ranks)
        # message queue broadcaster is only used in tensor model parallel group
        _TP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        use_message_queue_broadcaster=True,
                                        group_name="tp")

        # Build the pipeline model-parallel groups.
        from vllm.distributed.parallel_state import _PP
        assert _PP is None, (
            "pipeline model parallel group is already initialized")
        group_ranks = all_ranks.transpose(2, 4).reshape(
            -1, pipeline_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        _PP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="pp")

        # Build the sequence parallel groups.
        ulysses_model_parallel_size = tensor_model_parallel_size \
            * sequence_parallel_size
        assert parallel_state._SP is None, (
            "sequence parallel group is already initialized")
        group_ranks = []
        for i in range(pipeline_model_parallel_size):
            for j in range(tensor_model_parallel_size):
                ranks = list(
                    range(i * ulysses_model_parallel_size + j,
                        (i + 1) * ulysses_model_parallel_size + j,
                        tensor_model_parallel_size))
                group_ranks.append(ranks)
        _SP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="sp")

        from vllm.distributed.parallel_state import _DP
        assert _DP is None, ("data parallel group is already initialized")
        group_ranks = all_ranks.transpose(1,
                                          4).reshape(-1,
                                                     data_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        _DP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="dp")

        parallel_state.logger.info(
            "rank %s in world size %s is assigned as "
            "DP rank %s, PP rank %s, SP rank %s, TP rank %s", rank,
            world_size, _DP.rank_in_group, _PP.rank_in_group, _SP.rank_in_group,
            _TP.rank_in_group)

        parallel_state._TP = _TP
        parallel_state._PP = _PP
        parallel_state._SP = _SP
        parallel_state._DP = _DP


class UlyssesMultiprocExecutorPatch(ArcticPatch[MultiprocExecutor]):

    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)

        # The child processes will send SIGUSR1 when unrecoverable
        # errors happen.
        def sigusr1_handler(signum, frame):
            vllm.v1.executor.multiproc_executor.logger.fatal(
                "MulitprocExecutor got fatal signal from worker processes, "
                "shutting down. See stack trace above for root cause issue.")
            # Propagate error up to parent process.
            parent_process = psutil.Process().parent()
            parent_process.send_signal(signal.SIGUSR1)
            self.shutdown()

        signal.signal(signal.SIGUSR1, sigusr1_handler)

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        sequence_parallel_size = self.parallel_config.sequence_parallel_size
        assert self.world_size == tensor_parallel_size \
            * sequence_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size * sequence_parallel_size "
            f"({tensor_parallel_size * sequence_parallel_size}). "
            f"Pipeline parallelism is not yet implemented in v1")

        # Set multiprocessing envs that are common to V0 and V1
        set_multiprocessing_worker_envs(self.parallel_config)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # 127.0.0.1 for communication.
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port())

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        self.rpc_broadcast_mq = MessageQueue(self.world_size, self.world_size)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Create workers
        self.workers: List[WorkerProcHandle] = []
        for rank in range(self.world_size):
            worker = WorkerProc.make_worker_process(self.vllm_config, rank,
                                                    rank,
                                                    distributed_init_method,
                                                    scheduler_output_handle)
            self.workers.append(worker)

        # Ensure message queues are ready. Will deadlock if re-ordered
        # Must be kept consistent with the WorkerProc
        self.rpc_broadcast_mq.wait_until_ready()
        for w in self.workers:
            w.worker_response_mq.wait_until_ready()


class UlyssesAttentionPatch(ArcticPatch[Attention]):

    _orig_init = Attention.__init__
    _orig_forward = Attention.forward

    def __init__(self, num_heads, *args, **kwargs) -> None:
        self.sp_size = parallel_state._SP.world_size
        num_heads //= self.sp_size
        kwargs["num_kv_heads"] //= self.sp_size
        return self._orig_init(num_heads, *args, **kwargs)

    def forward(self, query, key, value, **kwargs):
        if self.sp_size == 1:
            return self._orig_forward(query, key, value, **kwargs)

        from vllm.forward_context import get_forward_context
        if self.calculate_kv_scales:
            attn_metadata = get_forward_context().attn_metadata
            if attn_metadata.enable_kv_scales_calculation:
                self.calc_kv_scales(key, value)
        hidden_size = query.shape[-1]
        output = torch.empty_like(query)
        torch.ops.vllm.unified_attention_with_output(
                    query, key, value, output, self.layer_name)
        return output.view(-1, hidden_size)


class UlyssesFlashAttentionImplPatch(ArcticPatch[FlashAttentionImpl]):

    _orig_init = FlashAttentionImpl.__init__
    _orig_forward = FlashAttentionImpl.forward

    def __init__(self, *args, **kwargs):
        self._orig_init(*args, **kwargs)

        self.sp_size = vllm.distributed.parallel_state._SP.world_size
        self.device_group = vllm.distributed.parallel_state._SP.device_group

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output):
        if self.sp_size == 1:
            return self._orig_forward(layer, query, key, value, kv_cache,
                                      attn_metadata, output)

        qkv = torch.cat(
            (query.view(-1, self.sp_size, self.num_heads * self.head_size),
             key.view(-1, self.sp_size, self.num_kv_heads * self.head_size),
             value.view(-1, self.sp_size, self.num_kv_heads * self.head_size)),
            dim=-1).transpose(0, 1).reshape(
                -1, (self.num_heads + 2 * self.num_kv_heads) * self.head_size)
        # all-to-all
        qkv_ = torch.empty_like(qkv)
        torch.distributed.all_to_all_single(qkv_, qkv, group=self.device_group)
        # unpack
        q_, k_, v_ = qkv_.split([
            self.num_heads * self.head_size, self.num_kv_heads *
            self.head_size, self.num_kv_heads * self.head_size
        ], dim=-1)
        # prepare
        q_ = q_.reshape(-1, self.num_heads, self.head_size)
        k_ = k_.reshape(-1, self.num_kv_heads, self.head_size)
        v_ = v_.reshape(-1, self.num_kv_heads, self.head_size)
        c_ = output.view(-1, self.num_heads, self.head_size)
        # original attention
        self._orig_forward(layer, q_, k_, v_, kv_cache, attn_metadata, c_)
        # Ulysses all-to-all 2/2
        c = torch.empty_like(c_)
        torch.distributed.all_to_all_single(c, c_, group=self.device_group)
        output.copy_(
            torch.transpose(
                c.view(self.sp_size, -1, self.num_heads * self.head_size), 0,
                1).reshape(-1, self.num_heads * self.sp_size * self.head_size))
        return output
