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
from typing import List, Optional, Union

import torch
import vllm.distributed.parallel_state as parallel_state
import vllm.v1.executor.multiproc_executor
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from vllm.attention.layer import Attention
from vllm.config import VllmConfig
from vllm.config import ModelConfig, ParallelConfig
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed.parallel_state import (init_model_parallel_group,
                                             get_world_group)
from vllm.distributed import split_tensor_along_last_dim
from vllm.executor.multiproc_worker_utils import (
    set_multiprocessing_worker_envs)
from vllm.utils import get_distributed_init_method, get_open_port
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.model_executor.models.llama import LlamaAttention
from vllm.v1.executor.multiproc_executor import (MultiprocExecutor, WorkerProc,
                                                 WorkerProcHandle)
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.linear import (UnquantizedLinearMethod, 
                                               RowParallelLinear, 
                                               ColumnParallelLinear)
from vllm.model_executor.models.llama import (LlamaModel,
                                              LlamaForCausalLM,
                                              LlamaDecoderLayer)

from vllm.compilation.decorators import support_torch_compile
from vllm.sequence import IntermediateTensors

from arctic_inference.patching import ArcticPatch


def apply_shift_parallel_patches():
    UlyssesModelConfigPatch.apply_patch()
    UlyssesParallelStatePatch.apply_patch()
    UlyssesMultiprocExecutorPatch.apply_patch()
    UlyssesAttentionPatch.apply_patch()
    UlyssesFlashAttentionImplPatch.apply_patch()
    ShiftParallelLlamaForCausalLM.apply_patch()
    ShiftParallelLlamaAttention.apply_patch()
    ShiftParallelRowParallelLinear.apply_patch()
    ShiftParallelColumnParallelLinear.apply_patch()
    ShiftParallelUnquantizedLinearMethod.apply_patch()
    ShiftParallelFP8LinearMethod.apply_patch()


class ShiftParallelFP8LinearMethod(ArcticPatch[Fp8LinearMethod]):
    _orig_process_weights_after_loading = Fp8LinearMethod.process_weights_after_loading
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._orig_process_weights_after_loading(layer)

        # TODO: skip the rest if shift parallel threshold is 0
        sp_size = parallel_state._SP.world_size
        sp_rank = parallel_state._SP.rank_in_group
        output_partition_sizes = layer.logical_widths
        if isinstance(layer, RowParallelLinear):
            assert layer.weight.shape[0] % sp_size == 0
            chunk_size = layer.weight.shape[0] // sp_size
            self.sp_tp_weight = layer.weight.split(
                chunk_size, dim=0)[sp_rank].t().contiguous().t()
        elif isinstance(layer, ColumnParallelLinear):
            assert layer.weight.shape[1] % sp_size == 0
            chunk_sizes = []
            for size in output_partition_sizes:
                chunk_size = size // sp_size
                chunk_sizes.extend([chunk_size] * sp_size)
            split = layer.weight.split(chunk_sizes, dim=1)
            self.sp_tp_weight = torch.cat(
                [split[i] for i in range(sp_rank, len(split), sp_size)],
                dim=1).t().contiguous().t()
        else:
            # replicated linear
            self.sp_tp_weight = layer.weight

        # print (remove later)
        if parallel_state._SP.rank == 0:
            if output_partition_sizes == [layer.weight.shape[1]]:
                print(f"row parallel SP: {sp_size}.")
            else:
                print(f"column parallel {sp_size}.")
            print(f"loaded weight shape: {layer.weight.shape} "
                  f"stride {layer.weight.stride()} "
                  f"contiguous {layer.weight.is_contiguous()} "
                  f"tcontiguous {layer.weight.t().is_contiguous()} "
                  f" {layer.weight.dtype}")
            print(f"     logical widths: {layer.logical_widths}")
            print(f"      SP_TP weights: {self.sp_tp_weight.shape} "
                  f"stride {self.sp_tp_weight.stride()} "
                  f"contiguous {self.sp_tp_weight.is_contiguous()} "
                  f"tcontiguous {self.sp_tp_weight.t().is_contiguous()} "
                  f" {self.sp_tp_weight.dtype}")
            
    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_marlin:
            return apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias)
        if self.block_quant:
            assert self.quant_config.weight_block_size is not None
            return torch.ops.vllm.apply_w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=layer.input_scale,
                bias=bias,
                cutlass_block_fp8_supported=self.cutlass_block_fp8_supported,
            )

        from .model_runner import SP_TP_MODE
        return self.fp8_linear.apply(
            input=x,
            weight=self.sp_tp_weight if SP_TP_MODE else layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias)

class ShiftParallelUnquantizedLinearMethod(ArcticPatch[UnquantizedLinearMethod]):

    _orig_create_weighs = UnquantizedLinearMethod.create_weights
    def create_weights(self, *args, **kwargs):
        self.output_partition_sizes = kwargs["output_partition_sizes"]
        return self._orig_create_weighs(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        # TODO: skip weight duplication
        # if ParallelConfig.shift_parallel_threshold == 0:
        #     return

        if torch.distributed.get_rank() == 0:
            print(f"ShiftParallelUnquantizedLinearMethod: "
                  f"output_partition_sizes: {self.output_partition_sizes}")
        output_partition_sizes = self.output_partition_sizes
        sp_size = parallel_state._SP.world_size
        sp_rank = parallel_state._SP.rank_in_group
        if isinstance(layer, RowParallelLinear):
            # row parallel linear
            assert layer.weight.shape[1] % sp_size == 0
            chunk_size = layer.weight.shape[1] // sp_size
            self.sp_tp_weight = layer.weight.split(
                chunk_size, dim=1)[sp_rank].contiguous()
        elif isinstance(layer, ColumnParallelLinear):
            # column parallel linear
            assert layer.weight.shape[0] % sp_size == 0
            chunk_sizes = []
            for size in output_partition_sizes:
                chunk_size = size // sp_size
                chunk_sizes.extend([chunk_size] * sp_size)
            split = layer.weight.split(chunk_sizes, dim=0)
            self.sp_tp_weight = torch.cat(
                [split[i] for i in range(sp_rank, len(split), sp_size)])
        else:
            # replicated linear
            self.sp_tp_weight = layer.weight

        # print (remove later)
        if torch.distributed.get_rank() == 0:
            if output_partition_sizes == [layer.weight.shape[0]]:
                print("row parallel linear")
            else:
                print("column parallel linear")
            print(f"loaded weight shape: {layer.weight.shape} "
                  f"stride {layer.weight.stride()} "
                  f"contiguous {layer.weight.is_contiguous()} "
                  f"tcontiguous {layer.weight.t().is_contiguous()} "
                  f" {layer.weight.dtype}")
            print(f"     output_partition_sizes {output_partition_sizes}")
            print(f"     SP_TP weights: {self.sp_tp_weight.shape} "
                  f"stride {self.sp_tp_weight.stride()} "
                  f"contiguous {self.sp_tp_weight.is_contiguous()} "
                  f"tcontiguous {self.sp_tp_weight.t().is_contiguous()} "
                  f" {self.sp_tp_weight.dtype}")

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        from .model_runner import SP_TP_MODE
        # ShiftParallel
        if SP_TP_MODE:
            return F.linear(x, self.sp_tp_weight, bias)
        else:
            return F.linear(x, layer.weight, bias)

class ShiftParallelRowParallelLinear(ArcticPatch[RowParallelLinear]):
    def forward(
        self, input_
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        from .model_runner import SP_TP_MODE
        sp_tp_size = parallel_state._SP_TP.world_size
        sp_tp_rank = parallel_state._SP_TP.rank_in_group
        if self.input_is_parallel:
            input_parallel = input_
        elif SP_TP_MODE:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=sp_tp_size)
            input_parallel = splitted_input[sp_tp_rank].contiguous()
        else:
            tp_rank = parallel_state._TP.rank_in_group
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()
        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        if SP_TP_MODE:
            bias_ = None if (sp_tp_rank > 0
                             or self.skip_bias_add) else self.bias
        else:
            bias_ = None if (self.tp_rank > 0
                             or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_)
        if self.reduce_results and SP_TP_MODE and sp_tp_size > 1:
            output = parallel_state._SP_TP.all_reduce(output_parallel)
        elif self.reduce_results and not SP_TP_MODE and self.tp_size > 1:
            output = parallel_state._TP.all_reduce(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

class ShiftParallelColumnParallelLinear(ArcticPatch[ColumnParallelLinear]):
    def forward(
        self, input_
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_, bias)
        if self.gather_output:
            # All-gather across the partitions.
            from .model_runner import SP_TP_MODE
            if SP_TP_MODE:
                output = parallel_state._SP_TP.all_gather(output_parallel)
            else:
                output = parallel_state._TP.all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

class ShiftParallelLlamaAttention(ArcticPatch[LlamaAttention]):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        import arctic_inference.vllm.model_runner as model_runner
        SP = parallel_state._SP.world_size
        if model_runner.SP_TP_MODE:
            q_size = self.q_size // SP
            kv_size = self.kv_size // SP
        else:
            q_size = self.q_size
            kv_size = self.kv_size
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

@support_torch_compile
class LlamaModelTP(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 model: LlamaModel,
                 prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self._model = [model]  # Box it to avoid recursive registration

    @property
    def model(self) -> LlamaModel:
        return self._model[0]

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model.forward(input_ids, positions, intermediate_tensors,
                                  inputs_embeds)
    
class ShiftParallelLlamaForCausalLM(ArcticPatch[LlamaForCausalLM]):

    # TODO: make the below work
    # _orig_init = LlamaForCausalLM.__init__
    # def __init__(self, vllm_config, model, **kwargs):
    #     return self._orig_init(vllm_config=vllm_config, model=model, **kwargs)

    def _init_model(self,
                vllm_config: VllmConfig,
                prefix: str = "",
                layer_type: type[nn.Module] = LlamaDecoderLayer):
        # original model
        model = LlamaModel(vllm_config=vllm_config,
                          prefix=prefix,
                          layer_type=layer_type)
        # tp-only model
        vllm_config.compilation_config = (
            vllm_config.compilation_config.model_copy())
        vllm_config.compilation_config.inductor_compile_config = (
            vllm_config.compilation_config.inductor_compile_config.copy())
        self.model_tp = LlamaModelTP(vllm_config=vllm_config, model=model)
        # stats
        self.prefill = 0
        self.decode = 0
        self.mixed = 0
        self.numiter = 0
        # return original model
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        from .model_runner import SP_TP_MODE
        assert SP_TP_MODE is not None

        from vllm.forward_context import get_forward_context
        metadata = get_forward_context().attn_metadata
        if torch.distributed.get_rank() == 0:
            print(f"numiter: {self.numiter} "
                  f"input_ids: {input_ids.shape} SP_TP_MODE: {SP_TP_MODE} ")
            if metadata is None:
                print("metadata: None")
            else:
                seq_lens = metadata.seq_lens.tolist()
                num_actual_tokens = metadata.num_actual_tokens
                self.numiter += 1
                if len(seq_lens) == num_actual_tokens:
                    self.decode += 1
                else:
                    if len(seq_lens) == 1 and num_actual_tokens > 1:
                        self.prefill += 1
                    else:
                        self.mixed += 1
                print(f"metadata: "
                      f"actual tokens: {num_actual_tokens} "
                      f"seq. lens: {seq_lens} "
                      f"prefill {self.prefill} "
                      f"decode {self.decode} "
                      f"mixed {self.mixed}")

        # ShiftParallel
        if SP_TP_MODE:
            model_output = self.model_tp(input_ids, positions,
                                         intermediate_tensors, inputs_embeds)
        else:
            model_output = self.model(input_ids, positions,
                                      intermediate_tensors, inputs_embeds)
        return model_output

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
            1, total_num_kv_heads // (
                parallel_config.tensor_parallel_size *
                parallel_config.ulysses_sequence_parallel_size))

    def get_num_attention_heads(self: ModelConfig,
                                parallel_config: ParallelConfig) -> int:
        num_heads = getattr(self.hf_text_config, "num_attention_heads", 0)
        return num_heads // (parallel_config.tensor_parallel_size *
                             parallel_config.ulysses_sequence_parallel_size)
    
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
                    parallel_config.ulysses_sequence_parallel_size)
                   ) % parallel_config.pipeline_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
        return start, end


class UlyssesParallelStatePatch(ArcticPatch[parallel_state]):

    _SP = None
    _SP_TP = None

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
            config.parallel_config.ulysses_sequence_parallel_size

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
        ulysses_parallel_size = tensor_model_parallel_size \
            * sequence_parallel_size
        assert parallel_state._SP is None, (
            "sequence parallel group is already initialized")
        group_ranks = []
        for i in range(pipeline_model_parallel_size):
            for j in range(tensor_model_parallel_size):
                ranks = list(
                    range(i * ulysses_parallel_size + j,
                        (i + 1) * ulysses_parallel_size + j,
                        tensor_model_parallel_size))
                group_ranks.append(ranks)
        _SP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="sp")
        
        # Build full-TP groups for ShiftParallel
        assert parallel_state._SP_TP is None, (
            "full-TP group is already initialized")
        group_ranks = []
        for i in range(pipeline_model_parallel_size):
            ranks = list(
                range(i * ulysses_parallel_size, (i + 1) * ulysses_parallel_size))
            group_ranks.append(ranks)
        _SP_TP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="sp_tp")

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
            "DP rank %s, PP rank %s, SP_TP rank %s, SP rank %s, TP rank %s", rank,
            world_size, _DP.rank_in_group, _PP.rank_in_group, _SP_TP.rank_in_group,
            _SP.rank_in_group, _TP.rank_in_group)

        parallel_state._TP = _TP
        parallel_state._PP = _PP
        parallel_state._SP = _SP
        parallel_state._SP_TP = _SP_TP
        parallel_state._DP = _DP

    from contextlib import contextmanager
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
        sequence_parallel_size = self.parallel_config.ulysses_sequence_parallel_size
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

    def __init__(self, num_heads, *args, **kwargs):
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

        self.SP = vllm.distributed.parallel_state._SP.world_size
        self.device_group = vllm.distributed.parallel_state._SP.device_group

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output):
        if self.SP == 1:
            return self._orig_forward(layer, query, key, value, kv_cache,
                                      attn_metadata, output)
        from .model_runner import SP_TP_MODE
        if SP_TP_MODE:
            q_ = query.reshape(-1, self.num_heads, self.head_size)
            k_ = key.reshape(-1, self.num_kv_heads, self.head_size)
            v_ = value.reshape(-1, self.num_kv_heads, self.head_size)
            c_ = output.reshape(-1, self.num_heads, self.head_size)
        else:
            qkv = torch.cat(
                (query.view(-1, self.SP, self.num_heads * self.head_size),
                 key.view(-1, self.SP, self.num_kv_heads * self.head_size),
                 value.view(-1, self.SP, self.num_kv_heads * self.head_size)),
                dim=-1).transpose(0, 1).reshape(-1,
                    (self.num_heads + 2 * self.num_kv_heads) * self.head_size)
            # all-to-all
            qkv_ = torch.empty_like(qkv)
            torch.distributed.all_to_all_single(qkv_,
                                                qkv,
                                                group=self.device_group)
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
        if SP_TP_MODE:
            output = c_.reshape(output.shape)
        else:
            c = torch.empty_like(c_)
            torch.distributed.all_to_all_single(c, c_, group=self.device_group)
            output.copy_(
                torch.transpose(c.view(self.SP, -1, self.num_heads * self.head_size), 0, 1)
                .reshape(-1, self.num_heads * self.SP * self.head_size))
        return output
