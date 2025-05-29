from typing import List, Optional
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import (FusedMoE)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, convert_to_channelwise,
    cutlass_block_fp8_supported, cutlass_fp8_supported,
    maybe_create_device_identity, normalize_e4m3fn_to_e4m3fnuz,
    requantize_with_max_scale)
from vllm.model_executor.parameter import (BlockQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.model_executor.layers.quantization.fp8 import (Fp8MoEMethod,
                                                         Fp8KVCacheMethod,
                                                         Fp8Config)

class OriginalFp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.cutlass_block_fp8_supported = cutlass_block_fp8_supported()
        self.out_dtype = torch.get_default_dtype()

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = (not current_platform.has_device_capability(89)
                           or envs.VLLM_TEST_FORCE_FP8_MARLIN)
        # Disable marlin for rocm
        if current_platform.is_rocm():
            self.use_marlin = False

        self.block_quant = self.quant_config.weight_block_size is not None
        if self.block_quant:
            # Marlin doesn't support block-wise fp8
            self.use_marlin = False

        self.fp8_linear = Fp8LinearOp(
            # Default to using per_token quantization if cutlass is supported
            use_per_token_if_dynamic=cutlass_fp8_supported())

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        maybe_create_device_identity()

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        if self.block_quant:
            tp_size = get_tensor_model_parallel_world_size()
            assert self.quant_config.weight_block_size is not None
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            # Required by row parallel
            if (tp_size > 1
                    and input_size // input_size_per_partition == tp_size
                    and input_size_per_partition % block_k != 0):
                raise ValueError(
                    f"Weight input_size_per_partition = "
                    f"{input_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}.")
            # Required by column parallel or enabling merged weights
            if (tp_size > 1 and output_size // output_size_per_partition
                    == tp_size) or len(output_partition_sizes) > 1:
                for output_partition_size in output_partition_sizes:
                    if output_partition_size % block_n != 0:
                        raise ValueError(
                            f"Weight output_partition_size = "
                            f"{output_partition_size} is not divisible by "
                            f"weight quantization block_n = {block_n}.")

        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT
        weight_dtype = (torch.float8_e4m3fn
                        if self.quant_config.is_checkpoint_fp8_serialized else
                        params_dtype)

        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=weight_dtype),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            if not self.block_quant:
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes),
                                     dtype=torch.float32),
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                set_weight_attrs(scale, {"scale_type": "weight_scale"})
                layer.register_parameter("weight_scale", scale)
            else:
                assert self.quant_config.activation_scheme == "dynamic"
                scale = BlockQuantScaleParameter(
                    data=torch.empty(
                        (output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=torch.float32,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                set_weight_attrs(scale, {"scale_type": "weight_scale"})
                # The weight_scale_inv name is intentional for deepseekv3
                layer.register_parameter("weight_scale_inv", scale)

            # INPUT ACTIVATION SCALE
            if self.quant_config.activation_scheme == "static":
                scale = PerTensorScaleParameter(data=torch.empty(
                    len(output_partition_sizes), dtype=torch.float32),
                                                weight_loader=weight_loader)

                scale[:] = torch.finfo(torch.float32).min
                set_weight_attrs(scale, {"scale_type": "input_scale"})
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Pad the weight tensor. This is an optimization on ROCm platform, which
        # can benefit from tensors located far enough from one another in memory
        if (envs.VLLM_ROCM_FP8_PADDING and current_platform.is_rocm()
                and weight.stride(-1) == 1
                and (weight.stride(-2) * weight.element_size()) % 512 == 0):
            num_pad = 256 // weight.element_size()
            weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
            torch.cuda.empty_cache()
        return weight

    def process_weights_after_loading(self, layer: Module) -> None:
        # TODO(rob): refactor block quant into separate class.
        if self.block_quant:
            assert self.quant_config.activation_scheme == "dynamic"
            if current_platform.is_fp8_fnuz():
                weight, weight_scale_inv, _ = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        weight=layer.weight,
                        weight_scale=layer.weight_scale_inv)
            else:
                weight = layer.weight.data
                weight_scale_inv = layer.weight_scale_inv.data

            weight = self._maybe_pad_weight(weight)

            # Torch.compile cannot use Parameter subclasses.
            layer.weight = Parameter(weight, requires_grad=False)
            layer.weight_scale_inv = Parameter(weight_scale_inv,
                                               requires_grad=False)
            return

        # If checkpoint not serialized fp8, quantize the weights.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            qweight, weight_scale = ops.scaled_fp8_quant(layer.weight,
                                                         scale=None)

            # If using marlin (w8a16), kernel uses channelwise weights,
            # so extend the weight scales to be channelwise.
            if self.use_marlin:
                assert weight_scale.numel() == 1
                weight_scale = convert_to_channelwise(
                    weight_scale.expand(len(layer.logical_widths)),
                    layer.logical_widths)

            # Update the layer with the new values.
            layer.weight = Parameter(qweight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.input_scale = None

        # If checkpoint is fp8, handle that there are N scales for N
        # shards in a fused module
        else:
            layer.weight_scale = torch.nn.Parameter(layer.weight_scale.data,
                                                    requires_grad=False)
            if self.quant_config.activation_scheme == "static":
                layer.input_scale = torch.nn.Parameter(layer.input_scale.data,
                                                       requires_grad=False)
            # If using marlin (w8a16), kernel uses channelwise weights,
            # so extend the weight scales to be channelwise.
            if self.use_marlin:
                weight = layer.weight
                weight_scale = convert_to_channelwise(layer.weight_scale,
                                                      layer.logical_widths)

            # If using w8a8, torch._scaled_mm needs per tensor, so
            # requantize the logical shards as a single weight.
            else:
                # Dequant -> Quant with max scale so we can run per tensor.
                weight = layer.weight
                weight_scale = layer.weight_scale

                if current_platform.is_fp8_fnuz():
                    weight, weight_scale, input_scale = \
                        normalize_e4m3fn_to_e4m3fnuz(
                            weight=weight,
                            weight_scale=weight_scale,
                            input_scale=layer.input_scale)
                    if input_scale is not None:
                        layer.input_scale = Parameter(input_scale,
                                                      requires_grad=False)

                weight_scale, weight = requantize_with_max_scale(
                    weight=weight,
                    weight_scale=weight_scale,
                    logical_widths=layer.logical_widths,
                )

            weight = self._maybe_pad_weight(weight)
            # Update layer with new values.
            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            if self.quant_config.activation_scheme == "static":
                layer.input_scale = Parameter(layer.input_scale.max(),
                                              requires_grad=False)

        if self.use_marlin:
            prepare_fp8_layer_for_marlin(layer)
            # Activations not quantized for marlin.
            del layer.input_scale

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

        return self.fp8_linear.apply(input=x,
                                     weight=layer.weight,
                                     weight_scale=layer.weight_scale,
                                     out_dtype=self.out_dtype,
                                     input_scale=layer.input_scale,
                                     bias=bias)

class Fp8LinearMethodEmbedding(OriginalFp8LinearMethod):
    def __init__(self, config: Fp8Config):
        super().__init__(config)

    def embedding(self, layer: torch.nn.Module,
                  input_: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        return F.embedding(input_, layer.weight)


class Fp8ConfigWithEmbedding(Fp8Config):

    def get_quant_method_patch(self, layer: torch.nn.Module,
                               prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import
        from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
    
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return OriginalFp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return Fp8MoEMethod(self)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            return Fp8LinearMethodEmbedding(self)
        return None