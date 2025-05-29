from typing import Optional, Tuple

import torch
from torch.nn import Parameter

from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.parameter import BasevLLMParameter

logger = init_logger(__name__)


class SwiftKVLinear(ColumnParallelLinear):
    """Linear layer that immediately computes the KV transformations for all
    attentions of the decode-only layers after the prefill layers in SwiftKV.

    Code is based on `vllm.model_executor.layers.linear.QKVParallelLinear`.

    Args:
        num_layers: number of layers after the prefill that still require KV.
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_kv_heads: total number of attention key/value heads.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        head_size: int,
        total_num_kv_heads: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
    ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size,
                                               self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (self.num_layers *
                       2 * self.num_kv_heads) * tp_size * self.head_size
        self.output_sizes = [
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj 
        ] * self.num_layers

        super().__init__(input_size=input_size,
                         output_size=output_size,
                         bias=bias,
                         gather_output=False,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix,
                         return_bias=return_bias)

    def _get_shard_offset_mapping(self, loaded_shard_id: Tuple[int, str]):
        layer_id, shard_id = loaded_shard_id
        layer_size = self.num_kv_heads * self.head_size
        shard_offset_mapping = {
            "k": layer_id * layer_size,
            "v": (self.num_layers + layer_id) * layer_size,
            "total": 2 * self.num_layers * layer_size,
        }
        return shard_offset_mapping.get(shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: Tuple[int, str]):
        _, shard_id = loaded_shard_id
        shard_size_mapping = {
            "k": self.num_kv_heads * self.head_size,
            "v": self.num_kv_heads * self.head_size,
        }
        return shard_size_mapping.get(shard_id)

    def weight_loader_v2(self,
                         param: BasevLLMParameter,
                         loaded_weight: torch.Tensor,
                         loaded_shard_id: Optional[str] = None):
        logger.info_once(
            "Falling back to original weight loader for SwiftKVLinear")
        return self.weight_loader(param, loaded_weight, loaded_shard_id)

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[str] = None):
        if loaded_shard_id is None:  # special case for certain models
            raise NotImplementedError(
                "SwiftKV does not support fused QKV weights")

        param_data = param.data
        output_dim = param.output_dim

        tp_rank = get_tensor_model_parallel_rank()
        assert loaded_shard_id[1] in ["k", "v"]

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)

        param_data = param_data.narrow(output_dim, shard_offset, shard_size)
        shard_id = tp_rank // self.num_kv_head_replicas
        start_idx = shard_id * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
