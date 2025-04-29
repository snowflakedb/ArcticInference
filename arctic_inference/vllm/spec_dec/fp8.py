from typing import Optional
import torch

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.model_executor.layers.quantization.fp8 import (Fp8LinearMethod, 
                                                         Fp8MoEMethod,
                                                         Fp8KVCacheMethod,
                                                         Fp8Config)


class Fp8LinearMethodEmbedding(Fp8LinearMethod):
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
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return Fp8MoEMethod(self)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            return Fp8LinearMethodEmbedding(self)
        return None