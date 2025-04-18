# SPDX-License-Identifier: Apache-2.0

"""A layer that compute logits from hidden_stats."""
from typing import Optional

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.distributed import (tensor_model_parallel_all_gather,
                              tensor_model_parallel_gather)
from arctic_inference.vllm.spec_dec.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.platforms import current_platform


class LogitsProcessorOpt(nn.Module):
    """Process logits and apply logits processors from sampling metadata.
    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    def __init__(self,
                 vocab_size: int,
                 org_vocab_size: Optional[int] = None,
                 scale: float = 1.0,
                 logits_as_input: bool = False,
                 soft_cap: Optional[float] = None,
                 skip_last_gather: bool = False) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # Whether the input is logits (default is hidden states).
        self.logits_as_input = logits_as_input
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size
        # Soft cap the logits. Used in Gemma 2.
        self.soft_cap = soft_cap
        # Whether to use gather or all-gather to gather the logits.

        self.use_gather = not current_platform.is_tpu(
        ) and not envs.VLLM_USE_V1

        self.skip_last_gather = skip_last_gather

    def _get_logits_and_post_processing(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self.logits_as_input:
            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)

        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale

        return logits

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self._get_logits_and_post_processing(lm_head, hidden_states,
                                                      None)

        return logits

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        logits = lm_head.quant_method.apply(lm_head,
                                             hidden_states,
                                             bias=embedding_bias)

        if not self.skip_last_gather:
            if self.use_gather:
                # None may be returned for rank > 0
                logits = tensor_model_parallel_gather(logits)
            else:
                # Gather is not supported for some devices such as TPUs.
                # Use all-gather instead.
                # NOTE(woosuk): Here, the outputs of every device should not be None
                # because XLA requires strict SPMD among all devices. Every device
                # should execute the same operations after gathering the logits.
                logits = tensor_model_parallel_all_gather(logits)
        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., :self.org_vocab_size]
        return logits

