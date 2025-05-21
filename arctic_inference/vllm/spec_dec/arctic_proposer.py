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
    
import copy
from typing import Optional, Union

from vllm.config import VllmConfig, SpeculativeConfig
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

import numpy as np
import torch

from arctic_inference.vllm.spec_dec.arctic_speculator import ArcticMLPSpeculator, ArcticLSTMSpeculator


class ArcticProposer:

    def __init__(
        self,
        vllm_config: VllmConfig,
        speculative_config: SpeculativeConfig,
    ):
        self.vllm_config = vllm_config
        self.speculative_config = speculative_config
        self.num_predict_tokens = self.speculative_config.num_speculative_tokens

        self.model = None
        self.device = None

    def load_model(
        self,
        model: Union[ArcticMLPSpeculator, ArcticLSTMSpeculator],
    ):
        from vllm.config import VllmConfig

        draft_config_model_config = self.speculative_config.draft_model_config
        draft_config_quant_config = VllmConfig._get_quantization_config(
            self.vllm_config.model_config,
            self.vllm_config.load_config,
        )
        self.speculative_config.draft_parallel_config.worker_cls =\
            self.vllm_config.parallel_config.sd_worker_cls
        draft_config_parallel_config = self.speculative_config.draft_parallel_config

        # We cannot use deepcopy here because Ulysses introduces
        # torch._C._distributed_c10d.ProcessGroup objects that are not
        # designed to be pickled.
        draft_worker_config = VllmConfig(
            model_config=draft_config_model_config,
            quant_config=draft_config_quant_config,
            parallel_config=draft_config_parallel_config,
            load_config=self.vllm_config.load_config,
            device_config=self.vllm_config.device_config,
        )

        self.model = get_model(vllm_config=draft_worker_config)
        self.device = next(model.parameters()).device

    def prepare_hidden_states(
        self,
        sample_hidden_states: torch.Tensor,
        sampled_token_ids: np.ndarray,
        spec_decode_metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            return sample_hidden_states

        assert spec_decode_metadata is not None
        valid_mask = sampled_token_ids != -1
        gen_lens = valid_mask.sum(dim=1)
        num_sampled_tokens = np.array(spec_decode_metadata.num_draft_tokens)
        num_sampled_tokens = torch.tensor(num_sampled_tokens,
                                          device=gen_lens.device) + 1
        hidden_states_idx = (gen_lens - 1) + torch.cumsum(
            num_sampled_tokens, 0) - num_sampled_tokens
        previous_hidden_states = sample_hidden_states[hidden_states_idx]

        return previous_hidden_states

    def propose(
        self,
        context_token_ids: np.ndarray,
        previous_hidden_states: torch.Tensor,
    ) -> Optional[np.ndarray]:
        input_ids = torch.tensor(context_token_ids, device=self.device)

        next_tokens = self.model.generate_proposals(
            input_ids=input_ids,
            previous_hidden_states=previous_hidden_states,
            num_predict_tokens=self.num_predict_tokens,
        )

        return next_tokens.cpu().numpy()
