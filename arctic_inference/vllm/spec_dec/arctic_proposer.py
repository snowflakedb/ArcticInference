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

from typing import Optional, Union, Tuple, List

from vllm.config import VllmConfig
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_model_runner import logger
from vllm.v1.utils import CpuGpuBuffer 
from vllm.utils.platform_utils import is_pin_memory_available 

import numpy as np
import torch

from arctic_inference.vllm.spec_dec.arctic_speculator import ArcticMLPSpeculator, ArcticLSTMSpeculator
from arctic_inference.envs import ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK


class ArcticProposer:

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config

        self.model = None
        self.device = None

        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.backup_next_token_ids = None  # type: Optional[CpuGpuBuffer]

    def load_model(
        self,
        model: Union[ArcticMLPSpeculator, ArcticLSTMSpeculator],
    ):
        from vllm.config import VllmConfig

        draft_config_model_config = self.speculative_config.draft_model_config

        spec_model_archs = draft_config_model_config.hf_config.architectures
        if not isinstance(spec_model_archs, list):
            logger.error(
                f"Draft model architectures {spec_model_archs} is not a list. "
            )
            raise TypeError()
        if len(spec_model_archs) != 1:
            logger.error(
                f"Draft model architectures {spec_model_archs} does not have exactly one architecture. "
            )
            raise ValueError()
        if spec_model_archs[0] not in [
                "ArcticMLPSpeculatorPreTrainedModel",
                "ArcticLSTMSpeculatorPreTrainedModel",
                "MLPVariantSpeculatorPreTrainedModel",
        ]:
            logger.error(
                f"Draft model architecture {spec_model_archs} is not supported by Arctic Speculator. "
            )
            raise ValueError()

        if not ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK:
            base_model_arch = self.vllm_config.model_config.architectures[0]
            if not hasattr(draft_config_model_config.hf_config, "base_model_archs"):
                logger.error(
                    "Draft model config does not have base_model_archs attribute. "
                    "Set ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK=1 to skip this assertion."
                )
                assert False
            base_model_archs_in_spec_config = draft_config_model_config.hf_config.base_model_archs
            if base_model_arch not in base_model_archs_in_spec_config:
                logger.error(
                    f"Draft model trained with base model architectures {base_model_archs_in_spec_config} "
                    f"does not match the base model architecture {base_model_arch} in the vLLM config. "
                    "Set ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK=1 to skip this assertion."
                )
                assert False

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
        self.device = next(self.model.parameters()).device

        self.input_hidden_dim = self.model.input_hidden_dim if isinstance(
            self.model, ArcticLSTMSpeculator) else self.model.emb_dim

        self.backup_next_token_ids = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=self.device,
            with_numpy=True,
        )

    def prepare_hidden_states(
        self,
        sample_hidden_states: torch.Tensor,
        sampled_token_ids: Union[torch.Tensor, np.ndarray],
        spec_decode_metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        # TODO: fuse it into one kernel
        assert sample_hidden_states is not None, "sample_hidden_states must be provided"

        if isinstance(sampled_token_ids, np.ndarray):
            sampled_token_ids = torch.as_tensor(
                sampled_token_ids, device=sample_hidden_states.device, dtype=torch.long
            )
        elif sampled_token_ids.device != sample_hidden_states.device:
            sampled_token_ids = sampled_token_ids.to(sample_hidden_states.device, non_blocking=True)

        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            return sample_hidden_states

        valid_mask = (sampled_token_ids != -1)
        gen_lens = valid_mask.sum(dim=1).to(dtype=torch.int64) 

        if hasattr(spec_decode_metadata, "cu_num_draft_tokens") and spec_decode_metadata.cu_num_draft_tokens is not None:
            cu = spec_decode_metadata.cu_num_draft_tokens
            num_draft_tokens_gpu = torch.cat([cu[0:1], cu[1:] - cu[:-1]])
        else:
            num_draft_tokens_gpu = torch.as_tensor(
                spec_decode_metadata.num_draft_tokens, device=sample_hidden_states.device, dtype=torch.int64
            )

        num_sampled_tokens_per_req = num_draft_tokens_gpu + 1 

        offsets = torch.cumsum(num_sampled_tokens_per_req, dim=0) - num_sampled_tokens_per_req

        hidden_states_idx = offsets + (gen_lens - 1)

        previous_hidden_states = sample_hidden_states.index_select(
            dim=0, index=hidden_states_idx
        )
  
        assert previous_hidden_states.size(-1) == self.input_hidden_dim, (
            f"hidden_states dim {previous_hidden_states.size(-1)} != speculator expected {self.input_hidden_dim}. "
            "Make sure the spec model is trained with the same base model."
        )
        return previous_hidden_states

    def propose(
        self,
        context_token_ids: Union[torch.Tensor, np.ndarray, List[int]],
        previous_hidden_states: Optional[torch.Tensor],
        num_predict_tokens: int,
    ) -> Optional[np.ndarray]:
        assert num_predict_tokens > 0
        if isinstance(context_token_ids, torch.Tensor):
            input_ids = context_token_ids.to(self.device, dtype=torch.long, non_blocking=True)
        else:
            input_ids = torch.as_tensor(context_token_ids, device=self.device, dtype=torch.long)

        next_tokens = self.model.generate_proposals(
            input_ids=input_ids,
            previous_hidden_states=previous_hidden_states,
            num_predict_tokens=num_predict_tokens,
        )
        return next_tokens


    @torch.inference_mode()
    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata,                 # CommonAttentionMetadata
        sampled_token_ids: torch.Tensor,      # [B, 1 + K], device
        requests: dict,                       # req_id -> CachedRequestState
        gpu_input_batch,                      # InputBatch
        discard_request_indices: torch.Tensor,# [<=B] long, device
        num_discarded_requests: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(sampled_token_ids, torch.Tensor)
        assert self.backup_next_token_ids is not None, "Call load_model() first."

        device = sampled_token_ids.device
        B = sampled_token_ids.size(0)

        self.backup_next_token_ids.np[:B] = np.array(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(
                    int(common_attn_metadata.seq_lens_cpu[i].item())
                )
                for i in range(B)
            ],
            dtype=np.int32,
        )
        self.backup_next_token_ids.copy_to_gpu(B)

        valid_grid = sampled_token_ids.clone()
        if num_discarded_requests:
            valid_grid.index_fill_(0, discard_request_indices[:num_discarded_requests], -1)

        valid_mask = (valid_grid != -1) & (valid_grid < gpu_input_batch.vocab_size)

        valid_counts = valid_mask.sum(dim=1).to(torch.int64)        
        last_idx = torch.clamp(valid_counts - 1, min=0)          
        last_tok = valid_grid.gather(1, last_idx.view(-1, 1)).squeeze(1) 

        backup_gpu = self.backup_next_token_ids.gpu[:B]
        if backup_gpu.dtype != last_tok.dtype:
            backup_gpu = backup_gpu.to(last_tok.dtype)

        next_token_ids = torch.where(valid_counts > 0, last_tok, backup_gpu)  
        return next_token_ids, valid_counts


class SuffixProposer:
    def __init__(self):
        pass

    def load_model(
        self,
        model: None,
    ):
        pass
