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

from pydantic.dataclasses import dataclass
import logging

import vllm
from vllm.config import ParallelConfig, SpeculativeConfig, VllmConfig
from vllm.transformers_utils.configs.mlp_speculator import MLPSpeculatorConfig

from arctic_inference.patching import ArcticPatch

logger = logging.getLogger(__name__)


@dataclass
class ArcticParallelConfig(ParallelConfig):

    ulysses_sequence_parallel_size: int = 1
    enable_shift_parallel: bool = False
    shift_parallel_threshold: int = 512

    def __post_init__(self, *args, **kwargs):
        if (self.enable_shift_parallel
                and self.ulysses_sequence_parallel_size == 1):
            raise ValueError("ulysses_sequence_parallel_size must be > 1 "
                             "when enable_shift_parallel is True.")
        super().__post_init__(*args, **kwargs)

    @property
    def world_size(self) -> int:
        return (self.pipeline_parallel_size * self.tensor_parallel_size *
                self.ulysses_sequence_parallel_size)

    @world_size.setter
    def world_size(self, value: int) -> None:
        # ParallelConfig.__post_init__ will assign world_size to PP * TP, while
        # we want PP * TP * SP to be the world size. So we define world_size as
        # a property with a no-op setter to ignore the value later assigned by
        # ParallelConfig.__post_init__.
        pass


@dataclass
class ArcticSpeculativeConfig(SpeculativeConfig):

    method: str | None = None
    enable_suffix_decoding: bool = False
    suffix_cache_max_depth: int = 64
    suffix_speculative_tokens: int = 0
    suffix_cache_max_requests: int = 100000
    suffix_max_spec_factor: float = 1.0
    suffix_max_spec_offset: float = 0.0
    suffix_min_token_prob: float = 0.1


class ParallelConfigPatch(ArcticPatch[ParallelConfig]):

    def __new__(cls, *args, **kwargs):
        # Override __new__ to return an ArcticParallelConfig instead of a
        # ParallelConfig when creating a new instance of the class.
        if cls is ParallelConfig:
            return ArcticParallelConfig.__new__(ArcticParallelConfig, *args,
                                                **kwargs)
        return super(ParallelConfig, cls).__new__(cls)


class SpeculativeConfigPatch(ArcticPatch[SpeculativeConfig]):

    _orig_post_init = SpeculativeConfig.__post_init__

    def __new__(cls, *args, **kwargs):
        if cls is SpeculativeConfig:
            return ArcticSpeculativeConfig.__new__(ArcticSpeculativeConfig,
                                                   *args, **kwargs)
        return super(SpeculativeConfig, cls).__new__(cls)

    def __post_init__(self):
        is_arctic_method = self.method in ("arctic", "mlp_speculator")
        use_suffix = (self.method == "suffix") or (self.method is None 
                                                   and self.enable_suffix_decoding)
        use_hybrid = (self.method == "arctic" and self.enable_suffix_decoding)

        if (use_suffix or is_arctic_method) and self.disable_by_batch_size is None:
            logger.info("Defaulting disable_by_batch_size to 64")
            self.disable_by_batch_size = 64

        if use_hybrid:
            self.suffix_speculative_tokens = self.suffix_cache_max_depth
            
        if use_suffix:
            self.method = "suffix"
            self.enable_suffix_decoding = True
            self.num_speculative_tokens = self.suffix_cache_max_depth
            self._verify_args()
            return 

        if is_arctic_method:
            actual_draft_model = getattr(self, "draft_model", None)
            
            self.draft_model = None 
            
            try:
                self._orig_post_init()
            finally:
                self.draft_model = actual_draft_model
            
            if self.num_speculative_tokens == 0:
                self.num_speculative_tokens = getattr(self, "num_lookahead_slots", 1)
        else:
            self._orig_post_init()


class VllmConfigPatch(ArcticPatch[VllmConfig]):

    _orig_str = VllmConfig.__str__
    _orig_post_init = VllmConfig.__post_init__

    from typing import Literal
    OldEagleModelTypes = vllm.config.speculative.EagleModelTypes
    NewEagleModelTypes = Literal["arctic", OldEagleModelTypes]

    def __str__(self, *args, **kwargs):
        string = self._orig_str(*args, **kwargs)
        string += f", ulysses_sequence_parallel_size={self.parallel_config.ulysses_sequence_parallel_size}"
        string += f", enable_shift_parallel={self.parallel_config.enable_shift_parallel}"
        string += f", shift_parallel_threshold={self.parallel_config.shift_parallel_threshold}"
        return string

    def __post_init__(self, *args, **kwargs):
        # if self.speculative_config is not None:
        #     if self.speculative_config.method not in get_args(EagleModelTypes):
        #         raise ValueError(
        #             "Currently, async scheduling is only supported "
        #             "with EAGLE/MTP kind of speculative decoding"
        #         )
        import sys
        from typing import Literal
        target_module = sys.modules[VllmConfig.__module__]
        original_types = getattr(target_module, "EagleModelTypes")
        NewEagleModelTypes = Literal["mlp_speculator", original_types]
        setattr(target_module, "EagleModelTypes", NewEagleModelTypes)
        try:
            self._orig_post_init(*args, **kwargs)
        finally:
            setattr(target_module, "EagleModelTypes", original_types)


class MLPSpeculatorConfigPatch(ArcticPatch[MLPSpeculatorConfig]):

    _orig_init = MLPSpeculatorConfig.__init__

    def __init__(self, *args, **kwargs):
        self.base_model_arch = kwargs.pop("base_model_arch", "")
        self._orig_init(*args, **kwargs)

        # Inject dummy attributes required by vLLM's ModelArchConfigConvertor
        # The convertor tries to calculate head_size = hidden_size // num_attention_heads
        if not hasattr(self, "num_attention_heads"):
            self.num_attention_heads = 1
        
        if not hasattr(self, "hidden_size"):
            # Fallback to n_embd if present, otherwise default to a safe dummy value
            self.hidden_size = getattr(self, "n_embd", 1024)

        # Ensure hidden_size is an integer to prevent TypeError during division
        if hasattr(self, "hidden_size"):
            self.hidden_size = int(self.hidden_size)
