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

from dataclasses import dataclass
import logging

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

    enable_suffix_decoding: bool = False
    suffix_cache_max_depth: int = 64
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

    _orig_from_dict = SpeculativeConfig.__dict__["from_dict"].__wrapped__
    _orig_post_init = SpeculativeConfig.__post_init__

    def __new__(cls, *args, **kwargs):
        # Override __new__ to return an ArcticSpeculativeConfig instead of a
        # SpeculativeConfig when creating a new instance of the class.
        if cls is SpeculativeConfig:
            return ArcticSpeculativeConfig.__new__(ArcticSpeculativeConfig,
                                                   *args, **kwargs)
        return super(SpeculativeConfig, cls).__new__(cls)

    def __post_init__(self):
        use_suffix = (self.method
                      == "suffix") or (self.method is None
                                       and self.enable_suffix_decoding)
        if (use_suffix or self.method == "arctic") and \
            self.disable_by_batch_size is None:
            logger.info("Defaulting disable_by_batch_size to 64")
            self.disable_by_batch_size = 64
            
        if use_suffix:
            self.method = "suffix"
            self.enable_suffix_decoding = True
            self.num_speculative_tokens = self.suffix_cache_max_depth
            self._verify_args()
        else:
            self._orig_post_init()

    @classmethod
    def from_dict(cls, dict_value: dict) -> SpeculativeConfig:
        """Parse the CLI value for the speculative config."""
        if cls is SpeculativeConfig:
            return SpeculativeConfigPatch._orig_from_dict(
                ArcticSpeculativeConfig, dict_value)
        return SpeculativeConfigPatch._orig_from_dict(cls, dict_value)


class VllmConfigPatch(ArcticPatch[VllmConfig]):

    _orig_str = VllmConfig.__str__

    def __str__(self, *args, **kwargs):
        string = self._orig_str(*args, **kwargs)
        string += f", ulysses_sequence_parallel_size={self.parallel_config.ulysses_sequence_parallel_size}"
        string += f", enable_shift_parallel={self.parallel_config.enable_shift_parallel}"
        string += f", shift_parallel_threshold={self.parallel_config.shift_parallel_threshold}"
        return string


class MLPSpeculatorConfigPatch(ArcticPatch[MLPSpeculatorConfig]):

    _orig_init = MLPSpeculatorConfig.__init__

    def __post_init__(self, *args, **kwargs):
        self.base_model_arch = kwargs.pop("base_model_arch", "")
        self._orig_init(*args, **kwargs)
