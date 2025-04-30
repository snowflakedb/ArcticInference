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

from vllm.config import ParallelConfig, SpeculativeConfig, VllmConfig

from arctic_inference.patching import ArcticPatch
from arctic_inference.vllm.args import get_current_arctic_args


@dataclass
class ArcticParallelConfig(ParallelConfig):

    sequence_parallel_size: int = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        arctic_args = get_current_arctic_args()
        self.sequence_parallel_size = arctic_args.sequence_parallel_size

    @property
    def world_size(self) -> int:
        return (self.pipeline_parallel_size *
                self.tensor_parallel_size *
                self.sequence_parallel_size)

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
            return ArcticParallelConfig.__new__(ArcticParallelConfig,
                                                *args, **kwargs)
        return super(ParallelConfig, cls).__new__(cls)


class SpeculativeConfigPatch(ArcticPatch[SpeculativeConfig]):

    _orig_from_dict = SpeculativeConfig.__dict__["from_dict"].__wrapped__
    _orig_post_init = SpeculativeConfig.__post_init__

    def __new__(cls, *args, **kwargs):
        # Override __new__ to return an ArcticSpeculativeConfig instead of a
        # SpeculativeConfig when creating a new instance of the class.
        if cls is SpeculativeConfig:
            return ArcticSpeculativeConfig.__new__(
                ArcticSpeculativeConfig, *args, **kwargs)
        return super(SpeculativeConfig, cls).__new__(cls)

    def __post_init__(self):
        if self.method == "suffix" or (self.method is None and
                                       self.enable_suffix_decoding):
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
        string += f", sequence_parallel_size={self.parallel_config.sequence_parallel_size}"
        return string
