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

from vllm.config import SpeculativeConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from arctic_inference.patching import ArcticPatch    

def apply_spec_decoding_patches():
    EngineArgsPatch.apply_patch()
    Fp8ConfigPatch.apply_patch()
    SpeculativeConfigPatch.apply_patch()


class EngineArgsPatch(ArcticPatch[EngineArgs]):

    _orig_is_v1_supported_oracle = EngineArgs._is_v1_supported_oracle

    def _is_v1_supported_oracle(self, *args, **kwargs):
        # Hack for 0.8.1, [ngram] is used to activate suffix decoding
        spec_model = self.speculative_model
        self.speculative_model = "[ngram]"
        bool_val = self._orig_is_v1_supported_oracle(*args, **kwargs)
        self.speculative_model = spec_model
        return bool_val


class Fp8ConfigPatch(ArcticPatch[Fp8Config]):

    def get_quant_method(self, *args, **kwargs):
        from arctic_inference.vllm.spec_dec.fp8 import get_quant_method_patch
        return get_quant_method_patch(*args, **kwargs)


class SpeculativeConfigPatch(ArcticPatch[SpeculativeConfig]):

    _orig_maybe_create_spec_config = SpeculativeConfig.maybe_create_spec_config

    @staticmethod
    def maybe_create_spec_config(*args, **kwargs):
        ngram_prompt_lookup_max = kwargs.get("ngram_prompt_lookup_max")
        ngram_prompt_lookup_min = kwargs.get("ngram_prompt_lookup_min")
        spec_config = SpeculativeConfigPatch._orig_maybe_create_spec_config(*args, **kwargs)

        if spec_config is None:
            return None
        
        spec_config.ngram_prompt_lookup_max = ngram_prompt_lookup_max
        spec_config.ngram_prompt_lookup_min = ngram_prompt_lookup_min
        return spec_config

