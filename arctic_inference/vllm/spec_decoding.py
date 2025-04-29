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

from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from arctic_inference.patching import ArcticPatch    

def apply_spec_decoding_patches():
    EngineArgsPatch.apply_patch()


class EngineArgsPatch(ArcticPatch[EngineArgs]):

    _orig_is_v1_supported_oracle = EngineArgs._is_v1_supported_oracle

    def _is_v1_supported_oracle(self, *args, **kwargs):
        
        is_ngram_enabled = False
        is_eagle_enabled = False
        is_arctic_enabled = False
        is_suffix_enabled = False
        if self.speculative_config is not None:
            # This is supported but experimental (handled below).
            speculative_method = self.speculative_config.get("method")
            if speculative_method:
                if speculative_method in ("ngram", "[ngram]"):
                    is_ngram_enabled = True
                elif speculative_method == "eagle":
                    is_eagle_enabled = True
                elif speculative_method == "arctic":
                    is_arctic_enabled = True
                elif speculative_method == "suffix":
                    is_suffix_enabled = True
            else:
                speculative_model = self.speculative_config.get("model")
                if speculative_model in ("ngram", "[ngram]"):
                    is_ngram_enabled = True
            if not (is_ngram_enabled or is_eagle_enabled or is_arctic_enabled
                    or is_suffix_enabled):
                # Other speculative decoding methods are not supported yet.
                from vllm.engine.arg_utils import _raise_or_fallback
                _raise_or_fallback(feature_name="Speculative Decoding",
                                   recommend_to_remove=False)
                return False

        speculative_config_archive = self.speculative_config

        self.speculative_config = None
        res = self._orig_is_v1_supported_oracle(*args, **kwargs)
        self.speculative_config = speculative_config_archive

        return res

