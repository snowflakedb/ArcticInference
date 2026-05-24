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

from vllm.logger import init_logger
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

from arctic_inference.patching import ArcticPatch

logger = init_logger(__name__)


class XgrammarBackendPatch(ArcticPatch[XgrammarBackend]):
    """Patch for XgrammarBackend to handle additional structured output."""

    _orig_post_init = XgrammarBackend.__post_init__

    def __post_init__(self):
        self._orig_post_init()

        spec_cfg = self.vllm_config.speculative_config
        if spec_cfg is not None:
            suffix_tokens = getattr(spec_cfg, 'suffix_speculative_tokens', 0)
            base_tokens = spec_cfg.num_speculative_tokens or 0
            self.num_speculative_tokens = max(base_tokens, suffix_tokens)

        logger.info("XgrammarBackendPatch: num_speculative_tokens=%d",
                    self.num_speculative_tokens)
