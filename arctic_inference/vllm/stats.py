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

from dataclasses import dataclass, field
from vllm.v1.spec_decode.metrics import SpecDecodingStats, SpecDecodingLogging
from vllm.logger import init_logger

from arctic_inference.patching import ArcticPatch

logger = init_logger(__name__)


@dataclass
class SpecDecodingStatsPatch(ArcticPatch[SpecDecodingStats]):
    """Patch for SpecDecodingStats to handle additional metrics."""

    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)

    _orig_observe_draft = SpecDecodingStats.observe_draft

    def __init__(self, num_spec_tokens: int,
                 num_accepted_tokens_per_pos: list[int]):
        self.num_spec_tokens = num_spec_tokens
        self.num_accepted_tokens_per_pos = num_accepted_tokens_per_pos

    def observe_draft(self, num_draft_tokens: int, num_accepted_tokens: int):
        if num_draft_tokens > self.num_spec_tokens:
            self.num_spec_tokens = num_draft_tokens
            self.num_accepted_tokens_per_pos.extend(
                [0] *
                (num_draft_tokens - len(self.num_accepted_tokens_per_pos)))

        self._orig_observe_draft(num_draft_tokens, num_accepted_tokens)


class SpecDecodingLoggingPatch(ArcticPatch[SpecDecodingLogging]):
    """Patch for SpecDecodingLogging to handle additional logging."""

    _orig_log = SpecDecodingLogging.log

    def log(self, log_fn=logger.info):
        if not self.num_drafts:
            return

        max_length = max(
            len(lst) for lst in self.accepted_tokens_per_pos_lists)

        for i in range(len(self.accepted_tokens_per_pos_lists)):
            self.accepted_tokens_per_pos_lists[i].extend(
                [0] *
                (max_length - len(self.accepted_tokens_per_pos_lists[i])))

        self._orig_log(log_fn)
