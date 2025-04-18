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

from typing import Optional

import numpy as np
import torch

from arctic_inference.vllm.spec_dec.arctic_speculator import ArcticMLPSpeculator, ArcticLSTMSpeculator

class ArcticProposer:

    def link_model(
        self,
        model: ArcticMLPSpeculator,
    ):
        self.model = model
        self.device = next(model.parameters()).device

    def propose(
        self,
        context_token_ids: np.ndarray,
        previous_hidden_states: torch.Tensor,
    ) -> Optional[np.ndarray]:
        input_ids = torch.tensor(context_token_ids, device=self.device)

        next_tokens = self.model.generate_proposals(
            input_ids=input_ids,
            previous_hidden_states=previous_hidden_states,
            num_predict_tokens=3,
        )

        return next_tokens.cpu().numpy()
