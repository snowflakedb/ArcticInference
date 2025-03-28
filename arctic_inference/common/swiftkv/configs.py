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

from transformers import LlamaConfig


class LlamaSwiftKVConfig(LlamaConfig):
    """
    Args:
        num_key_value_layers (int, optional):
            The number of layers, from the first layer, that have keys and
            values. If None, all layers have keys and values.
    """

    model_type = "llama_swiftkv"

    def __init__(
        self,
        num_key_value_layers: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_key_value_layers = num_key_value_layers or self.num_hidden_layers
