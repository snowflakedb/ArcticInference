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

import vllm
from vllm import LLM, SamplingParams

vllm.plugins.load_general_plugins()

llm = LLM(
    model="Snowflake/Llama-3.1-SwiftKV-8B-Instruct",
    #model="/checkpoint/llama3.3_70b_swiftkv_16bit",
    tensor_parallel_size=1,
    sequence_parallel_size=4,
    shapeshifter_threshold=512,
    seed=0,
    distributed_executor_backend="mp",
    quantization="fp8",
)

print("=" * 80)

conversation = [
    {
        "role": "user",
        "content": "Hello"
    },
    {
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
    },
    {
        "role": "user",
        "content": "Write an essay about the importance of higher education.",
    },
]

sampling_params = SamplingParams(temperature=0, max_tokens=800)

outputs = llm.chat(conversation, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
