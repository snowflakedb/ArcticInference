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

from __future__ import annotations

import pytest

import vllm
from vllm import LLM, SamplingParams

MAX_MODEL_LEN = 8192


@pytest.fixture
def test_prompts():
    prompt = ""
    return [prompt]


@pytest.fixture
def sampling_configs():
    return [
        SamplingParams(temperature=0,
                       max_tokens=MAX_MODEL_LEN,
                       ignore_eos=True),
        SamplingParams(temperature=0,
                       max_tokens=MAX_MODEL_LEN - 1,
                       ignore_eos=True),
        SamplingParams(temperature=0,
                       max_tokens=MAX_MODEL_LEN - 2,
                       ignore_eos=True),
        SamplingParams(temperature=0,
                       max_tokens=MAX_MODEL_LEN - 3,
                       ignore_eos=True)
    ]


@pytest.fixture
def model_name():
    return "Snowflake/Llama-3.1-SwiftKV-8B-Instruct"


# Define the speculative configurations that will be tested
ARCTIC_SPEC_CONFIG = {
    "method": "arctic",
    "model": "Snowflake/Arctic-LSTM-Speculator-Llama-3.1-8B-Instruct",
    "num_speculative_tokens": 3,
    "disable_by_batch_size": 64,
    "enable_suffix_decoding": True,
}

SUFFIX_SPEC_CONFIG = {
    "method": "suffix",
    "disable_by_batch_size": 64,
}


@pytest.mark.parametrize("spec_config, test_id", [
    (ARCTIC_SPEC_CONFIG, "arctic_spec_decoding"),
    (SUFFIX_SPEC_CONFIG, "suffix_decoding"),
],
                         ids=["arctic", "suffix"])
def test_speculative_decoding(
    monkeypatch: pytest.MonkeyPatch,
    test_prompts: list[str],
    sampling_configs: list[SamplingParams],
    model_name: str,
    spec_config: dict,
    test_id: str,
):
    '''
    Tests that different speculative decoding methods run without raising errors.
    This test is parameterized to cover 'arctic' and 'suffix' methods.
    '''
    with monkeypatch.context() as m:
        m.setenv("VLLM_PLUGINS", "arctic_inference")
        m.setenv("VLLM_USE_V1", "1")

        vllm.plugins.load_general_plugins()

        spec_llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            quantization="fp8",
            speculative_config=spec_config,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
        )

        for sampling_config in sampling_configs:
            try:
                spec_llm.generate(test_prompts, sampling_config)
            except Exception as e:
                method = spec_config.get('method', 'unknown')
                pytest.fail(
                    f"Speculative decoding with method '{method}' failed with error: {e}"
                )
                
        del spec_llm
