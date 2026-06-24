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

import os
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK: bool = False
    ARCTIC_FP32_LM_HEAD: bool = False


def arctic_inference_effective_enabled(
    extra_env: dict[str, str] | None = None,
) -> bool:
    """True if the Arctic vLLM plugin should load for this process or worker env.

    When ``extra_env`` is passed (e.g. ``ModelConfig.extra_env``), it is checked
    in addition to ``os.environ`` so the driver can omit Arctic-only engine
    kwargs when workers will not enable the plugin.
    """
    if os.getenv("ARCTIC_INFERENCE_ENABLED", "0") == "1":
        return True
    if extra_env and str(extra_env.get("ARCTIC_INFERENCE_ENABLED", "0")) == "1":
        return True
    return False


environment_variables: dict[str, Callable[[], Any]] = {
    "ARCTIC_INFERENCE_ENABLED":
    lambda: os.getenv("ARCTIC_INFERENCE_ENABLED", "0") == "1",
    "ARCTIC_INFERENCE_SKIP_PLATFORM_CHECK":
    lambda: os.getenv("ARCTIC_INFERENCE_SKIP_PLATFORM_CHECK", "0") == "1",
    "ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK":
    lambda: os.getenv("ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK", "0") == "1",
    "ARCTIC_INFERENCE_SKIP_VERSION_CHECK":
    lambda: os.getenv("ARCTIC_INFERENCE_SKIP_VERSION_CHECK", "0") == "1",
    # Run the lm_head matmul in fp32 (weights stay in their native
    # dtype; we upcast on the fly). vLLM's V1 sampler already does
    # softmax in fp32, so the full final stage is fp32. Needed for
    # RL workloads that require precise log-probs / token
    # probabilities for off-policy correction. Equivalent to the
    # ``--fp32-lm-head`` CLI flag.
    "ARCTIC_FP32_LM_HEAD":
    lambda: os.getenv("ARCTIC_FP32_LM_HEAD", "0") == "1",
}

# temporary workaround for gpt-oss model
ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK = 1

def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"Environment variable '{name}' not found.")
