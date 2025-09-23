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

import sys

import vllm

import arctic_inference.envs as envs
from arctic_inference.utils import get_compatible_vllm_version


def arctic_inference_plugin():
    if not envs.ARCTIC_INFERENCE_ENABLED:
        print("\x1b[35;1mArctic Inference plugin is disabled. To enable, set "
              "ARCTIC_INFERENCE_ENABLED=1.\x1b[0m", file=sys.stderr)
        return

    if not envs.ARCTIC_INFERENCE_SKIP_VERSION_CHECK:
        compatible_version = get_compatible_vllm_version()
        if vllm.__version__ != compatible_version:
            raise RuntimeError(
                f"Arctic Inference plugin requires vllm=={compatible_version} "
                f"but found vllm=={vllm.__version__}!")
    
    if not envs.ARCTIC_INFERENCE_SKIP_PLATFORM_CHECK:
        if not vllm.platforms.current_platform.is_cuda():
            raise RuntimeError(
                f"Arctic Inference plugin requires the cuda platform!")

    print("\x1b[36;1mArctic Inference plugin is enabled!\x1b[0m",
          file=sys.stderr)

    # Lazy import to avoid potential errors when the plugin is disabled.
    from .patches import apply_arctic_patches
    apply_arctic_patches()
