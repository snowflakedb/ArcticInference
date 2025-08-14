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

environment_variables: dict[str, Callable[[], Any]] = {
    "ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK":
    lambda: os.getenv("ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK", "0") == "1",
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"Environment variable '{name}' not found.")
