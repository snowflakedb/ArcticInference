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

import re
from importlib.metadata import requires


def get_compatible_vllm_version():
    reqs = requires("arctic_inference")
    for req in reqs:
        match = re.match("vllm==(.*); extra == \"vllm\"", req)
        if match is not None:
            return match.groups()[0]
        

# For debugging
def print0(*args, **kwargs):
    from vllm.distributed.parallel_state import get_tp_group
    if get_tp_group().is_first_rank:
        print(*args, **kwargs)
