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
from vllm.logger import init_logger
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.worker.worker_base import WorkerBase

from arctic_inference.patching import ArcticPatch
from arctic_inference.utils import get_compatible_vllm_version
from arctic_inference.vllm.args import EngineArgsPatch, AsyncEngineArgsPatch
from arctic_inference.vllm.config import ParallelConfigPatch, VllmConfigPatch
from arctic_inference.vllm.ulysses import apply_ulysses_patches

logger = init_logger(__name__)


class EngineCoreProcPatch(ArcticPatch[EngineCoreProc]):

    _orig_run_engine_core = EngineCoreProc.run_engine_core

    @staticmethod
    def run_engine_core(*args, **kwargs):
        # When starting the API server, it will spawn a new process to run the
        # EngineCore. We need to load the plugins in the new process before it
        # initializes the Executor.
        vllm.plugins.load_general_plugins()
        return EngineCoreProcPatch._orig_run_engine_core(*args, **kwargs)


class WorkerBasePatch(ArcticPatch[WorkerBase]):

    _orig_init = WorkerBase.__init__

    def __init__(self, *args, **kwargs):
        # Some patches like the GPUModelRunner will import CUDA libraries when
        # they are initialized, which will cause process forking to fail. For
        # these patches, we need to delay the initialization until after the
        # process has been forked (i.e., in the WorkerBase initializer).
        from arctic_inference.vllm.model_runner import GPUModelRunnerPatch

        GPUModelRunnerPatch.apply_patch()

        return self._orig_init(*args, **kwargs)


def arctic_inference_plugin():

    if get_compatible_vllm_version() != vllm.__version__:
        logger.warning(
            f"ArcticInference requires vllm=={get_compatible_vllm_version()} "
            f"but found vllm=={vllm.__version__}. Ignoring plugin!")
        return

    if not vllm.platforms.current_platform.is_cuda():
        logger.warning(
            f"ArcticInference requires the cuda platform. Ignoring plugin!")
        return

    from transformers import AutoConfig
    from arctic_inference.common.swiftkv import LlamaSwiftKVConfig

    # Register SwiftKV model configurations to transformers.
    AutoConfig.register("llama_swiftkv", LlamaSwiftKVConfig)

    from vllm import ModelRegistry
    from arctic_inference.vllm.swiftkv import LlamaSwiftKVForCausalLM

    # Register SwiftKV model definitions to vLLM.
    ModelRegistry.register_model("LlamaSwiftKVForCausalLM",
                                 LlamaSwiftKVForCausalLM)

    # Patches that make later patches work properly.
    EngineCoreProcPatch.apply_patch()
    WorkerBasePatch.apply_patch()

    # Patches to vLLM arguments and configuration objects.
    EngineArgsPatch.apply_patch()
    AsyncEngineArgsPatch.apply_patch()
    ParallelConfigPatch.apply_patch()
    VllmConfigPatch.apply_patch()

    # Main optimization patches.
    apply_ulysses_patches()
