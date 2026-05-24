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

import torch
from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.worker_base import WorkerBase

import arctic_inference.envs as envs
from arctic_inference.patching import ArcticPatch
from arctic_inference.utils import get_compatible_vllm_version
from arctic_inference.vllm.args import EngineArgsPatch, AsyncEngineArgsPatch
from arctic_inference.vllm.config import (ParallelConfigPatch,
                                          SpeculativeConfigPatch,
                                          VllmConfigPatch,
                                          MLPSpeculatorConfigPatch)
from arctic_inference.vllm.fp32_lm_head import (
    apply_fp32_lm_head_patches, set_fp32_lm_head_enabled)
from arctic_inference.vllm.stats import (SpecDecodingStatsPatch,
                                         SpecDecodingLoggingPatch)
from arctic_inference.vllm.structured_output import XgrammarBackendPatch
from arctic_inference.vllm.attention import apply_forest_cascade_patches
from arctic_inference.vllm.ulysses import apply_shift_parallel_patches


logger = init_logger(__name__)


class AsyncSchedulerPatch(ArcticPatch[AsyncScheduler]):
    """Patch AsyncScheduler to:
    1. Respect ``disable_by_batch_size`` when allocating spec token
       placeholders (the worker only drafts for the first N requests).
    2. Use the previous step's actual draft length for dynamic placeholder
       allocation, avoiding wasted verification compute when the real draft
       width (e.g. Arctic n_predict=3) is much smaller than
       num_speculative_tokens (e.g. 12).
    3. Store ``_scheduled_spec_count`` so that the post-fix in
       ``update_from_output`` can compensate for worker-side trimming.
    """

    _orig_update_after_schedule = AsyncScheduler._update_after_schedule

    def _update_after_schedule(self, scheduler_output):
        # Call Scheduler._update_after_schedule (the grandparent),
        # skipping the base AsyncScheduler override which we are replacing.
        Scheduler._update_after_schedule(self, scheduler_output)

        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens

        # Respect disable_by_batch_size: only add spec token placeholders
        # for the first N decode requests (matching the worker's draft_limit
        # in propose_draft_token_ids).
        spec_config = getattr(self.vllm_config, 'speculative_config', None)
        disable_bs = (
            spec_config.disable_by_batch_size if spec_config else None
        )
        decode_with_spec_count = 0
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output
                and request.num_output_placeholders > 0
            )

            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            # Store the originally-scheduled spec count so that
            # update_from_output can compensate for worker-side trimming.
            request._scheduled_spec_count = cur_num_spec_tokens

            request.num_output_placeholders += 1 + cur_num_spec_tokens

            decode_with_spec_count += 1
            if disable_bs and decode_with_spec_count > disable_bs:
                request.spec_token_ids = []
                continue

            # Use previous step's actual draft length to size
            # placeholders.  When suffix had a good match (actual
            # > n_predict), allocate the full width so the next
            # step can verify all suffix tokens.  When suffix
            # didn't match (actual = n_predict from arctic),
            # allocate only that many to avoid wasting attention
            # compute on zero-padded positions.
            # Cold start: allocate full width (generous).
            prev_actual = getattr(
                request, '_prev_actual_draft_len', None)
            if prev_actual is not None:
                num_placeholders = min(
                    max(prev_actual, 1), self.num_spec_tokens)
            else:
                num_placeholders = self.num_spec_tokens

            request.spec_token_ids = [-1] * num_placeholders

    def update_from_output(self, scheduler_output, model_runner_output):
        """Wrap Scheduler.update_from_output to store actual draft counts.

        We infer the drafter's real capability from the acceptance
        results so that the next ``_update_after_schedule`` can size
        placeholders correctly.  This works even when the worker runs
        in a separate process (where scheduler_output._actual_draft_lens
        set by the worker doesn't survive serialisation back to the
        scheduler).

        Strategy:
        1. **Primary path**: read ``_actual_draft_lens`` from the
           ``model_runner_output`` object (attached by the model runner
           to the ``ModelRunnerOutput`` dataclass, which reliably
           survives the async pipeline).
        2. **Legacy path**: read from ``scheduler_output._actual_draft_lens``
           (works in same-process non-async mode).
        3. **Fallback**: infer from acceptance results with exponential
           growth — when all drafted tokens are accepted, double the
           allocation so suffix decoding reaches full capacity in
           O(log n) steps instead of O(n).

        In v0.18, ``update_from_output`` returns ``dict[int, EngineCoreOutputs]``.
        """
        sampled_token_ids = model_runner_output.sampled_token_ids
        req_id_to_index = model_runner_output.req_id_to_index

        result = Scheduler.update_from_output(
            self, scheduler_output, model_runner_output)

        # Primary path: read from model_runner_output (most reliable
        # for async scheduling — the ModelRunnerOutput object is
        # returned by get_output() and guaranteed to survive).
        actual_lens = getattr(
            model_runner_output, '_actual_draft_lens', None)

        # Legacy path: read from scheduler_output (works for non-async
        # or same-process setups where the attribute is preserved).
        if not actual_lens:
            actual_lens = getattr(
                scheduler_output, '_actual_draft_lens', None)

        if actual_lens:
            for req_id, actual_len in actual_lens.items():
                request = self.requests.get(req_id)
                if request is not None:
                    request._prev_actual_draft_len = actual_len
            return result

        # Fallback: infer from acceptance results (multi-process case).
        # Uses exponential growth when all drafted tokens are accepted
        # (indicating the drafter / suffix cache can handle more), so
        # the allocation converges to num_spec_tokens in O(log n)
        # steps:
        #   step 0: 1 position  → accept 1/1 → prev = 2
        #   step 1: 2 positions → accept 2/2 → prev = 4
        #   step 2: 4 positions → accept 4/4 → prev = 8
        #   ...
        # When not all are accepted (normal drafter), linear growth:
        #   step 0: 1 position  → accept 1 → prev = 2
        #   step 1: 2 positions → accept 2 → prev = 3
        #   step 2: 3 positions → steady state (n_predict = 3)
        if not sampled_token_ids:
            return result

        for req_id in scheduler_output.num_scheduled_tokens:
            scheduled_spec = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if not scheduled_spec:
                continue

            req_index = req_id_to_index.get(req_id)
            if req_index is None:
                continue

            request = self.requests.get(req_id)
            if request is None:
                continue

            generated = sampled_token_ids[req_index]
            num_accepted = (len(generated) - 1) if generated else 0
            num_draft = len(scheduled_spec)

            prev = getattr(request, '_prev_actual_draft_len', None)

            if num_accepted > 0:
                if num_accepted >= num_draft and num_draft > 0:
                    # All drafted tokens accepted — the drafter (or
                    # suffix cache) could produce more if given room.
                    # Double the allocation for exponential ramp-up.
                    new_val = min(
                        num_draft * 2, self.num_spec_tokens)
                else:
                    # Partial acceptance: grow linearly.
                    new_val = min(
                        num_accepted + 1, self.num_spec_tokens)
                request._prev_actual_draft_len = max(
                    prev or 0, new_val)
            elif prev is None:
                # First step for this request, zero acceptance.
                # Seed with 1 so _update_after_schedule doesn't
                # fall back to the full num_spec_tokens next time.
                request._prev_actual_draft_len = 1

        return result


class WorkerBasePatch(ArcticPatch[WorkerBase]):

    _orig_init = WorkerBase.__init__

    def __init__(self, *args, **kwargs):
        # Some patches like the GPUModelRunner will import CUDA libraries when
        # they are initialized, which will cause process forking to fail. For
        # these patches, we need to delay the initialization until after the
        # process has been forked (i.e., in the WorkerBase initializer).
        from arctic_inference.vllm.model_runner import GPUModelRunnerPatch

        GPUModelRunnerPatch.apply_patch()
        WorkerPatch.apply_patch()

        return self._orig_init(*args, **kwargs)


class WorkerPatch(ArcticPatch[Worker]):
    """Fix weights offloading for sleep mode and add drafter support.

    The upstream ``load_model`` context-manager chaining bug was fixed in
    v0.18 (see https://github.com/vllm-project/vllm/pull/32947); we no
    longer patch ``load_model`` here.

    sleep/wake_up: Level-2 sleep discards all GPU weights as designed.
    On wake-up the main model is reloaded from disk via the upstream
    ``reload_weights()`` path.  The drafter (speculative model) is *not*
    covered by ``reload_weights()`` and ``drafter.load_model()`` would
    allocate a new model outside the CuMemAllocator pool, so we save the
    drafter state to CPU during sleep and restore it on wake-up instead.
    The drafter is small so the CPU cost is negligible.

    Note: FP8 quantization replaces ``ModelWeightParameter`` with plain
    ``Parameter`` during ``process_weights_after_loading``, which strips
    the TP-sharding methods that ``reload_weights()`` depends on.
    Level-2 sleep therefore only works for non-FP8 models at present.
    """

    _orig_sleep = Worker.sleep
    _orig_wake_up = Worker.wake_up

    @staticmethod
    def _save_module_state(module) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for name, param in module.named_parameters():
            state[f"param.{name}"] = param.data.cpu().clone()
        for name, buf in module.named_buffers():
            state[f"buffer.{name}"] = buf.cpu().clone()
        return state

    @staticmethod
    def _restore_module_state(
        module, state: dict[str, torch.Tensor]
    ) -> None:
        for name, param in module.named_parameters():
            key = f"param.{name}"
            if key in state:
                param.data.copy_(state[key])
        for name, buf in module.named_buffers():
            key = f"buffer.{name}"
            if key in state:
                buf.data.copy_(state[key])

    def sleep(self, level: int = 1) -> None:
        if level == 2:
            drafter = getattr(self.model_runner, "drafter", None)
            if drafter is not None and getattr(drafter, "model", None) is not None:
                self._sleep_saved_drafter_state = self._save_module_state(
                    drafter.model
                )
            else:
                self._sleep_saved_drafter_state = {}
            self._sleep_level = 2
        self._orig_sleep(level=level)

    def wake_up(self, tags: list[str] | None = None) -> None:
        self._orig_wake_up(tags=tags)

        if getattr(self, "_sleep_level", 0) == 2:
            from arctic_inference.vllm.model_runner import GPUModelRunnerPatch
            GPUModelRunnerPatch._orig_reload_weights(self.model_runner)

            saved_drafter = getattr(self, "_sleep_saved_drafter_state", {})
            if saved_drafter:
                drafter = getattr(self.model_runner, "drafter", None)
                if drafter is not None and getattr(drafter, "model", None) is not None:
                    self._restore_module_state(drafter.model, saved_drafter)
                self._sleep_saved_drafter_state = {}

            self._sleep_level = 0


def apply_arctic_patches():

    from transformers import AutoConfig
    from arctic_inference.common.swiftkv import LlamaSwiftKVConfig

    # Register SwiftKV model configurations to transformers.
    AutoConfig.register("llama_swiftkv", LlamaSwiftKVConfig)

    from vllm import ModelRegistry

    # Register SwiftKV model definitions to vLLM.
    ModelRegistry.register_model(
        "LlamaSwiftKVForCausalLM",
        "arctic_inference.vllm.swiftkv:LlamaSwiftKVForCausalLM")

    # Register ArcticSpeculator models to vLLM.
    from arctic_inference.vllm.spec_dec.arctic_speculator import (
        ArcticMLPSpeculator, ArcticLSTMSpeculator)
    ModelRegistry.register_model("ArcticMLPSpeculatorPreTrainedModel",
                                 ArcticMLPSpeculator)
    ModelRegistry.register_model("ArcticLSTMSpeculatorPreTrainedModel",
                                 ArcticLSTMSpeculator)
    # This name is currently used in corvo
    ModelRegistry.register_model("MLPVariantSpeculatorPreTrainedModel",
                                 ArcticLSTMSpeculator)

    # Patches that make later patches work properly.
    WorkerBasePatch.apply_patch()

    # Async scheduler patches for spec decode (disable_by_batch_size
    # interaction + dynamic draft width allocation).
    AsyncSchedulerPatch.apply_patch()

    # Patches to vLLM arguments and configuration objects.
    EngineArgsPatch.apply_patch()
    AsyncEngineArgsPatch.apply_patch()
    ParallelConfigPatch.apply_patch()
    SpeculativeConfigPatch.apply_patch()
    SpecDecodingStatsPatch.apply_patch()
    SpecDecodingLoggingPatch.apply_patch()
    VllmConfigPatch.apply_patch()
    XgrammarBackendPatch.apply_patch()
    MLPSpeculatorConfigPatch.apply_patch()

    # Forest Cascade Attention backend (always registered; runtime-gated
    # by --forest-cascade-attn-configs).
    apply_forest_cascade_patches()

    # Main optimization patches.
    apply_shift_parallel_patches()

    # FP32 LM head: run the lm_head matmul in fp32 (weights stay in
    # their native dtype, on-the-fly upcast). The patch is always
    # installed but is a no-op unless ARCTIC_FP32_LM_HEAD=1 (or the
    # --fp32-lm-head CLI flag) is set before model construction.
    if envs.ARCTIC_FP32_LM_HEAD:
        set_fp32_lm_head_enabled(True)
    apply_fp32_lm_head_patches()

    # kvcached prefix-cache patches (only when kvcached autopatch is active).
    if os.environ.get("KVCACHED_AUTOPATCH", "").lower() in ("1", "true"):
        from arctic_inference.vllm.kvcached.patches import (
            apply_kvcached_prefix_cache_patches,
        )
        apply_kvcached_prefix_cache_patches()
