"""Integration tests for spec decoding drafter support during sleep/wake.

Loads a real LLM (BF16, no FP8) with Arctic speculative decoding and
verifies that generation produces identical output before and after
level-1 and level-2 sleep/wake cycles.

Why no FP8
----------
Level-2 sleep discards all GPU weights and relies on ``reload_weights()``
to restore them from disk on wake-up.  ``reload_weights()`` uses the
``weight_loader`` / ``weight_loader_v2`` path which calls TP-sharding
methods on the parameter (``load_row_parallel_weight``, etc.).

FP8 quantization's ``process_weights_after_loading`` calls
``replace_parameter()`` (``vllm.model_executor.utils``), which creates a
new plain ``torch.nn.Parameter`` to hold the quantised data.  Although it
copies the ``weight_loader`` *attribute*, the new object is no longer a
``ModelWeightParameter`` subclass, so the class methods like
``load_row_parallel_weight`` are gone.  The next ``reload_weights()``
call therefore crashes with::

    AttributeError: 'Parameter' object has no attribute
        'load_row_parallel_weight'

This is an upstream vLLM v0.14.1 limitation that affects all TP sizes.

Two fixtures are used:
  - ``llm_engine``       : no sleep mode, validates spec dec baseline.
  - ``llm_engine_sleep`` : sleep mode enabled (+ kv_cache_memory_bytes
                           cap to work around CuMemAllocator profiler bug).

Requires GPU(s) and model access.  Marked with ``@pytest.mark.gpu``.
Run with:  python -m pytest tests/unit_tests/test_spec_dec_sleep.py -v -s
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("ARCTIC_INFERENCE_ENABLED", "1")

import vllm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

vllm.plugins.load_general_plugins()

MODEL = "meta-llama/Llama-3.1-70B-Instruct"
SPEC_MODEL = "Snowflake/Arctic-LSTM-Speculator-Llama-3.1-70B-Instruct"

SPEC_CONFIG = {
    "method": "arctic",
    "model": SPEC_MODEL,
    "num_speculative_tokens": 3,
    "enable_suffix_decoding": False,
    "disable_by_batch_size": 64,
}

CONVERSATION = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": "What is 2 + 2?"},
]
SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=64)


def _generate(llm: LLM) -> str:
    outputs = llm.chat(CONVERSATION, sampling_params=SAMPLING_PARAMS)
    return outputs[0].outputs[0].text


# ---------------------------------------------------------------------------
# Fixture 1: no sleep mode — validates spec dec works at all
# ---------------------------------------------------------------------------
@pytest.fixture(scope="class")
def llm_engine():
    """LLM without sleep mode.  Proves spec dec generation is correct."""
    engine = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        speculative_config=SPEC_CONFIG,
        async_scheduling=True,
        max_model_len=16384,
        seed=0,
    )
    yield engine
    del engine


# ---------------------------------------------------------------------------
# Fixture 2: sleep mode enabled
#
# Note: with FP8, enable_sleep_mode caused the KV cache memory profiler
# to over-allocate and OOM (requiring kv_cache_memory_bytes to cap it).
# With BF16 the issue does not reproduce; the root cause is unclear.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="class")
def llm_engine_sleep():
    """LLM with sleep mode for testing sleep/wake cycles."""
    engine = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        speculative_config=SPEC_CONFIG,
        enable_sleep_mode=True,
        async_scheduling=True,
        max_model_len=16384,
        seed=0,
    )
    yield engine
    del engine


@pytest.mark.gpu
class TestSpecDecBaseline:
    """Spec dec works without sleep mode."""

    def test_baseline_generation(self, llm_engine):
        """Engine with spec dec generates non-empty output."""
        text = _generate(llm_engine)
        assert len(text) > 0, "Baseline generation returned empty text"


@pytest.mark.gpu
class TestSpecDecSleep:
    """Sleep/wake cycles preserve generation correctness."""

    def test_baseline_generation(self, llm_engine_sleep):
        """Sanity check: the sleep-mode engine generates non-empty output."""
        text = _generate(llm_engine_sleep)
        assert len(text) > 0, "Baseline generation returned empty text"

    def test_sleep_wake_level1(self, llm_engine_sleep):
        """Level-1 sleep/wake preserves generation correctness."""
        baseline = _generate(llm_engine_sleep)

        llm_engine_sleep.sleep(level=1)
        llm_engine_sleep.wake_up()

        after = _generate(llm_engine_sleep)
        assert after == baseline, (
            f"Output changed after level-1 sleep/wake.\n"
            f"  Before: {baseline!r}\n"
            f"  After:  {after!r}"
        )

    def test_sleep_wake_level2(self, llm_engine_sleep):
        """Level-2 sleep/wake preserves generation correctness."""
        baseline = _generate(llm_engine_sleep)

        llm_engine_sleep.sleep(level=2)
        llm_engine_sleep.wake_up()

        after = _generate(llm_engine_sleep)
        assert after == baseline, (
            f"Output changed after level-2 sleep/wake.\n"
            f"  Before: {baseline!r}\n"
            f"  After:  {after!r}"
        )

    def test_sleep_wake_level2_twice(self, llm_engine_sleep):
        """Two consecutive level-2 cycles still produce correct output."""
        baseline = _generate(llm_engine_sleep)

        for _ in range(2):
            llm_engine_sleep.sleep(level=2)
            llm_engine_sleep.wake_up()

        after = _generate(llm_engine_sleep)
        assert after == baseline, (
            f"Output changed after two level-2 sleep/wake cycles.\n"
            f"  Before: {baseline!r}\n"
            f"  After:  {after!r}"
        )
