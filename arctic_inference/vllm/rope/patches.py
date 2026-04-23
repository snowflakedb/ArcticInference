# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Patches to wire multi-cache dynamic NTK RoPE into vLLM's get_rope dispatch.

We cannot add a new branch to
:func:`vllm.model_executor.layers.rotary_embedding.get_rope` directly
because that function lives upstream.  Instead we wrap it: our wrapper
handles ``rope_type`` values that the multi-cache subsystem owns and
delegates everything else to the original.  The wrapper reuses vLLM's
existing :data:`_ROPE_DICT` memoization so repeated calls with the
same key return a singleton.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

#: rope_type values we dispatch to :class:`MultiCacheDynamicNTKRotaryEmbedding`.
_MULTI_CACHE_ROPE_TYPES = frozenset(
    [
        "multi_cache_ntk",
        # Short alias, reads as "the multi-cache variant of dynamic".
        "dynamic_multi_cache",
    ]
)


def _build_multi_cache_ntk(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool,
    rope_parameters: dict[str, Any],
    dtype: torch.dtype,
):
    """Build a :class:`MultiCacheDynamicNTKRotaryEmbedding` from rope params.

    Supported ``rope_parameters`` keys (all optional):

    * ``factors``: list of scaling factors (e.g. ``[1, 2, 3, 4, 5, 6]``).
      Defaults to
      :data:`arctic_inference.vllm.rope.multi_cache_ntk.DEFAULT_FACTORS`.
    * ``original_max_position_embeddings``: the trained context (``m``).
      Each factor ``F`` gets a cache spanning ``ceil(F * m)`` positions.
      Defaults to ``max_position`` when omitted.

    The trailing ``factor`` (singular) key from legacy dynamic configs is
    folded into ``factors`` as ``[factor]`` when present and ``factors``
    is not explicitly set.  This keeps backward compat with configs
    written for the old static ``dynamic`` rope.

    ``max_num_batched_tokens`` is looked up from the active
    :class:`vllm.config.VllmConfig` to size the runtime offset buffer.
    The module falls back to a conservative default when no config is
    active (typical in unit tests).
    """
    from arctic_inference.vllm.rope.multi_cache_ntk import (
        DEFAULT_FACTORS,
        MultiCacheDynamicNTKRotaryEmbedding,
    )

    orig_max = rope_parameters.get(
        "original_max_position_embeddings", max_position
    )

    # Resolve the factor list: explicit ``factors`` wins; otherwise
    # fall back to a single-value list from ``factor``; otherwise
    # defaults.  This ordering is deliberate so that configs that
    # already set ``factors`` are not second-guessed.
    if "factors" in rope_parameters:
        raw_factors = list(rope_parameters["factors"])
    elif "factor" in rope_parameters:
        raw_factors = [float(rope_parameters["factor"])]
    else:
        raw_factors = list(DEFAULT_FACTORS)

    max_num_batched_tokens: int | None = None
    try:
        from vllm.config import get_current_vllm_config_or_none

        cfg = get_current_vllm_config_or_none()
        if (
            cfg is not None
            and getattr(cfg, "scheduler_config", None) is not None
        ):
            max_num_batched_tokens = int(
                cfg.scheduler_config.max_num_batched_tokens
            )
    except Exception:  # pragma: no cover - defensive
        max_num_batched_tokens = None

    return MultiCacheDynamicNTKRotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=int(orig_max),
        base=base,
        is_neox_style=is_neox_style,
        dtype=dtype,
        factors=raw_factors,
        max_num_batched_tokens=max_num_batched_tokens,
    )


def _install_apply_dict_overrides_patch() -> None:
    """Re-run ``patch_rope_parameters`` after dict-valued hf_overrides apply.

    In vLLM 0.14.1, :meth:`ModelConfig.__post_init__` splits its
    ``hf_overrides`` into *flat* values (passed into ``get_config``, which
    runs ``patch_rope_parameters``) and *dict* values (applied via
    ``_apply_dict_overrides`` *after* the HF config has already been
    loaded and patched).  ``rope_scaling`` is a dict, so it always lands
    in the second bucket.  The consequence is that
    ``config.rope_parameters`` never picks up the override, so the
    scaling factor in ``_get_and_verify_max_len`` stays at ``1.0`` and
    ``derived_max_model_len`` stays at the unscaled ``max_position_embeddings``.

    We wrap ``_apply_dict_overrides`` to invoke ``patch_rope_parameters``
    on the target config (and its text sub-config) after the dict values
    land, which re-materializes ``rope_parameters`` from the just-written
    ``rope_scaling``.  Idempotent; a second install is a no-op.
    """
    from vllm.config.model import ModelConfig
    from vllm.transformers_utils.config import patch_rope_parameters

    orig = ModelConfig._apply_dict_overrides
    if getattr(orig, "_arctic_wrapped", False):
        return

    def _apply_dict_overrides_arctic(self, config, overrides):
        orig(self, config, overrides)
        if "rope_scaling" not in overrides:
            return
        patch_rope_parameters(config)
        try:
            text_cfg = config.get_text_config()
        except Exception:  # pragma: no cover - non-text configs
            text_cfg = None
        if text_cfg is not None and text_cfg is not config:
            patch_rope_parameters(text_cfg)

    _apply_dict_overrides_arctic._arctic_wrapped = True  # type: ignore[attr-defined]
    _apply_dict_overrides_arctic._arctic_original = orig  # type: ignore[attr-defined]
    ModelConfig._apply_dict_overrides = _apply_dict_overrides_arctic  # type: ignore[assignment]


def _install_get_rope_patch() -> None:
    """Install the ``get_rope`` wrapper.

    Idempotent: installing twice is a no-op.  The wrapper is attached as
    ``_arctic_wrapped`` so we can detect it on subsequent imports.
    """
    import vllm.model_executor.layers.rotary_embedding as rope_mod

    orig = rope_mod.get_rope
    if getattr(orig, "_arctic_wrapped", False):
        return

    def get_rope_arctic(
        head_size: int,
        max_position: int,
        is_neox_style: bool = True,
        rope_parameters: dict[str, Any] | None = None,
        dtype: torch.dtype | None = None,
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ):
        # Optionally promote legacy dynamic rope configs to multi-cache.
        rope_parameters = maybe_promote_rope_parameters(rope_parameters)

        scaling_type = None
        if rope_parameters is not None:
            scaling_type = rope_parameters.get("rope_type")

        if scaling_type not in _MULTI_CACHE_ROPE_TYPES:
            return orig(
                head_size=head_size,
                max_position=max_position,
                is_neox_style=is_neox_style,
                rope_parameters=rope_parameters,
                dtype=dtype,
                dual_chunk_attention_config=dual_chunk_attention_config,
            )

        if dual_chunk_attention_config is not None:
            raise ValueError(
                "multi_cache_ntk rope is incompatible with "
                "dual_chunk_attention_config"
            )

        if dtype is None:
            dtype = torch.get_default_dtype()
        assert rope_parameters is not None

        base = rope_parameters.get("rope_theta", 10000)
        partial_rotary_factor = rope_parameters.get(
            "partial_rotary_factor", 1.0,
        )
        if partial_rotary_factor <= 0.0 or partial_rotary_factor > 1.0:
            raise ValueError(
                f"{partial_rotary_factor=} must be between 0.0 and 1.0"
            )
        rotary_dim = int(head_size * partial_rotary_factor)

        # Reuse vLLM's cache so repeated calls for the same layer family
        # get the same singleton module.
        rope_parameters_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in rope_parameters.items()
        }
        rope_parameters_args = tuple(rope_parameters_tuple.items())
        key = (
            head_size,
            rotary_dim,
            max_position,
            is_neox_style,
            rope_parameters_args,
            None,  # dual_chunk_attention_args
            dtype,
        )
        cache = rope_mod._ROPE_DICT
        if key in cache:
            return cache[key]

        module = _build_multi_cache_ntk(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=base,
            is_neox_style=is_neox_style,
            rope_parameters=rope_parameters,
            dtype=dtype,
        )
        cache[key] = module
        logger.info(
            "ArcticInference installed MultiCacheDynamicNTKRotaryEmbedding "
            "(head_size=%d, rotary_dim=%d, factors=%s, orig_max=%s, "
            "max_position=%s, runtime_buffer_size=%s, total_cache_len=%s)",
            head_size,
            rotary_dim,
            module.factors,
            rope_parameters.get(
                "original_max_position_embeddings", max_position
            ),
            max_position,
            module._runtime_buffer_size,
            module._total_cache_len,
        )
        return module

    get_rope_arctic._arctic_wrapped = True  # type: ignore[attr-defined]
    get_rope_arctic._arctic_original = orig  # type: ignore[attr-defined]
    rope_mod.get_rope = get_rope_arctic

    # Also patch the re-export path used by some model implementations
    # that import ``get_rope`` from the layers package root.
    try:
        import vllm.model_executor.layers as layers_mod

        if hasattr(layers_mod, "get_rope"):
            layers_mod.get_rope = get_rope_arctic
    except Exception:  # pragma: no cover - defensive
        pass


# --------------------------------------------------------------------------
# Config promotion helper
# --------------------------------------------------------------------------

_ARCTIC_MULTI_CACHE_ROPE_ENV = "ARCTIC_INFERENCE_MULTI_CACHE_ROPE"


def multi_cache_rope_enabled() -> bool:
    """Return True if ArcticInference should promote dynamic rope to multi-cache.

    Controlled by the env var ``ARCTIC_INFERENCE_MULTI_CACHE_ROPE``:
    ``"1"`` enables, anything else (or unset) disables.
    """
    return os.getenv(_ARCTIC_MULTI_CACHE_ROPE_ENV, "0") == "1"


def maybe_promote_rope_parameters(
    rope_parameters: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Rewrite ``rope_parameters["rope_type"]`` to the multi-cache variant.

    Only fires when :func:`multi_cache_rope_enabled` returns True and the
    incoming ``rope_type`` is the static-dynamic alias.  The rewrite is
    a shallow copy so the caller's dict is not mutated.  Users who want
    to opt-in per-model can set ``rope_type="multi_cache_ntk"`` directly
    in the HF config.

    If the incoming config specifies a single ``factor`` but no
    ``factors``, the promoter leaves it alone and
    :func:`_build_multi_cache_ntk` will fold it into a single-bucket
    multi-cache (degenerate but valid).  To activate the full default
    factor set, the caller should set ``factors`` explicitly.  The
    legacy "alpha" variant is not promoted because it uses a different
    base formula.
    """
    if rope_parameters is None or not multi_cache_rope_enabled():
        return rope_parameters
    rope_type = rope_parameters.get("rope_type")
    if rope_type in ("dynamic",):
        if "alpha" in rope_parameters and "factor" not in rope_parameters:
            # The alpha variant has no multi-cache analogue; leave it.
            return rope_parameters
        promoted = dict(rope_parameters)
        promoted["rope_type"] = "multi_cache_ntk"
        logger.info(
            "ArcticInference: promoting rope_type='dynamic' to "
            "'multi_cache_ntk' (factors=%s)",
            promoted.get("factors", promoted.get("factor", "default")),
        )
        return promoted
    return rope_parameters
