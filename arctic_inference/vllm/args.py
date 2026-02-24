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

import argparse
import json
from dataclasses import dataclass, fields

from vllm.config import ParallelConfig
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

from arctic_inference.patching import ArcticPatch
from arctic_inference.vllm.config import ArcticParallelConfig


@dataclass
class ArcticArgs:

    ulysses_sequence_parallel_size: int = 1
    enable_shift_parallel: bool = False
    shift_parallel_threshold: int = 512
    forest_cascade_attn_configs: str | None = None


@dataclass
class ArcticEngineArgs(EngineArgs, ArcticArgs):
    pass


@dataclass
class ArcticAsyncEngineArgs(AsyncEngineArgs, ArcticArgs):
    pass


class EngineArgsPatch(ArcticPatch[EngineArgs]):

    _orig_post_init = EngineArgs.__post_init__
    _orig_add_cli_args = EngineArgs.add_cli_args
    _orig_from_cli_args = EngineArgs.__dict__["from_cli_args"].__wrapped__
    _orig_create_engine_config = EngineArgs.create_engine_config

    def __new__(cls, *args, **kwargs):
        # Override __new__ to return an ArcticEngineArgs instead of an
        # EngineArgs when creating a new instance of the class.
        if cls is EngineArgs:
            return ArcticEngineArgs.__new__(ArcticEngineArgs,
                                            *args, **kwargs)
        return super(EngineArgs, cls).__new__(cls)

    def __post_init__(self):
        # Explicitly set the distributed executor backend if ulysses is enabled
        # since the ulysses parameter is not passed to ParallelConfig.__init__,
        # which leads to the backend being defaulted incorrectly to "uni".
        if (self.ulysses_sequence_parallel_size > 1 and
                self.distributed_executor_backend is None):
            self.distributed_executor_backend = "mp"

        self._orig_post_init()

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        parser = EngineArgsPatch._orig_add_cli_args(parser)
        arctic_group = parser.add_argument_group(
            title="Arctic Inference",
            description="Arctic Inference configuration.",
        )
        arctic_group.add_argument(
            "--ulysses-sequence-parallel-size",
            type=int,
            default=ArcticEngineArgs.ulysses_sequence_parallel_size,
            help="Number of Ulysses sequence parallel replicas",
        )
        arctic_group.add_argument(
            "--enable-shift-parallel",
            action='store_true',
            help='If True, enable shift parallelism.')
        arctic_group.add_argument(
            "--shift-parallel-threshold",
            type=int,
            default=ArcticEngineArgs.shift_parallel_threshold,
            help=("Ulysses sequence parallel if batch size > threshold, "
                  "otherwise tensor parallel across the whole world size"),
        )
        arctic_group.add_argument(
            "--forest-cascade-attn-configs",
            type=str,
            default=None,
            help=('Enable Forest Cascade Attention with a JSON config. '
                  'Example: \'{"max_query_len": 16, "min_group_size": 2, '
                  '"min_additional_prefix_blocks": 1, '
                  '"min_non_singleton_fraction": 0.25, '
                  '"max_non_singleton_groups": 256}\'. '
                  'Pass \'{}\' to enable with all defaults.'),
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        if cls is EngineArgs:
            return EngineArgsPatch._orig_from_cli_args(ArcticEngineArgs, args)
        if cls is AsyncEngineArgs:
            return EngineArgsPatch._orig_from_cli_args(ArcticAsyncEngineArgs,
                                                       args)
        return EngineArgsPatch._orig_from_cli_args(cls, args)

    def create_engine_config(self, *args, **kwargs):
        if (self.ulysses_sequence_parallel_size > 1 and
                self.distributed_executor_backend is None):
            self.distributed_executor_backend = "mp"
        
        # Store ulysses_sequence_parallel_size for access during config initialization
        from arctic_inference.vllm import ulysses
        ulysses._ulysses_sp_size = self.ulysses_sequence_parallel_size
        
        vllm_config = self._orig_create_engine_config(*args, **kwargs)
        # Recreate the parallel config with Arctic parameters since they might
        # not be passed to the parallel config __init__ when first initialized.
        kwargs = {f.name: getattr(vllm_config.parallel_config, f.name)
                  for f in fields(vllm_config.parallel_config) if f.init}
        kwargs["ulysses_sequence_parallel_size"] = (
            self.ulysses_sequence_parallel_size)
        kwargs["enable_shift_parallel"] = self.enable_shift_parallel
        kwargs["shift_parallel_threshold"] = self.shift_parallel_threshold
        vllm_config.parallel_config = ArcticParallelConfig(**kwargs)

        if self.forest_cascade_attn_configs is not None:
            try:
                fca_cfg = json.loads(self.forest_cascade_attn_configs)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"--forest-cascade-attn-configs must be valid JSON: {e}"
                ) from e
            if not isinstance(fca_cfg, dict):
                raise ValueError(
                    "--forest-cascade-attn-configs must be a JSON object"
                )
            vllm_config._forest_cascade_attn_config = fca_cfg
        else:
            vllm_config._forest_cascade_attn_config = None

        return vllm_config


class AsyncEngineArgsPatch(ArcticPatch[AsyncEngineArgs]):

    def __new__(cls, *args, **kwargs):
        # Override __new__ to return an ArcticAsyncEngineArgs instead of an
        # AsyncEngineArgs when creating a new instance of the class.
        if cls is AsyncEngineArgs:
            return ArcticAsyncEngineArgs.__new__(ArcticAsyncEngineArgs,
                                                 *args, **kwargs)
        return super(AsyncEngineArgs, cls).__new__(cls)
