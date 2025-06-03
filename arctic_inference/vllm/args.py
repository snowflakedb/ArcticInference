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
from dataclasses import dataclass, fields

from vllm.config import ParallelConfig
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.utils import FlexibleArgumentParser

from arctic_inference.patching import ArcticPatch
from arctic_inference.vllm.config import ArcticParallelConfig


@dataclass
class ArcticArgs:

    ulysses_sequence_parallel_size: int = 1
    enable_shift_parallel: bool = False
    shift_parallel_threshold: int = 512


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
    _orig_is_v1_supported_oracle = EngineArgs._is_v1_supported_oracle

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
            self.distributed
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
        return vllm_config

    def _is_v1_supported_oracle(self, *args, **kwargs):
        orig_speculative_config = self.speculative_config

        # Since Arctic Inference is only compatible with v1 and we already
        # check it earlier, we can just disable this check altogether.
        if (self.speculative_config is not None and
                self.speculative_config.get("method") in ("arctic", "suffix")):
            self.speculative_config = None

        res = self._orig_is_v1_supported_oracle(*args, **kwargs)

        self.speculative_config = orig_speculative_config

        return res


class AsyncEngineArgsPatch(ArcticPatch[AsyncEngineArgs]):

    def __new__(cls, *args, **kwargs):
        # Override __new__ to return an ArcticAsyncEngineArgs instead of an
        # AsyncEngineArgs when creating a new instance of the class.
        if cls is AsyncEngineArgs:
            return ArcticAsyncEngineArgs.__new__(ArcticAsyncEngineArgs,
                                                 *args, **kwargs)
        return super(AsyncEngineArgs, cls).__new__(cls)
