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

from typing import Any, Callable, Optional
from dataclasses import field

from vllm.v1.engine.core import EngineCoreProc

ORIG_RUN_ENGINE_CORE = EngineCoreProc.run_engine_core


def run_engine_core(*args, **kwargs):
    # When starting the API server, it will spawn a new process to run the
    # EngineCore. We need to load the plugins in the new process before it
    # initializes the Executor. This function also has to be pickleable so
    # we define it at the top level.
    import vllm
    vllm.plugins.load_general_plugins()
    return ORIG_RUN_ENGINE_CORE(*args, **kwargs)


def arctic_inference_plugin():

    from transformers import AutoConfig
    from arctic_inference.common.swiftkv import LlamaSwiftKVConfig

    # Register SwiftKV model configurations to transformers.
    AutoConfig.register("llama_swiftkv", LlamaSwiftKVConfig)

    from vllm import ModelRegistry
    from arctic_inference.vllm.swiftkv import LlamaSwiftKVForCausalLM

    # Register SwiftKV model definitions to vLLM.
    ModelRegistry.register_model("LlamaSwiftKVForCausalLM",
                                 LlamaSwiftKVForCausalLM)

    from vllm.v1.worker.worker_base import WorkerBase

    # We need to replace vllm's GPUModelRunner with our own, but we do the
    # monkey-patch from the WorkerBase init to avoid importing CUDA libraries
    # before the process fork.
    @replace_func(WorkerBase, "__init__")
    def __init__(orig, *args, **kwargs):
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        # Replace the __new__ method of GPUModelRunner to return our
        # ArcticGPUModelRunner instead of the original GPUModelRunner.
        def __new__(cls, *args, **kwargs):
            from arctic_inference.vllm.model_runner import ArcticGPUModelRunner
            if cls is GPUModelRunner:
                return ArcticGPUModelRunner.__new__(ArcticGPUModelRunner,
                                                    *args, **kwargs)
            return super(GPUModelRunner, cls).__new__(cls)

        GPUModelRunner.__new__ = __new__

        return orig(*args, **kwargs)


    # Ulysses Parameter
    # SP = 4

    import vllm

    from vllm.engine.arg_utils import EngineArgs
    EngineArgs.sequence_parallel_size = 1
    @replace_func(EngineArgs, "__init__")
    def __init__(orig, *args, **kwargs):
        print("monkeypatch EngineArgs __init__")
        EngineArgs.sequence_parallel_size = kwargs.pop("sequence_parallel_size", 1)
        return orig(*args, **kwargs)

    @replace_func(EngineArgs, "add_cli_args")
    def add_cli_args(orig, parser):
        print("monkeypatch EngineArgs add_cli_args")
        parser.add_argument('--sequence-parallel-size',
                            '-sp',
                            type=int,
                            default=EngineArgs.sequence_parallel_size,
                            help='Number of sequence parallel replicas.')
        return orig(parser)

    from vllm.engine.arg_utils import AsyncEngineArgs
    @replace_func(AsyncEngineArgs, "from_cli_args")
    def from_cli_args(orig, args):
        print("monkeypatch AsyncEngineArgs from_cli_args")
        engine_args = orig(args)
        EngineArgs.sequence_parallel_size = args.sequence_parallel_size
        return engine_args

    from vllm.config import ParallelConfig
    @replace_func(ParallelConfig, "__init__")
    def __init__(orig, self, *args, **kwargs):
        print(f"monkeypatch ParallelConfig __init__")
        # self.sequence_parallel_size = SP
        self.sequence_parallel_size = EngineArgs.sequence_parallel_size
        return orig(self, *args, **kwargs)

    @replace_func(ParallelConfig, "__post_init__")
    def __post_init__(orig, self) -> None:
        print("monkeypatch ParallelConfig __post_init__")
        self.world_size = self.pipeline_parallel_size * \
            self.tensor_parallel_size * \
            self.sequence_parallel_size

        self.data_parallel_size = vllm.envs.VLLM_DP_SIZE
        self.data_parallel_rank = vllm.envs.VLLM_DP_RANK
        self.data_parallel_master_ip = vllm.envs.VLLM_DP_MASTER_IP
        self.data_parallel_master_port = vllm.envs.VLLM_DP_MASTER_PORT
        self.world_size_across_dp = self.world_size * self.data_parallel_size

        if self.distributed_executor_backend == "external_launcher":
            import os
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            vllm.config.logger.info("Disabling V1 multiprocessing for external launcher.")

        ray_only_devices = ["tpu"]
        from vllm.platforms import current_platform
        if (current_platform.device_type in ray_only_devices
                and self.world_size > 1):
            if self.distributed_executor_backend is None:
                self.distributed_executor_backend = "ray"
            if self.distributed_executor_backend != "ray":
                raise ValueError(
                    f"{current_platform.device_type.upper()} backend only "
                    "supports Ray for distributed inference.")

        if self.distributed_executor_backend is None and self.world_size > 1:
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.

            from vllm.executor import ray_utils
            backend = "mp"
            ray_found = ray_utils.ray_is_available()
            if current_platform.is_neuron():
                # neuron uses single process to control multiple devices
                backend = "uni"
            elif (current_platform.is_cuda()
                  and vllm.utils.cuda_device_count_stateless() < self.world_size):
                if not ray_found:
                    raise ValueError("Unable to load Ray which is "
                                     "required for multi-node inference, "
                                     "please install Ray with `pip install "
                                     "ray`.") from ray_utils.ray_import_err
                backend = "ray"
            elif ray_found:
                if self.placement_group:
                    backend = "ray"
                else:
                    from ray import is_initialized as ray_is_initialized
                    if ray_is_initialized():
                        from ray.util import get_current_placement_group
                        if get_current_placement_group():
                            backend = "ray"
            self.distributed_executor_backend = backend
            vllm.config.logger.info("Defaulting to use %s for distributed inference",
                        backend)

        if self.distributed_executor_backend is None and self.world_size == 1:
            self.distributed_executor_backend = "uni"

        self._verify_args()
        
    from vllm.config import VllmConfig
    @replace_func(VllmConfig, "__str__")
    def __str__(orig, self, *args, **kwargs):
        print("monkeypatch VllmConfig __str__")
        string = orig(self, *args, **kwargs)
        string += f", sequence_parallel_size={self.parallel_config.sequence_parallel_size}"
        return string
    
    from vllm.config import ModelConfig
    @replace_func(ModelConfig, "get_num_kv_heads")
    def get_num_kv_heads(orig, self, parallel_config: "ParallelConfig") -> int:
        print("monkeypatch ModelConfig get_num_kv_heads")
        """Returns the number of KV heads per GPU."""
        if self.use_mla:
            # When using MLA during decode it becomes MQA
            return 1
        total_num_kv_heads = self.get_total_num_kv_heads()
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(
            1, total_num_kv_heads // (parallel_config.tensor_parallel_size *
                                      parallel_config.sequence_parallel_size))
    
    @replace_func(ModelConfig, "get_num_attention_heads")
    def get_num_attention_heads(orig, self,
                                parallel_config: "ParallelConfig") -> int:
        print("monkeypatch ModelConfig get_num_attention_heads_heads")
        num_heads = getattr(self.hf_text_config, "num_attention_heads", 0)
        return num_heads // (parallel_config.tensor_parallel_size *
                             parallel_config.sequence_parallel_size)
    
    @replace_func(ModelConfig, "get_layers_start_end_indices")
    def get_layers_start_end_indices(orig,
            self, parallel_config: "ParallelConfig") -> tuple[int, int]:
        print("monkeypatch ModelConfig get_layers_start_end_indices")
        from vllm.distributed.utils import get_pp_indices
        if self.hf_text_config.model_type == "deepseek_mtp":
            total_num_hidden_layers = getattr(self.hf_text_config,
                                              "num_nextn_predict_layers", 0)
        else:
            total_num_hidden_layers = getattr(self.hf_text_config,
                                              "num_hidden_layers", 0)
        # the layout order is: DP x PP x SP x TP
        pp_rank = (parallel_config.rank //
                   (parallel_config.tensor_parallel_size *
                    parallel_config.sequence_parallel_size)
                   ) % parallel_config.pipeline_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
        return start, end

    from vllm.distributed.parallel_state import init_model_parallel_group, get_world_group
    import torch
    vllm.distributed.parallel_state._SP = None
    @replace_func(vllm.distributed.parallel_state, "initialize_model_parallel")
    def initialize_model_parallel(
        orig,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        backend: Optional[str] = None,
    ) -> None:
        print("monkeypatch parallel_state initialize_model_parallel")
        """
        Initialize model parallel groups.

        Arguments:
            tensor_model_parallel_size: number of GPUs used for tensor model
                parallelism.
            pipeline_model_parallel_size: number of GPUs used for pipeline model
                parallelism.

        Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
        use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
        the model pipeline. The present function will
        create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
            4 tensor model-parallel groups:
                [g0, g1], [g2, g3], [g4, g5], [g6, g7]
            2 pipeline model-parallel groups:
                [g0, g2, g4, g6], [g1, g3, g5, g7]
        Note that for efficiency, the caller should make sure adjacent ranks
        are on the same DGX box. For example if we are using 2 DGX-1 boxes
        with a total of 16 GPUs, rank 0 to 7 belong to the first box and
        ranks 8 to 15 belong to the second box.
        """
        # Get world size and rank. Ensure some consistencies.
        assert torch.distributed.is_initialized()
        world_size: int = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        backend = backend or torch.distributed.get_backend(
            get_world_group().device_group)

        data_parallel_size = 1
        has_external_dp = False
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        sequence_parallel_size = \
            config.parallel_config.sequence_parallel_size
        if config is not None:
            if config.parallel_config.world_size != world_size:
                # detect external data parallelism.
                # dp in vllm means all dp instances need to run together.
                # if the world size does not match, it means this dp is external,
                # and the dp instances can run independently, e.g. in rlhf workflow
                # from https://github.com/volcengine/verl .
                # in that case, we treat the rest dimensions as if they are
                # data parallel, and create a dummy dp group that is not used.
                data_parallel_size = world_size // (pipeline_model_parallel_size *
                                                    sequence_parallel_size *
                                                    tensor_model_parallel_size)
                has_external_dp = True
            else:
                data_parallel_size = config.parallel_config.data_parallel_size

        # the layout order is: DP x PP x SP x TP
        # to get group_ranks for each dimension, transpose that dimension to the
        # last dimension, then reshape to 2D, then unbind the last dimension
        all_ranks = torch.arange(world_size).reshape(
            data_parallel_size, pipeline_model_parallel_size,
            sequence_parallel_size, tensor_model_parallel_size)  # noqa

        from vllm.distributed.parallel_state import _TP, _PP, _SP, _DP
        # Build the tensor model-parallel groups.
        assert _TP is None, ("tensor model parallel group is already initialized")
        group_ranks = []
        for i in range(world_size // tensor_model_parallel_size):
            ranks = list(
                range(i * tensor_model_parallel_size,
                    (i + 1) * tensor_model_parallel_size))
            group_ranks.append(ranks)
        _TP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="tp")

        # Build the pipeline model-parallel groups.
        assert _PP is None, (
            "pipeline model parallel group is already initialized")
        group_ranks = all_ranks.transpose(1, 2).reshape(
            -1, pipeline_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        _PP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="pp")

        # Build the sequence parallel groups.
        ulysses_model_parallel_size = tensor_model_parallel_size \
            * sequence_parallel_size
        assert _SP is None, (
            "sequence parallel group is already initialized")
        group_ranks = []
        for i in range(pipeline_model_parallel_size):
            for j in range(tensor_model_parallel_size):
                ranks = list(
                    range(i * ulysses_model_parallel_size + j,
                        (i + 1) * ulysses_model_parallel_size + j,
                        tensor_model_parallel_size))
                group_ranks.append(ranks)
        _SP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="sp")

        assert _DP is None, ("data parallel group is already initialized")
        group_ranks = all_ranks.transpose(0,
                                        2).reshape(-1,
                                                    data_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        if has_external_dp:
            # create a dummy dp group that is not used actually,
            # since this dp is external.
            # a dummy dp group means every rank is a group itself.
            # this way, no communication is needed, no memory is wasted.
            group_ranks = [[x] for x in range(world_size)]
        _DP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="dp")

        vllm.distributed.parallel_state.logger.info(
            "rank %s in world size %s is assigned as "
            "DP rank %s, PP rank %s, SP rank %s, TP rank %s", rank,
            world_size, _DP.rank_in_group, _PP.rank_in_group, _SP.rank_in_group,
            _TP.rank_in_group)
        
        vllm.distributed.parallel_state._TP = _TP
        vllm.distributed.parallel_state._PP = _PP
        vllm.distributed.parallel_state._SP = _SP
        vllm.distributed.parallel_state._DP = _DP

    EngineCoreProc.run_engine_core = run_engine_core

    from vllm.v1.executor.multiproc_executor import MultiprocExecutor
    import weakref
    import signal
    import psutil
    from vllm.executor.multiproc_worker_utils import set_multiprocessing_worker_envs
    from vllm.utils import get_distributed_init_method, get_open_port
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
    from vllm.v1.executor.multiproc_executor import WorkerProc, WorkerProcHandle
    @replace_func(MultiprocExecutor, "_init_executor")
    def _init_executor(orig, self) -> None:
        print("monkeypatch MultiprocExecutor _init_executor")
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)

        # The child processes will send SIGUSR1 when unrecoverable
        # errors happen.
        def sigusr1_handler(signum, frame):
            vllm.v1.executor.multiproc_executor.logger.fatal(
                "MulitprocExecutor got fatal signal from worker processes, "
                "shutting down. See stack trace above for root cause issue.")
            # Propagate error up to parent process.
            parent_process = psutil.Process().parent()
            parent_process.send_signal(signal.SIGUSR1)
            self.shutdown()

        signal.signal(signal.SIGUSR1, sigusr1_handler)

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        sequence_parallel_size = self.parallel_config.sequence_parallel_size
        assert self.world_size == tensor_parallel_size \
            * sequence_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size * sequence_parallel_size "
            f"({tensor_parallel_size * sequence_parallel_size}). "
            f"Pipeline parallelism is not yet implemented in v1")

        # Set multiprocessing envs that are common to V0 and V1
        set_multiprocessing_worker_envs(self.parallel_config)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # 127.0.0.1 for communication.
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port())

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        self.rpc_broadcast_mq = MessageQueue(self.world_size, self.world_size)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Create workers
        self.workers: list[WorkerProcHandle] = []
        for rank in range(self.world_size):
            worker = WorkerProc.make_worker_process(self.vllm_config, rank,
                                                    rank,
                                                    distributed_init_method,
                                                    scheduler_output_handle)
            self.workers.append(worker)

        # Ensure message queues are ready. Will deadlock if re-ordered
        # Must be kept consistent with the WorkerProc
        self.rpc_broadcast_mq.wait_until_ready()
        for w in self.workers:
            w.worker_response_mq.wait_until_ready()
    
    from vllm.model_executor.models.llama import LlamaForCausalLM
    from typing import Union
    from vllm.sequence import IntermediateTensors
    @replace_func(LlamaForCausalLM, "forward")
    def forward(
        orig,
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Ulysses
        N = input_ids.shape[0]
        SP = vllm.distributed.parallel_state._SP.world_size
        SP_rank = vllm.distributed.parallel_state._SP.rank_in_group
        device_group = vllm.distributed.parallel_state._SP.device_group
        N_ulysses = N // SP
        N_offset = N_ulysses * SP_rank
        # narrow the input
        input_ids[:N_ulysses] = input_ids[N_offset:N_offset + N_ulysses]
        positions[:N_ulysses] = positions[N_offset:N_offset + N_ulysses]
        # model forward
        output = self.model(input_ids[:N_ulysses], positions[:N_ulysses],
                            intermediate_tensors, inputs_embeds)
        # all-gather model_output
        model_output = torch.empty((N, self.config.hidden_size),
                                   dtype=output.dtype,
                                   device=output.device)
        torch.distributed.all_gather_into_tensor(
            model_output, output, group=device_group)
        return model_output
    
    from vllm.model_executor.models.llama import Qwen2ForCausalLM
    @replace_func(LlamaForCausalLM, "forward")
    def forward(
        orig,
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Ulysses
        N = input_ids.shape[0]
        SP = vllm.distributed.parallel_state._SP.world_size
        SP_rank = vllm.distributed.parallel_state._SP.rank_in_group
        device_group = vllm.distributed.parallel_state._SP.device_group
        N_ulysses = N // SP
        N_offset = N_ulysses * SP_rank
        # narrow the input
        input_ids[:N_ulysses] = input_ids[N_offset:N_offset + N_ulysses]
        positions[:N_ulysses] = positions[N_offset:N_offset + N_ulysses]
        # model forward
        output = self.model(input_ids[:N_ulysses], positions[:N_ulysses],
                            intermediate_tensors, inputs_embeds)
        # all-gather model_output
        model_output = torch.empty((N, self.config.hidden_size),
                                   dtype=output.dtype,
                                   device=output.device)
        torch.distributed.all_gather_into_tensor(
            model_output, output, group=device_group)
        return model_output
    
    from vllm.attention.layer import Attention
    @replace_func(Attention, "__init__")
    def __init__(orig, self, num_heads, *args, **kwargs) -> None:
        self.SP = vllm.distributed.parallel_state._SP.world_size
        num_heads //= self.SP
        kwargs["num_kv_heads"] //= self.SP
        return orig(self, num_heads, *args, **kwargs)
    @replace_func(Attention, "forward")
    def forward(orig, self, query, key, value, **kwargs):
        from vllm.forward_context import get_forward_context
        if self.calculate_kv_scales:
            attn_metadata = get_forward_context().attn_metadata
            if attn_metadata.enable_kv_scales_calculation:
                self.calc_kv_scales(key, value)
        hidden_size = query.shape[-1]
        output = torch.empty_like(query)
        torch.ops.vllm.unified_attention_with_output(
                    query, key, value, output, self.layer_name)
        return output.view(-1, hidden_size)
    
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    @replace_func(FlashAttentionImpl, "__init__")
    def __init__(orig, self, *args, **kwargs):
        self.SP = vllm.distributed.parallel_state._SP.world_size
        self.device_group = vllm.distributed.parallel_state._SP.device_group
        return orig(self, *args, **kwargs)

    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
    @replace_func(FlashAttentionImpl, "forward")
    def forward(orig, self, layer, query, key, value, kv_cache, attn_metadata, output):
        qkv = torch.cat(
            (query.view(-1, self.SP, self.num_heads * self.head_size),
             key.view(-1, self.SP, self.num_kv_heads * self.head_size),
             value.view(-1, self.SP, self.num_kv_heads * self.head_size)),
            dim=-1).transpose(0, 1).reshape(
                -1, (self.num_heads + 2 * self.num_kv_heads) * self.head_size)
        # all-to-all
        qkv_ = torch.empty_like(qkv)
        torch.distributed.all_to_all_single(qkv_, qkv, group=self.device_group)
        # unpack
        q_, k_, v_ = qkv_.split([
            self.num_heads * self.head_size, self.num_kv_heads *
            self.head_size, self.num_kv_heads * self.head_size
        ], dim=-1)
        # prepare
        q_ = q_.reshape(-1, self.num_heads, self.head_size)
        k_ = k_.reshape(-1, self.num_kv_heads, self.head_size)
        v_ = v_.reshape(-1, self.num_kv_heads, self.head_size)
        c_ = output.view(-1, self.num_heads, self.head_size)
        # original attention
        orig(self, layer, q_, k_, v_, kv_cache, attn_metadata, c_)
        # Ulysses all-to-all 2/2
        c = torch.empty_like(c_)
        torch.distributed.all_to_all_single(c, c_, group=self.device_group)
        output.copy_(
            torch.transpose(
                c.view(self.SP, -1, self.num_heads * self.head_size), 0,
                1).reshape(-1, self.num_heads * self.SP * self.head_size))
        return output
    

def replace_func(obj: Any, name: str,
                 func: Optional[Callable] = None) -> Callable:
    """
    Replace a method on a class/object with a wrapped version of the original.
    It works in two modes:

    As a decorator:

    ```
    @replace_func(SomeClass, 'some_method')
    def new_method(original_method, *args, **kwargs):
        # Optionally modify behavior using original_method
        return original_method(*args, **kwargs)
    ```

    As a direct function call:

    ```
    def new_method(original_method, *args, **kwargs):
        # Optionally modify behavior using original_method
        return original_method(*args, **kwargs)

    replace_func(SomeClass, 'some_method', new_method)
    ```

    Args:
        obj (Any): The object whose method is to be replaced.
        name (str): The name of the method to replace.
        func (Callable, optional): A function that wraps the original method.
            `func` should accept the original method as its first parameter.
            If not provided, the replacement is set up for decorator usage.

    Returns:
         Callable: The provided function after wrapping, which can be used as
            the new method.
    """
    orig_fn = getattr(obj, name)

    if func is None:
        # Use as a decorator

        def decorator(func):

            def wrapper(*args, **kwargs):
                return func(orig_fn, *args, **kwargs)

            setattr(obj, name, wrapper)
            return wrapper
        return decorator

    # Use as a function call

    def wrapper(*args, **kwargs):
        return func(orig_fn, *args, **kwargs)

    setattr(obj, name, wrapper)
    return wrapper
