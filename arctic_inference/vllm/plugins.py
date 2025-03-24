from typing import Any, Callable, Optional


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
