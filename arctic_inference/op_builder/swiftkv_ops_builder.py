import os
from .builder import CUDAOpBuilder

class SwiftKVOpsBuilder(CUDAOpBuilder):
    def __init__(self):
        # Set the name of the operator, this will be the name of the compiled module.
        super().__init__(name="reshape_and_cache_flash_bulk")

    def absolute_name(self):
        # This is the name of the JIT-compiled module.
        return f'arctic_inference.swiftkv_ops.{self.name}'

    def sources(self):
        # List all C++ and CUDA source files.
        # Paths should be relative to the root of the project.
        return [
            '../csrc/custom_ops/torch_bindings.cpp',
            '../csrc/custom_ops/kernels.cu',
        ]

    def include_paths(self):
        # Add the 'csrc' directory to the include path so files like
        # 'custom_ops.h' can be found.
        return ['../csrc/custom_ops']
