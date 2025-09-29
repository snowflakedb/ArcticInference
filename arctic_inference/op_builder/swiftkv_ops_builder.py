import os
from .builder import CUDAOpBuilder

class SwiftKVOpsBuilder(CUDAOpBuilder):
    def __init__(self):
        super().__init__(name="reshape_and_cache_flash_bulk")

    def absolute_name(self):
        return f'arctic_inference.swiftkv_ops.{self.name}'

    def sources(self):
        return [
            '../csrc/custom_ops/torch_bindings.cpp',
            '../csrc/custom_ops/kernels.cu',
        ]

    def include_paths(self):
        return ['../csrc/custom_ops']
