import os
from .builder import CUDAOpBuilder

class SwiftKVOpsBuilder(CUDAOpBuilder):
    def __init__(self):
        super().__init__(name="reshape_and_cache_flash_bulk")

    def absolute_name(self):
        return f'arctic_inference.swiftkv_ops.{self.name}'

    def get_prefix(self):
        # borrowed from moe_op. refactor later
        ai_path = self._src_path("arctic_inference")
        return "arctic_inference" if os.path.isdir(ai_path) else ".."

    def sources(self):
        return [
            'csrc/custom_ops/torch_bindings.cpp',
            'csrc/custom_ops/kernels.cu',
        ]
        prefix = self.get_prefix()
        sources = [os.path.join(prefix, src) for src in sources]
        return sources

    def include_paths(self):
        return ['csrc/custom_ops']
        prefix = self.get_prefix()
        sources = [os.path.join(prefix, src) for src in sources]
        return sources
