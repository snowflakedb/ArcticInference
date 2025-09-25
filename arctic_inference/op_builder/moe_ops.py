
import os

from .builder import CUDAOpBuilder, installed_cuda_version


class MoEOpsBuilder(CUDAOpBuilder):
    BUILD_VAR = "AI_BUILD_MOE_OPS"
    NAME = "moe_device_ops"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'arctic_inference.moe_ops.{self.NAME}'

    def is_compatible(self, verbose=False):
        try:
            import torch
        except ImportError:
            if verbose:
                self.warning("Please install torch if trying to pre-compile arctic_inference kernels")
            return False

        cuda_okay = True
        if torch.cuda.is_available():  #ignore-cuda
            sys_cuda_major, _ = installed_cuda_version()
            torch_cuda_major = int(torch.version.cuda.split('.')[0])
            cuda_capability = torch.cuda.get_device_properties(0).major  #ignore-cuda
            if cuda_capability < 6:
                if verbose:
                    self.warning("NVIDIA Inference is only supported on Pascal and newer architectures")
                cuda_okay = False
            if cuda_capability >= 8:
                if torch_cuda_major < 11 or sys_cuda_major < 11:
                    if verbose:
                        self.warning("On Ampere and higher architectures please use CUDA 11+")
                    cuda_okay = False
        return super().is_compatible(verbose) and cuda_okay

    def filter_ccs(self, ccs):
        ccs_retained = []
        ccs_pruned = []
        for cc in [cc.split('.') for cc in ccs]:
            if int(cc[0]) >= 8:
                # Blocked flash has a dependency on Ampere + newer
                ccs_retained.append(cc)
            else:
                ccs_pruned.append(cc)
        if len(ccs_pruned) > 0:
            self.warning(f"Filtered compute capabilities {ccs_pruned}")
        return ccs_retained

    def get_prefix(self):
        ai_path = self._src_path("arctic_inference")
        return "arctic_inference" if os.path.isdir(ai_path) else ".."

    def sources(self):
        sources = [
            "csrc/moe_ops/topk_router.cpp",
            "csrc/moe_ops/topk_router.cu",
            "csrc/moe_ops/moe_apis.cpp"
        ]

        prefix = self.get_prefix()
        sources = [os.path.join(prefix, src) for src in sources]
        return sources

    def extra_ldflags(self):
        return []

    def include_paths(self):
        sources = [
            'csrc/moe_ops/',
        ]

        prefix = self.get_prefix()
        sources = [os.path.join(prefix, src) for src in sources]
        return sources
