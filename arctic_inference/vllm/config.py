from dataclasses import dataclass

from vllm.config import ParallelConfig, VllmConfig

from arctic_inference.patching import ArcticPatch
from arctic_inference.vllm.args import get_current_arctic_args


@dataclass
class ArcticParallelConfig(ParallelConfig):

    sequence_parallel_size: int = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        arctic_args = get_current_arctic_args()
        self.sequence_parallel_size = arctic_args.sequence_parallel_size


class ParallelConfigPatch(ArcticPatch[ParallelConfig]):

    def __new__(cls, *args, **kwargs):
        # Override __new__ to return an ArcticParallelConfig instead of a
        # ParallelConfig when creating a new instance of the class.
        if cls is ParallelConfig:
            return ArcticParallelConfig.__new__(ArcticParallelConfig,
                                                *args, **kwargs)
        return super(ParallelConfig, cls).__new__(cls)

    @property
    def world_size(self) -> int:
        return (self.pipeline_parallel_size *
                self.tensor_parallel_size *
                self.sequence_parallel_size)

    @world_size.setter
    def world_size(self, value: int) -> None:
        # ParallelConfig.__post_init__ will assign world_size to PP * TP, while
        # we want PP * TP * SP to be the world size. So we define world_size as
        # a property with a no-op setter to ignore the value later assigned by
        # ParallelConfig.__post_init__.
        pass


class VllmConfigPatch(ArcticPatch[VllmConfig]):

    _orig_str = VllmConfig.__str__

    def __str__(self, *args, **kwargs):
        string = self._orig_str(*args, **kwargs)
        string += f", sequence_parallel_size={self.parallel_config.sequence_parallel_size}"
        return string
