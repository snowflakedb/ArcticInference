"""Weight synchronization via pipelined NCCL send/recv or shared memory.

Public API
----------
- :class:`NCCLEngine`           — low-level pipelined NCCL transfer
- :class:`WeightSender`         — sender-side manager (wraps NCCLEngine)
- :class:`WeightSyncExtension`  — vLLM worker extension (receiver)
- :class:`TransferSchedule`     — static topology planner
- :class:`TransferGroup`        — one independent NCCL transfer group
- :class:`WeightInfo`           — lightweight parameter descriptor
- :func:`build_weights_info`    — build descriptors from safetensors
- :func:`save_weights_to_shm`   — sender-side shared-memory transfer (colocated)
- :func:`load_weights_from_shm` — receiver-side shared-memory transfer (colocated)
"""

from arctic_inference.server.weight_sync.utils import (
    WeightInfo,
    build_weights_info,
    load_spec_checkpoint,
    spec_bucket_size,
    _DirectParamWriter,
    _FP8InplaceUpdater,
)
from arctic_inference.server.weight_sync.engine import NCCLEngine
from arctic_inference.server.weight_sync.sender import WeightSender, send_spec_weights
from arctic_inference.server.weight_sync.receiver import WeightSyncExtension
from arctic_inference.server.weight_sync.schedule import TransferSchedule, TransferGroup
from arctic_inference.server.weight_sync.ipc_engine import (
    save_weights_to_shm,
    load_weights_from_shm,
    cleanup_shm,
)

__all__ = [
    "NCCLEngine",
    "WeightSender",
    "WeightSyncExtension",
    "TransferSchedule",
    "TransferGroup",
    "WeightInfo",
    "build_weights_info",
    "load_spec_checkpoint",
    "spec_bucket_size",
    "send_spec_weights",
    "save_weights_to_shm",
    "load_weights_from_shm",
    "cleanup_shm",
]
