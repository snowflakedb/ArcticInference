import pytest
import torch
from typing import List

from arctic_inference.py_custom_ops import copy_caches_with_index

#torch.ops.load_library("arctic_inference/libCustomOps.so")

CUDA_DEVICES = [
    f"cuda:{0}" 
]

def copy_caches_with_index_ref(
    src_caches: List[torch.Tensor],
    dst_caches: List[torch.Tensor],
    shared_indices: torch.Tensor
) -> None:
    for i in range(len(src_caches)):
        dst_caches[i][:shared_indices.shape[0]].copy_(src_caches[i][shared_indices])
    return None

@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_copy_caches_with_index(
    device: str
) -> None:
    torch.set_default_device(device)

    src_caches = [
        torch.randn(3, 4, device=device),
        torch.randn(3, 4, device=device),
    ]

    dst_caches = [
        torch.empty(3, 4, device=device),
        torch.empty(3, 4, device=device),
    ]

    dst_caches_ref = [
        torch.empty(3, 4, device=device),
        torch.empty(3, 4, device=device),
    ]

    shared_indices = torch.tensor([0, 1], device=device)
    # Call the custom op
    copy_caches_with_index(src_caches, dst_caches, shared_indices)
    # Call the reference implementation
    copy_caches_with_index_ref(src_caches, dst_caches_ref, shared_indices)
    # Verify the results
    for i in range(len(src_caches)):
        assert torch.allclose(dst_caches[i], dst_caches_ref[i]), f"Mismatch in cache {i} on device {device}"