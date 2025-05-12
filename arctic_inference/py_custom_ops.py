import torch
import os

package_name = 'arctic_inference'
library_name = 'libCustomOps'

package_path = __import__(package_name).__path__[0]
print(f"package_path: {package_path}")
library_path = os.path.join(package_path, f'{library_name}.so')
print(f"library_path: {library_path}")
torch.ops.load_library(library_path)

def reshape_and_cache_flash_bulk(
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    key_caches: list[torch.Tensor],
    value_caches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scales: list[torch.Tensor],
    v_scales: list[torch.Tensor],
) -> None:
    torch.ops.arctic_inference.reshape_and_cache_flash_bulk(
        keys, values, key_caches, value_caches, slot_mapping, kv_cache_dtype,
        k_scales, v_scales)
    
def copy_caches_with_index(src_caches: list[torch.Tensor],
                           dst_caches: list[torch.Tensor],
                           shared_indices: torch.Tensor) -> None:
    torch.ops.arctic_inference.copy_caches_with_index(src_caches, 
                                                      dst_caches,
                                                      shared_indices)
