import torch
import os

import logging

logger = logging.getLogger(__name__)


def try_load_torch_library() -> bool:
    package_name = 'arctic_inference'
    module_basename = 'custom_ops'

    package_path = __import__(package_name).__path__[0]

    # Dynamically locate the compiled extension (handles .cpython-310... suffix)
    for file in os.listdir(package_path):
        if file.startswith(module_basename) and file.endswith('.so'):
            library_path = os.path.join(package_path, file)
            break
    else:
        logger.info("Could not find compiled custom_ops library in package.")
        return False

    try:
        logger.info(f"Attempting to load custom ops from {library_path}...")
        torch.ops.load_library(library_path)
        return True
    except RuntimeError as e:
        logger.info(
            f"Unable to load custom library from {library_path}. RuntimeError: {e}. Falling back to original implementation."
        )
        return False
    except Exception as e:
        logger.info(
            f"Unable to load custom library from {library_path}. Exception: {e}. Falling back to original implementation."
        )
        return False


from arctic_inference.op_builder.swiftkv_ops_builder import SwiftKVOpsBuilder

swiftkv_ops_module = SwiftKVOpsBuilder().load()

def reshape_and_cache_flash_bulk(
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    key_caches: list[torch.Tensor],
    value_caches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scales: list[torch.Tensor],
    v_scales: list[torch.Tensor],
    num_heads: int,
    head_size: int,
) -> None:
    torch.ops.arctic_inference.reshape_and_cache_flash_bulk(
        keys, values, key_caches, value_caches, slot_mapping, kv_cache_dtype,
        k_scales, v_scales, num_heads, head_size)

# --- Example Usage ---
if __name__ == '__main__':
    # NOTE: This is a placeholder for demonstrating the call.
    # You would need to create tensors with the correct shapes and types
    # as required by your kernel.
    print("\n--- Running Example ---")
    
    # Example dummy tensors (replace with actual data)
    num_tokens = 4
    num_layers = 2
    num_heads = 8
    head_size = 64
    block_size = 16
    
    keys_tensor = torch.randn(num_tokens, num_layers * num_heads * head_size, dtype=torch.float16, device='cuda')
    values_tensor = torch.randn(num_tokens, num_layers * num_heads * head_size, dtype=torch.float16, device='cuda')
    
    key_caches_list = [
        torch.zeros(1024, block_size, num_heads, head_size, dtype=torch.float16, device='cuda')
        for _ in range(num_layers)
    ]
    value_caches_list = [
        torch.zeros(1024, block_size, num_heads, head_size, dtype=torch.float16, device='cuda')
        for _ in range(num_layers)
    ]
    
    slot_mapping_tensor = torch.arange(num_tokens, dtype=torch.int64, device='cuda')
    k_scales_list = [torch.ones(1, device='cuda') for _ in range(num_layers)]
    v_scales_list = [torch.ones(1, device='cuda') for _ in range(num_layers)]

    print("Calling custom CUDA kernel...")
    reshape_and_cache_flash_bulk(
        keys=keys_tensor,
        values=values_tensor,
        key_caches=key_caches_list,
        value_caches=value_caches_list,
        slot_mapping=slot_mapping_tensor,
        kv_cache_dtype="auto",
        k_scales=k_scales_list,
        v_scales=v_scales_list,
        num_heads=num_heads,
        head_size=head_size
    )
    print("Kernel execution finished.")
