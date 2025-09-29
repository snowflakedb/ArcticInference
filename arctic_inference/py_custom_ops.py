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
        torch.ops.load_library(library_path)
        logger.info(f"Successfully loaded custom ops from {library_path}.")
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


def try_load_jit_library() -> bool:
    try:
        from arctic_inference.op_builder.swiftkv_ops_builder import SwiftKVOpsBuilder
        swiftkv_ops_module = SwiftKVOpsBuilder().load()

        logger.info("Successfully loaded SwiftKVOpsBuilder JIT library.")
        return True
    except ImportError as e:
        logger.info(
            f"Unable to import SwiftKVOpsBuilder. ImportError: {e}. Falling back to original implementation."
        )
        return False
    except Exception as e:
        logger.info(
            f"Unable to load JIT library. Exception: {e}. Falling back to original implementation."
        )
        return False


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