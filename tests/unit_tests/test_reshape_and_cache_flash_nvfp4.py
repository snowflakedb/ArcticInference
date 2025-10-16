import math
import torch


def unpack_fp4_e2m1_to_float32(packed_u8: torch.Tensor) -> torch.Tensor:
    """
    packed_u8: uint8 tensor [..., D_bytes] where each byte packs 2 FP4 codes.
    Returns float32 tensor [..., 2*D_bytes] with values in [-6, 6].
    """
    assert packed_u8.dtype == torch.uint8
    low = packed_u8 & 0x0F  # low nibble
    high = packed_u8 >> 4  # high nibble
    nibbles = torch.stack((low, high), dim=-1)  # [..., D_bytes, 2]
    nibbles = nibbles.reshape(*packed_u8.shape[:-1],
                              packed_u8.shape[-1] * 2)  # [..., D]

    mag_code = (nibbles & 0x07).long()  # 3 bits
    sign_bit = (nibbles & 0x08) != 0  # bit 3

    mag_lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                           dtype=torch.float32,
                           device=packed_u8.device)
    mag = mag_lut[mag_code]
    sgn = torch.where(sign_bit, -1.0, 1.0).to(torch.float32)
    return sgn * mag


def recompute_e4m3_scales_from_src(x: torch.Tensor,
                                   group_size: int = 16) -> torch.Tensor:
    """
    x: [T, H, D] float16/float32
    Returns scales: [T, (H*D)//group_size] float32
    """
    T, H, D = x.shape
    M = H * D
    xf = x.to(torch.float32).reshape(T, M)
    vec_max = xf.abs().reshape(T, M // group_size,
                               group_size).amax(dim=-1)  # [T, M/16]
    sf = vec_max / 6.0

    sf = torch.clamp(sf, 0, 448)
    sf_q = sf.to(torch.float8_e4m3fn).to(
        torch.float32)  # rounded SFValue used by kernel
    return sf_q  # [T, M/16]


def expand_scales_16(scales_2d: torch.Tensor, H: int, D: int) -> torch.Tensor:
    """
    scales_2d: [T, (H*D)//16] float32
    Returns: [T, H, D] float32 with each scale repeated for its 16-element chunk.
    """
    T = scales_2d.shape[0]
    M = H * D
    s = scales_2d.unsqueeze(-1).expand(T, M // 16, 16).reshape(T, H, D)
    return s


def gather_cache_rows(cache: torch.Tensor,
                      slot_mapping: torch.Tensor,
                      block_size: int,
                      cache_layout: str = "NHD") -> torch.Tensor:
    """
    cache_layout:
      - "NHD": [num_blocks, block_size, num_heads, last_dim]
      - "HND": [num_blocks, num_heads, block_size, last_dim]
    Returns: [num_tokens, num_heads, last_dim]
    """
    num_tokens = slot_mapping.numel()
    block_idx = torch.div(slot_mapping, block_size, rounding_mode='floor')
    block_off = slot_mapping % block_size

    if cache_layout.upper() == "HND":
        cache_nhd = cache.permute(0, 2, 1, 3).contiguous()
    else:
        cache_nhd = cache

    return cache_nhd[block_idx, block_off]  # [T, H, last_dim]


def dequantize_fp4_cache_to_fp16(packed_bytes: torch.Tensor,
                                 src_fp16: torch.Tensor,
                                 slot_mapping: torch.Tensor,
                                 block_size: int,
                                 cache_layout: str = "NHD") -> torch.Tensor:
    """
    packed_bytes: uint8, [B, page, H, D/2] (NHD) or [B, H, page, D/2] (HND)
    src_fp16:     float16, [T, H, D]  (original inputs used during quantization)
    Returns float16 dequantized tensor [T, H, D]
    """
    T, H, D = src_fp16.shape

    packed_tok = gather_cache_rows(packed_bytes, slot_mapping, block_size,
                                   cache_layout)  # [T, H, D/2]

    e2m1_vals = unpack_fp4_e2m1_to_float32(packed_tok)  # [T, H, D]

    scales_2d = recompute_e4m3_scales_from_src(src_fp16)  # [T, (H*D)/16]
    scales = expand_scales_16(scales_2d, H, D)  # [T, H, D]

    dequant = (e2m1_vals * scales).to(torch.float16)  # [T, H, D]
    return dequant


def parity_check_fp4_vs_fp16(
    num_tokens: int = 64,
    num_heads: int = 16,
    head_size: int = 128,
    block_size: int = 16,
    cache_layout: str = "NHD",
    device: str = "cuda",
    dtype=torch.float16,
    seed: int = 0,
):
    torch.manual_seed(seed)
    T, H, D = num_tokens, num_heads, head_size
    assert D % 16 == 0, "head_size must be divisible by 16 for FP4 groups"

    key = torch.randn(T, H, D, device=device, dtype=dtype) * 0.5
    value = torch.randn(T, H, D, device=device, dtype=dtype) * 0.5
    slot_mapping = torch.arange(T, device=device,
                                dtype=torch.long)  # sequential mapping
    num_blocks = math.ceil(T / block_size)

    if cache_layout.upper() == "NHD":
        # [B, page, H, last_dim]
        key_cache_fp16 = torch.empty(num_blocks,
                                     block_size,
                                     H,
                                     D,
                                     device=device,
                                     dtype=torch.float16)
        value_cache_fp16 = torch.empty(num_blocks,
                                       block_size,
                                       H,
                                       D,
                                       device=device,
                                       dtype=torch.float16)
        key_cache_fp4 = torch.empty(num_blocks,
                                    block_size,
                                    H,
                                    D // 2,
                                    device=device,
                                    dtype=torch.uint8)
        value_cache_fp4 = torch.empty(num_blocks,
                                      block_size,
                                      H,
                                      D // 2,
                                      device=device,
                                      dtype=torch.uint8)

        key_scale_cache = torch.empty(num_blocks,
                                      block_size,
                                      H,
                                      D // 16,
                                      device=device,
                                      dtype=torch.uint8)
        value_scale_cache = torch.empty(num_blocks,
                                        block_size,
                                        H,
                                        D // 16,
                                        device=device,
                                        dtype=torch.uint8)
    else:
        # HND: [B, H, page, last_dim]
        key_cache_fp16 = torch.empty(num_blocks,
                                     H,
                                     block_size,
                                     D,
                                     device=device,
                                     dtype=torch.float16)
        value_cache_fp16 = torch.empty(num_blocks,
                                       H,
                                       block_size,
                                       D,
                                       device=device,
                                       dtype=torch.float16)
        key_cache_fp4 = torch.empty(num_blocks,
                                    H,
                                    block_size,
                                    D // 2,
                                    device=device,
                                    dtype=torch.uint8)
        value_cache_fp4 = torch.empty(num_blocks,
                                      H,
                                      block_size,
                                      D // 2,
                                      device=device,
                                      dtype=torch.uint8)
        key_scale_cache = torch.empty(num_blocks,
                                      H,
                                      block_size,
                                      D // 16,
                                      device=device,
                                      dtype=torch.uint8)
        value_scale_cache = torch.empty(num_blocks,
                                        H,
                                        block_size,
                                        D // 16,
                                        device=device,
                                        dtype=torch.uint8)

    from vllm import _custom_ops as _
    k_scale = torch.tensor(0.0, device=device, dtype=torch.float32)
    v_scale = torch.tensor(0.0, device=device, dtype=torch.float32)
    torch.ops._C_cache_ops.reshape_and_cache_flash(key, value, key_cache_fp16,
                                                   value_cache_fp16,
                                                   slot_mapping, "auto",
                                                   k_scale, v_scale)

    from arctic_inference.py_custom_ops import (try_load_torch_library,
                                                reshape_and_cache_flash_fp4)
    if not try_load_torch_library():
        raise RuntimeError(
            "Custom FP4 ops not available (try_load_torch_library() returned False)"
        )

    fp4_called = False
    try:
        reshape_and_cache_flash_fp4(key, value, key_cache_fp4, value_cache_fp4,
                                    key_scale_cache, value_scale_cache,
                                    slot_mapping, "fp4", k_scale, v_scale)
        fp4_called = True
    except TypeError:
        print("Could not call reshape_and_cache_flash_fp4 with full signature")

    if not fp4_called:
        raise RuntimeError(
            "Could not invoke reshape_and_cache_flash_fp4 with either signature."
        )

    key_from_fp4 = dequantize_fp4_cache_to_fp16(key_cache_fp4, key,
                                                slot_mapping, block_size,
                                                cache_layout)
    value_from_fp4 = dequantize_fp4_cache_to_fp16(value_cache_fp4, value,
                                                  slot_mapping, block_size,
                                                  cache_layout)

    key_fp16_tok = gather_cache_rows(key_cache_fp16, slot_mapping, block_size,
                                     cache_layout)
    value_fp16_tok = gather_cache_rows(value_cache_fp16, slot_mapping,
                                       block_size, cache_layout)

    def stats(a: torch.Tensor, b: torch.Tensor):
        diff = (a.to(torch.float32) - b.to(torch.float32))
        mae = diff.abs().mean().item()
        mse = (diff * diff).mean().item()
        rmse = math.sqrt(mse)
        mx = diff.abs().max().item()
        return mae, mse, rmse, mx

    k_mae, k_mse, k_rmse, k_max = stats(key_from_fp4, key_fp16_tok)
    v_mae, v_mse, v_rmse, v_max = stats(value_from_fp4, value_fp16_tok)

    print(
        f"[K]  MAE={k_mae:.6f}  RMSE={k_rmse:.6f}  MaxAbs={k_max:.6f}  MSE={k_mse:.6f}"
    )
    print(
        f"[V]  MAE={v_mae:.6f}  RMSE={v_rmse:.6f}  MaxAbs={v_max:.6f}  MSE={v_mse:.6f}"
    )

    return {
        "key_from_fp4": key_from_fp4,
        "key_fp16": key_fp16_tok,
        "value_from_fp4": value_from_fp4,
        "value_fp16": value_fp16_tok,
        "metrics": {
            "K": {
                "MAE": k_mae,
                "RMSE": k_rmse,
                "MaxAbs": k_max,
                "MSE": k_mse
            },
            "V": {
                "MAE": v_mae,
                "RMSE": v_rmse,
                "MaxAbs": v_max,
                "MSE": v_mse
            },
        },
    }


def show_example_row(
    results: dict,
    tensor_name: str = "key",
    token_idx: int = 0,
    head_idx: int = 0,
    num_vals: int = 16,
):
    original_tensor = results[f"{tensor_name}_fp16"]
    dequant_tensor = results[f"{tensor_name}_from_fp4"]

    original_slice = original_tensor[token_idx,
                                     head_idx, :num_vals].to(torch.float32)
    dequant_slice = dequant_tensor[token_idx,
                                   head_idx, :num_vals].to(torch.float32)
    diff = (original_slice - dequant_slice).abs()

    print("\n" + "-" * 50)
    print(
        f"Example Comparison: '{tensor_name.upper()}' (Token {token_idx}, Head {head_idx}, First {num_vals} values)"
    )
    print("-" * 50)

    torch.set_printoptions(precision=4, sci_mode=False)

    print(f"Original FP16 : {original_slice.cpu().numpy()}")
    print(f"From FP4      : {dequant_slice.cpu().numpy()}")
    print(f"Abs Difference: {diff.cpu().numpy()}")
    print("-" * 50 + "\n")


if __name__ == "__main__":
    print("NHD Cache Layout Results:")
    results_nhd = parity_check_fp4_vs_fp16(num_tokens=64,
                                           num_heads=16,
                                           head_size=128,
                                           block_size=16,
                                           cache_layout="NHD",
                                           device="cuda")
    show_example_row(results_nhd, tensor_name="key")
    show_example_row(results_nhd, tensor_name="value")

    print("\n\nHND Cache Layout Results:")
    results_hnd = parity_check_fp4_vs_fp16(num_tokens=64,
                                           num_heads=16,
                                           head_size=128,
                                           block_size=16,
                                           cache_layout="HND",
                                           device="cuda")
    show_example_row(results_hnd, tensor_name="key")
    show_example_row(results_hnd, tensor_name="value")
