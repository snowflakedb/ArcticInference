import math
import torch


def unpack_fp4_e2m1_to_float32(packed_u8: torch.Tensor) -> torch.Tensor:
    assert packed_u8.dtype == torch.uint8
    low = packed_u8 & 0x0F
    high = packed_u8 >> 4
    nibbles = torch.stack((low, high), dim=-1)  # [..., D_bytes, 2]
    nibbles = nibbles.reshape(*packed_u8.shape[:-1],
                              packed_u8.shape[-1] * 2)  # [..., D]

    mag_code = (nibbles & 0x07).long()
    sign_bit = (nibbles & 0x08) != 0

    mag_lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                           dtype=torch.float32,
                           device=packed_u8.device)
    mag = mag_lut[mag_code]
    sgn = torch.where(sign_bit, -1.0, 1.0).to(torch.float32)
    return sgn * mag


def decode_fp8_e4m3(u8: torch.Tensor) -> torch.Tensor:
    assert u8.dtype == torch.uint8

    s = (u8 >> 7).to(torch.int32)
    e = ((u8 >> 3) & 0x0F).to(torch.int32)
    m = (u8 & 0x07).to(torch.int32)

    sign = torch.where(s != 0, -1.0, 1.0).to(torch.float32)

    is_sub = (e == 0)
    exp_norm = (e - 7).to(torch.float32)
    exp_sub = torch.full_like(exp_norm, -6.0)

    mant_norm = 1.0 + (m.to(torch.float32) / 8.0)
    mant_sub = (m.to(torch.float32) / 8.0)

    base = torch.where(is_sub, exp_sub, exp_norm)
    mant = torch.where(is_sub, mant_sub, mant_norm)

    val = sign * torch.pow(torch.tensor(2.0, device=u8.device), base) * mant
    return torch.clamp(val, -448.0, 448.0)  # finite-range variant


def expand_scales_from_bytes(scales_u8_tok: torch.Tensor, H: int,
                             D: int) -> torch.Tensor:
    sf = decode_fp8_e4m3(scales_u8_tok)  # [T, H, D/16] as f32

    return sf.unsqueeze(-1).expand(-1, -1, -1, 16).reshape(-1, H, D)


def gather_cache_rows(cache: torch.Tensor, slot_mapping: torch.Tensor,
                      block_size: int) -> torch.Tensor:
    block_idx = torch.div(slot_mapping, block_size, rounding_mode='floor')
    block_off = slot_mapping % block_size

    return cache[block_idx, block_off]


def dequantize_fp4_cache_to_fp16_using_cache_scales(
    packed_bytes: torch.Tensor,  # uint8 [B, page, H, D/2] or [B, H, page, D/2]
    scale_bytes: torch.
    Tensor,  # uint8 [B, page, H, D/16] or [B, H, page, D/16]
    slot_mapping: torch.Tensor,  # int64 [T]
    block_size: int,
) -> torch.Tensor:
    packed_tok = gather_cache_rows(packed_bytes, slot_mapping,
                                   block_size)  # [T, H, D/2]
    scales_u8_tok = gather_cache_rows(scale_bytes, slot_mapping,
                                      block_size)  # [T, H, D/16]

    T, H, D_half = packed_tok.shape
    D = D_half * 2

    e2m1_vals = unpack_fp4_e2m1_to_float32(packed_tok)

    scales = expand_scales_from_bytes(scales_u8_tok, H, D)

    return (e2m1_vals * scales).to(torch.float16)


def parity_check_fp4_vs_fp16(
    num_tokens: int = 64,
    num_heads: int = 16,
    head_size: int = 128,
    block_size: int = 16,
    device: str = "cuda",
    dtype=torch.float16,
    seed: int = 0,
):
    torch.manual_seed(seed)
    T, H, D = num_tokens, num_heads, head_size
    assert D % 16 == 0, "head_size must be divisible by 16 for FP4 groups"

    key = torch.randn(T, H, D, device=device, dtype=dtype) * 0.5
    value = torch.randn(T, H, D, device=device, dtype=dtype) * 0.5
    slot_mapping = torch.arange(T, device=device, dtype=torch.long)
    num_blocks = math.ceil(T / block_size)

    # [B, page, H, last_dim]
    key_cache_fp16 = torch.zeros(num_blocks,
                                 block_size,
                                 H,
                                 D,
                                 device=device,
                                 dtype=torch.float16)
    value_cache_fp16 = torch.zeros(num_blocks,
                                   block_size,
                                   H,
                                   D,
                                   device=device,
                                   dtype=torch.float16)
    key_cache_fp4 = torch.zeros(num_blocks,
                                block_size,
                                H,
                                D // 2,
                                device=device,
                                dtype=torch.uint8)
    value_cache_fp4 = torch.zeros(num_blocks,
                                  block_size,
                                  H,
                                  D // 2,
                                  device=device,
                                  dtype=torch.uint8)
    key_scale_cache = torch.zeros(num_blocks,
                                  block_size,
                                  H,
                                  D // 16,
                                  device=device,
                                  dtype=torch.uint8)
    value_scale_cache = torch.zeros(num_blocks,
                                    block_size,
                                    H,
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

    key_from_fp4 = dequantize_fp4_cache_to_fp16_using_cache_scales(
        key_cache_fp4, key_scale_cache, slot_mapping, block_size)
    value_from_fp4 = dequantize_fp4_cache_to_fp16_using_cache_scales(
        value_cache_fp4, value_scale_cache, slot_mapping, block_size)

    key_fp16_tok = gather_cache_rows(key_cache_fp16, slot_mapping, block_size)
    value_fp16_tok = gather_cache_rows(value_cache_fp16, slot_mapping,
                                       block_size)

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
    num_vals: int = 64,
):
    original_tensor = results[f"{tensor_name}_fp16"]
    dequant_tensor = results[f"{tensor_name}_from_fp4"]

    original_slice = original_tensor[:num_vals].to(torch.float32)
    dequant_slice = dequant_tensor[:num_vals].to(torch.float32)
    diff = (original_slice - dequant_slice).abs()

    print("\n" + "-" * 50)
    print(
        f"Example Comparison: '{tensor_name.upper()}' (First {num_vals} values)"
    )
    print("-" * 50)

    torch.set_printoptions(precision=4, sci_mode=False)

    print(f"Original FP16 : {original_slice.cpu().numpy()}")
    print(f"From FP4      : {dequant_slice.cpu().numpy()}")
    print(f"Abs Difference: {diff.cpu().numpy()}")
    print("-" * 50 + "\n")


if __name__ == "__main__":
    for block_size in [16, 32]:
        for num_tokens in [1, 16, 256, 2048]:
            for num_heads in [8, 16]:
                for head_size in [64, 128]:
                    print(
                        f"Running parity check: num_tokens={num_tokens}, num_heads={num_heads}, head_size={head_size}, block_size={block_size}"
                    )
                    results = parity_check_fp4_vs_fp16(num_tokens=num_tokens,
                                                       num_heads=num_heads,
                                                       head_size=head_size,
                                                       block_size=block_size,
                                                       device="cuda")
