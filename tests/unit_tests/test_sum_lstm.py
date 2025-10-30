import argparse
import math
import os
import random
from typing import Optional, Tuple

import torch

from arctic_inference.py_custom_ops import try_load_torch_library

if try_load_torch_library():
    from arctic_inference.py_custom_ops import sum_lstm
else:
    raise RuntimeError(
        "The fused CUDA extension is not available. "
        "Compile your extension so that arctic_inference.py_custom_ops exposes `sum_lstm`."
    )


def rms_norm(x: torch.Tensor, eps: float, weight: Optional[torch.Tensor],
             bias: Optional[torch.Tensor]) -> torch.Tensor:
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = x * inv_rms
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def reference_sum_lstm(states4: torch.Tensor, z4: torch.Tensor,
                       prev_cell: torch.Tensor, w_cell: Optional[torch.Tensor],
                       b_cell: Optional[torch.Tensor],
                       w_state: Optional[torch.Tensor],
                       b_state: Optional[torch.Tensor], alpha: float,
                       eps_cell: float, eps_state: float,
                       fast_gelu: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch reference for:

      added_states = states + alpha * z4            # gatewise
      f, i, o = sigmoid(pre_f), sigmoid(pre_i), sigmoid(pre_o)
      c_pre = added_states[..., 3D:4D]
      c_new = prev_cell * f + GELU(RMSNorm(c_pre)) * i
      state = GELU(RMSNorm(c_new)) * o
    """
    D = prev_cell.size(-1)
    s_f, s_i, s_o, s_c = states4.split(D, dim=-1)
    z_f, z_i, z_o, z_c = z4.split(D, dim=-1)

    pre_f = s_f + alpha * z_f
    pre_i = s_i + alpha * z_i
    pre_o = s_o + alpha * z_o
    c_pre = s_c + alpha * z_c

    f = torch.sigmoid(pre_f)
    i = torch.sigmoid(pre_i)

    cn = rms_norm(c_pre, eps_cell, w_cell, b_cell)
    c_act = torch.nn.functional.gelu(
        cn, approximate="tanh" if fast_gelu else "none")
    c_new = prev_cell * f + c_act * i

    sn = rms_norm(c_new, eps_state, w_state, b_state)
    s_act = torch.nn.functional.gelu(
        sn, approximate="tanh" if fast_gelu else "none")

    o = torch.sigmoid(pre_o)
    state = s_act * o
    return state, c_new


def make_pitched_view(rows: int, cols: int, dtype: torch.dtype,
                      device: torch.device, pad: int) -> torch.Tensor:
    """
    Create a 2D tensor view with contiguous last dimension but a *larger row stride*
    (so row_stride != cols). This exercises the kernel's stride handling.
    """
    base = torch.empty((rows, cols + pad), dtype=dtype, device=device)
    view = base[:, :cols]
    return view


def gen_optional_vec(D: int, dtype: torch.dtype, device: torch.device,
                     enable: bool) -> Optional[torch.Tensor]:
    if not enable:
        return None
    t = torch.randn(D, dtype=dtype, device=device)
    return t.contiguous()


def run_one_case(rows: int, D: int, dtype: torch.dtype, device: torch.device,
                 fast_gelu: bool, use_wb_cell: bool, use_wb_state: bool,
                 seed: int, pad4: int, padD: int, atol: float,
                 rtol: float) -> None:
    torch.manual_seed(seed)

    states4 = make_pitched_view(rows, 4 * D, dtype, device, pad4)
    z = torch.randn(rows, D, dtype=dtype, device=device)

    z4 = make_pitched_view(rows, 4 * D, dtype, device, pad4)
    z4.copy_(z.repeat(1, 4))

    prev_cell = make_pitched_view(rows, D, dtype, device, padD)

    states4.copy_(torch.randn_like(states4))
    prev_cell.copy_(torch.randn_like(prev_cell))

    w_cell = gen_optional_vec(D, dtype, device, use_wb_cell)
    b_cell = gen_optional_vec(D, dtype, device, use_wb_cell)
    w_state = gen_optional_vec(D, dtype, device, use_wb_state)
    b_state = gen_optional_vec(D, dtype, device, use_wb_state)

    alpha = 0.35
    eps_cell = 1e-6
    eps_state = 1e-6

    with torch.no_grad():
        ref_state, ref_cell = reference_sum_lstm(states4, z4, prev_cell,
                                                 w_cell, b_cell, w_state,
                                                 b_state, alpha, eps_cell,
                                                 eps_state, fast_gelu)

    with torch.no_grad():
        fused_state, fused_cell = sum_lstm(states4, z4, prev_cell, w_cell,
                                           b_cell, w_state, b_state,
                                           float(alpha), float(eps_cell),
                                           float(eps_state), bool(fast_gelu))

    diff_state = (ref_state - fused_state).abs()
    diff_cell = (ref_cell - fused_cell).abs()

    max_abs_state = diff_state.max().item()
    max_abs_cell = diff_cell.max().item()

    max_rel_state = (diff_state / (ref_state.abs() + 1e-6)).max().item()
    max_rel_cell = (diff_cell / (ref_cell.abs() + 1e-6)).max().item()

    print(
        f"[dtype={str(dtype):>8s}] rows={rows:4d} D={D:5d} fast_gelu={fast_gelu} "
        f"wb_cell={use_wb_cell} wb_state={use_wb_state} "
        f"pad4={pad4:3d} padD={padD:3d} | "
        f"state abs={max_abs_state:.3e} rel={max_rel_state:.3e} ; "
        f"cell abs={max_abs_cell:.3e} rel={max_rel_cell:.3e}")

    assert max_abs_state <= atol or max_rel_state <= rtol, "State parity check failed"
    assert max_abs_cell <= atol or max_rel_cell <= rtol, "Cell parity check failed"


def main():
    parser = argparse.ArgumentParser(
        description="Parity check for fused sum_lstm kernel")
    parser.add_argument("--device",
                        default="cuda",
                        help="cuda device, e.g., cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fast_gelu",
                        action="store_true",
                        help="use approximate GELU in both paths")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required for this parity test"
    device = torch.device(args.device)

    dtypes = [torch.float16]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)

    tol = {
        torch.float16: dict(atol=1.2e-2, rtol=1.5e-2),
        torch.bfloat16: dict(atol=1.2e-2, rtol=1.5e-2),
    }

    cases = [
        dict(rows=8, D=256, pad4=0, padD=0),
        dict(rows=5, D=257, pad4=5, padD=3),
        dict(rows=16, D=1024, pad4=16, padD=8),
        dict(rows=1, D=64, pad4=0, padD=0),
        dict(rows=5, D=128, pad4=7, padD=3),
        dict(rows=33, D=256, pad4=0, padD=0),
        dict(rows=64, D=512, pad4=16, padD=8),
        dict(rows=7, D=1000, pad4=3, padD=5),
        dict(rows=128, D=1024, pad4=32, padD=16),
        dict(rows=9, D=257, pad4=5, padD=3),
    ]

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    for dtype in dtypes:
        atol = tol[dtype]["atol"]
        rtol = tol[dtype]["rtol"]

        for cfg in cases:
            for wb_cell in (True, False):
                for wb_state in (True, False):
                    run_one_case(
                        rows=cfg["rows"],
                        D=cfg["D"],
                        dtype=dtype,
                        device=device,
                        fast_gelu=args.fast_gelu,
                        use_wb_cell=wb_cell,
                        use_wb_state=wb_state,
                        seed=random.randint(0, 10_000_000),
                        pad4=cfg["pad4"],
                        padD=cfg["padD"],
                        atol=atol,
                        rtol=rtol,
                    )


if __name__ == "__main__":
    main()
