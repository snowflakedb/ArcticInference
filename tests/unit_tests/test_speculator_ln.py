import torch
import torch.nn as nn
import random
import math

from arctic_inference.py_custom_ops import (try_load_torch_library,
                                            speculator_ln)


class MLPSpeculatorLayerNorm(nn.Module):
    """
    A L2 normalization implementation
    ...
    Args
    ----
    normalized_shape : int
        Dimensionality of input data (size of final tensor axis)
    eps : float
        Safety term to prevent division by zero. Make sure the chosen value
         fits in the range of your encoding scheme
         (i.e. fp16 requires eps >= 6e-8).
    elementwise_scale_and_shift : bool
        Include a learned scaling and shift term after normalization.
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-06,
        elementwise_scale_and_shift=True,
    ):
        super().__init__()
        self.elementwise_scale_and_shift = elementwise_scale_and_shift
        if self.elementwise_scale_and_shift:
            self.weight = nn.Parameter(torch.empty(normalized_shape))
            self.bias = nn.Parameter(torch.empty(normalized_shape))
        self.eps = eps

        assert try_load_torch_library(), "Custom ops library failed to load."

    def forward(self, x):
        xf = x
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        x = xf.type_as(x)
        if self.elementwise_scale_and_shift:
            x = self.weight * x
            x = x + self.bias
        return x

    def forward_opt(self, x):
        return speculator_ln(
            x,
            self.weight if self.elementwise_scale_and_shift else None,
            self.bias if self.elementwise_scale_and_shift else None,
            float(self.eps),
        )


def run_case(shape, dtype, affine, eps=1e-6):
    device = "cuda"
    hidden = shape[-1]

    x = torch.randn(*shape, device=device, dtype=dtype)

    if affine:
        ref = MLPSpeculatorLayerNorm(hidden,
                                     eps=eps,
                                     elementwise_scale_and_shift=True).to(
                                         device=device, dtype=dtype)
        with torch.no_grad():
            ref.weight.copy_(torch.randn(hidden, device=device, dtype=dtype))
            ref.bias.copy_(torch.randn(hidden, device=device, dtype=dtype))
        y_ref = ref(x)

        y = ref.forward_opt(x)
    else:
        ref = MLPSpeculatorLayerNorm(hidden,
                                     eps=eps,
                                     elementwise_scale_and_shift=False).to(
                                         device=device, dtype=dtype)
        y_ref = ref(x)

        y = ref.forward_opt(x)

    max_abs = (y - y_ref).abs().max().item()
    denom = y_ref.abs().max().item() + 1e-8
    max_rel = (y - y_ref).abs().max().item() / denom

    return max_abs, max_rel


def main():
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this parity check.")

    shapes = [
        (32, 128),  # multiple of 8
        (16, 260),  # multiple of 4, not 8
        (7, 513),  # tail
        (2, 3, 1024)  # higher rank
    ]
    dtypes = [torch.float16, torch.bfloat16]
    affines = [False, True]

    print("Running parity checks against MLPSpeculatorLayerNorm...")
    for dtype in dtypes:
        for affine in affines:
            for shape in shapes:
                max_abs, max_rel = run_case(shape, dtype, affine, eps=1e-6)
                print(
                    f"dtype={str(dtype).split('.')[-1]:>9}  affine={affine!s:>5}  shape={shape!s:<12} "
                    f"max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")


if __name__ == "__main__":
    main()
