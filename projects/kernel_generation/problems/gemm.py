"""
K-major bf16 gemm problem file.
"""

import torch
import torch.nn as nn
from problem_loader import Config, pick_device
from collections.abc import Callable


ABC_DTYPE = torch.bfloat16
# Evaluator knobs (read by scripts/evaluate.py).
TOLERANCE = {'rtol': 0.02, 'atol': 0.02}
WARMUP_ITERS = 3

class Reference(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        return torch.mm(A, B, out=C)

def _make_gemm(M: int, N: int, K: int) -> Callable[[], tuple]:
    device = pick_device()

    def factory() -> tuple:
        A = torch.randn(M, K, device=device, dtype=ABC_DTYPE)
        B = torch.randn(N, K, device=device, dtype=ABC_DTYPE).transpose(0, 1)
        C = torch.empty(M, N, device=device, dtype=ABC_DTYPE)
        return (A, B, C)
    return factory

def flops(inputs: tuple, **_) -> float:
    A, B, _C = inputs
    M, K = A.shape
    N = B.shape[1]
    return 2.0 * M * N * K

def bytes_moved(inputs: tuple, **_) -> float:
    A, B, C = inputs
    return float(A.element_size() * (A.numel() + B.numel() + C.numel()))

def make_configs() -> list[Config]:
    out: list[Config] = []
    for M, N, K in [(4096, 4096, 4096), (8192, 8192, 8192), (16384, 16384, 16384)]:
        out.append(Config(make_inputs=_make_gemm(M, N, K), name=f'gemm_M{M}_N{N}_K{K}'))
    return out
