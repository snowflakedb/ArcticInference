from __future__ import annotations
import importlib.util
import os
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import torch

@dataclass(frozen=True)
class Config:
    make_inputs: Callable[[], tuple] = field(default=lambda: ())
    init_args: tuple = ()
    init_kwargs: dict = field(default_factory=dict)
    name: str = ''
    # Benchmark this config but never correctness-check it. The reference is neither
    # built nor run, which is the whole point: for very large cases the oracle's
    # memory is the binding limit, not the comparison (attention prefill caps at 8K
    # on its naive O(Lq*Lk) fp32 oracle). The latency still counts toward the score,
    # so a kernel that is wrong ONLY here scores as if it were right.
    perf_only: bool = False
    # Timed iterations for this config, each on its OWN freshly drawn input set: N sets,
    # N timed calls, 1:1. No set is ever measured twice, so nothing can warm across
    # iterations. Sets are drawn inside the loop and freed, keeping memory at ~1 set --
    # only `forward` is inside the timing window, so the draw costs the measurement
    # nothing.
    #
    # `None` (the default) means the evaluator decides, sampling until the median stops
    # moving. Prefer that: the right count depends on the config's own variability AND on
    # the thermal state it happens to run in, so no fixed number is right for both. Pin an
    # int only to make a run reproducible or to cap a config whose inputs are ruinous to
    # regenerate.
    num_samples: int | None = None

# Rank identity, for problems whose inputs are sharded across GPUs. Reads the variables
# torchrun sets, and answers as a world of one when launched normally -- so a single-GPU
# problem never has to know any of this exists.
def rank() -> int:
    return int(os.environ.get('RANK', 0))

def world_size() -> int:
    return int(os.environ.get('WORLD_SIZE', 1))

def local_rank() -> int:
    return int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))

# CUDA only. macOS/MPS support was deleted in 02d01f9; the device argument survives
# because problems and the evaluator pass it around, not because there is a choice.
#
# Returns bare 'cuda', not 'cuda:N', on purpose: the evaluator calls
# `torch.cuda.set_device(local_rank())` before any problem code runs, so 'cuda' already
# resolves to this rank's GPU. That is what lets every existing single-GPU problem work
# unchanged under torchrun.
def pick_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError('no CUDA device available')
    return 'cuda'

def synchronize(device: str) -> None:
    torch.cuda.synchronize()
# Keys are matched as substrings of torch.cuda.get_device_name(0) ('NVIDIA GB300'). A
# device absent from these tables reports no pct_peak/pct_bandwidth at all, which is how
# every GB300 run before this silently lost its roofline. GB300 differs from B300 on
# BOTH axes: 2500 TF dense bf16 against the standalone part's 2250, and 7928 GB/s
# against 7672. Bandwidths are decimal GB/s, matching
# `achieved_gbps = bytes / s / 1e9`. All three Blackwell entries read 8183 until
# 2026-08-24, which overstated the peak and so understated pct_bandwidth by ~3%
# (GB300) and ~6.7% (B200/B300). Runs recorded before that date carry the old
# denominator; their pct_bandwidth is not comparable to newer runs without a
# backfill.
_CUDA_BF16_TFLOPS = {'GB300': 2500.0, 'B300': 2250.0, 'B200': 2250.0,
                     'H200': 989.0, 'H100': 989.0, 'A100': 312.0}
_CUDA_GBPS = {'GB300': 7928.0, 'B300': 7672.0, 'B200': 7672.0,
              'H200': 4800.0, 'H100': 3350.0, 'A100': 2039.0}

def _cuda_peak(table: dict[str, float]) -> float | None:
    try:
        name = torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        return None
    # Longest key first, so the most specific device wins: 'GB300' must beat 'B300' on an
    # 'NVIDIA GB300'. Leaving that to dict order is a trap for whoever adds the next part.
    return next((table[k] for k in sorted(table, key=len, reverse=True) if k in name), None)

def device_peak_tflops(device: str) -> float | None:
    env = os.environ.get('KG_PEAK_TFLOPS')
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return _cuda_peak(_CUDA_BF16_TFLOPS)

def device_peak_bandwidth(device: str) -> float | None:
    env = os.environ.get('KG_PEAK_GBPS')
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return _cuda_peak(_CUDA_GBPS)

def load_module(path: str | Path, name: str) -> types.ModuleType:
    path = Path(path)
    parent = str(path.resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def solution_class(sol_module: types.ModuleType):
    cls = getattr(sol_module, 'Solution', None) or getattr(sol_module, 'Reference', None)
    if cls is None:
        raise AttributeError('solution module exports neither `Solution` nor `Reference`')
    return cls
