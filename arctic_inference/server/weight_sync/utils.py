"""Shared utilities for weight synchronization.

Contains lightweight helpers, parameter descriptors, and model-aware
writers/updaters used by both sender and receiver code paths.
"""

from __future__ import annotations

import json
import logging
import struct

import torch
from torch import nn

logger = logging.getLogger(__name__)

_SF_DTYPE_MAP = {
    "F16": torch.float16, "BF16": torch.bfloat16,
    "F32": torch.float32, "F64": torch.float64,
    "I8": torch.int8, "I16": torch.int16,
    "I32": torch.int32, "I64": torch.int64,
    "U8": torch.uint8, "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}


# ---------------------------------------------------------------------------
# NCCL group creation (shared by sender + receiver)
# ---------------------------------------------------------------------------

def stateless_init_nccl(master_addr, master_port, rank, world_size, device,
                        *, is_server=None):
    """Create an independent PyNcclCommunicator via StatelessProcessGroup.

    When *is_server* is ``None`` (default), ``rank == 0`` creates the TCP
    listener (standard behaviour).  Pass an explicit bool to decouple the
    TCP listener role from the NCCL rank — needed when the network only
    allows one direction of connectivity (e.g. training → inference).
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    if is_server is None:
        pg = StatelessProcessGroup.create(
            host=master_addr, port=master_port, rank=rank, world_size=world_size
        )
    else:
        import socket
        from datetime import timedelta
        from torch.distributed import TCPStore

        listen_socket = None
        listen_fd = None
        if is_server:
            listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen_socket.bind(("0.0.0.0", master_port))
            listen_socket.listen()
            listen_fd = listen_socket.fileno()

        store = TCPStore(
            host_name=master_addr,
            port=master_port,
            world_size=world_size,
            is_master=is_server,
            timeout=timedelta(seconds=300),
            use_libuv=False,
            master_listen_fd=listen_fd,
        )
        pg = StatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            store=store,
            socket=listen_socket,
            data_expiration_seconds=3600,
        )

    return PyNcclCommunicator(pg, device=device)


# ---------------------------------------------------------------------------
# WeightInfo — lightweight parameter descriptor
# ---------------------------------------------------------------------------

_DTYPE_BYTES = {
    torch.float16: 2, torch.bfloat16: 2,
    torch.float32: 4, torch.float64: 8,
    torch.int8: 1, torch.int16: 2, torch.int32: 4, torch.int64: 8,
    torch.uint8: 1, torch.bool: 1,
    torch.float8_e4m3fn: 1, torch.float8_e5m2: 1,
}


class WeightInfo:
    __slots__ = ("name", "shape", "dtype")

    def __init__(self, name: str, shape: torch.Size, dtype: torch.dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype

    @property
    def nbytes(self) -> int:
        numel = 1
        for s in self.shape:
            numel *= s
        return numel * _DTYPE_BYTES[self.dtype]

    def to_dict(self) -> dict:
        return {"name": self.name, "shape": list(self.shape), "dtype": str(self.dtype)}

    @classmethod
    def from_dict(cls, d: dict) -> WeightInfo:
        return cls(d["name"], torch.Size(d["shape"]),
                   getattr(torch, d["dtype"].replace("torch.", "")))


def build_weights_info(model_path: str) -> list[WeightInfo]:
    """Build WeightInfo list from safetensors file headers (zero-copy)."""
    import glob
    from pathlib import Path

    p = Path(model_path)
    if not p.is_dir():
        from huggingface_hub import snapshot_download
        p = Path(snapshot_download(model_path))

    sf_files = sorted(glob.glob(str(p / "*.safetensors")))
    if not sf_files:
        raise FileNotFoundError(f"No safetensors files in {p}")

    infos: list[WeightInfo] = []
    for sf in sf_files:
        with open(sf, "rb") as fh:
            header_size = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_size))
        for k, v in header.items():
            if k == "__metadata__":
                continue
            dtype = _SF_DTYPE_MAP.get(v["dtype"], torch.float32)
            infos.append(WeightInfo(k, torch.Size(v["shape"]), dtype))
    return infos


# ---------------------------------------------------------------------------
# FP8 in-place weight update helpers
# ---------------------------------------------------------------------------

_STACKED_PARAMS = {
    "q_proj": "qkv_proj",
    "k_proj": "qkv_proj",
    "v_proj": "qkv_proj",
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
}

_SHARD_IDS = {
    "q_proj": "q", "k_proj": "k", "v_proj": "v",
    "gate_proj": 0, "up_proj": 1,
}

_SHARD_COUNTS = {"qkv_proj": 3, "gate_up_proj": 2}


class _FP8InplaceUpdater:
    """Accumulates BF16 weight shards and quantizes them back to FP8 in-place.

    Preserves the existing FP8 tensor GPU addresses so that CUDA graphs
    remain valid — no enforce_eager required.  Peak temporary memory is
    one module's BF16 buffer (~100-200 MB) rather than the full model.
    """

    def __init__(self, model: nn.Module, target_dtype: torch.dtype, device):
        from vllm.model_executor.parameter import BasevLLMParameter
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear, MergedColumnParallelLinear,
            QKVParallelLinear, RowParallelLinear,
        )
        self._linear_types = (ColumnParallelLinear, MergedColumnParallelLinear,
                              QKVParallelLinear, RowParallelLinear)
        self._model = model
        self._device = device
        self._dtype = target_dtype

        self._fp8_modules: dict[str, nn.Module] = {}
        self._params: dict[str, nn.Parameter] = dict(model.named_parameters())

        for name, mod in model.named_modules():
            if not isinstance(mod, self._linear_types):
                continue
            w = getattr(mod, "weight", None)
            if w is not None and not isinstance(w, BasevLLMParameter):
                self._fp8_modules[name] = mod

        self._bufs: dict[str, nn.Parameter] = {}
        self._shards_left: dict[str, int] = {}

    def _ensure_buf(self, mod_path: str) -> nn.Parameter:
        if mod_path in self._bufs:
            return self._bufs[mod_path]
        from vllm.model_executor.layers.linear import RowParallelLinear
        mod = self._fp8_modules[mod_path]
        w = mod.weight
        orig_shape = (w.shape[1], w.shape[0]) if w.ndim == 2 else w.shape
        buf = nn.Parameter(
            torch.empty(orig_shape, dtype=self._dtype, device=self._device),
            requires_grad=False)
        buf.weight_loader = mod.weight_loader
        if isinstance(mod, RowParallelLinear):
            buf.input_dim = 1
        else:
            buf.output_dim = 0
        merged_name = mod_path.rsplit(".", 1)[-1]
        self._shards_left[mod_path] = _SHARD_COUNTS.get(merged_name, 1)
        self._bufs[mod_path] = buf
        return buf

    def _flush_module(self, mod_path: str):
        from vllm._custom_ops import scaled_fp8_quant
        buf = self._bufs.pop(mod_path)
        mod = self._fp8_modules[mod_path]
        qweight, scale = scaled_fp8_quant(buf.data, scale=None)
        mod.weight.data.copy_(qweight.t().contiguous())
        mod.weight_scale.data.copy_(scale)
        del buf

    def _copy_param(self, param: nn.Parameter, tensor: torch.Tensor):
        """Copy tensor into param, using weight_loader for TP sharding if needed."""
        loader = getattr(param, "weight_loader", None)
        if loader is not None and param.data.shape != tensor.shape:
            loader(param, tensor)
        else:
            param.data.copy_(tensor)

    def _quant_and_copy(self, param: nn.Parameter, tensor: torch.Tensor,
                        scale_param: nn.Parameter | None = None):
        """FP8-quantize a BF16 tensor and copy into an FP8-transposed param."""
        from vllm._custom_ops import scaled_fp8_quant
        qweight, scale = scaled_fp8_quant(
            tensor.to(dtype=self._dtype, device=self._device), scale=None,
        )
        if param.data.ndim == 2 and param.data.shape == (qweight.shape[1], qweight.shape[0]):
            param.data.copy_(qweight.t().contiguous())
        else:
            param.data.copy_(qweight)
        if scale_param is not None:
            scale_param.data.copy_(scale)

    def _feed_fp8_module(self, mod_path: str, tensor: torch.Tensor,
                         shard_id=None) -> None:
        buf = self._ensure_buf(mod_path)
        if shard_id is not None:
            buf.weight_loader(buf, tensor, shard_id)
        else:
            buf.weight_loader(buf, tensor)
        self._shards_left[mod_path] -= 1
        if self._shards_left[mod_path] <= 0:
            self._flush_module(mod_path)

    def feed(self, sf_key: str, tensor: torch.Tensor):
        """Route one received BF16 tensor to the correct module."""
        if sf_key.endswith(".weight"):
            mod_candidate = sf_key[: -len(".weight")]
            prefix, _, leaf = mod_candidate.rpartition(".")

            merged = _STACKED_PARAMS.get(leaf)
            if merged and prefix:
                mod_path = f"{prefix}.{merged}"
                if mod_path in self._fp8_modules:
                    self._feed_fp8_module(mod_path, tensor, _SHARD_IDS[leaf])
                    return

            if mod_candidate in self._fp8_modules:
                self._feed_fp8_module(mod_candidate, tensor)
                return

        param = self._params.get(sf_key)
        if param is not None:
            if param.data.shape == tensor.shape:
                self._copy_param(param, tensor)
                return
            # Shape mismatch — the param is likely FP8-transposed.
            # Quantize the BF16 tensor to FP8 and copy with transposition.
            if param.data.dtype == torch.float8_e4m3fn:
                scale_name = sf_key.replace(".weight", ".weight_scale")
                scale_param = self._params.get(scale_name)
                self._quant_and_copy(param, tensor, scale_param)
                return
            self._copy_param(param, tensor)

    @property
    def pending(self) -> int:
        return len(self._bufs)


# ---------------------------------------------------------------------------
# Expected-HF-name oracle (for weight-sync param name validation)
# ---------------------------------------------------------------------------

# Param-name patterns that are legitimately NOT shipped by a BF16 sender:
#   - rotary frequency buffers (registered as buffers, not parameters)
#   - FP8 / GPTQ quantization scales / zero-points / metadata (only exist
#     on the receiver when the model is quantized).
_NON_SYNCED_HF_NAME_PATTERNS = (
    ".inv_freq",
    ".weight_scale",
    ".input_scale",
    ".weight_zero_point",
    ".weight_qzeros",
    ".weight_g_idx",
    ".weight_packed",
    ".weight_shape",
)


def _name_is_non_synced(name: str) -> bool:
    return any(pat in name for pat in _NON_SYNCED_HF_NAME_PATTERNS)


def compute_expected_hf_param_names(model: nn.Module) -> set[str]:
    """Compute the HF-style param name set the vLLM model can accept.

    Walks ``model.named_modules()`` to detect ``QKVParallelLinear`` and
    ``MergedColumnParallelLinear`` (gate_up) fusions; for each fused
    module, emits the corresponding unfused HF names
    (``q_proj``/``k_proj``/``v_proj`` or ``gate_proj``/``up_proj``).  For
    all other parameters, emits their name directly.

    Does not access ``param.data`` storage, so it works correctly when
    weights are offloaded (``param.data`` may be empty).

    Filters out param names that are legitimately not shipped by a BF16
    sender (rotary inv_freq buffers, FP8/GPTQ quantization metadata).
    """
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear, QKVParallelLinear,
    )

    expected: set[str] = set()
    fused_names: set[str] = set()

    def _parent_of(mod_path: str) -> str:
        return mod_path.rsplit(".", 1)[0] if "." in mod_path else ""

    for mod_path, mod in model.named_modules():
        if isinstance(mod, QKVParallelLinear):
            parent = _parent_of(mod_path)
            for orig in ("q_proj", "k_proj", "v_proj"):
                key = f"{parent}.{orig}.weight" if parent else f"{orig}.weight"
                expected.add(key)
            fused_names.add(f"{mod_path}.weight")
        elif isinstance(mod, MergedColumnParallelLinear):
            leaf = mod_path.rsplit(".", 1)[-1] if "." in mod_path else mod_path
            if leaf == "gate_up_proj":
                parent = _parent_of(mod_path)
                for orig in ("gate_proj", "up_proj"):
                    key = f"{parent}.{orig}.weight" if parent else f"{orig}.weight"
                    expected.add(key)
                fused_names.add(f"{mod_path}.weight")

    for name, _ in model.named_parameters():
        if name in fused_names:
            continue
        if _name_is_non_synced(name):
            continue
        expected.add(name)

    return expected


# ---------------------------------------------------------------------------
# Non-quantized direct-to-parameter writer (zero temp allocation for TP=1)
# ---------------------------------------------------------------------------

class _DirectParamWriter:
    """Pre-computes views into model parameter storage for zero-copy writes.

    For TP=1 non-quantized models, every safetensor weight maps exactly to
    either a model parameter or a narrow slice of a merged parameter
    (qkv_proj, gate_up_proj).
    """

    def __init__(self, model: nn.Module, device):
        from vllm.model_executor.layers.linear import (
            QKVParallelLinear, MergedColumnParallelLinear,
        )
        self._device = device
        self._views: dict[str, torch.Tensor] = {}

        params = dict(model.named_parameters())

        for mod_path, mod in model.named_modules():
            if not isinstance(mod, (QKVParallelLinear, MergedColumnParallelLinear)):
                continue
            weight = getattr(mod, "weight", None)
            if weight is None:
                continue

            output_dim = getattr(weight, "output_dim", None)
            if output_dim is None:
                output_dim = 0
            output_sizes = mod.output_sizes
            tp_size = mod.tp_size

            prefix = mod_path.rsplit(".", 1)[0] if "." in mod_path else ""
            leaf = mod_path.rsplit(".", 1)[-1] if "." in mod_path else mod_path

            if isinstance(mod, QKVParallelLinear):
                originals = [("q_proj", 0), ("k_proj", 1), ("v_proj", 2)]
            elif leaf == "gate_up_proj":
                originals = [("gate_proj", 0), ("up_proj", 1)]
            else:
                continue

            for orig_name, shard_idx in originals:
                sf_key = f"{prefix}.{orig_name}.weight" if prefix else f"{orig_name}.weight"
                offset = sum(output_sizes[:shard_idx]) // tp_size
                size = output_sizes[shard_idx] // tp_size
                self._views[sf_key] = weight.data.narrow(output_dim, offset, size)

        for name, param in params.items():
            if name not in self._views:
                self._views[name] = param.data

    def get_view(self, sf_key: str) -> torch.Tensor | None:
        return self._views.get(sf_key)

    def all_keys(self) -> list[str]:
        return list(self._views.keys())


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------

def load_spec_checkpoint(
    model_path: str,
) -> list[tuple[str, torch.Tensor]]:
    """Load all spec-model weights from a checkpoint directory.

    Supports both safetensors and pytorch_model.bin formats.
    Returns a list of ``(name, tensor)`` pairs with tensors on CPU.
    """
    import glob as _glob
    import os

    weights: list[tuple[str, torch.Tensor]] = []
    st_files = sorted(_glob.glob(os.path.join(model_path, "*.safetensors")))
    if st_files:
        from safetensors.torch import load_file
        for f in st_files:
            for name, tensor in load_file(f, device="cpu").items():
                weights.append((name, tensor))
    else:
        bin_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(bin_file):
            state = torch.load(bin_file, map_location="cpu", weights_only=True)
            for name, tensor in state.items():
                weights.append((name, tensor))
    return weights


def spec_bucket_size(
    model_path: str,
    min_bucket_size: int = 256 * 1024 * 1024,
) -> int:
    """Compute a bucket size large enough for the largest spec-model weight.

    Reads only ``config.json`` from *model_path*; no tensors are loaded.
    Falls back to *min_bucket_size* when the config is absent.
    """
    import os

    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return min_bucket_size
    with open(config_path) as f:
        cfg = json.load(f)

    def _parse_dim(val) -> list[int]:
        if isinstance(val, str):
            return [int(x) for x in val.split(".")]
        if isinstance(val, list):
            return [int(x) for x in val]
        if isinstance(val, int):
            return [val]
        return []

    vocab = cfg.get("vocab_size", 0)
    dims: list[int] = []
    for key in ("emb_dim", "inner_dim", "proj_dim", "input_hidden_dim"):
        dims.extend(_parse_dim(cfg.get(key, 0)))
    max_dim = max(dims) if dims else 0

    max_bytes = vocab * max_dim * 4  # float32
    return max(min_bucket_size, max_bytes)
