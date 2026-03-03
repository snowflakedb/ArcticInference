import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, "/code/users/yewang/arctic_inference_dev/ArcticInference-internal")


def _init_distributed():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl", world_size=1, rank=0)
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        ensure_model_parallel_initialized,
    )
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method="env://",
    )
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )


def test():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    _init_distributed()

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import (
        QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear,
    )
    from arctic_inference.server.weight_sync import _DirectParamWriter

    H, NH, NKH, HD, I = 256, 4, 2, 64, 512

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = MergedColumnParallelLinear(H, [I, I], bias=False)
            self.down_proj = RowParallelLinear(I, H, bias=False)

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = QKVParallelLinear(H, HD, NH, NKH)
            self.o_proj = RowParallelLinear(NH * HD, H, bias=False)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attn()
            self.mlp = MLP()
            self.norm = nn.LayerNorm(H)

    with set_current_vllm_config(VllmConfig()):
        m = M().to(device=device, dtype=torch.bfloat16)
    w = _DirectParamWriter(m, device)
    print("Views:", len(w._views))

    ok = True
    qs, ks = NH * HD, NKH * HD
    qw = m.attn.qkv_proj.weight.data

    for nm, off, sz in [
        ("attn.q_proj.weight", 0, qs),
        ("attn.k_proj.weight", qs, ks),
        ("attn.v_proj.weight", qs + ks, ks),
    ]:
        v = w.get_view(nm)
        leaf_off = off
        if v is None or v.data_ptr() != qw.narrow(0, leaf_off, sz).data_ptr():
            print("FAIL", nm, "view=", v is not None)
            ok = False
        else:
            print("OK", nm, list(v.shape))

    guw = m.mlp.gate_up_proj.weight.data
    for nm, off, sz in [
        ("mlp.gate_proj.weight", 0, I),
        ("mlp.up_proj.weight", I, I),
    ]:
        v = w.get_view(nm)
        if v is None or v.data_ptr() != guw.narrow(0, off, sz).data_ptr():
            print("FAIL", nm, "view=", v is not None)
            ok = False
        else:
            print("OK", nm, list(v.shape))

    for nm in ["attn.o_proj.weight", "mlp.down_proj.weight", "norm.weight", "norm.bias"]:
        v = w.get_view(nm)
        p = dict(m.named_parameters()).get(nm)
        if p is None:
            continue
        if v is not None and v.data_ptr() == p.data.data_ptr():
            print("OK", nm, list(v.shape))
        else:
            print("FAIL", nm)
            ok = False

    fq = torch.randn(qs, H, dtype=torch.bfloat16, device=device)
    w.get_view("attn.q_proj.weight").copy_(fq)
    if torch.equal(qw.narrow(0, 0, qs), fq):
        print("OK write-through")
    else:
        print("FAIL write-through")
        ok = False

    torch.cuda.reset_peak_memory_stats(device)
    mb = torch.cuda.memory_allocated(device)
    for nm in [
        "attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight",
        "attn.o_proj.weight", "mlp.down_proj.weight",
    ]:
        v = w.get_view(nm)
        if v is not None:
            v.copy_(torch.randn_like(v))
    pk = torch.cuda.max_memory_allocated(device)
    print("Extra mem:", round((pk - mb) / 1e6, 1), "MB")

    print("RESULT:", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    exit(0 if test() else 1)
