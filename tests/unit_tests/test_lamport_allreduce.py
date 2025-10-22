# test_lamport_allreduce.py
import os
import torch
import torch.distributed as dist
from lamport_vllm_custom_allreduce import LamportCustomAllreduce

def setup():
    backend = "gloo"  # we need object collectives for IPC handle exchange
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    torch.cuda.set_device(rank)  # 1 process per GPU assumed
    return rank, dist.get_world_size()

def main():
    rank, world = setup()
    device = f"cuda:{rank}"
    group = dist.group.WORLD

    lamport = LamportCustomAllreduce(group, device=device, max_elems=4 * 1024 * 1024)  # 4M elems max
    dtype = torch.float16
    tokens = 64
    hidden = 8192  # must be divisible by 8 for fp16/bf16
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)

    # expected via standard dist allreduce (CPU or NCCL both fine; do CPU to avoid NCCL dependency)
    x_cpu = x.detach().cpu()
    dist.all_reduce(x_cpu, op=dist.ReduceOp.SUM)
    expected = x_cpu.to(device)

    # Lamport AR
    y = lamport.custom_all_reduce(x)
    assert y is not None, "Lamport AR disabled for this config"

    # Check
    max_abs = (y - expected).abs().max().item()
    if rank == 0:
        print(f"[world={world}] dtype={dtype} tokens={tokens} hidden={hidden}  max_abs_err={max_abs:.3e}")
    lamport.close()
    dist.destroy_process_group()

if __name__ == "__main__":
    os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
    torch.backends.cuda.matmul.allow_tf32 = False
    main()

