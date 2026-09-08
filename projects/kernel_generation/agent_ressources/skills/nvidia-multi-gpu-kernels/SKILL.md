---
name: nvidia-multi-gpu-kernels
description: Write kernels which run on each gpu and allow for GPU-initiated communications (no cpu in the loop).
---

# How to do gpu-initiated copies

NVIDIA gpus all use the nvidia kernel driver which manages a virtual unified address space. When the cuda driver gets gpu memory addresses, the addresses are virtual and if (in the same process) you give an global memory address of one gpu to another, if the other gpu tries to access it, the hardware will make the copy go through the nvlink connection.
When you have multiple processes, you need to use cuda ipc stuff to properly get handles that are process safe and other things. you can read about this in the cuda docs.

# torch integration

Torch exposes this through the symetric memory api.


To write your own kernel doing communications with symmetric memory, you’ll need access to the addresses of mapped peer buffers and access to signal pads that are required for synchronization. In the kernel you’ll also need to perform correct synchronizations to make sure that peers are ready for communication, and signal to them that this GPU is ready.

PyTorch Symmetric Memory provides CUDA Graph-compatible synchronization primitives that operate on the signal pad accompanying each symmetric memory allocation. Kernels using symmetric memory can be written both in CUDA and in Triton. Here’s an example allocating symmetric tensor and exchanging handles:
```py
import torch.distributed._symmetric_memory as symm_mem

dist.init_process_group()
rank = dist.get_rank()

# Allocate a tensor
t = symm_mem.empty(4096, device=f"cuda:{rank}")
# Establish symmetric memory and obtain the handle
hdl = symm_mem.rendezvous(t, dist.group.WORLD)
```

Access to buffer pointers, multimem pointer, and signal pads is provided via:
```
hdl.buffer_ptrs
hdl.multicast_ptr
hdl.signal_pad_ptrs
```

# Advanced fault tolerance and copies

You can search for all fabric related things in nvidia docs (cuda and ptx docs mainly, it's all new).
