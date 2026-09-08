# torch.cuda._compile_kernel()

This function is a custom, new and different function from torch.cuda.load_inline.

This one uses nvrtc to compile the provided kernel source at lightspeed. It also doesn't give any flexibility on the kernel launcher code.
Kernels objects returned are functions which take 3 args: grid_size: tuple, block_size: tuple and args: list[Arg] where each Arg can be either a torch.tensor, a python float or a python int and finally an optional dynamic shared memory size argument (int) which represents the dynamic shared memory size amount in bytes.

How to add include paths:
```py
import torch
from torch.utils.cpp_extension import include_paths

kernel = torch.cuda._compile_kernel(
    kernel_source=KERNEL_SRC,
    kernel_name="my_kernel",
    cuda_include_dirs=[
        *include_paths("cuda"),
    ],
    compute_capability="100a",
)
```

## Shared memory size

you can use 

```py
libcuda = _get_cuda_library()
_check_cuda(libcuda.cuFuncSetAttribute(kernel.func, 8, smem_size))
```

to set the dynamic memory size when its above the treshold too (torch kernel wrapper doesn't always support the .set_shared_memory_config() method depending on torch version)


# TMA usage

This means that you cannot pass TMA descriptors (TensorMap) directly and need to create a torch byte tensor and put the contents of the descriptor inside and read it inside the kernel:

```py
import struct
import torch
import cuda.bindings.driver as cuda

def tm_to_bytes(tm):
    return b''.join(struct.pack('<Q', int(x)) for x in tm.opaque)

def make_tensormap_2d_bf16(gpu_ptr, rows, cols, box_rows, box_cols, swizzle='none'):
    sw_map = {
        'none':  cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
        '32b':   cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
        '64b':   cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
        '128b':  cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
    }
    dt = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
    intl = cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE
    l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B
    oob = cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    u64 = cuda.cuuint64_t
    u32 = cuda.cuuint32_t
    global_dim = [u64(cols), u64(rows)]
    global_strides = [u64(cols * 2)]
    box_dim = [u32(box_cols), u32(box_rows)]
    element_strides = [u32(1), u32(1)]
    err, tm = cuda.cuTensorMapEncodeTiled(
        dt, 2, gpu_ptr,
        global_dim, global_strides, box_dim, element_strides,
        intl, sw_map[swizzle], l2, oob,
    )
    assert int(err) == 0, f"cuTensorMapEncodeTiled err={err}"
    return tm_to_bytes(tm)


tm_bytes = make_tensormap_2d_bf16(my_tensor.data_ptr(), M, N, box_rows=BLOCK_K, box_cols=BLOCK_N_H, swizzle='128b')
tm_gpu = torch.frombuffer(bytearray(tm_bytes), dtype=torch.uint8).to("cuda")

kernel(grid, threads, (tm_gpu, my_tensor), smem_size) # and call kernel with it
```

This process of creating and copying the TensorMap over is fairly expensive in python, you only want to do it when the shapes changes. (don't benchmark tensor map creation ..)

## simpler version without cuda bindings but using ctypes directly

```py
import ctypes

libcuda = ctypes.CDLL("libcuda.so.1")

CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 9
CU_TENSOR_MAP_SWIZZLE_NONE = 0
CU_TENSOR_MAP_SWIZZLE_32B = 1
CU_TENSOR_MAP_SWIZZLE_64B = 2
CU_TENSOR_MAP_SWIZZLE_128B = 3
CU_TENSOR_MAP_INTERLEAVE_NONE = 0
CU_TENSOR_MAP_L2_PROMOTION_L2_128B = 2
CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0

cuTensorMapEncodeTiled = libcuda.cuTensorMapEncodeTiled
cuTensorMapEncodeTiled.restype = ctypes.c_int
cuTensorMapEncodeTiled.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


def create_tma_map_2d(tensor, box_inner, box_outer, swizzle_mode):
    """
    tensor: 2D CUDA bf16 tensor, row-major, shape (outer, inner).
    box_inner: innermost dim box size (elements)
    box_outer: outer dim box size (elements)
    """
    assert tensor.ndim == 2
    assert tensor.is_cuda and tensor.dtype == torch.bfloat16
    outer, inner = tensor.shape
    elem_size = tensor.element_size()

    global_dim = (ctypes.c_uint64 * 2)(inner, outer)
    global_strides = (ctypes.c_uint64 * 1)(inner * elem_size)
    box_dim = (ctypes.c_uint32 * 2)(box_inner, box_outer)
    element_strides = (ctypes.c_uint32 * 2)(1, 1)

    host_map = (ctypes.c_uint8 * 128)()

    res = cuTensorMapEncodeTiled(
        ctypes.addressof(host_map),
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        tensor.data_ptr(),
        global_dim,
        global_strides,
        box_dim,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_mode,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if res != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {res}")

    cpu_t = torch.frombuffer(bytes(host_map), dtype=torch.uint8).clone()
    dev_t = cpu_t.cuda()
    torch.cuda.synchronize()
    return dev_t
```


## Using tensormap type inside the kernel

NVRTC doesn't support including `cuda.h` which is where the CUtensormap type is defined. 
Instead you can define a stub tensor map type as an opaque 128B type:

```cu
struct alignas(128) CUtensorMap {
    unsigned long long opaque[16];
};
```