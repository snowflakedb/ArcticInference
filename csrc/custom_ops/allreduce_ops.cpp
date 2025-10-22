#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "allreduce_kernels.h"
#include "ipc_minimal.h"

namespace py = pybind11;
using namespace lamport_ar;

static inline DType to_dtype(at::ScalarType st) {
    if (st == at::kHalf)      return DType::kHalf;
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 11000)
    if (st == at::kBFloat16)  return DType::kBFloat16;
#endif
    if (st == at::kFloat)     return DType::kFloat;
    TORCH_CHECK(false, "Unsupported dtype: ", st);
}

struct OpenedIpc {
    void* ptr{nullptr};
    bool  is_self{false};
};

class LamportWorkspace : public torch::CustomClassHolder {
public:
    LamportWorkspace() = default;
    ~LamportWorkspace() override { destroy(); }

    void destroy() {
        if (device_ptr_array_.defined()) device_ptr_array_.reset();
        for (auto& o : opened_) {
            if (!o.is_self && o.ptr) cudaIpcCloseMemHandle(o.ptr);
        }
        opened_.clear();
        if (ctrl_dev_)    { cudaFree(ctrl_dev_);    ctrl_dev_ = nullptr; }
        if (data_triple_) { cudaFree(data_triple_); data_triple_ = nullptr; }
    }

    void init(const c10::intrusive_ptr<c10d::ProcessGroup>& pg, int64_t capacity_elems) {
        auto ws = build_ipc_triple_workspace(pg, capacity_elems);
        world_size_ = ws.world_size;
        rank_       = ws.rank;
        data_triple_= ws.data_triple;
        ctrl_dev_   = ws.ctrl_dev;
        device_ptr_array_ = ws.device_ptr_table;
        comm_size_bytes_  = ws.comm_size_bytes;
        capacity_elems_   = capacity_elems;

        opened_.resize(world_size_);
        for (int r = 0; r < world_size_; ++r) {
            opened_[r].ptr     = ws.opened_ptrs[r];
            opened_[r].is_self = (ws.is_self[r] != 0);
        }
    }

    at::Tensor ptrs_tensor() const { return device_ptr_array_; }
    int world_size() const { return world_size_; }
    int rank() const { return rank_; }
    size_t capacity_elems() const { return capacity_elems_; }
    size_t comm_size_bytes() const { return comm_size_bytes_; }

private:
    int world_size_{0};
    int rank_{-1};

    void* data_triple_{nullptr};
    void* ctrl_dev_{nullptr};
    std::vector<OpenedIpc> opened_;

    at::Tensor device_ptr_array_;
    size_t capacity_elems_{0};
    size_t comm_size_bytes_{0};
};

// ---- Ops ----

static at::Tensor all_reduce_outofplace(const at::Tensor& input,
                                        const c10::intrusive_ptr<LamportWorkspace>& ws,
                                        bool trigger_completion_at_end)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() >= 2, "input shape must be [..., hidden_dim] with hidden_dim last");

    auto st = input.scalar_type();
    auto dt = to_dtype(st);

    const auto numel = static_cast<size_t>(input.numel());
    TORCH_CHECK(numel <= ws->capacity_elems(),
        "numel (", numel, ") exceeds workspace capacity (", ws->capacity_elems(), ")");

    const int hidden_dim = static_cast<int>(input.size(input.dim() - 1));
    at::Tensor out = at::empty_like(input);

    AllReduceParams p{};
    p.nranks       = ws->world_size();
    p.rank         = ws->rank();
    p.dtype        = dt;
    p.size         = static_cast<int>(numel);
    p.hidden_dim   = hidden_dim;
    p.workspace    = reinterpret_cast<void**>(ws->ptrs_tensor().data_ptr());
    p.allreduce_in = input.data_ptr();
    p.allreduce_out= out.data_ptr();
    p.stream       = at::cuda::getCurrentCUDAStream(input.get_device());
    p.trigger_completion_at_end = trigger_completion_at_end;

    lamport_allreduce_run(p);
    return out;
}

static void all_reduce_inplace(at::Tensor& input,
                               const c10::intrusive_ptr<LamportWorkspace>& ws,
                               bool trigger_completion_at_end)
{
    auto out = all_reduce_outofplace(input, ws, trigger_completion_at_end);
    input.copy_(out, /*non_blocking=*/true);
}

// ---- Registration ----

TORCH_LIBRARY(lamport_ar, m) {
    m.class_<LamportWorkspace>("Workspace")
        .def(torch::init<>())
        .def("init", [](c10::intrusive_ptr<LamportWorkspace> self,
                        c10::intrusive_ptr<c10d::ProcessGroup> pg,
                        int64_t capacity_elems) {
                self->init(pg, capacity_elems);
                return self;
            })
        .def("destroy", &LamportWorkspace::destroy)
        .def("capacity_elems", &LamportWorkspace::capacity_elems)
        .def("world_size", &LamportWorkspace::world_size)
        .def("rank", &LamportWorkspace::rank);

    m.def("init_workspace(__torch__.torch.classes.lamport_ar.Workspace ws, "
         "__torch__.torch.classes.c10d.ProcessGroup pg, int capacity_elems) -> __torch__.torch.classes.lamport_ar.Workspace");

    m.impl("init_workspace", [](const c10::intrusive_ptr<LamportWorkspace>& ws,
                                c10::intrusive_ptr<c10d::ProcessGroup> pg,
                                int64_t capacity_elems){
        ws->init(pg, capacity_elems);
        return ws;
    });

    m.def("all_reduce(Tensor input, __torch__.torch.classes.lamport_ar.Workspace ws, bool trigger_completion_at_end=True) -> Tensor");
    m.def("all_reduce_(Tensor(a!) input, __torch__.torch.classes.lamport_ar.Workspace ws, bool trigger_completion_at_end=True) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(lamport_ar, CUDA, m) {
    m.impl("all_reduce", &all_reduce_outofplace);
    m.impl("all_reduce_", &all_reduce_inplace);
}

// Empty pybind init; ops are registered with TORCH_LIBRARY above
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

