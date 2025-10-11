
#include <torch/extension.h>
#include "topk_router.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // top_k_gating.h
    m.def("top_k_gating", &top_k_gating, "Top-k gating for MoE module.");
}
