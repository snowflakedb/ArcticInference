import torch
from arctic_inference.op_builder import MoEOpsBuilder
from torch.library import Library
from typing import Callable, Optional
from vllm.utils import direct_register_custom_op

inf_module = None

def _topk_gating_fake(
            expert_counts: torch.Tensor,
            scores: torch.Tensor,
            assignments: torch.Tensor,
            offsets: torch.Tensor,
            router_logits: torch.Tensor,
            e_score_correction_bias: torch.Tensor) -> None:
    pass

def _topk_gating(
    expert_counts: torch.Tensor,
    scores: torch.Tensor,
    assignments: torch.Tensor,
    offsets: torch.Tensor,
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor) -> None:
    inf_module.top_k_gating(
            expert_counts,
            scores,
            assignments,
            offsets,
            router_logits,
            e_score_correction_bias,
        )

class MoERouter(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        global inf_module
        if inf_module is None:
            inf_module = MoEOpsBuilder().load()
        self.expert_counts = None
        self.expert_cumsum = None
        
        # library = Library("arctic_inference", "FRAGMENT")
        direct_register_custom_op(
            op_name="_topk_gating",
            op_func=_topk_gating,
            mutates_args=["expert_counts",
            "scores",
            "assignments",
            "offsets",
            "router_logits",
            "e_score_correction_bias"],
            fake_impl=_topk_gating_fake,
            # target_lib=library,
        )

    def allocate_buffers(
        self, 
        n_tokens, 
        num_experts, 
        top_k, 
        model_dim, 
        device='cuda', 
        dtype=torch.float16, 
        normalize_scores=True
    ):
        assignments = torch.empty((n_tokens, top_k), dtype=torch.int32, device=device)
        scores = torch.empty((n_tokens, top_k), dtype=torch.float32, device=device)
        offsets = torch.empty((n_tokens, top_k), dtype=torch.int32, device=device)
        return assignments, scores, offsets

    def __call__(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        n_tokens, model_dim = hidden_states.shape
        _, num_experts = router_logits.shape
        if self.expert_counts is None:
            self.expert_counts = torch.zeros((num_experts,), dtype=torch.int32, device=hidden_states.device)
        assignments, scores, offsets = self.allocate_buffers(
            n_tokens,
            num_experts,
            top_k,
            model_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            normalize_scores=renormalize,
        )
        torch.ops.vllm._topk_gating(
            self.expert_counts,
            scores,
            assignments,
            offsets,
            router_logits,
            e_score_correction_bias,
        )
        return (
            scores,
            assignments,
            # self.expert_counts
        )
