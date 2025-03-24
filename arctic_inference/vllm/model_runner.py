from typing import List

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class ArcticGPUModelRunner(GPUModelRunner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prepare_inputs(self, *args, **kwargs):
        attn_metadata, logits_indices, *rest = (
            super()._prepare_inputs(*args, **kwargs))
        # SwiftKV requires knowing the logits indices from inside the model
        # definition in order to early-stop the prefill tokens.
        attn_metadata.swiftkv_logits_indices = logits_indices
        return attn_metadata, logits_indices, *rest
