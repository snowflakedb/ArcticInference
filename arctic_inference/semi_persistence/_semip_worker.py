"""Custom vLLM worker for TP semi-persistence.

Standalone ``worker_cls`` (subclasses vanilla vLLM ``Worker``; no
arctic_inference dependency).  Wired via ``vllm_config["worker_cls"] =
"_semip_worker.SemipGPUWorker"`` for TP>1 instances.

Two hooks:
  * ``init_device`` remaps the vLLM ``local_rank`` to a physical GPU via
    ``SEMIP_GPU_MAP`` (set by vllm_child before spawn), so a TP group can be
    placed on an arbitrary set of physical GPUs while keeping all GPUs
    visible (required for the cuda-checkpoint physical-GPU addressing).
  * ``compile_or_warm_up_model`` forces the single cold-start CUDA-graph
    capture onto the CustomAllreduce copy path (registered=False) with
    keep_graph=True, so the preserved graph is reuse-friendly and
    ``ca_graph_rebind`` can rewrite its baked addresses after CRIU restore.
    The patches are scoped to the capture window (try/finally) so runtime is
    unaffected.

Both hooks return whatever the base class returns, so this stays agnostic to
the ``compile_or_warm_up_model`` return type (a float on older vLLM, a
``CompilationTimes`` on newer).
"""
import os

from vllm.v1.worker.gpu_worker import Worker


class SemipGPUWorker(Worker):
    def init_device(self):
        gpu_map = os.environ.get("SEMIP_GPU_MAP")
        if gpu_map:
            self.local_rank = int(gpu_map.split(",")[self.local_rank])
        return super().init_device()

    def compile_or_warm_up_model(self):
        try:
            import ca_graph_rebind
        except Exception:
            return super().compile_or_warm_up_model()
        ca_graph_rebind.install_force_copy_patch()
        ca_graph_rebind.install_keepgraph_patch()
        try:
            return super().compile_or_warm_up_model()
        finally:
            ca_graph_rebind.restore_force_copy_patch()
            ca_graph_rebind.restore_keepgraph_patch()
