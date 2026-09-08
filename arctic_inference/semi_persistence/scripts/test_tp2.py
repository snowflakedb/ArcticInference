"""Tensor-parallel (TP=2 + expert parallel) save/restore with GPU migration.

Exercises the TP>1 primitives on a MoE model: the cold start captures on
one pair of GPUs, and the restore places the group on a different pair.

The TP-specific steps are the only difference from the single-GPU flow in
``test_weights.py``:

  * ``cuda_checkpoint()`` auto-inserts ``cleargraph`` + ``destroy_nccl``
    when tensor_parallel_size > 1, so the caller does not.
  * ``reinit_nccl()`` must run after ``cuda_restore``, and before anything
    that runs the model or replays a captured graph (``recapture_graphs``,
    ``generate``).  ``attach`` and ``load_weights`` are CPU-only per rank,
    so they are not constrained by it and run first below.
  * ``recapture_graphs("reuse")`` runs after ``wake_up_kv_cache`` and
    rebinds the preserved decode graphs' baked addresses.

TP size comes from the vLLM config; the ``gpus`` argument is placement
only and must have exactly tensor_parallel_size entries.

Run twice: the first run cold-starts and saves, the second restores.
"""
import os

from arctic_inference.semi_persistence import Instance

# TP2 + EP
config_qwen_35b = {"model": "Qwen/Qwen3.6-35B-A3B", "gpu_memory_utilization": 0.7,
                   "tensor_parallel_size": 2, "enable_expert_parallel": True}

MODEL_DIR = "/data-fast/image-cache/tp2test"

conversation = ["Write an essay about the importance of higher education."]
sampling_params = {"temperature": 0.0, "max_tokens": 800}


def init(inst: Instance, gpus):
    inst.init(gpus=gpus)
    inst.generate(conversation, sampling_params)
    inst.attach()
    inst.stage()
    inst.save_weights()
    inst.detach()
    inst.sleep()
    inst.cuda_checkpoint()  # TP>1: cleargraph + destroy_nccl inside
    inst.criu_dump()  # destroys the instance


def load_(inst: Instance, gpus):
    inst.criu_restore()
    inst.attach()
    inst.load_weights()
    inst.cuda_restore(gpus=gpus)
    inst.reinit_nccl()  # TP>1: rebuild NCCL after CRIU
    inst.wake_up_weights()
    inst.repin()
    inst.restore_weights()
    inst.wake_up_kv_cache()
    inst.recapture_graphs("reuse")  # TP>1: rebind preserved graphs
    inst.generate(conversation, sampling_params)


def main():
    inst1 = Instance(config_qwen_35b, os.path.join(MODEL_DIR, "qwen_35b"))
    cache_hit = os.path.isfile(
        os.path.join(MODEL_DIR, "qwen_35b", "image", "meta.json"))

    if cache_hit:
        print("[test] cache HIT — loading from image")
        load_(inst1, [0, 1])   # migrate onto 0,1 if the image was dumped elsewhere
    else:
        print("[test] cache MISS — cold-starting, saving image")
        init(inst1, [2, 3])

    print("[test] waiting for instances to finish")
    inst1.wait()
    inst1.teardown()


if __name__ == "__main__":
    main()
