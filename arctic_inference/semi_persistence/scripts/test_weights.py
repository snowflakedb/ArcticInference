"""Save/restore with the weights kept outside the CRIU image.

Contrast with ``test_image.py``, which leaves the staged weights inside
the image: here ``save_weights()`` writes them to ``<model_dir>/weights``
and ``detach()`` frees the pinned buffer before the dump, so the image
stays small.  The restore side rebuilds the buffer with ``attach()`` and
refills it with ``load_weights()``.

Two models are driven concurrently to exercise parallel dump/restore.
Run twice: the first run cold-starts and saves, the second restores.
"""
import os

from arctic_inference.semi_persistence import Instance

config_qwen_27b = {"model": "Qwen/Qwen3.8-27B", "gpu_memory_utilization": 0.7, "max_num_seqs": 512}
config_qwen_35b = {"model": "Qwen/Qwen3.6-35B-A3B", "gpu_memory_utilization": 0.7}
MODEL_DIR = "/data-fast/image-cache"

conversation = ["Write an essay about the importance of higher education."]
sampling_params = {"temperature": 0.0, "max_tokens": 800}


def init(inst: Instance, gpu=0):
    inst.init(gpu=gpu)
    inst.generate(conversation, sampling_params)
    inst.attach()
    inst.stage()
    inst.save_weights()
    inst.detach()
    inst.sleep()
    inst.cuda_checkpoint()
    inst.criu_dump()  # destroys the instance


def load(inst: Instance, gpu=0):
    inst.criu_restore()
    inst.cuda_restore(gpu=gpu)
    inst.attach()
    inst.load_weights()
    inst.wake_up_weights()
    inst.repin()
    # Force a small (4 GiB) weight-restore staging buffer.  This image was
    # dumped before the empty_cache() fix, so its checkpointed
    # restore_weights leaves the staging buffer in torch's caching allocator
    # (not returned to the driver); a large chunk would then starve
    # wake_up_kv_cache and OOM.  A small chunk stays negligible.
    inst.plan_restore_weights(max_buffer_bytes=4 * 1024**3)
    inst.restore_weights()
    inst.wake_up_kv_cache()
    inst.generate(conversation, sampling_params)


def main():
    inst1 = Instance(config_qwen_27b, os.path.join(MODEL_DIR, "qwen_27b"))
    inst2 = Instance(config_qwen_35b, os.path.join(MODEL_DIR, "qwen_35b"))
    cache_hit = (
        os.path.isfile(os.path.join(MODEL_DIR, "qwen_27b", "image", "meta.json"))
        and os.path.isfile(os.path.join(MODEL_DIR, "qwen_35b", "image", "meta.json"))
    )

    if cache_hit:
        print("[test] cache HIT — loading from image")
        load(inst1, 5)
        load(inst2, 6)
    else:
        print("[test] cache MISS — cold-starting, saving image")
        init(inst1, 2)
        init(inst2, 3)
        print("[test] image saved")

    inst1.wait()
    inst2.wait()


if __name__ == "__main__":
    main()
