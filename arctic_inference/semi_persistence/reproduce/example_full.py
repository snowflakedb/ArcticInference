#!/usr/bin/env python3

import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("VLLM_ALLREDUCE_USE_SYMM_MEM", "0")

from instance import Instance

MODELS_DIR = "/data-fast/hf_models"
IMAGE_ROOT = "/data-fast/image-cache/reproduce_example_full"
PROMPT = "Hi canyou introduce yourself?"
SAMPLING = {"max_tokens": 32, "ignore_eos": True, "temperature": 0.0}

# label, model_id, tp, gpu_memory_utilization, gpus, extra
MODEL_SPECS = [
    # 122b / 397b are hybrid-Mamba: vLLM 0.24's cudagraph check needs
    # max_num_seqs <= available Mamba cache blocks, so cap it (else init fails).
    # The near-card-filling models (122b/235b/397b) need util=0.9 to fit weights.
    ("Qwen3.5-122b-a10b-tp2", "Qwen/Qwen3.5-122B-A10B",  2, 0.9, [0, 1], {"max_num_seqs": 128}),
    ("Qwen3-32b-tp4",         "Qwen/Qwen3-32B",          4, 0.8, [0, 1, 2, 3], {}),
    ("Qwen3-235B-A22B-tp4",   "Qwen/Qwen3-235B-A22B",    4, 0.9, [0, 1, 2, 3], {}),
    ("Qwen3.5-397b-a17b-tp8", "Qwen/Qwen3.5-397B-A17B",  8, 0.9, list(range(8)), {"max_num_seqs": 128, "kernel_config": {"moe_backend": "triton", "enable_flashinfer_autotune": False}}),
    ("Qwen3.5-2B-tp1",        "Qwen/Qwen3.5-2B",         1, 0.2, [0], {}),
    ("Llama3.1-8B-tp1",       "meta-llama/Llama-3.1-8B", 1, 0.4, [0], {}),
    ("Qwen3.5-27B-tp1",       "Qwen/Qwen3.5-27B",        1, 0.8, [0], {}),
    ("Qwen3-30B-A3B-tp1",     "Qwen/Qwen3-30B-A3B",      1, 0.8, [0], {}),
]


def cleanup(inst):
    inst._cmd_queue.put(("teardown", {}))
    worker = getattr(inst, "_worker", None)
    if worker is not None:
        worker.join(timeout=15)
        if worker.is_alive():
            worker.kill()
    Instance._all.pop(inst.instance_id, None)


def run_one(label, model_id, tp, util, gpus, extra):
    local = os.path.join(MODELS_DIR, model_id)
    config = {"model": local if os.path.exists(local) else model_id,
              "gpu_memory_utilization": util, "enforce_eager": False,
              "tensor_parallel_size": tp, **extra}
    model_dir = os.path.join(IMAGE_ROOT, label)
    shutil.rmtree(model_dir, ignore_errors=True)
    os.makedirs(model_dir, exist_ok=True)
    print(f"\n{'=' * 70}\n{label}  {model_id}  tp={tp}  gpus={gpus}\n{'=' * 70}",
          flush=True)

    inst = Instance(dict(config), model_dir)
    times = {}

    print("== up ==", flush=True)
    inst.init(gpus=gpus).attach().repin().stage()
    inst.generate([PROMPT], SAMPLING).wait()
    print(f"  answer after cold start: {str(inst.last_generate_result)[:80]!r}",
          flush=True)

    print("\n== saved -> up ==", flush=True)
    inst.save_weights().detach().wait()

    t0 = time.perf_counter()
    inst.sleep().wait()
    t1 = time.perf_counter()
    times["sleep"] = t1 - t0

    # cuda_checkpoint already does cleargraph + destroy_nccl when tp>1.
    t0 = time.perf_counter()
    inst.cuda_checkpoint().wait()
    t1 = time.perf_counter()
    times["checkpoint_cuda"] = t1 - t0

    inst.criu_dump().wait()
    cleanup(inst)
    time.sleep(2)
    image_dir = os.path.join(model_dir, "image")
    if os.path.isdir(image_dir):
        size = sum(f.stat().st_size for f in os.scandir(image_dir) if f.is_file())
        print(f"  image on disk: {size / 2**30:.2f} GiB", flush=True)

    inst = Instance(dict(config), model_dir)
    inst.criu_restore().wait()

    t0 = time.perf_counter()
    inst.cuda_restore(gpus=gpus).wait()
    t1 = time.perf_counter()
    times["restore_cuda"] = t1 - t0

    t0 = time.perf_counter()
    inst.reinit_nccl().wait()
    t1 = time.perf_counter()
    times["reinit_nccl"] = t1 - t0

    inst.attach().load_weights().wait()

    t0 = time.perf_counter()
    inst.wake_up_weights().wait()
    t1 = time.perf_counter()
    times["wake_up_weights"] = t1 - t0

    inst.repin().plan_restore_weights().wait()

    t0 = time.perf_counter()
    inst.restore_weights().wait()
    t1 = time.perf_counter()
    times["restore_weights"] = t1 - t0

    t0 = time.perf_counter()
    inst.wake_up_kv_cache().wait()
    t1 = time.perf_counter()
    times["wake_up_kv_cache"] = t1 - t0

    t0 = time.perf_counter()
    inst.recapture_graphs().wait()
    t1 = time.perf_counter()
    times["recapture_graphs"] = t1 - t0

    inst.generate([PROMPT], SAMPLING).wait()
    print(f"  answer after saved: {str(inst.last_generate_result)[:80]!r}",
          flush=True)
    print("  times:", {k: round(v, 3) for k, v in times.items()}, flush=True)
    cleanup(inst)
    return times


def main():
    if os.geteuid() != 0:
        sys.exit("CUDA checkpoint process must run as root ")
    want = set(sys.argv[1:])
    known = {s[0] for s in MODEL_SPECS}
    if want - known:
        sys.exit(f"unknown label(s); known: {sorted(known)}")
    specs = [s for s in MODEL_SPECS if not want or s[0] in want]

    failed = []
    recap = []
    for spec in specs:
        t0 = time.time()
        try:
            times = run_one(*spec)
            recap.append((spec[0], times))
            print(f"[{spec[0]}] OK in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            failed.append(spec[0])
            short = str(e).split("\n", 1)[0][:120]
            print(f"[{spec[0]}] FAILED after {time.time() - t0:.0f}s: "
                  f"{type(e).__name__}: {short}", flush=True)
    print("\n== times ==", flush=True)
    for label, times in recap:
        print(f"  {label}: { {k: round(v, 3) for k, v in times.items()} }",
              flush=True)
    print(f"\ndone: {len(specs) - len(failed)}/{len(specs)} ok", flush=True)
    print("== run finished ==", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
