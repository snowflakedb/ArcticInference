# Semi-Persistence

Fast model swapping for vLLM: keep model weights resident in a pinned CPU-memory
pool, let their GPU copies come and go with demand, and restore a model onto a
GPU in about a second instead of paying a cold start.

Read the announcement:
[**Semi-Persistence: Blazing-Fast Model Swapping for Complex Scheduling**](https://www.snowflake.com/en/blog/engineering/semi-persistence-gpu-model-swapping/)
(Snowflake AI Research, Aug 2026).

<img src="assets/figure1-sleep-wake-latency.png" alt="End-to-end sleep and wake-up latency: vLLM level-2 sleep vs. Semi-Persistence, across eight models from 2B to 397B parameters." width="900">

*End-to-end sleep and wake-up latency across eight open-weight models. Compared
with vLLM's level-2 sleep mode under otherwise identical configuration,
Semi-Persistence is 5.6x to 19.9x faster: 214-801 ms for single-GPU models
(vs. 1.2-13.5 s) and 1.75-7 s for multi-GPU models (vs. 10.5-40.6 s). Measured
on AWS p5en.48xlarge (192 vCPU, 2 TiB host memory, 8x H200) with vLLM v0.18.0.*

## How it works

- **Skeletons are separated from weights.** The model skeleton (structure plus
  runtime state, typically a few GB) is checkpointed with CRIU and the CUDA
  checkpoint driver; the weights live in a long-lived pinned CPU pool in vLLM's
  native format, so any compatible skeleton can restore them without conversion.
- **Weights are restored at node bandwidth.** Weights are sharded across the
  node, streamed from the pinned pool through multiple GPUs over distinct PCIe
  switches, and gathered over NVLink, rather than through a single PCIe path.
- **Orchestration is asynchronous.** Each instance is an independent state
  machine walking the ladder `saved <-> checkpoint <-> sleep <-> up`; a global
  reservation system hands out fractional GPU slots (quarter, half, full, or
  multiple GPUs) so instances move concurrently without running out of memory.
- **Scheduling behaviours emerge.** Eviction, migration, consolidation, and
  pause/resume fall out of a few per-instance rules, which lets the system
  self-heal fragmentation and head-of-line blocking.

## Requirements

- Linux with NVIDIA GPUs, and a driver providing `cuda-checkpoint`.
- vLLM and ArcticInference installed (see the [repository README](../../README.md)).
- CRIU 4.2 with the CUDA plugin. The empty plugin directory it needs,
  `/usr/lib/criu/empty`, is created by the dump if missing.
- Passwordless `sudo`: checkpoint and restore shell out to `cuda-checkpoint`
  and `criu`, which need root.

Full setup, including a from-source CRIU build for hosts that cannot reach the
Ubuntu PPA, is in [`skills/INSTALL.md`](skills/INSTALL.md).

## Get started

All commands run from this directory. Modules import their siblings by bare name
(`import worker`), so scripts either run from here or bootstrap `sys.path`
themselves; the three public classes are also re-exported lazily from the
package.

### 1. Build an image cache

Registering a model cold-starts it once, checkpoints the skeleton to disk, and
pins its weights in CPU memory. Everything after this point restores from that
image.

```bash
cd arctic_inference/semi_persistence
python scripts/register.py          # edit the configs inside to taste
```

### 2. Run the orchestrator server

The server discovers every model in the image cache and exposes the HTTP control
plane and `/state` endpoint on `--port` (8157 by default).

```bash
sudo python orch_server.py \
    --image-cache /data-fast/image-cache/demo \
    --gpus 1,2,3 \
    --port 8157
```

### 3. Drive it from a client

`OrchestratorClient` is job-keyed: `init` optionally binds a session file so
bindings survive across interpreter restarts.

```python
from arctic_inference.semi_persistence import OrchestratorClient as client

client.init("demo_hol.jsonl")

client.generate("32b bird", "who are you?", 2000)
client.generate("model 8", "what is the capital of France?", 3000)
client.generate("model 10", "What is the meaning of life?", 6000)

client.pause("model 8")                       # frees its slot mid-generation
client.generate("spec 30b", "who are you?", 2000)
client.generate("spec 8b", "who are you?", 3000)
client.resume("model 8")                      # migrates, re-prefills, continues

client.wait()
```

### 4. Watch it live

The dashboard polls `/state` and renders slot occupancy, instance states, the
pinned pool, and the request log.

```bash
python dashboard.py --port 8157 --interval 0.1
```

## Using the primitives directly

The orchestrator is one policy built on top of `Instance`, which exposes the
semi-persistent lifecycle as chainable non-blocking primitives. Use it directly
to add semi-persistence to another serving stack:

```python
from arctic_inference.semi_persistence import Instance

inst = Instance({"model": "Qwen/Qwen3-8B-FP8", "enforce_eager": True})

inst.init(gpu=0).attach().stage().sleep().cuda_checkpoint()
inst.criu_dump("/data-fast/image-cache/qwen3-8b").wait()

inst.cuda_restore(gpu=3).wake_up_weights().restore_weights().wake_up_kv_cache()
inst.generate(["Hello, world!"], {}).wait().print_status()
```

`scripts/test_image.py` is the minimal save-then-restore walkthrough;
`scripts/main_test.py` shows five instances being multiplexed across two GPUs by
hand. `scripts/test_weights.py` keeps the weights outside the image
(`save_weights` / `load_weights`, so the dump runs against a detached, small
process), and `scripts/test_tp2.py` does the same at tensor_parallel_size=2 with
expert parallel, cold-starting on one GPU pair and restoring onto another.

## Repository layout

| Path | Contents |
|---|---|
| `instance.py`, `worker.py`, `vllm_child.py` | The instance handle and the spawned worker / vLLM child that own the GPU |
| `orchestrator.py`, `pipeline.py`, `slots.py` | State ladder, per-model op pipeline, buddy allocator for GPU slots |
| `orch_server.py`, `client.py`, `state_server.py` | HTTP control plane, job-keyed client, `/state` endpoint |
| `dashboard.py` | Live curses dashboard over the `/state` endpoint |
| `scripts/` | Imperative repros; these need real GPUs and real weights |
| `tests/` | CPU-only pytest suite, hermetic, no GPU required |
| `skills/` | Design docs and install notes |

## Documentation

| Doc | Covers |
|---|---|
| [`skills/SKILL.md`](skills/SKILL.md) | Orientation map: layers, process hierarchy, state ladder, gotchas |
| [`skills/INSTALL.md`](skills/INSTALL.md) | CRIU install (PPA and from source), draft model sync |
| [`skills/instance_DESIGN.md`](skills/instance_DESIGN.md) | Primitives table, pin management, pause/resume |
| [`skills/orchestrator_DESIGN.md`](skills/orchestrator_DESIGN.md) | State machine, slot allocation, eviction policy |
| [`skills/pipeline_DESIGN.md`](skills/pipeline_DESIGN.md) | Op model, interrupts, cross-model eviction |
| [`skills/slots_DESIGN.md`](skills/slots_DESIGN.md) | Buddy allocator algorithms and invariants |
| [`skills/client_DESIGN.md`](skills/client_DESIGN.md) | Job/model split, calling shapes, session persistence |
| [`skills/CRIU_PLUMBING.md`](skills/CRIU_PLUMBING.md) | The CRIU complications and the FD keep-list |
| [`skills/async_generate_DETAILS.md`](skills/async_generate_DETAILS.md) | Async generate, IPC protocol, drain points |

## Tests

```bash
cd arctic_inference/semi_persistence
python -m pytest tests/ -q      # CPU-only, no GPU needed
```

## Notes and limitations

- This is **not** a KV-cache offload feature. Checkpointing captures the whole
  CUDA context and leaves no GPU residency behind; the on-disk form is a CRIU
  image of the entire child process tree.
- A CRIU dump is destructive: the child is killed once the image is written, so
  after `criu_dump` the model is `saved` with no live process.
- The orchestrator and its HTTP control plane are experimental research code
  with no authentication or per-user isolation. Do not expose the control port
  on an untrusted network.
