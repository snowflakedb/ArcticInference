# ArcticInference architecture — per-feature design

Detail behind [SKILL.md](SKILL.md). Snapshot at the vLLM 0.26.0 baseline; verify
paths/line numbers against the current tree.

## 1. Advanced parallelism (Arctic Ulysses + Shift Parallelism)

**Source:** `arctic_inference/vllm/ulysses.py` (~700 lines), 9 patches via
`apply_shift_parallel_patches()`.

Ulysses is sequence parallelism (SP): world layout becomes `DP x PP x SP x TP`,
splitting the **sequence** dim across SP ranks instead of / in addition to the
head dim across TP. Wins: lower TTFT during prefill (each rank computes only
`seq_len/SP` of attention); higher decode throughput (TP comms are the
bottleneck; SP avoids it for non-attention layers).

Knobs (added by `EngineArgsPatch.add_cli_args`):

| Flag | Default | Effect |
|------|---------|--------|
| `--ulysses-sequence-parallel-size N` | `1` | SP world size; multiplies total world by `N` |
| `--enable-shift-parallel` | `false` | Dynamically switch between full-TP and SP modes |
| `--shift-parallel-threshold N` | `512` | Per-batch token threshold for the SP<->TP decision |

Patches (each an `ArcticPatch`):

| Patch class | Target | What it does |
|-------------|--------|--------------|
| `UlyssesModelConfig` | `vllm.config.ModelConfig` | `num_kv_heads //= sp_size`, layer-range PP indexing |
| `UlyssesParallelState` | `vllm.distributed.parallel_state` | New `_SP` / `_SP_TP` groups; layout `DP x PP x SP x TP` |
| `UlyssesWorkerProc` | `WorkerProc` | SP-aware multiproc worker (extra rank coords) |
| `UlyssesMultiprocExecutor` | `MultiprocExecutor` | World-size = TP*SP*PP; map collectives across SP |
| `UlyssesAttention` | `Attention` | Pre-/post-attention all-to-all between SP and head dims |
| `UlyssesCudagraphDispatcher` | `CudagraphDispatcher` | Two CUDA graph sets: one TP-only, one SP |
| `UlyssesCompilationConfig` | `CompilationConfig` | Compile two model variants when shift-parallel enabled |
| `UlyssesVllmConfig` | `VllmConfig` | Validate compatible config (e.g. SP > 1 if shift on) |
| `UlyssesEngineCore` | `EngineCore` | Inject `is_shift_parallel_mode()` into the step loop |

**Shift parallelism** keeps *two* parallel-state configs (TP-only and SP) and
chooses per step by batch size: `tokens > threshold -> SP` (better prefill),
`tokens <= threshold -> TP` (better decode). The switch happens inside the engine
step (`UlyssesEngineCore`) and is invisible to the caller; CUDA graphs are
pre-captured for both modes.

Where to look: `projects/ulysses/README.md`; step-level decision
`is_shift_parallel_mode()` in `model_runner.py` (~line 96); layer A2A
`UlyssesAttention.forward` in `ulysses.py`.

## 2. Speculative decoding (Arctic Speculator + Suffix Decoding)

**Source:** `arctic_inference/vllm/spec_dec/`, plus `SpeculativeConfigPatch`
(`config.py`) and `AsyncSchedulerPatch` (`patches.py`).

| `method` | What runs | Use case |
|----------|-----------|----------|
| `"arctic"` | Arctic LSTM/MLP speculator drafts -> target verifies | General-purpose, +2-3x tokens/s |
| `"suffix"` | Suffix-tree draft from the request's own history | Code/agentic (long shared spans) |
| `"arctic"` + `enable_suffix_decoding=true` | Hybrid: Arctic fresh tokens, Suffix piggybacks | Repetitive/agentic |

### Arctic Speculator
Registered with `vllm.ModelRegistry`: `ArcticMLPSpeculatorPreTrainedModel` and
`ArcticLSTMSpeculatorPreTrainedModel` -> the classes in
`spec_dec/arctic_speculator.py` (`MLPVariantSpeculatorPreTrainedModel` is a corvo
legacy alias for the LSTM one). Design points:
- Token + embedding speculator (IBM, arXiv:2404.19124): input is
  `last_hidden_state (+) next_token_embedding`, output up to `n_predict=3` future
  token ids per call.
- TP-aware via `SpeculatorTPInit` (`spec_dec/vocab_parallel_embedding.py`): the LM
  head is `ParallelLMHead`-sharded across the TP group, so a TP=N target can use a
  TP=1 speculator without weight duplication.
- Custom CUDA ops (`csrc/custom_ops/`, `py_custom_ops.py`): `speculator_ln` (fused
  L2-LayerNorm for speculator head dims) and `sum_lstm` (fused LSTM-step+reduce).
- FP8-aware: `Fp8ConfigWithEmbedding` (`spec_dec/fp8.py`) loads the speculator
  under the target's FP8 config so drafts match precision.
- CUDA-graph captured per `(padding, head_index)`; key from
  `_generate_cg_key(padding_size, head_index)`.

### ArcticProposer (runtime glue)
`spec_dec/arctic_proposer.py::ArcticProposer` implements vLLM's V1 proposer:
1. `load_model()` — checks the draft model's `architectures` is a registered
   speculator type, optionally checks `base_model_archs` matches target
   (`ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK=1` disables for gpt-oss), builds a
   `VllmConfig` reusing the target's parallel/scheduler/load configs.
2. `propose()` — called by `GPUModelRunner.propose_draft_token_ids` each step;
   returns up to `num_speculative_tokens` candidates per request.
3. `disable_by_batch_size` — proposer only drafts for the first N requests;
   `AsyncSchedulerPatch` matches that cap when allocating spec-token placeholders.

### Suffix Decoding
- Native: `arctic_inference/suffix_decoding/_C...so` (from `csrc/`) — `SuffixTree`
  C++ class, O(L) match per position.
- Python: `suffix_decoding/cache.py::SuffixDecodingCache` keeps a *global* tree of
  all completed responses (cross-request reuse) and *per-request* trees of the
  current prompt+output (intra-request reuse, e.g. agentic loops re-quoting tool
  output).
- Defaults: `SpeculativeConfigPatch.__post_init__` sets
  `num_speculative_tokens = suffix_speculative_tokens or 16` for `method="suffix"`,
  `disable_by_batch_size=64`.

Knobs (`ArcticSpeculativeConfig`): `suffix_cache_max_depth=64`,
`suffix_speculative_tokens=0`, `suffix_cache_max_requests=100_000`,
`suffix_max_spec_factor=1.0`, `suffix_max_spec_offset=0.0`,
`suffix_min_token_prob=0.1`.

### Hybrid mode + the async scheduler patch
For `method="arctic"` + `enable_suffix_decoding=true`,
`SpeculativeConfigPatch.__post_init__` sets
`suffix_speculative_tokens = suffix_cache_max_depth` so suffix drafts extend to
tree depth while Arctic still drafts `n_predict`; both go into
`scheduled_spec_decode_tokens`. The `AsyncSchedulerPatch` (`patches.py` ~45-243):
1. `disable_by_batch_size` placeholder allocation — only first N decode requests
   get spec-token placeholders (matches worker draft cap).
2. Dynamic placeholder sizing — reads the previous step's *actual* draft length
   from model-runner output and sizes next step's placeholders accordingly (avoid
   wasted verification when Arctic drafts 3 but the slot is sized for suffix max).
3. Exponential ramp-up fallback — when `_actual_draft_lens` doesn't survive the
   multiproc round-trip, infer from acceptance (100% -> double; partial -> linear;
   zero -> seed at 1).

Most-tested path in the repo: `tests/unit_tests/test_arctic_spec_max_len.py`,
`test_spec_dec_sleep.py`. Validate it after a rebase via the
`rebase-arctic-inference` skill's Phase-6 / `REBASE_TEST.md` spec-decode ladder.

### Other spec-decode patches
| File | Role |
|------|------|
| `spec_dec/logits_processor_opt.py` | Faster greedy `argmax` path for `ArcticMLPSpeculator` head selection |
| `spec_dec/vocab_parallel_embedding.py` | TP-aware speculator embedding/LM-head (`SpeculatorTPInit`) |
| `spec_dec/fp8.py` | `Fp8ConfigWithEmbedding` so the speculator inherits target's FP8 quant |
| `structured_output.py::XgrammarBackendPatch` | `num_speculative_tokens = max(base, suffix)` for guided-decoding bitmask |
| `stats.py::SpecDecodingStatsPatch` | Grow acceptance-per-position stats when `num_draft_tokens` > cap |
| `stats.py::SpecDecodingLoggingPatch` | Pad `accepted_tokens_per_pos_lists` to uniform length for clean logs |

## 3. Forest Cascade Attention (FCA)

**Source:** `arctic_inference/vllm/attention/`, applied by
`apply_forest_cascade_patches()`. README: `arctic_inference/vllm/attention/README.md`.

Batched-attention optimization for many concurrent requests sharing long KV
prefixes (system prompts, few-shot, agentic tool outputs). Splits each attention
call into: (1) **Prefix FA** — one grouped `flash_attn_varlen_func` over
shared-prefix KV blocks, all requests in a group as one virtual sequence
(`causal=False`); (2) **Suffix FA** — per-request causal call over each request's
unique suffix; (3) **Merge** — log-sum-exp via `merge_attn_states`.

- Drop-in replacement of `vllm.v1.attention.backends.flash_attn.FlashAttentionBackend`
  with `ForestFlashAttentionBackend` (`attention/__init__.py`).
- Always registered; FCA paths fire only when `--forest-cascade-attn-configs` is
  set — otherwise byte-for-byte equivalent to upstream FlashAttention.
- Five JSON knobs (`max_query_len`, `min_group_size`,
  `min_additional_prefix_blocks`, `min_non_singleton_fraction`,
  `max_non_singleton_groups`) with default-on logic (see README).
- Implicit eligibility: `num_reqs >= 8`, `causal=True`, `dcp=1`, full-CUDA-graph
  off, block-aligned global common prefix.
- `forward_includes_kv_cache_update = True` is intentional: 0.18 split
  `reshape_and_cache_flash` out of `forward()` by default, but the FCA impl needs
  the inline behavior, so it opts back in.

## 4. SwiftKV

**Source:** `arctic_inference/vllm/swiftkv/llama_swiftkv.py` (~830 lines) +
`arctic_inference/common/swiftkv/configs.py` (HF config class).

Model rewiring + self-distillation (arXiv:2410.03960): skip KV computation in the
final ~50% of layers, replace with a learned single-layer KV projection -> ~2x
faster prefill + lower KV memory, no quality loss after the SwiftKV fine-tune.

```python
AutoConfig.register("llama_swiftkv", LlamaSwiftKVConfig)
ModelRegistry.register_model("LlamaSwiftKVForCausalLM",
                             "arctic_inference.vllm.swiftkv:LlamaSwiftKVForCausalLM")
```
Any HF config with `model_type: "llama_swiftkv"` loads this path (released model:
`Snowflake/Llama-3.1-SwiftKV-8B-Instruct`). Design: subclasses upstream
`LlamaModel` (standard weight loader works); a `swiftkv_kv_proj` linear per skipped
layer projects the post-attention hidden state straight to K/V; forward forks at
`swiftkv_layer_idx` (layers below normal, above reuse projected KV — only attention
output + MLP run per token); FlashInfer-aware defensive import. **Llama-only** —
other architectures need a parallel implementation file.

## 5. KV cache & memory management

### kvcached prefix caching (elastic block pool)
**Source:** `arctic_inference/vllm/kvcached/`, gated by `KVCACHED_AUTOPATCH=1`.
`PrefixCacheableElasticBlockPool` (`kvcached/prefix_block_pool.py`) replaces
vLLM's internal `BlockPool`, delegating physical block management to
[`kvcached`](https://github.com/snowflakedb/kvcached) (CUDA VMM page alloc/free)
while keeping prefix caching: freed blocks stay in an LRU queue with pages still
mapped (sub-us prefix hits); when `kvcached.available_size()` < 10%
(`pressure_threshold_pct=0.10`) the queue flushes, releasing pages to other
processes on the GPU (multi-tenant sharing); `set_caching_enabled(bool)` toggles
at runtime. Installed by hot-replacing
`KVCacheCoordinator._setup_kvcached_coordinator` with `_setup_with_prefix_cache`
(`kvcached/patches.py`). Substrate for the multi-model server's GPU sharing.

### Sleep / wake_up with drafter preservation
**Source:** `WorkerPatch` in `patches.py` (~lines 262-335). Level 1 frees KV pages
(weights stay); level 2 frees everything and `wake_up()` restores main weights via
`reload_weights()`. Problem: `reload_weights()` only handles the **main** model;
the drafter isn't in the upstream weight-sync registry and its `load_model()`
would allocate outside the `CuMemAllocator` pool. Fix: on level-2 `sleep`, CPU-clone
the drafter state (`_save_module_state`); on `wake_up`, call
`GPUModelRunnerPatch._orig_reload_weights(...)` then restore the drafter. Drafters
are small (100s of MB) so the round-trip is negligible. Verified by
`tests/unit_tests/test_spec_dec_sleep.py`. **FP8 limitation:** upstream
`process_weights_after_loading` swaps `ModelWeightParameter` for plain `Parameter`,
stripping TP-sharding methods `reload_weights()` needs -> level-2 sleep is BF16-only.

## 6. RL / training-loop features

### FP32 LM head
**Source:** `arctic_inference/vllm/fp32_lm_head.py` (~115 lines). RL (PPO/DPO/GRPO)
needs precise log-probs. V1's sampler already softmaxes in fp32, but the `lm_head`
matmul runs in bf16/fp16 — this upcasts the matmul operands on the fly
(`F.linear(hidden.to(fp32), weight.to(fp32), bias.to(fp32))`). No weight
modification, no extra VRAM (no fp32 weight copy); per-step cost is one extra HBM
read of `vocab*hidden*2` bytes. Patches `LogitsProcessor._get_logits` and the
`get_top_tokens` greedy fast path; falls through to upstream for *quantized* LM
heads. Enable `--fp32-lm-head` or `ARCTIC_FP32_LM_HEAD=1`; the CLI flag in
`EngineArgsPatch.__post_init__` exports the env var so worker subprocesses inherit
it before applying patches.

### Rollout replay (per-sequence output length for `n > 1`)
**Source:** `arctic_inference/vllm/sampling.py::ParentRequestPatch`. RL rollout
replay needs each of the `n` sampled sequences to stop at a *prescribed* length
(replaying a recorded trajectory). Upstream `SamplingParams` only has a single
`max_tokens` shared by all `n` children. The patch overrides
`ParentRequest._get_child_sampling_params` (`vllm/v1/engine/parallel_sampling.py`)
so that child `i` gets `max_tokens = max_tokens_n[i]` when the parent request
carries a `max_tokens_n` list; combine with `ignore_eos=True` to force exact
lengths. Requests without `max_tokens_n` are untouched (normal `max_tokens` +
child-param caching preserved).

The lengths travel in `SamplingParams.extra_args["max_tokens_n"]`, **not** a new
field: `SamplingParams` is a `msgspec.Struct` (fixed fields, `__slots__`,
msgspec-encoded across the front-end -> EngineCore boundary), so a plugin cannot
add a real serialized field at runtime — `extra_args` is an existing serialized
field we ride on. Usage:
```python
SamplingParams(n=2, ignore_eos=True, extra_args={"max_tokens_n": [25, 50]})
```
Rebase note: the override mirrors an upstream method body, so it is a Phase-3
behavioral-parity surface — re-diff `_get_child_sampling_params` against the
target `parallel_sampling.py` each bump. (Formerly a standalone
`benchmark/rollout/*.patch`; folded into the plugin.) See
`benchmark/rollout/README.md`.

### NCCL weight sync (training -> inference)
**Source:** `arctic_inference/server/weight_sync/` (5 files). Pushes fresh weights
from a training cluster to inference replicas each training step without blocking
inference (Snowflake RL pipeline, DSS `TrainingJobEngine`).
```
[Training GPU] NCCLEngine.send_weights -> [Inference GPU] NCCLEngine.recv_weights
   -> WeightSyncExtension.recv_into_params -> loaded directly into model params
```
Modes: **bucket** (default, 256 MB, pipelined double-buffered send/recv,
`_nccl_stream` overlaps NCCL with packing); **direct** (BF16 + TP=1, manifest then
per-tensor `nccl.send/recv` into pre-computed views, zero intermediate buffer).
`TransferSchedule.build()` computes `(training_rank) -> [(replica_id, tp_rank),...]`;
DP/FSDP fan out up to `min(training_gpus, R*TP)` senders; ZeRO-3 uses
`GatheredParameters`. Receiver: `server.weight_sync.receiver.WeightReceiver` + a
`WeightSyncExtension` hosted in each `Worker`
(`worker_extension_cls="arctic_inference.server.weight_sync.WeightSyncExtension"`).
HTTP: `GET /weights_info` (manifest), `POST /sync_weights` (open group, receive,
optional hot-swap), `POST /close_weight_sync`. Ray/OPD binds an init-time
trainer dest contract (`bind_weight_sync_contract`) before the first payload;
later syncs check count + name hash. Tests:
`tests/weight_sync/{test_direct_param_writer.py, test_parallel_weight_sync.py, test_weight_sync_contract.py, benchmark_weight_sync.py}`.

## 7. Serving infrastructure

### Multi-model gRPC/HTTP server
**Source:** `arctic_inference/server/` (12 files), console script
`arctic_inference_server` (`server/cli.py`). See `projects/server/README.md`.
```
Client -> FastAPI -> Driver -> ReplicaPool(model-a) -> Scheduler -> Worker 0/1
                          `-> ReplicaPool(model-b) -> Scheduler -> Worker 0
```
| Class | Responsibility |
|-------|----------------|
| `Driver` | Tracks pools, manages GPU allocation, routes by `model_id` |
| `ReplicaPool` | Per-model: owns Ray workers + scheduler, scaling, health, weight sync |
| `Scheduler` | Routes to least-loaded worker; dynamic concurrency control |
| `InferenceWorker` | Ray actor wrapping `vllm.v1.engine.async_llm.AsyncLLM` |
| `Pipeline` | Dataset-processing wrapper with managed concurrency |

GPU sharing: loading a new model auto-rebalances pools
(`Driver._compute_even_share`) — e.g. model-a 4 replicas -> load model-b -> each
2 -> shutdown model-a -> model-b back to 4. Dynamic concurrency:
`scheduler.py::utilization_based_concurrency` uses KV-cache utilization (grow < 90%,
back off > 95%, random-probe ramp-up); default `least_loaded_routing`. Pre-tuned
`server/config.py::MODEL_CONFIGS` for common H200/B200 Qwen3 models. HTTP:
`/init`, `/generate`, `/sleep` `/wake_up`, `/weights_info`, `/sync_weights`,
`/close_weight_sync`, `/status`, `/shutdown`.

### Embeddings server
**Source:** `arctic_inference/embedding/` (5 files), README
`arctic_inference/embedding/README.md`. Separate gRPC stack for embedding
workloads (short, parallel requests): proto in `embedding/proto/` (generated via
`python arctic_inference/embedding/generate_proto.py`); `replica_manager.py`
launches N replicas and load-balances (round-robin / random / least-loaded);
"dedicated" vs "shared" GPU assignment.
```bash
python -m arctic_inference.embedding.replica_manager \
    --model Snowflake/snowflake-arctic-embed-m-v1.5 \
    --num-replicas 32 --load-balancing round_robin
```

### Dynasor (reasoning early-stop)
**Source:** `arctic_inference/dynasor/` (7 files), README `projects/dynasor/README.md`.
Proxy in front of any OpenAI-compatible vLLM deployment; periodically *probes* the
running generation and cuts it early once the model has converged (entropy below
threshold for `certainty_window` consecutive probes) — saves 30-70% decode tokens
on R1-style CoT. Modes: one-shot (`arctic_inference.dynasor.vllm_server`) and
front-end proxy (`arctic_inference.dynasor.openai_server`). Per-request knobs via
OpenAI `extra_body.dynasor`: `probe_interval`, `certainty_window`. Impl: `cot.py`,
`entropy.py`, `openai_server.py`, `vllm_server.py`.

## Ancillary patches / infrastructure
| Patch / module | Purpose |
|----------------|---------|
| `EngineArgsPatch`, `AsyncEngineArgsPatch` | Wrap `EngineArgs` -> `ArcticEngineArgs` (adds five Arctic CLI flags) |
| `ParallelConfigPatch` | Wrap `ParallelConfig` -> `ArcticParallelConfig` (SP fields) |
| `SpeculativeConfigPatch` | Wrap `SpeculativeConfig` -> `ArcticSpeculativeConfig` (suffix knobs, hybrid) |
| `MLPSpeculatorConfigPatch` | Inject dummy `num_attention_heads`/`hidden_size` for vLLM's arch convertor |
| `VllmConfigPatch` | Add Arctic fields to `VllmConfig.__str__`; extend `EagleModelTypes` literal |
| `WorkerBasePatch` | Lazy-apply `GPUModelRunnerPatch`/`WorkerPatch` post-fork (CUDA-init safety) |
| `arctic_inference/envs.py` | `ARCTIC_INFERENCE_ENABLED`, `_SKIP_PLATFORM_CHECK`, `_SKIP_VERSION_CHECK`, `_SKIP_SPEC_MODEL_CHECK`, `ARCTIC_FP32_LM_HEAD` |
| `arctic_inference/utils.py::get_compatible_vllm_version` | Reads pyproject `vllm==X.Y.Z` pin for the runtime version check |
| `arctic_inference/py_custom_ops.py` | Loads compiled CUDA ops (`speculator_ln`, `sum_lstm`); PyTorch fallback |
| `arctic_inference/op_builder/` | CUDA-op JIT build helpers |
| `arctic_inference/semi_persistence/` | KV-cache offload-to-CPU experiment (off by default) |

## References
- Patch entry point: `arctic_inference/vllm/plugin.py::arctic_inference_plugin`
- Patch composition: `arctic_inference/vllm/patches.py::apply_arctic_patches`
- Patch framework: `arctic_inference/patching.py::ArcticPatch`
- Per-feature READMEs: `arctic_inference/vllm/attention/README.md`,
  `arctic_inference/embedding/README.md`, `projects/server/README.md`,
  `projects/spec_dec/README.md`, `projects/swiftkv/README.md`,
  `projects/ulysses/README.md`, `projects/dynasor/README.md`
- Papers: arXiv:2507.11830 / 2509.16495 (Shift Parallelism), 2411.04975 (Suffix
  Decoding), 2410.03960 (SwiftKV), 2412.20993 (Dynasor), 2404.19124 (MLP Speculator)
