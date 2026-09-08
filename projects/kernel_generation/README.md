# kernelguy

A domain-specific autonomous-agent harness for developing CUDA kernels.

## Requirements

A Linux host with an NVIDIA GPU, the CUDA 13 toolkit, and:

- Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- bubblewrap: `sudo apt install bubblewrap -y`
- [uv](https://docs.astral.sh/uv/), for the Python side
- GPU performance counters readable by non-root users — every candidate kernel is
  profiled with `ncu`

Python dependencies live in `pyproject.toml`, so one command builds the `.venv/` at
the repo root, which is then picked up automatically with no activation needed:

```sh
uv sync                  # evaluator + problems: what a run needs
uv sync --extra vendor   # also vendor_solutions/ (flashinfer, FlashAttention 4)
```

Do not install these by hand. `uv pip install` resolves one package at a time and
will break constraints the lockfile keeps: torch pins `nvidia-cuda-cupti` to one
version and a plain `uv pip install cupti-python` upgrades it out from under torch,
which then still imports and appears to work.

Startup validation exercises all of this inside the agent sandbox and refuses to
start a run if any of it is missing, so problems surface before a run burns time.
See [Troubleshooting](#troubleshooting) when it does.

## Quick start

### Credentials

kernelguy does not implement any login flow. Each provider file names an auth
command, and kernelguy runs it and reads a token from its stdout — nothing more.
The shipped Snowflake Cortex provider uses
[cortex-oauth-helper](https://github.com/Snowflake-AI-Research/cortex-oauth-helper), so
set that up per its own README; anything that prints a bare token to stdout works
just as well.

Configuration for that command — which account or role it authenticates against —
belongs to the command, not here. See its documentation.

### Providers and models

Providers are JSON files in `~/.kernelguy/providers/`. On first launch the
defaults from `src/default_providers/` are copied there, and existing files are never
overwritten, so your edits survive upgrades.

`--provider` names one of those files without the `.json`, and `--model` must name a
model that file lists — the listing endpoint determines both the wire protocol and
the context window, so adding a model or retargeting a host is a file edit, not a
rebuild. An unknown slug is an error that prints what *is* served.

See [docs/providers.md](docs/providers.md) for the file format, the three auth
forms, and custom headers.

### Launch a run

```sh
# Optimize GEMM for one hour.
cargo run -r -- avo problems/gemm.py \
  --provider snowflake_cortex --model claude-opus-5 \
  --max-wall-secs 3600

# Launch the current 10-hour batched-attention run. (**Production Run**)
cargo run -r -- avo problems/batched_attention.py \
  --policy ucb --provider snowflake_cortex --model claude-opus-5 \
  --beam-width 4 --ucb-c 0.5 \
  --max-wall-secs 36000 > batched_attn_ucb_10h_$(date +%Y%m%d_%H%M%S).log 2>&1

# Resume a previous run.
cargo run -r -- avo --resume <run-id>

# Show every search, model, supervisor, and diversity option.
cargo run -r -- avo --help
```

Included workloads are `problems/gemm.py`, `problems/attention.py`,
`problems/batched_attention.py`, and `problems/glm_sparse_mla.py` — the last being
GLM-5.2's sparse MLA decode (DeepSeek Sparse Attention) at SGLang's GB300
low-latency shapes, where the top-2048 selection is an input rather than something
the kernel computes. Results and resumable search history are written under
`runs/<run-id>/`.

### Vendor solutions

`vendor_solutions/` holds the tuned implementations a problem is measured *against*
— the number to beat, not a starting point. Each one wraps a real production kernel
and runs through the ordinary evaluator:

```sh
python scripts/evaluate.py problems/attention.py \
    vendor_solutions/attention_fa4.py --stage full
```

| file | problem | kernel |
| --- | --- | --- |
| `attention_fa4.py` | `attention.py` | FlashAttention 4 (`flash_attn.cute`) |
| `attention_torch_sdpa.py` | `attention.py` | cuDNN SM100 flash, via torch SDPA |
| `glm_sparse_mla_sglang.py` | `glm_sparse_mla.py` | TRT-LLM-gen sparse MLA, autotuned |

Do not confuse these with a problem's `Reference` class, which is the deliberately
naive fp32 oracle and the speedup denominator. These are the opposite end.

**They are not mounted into the sandbox.** `build_mount_spec` stages only
`agent_ressources/{docs,skills}`, the problem file, and three named files out of
`scripts/`, so the agent never sees them — which is the point. Each file names the
extra dependency it needs in its docstring; none is required for a normal run.

### Selecting GPUs

Set `CUDA_VISIBLE_DEVICES`, either in the environment or in a `.env` at the repo
root, to restrict a run to specific devices. The eval device pool is sized to the
visible set, and each eval is pinned to one of those devices.

Prefer UUIDs (`nvidia-smi -L`) over indices: `nvidia-smi` enumerates in PCI-bus
order while the CUDA runtime defaults to fastest-first, so the same index can name
different cards to each.

## Visualize a run

```sh
# Score, search tree, timeline, and text summary.
python3 scripts/viz_run.py runs/<run-id>

# Every configuration on the device roofline, plus throughput over the run.
python3 scripts/viz_roofline.py runs/<run-id>
```

The generated PNGs are written into the run directory.

`viz_roofline.py` works for any problem that defines `flops` and `bytes_moved`: it
derives each configuration's arithmetic intensity from the recorded throughputs and
puts it on the axis that actually binds it, so nothing depends on what the
configurations are named. It also prints the per-configuration table it plots.

## Generate docs/skills

`agent_ressources/docs/` is mounted read-only at `/workspace/docs/` for the agent,
and `agent_ressources/skills/` supplies its startup skills. Most of that tree is
scraped from upstream and committed, so it needs a refresh whenever a new CUDA
toolkit or PTX ISA ships:

```sh
./agent_ressources/.generate/generate.sh
```

Prerequisites are [uv](https://docs.astral.sh/uv/) — which fetches the one Python
dependency per-run, so there is nothing to install first — and, for the papers,
`cargo install --git https://github.com/snowflakedb/ArcticInference texmark` **(available in ./tools/texmark)**.

The script is one line per document. To refresh a single doc run just its line, and
to add one copy a line. Everything it writes is committed, so `git diff --stat`
after a run is the changelog and `git checkout` is the undo.

Review that diff rather than committing it blind — upstream occasionally degrades a
page, so a doc that got much **smaller** is worth a look.

See [docs/regenerating_resources.md](docs/regenerating_resources.md) for what
regenerates versus what is hand-written, how to add or retarget a document, and what
else to check in the diff.

## Troubleshooting

Startup validation runs the whole toolchain inside the agent sandbox — required
tools on `PATH`, tool versions, `torch.cuda.is_available()`, an `nvcc` compile, a
`load_inline` build, `cuobjdump`/`nvdisasm`, and an `ncu` profile — and reports
the exact command it ran on failure.

### Validation fails on the PyTorch check

Check which interpreter was selected, printed at startup:

```
kernelguy: agent python3 from /path/to/kernel-proj/.venv/bin
```

It resolves `$VIRTUAL_ENV`, then `<repo>/.venv`, then `python3` on `PATH` — so a
system `python3` without `torch` is the usual cause.

### `ncu` fails with `ERR_NVGPUCTRPERM`

Performance counters are admin-only by default. To make them readable by all
users:

```sh
sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" \
  > /etc/modprobe.d/nvidia-profiling.conf'
sudo update-initramfs -u && sudo reboot
```

### `load_inline` fails its arch probe with `Invalid device id`

On hosts whose GPUs cannot all be used at once, the CUDA runtime initializes fewer
devices than NVML reports. Mask down to the usable devices with
`CUDA_VISIBLE_DEVICES` (see [Selecting GPUs](#selecting-gpus)).

## Acknowledgements

kernelguy's harness borrows design from three open-source coding agents. None is
a dependency and no code is vendored:

- [pi](https://github.com/earendil-works/pi) — the layered harness shape, and the
  compaction defaults it settled on (`keepRecentTokens`, and a real-window trigger
  of `contextWindow - reserveTokens`).
- [opencode](https://github.com/anomalyco/opencode) — the context accounting for
  that trigger (`usable() = limit.input - min(20k, maxOut)`) and the summarizer
  overflow self-guard, so the call that shrinks the context cannot itself
  overflow.
- [tau](https://github.com/huggingface/tau) — read alongside pi for module
  layering. The two place the provider's wire vocabulary differently, and that
  contrast is what settled kernelguy's own arrangement: the transport types belong
  to the layer that serializes them (`src/ai/protocol.rs`), not to the tool layer
  above it.

Where pi and opencode informed specific mechanisms, they are cited at the code:
`src/orchestrator/compaction.rs` and the full-list `todo_write` in
`src/tool/todo.rs`.
