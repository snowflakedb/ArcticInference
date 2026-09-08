#!/usr/bin/env bash
#
# Regenerates agent_ressources/docs/ and the skills' copies of those docs.
# Run when upstream moves: a new CUDA toolkit, a new PTX ISA, a new paper.
#
# One line per document. Run a single line to refresh one document; copy a line
# to add one. Output is committed, so `git diff` is the changelog.
#
# Needs uv, and texmark:
#   cargo install --git https://github.com/snowflakedb/ArcticInference texmark

set -euo pipefail
cd "$(dirname "$0")"

# Tools checked up front: the papers section clears its output before writing.
command -v uv >/dev/null || { echo "need uv: https://docs.astral.sh/uv/" >&2 && exit 1; }
command -v texmark >/dev/null || {
    echo "need texmark: cargo install --git https://github.com/snowflakedb/ArcticInference texmark" >&2
    exit 1
}

N=../docs/nvidia
P=../docs/papers

doc() { uv run --quiet --with beautifulsoup4 python -m nvidia "$@"; }
blog() { uv run --quiet --with beautifulsoup4 python -m nvidia_blog "$@"; }
pdf() { curl -fsSL "$1" -o "$2" || echo "WARNING: fetch failed, keeping committed $(basename "$2")" >&2; }

# ---------------------------------------------------------------- papers ------
# texmark converts the arXiv LaTeX source, not the PDF. Wiped first so a stale
# figure can't survive a rerun. A 429 means arXiv throttled a burst: rerun the line.

rm -rf $P/*/

texmark --arxiv 2205.14135 -o $P/FlashAttention_Fast_and_Memory_Efficient_Exact_Attention_with_IO_Awareness
texmark --arxiv 2307.08691 -o $P/FlashAttention_2_Faster_Attention_with_Better_Parallelism_and_Work_Partitioning
texmark --arxiv 2407.08608 -o $P/FlashAttention_3_Fast_and_Accurate_Attention_with_Asynchrony_and_Low_precision
texmark --arxiv 2603.05451 -o $P/FlashAttention_4_Algorithm_and_Kernel_Pipelining_Co_Design_for_Asymmetric_Hardware_Scaling

# texmark names the Markdown after the LaTeX main file; count_papers wants paper.md.
for p in $P/*/; do [ -f "$p/paper.md" ] || mv "$p"/*.md "$p/paper.md"; done

# ------------------------------------------------------------------ docs ------
# The Sphinx theme is auto-detected per site. Don't pass --style: pinning the wrong
# one silently truncates a multi-page doc to its first page (the CUDA guide goes
# 1.5MB -> 4KB, exit 0). Detection raises if it can't tell, which is the safe
# failure, and only then is --style the answer.
#
# torch_nvrtc.md and how_to_read_ptx_docs.md live in $N but are hand-written —
# nothing below regenerates them, so don't wipe the directory.

doc https://docs.nvidia.com/cuda/parallel-thread-execution/ -o $N/ptx.md
doc https://docs.nvidia.com/cuda/cuda-programming-guide/index.html -o $N/cuda.md
doc https://docs.nvidia.com/cuda/inline-ptx-assembly/ -o $N/inline_ptx_assembly.md
doc https://docs.nvidia.com/cuda/blackwell-tuning-guide/ -o $N/blackwell_tuning_guide.md
doc https://docs.nvidia.com/cuda/blackwell-compatibility-guide/ -o $N/blackwell_compatibility_guide.md
doc https://docs.nvidia.com/cuda/hopper-tuning-guide/ -o $N/hopper_tuning_guide.md
doc https://docs.nvidia.com/cuda/hopper-compatibility-guide/ -o $N/hopper_compatibility_guide.md
doc https://docs.nvidia.com/nsight-compute/ProfilingGuide/ -o $N/nsight_compute_profiling_guide.md
doc https://docs.nvidia.com/nsight-compute/NsightCompute/ -o $N/nsight_compute_ui.md
doc https://docs.nvidia.com/nsight-compute/NsightComputeCli/ -o $N/nsight_compute_cli.md
blog https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/ -o $N/blackwell_ultra_blog.md

# --split-... also writes one file per tool; the binary-utilities skill uses those.
doc https://docs.nvidia.com/cuda/cuda-binary-utilities/contents.html -o $N/cuda_binary_utilities.md --split-cuda-binary-utilities $N/cuda_binary_utilities

# Published as PDFs rather than docs sites. Keep the links unsigned — a signed URL
# expires. To find a fresh id, read the landing page and take the asset link it
# repeats most, then confirm by page count (brief 31, datasheet 7):
#   https://resources.nvidia.com/en-us-blackwell-architecture
#   https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet
pdf https://dam-cdn.nvd.orangelogic.com/AssetLink/gl2l4l4812s5fw0p614s6i8bv6mi3vx5.pdf $N/nvidia-blackwell-architecture-technical-brief.pdf
pdf https://dam-cdn.nvd.orangelogic.com/AssetLink/1k0p832eq8r5ca0u5383ie5o4tp3bst1.pdf $N/nvidia-blackwell-ultra-datasheet.pdf

# ---------------------------------------------------------------- skills ------
# Each skill is mounted on its own, so its references are copies, not links. Every
# copy that exists is refreshed from the matching filename under docs/, so adding a
# reference to a skill needs no edit here. One with no counterpart is an error.

for ref in ../skills/*/references/*.md; do
    src=$(find ../docs -type f -name "$(basename "$ref")" -print -quit)
    [ -n "$src" ] || {
        echo "no docs/ source for $ref" >&2
        exit 1
    }
    cp "$src" "$ref"
done

echo "done — review with: git -C .. diff --stat"
