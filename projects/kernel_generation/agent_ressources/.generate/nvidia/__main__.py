"""CLI for the NVIDIA docs Markdown recovery package.

Examples::

    python -m nvidia https://docs.nvidia.com/cuda/parallel-thread-execution/ -o ptx.md
    python -m nvidia https://docs.nvidia.com/cuda/cuda-programming-guide/index.html -o cuda.md
    python -m nvidia https://docs.nvidia.com/cuda/blackwell-tuning-guide/ -o bw_tune.md --style old
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from .pipeline import RecoverOptions, recover_markdown

_CUDA_BINARY_UTILITIES_SECTION_FILES = {
    "1. Overview": "overview.md",
    "2. cuobjdump": "cuobjdump.md",
    "3. nvdisasm": "nvdisasm.md",
    "4. Instruction Set Reference": "instruction_set_reference.md",
    "5. cu++filt": "cuplusplusfilt.md",
    "6. nvprune": "nvprune.md",
}
_CUDA_BINARY_UTILITIES_APPENDIX_TARGET = "nvdisasm.md"
_TOP_LEVEL_NUMBERED_HEADING_RE = re.compile(r"^# (\d+\. .+)$", re.MULTILINE)


def _default_output_path(url: str) -> Path:
    """Derive a reasonable output filename from the URL slug.

    Picks the last non-empty path segment that isn't a trailing
    ``index.html``-ish file. e.g. ``.../cuda/parallel-thread-execution/``
    -> ``parallel-thread-execution.md``.
    """

    parts = [p for p in urlparse(url).path.split("/") if p]
    while parts and parts[-1].endswith(".html"):
        parts.pop()
    slug = parts[-1] if parts else "nvidia-docs"
    return Path(f"{slug}.md")


def _strip_page_markers(md: str) -> str:
    md = re.sub(r"^<!-- Reconstructed from .*? -->\n\n", "", md)
    return re.sub(r"^<!-- =====.*?===== -->\n\n", "", md, flags=re.MULTILINE)


def _split_cuda_binary_utilities(md: str) -> dict[str, str]:
    md = _strip_page_markers(md)
    matches = list(_TOP_LEVEL_NUMBERED_HEADING_RE.finditer(md))
    if not matches:
        raise RuntimeError("No top-level numbered sections found")

    preamble = md[: matches[0].start()].strip()
    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        title = match.group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        sections[title] = md[match.start() : end].strip()

    outputs: dict[str, str] = {}
    for section_title, filename in _CUDA_BINARY_UTILITIES_SECTION_FILES.items():
        body = sections.get(section_title)
        if body is None:
            raise RuntimeError(f"Missing CUDA Binary Utilities section: {section_title}")
        if section_title == "1. Overview" and preamble:
            body = f"{preamble}\n\n{body}"
        if (
            filename == _CUDA_BINARY_UTILITIES_APPENDIX_TARGET
            and "7. Appendix" in sections
        ):
            body = f"{body}\n\n{sections['7. Appendix']}"
        outputs[filename] = body.rstrip() + "\n"
    return outputs


def _write_split_cuda_binary_utilities(md: str, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, body in _split_cuda_binary_utilities(md).items():
        path = output_dir / filename
        path.write_text(body, encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m nvidia",
        description=(
            "Recover the original Markdown for an NVIDIA Sphinx docs site, "
            "auto-detecting whether it uses the new (pydata-sphinx-theme) or "
            "old (Read-the-Docs) layout."
        ),
    )
    ap.add_argument("url", help="Index page URL of the documentation site")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output Markdown path (default: <slug>.md derived from URL)",
    )
    ap.add_argument(
        "--style",
        choices=("auto", "old", "new"),
        default="auto",
        help="Override theme detection (default: auto)",
    )
    ap.add_argument(
        "--no-mirror",
        action="store_true",
        help="Skip the .html.md mirror; render every page from HTML",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for caching HTTP fetches between runs",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=3,
        help="HTTP retry attempts per URL (default: 3)",
    )
    ap.add_argument(
        "--split-cuda-binary-utilities",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Also split the CUDA Binary Utilities page into per-section "
            "Markdown files under DIR"
        ),
    )
    args = ap.parse_args(argv)

    output_path = args.output or _default_output_path(args.url)

    md = recover_markdown(
        args.url,
        RecoverOptions(
            style=args.style,
            no_mirror=args.no_mirror,
            cache_dir=args.cache_dir,
            retries=args.retries,
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(
        f"Wrote {output_path} ({output_path.stat().st_size:,} bytes)",
        file=sys.stderr,
    )
    if args.split_cuda_binary_utilities is not None:
        paths = _write_split_cuda_binary_utilities(md, args.split_cuda_binary_utilities)
        for path in paths:
            print(f"Wrote {path} ({path.stat().st_size:,} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
