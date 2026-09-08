"""CLI for the NVIDIA developer-blog markdown recovery package.

Examples::

    python -m nvidia_blog \\
        https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/ \\
        -o blackwell_ultra.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

from . import RecoverOptions, recover_markdown


def _default_output_path(url: str) -> Path:
    """Derive ``<slug>.md`` from the trailing path segment."""

    parts = [p for p in urlparse(url).path.split("/") if p]
    while parts and parts[-1].endswith(".html"):
        parts.pop()
    slug = parts[-1] if parts else "nvidia-blog-post"
    return Path(f"{slug}.md")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m nvidia_blog",
        description="Recover markdown for a developer.nvidia.com blog post.",
    )
    ap.add_argument("url", help="URL of the blog post to recover")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output markdown path (default: <slug>.md derived from URL)",
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
        help="HTTP retry attempts (default: 3)",
    )
    args = ap.parse_args(argv)

    output_path = args.output or _default_output_path(args.url)

    md = recover_markdown(
        args.url,
        RecoverOptions(cache_dir=args.cache_dir, retries=args.retries),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(
        f"Wrote {output_path} ({output_path.stat().st_size:,} bytes)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
