"""Recover the article markdown for an NVIDIA Developer Blog post.

The blog (``developer.nvidia.com/blog/...``) is WordPress, not Sphinx,
so this is a separate parser from :mod:`nvidia`. The
shape of the work is the same — fetch the page, locate the article body,
walk it, and emit clean markdown — but the locators (``<main>`` >
``div.entry-content``) and walker (flat, not section-nested) are
different enough that mixing them into the docs parser would just dilute
both.

Public entry point::

    from nvidia_blog import recover_markdown
    md = recover_markdown(
        "https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/"
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import sys

# Absolute, not `..nvidia`: both are top-level packages side by side in .generate/,
# so a relative import walks off the top and raises ImportError.
from nvidia.fetch import Fetcher
from .extract import extract_post
from .render import render_post_to_markdown

LogFn = Callable[[str], None]


def _default_log(msg: str) -> None:
    print(msg, file=sys.stderr)


@dataclass
class RecoverOptions:
    """User-facing knobs for :func:`recover_markdown`.

    Attributes:
        cache_dir: optional directory for caching HTTP fetches.
        retries: HTTP retry attempts before giving up.
        log: callable for progress messages; defaults to stderr.
    """

    cache_dir: Path | None = None
    retries: int = 3
    log: LogFn = field(default=_default_log)


def recover_markdown(post_url: str, options: RecoverOptions | None = None) -> str:
    """Reconstruct the markdown source of a developer-blog post.

    Returns the markdown as a string. The CLI is what writes to disk.
    """

    opts = options or RecoverOptions()
    fetcher = Fetcher(retries=opts.retries, cache_dir=opts.cache_dir)
    opts.log(f"Fetching {post_url}")
    html = fetcher.get(post_url)
    post = extract_post(html)
    opts.log(f"Extracted post: {post.title!r}")
    return render_post_to_markdown(post, post_url)


__all__ = ["RecoverOptions", "recover_markdown"]
