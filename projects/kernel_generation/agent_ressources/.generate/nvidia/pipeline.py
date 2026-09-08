"""Orchestrate theme detection, TOC traversal, and per-page recovery.

Per page, the pipeline tries two strategies in order:

1. **mirror** — fetch ``<page>.html.md`` and clean up the Jina-Reader
   wrapper. Empty (HTTP 200, zero bytes), missing (404), and broken
   (collapsed-into-one-giant-paragraph) mirrors transparently fall through
   to (2).
2. **render** — fetch ``<page>.html`` and convert it locally with the
   structural Sphinx walker in ``html_render``.

The mirror is tried for every style. Most old-style (Read-the-Docs) sites
serve an empty mirror so they still end up rendered from HTML, but a few —
e.g. the Nsight Compute sub-docs — ship rich mirrors despite the old
theme, and those are recovered cleanly via (1) for free.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from .detect import Style, detect_style
from .fetch import Fetcher, NotFound
from .html_render import render_html_to_markdown
from .md_mirror import (
    absolutize_links,
    clean_mirror_body,
    is_broken_paragraph_page,
    strip_reader_wrapper,
)
from .toc import discover_pages

StyleSetting = Literal["auto", "old", "new"]
LogFn = Callable[[str], None]


def _default_log(msg: str) -> None:
    print(msg, file=sys.stderr)


@dataclass
class RecoverOptions:
    """User-facing knobs for ``recover_markdown``.

    Attributes:
        style: ``"auto"`` (default) detects from HTML; ``"old"`` / ``"new"``
            override.
        no_mirror: when ``True``, skip the ``.html.md`` mirror entirely and
            render every page from HTML. Useful when the mirror is fresh
            but broken, or when you want a pure HTML-derived rendering.
        cache_dir: optional directory for caching HTTP fetches between
            runs. Created if missing.
        retries: number of HTTP attempts per URL before giving up.
        log: callable that receives one progress line at a time. Defaults
            to writing to ``sys.stderr``.
    """

    style: StyleSetting = "auto"
    no_mirror: bool = False
    cache_dir: Path | None = None
    retries: int = 3
    log: LogFn = field(default=_default_log)


def _render_page(fetcher: Fetcher, html_url: str) -> str:
    html = fetcher.get(html_url)
    return render_html_to_markdown(html, html_url)


def _mirror_url(html_url: str) -> str:
    """Derive the ``.html.md`` mirror URL for an HTML page URL.

    Handles the three forms NVIDIA serves:

    * ``.../foo.html``  -> ``.../foo.html.md``
    * ``.../foo/``      -> ``.../foo/index.html.md`` (the server resolves the
      bare directory to ``index.html`` but the mirror has no such rewrite, so
      we have to spell it out)
    * ``.../foo``       -> ``.../foo/index.html.md`` (rare; treat as directory)
    """

    if html_url.endswith(".html"):
        return html_url + ".md"
    if html_url.endswith("/"):
        return html_url + "index.html.md"
    return html_url + "/index.html.md"


def _mirror_page(fetcher: Fetcher, html_url: str) -> str | None:
    """Try to recover a page from its ``.html.md`` mirror.

    Returns the cleaned body, or ``None`` if the mirror was missing
    (404), empty, or collapsed into one giant paragraph — in any of those
    cases the caller falls back to HTML rendering.
    """

    md_url = _mirror_url(html_url)
    try:
        raw = fetcher.get(md_url, allow_empty=True)
    except NotFound:
        return None
    if not raw.strip():
        return None
    body, title = strip_reader_wrapper(raw)
    if is_broken_paragraph_page(body):
        return None
    return clean_mirror_body(body, title=title, page_url=html_url)


def _fetch_page_markdown(
    fetcher: Fetcher,
    html_url: str,
    *,
    no_mirror: bool,
    log: LogFn,
) -> str:
    if not no_mirror:
        body = _mirror_page(fetcher, html_url)
        if body is not None:
            return body
        log(f"  -> mirror empty/broken, rendering HTML for {html_url}")

    rendered = _render_page(fetcher, html_url)
    return absolutize_links(rendered, html_url)


def recover_markdown(
    index_url: str, options: RecoverOptions | None = None
) -> str:
    """Reconstruct the full Markdown for a doc site rooted at ``index_url``.

    Returns the stitched Markdown as a single string (does not write to
    disk; that's the CLI's job).
    """

    opts = options or RecoverOptions()
    fetcher = Fetcher(retries=opts.retries, cache_dir=opts.cache_dir)

    index_html = fetcher.get(index_url)

    if opts.style == "auto":
        style: Style = detect_style(index_html)
    else:
        style = opts.style

    opts.log(f"Detected style: {style}")

    pages = discover_pages(index_html, index_url, style)
    opts.log(f"Discovered {len(pages)} pages from TOC")

    parts: list[str] = [f"<!-- Reconstructed from {index_url} -->\n\n"]

    for i, (url, label) in enumerate(pages, 1):
        opts.log(f"  [{i:3d}/{len(pages)}] {label}  <-  {url}")
        body = _fetch_page_markdown(
            fetcher, url, no_mirror=opts.no_mirror, log=opts.log
        )
        parts.append(f"<!-- ===== {label}  ({url}) ===== -->\n\n")
        parts.append(body.rstrip() + "\n\n")

    return "".join(parts)
