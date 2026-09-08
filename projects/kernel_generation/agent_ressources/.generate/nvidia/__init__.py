"""Recover the original Markdown for NVIDIA Sphinx-generated documentation sites.

Two layouts are supported and auto-detected:

* **new style** (pydata-sphinx-theme).
* **old style** (Read-the-Docs ``wy-nav-side`` theme, Docutils 0.17.1).

The style only governs how the sidebar is traversed for TOC discovery. For
every page the pipeline first tries the ``<page>.html.md`` mirror and only
falls back to local HTML rendering when the mirror is missing, empty, or
collapsed into one giant paragraph. Most old-style sites ship an empty
mirror and end up rendered from HTML, but a few (e.g. the Nsight Compute
sub-docs) ship a non-empty mirror despite the old theme, and the pipeline
recovers them cleanly via the mirror.

Public entry point::

    from nvidia import recover_markdown
    md = recover_markdown("https://docs.nvidia.com/cuda/parallel-thread-execution/")
"""

from __future__ import annotations

from .pipeline import RecoverOptions, recover_markdown

__all__ = ["RecoverOptions", "recover_markdown"]
