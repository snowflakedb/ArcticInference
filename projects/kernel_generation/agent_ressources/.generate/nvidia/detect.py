"""Sniff which Sphinx theme a doc site uses, so the pipeline can pick a path.

We look for unambiguous markers that differ between the pydata-sphinx-theme
(``bd-toc-item`` sidebar, sometimes a ``meta[name=generator]`` containing
``pydata-sphinx-theme``) and the older Read-the-Docs theme (``wy-nav-side``
nav, Docutils 0.17.1 generator). Falling back to the meta tag matters: a few
NVIDIA pydata pages omit the sidebar entirely (single-page docs collapse the
TOC into a header dropdown) but still carry the generator meta.
"""

from __future__ import annotations

from typing import Literal

from bs4 import BeautifulSoup

Style = Literal["new", "old"]


class StyleDetectionError(RuntimeError):
    """Raised when neither a pydata nor an RTD marker is present."""


def detect_style(html: str) -> Style:
    soup = BeautifulSoup(html, "html.parser")

    if soup.find("div", class_="bd-toc-item") is not None:
        return "new"
    if soup.find(attrs={"class": lambda c: bool(c) and "bd-sidebar" in c}) is not None:
        return "new"
    if soup.find("nav", class_="wy-nav-side") is not None:
        return "old"
    if soup.find("div", class_="wy-menu-vertical") is not None:
        return "old"

    gen = soup.find("meta", attrs={"name": "generator"})
    content = (gen.get("content") if gen else "") or ""
    content = content.lower()
    if "pydata-sphinx-theme" in content or "pydata sphinx theme" in content:
        return "new"
    if "docutils" in content:
        return "old"

    raise StyleDetectionError(
        "Could not detect theme: no pydata (bd-toc-item / bd-sidebar) or "
        "RTD (wy-nav-side / wy-menu-vertical) markers found. "
        "Pass --style {old,new} to override."
    )
