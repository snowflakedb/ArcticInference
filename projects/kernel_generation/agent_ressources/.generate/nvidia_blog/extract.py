"""Locate the article body inside an NVIDIA Developer Blog page.

The blog (``developer.nvidia.com/blog/...``) is WordPress, not Sphinx, so
the structural anchors are different from the docs site:

    <main class="main-content ...">
      <div class="post-card--single">      ← title / byline / share buttons
      <div class="entry-meta-social">      ← share/like rail
      <div class="collapsed">              ← AI-generated summary widget
      <div class="entry-content">          ← THE ACTUAL POST BODY
      <div class="card--post-attributes-secondary">
      <div class="... tags">
      <div class="post-authors-list">      ← author bios
      <div class="entry-content-comments">

Everything outside ``div.entry-content`` is page chrome and is dropped.
The ``<h1>`` lives in ``post-card--single``, not inside the body, so we
extract it separately and let the renderer prepend it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag


_WS_RE = re.compile(r"\s+")


def _clean(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


@dataclass
class ExtractedPost:
    """Parsed handles into the post DOM.

    ``body`` is the ``div.entry-content`` element; the renderer walks its
    direct children. ``title`` is the H1 text (already stripped of any
    ``| Site Name`` trailer pulled from ``<title>`` fallback paths).
    """

    body: Tag
    title: str


def extract_post(html: str) -> ExtractedPost:
    """Find the post body and title in a developer.nvidia.com/blog page.

    Raises ``RuntimeError`` if the expected layout isn't present, which
    is the right thing to do for a parser intentionally narrow to one
    site — silently returning a partial result would mask mis-aimed input.
    """

    soup = BeautifulSoup(html, "html.parser")

    main = soup.find("main")
    if main is None:
        raise RuntimeError("No <main> element on page; not a blog post layout?")

    body = main.find("div", class_="entry-content")
    if body is None:
        raise RuntimeError(
            "No <div class='entry-content'> inside <main>; "
            "page layout doesn't match the developer-blog template."
        )

    title = _resolve_title(soup, main)
    return ExtractedPost(body=body, title=title)


def _resolve_title(soup: BeautifulSoup, main: Tag) -> str:
    """Try the H1 in the post-card header first, then meta tags, then
    ``<title>`` (with the ``| NVIDIA Technical Blog`` trailer trimmed)."""

    h1 = main.find("h1")
    if h1:
        text = _clean(h1.get_text())
        if text:
            return text

    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return _clean(og["content"])

    t = soup.find("title")
    if t:
        text = _clean(t.get_text())
        if "|" in text:
            text = text.rsplit("|", 1)[0].strip()
        return text

    return ""
