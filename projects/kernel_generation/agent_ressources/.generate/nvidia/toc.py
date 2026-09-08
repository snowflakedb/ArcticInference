"""Discover ordered sub-page URLs from a docs site's sidebar.

The two themes lay out their sidebars differently:

* pydata: ``div.bd-toc-item`` containing ``a.reference``
* RTD:    ``div.wy-menu-vertical`` containing ``a.reference.internal``

For a single-page doc (PTX, Blackwell guides, etc.) there is nothing useful
in the sidebar that points to *other* pages, and we want the recovery to
gracefully degrade to a single-entry list pointing at the index URL itself
rather than failing. The collectors below filter to in-tree ``.html`` links
only and drop anchors / external URLs / duplicates while preserving order,
then ``discover_pages`` falls back to ``[index_url]`` if nothing was found.
"""

from __future__ import annotations

from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup, Tag

from .detect import Style


def _base_dir(index_url: str) -> str:
    return index_url.rsplit("/", 1)[0] + "/"


def _collect(
    container: Tag,
    *,
    index_url: str,
    link_selector: dict,
) -> list[tuple[str, str]]:
    base_dir = _base_dir(index_url)
    pages: list[tuple[str, str]] = []
    seen: set[str] = set()

    for a in container.find_all("a", **link_selector):
        href = (a.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue
        absolute, _ = urldefrag(urljoin(index_url, href))
        if not absolute.startswith(base_dir):
            continue
        if not absolute.endswith(".html"):
            continue
        if absolute in seen or absolute == index_url:
            continue
        seen.add(absolute)
        label = " ".join((a.get_text() or "").split()) or absolute
        pages.append((absolute, label))

    return pages


def _new_style_pages(soup: BeautifulSoup, index_url: str) -> list[tuple[str, str]]:
    toc = soup.find("div", class_="bd-toc-item")
    if toc is None:
        return []
    return _collect(toc, index_url=index_url, link_selector={"class_": "reference"})


def _old_style_pages(soup: BeautifulSoup, index_url: str) -> list[tuple[str, str]]:
    nav = soup.find("div", class_="wy-menu-vertical")
    if nav is None:
        nav = soup.find("nav", class_="wy-nav-side")
    if nav is None:
        return []
    return _collect(
        nav,
        index_url=index_url,
        link_selector={"class_": ["reference", "internal"]},
    )


def discover_pages(
    index_html: str, index_url: str, style: Style
) -> list[tuple[str, str]]:
    """Return ordered ``(absolute_html_url, link_label)`` for every TOC entry.

    The index URL is always the first entry. If the sidebar is empty (or
    absent — single-page docs) the result is just the index, which is
    exactly what the rest of the pipeline expects.
    """

    soup = BeautifulSoup(index_html, "html.parser")
    pages: list[tuple[str, str]] = []

    if style == "new":
        pages.extend(_new_style_pages(soup, index_url))
    else:
        pages.extend(_old_style_pages(soup, index_url))

    if urlparse(index_url).path.endswith("/contents.html") and pages:
        return pages

    return [(index_url, "index"), *pages]
