"""Convert a developer-blog ``div.entry-content`` body to Markdown.

Unlike the Sphinx renderer in :mod:`nvidia.html_render`,
the blog body is *flat*: a sequence of paragraphs, headings (h2/h3),
lists, figures, and tables, with no nested ``<section>`` containers. So
the walker is a single pass over the direct children of
``div.entry-content``.

Inline formatting matters here in a way it doesn't for most Sphinx
content — every other paragraph has a ``<strong>`` lead-in or a
``<a>`` reference — so this module renders inline tags (``<strong>``,
``<em>``, ``<code>``, ``<a>``, ``<br>``) instead of flattening to plain
text. Tables emit GFM pipe-tables with a header separator row.
"""

from __future__ import annotations

import re
from typing import Iterable
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString, Tag

from .extract import ExtractedPost


# Lightbox/share-button chrome that gets injected inside <figure>s and
# elsewhere in the entry body. Stripped before walking so the renderer
# doesn't have to special-case them at every level.
_CHROME_TAGS = ("noscript", "button", "svg", "script", "style")

# WordPress wraps the actual image inside a <noscript> for JS-disabled
# fallbacks. We unwrap it back into the figure before stripping noscript.
_NOSCRIPT_IMG_RE = re.compile(r"<noscript>(\s*<img\b[^>]*>\s*)</noscript>", re.I)


def _clean_ws(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()


def _escape_md_inline(s: str) -> str:
    """Escape characters with markdown meaning in inline text.

    Conservative: only escape characters that would otherwise be parsed
    as markdown syntax in this context. Underscores are deliberately
    left alone — many of the post's identifiers contain them and CommonMark
    parsers tolerate raw ``_`` outside of word boundaries.
    """

    return s.replace("\\", "\\\\").replace("*", "\\*").replace("`", "\\`")


def _render_inline(node: Tag | NavigableString) -> str:
    """Walk a node's children, returning markdown-formatted inline text.

    Whitespace is preserved as-is (modulo collapsing runs to a single
    space) so adjacency in the source HTML survives — e.g. a trailing
    space inside ``<strong>foo </strong>bar`` doesn't get dropped and
    glue ``foobar`` together.
    """

    if isinstance(node, NavigableString):
        return _escape_md_inline(str(node))

    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(_escape_md_inline(str(child)))
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name in _CHROME_TAGS:
            continue
        if name in ("strong", "b"):
            inner = _render_inline(child).strip()
            if inner:
                parts.append(f"**{inner}**")
        elif name in ("em", "i"):
            inner = _render_inline(child).strip()
            if inner:
                parts.append(f"*{inner}*")
        elif name == "code":
            parts.append(f"`{child.get_text()}`")
        elif name == "a":
            href = (child.get("href") or "").strip()
            label = _render_inline(child).strip()
            if not href:
                parts.append(label)
            elif not label:
                parts.append(f"<{href}>")
            else:
                parts.append(f"[{label}]({href})")
        elif name == "br":
            parts.append("  \n")
        elif name in ("sub", "sup"):
            parts.append(f"<{name}>{_render_inline(child)}</{name}>")
        elif name == "img":
            alt = _clean_ws(child.get("alt") or "")
            src = (child.get("src") or "").strip()
            if src:
                parts.append(f"![{alt}]({src})")
        else:
            parts.append(_render_inline(child))

    text = "".join(parts)
    return re.sub(r"[ \t]+", " ", text)


def _resolve_url(href: str, base: str) -> str:
    href = (href or "").strip()
    if not href or href.startswith(("http://", "https://", "data:", "mailto:", "#")):
        return href
    if href.startswith("//"):
        scheme = urlparse(base).scheme or "https"
        return f"{scheme}:{href}"
    return urljoin(base, href)


def _absolutize_links(md: str, page_url: str) -> str:
    """Rewrite every ``](...)`` and ``](<...>)`` link target to absolute.

    Only operates on URL-shaped targets; anchors and absolute URLs pass
    through. Image targets share the same syntax so this handles them
    too.
    """

    pattern = re.compile(
        r"(!?\[[^\]]*\])"
        r"\(\s*(?P<target><[^>]+>|[^)\s]+)(?P<title>\s+\"[^\"]*\")?\s*\)"
    )

    def repl(m: re.Match[str]) -> str:
        target = m.group("target")
        title = m.group("title") or ""
        if target.startswith("<") and target.endswith(">"):
            inner = _resolve_url(target[1:-1], page_url)
            return f"{m.group(1)}(<{inner}>{title})"
        return f"{m.group(1)}({_resolve_url(target, page_url)}{title})"

    return pattern.sub(repl, md)


def _strip_chrome_in_place(body: Tag) -> None:
    """Remove every chrome tag anywhere under ``body``.

    Includes ``<noscript>`` (after first pulling out any ``<img>`` it
    wraps so we keep the actual image source), ``<button>`` / ``<svg>``
    lightbox controls, ``<script>`` / ``<style>``, and the
    ``a.heading-anchor-link`` permalinks WordPress drops next to every
    heading. Mutates ``body`` directly — the renderer never re-reads
    the soup after this, so in-place is safe and avoids a deep clone.
    """

    for ns in body.find_all("noscript"):
        img = ns.find("img")
        if img:
            ns.replace_with(img)
        else:
            ns.decompose()
    for tag in body.find_all(_CHROME_TAGS):
        tag.decompose()
    for a in body.find_all(
        "a", class_=lambda c: bool(c) and "heading-anchor-link" in c
    ):
        a.decompose()


def _figcaption_plain(fig: Tag) -> str:
    """Get the figcaption as plain inline text — no markdown emphasis.

    The wrapper italicizes the whole caption already, so emitting nested
    ``<em>`` markers would just produce broken syntax. Some WordPress
    captions also contain accidentally-truncated ``<em>`` tags (e.g. the
    Table 1 caption closes ``</em>`` mid-word), and using ``get_text``
    sidesteps that whole failure mode at the cost of dropping the rare
    intra-caption link.
    """

    cap = fig.find("figcaption")
    if cap is None:
        return ""
    text = _clean_ws(cap.get_text(" ", strip=True))
    return text.replace("\\", "\\\\").replace("*", "\\*")


def _figure_image(fig: Tag) -> tuple[str, str]:
    """Return ``(src, alt)`` for the first ``<img>`` in ``fig``."""

    img = fig.find("img")
    if img is None:
        return "", ""
    return (img.get("src") or "").strip(), _clean_ws(img.get("alt") or "")


def _render_figure(fig: Tag) -> str:
    """Render a ``<figure>`` as ``![alt](src)`` plus an italic caption.

    Uses the image's *original* ``alt`` attribute for the alt text (which
    is typically a rich screen-reader description) rather than echoing
    the caption — that keeps the alt informative and avoids the caption
    text bleeding into the alt with broken emphasis markers.
    """

    src, alt = _figure_image(fig)
    caption = _figcaption_plain(fig)
    if not src and not caption:
        return ""
    out = ""
    if src:
        alt_text = alt or caption or "image"
        out += f"![{alt_text}]({src})\n\n"
    if caption:
        out += f"*{caption}*\n\n"
    return out


def _render_table_figure(fig: Tag) -> str:
    """Render a ``<figure class='wp-block-table'>`` to a GFM table + caption.

    WordPress wraps table blocks in ``<figure>`` with the table as a
    child and the caption as a ``<figcaption>`` sibling. Without this
    branch the walker treats it as an image figure and silently drops
    the table.
    """

    table = fig.find("table")
    if table is None:
        return _render_figure(fig)
    out = _render_table(table) + "\n"
    cap = _figcaption_plain(fig)
    if cap:
        out += f"*{cap}*\n\n"
    return out


def _is_wp_block_image(div: Tag) -> bool:
    classes = div.get("class") or []
    return "wp-block-image" in classes


def _render_list(el: Tag, ordered: bool, indent: str = "") -> str:
    """Render a ``<ul>``/``<ol>``, including nested lists.

    Nested lists are indented by 4 spaces per level so CommonMark
    parsers attach them to the parent item.
    """

    out: list[str] = []
    for i, li in enumerate(el.find_all("li", recursive=False), 1):
        marker = f"{i}." if ordered else "-"
        # Split children into inline content + nested block lists.
        nested: list[Tag] = []
        inline_nodes: list[Tag | NavigableString] = []
        for child in li.children:
            if isinstance(child, Tag) and child.name in ("ul", "ol"):
                nested.append(child)
            else:
                inline_nodes.append(child)
        text = _render_inline_nodes(inline_nodes).strip()
        if text:
            out.append(f"{indent}{marker} {text}")
        else:
            out.append(f"{indent}{marker}")
        for sub in nested:
            out.append(_render_list(sub, sub.name == "ol", indent + "    "))
    return "\n".join(filter(None, out)) + "\n"


def _render_inline_nodes(nodes: Iterable[Tag | NavigableString]) -> str:
    parts: list[str] = []
    for n in nodes:
        if isinstance(n, NavigableString):
            parts.append(_escape_md_inline(str(n)))
        elif isinstance(n, Tag):
            parts.append(_render_inline(n))
    return re.sub(r"[ \t]+", " ", "".join(parts))


def _render_table(table: Tag) -> str:
    """Emit a GFM pipe table with a header separator.

    If the table has no ``<thead>`` we treat the first row as the header
    (which is what every table in this corpus looks like). Cells render
    inline markdown so ``<strong>`` / ``<a>`` survive.
    """

    rows: list[list[str]] = []
    has_thead = table.find("thead") is not None
    if has_thead:
        header_rows = table.find("thead").find_all("tr")
        body_rows = []
        tbody = table.find("tbody")
        if tbody:
            body_rows = tbody.find_all("tr", recursive=False)
        else:
            body_rows = [
                tr for tr in table.find_all("tr", recursive=False)
                if tr.parent is table
            ]
        all_rows = list(header_rows) + list(body_rows)
    else:
        all_rows = table.find_all("tr")

    for tr in all_rows:
        cells: list[str] = []
        for cell in tr.find_all(("th", "td"), recursive=False):
            cells.append(
                _render_inline(cell).strip().replace("|", "\\|").replace("\n", " ")
            )
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out: list[str] = []
    out.append("| " + " | ".join(rows[0]) + " |")
    out.append("|" + "|".join([" --- "] * width) + "|")
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out) + "\n"


def _render_block(el: Tag) -> str:
    """Render a single direct child of ``entry-content`` to markdown.

    Returns the trailing ``\\n\\n`` so blocks join cleanly when
    concatenated. Unknown block types fall through to inline rendering
    inside a paragraph, which is the right thing for the rare
    ``<div>`` that sneaks in carrying flowing text.
    """

    name = el.name
    if name == "p":
        text = _render_inline(el).strip()
        return f"{text}\n\n" if text else ""
    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        level = int(name[1])
        text = _render_inline(el).strip()
        return f"{'#' * level} {text}\n\n" if text else ""
    if name == "ul":
        return _render_list(el, ordered=False) + "\n"
    if name == "ol":
        return _render_list(el, ordered=True) + "\n"
    if name == "figure":
        if el.find("table") is not None:
            return _render_table_figure(el)
        return _render_figure(el)
    if name == "img":
        src = (el.get("src") or "").strip()
        alt = _clean_ws(el.get("alt") or "")
        return f"![{alt}]({src})\n\n" if src else ""
    if name == "table":
        return _render_table(el) + "\n"
    if name == "blockquote":
        text = _render_inline(el).strip()
        if not text:
            return ""
        return "\n".join(f"> {ln}" for ln in text.split("\n")) + "\n\n"
    if name == "pre":
        return f"```\n{el.get_text()}\n```\n\n"
    if name == "div":
        if _is_wp_block_image(el):
            fig = el.find("figure")
            if fig is not None:
                return _render_figure(fig)
            img = el.find("img")
            if img is not None:
                src = (img.get("src") or "").strip()
                alt = _clean_ws(img.get("alt") or "")
                if src:
                    return f"![{alt}]({src})\n\n"
        # Fallback: walk inner blocks. WordPress emits a few
        # nested-div decorations (``wp-block-group`` and friends) that
        # carry real content one level down.
        return "".join(_render_block(c) for c in el.children if isinstance(c, Tag))
    return ""


def render_post_to_markdown(post: ExtractedPost, page_url: str) -> str:
    """Render an ``ExtractedPost`` to a single markdown string.

    Includes the H1 title at the top. Resolves all image and link URLs
    against ``page_url`` so the output is portable when the file moves.
    """

    _strip_chrome_in_place(post.body)

    parts: list[str] = []
    if post.title:
        parts.append(f"# {post.title}\n\n")

    for child in post.body.children:
        if not isinstance(child, Tag):
            continue
        parts.append(_render_block(child))

    md = "".join(parts)
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = _absolutize_links(md, page_url)
    return md.rstrip() + "\n"
