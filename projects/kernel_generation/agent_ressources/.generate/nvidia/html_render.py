"""Convert a single Sphinx-rendered HTML page to Markdown.

Used both as the primary path for old-style (RTD) docs whose ``.html.md``
mirror is empty, and as a fallback for new-style pages whose mirror collapsed
into one giant paragraph.

The traversal is deliberately structural: we parse the article body into a
tree of section dicts (``id``/``title``/``level``/``blocks``/``children``)
and only render to Markdown at the end, which keeps the rendering rules in
one place and makes it easy to special-case a block type without
restructuring the walker.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup, Tag

# h7/h8 are not real HTML, but Sphinx emits them for sections nested 7+ deep — which
# the PTX doc reaches in the tcgen05 MMA layout material (9.7.17.10.8.4.1 and
# friends). Without them those sections lose their title and their body renders as a
# `[raw]` placeholder. Ordered, because the search below takes the first match and a
# set would make that hash-order dependent.
HEADER_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8")
SKIP_HEADER_TAGS = frozenset(HEADER_TAGS)


def image_base_url_from_page_url(page_url: str) -> str:
    """Turn ``https://host/a/b/index.html`` into ``https://host/a/b/`` for
    ``urljoin``.

    Returns an empty string if ``page_url`` is empty or not absolute, which
    lets callers pass it unconditionally without first sniffing.
    """

    if not page_url:
        return ""
    u = urlparse(page_url.strip())
    if not u.scheme or not u.netloc:
        return ""
    raw_path = u.path or "/"
    # If the URL is already a directory (ends in "/"), keep it as-is.
    # Otherwise drop the final segment so urljoin resolves siblings correctly.
    if raw_path.endswith("/"):
        dirpath = raw_path
    elif "/" in raw_path:
        dirpath = raw_path.rsplit("/", 1)[0] + "/"
    else:
        dirpath = "/"
    return urlunparse((u.scheme, u.netloc, dirpath, "", "", ""))


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _markdown_image_alt(s: str) -> str:
    """Escape characters that break CommonMark image description brackets."""

    return s.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _absolute_image_url(src: str, base: str) -> str:
    """Resolve relative ``src`` to an http(s) URL; pass through absolute and
    data URLs unchanged."""

    src = (src or "").strip()
    if not src:
        return ""
    if src.startswith(("http://", "https://", "data:")):
        return src
    if src.startswith("//"):
        return "https:" + src
    base = base.strip()
    if not base:
        return src
    if not base.endswith("/"):
        base = base + "/"
    return urljoin(base, src)


def _figcaption_caption(fig: Tag) -> str:
    """Sphinx figcaption: caption-number + caption-text (+ optional legend),
    no headerlink noise."""

    parts: list[str] = []
    num = fig.find("span", class_=lambda c: c and "caption-number" in c)
    txt = fig.find("span", class_=lambda c: c and "caption-text" in c)
    if num:
        parts.append(_clean_text(num.get_text()))
    if txt:
        parts.append(_clean_text(txt.get_text()))
    leg = fig.find("div", class_=lambda c: c and "legend" in c)
    if leg:
        parts.append(_clean_text(leg.get_text()))
    if parts:
        return " ".join(parts)
    return _clean_text(fig.get_text(" ", strip=True))


def _table_to_text(table: Tag) -> str:
    """Join table cells by row; avoids the one-newline-per-cell artifacts
    you get from ``get_text('\\n')``."""

    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells: list[str] = []
        for cell in tr.find_all(("th", "td"), recursive=False):
            cells.append(_clean_text(cell.get_text(" ", strip=True)))
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _flow_to_blocks(container: Tag) -> list[dict[str, Any]]:
    """Collect blocks from direct children (used for ``<li>`` bodies, etc.)."""

    blocks: list[dict[str, Any]] = []
    for child in container.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "blockquote":
            for inner in child.children:
                if isinstance(inner, Tag) and inner.name == "div":
                    blocks.extend(_flow_to_blocks(inner))
                elif isinstance(inner, Tag):
                    blk = _element_to_block(inner)
                    if blk is not None:
                        blocks.append(blk)
            continue
        blk = _element_to_block(child)
        if blk is not None:
            blocks.append(blk)
    return blocks


def _heading_level(h: Tag) -> int:
    return int(h.name[1]) if h.name and h.name[0] == "h" else 0


def _parse_heading(h: Tag) -> tuple[str, str]:
    """Return ``(section_number, title_text)``."""

    # Drop Sphinx headerlink anchors (¶ in old themes, # in pydata-sphinx-theme)
    # so they don't end up appended to the title text.
    for hl in h.find_all("a", class_="headerlink"):
        hl.extract()

    num = ""
    sn = h.find("span", class_="section-number")
    if sn:
        num = _clean_text(sn.get_text())
    link = h.find("a", class_="reference")
    if link:
        title = _clean_text(link.get_text())
    else:
        title = _clean_text(h.get_text())
    if sn and title.startswith(num):
        title = _clean_text(title[len(num) :])
    return num, title


def _element_to_block(el: Tag) -> dict[str, Any] | None:
    if el.name == "p":
        classes = el.get("class") or []
        text = el.get_text(" ", strip=True)
        if "rubric" in classes:
            return {"type": "rubric", "text": _clean_text(text)}
        return {"type": "paragraph", "text": text}

    if el.name == "div":
        pre = el.find("pre")
        if pre:
            return {"type": "code", "text": pre.get_text()}
        cls = " ".join(el.get("class") or [])
        if "highlight" in cls:
            return {"type": "code", "text": el.get_text()}
        # Sphinx wraps real content in plain divs: `line`/`line-block` for RST line
        # blocks (instruction syntax) and `admonition` for notes. Emitting a
        # placeholder here silently dropped all of it, so recurse instead. A
        # `line` holds inline content with no block child, so take its text.
        if "line" in cls.split():
            text = el.get_text(" ", strip=True)
            return {"type": "paragraph", "text": text} if text else None
        blocks = _flow_to_blocks(el)
        if blocks:
            return {"type": "container", "blocks": blocks}
        text = el.get_text(" ", strip=True)
        return {"type": "paragraph", "text": text} if text else None

    if el.name == "figure":
        cap = el.find("figcaption")
        caption = _figcaption_caption(cap) if cap else ""
        img = el.find("img")
        src = (img.get("src") or "") if img else ""
        alt = _clean_text(img.get("alt") or "") if img else ""
        return {"type": "figure", "caption": caption, "src": src, "alt": alt}

    if el.name == "img":
        return {
            "type": "image",
            "src": (el.get("src") or "").strip(),
            "alt": _clean_text(el.get("alt") or ""),
            "caption": "",
        }

    if el.name in ("ul", "ol"):
        items: list[list[dict[str, Any]]] = []
        for li in el.find_all("li", recursive=False):
            items.append(_flow_to_blocks(li))
        return {"type": el.name, "items": items}

    if el.name == "table":
        return {"type": "table", "text": _table_to_text(el), "html": str(el)[:300_000]}

    if el.name in ("blockquote", "aside"):
        return {"type": el.name, "text": el.get_text(" ", strip=True)}

    # Definition lists carry the per-instruction Syntax/Description/Semantics
    # structure in the PTX docs; they used to fall through to a `[raw]` placeholder.
    if el.name == "dl":
        items: list[dict[str, Any]] = []
        term = ""
        for child in el.find_all(("dt", "dd"), recursive=False):
            if child.name == "dt":
                term = child.get_text(" ", strip=True)
                continue
            body = _flow_to_blocks(child)
            if not body:
                text = child.get_text(" ", strip=True)
                body = [{"type": "paragraph", "text": text}] if text else []
            if term or body:
                items.append({"term": term, "blocks": body})
            term = ""
        if term:
            items.append({"term": term, "blocks": []})
        return {"type": "dl", "items": items} if items else None

    if el.name == "span":
        return None

    return {"type": "raw", "tag": el.name or "", "html": str(el)[:200_000]}


def _parse_section(section: Tag) -> dict[str, Any]:
    h = None
    for tag in HEADER_TAGS:
        found = section.find(tag, recursive=False)
        if found:
            h = found
            break

    sec_num, title = ("", "")
    level = 0
    if h:
        sec_num, title = _parse_heading(h)
        level = _heading_level(h)

    blocks: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []

    for child in section.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "span":
            continue
        if child.name in SKIP_HEADER_TAGS:
            continue
        if child.name == "section":
            children.append(_parse_section(child))
            continue
        blk = _element_to_block(child)
        if blk is not None:
            blocks.append(blk)

    return {
        "id": section.get("id") or "",
        "section_number": sec_num,
        "title": title,
        "level": level,
        "blocks": blocks,
        "children": children,
    }


def _parse_article_body(body: Tag) -> dict[str, Any]:
    preamble: list[dict[str, Any]] = []
    top_sections: list[dict[str, Any]] = []

    for child in body.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "section":
            top_sections.append(_parse_section(child))
            continue
        blk = _element_to_block(child)
        if blk is not None:
            preamble.append(blk)

    title_el = body.find("p", class_=lambda c: c and "rubric-h1" in c)
    doc_title = _clean_text(title_el.get_text()) if title_el else ""

    return {
        "document_title": doc_title,
        "preamble": preamble,
        "sections": top_sections,
    }


def parse_sphinx_html(html: str) -> dict[str, Any]:
    """Parse a single-page Sphinx HTML document into a tree dict.

    Accepts the HTML body as a string (not a file path) so callers don't
    have to round-trip through a tempfile when they already have the page
    text from ``Fetcher.get``.
    """

    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("div", attrs={"role": "main", "class": lambda c: c and "document" in c})
    if not main:
        main = soup.find("div", role="main")
    if not main:
        main = soup.find("main", role="main")
    if not main:
        main = soup.find("main")
    if not main:
        raise RuntimeError(
            "Could not find main content (div/main[role='main']) — wrong HTML?"
        )

    body = main.find("div", itemprop="articleBody")
    if not body:
        body = main.find("article")
    if not body:
        body = main

    meta_title = ""
    t = soup.find("title")
    if t:
        meta_title = _clean_text(t.get_text())

    tree = _parse_article_body(body)
    return {"html_title": meta_title, **tree}


def _figure_or_image_to_markdown(
    b: dict[str, Any], indent: str, image_base_url: str
) -> str:
    cap = _clean_text(b.get("caption") or "")
    alt_raw = _clean_text(b.get("alt") or "")
    alt = cap or alt_raw or "image"
    src = b.get("src") or ""
    url = _absolute_image_url(src, image_base_url)
    if url.startswith(("http://", "https://", "data:")):
        return f"{indent}![{_markdown_image_alt(alt)}]({url})\n\n"
    if cap or src:
        extra = f" ({src})" if src else ""
        return f"{indent}*{cap}{extra}*\n\n"
    return ""


def _blocks_to_markdown(
    blocks: list[dict[str, Any]], indent: str = "", image_base_url: str = ""
) -> str:
    lines: list[str] = []
    cont = "    "

    for b in blocks:
        t = b.get("type")
        if t == "rubric":
            lines.append(f"{indent}**{_clean_text(b.get('text', ''))}**\n")
        elif t == "paragraph":
            lines.append(f"{indent}{b.get('text', '')}\n\n")
        elif t == "code":
            body = b.get("text", "").rstrip()
            lines.append(f"{indent}```\n")
            for line in body.split("\n"):
                lines.append(f"{indent}{line}\n")
            lines.append(f"{indent}```\n\n")
        elif t == "figure":
            lines.append(_figure_or_image_to_markdown(b, indent, image_base_url))
        elif t == "image":
            lines.append(_figure_or_image_to_markdown(b, indent, image_base_url))
        elif t == "ul":
            for item_blocks in b.get("items") or []:
                inner = _blocks_to_markdown(
                    item_blocks, indent + cont, image_base_url
                ).strip()
                if not inner:
                    lines.append(f"{indent}-\n\n")
                    continue
                first, _, rest = inner.partition("\n")
                lines.append(f"{indent}- {first}\n")
                if rest:
                    for ln in rest.split("\n"):
                        lines.append(f"{indent}{cont}{ln}\n")
                lines.append("\n")
        elif t == "ol":
            for i, item_blocks in enumerate(b.get("items") or [], 1):
                inner = _blocks_to_markdown(
                    item_blocks, indent + cont, image_base_url
                ).strip()
                if not inner:
                    lines.append(f"{indent}{i}.\n\n")
                    continue
                first, _, rest = inner.partition("\n")
                lines.append(f"{indent}{i}. {first}\n")
                if rest:
                    for ln in rest.split("\n"):
                        lines.append(f"{indent}{cont}{ln}\n")
                lines.append("\n")
        elif t == "table":
            txt = (b.get("text") or "").strip()
            if txt:
                for ln in txt.split("\n"):
                    lines.append(f"{indent}{ln}\n")
                lines.append("\n")
            else:
                lines.append(f"{indent}[table: {len(b.get('html', '') or '')} chars]\n\n")
        elif t in ("blockquote", "aside"):
            txt = (b.get("text") or "").strip()
            if txt:
                lines.append(f"{indent}{txt}\n\n")
        elif t == "container":
            lines.append(_blocks_to_markdown(b.get("blocks") or [], indent, image_base_url))
        elif t == "dl":
            for item in b.get("items") or []:
                term = (item.get("term") or "").strip()
                if term:
                    lines.append(f"{indent}**{term}**\n\n")
                inner = _blocks_to_markdown(item.get("blocks") or [], indent, image_base_url)
                if inner.strip():
                    lines.append(inner if inner.endswith("\n") else f"{inner}\n")
        else:
            lines.append(f"{indent}[{t}]\n\n")
    return "".join(lines)


def _section_to_markdown(sec: dict[str, Any], image_base_url: str = "") -> str:
    lvl = sec.get("level") or 1
    hashes = "#" * min(6, max(1, lvl))
    num = sec.get("section_number") or ""
    title = sec.get("title") or ""
    head = f"{hashes} {num} {title}".rstrip() + "\n\n"
    out = [
        head,
        _blocks_to_markdown(sec.get("blocks") or [], "", image_base_url),
    ]
    for ch in sec.get("children") or []:
        out.append(_section_to_markdown(ch, image_base_url))
    return "".join(out)


def tree_to_markdown(data: dict[str, Any], image_base_url: str = "") -> str:
    parts: list[str] = []
    if data.get("document_title"):
        parts.append(f"# {data['document_title']}\n\n")
    parts.append(_blocks_to_markdown(data.get("preamble") or [], "", image_base_url))
    for sec in data.get("sections") or []:
        parts.append(_section_to_markdown(sec, image_base_url))
    return "".join(parts)


def render_html_to_markdown(html: str, page_url: str) -> str:
    """One-shot helper used by the pipeline. Equivalent to
    ``tree_to_markdown(parse_sphinx_html(html), image_base_url=...)`` with
    the image base derived from the page URL."""

    data = parse_sphinx_html(html)
    return tree_to_markdown(data, image_base_url=image_base_url_from_page_url(page_url))
