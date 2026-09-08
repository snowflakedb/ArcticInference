//! Serializing the document tree to portable GitHub Flavored Markdown.
//!
//! "Portable" means the output renders on GitHub without an accompanying asset
//! directory: math uses GFM's `$...$` / `$$...$$`, tables use pipe syntax, and
//! images must resolve to a hosted URL or a data URI. Image `src`s are made
//! portable by [`crate::image::resolve_images`] before this backend runs.
//!
//! The tree comes from [`crate::engine`]; every element maps to a small, fixed
//! Markdown shape. Anything unrecognized falls back to rendering its children,
//! so unknown wrappers never drop content.

use super::{attr, capitalize, join_and, slug, split_off_child, table_rows};
use crate::node::{Element, Node};
use std::collections::HashMap;

/// Rendering options for the Markdown backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct MdOptions {
    /// Also emit a figure's caption as a visible italic line, even though it is
    /// already the image's alt text. Off by default (best for LLM consumption —
    /// no redundancy); on for human viewing, where most renderers hide alt text.
    pub duplicate_captions: bool,
}

/// Render a YAML frontmatter block from `fields`, ready to prepend to a Markdown
/// document. Empty-valued fields are skipped. Values are double-quoted and
/// escaped (so titles with `:` are safe, and numeric-looking ids like
/// `2205.14135` stay strings), except the booleans `true`/`false`, which are
/// emitted bare so they parse as YAML booleans.
pub fn frontmatter(fields: &[(&str, &str)]) -> String {
    frontmatter_with_authors(fields, &[])
}

pub fn frontmatter_with_authors(fields: &[(&str, &str)], authors: &[String]) -> String {
    let mut out = String::from("---\n");
    for (key, value) in fields {
        if value.is_empty() {
            continue;
        }
        out.push_str(key);
        out.push_str(": ");
        write_yaml_value(&mut out, value);
        out.push('\n');
    }
    if !authors.is_empty() {
        out.push_str("authors:\n");
        for author in authors {
            out.push_str("  - ");
            write_yaml_value(&mut out, author);
            out.push('\n');
        }
    }
    out.push_str("---\n\n");
    out
}

fn write_yaml_value(out: &mut String, value: &str) {
    if value == "true" || value == "false" {
        out.push_str(value);
        return;
    }
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    out.push('"');
}

/// The document's title as plain text, if it has one — useful metadata for
/// [`frontmatter`].
pub fn title_text(root: &Element) -> Option<String> {
    root.children.iter().find_map(|n| match n {
        Node::Element(e) if e.name == "title" => Some(heading_text(&e.children)),
        _ => None,
    })
}

pub fn author_texts(root: &Element) -> Vec<String> {
    root.children
        .iter()
        .filter_map(|node| match node {
            Node::Element(element) if element.name == "author" => {
                let text = inline(&element.children);
                text.chars()
                    .any(char::is_alphanumeric)
                    .then(|| text.trim().to_string())
            }
            _ => None,
        })
        .collect()
}

/// The Markdown backend. Wraps [`to_markdown_with`] as a [`crate::Backend`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Markdown {
    pub opts: MdOptions,
}

impl super::Backend for Markdown {
    fn serialize(&self, root: &Element) -> String {
        to_markdown_with(root, &self.opts)
    }
}

/// Render a `<document>` element as a Markdown document (default options).
pub fn to_markdown(root: &Element) -> String {
    to_markdown_with(root, &MdOptions::default())
}

/// Render a `<document>` element as a Markdown document with explicit options.
pub fn to_markdown_with(root: &Element, opts: &MdOptions) -> String {
    // Two GFM-specific normalizations run on a clone so the tree stays neutral
    // for other backends: number footnotes (collecting their definitions for the
    // end of the document) and collapse same-kind nested emphasis.
    let mut root = root.clone();
    let mut footnotes: Vec<(usize, Vec<Node>)> = Vec::new();
    collect_footnotes(
        &mut root.children,
        &mut 1,
        &mut HashMap::new(),
        &mut footnotes,
    );
    root.children = dedup_emphasis(&root.children, false, false);
    let root = &root;

    let mut out = String::new();
    // authblk uses one `\author` per author, so the tree holds several `<author>`
    // nodes; collect their names to render as one line rather than a stack.
    let authors = author_texts(root);
    let mut authors_done = false;
    for child in &root.children {
        match child {
            Node::Element(e) if e.name == "title" => {
                out.push_str("# ");
                out.push_str(&heading_text(&e.children));
                out.push_str("\n\n");
            }
            Node::Element(e) if e.name == "author" => {
                // Emit the whole author list once, on the first author node.
                if !authors_done && !authors.is_empty() {
                    let names: Vec<&str> = authors.iter().map(String::as_str).collect();
                    out.push('*');
                    out.push_str(&join_and(&names));
                    out.push_str("*\n\n");
                }
                authors_done = true;
            }
            Node::Element(e) if e.name == "date" => {
                out.push_str(inline(&e.children).trim());
                out.push_str("\n\n");
            }
            Node::Element(e) => block_element(e, &mut out, 0, opts),
            Node::Text(t) => push_paragraph(&mut out, inline_text(t)),
        }
    }
    emit_footnote_definitions(&mut out, &footnotes);
    normalize_blank_lines(&out)
}

/// Append the GFM footnote definitions (`[^n]: text`) collected from the tree.
/// Each definition is a single line — a footnote body is inline, so any interior
/// hard break is folded to a space to keep the definition well-formed.
fn emit_footnote_definitions(out: &mut String, footnotes: &[(usize, Vec<Node>)]) {
    for (n, children) in footnotes {
        let body = inline(children);
        let body = body.split_whitespace().collect::<Vec<_>>().join(" ");
        if !body.is_empty() {
            push_paragraph(out, format!("[^{n}]: {body}"));
        }
    }
}

// --- block level --------------------------------------------------------

/// Render a sequence of block-level nodes into `out`. `indent` is the current
/// left margin (used inside list items), in spaces.
fn blocks(nodes: &[Node], out: &mut String, indent: usize, opts: &MdOptions) {
    for node in nodes {
        match node {
            Node::Element(e) => block_element(e, out, indent, opts),
            Node::Text(t) => {
                if !t.trim().is_empty() {
                    push_paragraph(out, inline_text(t));
                }
            }
        }
    }
}

fn block_element(e: &Element, out: &mut String, indent: usize, opts: &MdOptions) {
    match e.name.as_str() {
        "section" => render_section(e, out, opts),
        "p" if is_orphan_punctuation_para(e) => {
            // A paragraph with no words — only stray punctuation, e.g. the lone
            // `.` a source leaves between a `\captionof{…}` and its `\label`.
            // Drop the punctuation but keep any `\label` anchors so cross-refs
            // still resolve.
            for child in &e.children {
                if let Node::Element(l) = child
                    && l.name == "label"
                {
                    block_element(l, out, indent, opts);
                }
            }
        }
        "p" => push_paragraph(out, inline(&e.children)),
        "abstract" => {
            out.push_str("## Abstract\n\n");
            blocks(&e.children, out, indent, opts);
        }
        "itemize" => render_list(e, out, indent, false, opts),
        "bibliography" => {
            // A heading so references don't read as part of the last section.
            out.push_str("## References\n\n");
            render_list(e, out, indent, false, opts);
        }
        "enumerate" => render_list(e, out, indent, true, opts),
        "description" => render_list(e, out, indent, false, opts),
        "blockquote" => render_blockquote(e, out, opts),
        "verbatim" => render_code_block(e, out),
        "math" => {
            // Emit a link anchor for each equation label the resolve pass
            // recorded, so `\eqref`/`\cref` to an equation lands here.
            if let Some(ids) = attr(e, "labels") {
                let anchors: String = ids
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|id| format!("<a id=\"{}\"></a>", slug(id)))
                    .collect();
                if !anchors.is_empty() {
                    push_paragraph(out, anchors);
                }
            }
            // A display-math element sits at block level.
            push_block(out, format!("$$\n{}\n$$", katex_body(e)));
        }
        "tabular" => render_table(e, out),
        "float" => render_float(e, out, opts),
        "caption" => {
            // Skip the visible caption when it is already the image's alt text
            // (marked `mirrored` by the engine), unless duplication is requested.
            if opts.duplicate_captions || attr(e, "mirrored").is_none() {
                let body = inline(&e.children);
                let body = body.trim();
                // A numbered float caption leads with its "Figure 3:" label.
                let text = match attr(e, "prefix") {
                    Some(prefix) => format!("**{prefix}:** {body}"),
                    None => format!("*{body}*"),
                };
                push_paragraph(out, text);
            } else {
                emit_label_anchors(&e.children, out);
            }
        }
        "image" => push_paragraph(out, render_image(e)),
        "label" => {
            // Emit an anchor so cross-reference links land (LaTeX's `\label`).
            if let Some(id) = attr(e, "id") {
                push_paragraph(out, format!("<a id=\"{}\"></a>", slug(id)));
            }
        }
        "environment" => {
            // Lead with a bold label so theorem/proof blocks stay recognizable.
            // Prefer a resolved number ("Theorem 1"), then a proof's captured
            // lead ("Proof of Theorem 1"), else the bare environment name.
            let (term, body) = split_off_child(e, "term");
            let term_lead = term.map(|t| inline(&t.children).trim().to_string());
            // A theorem's optional note (captured as a <term>) shows in parens
            // after the number: "Theorem 1 (Pythagoras)". A proof's <term> is a
            // standalone lead ("Proof of Theorem 1") with no prefix.
            let lead = match (attr(e, "prefix"), &term_lead) {
                (Some(p), Some(t)) => Some(format!("{p} ({t})")),
                (Some(p), None) => Some(p.to_string()),
                (None, Some(t)) => Some(t.clone()),
                (None, None) => attr(e, "name").map(|n| capitalize(n)),
            };
            if let Some(lead) = lead {
                push_paragraph(out, format!("**{lead}.**"));
            }
            blocks(&body, out, indent, opts);
        }
        // "align" (center/flush) and any other wrapper: render transparently.
        _ => blocks(&e.children, out, indent, opts),
    }
}

fn emit_label_anchors(nodes: &[Node], out: &mut String) {
    for node in nodes {
        if let Node::Element(element) = node {
            if element.name == "label" {
                if let Some(id) = attr(element, "id") {
                    push_paragraph(out, format!("<a id=\"{}\"></a>", slug(id)));
                }
            } else {
                emit_label_anchors(&element.children, out);
            }
        }
    }
}

fn render_section(e: &Element, out: &mut String, opts: &MdOptions) {
    let level = super::section_display_level(e);
    // Document title is h1; sections (level=1) start at h2. Clamp to GFM's six levels.
    let hashes = (level + 1).clamp(1, 6) as usize;

    let (title, body) = split_off_child(e, "title");
    if let Some(t) = title {
        // Collapse hard line-breaks in headings to a single space so
        // e.g. \title{Foo:\\ Bar} becomes "# Foo: Bar" not two lines.
        // GFM/GitHub does not process `$…$` inside ATX headings, so lower heading
        // math to text/Unicode (`P$_{os}$`→`Pos`) from the tree's `<math>` nodes —
        // not by re-parsing the serialized string, which would mistake an escaped
        // literal `$` (from `\$5`) for a math delimiter.
        let title = match attr(e, "number") {
            Some(number) => format!("{number} {}", heading_text(&t.children)),
            None => heading_text(&t.children),
        };
        out.push_str(&"#".repeat(hashes));
        out.push(' ');
        out.push_str(&title);
        out.push_str("\n\n");
    }
    blocks(&body, out, 0, opts);
}

fn heading_text(nodes: &[Node]) -> String {
    inline(&lower_math_nodes(nodes))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Replace `<math>` elements in an inline node tree with their lowered plain-text
/// form, leaving every other node — and any literal `$` in text — untouched.
/// Used for headings, where GFM won't render `$…$`; lowering from the tree
/// (rather than the serialized string) keeps escaped literal `$` intact.
fn lower_math_nodes(nodes: &[Node]) -> Vec<Node> {
    nodes
        .iter()
        .map(|n| match n {
            Node::Element(e) if e.name == "math" => Node::Text(math_to_text(&text_content(e))),
            Node::Element(e) => {
                let mut ne = e.clone();
                ne.children = lower_math_nodes(&e.children);
                Node::Element(ne)
            }
            Node::Text(t) => Node::Text(t.clone()),
        })
        .collect()
}

fn render_blockquote(e: &Element, out: &mut String, opts: &MdOptions) {
    let mut inner = String::new();
    blocks(&e.children, &mut inner, 0, opts);
    for line in inner.trim_end().lines() {
        out.push_str("> ");
        out.push_str(line);
        out.push('\n');
    }
    out.push('\n');
}

fn render_code_block(e: &Element, out: &mut String) {
    if let Some(ids) = attr(e, "labels") {
        let anchors = ids
            .split(',')
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(|id| format!("<a id=\"{}\"></a>", slug(id)))
            .collect::<String>();
        if !anchors.is_empty() {
            push_paragraph(out, anchors);
        }
    }
    let text = text_content(e);
    let language = attr(e, "language").map(String::as_str).unwrap_or("");
    push_block(
        out,
        format!("```{language}\n{}\n```", text.trim_matches('\n')),
    );
}

fn render_float(e: &Element, out: &mut String, opts: &MdOptions) {
    // A figure/table's caption becomes its image's alt text (Markdown's caption
    // field). Work on a clone: set alt on images from the caption, and mark the
    // caption `mirrored` so the visible line is suppressed by default (avoiding
    // redundancy) unless `duplicate_captions` is set. This is a Markdown-only
    // concern; the tree stays neutral.
    //
    // Flatten first so the caption/image pairing sees them as siblings regardless
    // of how the source grouped them: `\begin{center}` wraps the body in an
    // `<align>`, and `\captionof` puts its caption in a caption-only nested float.
    let mut children = flatten_float_children(&e.children);

    // Collect each `<caption>`'s alt text: the SAME resolved render the visible
    // caption uses (`inline`), reduced to a plain-text alt so it carries
    // rendered math and resolved `\ref` numbers, unlike the raw source text.
    let captions: Vec<(usize, String)> = children
        .iter()
        .enumerate()
        .filter_map(|(i, n)| match n {
            Node::Element(c) if c.name == "caption" => {
                let alt = alt_from_markdown(&inline(&c.children));
                (!alt.trim().is_empty()).then_some((i, alt))
            }
            _ => None,
        })
        .collect();

    // Pair captions with the images they describe, marking each paired caption
    // `mirrored`. A single caption describes the whole float (apply to every
    // image, regardless of caption-above vs -below). Several captions are
    // side-by-side panels — one float, one `\caption` per panel — so each
    // captions only the images in its own segment and each keeps its own number.
    let mirrored: Vec<usize> = if captions.len() <= 1 {
        match captions.first() {
            Some((i, alt)) if set_image_alt(&mut children, alt) => vec![*i],
            _ => vec![],
        }
    } else {
        // Whether captions sit below their images (the LaTeX default) or above
        // is decided once for the whole float, by where the first image falls
        // relative to the first caption, so every panel is paired the same way.
        // Guessing wrong would hand each image to the neighbouring panel's
        // caption and leave the true caption unmirrored, leaking a stray line.
        let first_cap = captions[0].0;
        let caption_below = children[..first_cap].iter().any(contains_free_image);
        let mut mirrored = Vec::new();
        for (k, (cap_i, alt)) in captions.iter().enumerate() {
            // The half-open node range this caption owns (the caption node
            // itself is skipped below): the images on its own side, up to the
            // neighbouring caption. The outermost caption also sweeps any images
            // before the first / past the last.
            let (lo, hi) = if caption_below {
                let lo = if k == 0 { 0 } else { captions[k - 1].0 + 1 };
                let hi = if k + 1 == captions.len() {
                    children.len()
                } else {
                    *cap_i
                };
                (lo, hi)
            } else {
                let lo = if k == 0 { 0 } else { *cap_i + 1 };
                let hi = if k + 1 == captions.len() {
                    children.len()
                } else {
                    captions[k + 1].0
                };
                (lo, hi)
            };
            let mut set = false;
            for (j, child) in children[lo..hi].iter_mut().enumerate() {
                if lo + j == *cap_i {
                    continue; // don't treat the caption node as captioned content
                }
                set |= set_image_alt(std::slice::from_mut(child), alt);
            }
            if set {
                mirrored.push(*cap_i);
            }
        }
        mirrored
    };

    for i in mirrored {
        if let Node::Element(c) = &mut children[i] {
            c.attributes.push(("mirrored".into(), "1".into()));
        }
    }
    blocks(&children, out, 0, opts);
}

/// Recursively set `alt` on images lacking it, without descending into nested
/// floats (a subfigure keeps its own caption). Returns whether any image got it.
fn set_image_alt(nodes: &mut [Node], alt: &str) -> bool {
    let mut set = false;
    for node in nodes {
        let Node::Element(e) = node else { continue };
        match e.name.as_str() {
            "float" => {}
            "image" => {
                if !e.attributes.iter().any(|(k, _)| k == "alt") {
                    e.attributes.push(("alt".into(), alt.to_string()));
                    set = true;
                }
            }
            _ => set |= set_image_alt(&mut e.children, alt),
        }
    }
    set
}

/// A `<p>` that is a stray punctuation fragment stranded beside a `\label`
/// anchor — e.g. the lone `.` a source leaves between a `\captionof{…}` and its
/// `\label`. It must contain a `<label>` (the structural anchor that identifies
/// this as a float/caption boundary, not prose) and, apart from that, only
/// sentence punctuation and whitespace. The backend then drops the punctuation
/// and keeps the anchor. Requiring the label keeps a genuine standalone `?`/`!`/
/// `…` prose paragraph — which has no label — from being removed.
fn is_orphan_punctuation_para(e: &Element) -> bool {
    let mut has_punct = false;
    let mut has_label = false;
    for child in &e.children {
        match child {
            Node::Text(t) => {
                for c in t.chars() {
                    if c.is_whitespace() {
                        continue;
                    }
                    if matches!(c, '.' | ',' | ';' | ':' | '!' | '?') {
                        has_punct = true;
                    } else {
                        return false; // real content (letters, digits, symbols)
                    }
                }
            }
            Node::Element(el) if el.name == "label" => has_label = true,
            _ => return false,
        }
    }
    has_punct && has_label
}

/// Whether `node` contains an image that [`set_image_alt`] could reach — i.e.
/// one not walled off inside a nested float. Used to decide, once per
/// multi-caption float, whether the captions sit above or below their images.
fn contains_free_image(node: &Node) -> bool {
    match node {
        Node::Element(e) => match e.name.as_str() {
            "float" => false,
            "image" => true,
            _ => e.children.iter().any(contains_free_image),
        },
        _ => false,
    }
}

/// Expose a float's captions and images as direct siblings so the caption/image
/// pairing works regardless of source grouping:
///   - an `<align>` (from `\begin{center}`/`\centering`) is a transparent
///     centering wrapper — splice its children in;
///   - a caption-only nested `<float>` (a `\captionof`, which carries a caption
///     for adjacent content but no image of its own) is replaced by its caption,
///     so the caption pairs with the sibling image it describes.
///
/// A real subfigure (a nested float WITH an image) is left intact — it keeps its
/// own caption and is rendered by its own `render_float`.
fn flatten_float_children(children: &[Node]) -> Vec<Node> {
    let mut out = Vec::with_capacity(children.len());
    for child in children {
        match child {
            Node::Element(e) if e.name == "align" => {
                out.extend(flatten_float_children(&e.children));
            }
            Node::Element(e) if e.name == "float" && is_caption_carrier(e) => {
                out.extend(e.children.iter().cloned());
            }
            other => out.push(other.clone()),
        }
    }
    out
}

/// A nested float that carries a caption but no image of its own (a `\captionof`
/// for sibling content), as opposed to a subfigure (which has its own image).
fn is_caption_carrier(float: &Element) -> bool {
    float
        .children
        .iter()
        .any(|n| matches!(n, Node::Element(e) if e.name == "caption"))
        && !float.children.iter().any(contains_free_image)
}

/// Render a `<math>` element's TeX for KaTeX (GFM math), without `$…$`/`$$…$$`
/// delimiters. The KaTeX post-processing is shared with the HTML backend and
/// lives in the `texmath` submodule, so the tree stays format-neutral.
fn katex_body(e: &Element) -> String {
    super::texmath::katex(&text_content(e), attr(e, "env").map(String::as_str))
}

fn render_image(e: &Element) -> String {
    let src = attr(e, "src").cloned().unwrap_or_default();
    let alt = attr(e, "alt").map(|s| alt_text(s)).unwrap_or_default();
    if crate::image::is_pdf(&src) {
        let label = if alt.is_empty() { "Figure" } else { &alt };
        format!("[{label}]({src})")
    } else {
        format!("![{alt}]({src})")
    }
}

/// Turn a resolved numeric citation marker into one whose numbers link to their
/// reference entries. `text` is the finished marker (`[1, 2]`); `anchors` is the
/// per-number anchor list the resolve pass recorded, aligned with the numbers in
/// order (`ref-a,ref-b`). Each `N` becomes `[N](#ref-key)`. Without an anchor
/// list (author-year, or any other marker), the text is returned unchanged.
fn link_cite(text: &str, anchors: Option<&String>) -> String {
    let Some(anchors) = anchors else {
        return text.to_string();
    };
    // The numeric marker is bracketed numbers: `[1, 2]`. Link the inner numbers
    // and keep the surrounding brackets literal so the shape is unchanged.
    let Some(inner) = text.strip_prefix('[').and_then(|s| s.strip_suffix(']')) else {
        return text.to_string();
    };
    let anchors: Vec<&str> = anchors.split(',').filter(|s| !s.is_empty()).collect();
    let numbers: Vec<&str> = inner.split(", ").collect();
    // Alignment failure (shouldn't happen): fall back to the plain marker.
    if numbers.len() != anchors.len() {
        return text.to_string();
    }
    let linked: Vec<String> = numbers
        .iter()
        .zip(anchors.iter())
        .map(|(num, anchor)| format!("[{num}](#{})", slug(anchor)))
        .collect();
    format!("[{}]", linked.join(", "))
}

/// Sanitize caption text for use as image alt text: collapse to a single line
/// and escape brackets so they don't break the `![...]()` syntax.
fn alt_text(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace('[', "\\[")
        .replace(']', "\\]")
}

/// Reduce a rendered-Markdown caption to a plain-text alt attribute. The alt
/// text mirrors what the visible caption shows, minus Markdown/math syntax that
/// is illegal or noisy in an alt attribute:
///   - `$…$` / `$$…$$` math delimiters are dropped, and a small, general set of
///     common TeX math tokens is mapped to Unicode (`\times`→×, `\alpha`→α, …)
///     so the alt reads as text rather than raw TeX;
///   - Markdown links `[text](url)` reduce to `text` (so a resolved `\ref`
///     shows its number, e.g. "Figure 3", not a link or an empty "Fig. .");
///   - HTML tags the resolved render can emit — a `\label`'s `<a id="…"></a>`
///     anchor, `<sup>`/`<sub>`/`<u>` wrappers — are dropped, keeping their inner
///     text;
///   - emphasis markers `**`/`*` and code backticks are stripped.
///
/// Whitespace/bracket sanitization is left to [`alt_text`], applied at render.
fn alt_from_markdown(md: &str) -> String {
    let mut s = strip_footnote_markers(md);
    s = strip_md_links(&s);
    s = strip_math(&s);
    s = strip_html_tags(&s);
    // Drop residual emphasis/code markers. Order-independent: these are literal
    // in prose (the source escapes intentional ones), so removing them is safe.
    s = s.replace("**", "").replace(['*', '`'], "");
    s
}

/// Drop GFM footnote markers `[^id]` from `s`. A footnote that appears inside a
/// caption renders as `[^n]`; in the caption's alt text that marker is noise (it
/// references a definition that only renders elsewhere), so it is removed.
fn strip_footnote_markers(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '['
            && chars.get(i + 1) == Some(&'^')
            && let Some(end) = find_char(&chars, i + 2, ']')
        {
            i = end + 1;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Drop HTML tags (`<a id="…">`, `</a>`, `<sup>`, …) from `s`, keeping the text
/// between them. The resolved caption render emits these for inline `\label`
/// anchors and super/subscripts; as raw markup they are alt-text noise. A lone
/// `<` with no closing `>` is kept literally (it isn't a tag).
fn strip_html_tags(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '<'
            && let Some(end) = find_char(&chars, i + 1, '>')
        {
            i = end + 1;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Replace Markdown links `[text](url)` with just their `text`. A bare `[x]`
/// with no following `(...)` is left untouched (it isn't a link).
fn strip_md_links(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '['
            && let Some(text_end) = find_char(&chars, i + 1, ']')
            && text_end + 1 < chars.len()
            && chars[text_end + 1] == '('
            && let Some(url_end) = find_char(&chars, text_end + 2, ')')
        {
            out.extend(&chars[i + 1..text_end]);
            i = url_end + 1;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Index of the next `c` at or after `from`, if any.
fn find_char(chars: &[char], from: usize, c: char) -> Option<usize> {
    (from..chars.len()).find(|&i| chars[i] == c)
}

/// Strip `$…$` / `$$…$$` math delimiters and rewrite the body into readable
/// plain text: common TeX macros mapped to Unicode, remaining control words and
/// braces/`^`/`_`/whitespace normalized away. Text outside math is untouched.
fn strip_math(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '$' {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        // Enter a math run; skip the opening `$`/`$$` and find the close.
        i += 1;
        if i < chars.len() && chars[i] == '$' {
            i += 1;
        }
        let mut body = String::new();
        while i < chars.len() && chars[i] != '$' {
            body.push(chars[i]);
            i += 1;
        }
        // Skip the closing `$`/`$$`.
        if i < chars.len() && chars[i] == '$' {
            i += 1;
        }
        if i < chars.len() && chars[i] == '$' {
            i += 1;
        }
        out.push_str(&math_to_text(&body));
    }
    out
}

/// Reduce a math body (no `$` delimiters) to readable plain text: map a small,
/// general set of common macros to Unicode, then drop remaining control words,
/// braces, and `^`/`_` structure markers. General on purpose — not paper- or
/// domain-specific — favoring "readable-ish text" over faithful math.
fn math_to_text(body: &str) -> String {
    let chars: Vec<char> = body.chars().collect();
    let mut out = String::with_capacity(body.len());
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '\\' => {
                // Read the control word (letters); a control symbol is a single
                // non-letter char (e.g. `\,`, `\%`).
                let start = i + 1;
                let mut j = start;
                while j < chars.len() && chars[j].is_ascii_alphabetic() {
                    j += 1;
                }
                if j == start {
                    // Control symbol: keep an escaped literal (`\%`→`%`, `\{`→`{`);
                    // drop spacing symbols (`\,` `\;` `\!` `\:`).
                    if j < chars.len() {
                        let c = chars[j];
                        if !matches!(c, ',' | ';' | '!' | ':' | ' ') {
                            out.push(c);
                        }
                        i = j + 1;
                    } else {
                        i = j;
                    }
                    continue;
                }
                let word: String = chars[start..j].iter().collect();
                if let Some(sym) = math_symbol(&word) {
                    out.push_str(sym);
                }
                // Unknown macros drop their name but keep any `{...}` argument
                // content (handled as ordinary chars below).
                i = j;
            }
            '{' | '}' => i += 1,
            '^' | '_' => {
                // Keep the scripted token as adjacent text (`x_i`→`xi`, `N^2`→`N2`).
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    // Collapse the whitespace TeX leaves between tokens.
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Map a common TeX math control word to a Unicode glyph, if known. Small and
/// general — Greek letters and a handful of ubiquitous operators/relations —
/// not an exhaustive or domain-specific table.
fn math_symbol(word: &str) -> Option<&'static str> {
    Some(match word {
        "times" => "×",
        "cdot" | "cdots" => "·",
        "div" => "÷",
        "pm" => "±",
        "mp" => "∓",
        "leq" | "le" => "≤",
        "geq" | "ge" => "≥",
        "neq" | "ne" => "≠",
        "approx" => "≈",
        "sim" => "∼",
        "to" | "rightarrow" => "→",
        "leftarrow" => "←",
        "infty" => "∞",
        "sum" => "∑",
        "prod" => "∏",
        "int" => "∫",
        "partial" => "∂",
        "nabla" => "∇",
        "in" => "∈",
        "forall" => "∀",
        "exists" => "∃",
        "alpha" => "α",
        "beta" => "β",
        "gamma" => "γ",
        "delta" => "δ",
        "epsilon" | "varepsilon" => "ε",
        "zeta" => "ζ",
        "eta" => "η",
        "theta" => "θ",
        "iota" => "ι",
        "kappa" => "κ",
        "lambda" => "λ",
        "mu" => "μ",
        "nu" => "ν",
        "xi" => "ξ",
        "pi" => "π",
        "rho" => "ρ",
        "sigma" => "σ",
        "tau" => "τ",
        "phi" | "varphi" => "φ",
        "chi" => "χ",
        "psi" => "ψ",
        "omega" => "ω",
        "Gamma" => "Γ",
        "Delta" => "Δ",
        "Theta" => "Θ",
        "Lambda" => "Λ",
        "Xi" => "Ξ",
        "Pi" => "Π",
        "Sigma" => "Σ",
        "Phi" => "Φ",
        "Psi" => "Ψ",
        "Omega" => "Ω",
        _ => return None,
    })
}

fn render_list(e: &Element, out: &mut String, indent: usize, ordered: bool, opts: &MdOptions) {
    let pad = " ".repeat(indent);
    let mut number = 1;
    for child in &e.children {
        let Node::Element(item) = child else { continue };
        let marker = if ordered {
            format!("{number}. ")
        } else {
            "- ".to_string()
        };
        number += 1;

        // A description term becomes a bold lead-in; a bibitem is keyed below.
        let mut lead = String::new();
        let mut body: Vec<Node> = Vec::new();
        for grandchild in &item.children {
            match grandchild {
                Node::Element(t) if t.name == "term" => {
                    lead = format!("**{}** ", inline(&t.children).trim());
                }
                other => body.push(other.clone()),
            }
        }
        // A bibitem is keyed by its resolved citation number (`[7]`) in numeric
        // mode; author-year mode marks it `unlabeled` (the entry stands alone,
        // no bracket). A description term becomes a bold lead-in.
        if let Some(number) = attr(item, "number") {
            lead = format!("[{number}] ");
        } else if attr(item, "unlabeled").is_some() {
            lead = String::new();
        } else if let Some(key) = attr(item, "key") {
            lead = format!("[{key}] ");
        }
        // A reference entry carries a link anchor (`ref-<key>`) so in-text cites
        // can jump to it; emit it inline before the lead, matching how `<label>`
        // anchors are emitted for cross-references.
        if let Some(anchor) = attr(item, "anchor") {
            lead = format!("<a id=\"{}\"></a>{lead}", slug(anchor));
        }

        let mut inner = String::new();
        blocks(&body, &mut inner, 0, opts);
        let inner = format!("{lead}{}", inner.trim_start());
        write_item(out, &pad, &marker, inner.trim_end());
    }
    out.push('\n');
}

/// Write one list item: the marker on the first line, hanging indent on the
/// rest so nested blocks stay inside the item.
fn write_item(out: &mut String, pad: &str, marker: &str, content: &str) {
    let hang = " ".repeat(marker.len());
    let mut lines = content.lines();
    if let Some(first) = lines.next() {
        out.push_str(pad);
        out.push_str(marker);
        out.push_str(first);
        out.push('\n');
    }
    for line in lines {
        if line.is_empty() {
            out.push('\n');
        } else {
            out.push_str(pad);
            out.push_str(&hang);
            out.push_str(line);
            out.push('\n');
        }
    }
}

fn render_table(e: &Element, out: &mut String) {
    let rows = table_rows(e);
    if rows.is_empty() {
        return;
    }

    let all_cells: Vec<Vec<String>> = rows.iter().map(|r| cells_of(r)).collect();
    let width = all_cells.iter().map(Vec::len).max().unwrap_or(0);
    if width == 0 {
        return;
    }

    // GFM requires a header row; use the first row as the header.
    out.push_str(&table_row(&all_cells[0], width));
    out.push('|');
    for _ in 0..width {
        out.push_str(" --- |");
    }
    out.push('\n');
    for cells in &all_cells[1..] {
        out.push_str(&table_row(cells, width));
    }
    out.push('\n');
}

/// The rendered inline text of a row's cells.
fn cells_of(row: &Element) -> Vec<String> {
    row.children
        .iter()
        .filter_map(|n| match n {
            Node::Element(c) if c.name == "cell" => {
                let text = inline(&c.children).replace('|', "\\|");
                // A pipe-table cell must be a single physical line, so fold any
                // interior line break (e.g. from `\makecell{a\\b}`) into `<br>`.
                Some(fold_cell_breaks(&text))
            }
            _ => None,
        })
        .collect()
}

/// Collapse a cell's interior line breaks into GFM `<br>` so they don't split
/// the pipe-table row. Blank segments are dropped; the result is trimmed.
fn fold_cell_breaks(text: &str) -> String {
    text.split('\n')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("<br>")
}

/// One `| a | b |` table line, padded to `width` columns.
fn table_row(cells: &[String], width: usize) -> String {
    let mut s = String::from("|");
    for i in 0..width {
        s.push(' ');
        s.push_str(cells.get(i).map(String::as_str).unwrap_or(""));
        s.push_str(" |");
    }
    s.push('\n');
    s
}

// --- inline level -------------------------------------------------------

/// Render inline nodes to a Markdown string.
fn inline(nodes: &[Node]) -> String {
    let mut s = String::new();
    for node in coalesce_code_spans(nodes) {
        match node {
            Node::Text(t) => s.push_str(&inline_text(&t)),
            Node::Element(e) => s.push_str(&inline_element(&e)),
        }
    }
    s
}

/// Merge runs of directly-adjacent plain-text code spans into one, so
/// `\texttt{L1/}\texttt{x.py}` renders as `` `L1/x.py` `` rather than two
/// touching spans (which GFM parses as broken adjacent code). Empty text nodes
/// (e.g. from a dropped `\allowbreak`) don't separate the spans; a real space
/// does.
fn coalesce_code_spans(nodes: &[Node]) -> Vec<Node> {
    let is_plain_code = |n: &Node| {
        matches!(n, Node::Element(e)
            if e.name == "code" && e.children.iter().all(|c| matches!(c, Node::Text(_))))
    };
    let mut out: Vec<Node> = Vec::with_capacity(nodes.len());
    for node in nodes {
        // Drop empties so a dropped command between two spans doesn't split them.
        if matches!(node, Node::Text(t) if t.is_empty()) {
            continue;
        }
        if is_plain_code(node)
            && out.last().is_some_and(is_plain_code)
            && let (Some(Node::Element(prev)), Node::Element(e)) = (out.last_mut(), node)
        {
            prev.children.extend(e.children.iter().cloned());
            continue;
        }
        out.push(node.clone());
    }
    out
}

fn inline_element(e: &Element) -> String {
    let inner = inline(&e.children);
    match e.name.as_str() {
        "bold" => wrap_emphasis("**", &inner),
        "italic" | "emph" => wrap_emphasis("*", &inner),
        // A code span is verbatim, and GFM can't nest emphasis inside backticks.
        // When the body is plain text (the common case: identifiers, filenames)
        // emit a code span with the raw, unescaped text so `\texttt{__init__}`
        // stays `__init__`. When it carries inline formatting, keep that
        // formatting instead — the monospace can't be expressed in GFM anyway.
        "code" => {
            if e.children.iter().all(|n| matches!(n, Node::Text(_))) {
                format!("`{}`", text_content(e))
            } else {
                inner
            }
        }
        "underline" => format!("<u>{inner}</u>"),
        "smallcaps" => inner,
        "superscript" => format!("<sup>{inner}</sup>"),
        "subscript" => format!("<sub>{inner}</sub>"),
        "linebreak" => "  \n".to_string(),
        "math" => {
            if attr(e, "mode").map(String::as_str) == Some("display") {
                format!("$$\n{}\n$$", katex_body(e))
            } else {
                format!("${}$", katex_body(e))
            }
        }
        "link" => {
            let href = attr(e, "href").cloned().unwrap_or_default();
            let text = if inner.trim().is_empty() {
                href.clone()
            } else {
                inner
            };
            format!("[{text}]({href})")
        }
        "ref" => {
            let target = attr(e, "target").cloned().unwrap_or_default();
            // The resolve pass fills `resolved`/`anchor` when the target numbered;
            // otherwise fall back to a raw link on the label name.
            match attr(e, "resolved") {
                Some(text) => {
                    let anchor = attr(e, "anchor").map(String::as_str).unwrap_or(&target);
                    format!("[{text}](#{})", slug(anchor))
                }
                None => format!("[{target}](#{})", slug(&target)),
            }
        }
        "cite" => match attr(e, "resolved") {
            // The resolve pass produces the full marker including any brackets or
            // parens (numeric "[1, 2]" vs author-year "(Smith, 2020)"). In numeric
            // mode it also records a per-number `anchors` list, so each `[N]` is
            // linked to its reference entry (`[[1](#ref-a), [2](#ref-b)]`).
            Some(text) => link_cite(text, attr(e, "anchors")),
            None => {
                let keys = attr(e, "keys").cloned().unwrap_or_default();
                format!("[{keys}]")
            }
        },
        "image" => render_image(e),
        "footnote" => match attr(e, "number") {
            // Footnotes are lowered to GFM footnotes: an inline `[^n]` reference
            // here, with the definition emitted once at the end of the document
            // (see `collect_footnotes`). `number` is assigned by that pre-pass.
            Some(n) => format!("[^{n}]"),
            None => String::new(),
        },
        "label" => attr(e, "id")
            .map(|id| format!("<a id=\"{}\"></a>", slug(id)))
            .unwrap_or_default(),
        // Unknown inline wrapper: keep its text.
        _ => inner,
    }
}

/// Wrap `inner` in an emphasis run (`*`/`**`), keeping any leading/trailing
/// whitespace OUTSIDE the delimiters. CommonMark requires a closing delimiter to
/// be right-flanking, which a trailing space defeats — so `\textbf{Memory: }`
/// must emit `**Memory:** `, not the non-rendering `**Memory: **`. An all-blank
/// (or empty) body carries no emphasis and is returned as-is.
fn wrap_emphasis(delim: &str, inner: &str) -> String {
    let trimmed = inner.trim();
    if trimmed.is_empty() {
        return inner.to_string();
    }
    let lead = &inner[..inner.len() - inner.trim_start().len()];
    let trail = &inner[inner.trim_end().len()..];
    format!("{lead}{delim}{trimmed}{delim}{trail}")
}

/// Collapse emphasis nested inside the same kind of emphasis, which GFM cannot
/// express: `\emph{\emph{x} y}` would emit the unbalanced `**x* y*`. LaTeX nests
/// by toggling, but the readable, always-valid Markdown is a single run, so an
/// inner `emph`/`italic` under an `emph`/`italic` (or `bold` under `bold`) is
/// unwrapped. Different kinds still nest fine (`***bold italic***`). Runs before
/// serialization on a clone of the tree, so other backends keep true nesting.
fn dedup_emphasis(nodes: &[Node], in_italic: bool, in_bold: bool) -> Vec<Node> {
    let mut out = Vec::with_capacity(nodes.len());
    for node in nodes {
        let Node::Element(e) = node else {
            out.push(node.clone());
            continue;
        };
        let is_italic = matches!(e.name.as_str(), "italic" | "emph");
        let is_bold = e.name == "bold";
        let children = dedup_emphasis(&e.children, in_italic || is_italic, in_bold || is_bold);
        if (is_italic && in_italic) || (is_bold && in_bold) {
            out.extend(children); // redundant wrapper: keep only its content
        } else {
            let mut ne = e.clone();
            ne.children = children;
            out.push(Node::Element(ne));
        }
    }
    out
}

/// Number every `<footnote>` in document order and collect its content, so the
/// backend can emit an inline `[^n]` marker (via [`inline_element`]) and a single
/// definition list at the end of the document. Mutates the tree in place (on the
/// backend's clone) to record each footnote's `number`. Nested footnotes are
/// illegal in LaTeX, so this does not descend into a footnote's own body.
fn collect_footnotes(
    nodes: &mut [Node],
    next: &mut usize,
    numbers: &mut HashMap<String, usize>,
    defs: &mut Vec<(usize, Vec<Node>)>,
) {
    for node in nodes {
        let Node::Element(e) = node else { continue };
        if e.name == "footnote" {
            // An empty footnote (`\footnote{}`) gets no number, so no dangling
            // `[^n]` marker and no orphan definition are emitted.
            if inline(&e.children).trim().is_empty() {
                continue;
            }
            let key = attr(e, "key").filter(|key| !key.is_empty()).cloned();
            let n = if let Some(number) = key.as_ref().and_then(|key| numbers.get(key)) {
                *number
            } else {
                let number = *next;
                *next += 1;
                if let Some(key) = key {
                    numbers.insert(key, number);
                }
                defs.push((number, e.children.clone()));
                number
            };
            e.attributes.push(("number".into(), n.to_string()));
        } else {
            collect_footnotes(&mut e.children, next, numbers, defs);
        }
    }
}

/// Escape Markdown metacharacters in literal text so it renders verbatim.
fn inline_text(s: &str) -> String {
    // Apply LaTeX text ligatures (dashes, curly quotes) before escaping Markdown
    // metacharacters. Runs only on prose text — code/math take other paths.
    let s = super::typographic(s);
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if matches!(c, '\\' | '`' | '*' | '_' | '<' | '[' | ']') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

// --- helpers ------------------------------------------------------------

/// The concatenated text of an element's direct text children (used for math,
/// whose body is stored as a single text node, and plain-text code spans).
/// Deliberately non-recursive — the shared [`super::text_content`] recurses.
fn text_content(e: &Element) -> String {
    e.children
        .iter()
        .filter_map(|n| match n {
            Node::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect()
}

/// Append `text` as a paragraph followed by a blank line, if non-empty.
fn push_paragraph(out: &mut String, text: String) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    out.push_str(text);
    out.push_str("\n\n");
}

/// Append a pre-formatted block (already internally laid out) plus a blank line.
fn push_block(out: &mut String, block: String) {
    out.push_str(&block);
    out.push_str("\n\n");
}

/// Collapse runs of three or more newlines down to a paragraph break.
fn normalize_blank_lines(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut newlines = 0;
    for c in s.chars() {
        if c == '\n' {
            newlines += 1;
            if newlines <= 2 {
                out.push(c);
            }
        } else {
            newlines = 0;
            out.push(c);
        }
    }
    let trimmed = out.trim_end();
    format!("{trimmed}\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::latex_to_tree;

    fn md(src: &str) -> String {
        to_markdown(&latex_to_tree(src))
    }

    #[test]
    fn frontmatter_quotes_and_skips() {
        let fm = frontmatter(&[
            ("source", "2205.14135"),                // numeric-looking id stays a string
            ("title", "A Study: Part 1 \"quoted\""), // colon + quotes must be escaped
            ("empty", ""),                           // skipped
            ("texmark_truncated", "false"),          // bare boolean
        ]);
        assert!(fm.starts_with("---\n") && fm.ends_with("---\n\n"), "{fm}");
        assert!(fm.contains("source: \"2205.14135\"\n"), "{fm}");
        assert!(
            fm.contains("title: \"A Study: Part 1 \\\"quoted\\\"\"\n"),
            "{fm}"
        );
        assert!(!fm.contains("empty:"), "empty field must be skipped: {fm}");
        assert!(
            fm.contains("texmark_truncated: false\n"),
            "bool must be bare: {fm}"
        );
    }

    #[test]
    fn frontmatter_emits_structured_authors() {
        let authors = vec!["Ada Lovelace".to_string(), "Grace Hopper".to_string()];
        let frontmatter = frontmatter_with_authors(&[("title", "Paper")], &authors);
        assert!(frontmatter.contains("authors:\n  - \"Ada Lovelace\"\n"));
        assert!(frontmatter.contains("  - \"Grace Hopper\"\n"));
    }

    #[test]
    fn document_title_collapses_forced_breaks() {
        let tree = latex_to_tree(
            "\\title{FlashAttention-2: \\\\ Faster Attention}\\begin{document}x\\end{document}",
        );
        assert_eq!(
            title_text(&tree).as_deref(),
            Some("FlashAttention-2: Faster Attention")
        );
        assert!(
            to_markdown(&tree).starts_with("# FlashAttention-2: Faster Attention\n"),
            "{}",
            to_markdown(&tree)
        );
    }

    #[test]
    fn figure_caption_becomes_image_alt() {
        // The caption is always the image's alt text (Markdown's caption field).
        // By default the redundant visible line is suppressed (LLM-friendly);
        // duplicate_captions restores it (human viewing).
        let doc = "\\begin{document}\\begin{figure}\\includegraphics{fig.png}\
             \\caption{A cat: sitting}\\end{figure}\\end{document}";
        let tree = latex_to_tree(doc);

        let plain = to_markdown(&tree);
        assert!(
            plain.contains("![A cat: sitting](fig.png)"),
            "alt missing: {plain}"
        );
        assert!(
            !plain.contains("*A cat: sitting*"),
            "visible caption not suppressed: {plain}"
        );

        let dup = to_markdown_with(
            &tree,
            &MdOptions {
                duplicate_captions: true,
            },
        );
        assert!(
            dup.contains("![A cat: sitting](fig.png)"),
            "alt missing: {dup}"
        );
        // The visible caption now leads with its float number.
        assert!(
            dup.contains("**Figure 1:** A cat: sitting"),
            "numbered caption missing: {dup}"
        );
    }

    #[test]
    fn pdf_figures_are_links() {
        let doc = "\\begin{document}\\includegraphics{figure.pdf}\\end{document}";
        let tree = latex_to_tree(doc);
        assert!(to_markdown(&tree).contains("[Figure](figure.pdf)"));
    }

    #[test]
    fn caption_above_panels_pair_each_image_with_its_own_caption() {
        // Side-by-side panels with the caption ABOVE each image (one float, two
        // `\caption`s). Each image must take the caption from its OWN panel; the
        // earlier bug handed panel 1's image the *next* panel's caption (and left
        // panel 1's caption unmirrored, leaking a stray visible line).
        let out = md("\\begin{document}\\begin{figure}\
             \\caption{First.}\\includegraphics{a.png}\
             \\caption{Second.}\\includegraphics{b.png}\
             \\end{figure}\\end{document}");
        assert!(out.contains("![First.](a.png)"), "panel 1 alt wrong: {out}");
        assert!(
            out.contains("![Second.](b.png)"),
            "panel 2 alt wrong: {out}"
        );
        assert!(
            !out.contains("![Second.](a.png)"),
            "panel 1 image must not inherit panel 2's caption: {out}"
        );
    }

    #[test]
    fn table_caption_always_visible() {
        // A table caption isn't mirrored into any alt, so it renders even in the
        // default (non-duplicating) mode.
        let out = md(
            "\\begin{document}\\begin{table}\\begin{tabular}{l}a\\\\\\end{tabular}\
             \\caption{Results}\\end{table}\\end{document}",
        );
        assert!(
            out.contains("**Table 1:** Results"),
            "table caption should render: {out}"
        );
    }

    #[test]
    fn plain_includegraphics_has_empty_alt() {
        let out = md("\\begin{document}\\includegraphics{lone.png}\\end{document}");
        assert!(out.contains("![](lone.png)"), "{out}");
    }

    #[test]
    fn bibliography_gets_references_heading() {
        let out = md("\\begin{document}\\begin{thebibliography}{9}\
             \\bibitem{k} Some Author. A title. 2020.\\end{thebibliography}\\end{document}");
        assert!(
            out.contains("## References"),
            "references heading missing: {out}"
        );
        assert!(
            out.contains("- <a id=\"ref-k\"></a>[1]"),
            "reference should be numbered and anchored: {out}"
        );
    }

    #[test]
    fn font_declarations_wrap_rest_of_group() {
        // `\bf`/`\it` are declarations: they style the rest of the group, not
        // just the next token.
        let out =
            md("\\begin{document}Normal {\\bf bold words} and {\\it slanted} end.\\end{document}");
        assert!(out.contains("**bold words**"), "whole group bold: {out}");
        assert!(out.contains("*slanted*"), "whole group italic: {out}");
        assert!(out.contains("Normal") && out.contains("end."), "{out}");
    }

    #[test]
    fn headings_and_emphasis() {
        let out = md("\\begin{document}\\section{Intro}\nHello \\textbf{world}.\n\\end{document}");
        assert!(out.contains("## 1 Intro"), "{out}");
        assert!(out.contains("**world**"), "{out}");
    }

    #[test]
    fn inline_and_display_math() {
        let out = md("\\begin{document}$a+b$\n\n\\[ x^2 \\]\n\\end{document}");
        assert!(out.contains("$a+b$"), "{out}");
        assert!(out.contains("$$\nx^2\n$$"), "{out}");
    }

    #[test]
    fn unordered_list() {
        let out = md(
            "\\begin{document}\\begin{itemize}\\item one\\item two\\end{itemize}\\end{document}",
        );
        assert!(out.contains("- one"), "{out}");
        assert!(out.contains("- two"), "{out}");
    }

    #[test]
    fn table_has_header_separator() {
        let out = md(
            "\\begin{document}\\begin{tabular}{ll}a & b \\\\ c & d \\\\\\end{tabular}\\end{document}",
        );
        assert!(out.contains("| a | b |"), "{out}");
        assert!(out.contains("| --- | --- |"), "{out}");
    }

    #[test]
    fn inline_code_is_verbatim_not_escaped() {
        // Inside a code span, Markdown metacharacters must survive literally:
        // `\texttt{fused\_experts}` is `fused_experts`, not the escaped
        // `fused\_experts`. (The source writes `\_` for a literal underscore.)
        let out = md("\\begin{document}\\texttt{fused\\_experts}\\end{document}");
        assert!(
            out.contains("`fused_experts`"),
            "underscore escaped in code: {out}"
        );
        assert!(!out.contains("\\_"), "code span must not escape `_`: {out}");
        // Nested inline formatting can't live inside a GFM code span, so the
        // formatting is kept and the (inexpressible) monospace is dropped.
        let nested = md("\\begin{document}\\texttt{\\textbf{Hello} world}\\end{document}");
        assert!(
            nested.contains("**Hello** world"),
            "nested formatting lost: {nested}"
        );
        assert!(
            !nested.contains('`'),
            "formatted body must not be a code span: {nested}"
        );
    }

    #[test]
    fn adjacent_code_spans_merge() {
        // `\texttt{L1/}\allowbreak\texttt{x.py}` is contiguous monospace: one
        // code span `L1/x.py`, not two touching spans.
        let out = md("\\begin{document}\\texttt{L1/}\\allowbreak\\texttt{x.py}\\end{document}");
        assert!(
            out.contains("`L1/x.py`"),
            "adjacent code spans not merged: {out}"
        );
        // A real space between them keeps them as separate spans.
        let spaced = md("\\begin{document}\\texttt{a} \\texttt{b}\\end{document}");
        assert!(
            spaced.contains("`a` `b`"),
            "spaced code spans must stay separate: {spaced}"
        );
    }

    #[test]
    fn ding_renders_check_and_cross() {
        // `\ding{51}`/`\ding{55}` (pifont) — as used by `\cmark`/`\xmark` — are
        // a check and a cross, not the raw numbers.
        let out = md("\\begin{document}\\ding{51} \\ding{55}\\end{document}");
        assert!(
            out.contains('✓') && out.contains('✗'),
            "ding glyphs missing: {out}"
        );
        assert!(
            !out.contains("51") && !out.contains("55"),
            "ding code leaked: {out}"
        );
    }

    #[test]
    fn linebreak_inside_cell_does_not_split_row() {
        // A `\\` nested in a braced cell argument (here `\makecell{A\\B}`) is a
        // line break within the cell, not a row separator. It must not split the
        // row nor leave the cell unbalanced and swallow following content.
        let out = md("\\begin{document}\
             \\begin{tabular}{lll}\
             \\makecell{A\\\\B} & x & 1 \\\\\
             \\end{tabular}\
             \\section{Marker}\nAfter the table.\\end{document}");
        assert!(
            out.contains("## 1 Marker"),
            "section swallowed by table: {out}"
        );
        assert!(
            out.contains("After the table."),
            "paragraph swallowed by table: {out}"
        );
        // The intra-cell `\\` folds to `<br>` and the row stays on one physical
        // line with its trailing cells intact.
        assert!(
            out.contains("| A<br>B | x | 1 |"),
            "cell break not folded: {out}"
        );
    }

    #[test]
    fn footnotes_become_gfm_footnotes() {
        // \footnote/\thanks lower to a GFM footnote: an inline `[^n]` reference
        // plus a definition at the end — not the old inline parenthetical, which
        // misattached notes (e.g. an author's "equal contributor" mark).
        let out = md("\\begin{document}Body\\footnote{A note}. More.\\end{document}");
        assert!(out.contains("Body[^1]"), "inline marker missing: {out}");
        assert!(out.contains("[^1]: A note"), "definition missing: {out}");
        assert!(
            !out.contains("(A note)"),
            "footnote must not inline as a parenthetical: {out}"
        );
        // An empty footnote leaves no dangling marker or orphan definition.
        let empty = md("\\begin{document}Body\\footnote{}. More.\\end{document}");
        assert!(!empty.contains("[^"), "empty footnote must vanish: {empty}");
    }

    #[test]
    fn run_in_bold_moves_trailing_space_outside_delimiters() {
        // A trailing space inside `**…**` defeats CommonMark right-flanking, so
        // the run-in heading fails to render; move the space outside.
        let out = md("\\begin{document}\\textbf{Note: } Body.\\end{document}");
        assert!(
            out.contains("**Note:**"),
            "space must move outside bold: {out}"
        );
        assert!(
            !out.contains("**Note: **"),
            "space left inside delimiter: {out}"
        );
    }

    #[test]
    fn nested_same_emphasis_collapses_to_one_span() {
        // Emphasis nested in the same kind can't be expressed in GFM (it would
        // emit unbalanced `**inner* tail*`); collapse to one run.
        let out = md("\\begin{document}\\emph{outer \\emph{inner} tail}\\end{document}");
        assert!(out.contains("*outer inner tail*"), "should collapse: {out}");
        assert!(
            !out.contains("**inner"),
            "unbalanced markers emitted: {out}"
        );
        // Different kinds still nest (`***bold italic***`).
        let mix = md("\\begin{document}\\emph{a \\textbf{b} c}\\end{document}");
        assert!(
            mix.contains("*a **b** c*"),
            "bold-in-italic should nest: {mix}"
        );
    }

    #[test]
    fn heading_math_is_lowered_for_gfm() {
        // GitHub does not render `$…$` inside an ATX heading, so lower it to text.
        let out = md("\\begin{document}\\section{S}\\subsection{P$_{os}$: States}\\end{document}");
        assert!(
            out.contains("### 1.1 Pos: States"),
            "heading math not lowered: {out}"
        );
        assert!(!out.contains('$'), "math delimiter left in heading: {out}");
        // A literal `$` in a heading (from `\$`) must survive — lowering works on
        // the tree's `<math>` nodes, not by re-parsing the serialized string.
        let money = md("\\begin{document}\\section{Save \\$5 on \\$100 orders}\\end{document}");
        assert!(
            money.contains("## 1 Save $5 on $100 orders"),
            "literal $ in heading corrupted: {money}"
        );
    }

    #[test]
    fn appendix_switches_to_lettered_numbering() {
        let out = md(
            "\\begin{document}\\section{Body}Text.\\appendix\\section{Extra}More.\\end{document}",
        );
        assert!(out.contains("## 1 Body"), "section number missing: {out}");
        assert!(out.contains("## A Extra"), "appendix letter missing: {out}");
        assert!(
            !out.contains("## Appendix"),
            "invented heading remains: {out}"
        );
    }

    #[test]
    fn paragraph_heading_level_stays_contiguous() {
        // `\paragraph` under a `\subsection` must render one level deeper (H4),
        // not jump to H5 and skip H4.
        let out = md(
            "\\begin{document}\\section{S}\\subsection{Sub}\\paragraph{Par}Text.\\end{document}",
        );
        assert!(out.contains("#### Par"), "paragraph should be H4: {out}");
        assert!(
            !out.contains("##### Par"),
            "heading level skipped to H5: {out}"
        );
        // A `\part` (level -1) above sections must not collide with them at the h1
        // floor: part→H1, section→H2, subsection→H3 stay distinct.
        let part = md("\\begin{document}\\part{P}\\section{S}\\subsection{Sub}x\\end{document}");
        assert!(part.contains("# P\n"), "part should be H1: {part}");
        assert!(
            part.contains("## 1 S"),
            "section under part should be H2: {part}"
        );
        assert!(
            part.contains("### 1.1 Sub"),
            "subsection should be H3: {part}"
        );
    }

    #[test]
    fn multicolumn_after_leading_hline_pads_spanned_columns() {
        // A leading `\hline` on the header row must not hide the `\multicolumn`
        // from span-padding, or the group labels drift left instead of aligning
        // over their spanned columns.
        let out = md("\\begin{document}\\begin{tabular}{|l|l|l|l|}\\hline \
             \\multicolumn{2}{|c|}{A} & \\multicolumn{2}{c|}{B} \\\\ \\hline \
             1 & 2 & 3 & 4 \\\\ \\hline\\end{tabular}\\end{document}");
        assert!(
            out.contains("| A |  | B |  |"),
            "spanned header columns not padded: {out}"
        );
    }

    #[test]
    fn centered_and_captionof_figures_get_alt_text() {
        // `\begin{center}` wraps the body in an `<align>`; the caption must still
        // reach the image as alt text.
        let center = md("\\begin{document}\\begin{figure}\\begin{center}\
             \\includegraphics{a.png}\\caption{Cap A}\\end{center}\\end{figure}\\end{document}");
        assert!(
            center.contains("![Cap A](a.png)"),
            "centered alt missing: {center}"
        );

        // `\captionof` puts its caption in a caption-only nested float; the alt
        // must still reach the sibling image.
        let capof = md(
            "\\documentclass{article}\\usepackage{caption}\\begin{document}\
             \\begin{figure}\\includegraphics{b.png}\\captionof{figure}{Cap B}\\end{figure}\\end{document}",
        );
        assert!(
            capof.contains("![Cap B](b.png)"),
            "captionof alt missing: {capof}"
        );
    }

    #[test]
    fn orphan_period_between_caption_and_label_is_dropped() {
        // A stray `.` the source leaves between `\captionof{…}` and `\label`
        // must not surface as its own fragment; the label anchor survives.
        let out = md(
            "\\documentclass{article}\\usepackage{caption}\\begin{document}\
             \\begin{figure}\\includegraphics{a.png}\\captionof{figure}{Cap}. \\label{f}\
             \\end{figure}\\end{document}",
        );
        assert!(!out.contains(". <a"), "stray period not dropped: {out}");
        assert!(out.contains("<a id=\"f\">"), "label anchor lost: {out}");
        // A standalone punctuation paragraph with NO label is real prose (e.g. an
        // ellipsis or a lone question mark) and must be kept.
        let prose = md("\\begin{document}First.\n\n?\n\nSecond.\\end{document}");
        assert!(
            prose.contains('?'),
            "standalone punctuation prose dropped: {prose}"
        );
    }
}
