//! Serializing the document tree to HTML.
//!
//! This backend exists to demonstrate — and enforce — that the document tree is
//! format-neutral: it reads exactly the same semantic elements and attributes
//! (`resolved`, `anchor`, `prefix`, `number`, `env`, `labels`, ...) that the
//! Markdown and XML backends do, and needs no changes to the engine or resolve
//! pass. Math targets a KaTeX auto-render script (delimiters `\(…\)` / `\[…\]`),
//! sharing the post-processing in the `texmath` submodule with the Markdown backend.

use super::{
    attr, capitalize, escape_attr as escape, join_and, slug, split_off_child, table_rows,
    text_content,
};
use crate::node::{Element, Node};

/// The HTML backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct Html;

impl super::Backend for Html {
    fn serialize(&self, root: &Element) -> String {
        to_html(root)
    }
}

/// Render a `<document>` element as an HTML fragment.
pub fn to_html(root: &Element) -> String {
    let mut out = String::new();
    // authblk keeps one `<author>` node per author; render the list on one line.
    let authors: Vec<String> = root
        .children
        .iter()
        .filter_map(|n| match n {
            Node::Element(e) if e.name == "author" => {
                let t = inline(&e.children);
                t.chars()
                    .any(|c| c.is_alphanumeric())
                    .then(|| t.trim().to_string())
            }
            _ => None,
        })
        .collect();
    let mut authors_done = false;
    for child in &root.children {
        match child {
            Node::Element(e) if e.name == "title" => {
                out.push_str(&format!("<h1>{}</h1>\n", inline(&e.children)));
            }
            Node::Element(e) if e.name == "author" => {
                if !authors_done && !authors.is_empty() {
                    let names: Vec<&str> = authors.iter().map(String::as_str).collect();
                    out.push_str(&format!("<p class=\"author\">{}</p>\n", join_and(&names)));
                }
                authors_done = true;
            }
            Node::Element(e) if e.name == "date" => {
                out.push_str(&format!(
                    "<p class=\"date\">{}</p>\n",
                    inline(&e.children).trim()
                ));
            }
            Node::Element(e) => block_element(e, &mut out),
            Node::Text(t) => push_para(&mut out, &escape(&super::typographic(t))),
        }
    }
    out
}

// --- block level ------------------------------------------------------------

fn blocks(nodes: &[Node], out: &mut String) {
    for node in nodes {
        match node {
            Node::Element(e) => block_element(e, out),
            Node::Text(t) if !t.trim().is_empty() => {
                push_para(out, &escape(&super::typographic(t)))
            }
            Node::Text(_) => {}
        }
    }
}

fn block_element(e: &Element, out: &mut String) {
    match e.name.as_str() {
        "section" => {
            let level = super::section_display_level(e);
            let h = (level + 1).clamp(1, 6);
            let (title, body) = split_off_child(e, "title");
            if let Some(t) = title {
                out.push_str(&format!("<h{h}>{}</h{h}>\n", inline(&t.children).trim()));
            }
            blocks(&body, out);
        }
        "p" => push_para(out, &inline(&e.children)),
        "abstract" => {
            out.push_str("<h2>Abstract</h2>\n");
            blocks(&e.children, out);
        }
        "itemize" | "description" => render_list(e, out, "ul"),
        "enumerate" => render_list(e, out, "ol"),
        "bibliography" => {
            out.push_str("<h2>References</h2>\n");
            render_list(e, out, "ul");
        }
        "blockquote" => {
            out.push_str("<blockquote>\n");
            blocks(&e.children, out);
            out.push_str("</blockquote>\n");
        }
        "verbatim" => {
            let lang = attr(e, "language")
                .map(|l| format!(" class=\"language-{l}\""))
                .unwrap_or_default();
            out.push_str(&format!(
                "<pre><code{lang}>{}</code></pre>\n",
                escape(&text_content(e))
            ));
        }
        "math" => {
            if let Some(ids) = attr(e, "labels") {
                for id in ids.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    out.push_str(&format!("<a id=\"{}\"></a>", slug(id)));
                }
            }
            out.push_str(&format!("<p>{}</p>\n", math_delimited(e)));
        }
        "tabular" => render_table(e, out),
        "float" => {
            out.push_str("<figure>\n");
            blocks(&e.children, out);
            out.push_str("</figure>\n");
        }
        "caption" => {
            let prefix = attr(e, "prefix")
                .map(|p| format!("<strong>{}:</strong> ", escape(p)))
                .unwrap_or_default();
            out.push_str(&format!(
                "<figcaption>{prefix}{}</figcaption>\n",
                inline(&e.children).trim()
            ));
        }
        "image" => push_para(out, &render_image(e)),
        "label" => {
            if let Some(id) = attr(e, "id") {
                out.push_str(&format!("<a id=\"{}\"></a>\n", slug(id)));
            }
        }
        "environment" => {
            let (term, body) = split_off_child(e, "term");
            let term_lead = term.map(|t| inline(&t.children).trim().to_string());
            // Theorem note (a `<term>`) shows in parens after the number; a
            // proof's `<term>` is a standalone lead with no prefix.
            let lead = match (attr(e, "prefix"), &term_lead) {
                (Some(p), Some(t)) => Some(format!("{} ({t})", escape(p))),
                (Some(p), None) => Some(escape(p)),
                (None, Some(t)) => Some(t.clone()),
                (None, None) => attr(e, "name").map(|s| capitalize(s)),
            };
            if let Some(lead) = lead {
                out.push_str(&format!("<p><strong>{lead}.</strong></p>\n"));
            }
            blocks(&body, out);
        }
        // Transparent wrappers (align/center and anything unknown).
        _ => blocks(&e.children, out),
    }
}

fn render_list(e: &Element, out: &mut String, tag: &str) {
    out.push_str(&format!("<{tag}>\n"));
    for child in &e.children {
        let Node::Element(item) = child else { continue };
        let mut lead = String::new();
        let mut body: Vec<Node> = Vec::new();
        for gc in &item.children {
            match gc {
                Node::Element(t) if t.name == "term" => {
                    lead = format!("<strong>{}</strong> ", inline(&t.children).trim());
                }
                other => body.push(other.clone()),
            }
        }
        if let Some(number) = attr(item, "number") {
            lead = format!("[{}] ", escape(number));
        } else if attr(item, "unlabeled").is_some() {
            lead = String::new();
        } else if let Some(key) = attr(item, "key") {
            lead = format!("[{}] ", escape(key));
        }
        // A reference entry emits its link anchor (`ref-<key>`) so in-text cites
        // resolve, mirroring the `<label>` anchor mechanism.
        if let Some(anchor) = attr(item, "anchor") {
            lead = format!("<a id=\"{}\"></a>{lead}", slug(anchor));
        }
        let mut inner = String::new();
        // Inline-only item content renders without a <p> wrapper; items with
        // block content (nested lists, paragraphs) go through the block path.
        if body.iter().any(is_block_node) {
            blocks(&body, &mut inner);
        } else {
            inner = inline(&body);
        }
        out.push_str(&format!("<li>{lead}{}</li>\n", inner.trim()));
    }
    out.push_str(&format!("</{tag}>\n"));
}

fn render_table(e: &Element, out: &mut String) {
    let rows = table_rows(e);
    if rows.is_empty() {
        return;
    }
    out.push_str("<table>\n");
    for (i, row) in rows.iter().enumerate() {
        let cell_tag = if i == 0 { "th" } else { "td" };
        out.push_str("<tr>");
        for cell in &row.children {
            if let Node::Element(c) = cell
                && c.name == "cell"
            {
                out.push_str(&format!(
                    "<{cell_tag}>{}</{cell_tag}>",
                    inline(&c.children).trim()
                ));
            }
        }
        out.push_str("</tr>\n");
    }
    out.push_str("</table>\n");
}

fn render_image(e: &Element) -> String {
    let src = attr(e, "src").map(|s| escape(s)).unwrap_or_default();
    let alt = attr(e, "alt").map(|s| escape(s)).unwrap_or_default();
    format!("<img src=\"{src}\" alt=\"{alt}\">")
}

// --- inline level -----------------------------------------------------------

fn inline(nodes: &[Node]) -> String {
    let mut out = String::new();
    for node in nodes {
        match node {
            Node::Text(t) => out.push_str(&escape(&super::typographic(t))),
            Node::Element(e) => out.push_str(&inline_element(e)),
        }
    }
    out
}

fn inline_element(e: &Element) -> String {
    let inner = inline(&e.children);
    match e.name.as_str() {
        "bold" => format!("<strong>{inner}</strong>"),
        "italic" | "emph" => format!("<em>{inner}</em>"),
        "code" => format!("<code>{}</code>", escape(&text_content(e))),
        "smallcaps" => format!("<span style=\"font-variant:small-caps\">{inner}</span>"),
        "underline" => format!("<u>{inner}</u>"),
        "superscript" => format!("<sup>{inner}</sup>"),
        "subscript" => format!("<sub>{inner}</sub>"),
        "linebreak" => "<br>".to_string(),
        "math" => math_delimited(e),
        "link" => {
            let href = attr(e, "href").map(|s| escape(s)).unwrap_or_default();
            let text = if inner.trim().is_empty() {
                href.clone()
            } else {
                inner
            };
            format!("<a href=\"{href}\">{text}</a>")
        }
        "ref" => {
            let raw = attr(e, "target").map(String::as_str).unwrap_or_default();
            match attr(e, "resolved") {
                Some(text) => {
                    let anchor = attr(e, "anchor").map(String::as_str).unwrap_or(raw);
                    format!("<a href=\"#{}\">{}</a>", slug(anchor), escape(text))
                }
                None => format!("<a href=\"#{}\">{}</a>", slug(raw), escape(raw)),
            }
        }
        "cite" => match attr(e, "resolved") {
            Some(text) => link_cite(text, attr(e, "anchors")),
            None => format!(
                "[{}]",
                attr(e, "keys").map(|s| escape(s)).unwrap_or_default()
            ),
        },
        "image" => render_image(e),
        "footnote" => format!(" ({})", inner.trim()),
        "label" => attr(e, "id")
            .map(|id| format!("<a id=\"{}\"></a>", slug(id)))
            .unwrap_or_default(),
        _ => inner,
    }
}

/// Turn a resolved numeric citation marker (`[1, 2]`) into HTML whose numbers
/// link to their reference entries, using the per-number `anchors` list (aligned
/// with the numbers in order). Each `N` becomes `<a href="#ref-key">N</a>`.
/// Without an anchor list (author-year, or any non-numeric marker) the text is
/// returned escaped and unchanged, matching the `<label>`/`\ref` link mechanism.
fn link_cite(text: &str, anchors: Option<&String>) -> String {
    let Some(anchors) = anchors else {
        return escape(text);
    };
    let Some(inner) = text.strip_prefix('[').and_then(|s| s.strip_suffix(']')) else {
        return escape(text);
    };
    let anchors: Vec<&str> = anchors.split(',').filter(|s| !s.is_empty()).collect();
    let numbers: Vec<&str> = inner.split(", ").collect();
    if numbers.len() != anchors.len() {
        return escape(text);
    }
    let linked: Vec<String> = numbers
        .iter()
        .zip(anchors.iter())
        .map(|(num, anchor)| format!("<a href=\"#{}\">{}</a>", slug(anchor), escape(num)))
        .collect();
    format!("[{}]", linked.join(", "))
}

/// A `<math>` node as delimited, HTML-safe TeX. The body's `<`/`>`/`&` are
/// escaped so the HTML parses (a browser decodes them back to text before a
/// KaTeX auto-render script reads them); the `\(…\)` / `\[…\]` delimiters are
/// literal (they contain no HTML metacharacters).
fn math_delimited(e: &Element) -> String {
    let body = escape(&super::texmath::katex(
        &text_content(e),
        attr(e, "env").map(String::as_str),
    ));
    if attr(e, "mode").map(String::as_str) == Some("display") {
        format!("\\[{body}\\]")
    } else {
        format!("\\({body}\\)")
    }
}

/// Whether a node is block-level (so a list item needs the block render path
/// rather than an inline run).
fn is_block_node(node: &Node) -> bool {
    matches!(node, Node::Element(e) if matches!(
        e.name.as_str(),
        "itemize" | "enumerate" | "description" | "bibliography" | "blockquote"
            | "verbatim" | "tabular" | "float" | "section" | "environment" | "abstract"
            | "image"
    ) || (e.name == "math" && attr(e, "mode").map(String::as_str) == Some("display")))
}

// --- helpers ----------------------------------------------------------------

fn push_para(out: &mut String, html: &str) {
    let trimmed = html.trim();
    if !trimmed.is_empty() {
        out.push_str(&format!("<p>{trimmed}</p>\n"));
    }
}

#[cfg(test)]
mod tests {
    use crate::latex_to_html;

    #[test]
    fn renders_basic_html_from_neutral_tree() {
        let src = r#"\begin{document}
\section{Intro}\label{sec:i}
Hello \textbf{world}, see \cref{sec:i}. Math $a+b$.
\begin{itemize}\item one\end{itemize}
\end{document}"#;
        let html = latex_to_html(src);
        assert!(html.contains("<h2>Intro</h2>"), "heading: {html}");
        assert!(html.contains("<strong>world</strong>"), "bold: {html}");
        assert!(html.contains("<a id=\"sec:i\">"), "anchor: {html}");
        assert!(
            html.contains("<a href=\"#sec:i\">Section 1</a>"),
            "cross-ref link: {html}"
        );
        assert!(html.contains("\\(a+b\\)"), "inline math: {html}");
        assert!(html.contains("<li>one</li>"), "list: {html}");
    }
}
