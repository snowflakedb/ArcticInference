//! Output backends: serializers from the format-neutral document tree to a
//! concrete output format.
//!
//! Each backend is a submodule ([`md`], [`xml`], [`html`]) implementing the
//! [`Backend`] trait. This `mod.rs` owns the trait and the behavior the backends
//! share — the small tree-walk helpers — so adding a format is implementing
//! `Backend` and reusing these, without duplicating traversal logic or touching
//! the engine/resolve pass. KaTeX math post-processing shared by the browser-math
//! backends (md, html) lives in the private `texmath` submodule.

use crate::node::{Element, Node};

pub mod html;
pub mod md;
mod texmath;
pub mod xml;

/// A serializer from the neutral tree to a concrete output format. The tree
/// carries no format-specific data, so a new format is a new `Backend` impl —
/// see [`md::Markdown`], [`xml::Xml`], [`html::Html`].
pub trait Backend {
    fn serialize(&self, root: &Element) -> String;
}

// --- shared tree helpers ----------------------------------------------------

/// The value of attribute `key` on `e`, if present.
pub(crate) fn attr<'a>(e: &'a Element, key: &str) -> Option<&'a String> {
    e.attributes.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

/// Concatenate an element's descendant text, dropping element wrappers.
pub(crate) fn text_content(e: &Element) -> String {
    node_text(&e.children)
}

/// Concatenate a node list's text, dropping element wrappers.
pub(crate) fn node_text(nodes: &[Node]) -> String {
    let mut out = String::new();
    for node in nodes {
        match node {
            Node::Text(t) => out.push_str(t),
            Node::Element(e) => out.push_str(&node_text(&e.children)),
        }
    }
    out
}

/// Uppercase the first character of `s`.
pub(crate) fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// Turn a `\label` key into a fragment identifier safe for an HTML `id`/URL
/// fragment, applied identically to both the emitted anchor and every link to
/// it so they still match. Alphanumerics (Unicode included — GFM permits them
/// in fragments) and `:._-` (already valid, and used by the common
/// `sec:intro`/`eq:x` conventions) pass through; any other run — notably spaces,
/// which break links on GitHub and strict renderers — collapses to a single `-`.
/// Edge `-`s are trimmed. Empty input maps to `_`.
///
/// Distinct labels differing only in a slugged run (`a b` vs `a-b`) can collapse
/// to the same fragment; such collisions are rare in practice and no worse than
/// the previously-broken spaced anchors.
pub(crate) fn slug(id: &str) -> String {
    let mut out = String::with_capacity(id.len());
    let mut pending_dash = false;
    for c in id.chars() {
        if c.is_alphanumeric() || matches!(c, ':' | '.' | '_' | '-') {
            if pending_dash && !out.is_empty() {
                out.push('-');
            }
            pending_dash = false;
            out.push(c);
        } else {
            pending_dash = true;
        }
    }
    if out.is_empty() { "_".to_string() } else { out }
}

/// Join items as an English list: "A", "A and B", "A, B and C".
pub(crate) fn join_and(items: &[&str]) -> String {
    match items {
        [] => String::new(),
        [a] => a.to_string(),
        [a, b] => format!("{a} and {b}"),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// Apply LaTeX text-mode typographic ligatures: `---`→—, `--`→–, and the quote
/// forms `` `` ``→“, `''`→”, `` ` ``→‘, `'`→’. Backends call this on PROSE text
/// only — never on code/verbatim/math, where these are literal. (XML keeps the
/// source characters.)
pub(crate) fn typographic(s: &str) -> String {
    s.replace("---", "\u{2014}") // em dash
        .replace("--", "\u{2013}") // en dash
        .replace("``", "\u{201C}") // left double quote
        .replace("''", "\u{201D}") // right double quote
        .replace('`', "\u{2018}") // left single quote
        .replace('\'', "\u{2019}") // right single quote / apostrophe
}

/// Escape `&`, `<`, `>` for XML/HTML text content.
pub(crate) fn escape_text(s: &str) -> String {
    escape_chars(s, false)
}

/// Escape `&`, `<`, `>`, and `"` — for attribute values, and for HTML text
/// (where escaping the quote too is harmless and keeps one code path).
pub(crate) fn escape_attr(s: &str) -> String {
    escape_chars(s, true)
}

fn escape_chars(s: &str, quote: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' if quote => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

/// A table element's `<row>` children, in order.
pub(crate) fn table_rows(e: &Element) -> Vec<&Element> {
    e.children
        .iter()
        .filter_map(|n| match n {
            Node::Element(r) if r.name == "row" => Some(r),
            _ => None,
        })
        .collect()
}

/// The heading level at which to render a `<section>`: its actual nesting
/// `depth` (set by the engine's section folding so levels stay contiguous — a
/// `\paragraph` under a `\subsection` renders one step deeper, never skipping a
/// level), falling back to the absolute `level` for trees without `depth`.
pub(crate) fn section_display_level(e: &Element) -> i32 {
    attr(e, "depth")
        .or_else(|| attr(e, "level"))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

/// Lift the first child element named `name` out of `e`, returning it alongside
/// the remaining children (cloned). Backends use this to separate a `<title>`
/// (section heading) or `<term>` (theorem/proof lead) from the body before
/// rendering each in its own format.
pub(crate) fn split_off_child<'a>(e: &'a Element, name: &str) -> (Option<&'a Element>, Vec<Node>) {
    let mut found = None;
    let mut body = Vec::new();
    for child in &e.children {
        match child {
            Node::Element(c) if c.name == name => {
                if found.is_none() {
                    found = Some(c);
                }
            }
            other => body.push(other.clone()),
        }
    }
    (found, body)
}
