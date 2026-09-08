//! texmark — a pure-Rust engine converting LaTeX projects to XML (and onward to
//! Markdown).
//!
//! The pipeline is deliberately small and mirrors TeX's own architecture,
//! reimplemented from scratch (LaTeXML is a design reference, not a source of
//! ported code):
//!
//! 1. [`tokenizer`] — tokenize characters into [`token::Token`]s using category
//!    codes.
//! 2. [`engine`] — expand macros and execute constructs into a document tree.
//! 3. [`backend`] — serialize the tree to Markdown/XML/HTML.
//!
//! The core carries no file I/O so it can run under WASM; the CLI supplies
//! source text and a [`state::Resolver`] and writes the output.

pub mod backend;
pub mod bibtex;
pub mod classes;
pub mod engine;
pub mod image;
pub mod node;
pub mod packages;
pub mod resolve;
pub mod state;
pub mod token;
pub mod tokenizer;

use engine::Engine;
use node::Element;
use state::{NoFiles, Resolver, State};

/// The serialization contract every output format implements; re-exported from
/// [`backend`] so callers can write `texmark::Backend`.
pub use backend::Backend;

pub use engine::Diagnostics;

/// The texmark library version, for provenance in generated output.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Convert a self-contained LaTeX source string into the document tree.
///
/// Convenience wrapper that discards [`Diagnostics`]; call [`Engine::parse`]
/// directly when you need to know whether the conversion was degraded.
pub fn latex_to_tree(source: &str) -> Element {
    let mut state = State::new();
    Engine::new(source, &mut state, &NoFiles).parse().0
}

/// Convert LaTeX source into the document tree, using `resolver` for
/// `\input`/`\include`.
pub fn latex_to_tree_with(source: &str, resolver: &dyn Resolver) -> Element {
    let mut state = State::new();
    Engine::new(source, &mut state, resolver).parse().0
}

/// Convert a self-contained LaTeX source string into an XML document.
pub fn latex_to_xml(source: &str) -> String {
    backend::xml::Xml.serialize(&latex_to_tree(source))
}

/// Convert a self-contained LaTeX source string into GitHub Flavored Markdown.
pub fn latex_to_markdown(source: &str) -> String {
    backend::md::Markdown::default().serialize(&latex_to_tree(source))
}

/// Convert a self-contained LaTeX source string into HTML.
pub fn latex_to_html(source: &str) -> String {
    backend::html::Html.serialize(&latex_to_tree(source))
}
