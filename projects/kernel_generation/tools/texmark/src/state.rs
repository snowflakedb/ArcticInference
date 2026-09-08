//! Persistent engine state and the file resolver.
//!
//! [`State`] holds everything that survives across the token stream: the
//! category codes and the user's macro and environment definitions. The
//! [`Resolver`] lets `\input`/`\include` pull in other files without the core
//! ever touching the filesystem — the CLI supplies a real one, WASM callers a
//! virtual one.

use crate::token::{CatCodeTable, Token};
use std::collections::HashMap;

/// A user-defined macro (`\def`, `\newcommand`, ...).
///
/// Parameters are undelimited `#1`..`#9`; `optional_default`, when present,
/// marks the first parameter as a bracketed optional argument (LaTeX's
/// `\newcommand[n][default]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Macro {
    pub params: usize,
    pub optional_default: Option<Vec<Token>>,
    pub body: Vec<Token>,
}

/// A user-defined environment (`\newenvironment`).
///
/// The `begin` tokens are inserted where `\begin{name}` appears and `end`
/// tokens where `\end{name}` appears; the body in between is processed normally.
#[derive(Debug, Clone)]
pub struct Environment {
    pub params: usize,
    pub optional_default: Option<Vec<Token>>,
    pub begin: Vec<Token>,
    pub end: Vec<Token>,
}

/// A theorem-like environment declared by `\newtheorem`.
///
/// The `printed` name is what LaTeX typesets ("Theorem", "Lemma", ...). The
/// `counter` names the counter the environment steps — several environments can
/// share one (`\newtheorem{lemma}[theorem]{Lemma}`), so they number in a single
/// sequence. `None` marks a starred (unnumbered) declaration.
#[derive(Debug, Clone)]
pub struct TheoremDef {
    pub printed: String,
    pub counter: Option<String>,
    /// The `[within]` counter (`\newtheorem{thm}{Thm}[section]`): theorem
    /// numbers are prefixed by that section/chapter number and reset when it
    /// steps ("Theorem 2.1"). `None` = flat, document-wide numbering.
    pub within: Option<String>,
}

/// Section-numbering configuration a document class supplies. `top_level` is the
/// `section_level` of the class's top numbered unit — 1 (`\section`) for article,
/// 0 (`\chapter`) for book/report — and `secnumdepth` is the deepest level still
/// numbered. Dotted numbers count from `top_level` (so book chapters are "1" and
/// their sections "1.1", while article sections are "1" and subsections "1.1").
#[derive(Debug, Clone, Copy)]
pub struct Numbering {
    pub top_level: i32,
    pub secnumdepth: i32,
}

impl Default for Numbering {
    /// The `article` defaults: number `\section`..`\subsubsection`.
    fn default() -> Self {
        Numbering {
            top_level: 1,
            secnumdepth: 3,
        }
    }
}

/// Everything that persists as the engine walks the token stream.
pub struct State {
    pub catcodes: CatCodeTable,
    pub macros: HashMap<String, Macro>,
    pub environments: HashMap<String, Environment>,
    /// `etoolbox` boolean toggles (`\newtoggle`, `\iftoggle`, ...).
    pub toggles: HashMap<String, bool>,
    /// LaTeX counters (`\newcounter`, `\stepcounter`, `\value`, `\arabic`, ...).
    pub counters: HashMap<String, i32>,
    /// Theorem-like environments declared by `\newtheorem`, keyed by environment
    /// name; consumed by the resolve pass to number and label them.
    pub theorems: HashMap<String, TheoremDef>,
    /// A scratchpad for native packages to persist their own state (they are
    /// stateless zero-sized structs). Keys are package-namespaced, e.g.
    /// `natbib` stores its citation mode here. The core does not interpret it.
    pub package_state: HashMap<String, String>,
    /// `\let\alias=\target` bindings whose target is NOT a user macro (a native
    /// or unknown control sequence). Keyed by alias name → target name, both
    /// without the leading `\`. Kept separate from [`macros`](Self::macros) so an
    /// alias is *not* an expandable macro: it dispatches to its target's meaning
    /// directly, which both matches TeX's snapshot semantics and avoids the
    /// infinite loop a later `\def\target{...\alias...}` would otherwise create
    /// (`\alias`→`\target`→`\alias`→…). See the `"let"` handler.
    pub let_aliases: HashMap<String, String>,
    /// Section-numbering config, set by `\documentclass`. Defaults to `article`.
    pub numbering: Numbering,
}

impl State {
    /// A fresh state with default LaTeX catcodes and no user definitions.
    pub fn new() -> Self {
        State {
            catcodes: CatCodeTable::default(),
            macros: HashMap::new(),
            environments: HashMap::new(),
            toggles: HashMap::new(),
            counters: HashMap::new(),
            theorems: HashMap::new(),
            package_state: HashMap::new(),
            let_aliases: HashMap::new(),
            numbering: Numbering::default(),
        }
    }
}

impl Default for State {
    fn default() -> Self {
        State::new()
    }
}

/// Supplies the source text of files pulled in by `\input` and `\include`.
///
/// Implementors keep all filesystem (or virtual-filesystem) access out of the
/// engine core. The `name` is the argument as written in the document, e.g.
/// `sections/intro`; the resolver is responsible for extension guessing and
/// path resolution.
pub trait Resolver {
    /// Return the source of the named file, or `None` if it cannot be found.
    fn resolve(&self, name: &str) -> Option<String>;

    /// Return the formatted bibliography source (the BibTeX-generated `.bbl`
    /// file), or `None` if there is none. Called for `\bibliography`; the `.bbl`
    /// is named after the top-level job, not after the `.bib` database, so the
    /// resolver — which owns filesystem knowledge — locates it.
    fn resolve_bibliography(&self) -> Option<String> {
        None
    }
}

/// A resolver that finds nothing — the default when no files should be pulled
/// in (e.g. a single self-contained document).
pub struct NoFiles;

impl Resolver for NoFiles {
    fn resolve(&self, _name: &str) -> Option<String> {
        None
    }
}
