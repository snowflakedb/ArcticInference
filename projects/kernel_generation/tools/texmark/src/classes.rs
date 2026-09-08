//! Native document classes.
//!
//! A document class is dispatched by name from `\documentclass{...}` exactly the
//! way a native package is dispatched from `\usepackage` (a compiled-in struct in
//! `CLASSES`, selected by name — nothing is loaded from disk). Its job here is
//! narrow: supply the section-[`Numbering`] configuration the resolve pass needs,
//! so numbering is a class decision (article's `\section` is "1", book's
//! `\chapter` is "1" and its `\section` "1.1") rather than hardcoded in the core.
//!
//! Classes texmark does not model fall back to the `article` default, which is
//! the right guess for the many `article`-derived paper classes.

use crate::state::Numbering;

/// A native document class: it names the `.cls` files it emulates and supplies
/// their numbering scheme. (Kept deliberately small; other class responsibilities
/// — title-block layout, `\abstract` — can be added as methods later.)
pub(crate) trait DocumentClass: Sync {
    /// The class names this handles (the argument of `\documentclass`).
    fn names(&self) -> &[&str];
    /// The section-numbering scheme for this class.
    fn numbering(&self) -> Numbering;
}

/// The standard `article` family: top numbered unit is `\section`, numbered
/// down to `\subsubsection`.
struct Article;
impl DocumentClass for Article {
    fn names(&self) -> &[&str] {
        &[
            "article",
            "proc",
            "minimal",
            "letter",
            "slides",
            "extarticle",
        ]
    }
    fn numbering(&self) -> Numbering {
        Numbering {
            top_level: 1,
            secnumdepth: 3,
        }
    }
}

/// The `book`/`report` family: top numbered unit is `\chapter`, numbered down to
/// `\subsection`, so sections read "1.1" within their chapter.
struct Book;
impl DocumentClass for Book {
    fn names(&self) -> &[&str] {
        &["book", "report", "memoir", "extbook", "extreport"]
    }
    fn numbering(&self) -> Numbering {
        Numbering {
            top_level: 0,
            secnumdepth: 2,
        }
    }
}

/// All registered native classes. Add one line here to add a class.
static CLASSES: &[&dyn DocumentClass] = &[&Article, &Book];

/// The class handling `name`, if any (first match wins).
pub(crate) fn for_class(name: &str) -> Option<&'static dyn DocumentClass> {
    CLASSES.iter().copied().find(|c| c.names().contains(&name))
}
