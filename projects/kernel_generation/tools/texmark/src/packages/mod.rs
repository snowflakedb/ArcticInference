//! Native package implementations.
//!
//! Each submodule emulates a LaTeX package as a zero-sized struct implementing
//! `Package`. A package claims control sequences and environments by name and,
//! when one fires, *runs inside the engine* — reading its own arguments, splicing
//! tokens, reaching the resolver — just as a real `.sty` does. It may also hook
//! the end of the document via `Package::finalize` (LaTeX's `\AtEndDocument`).
//!
//! The engine dispatches through `REGISTRY` and never names a specific package:
//! adding support for a package means adding a submodule and one line in
//! `REGISTRY`.

use crate::engine::{Engine, Event};
use crate::node::Element;

pub mod algorithmic;
pub mod caption;
pub mod cleveref;
pub mod comment;
pub mod doi;
pub mod enumitem;
pub mod fancyhdr;
pub mod float;
pub mod graphicx;
pub mod hyperref;
pub mod ifthen;
pub mod listings;
pub mod natbib;
pub mod nicefrac;

/// A native package: a bundle of command/environment handlers (and an optional
/// end-of-document hook) that plug into the engine.
///
/// All methods default to no-ops, so a package implements only what it provides.
pub(crate) trait Package: Sync {
    /// Control-sequence names this package handles (without the leading `\`).
    fn commands(&self) -> &[&str] {
        &[]
    }

    /// Environment names this package handles.
    fn environments(&self) -> &[&str] {
        &[]
    }

    /// Run one of this package's commands. The package reads its own arguments
    /// from `e` and returns the resulting event.
    fn command(&self, name: &str, e: &mut Engine) -> Event {
        let _ = (name, e);
        Event::Inline(vec![])
    }

    /// Run one of this package's environments (called just after `\begin{name}`).
    fn environment(&self, name: &str, e: &mut Engine) -> Event {
        let _ = (name, e);
        Event::Blocks(vec![])
    }

    /// The `.sty` name this package emulates (without extension), if it wants to
    /// react to being loaded. `\usepackage[opts]{name}` then calls [`configure`].
    ///
    /// [`configure`]: Package::configure
    fn package_names(&self) -> &[&str] {
        &[]
    }

    /// React to `\usepackage[options]{name}` (or `\PassOptionsToPackage`) for the
    /// package named by [`package_names`]. Lets a package record its own load
    /// options in [`State`] instead of the engine hardcoding package specifics.
    ///
    /// [`package_names`]: Package::package_names
    fn configure(&self, options: &str, state: &mut crate::state::State) {
        let _ = (options, state);
    }

    /// Adjust the finished document tree — the equivalent of `\AtEndDocument`.
    /// Runs after the full parse, with the engine still available for rendering.
    fn finalize(&self, tree: &mut Element, e: &mut Engine) {
        let _ = (tree, e);
    }
}

/// All registered native packages. Add one line here when adding a new package.
pub(crate) static REGISTRY: &[&dyn Package] = &[
    &algorithmic::Algorithmic,
    &caption::Caption,
    &cleveref::Cleveref,
    &comment::Comment,
    &doi::Doi,
    &enumitem::Enumitem,
    &fancyhdr::Fancyhdr,
    &float::Float,
    &graphicx::Graphicx,
    &hyperref::Hyperref,
    &ifthen::Ifthen,
    &listings::Listings,
    &natbib::Natbib,
    &nicefrac::Nicefrac,
];

/// The package that claims `name` as a command, if any (first match wins).
pub(crate) fn for_command(name: &str) -> Option<&'static dyn Package> {
    REGISTRY
        .iter()
        .copied()
        .find(|p| p.commands().contains(&name))
}

/// The package that claims `name` as an environment, if any (first match wins).
pub(crate) fn for_environment(name: &str) -> Option<&'static dyn Package> {
    REGISTRY
        .iter()
        .copied()
        .find(|p| p.environments().contains(&name))
}

/// The package emulating the `.sty` named `name`, if any — for reacting to
/// `\usepackage`/`\PassOptionsToPackage`.
pub(crate) fn for_package(name: &str) -> Option<&'static dyn Package> {
    REGISTRY
        .iter()
        .copied()
        .find(|p| p.package_names().contains(&name))
}

/// Run every package's end-of-document hook over the finished tree.
pub(crate) fn finalize_all(tree: &mut Element, e: &mut Engine) {
    for p in REGISTRY {
        p.finalize(tree, e);
    }
}
