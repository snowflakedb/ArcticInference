//! Native rendering for the `hyperref` package.
//!
//! Handles `\url{target}` → a link whose label is the URL itself.
//! `\href` stays in the engine because it renders its second argument inline.

use super::Package;
use crate::engine::{Engine, Event};
use crate::node::{Element, Node};

pub struct Hyperref;

impl Package for Hyperref {
    fn package_names(&self) -> &[&str] {
        &["hyperref"]
    }

    fn commands(&self) -> &[&str] {
        &["url", "Url"]
    }

    fn command(&self, _name: &str, e: &mut Engine) -> Event {
        // A link with no label: the backends use the raw href as the visible
        // text. (Do NOT add the URL as a text child — as prose it would get
        // typographic ligatures, corrupting `--`/quotes in the displayed URL;
        // `\url` is verbatim in LaTeX.)
        let target = e.grab_argument_text();
        let link = Element::new("link").attr("href", &target);
        Event::Inline(vec![Node::Element(link)])
    }
}
