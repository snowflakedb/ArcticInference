//! Native rendering for the `nicefrac` package.
//!
//! Handles `\nicefrac{num}{den}` → inline math `\frac{num}{den}`.

use super::Package;
use crate::engine::{Engine, Event};
use crate::node::{Element, Node};

pub struct Nicefrac;

impl Package for Nicefrac {
    fn package_names(&self) -> &[&str] {
        &["nicefrac"]
    }

    fn commands(&self) -> &[&str] {
        &["nicefrac"]
    }

    fn command(&self, _name: &str, e: &mut Engine) -> Event {
        let num = e.grab_argument_text();
        let den = e.grab_argument_text();
        let mut math = Element::new("math").attr("mode", "inline");
        math.push(Node::text(format!("\\frac{{{num}}}{{{den}}}")));
        Event::Inline(vec![Node::Element(math)])
    }
}
