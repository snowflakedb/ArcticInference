//! Native rendering for DOI bibliography macros.

use super::Package;
use crate::engine::{Engine, Event};
use crate::node::{Element, Node};

pub struct Doi;

impl Package for Doi {
    fn commands(&self) -> &[&str] {
        &["doi"]
    }

    fn command(&self, _name: &str, engine: &mut Engine) -> Event {
        let doi = engine.grab_argument_text();
        let identifier = identifier(&doi);
        if identifier.is_empty() {
            return Event::Inline(vec![]);
        }
        let mut link = Element::new("link").attr("href", format!("https://doi.org/{identifier}"));
        link.children = vec![Node::text(identifier)];
        Event::Inline(vec![Node::text("doi: "), Node::Element(link)])
    }
}

fn identifier(value: &str) -> &str {
    let value = value.trim();
    for prefix in [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ] {
        if let Some(identifier) = value.strip_prefix(prefix) {
            return identifier;
        }
    }
    value.strip_prefix("doi:").map(str::trim).unwrap_or(value)
}

#[cfg(test)]
mod tests {
    #[test]
    fn doi_macro_uses_an_absolute_link_but_can_be_redefined() {
        let native = crate::latex_to_markdown(
            r"\providecommand{\doi}[1]{doi: #1}\begin{document}\doi{10.1000/example_1}\end{document}",
        );
        assert!(
            native.contains("doi: [10.1000/example\\_1](https://doi.org/10.1000/example_1)"),
            "{native}"
        );

        let redefined = crate::latex_to_markdown(
            r"\renewcommand{\doi}[1]{CUSTOM:#1}\begin{document}\doi{10.1000/example}\end{document}",
        );
        assert!(redefined.contains("CUSTOM:10.1000/example"), "{redefined}");
        assert!(!redefined.contains("doi.org"), "{redefined}");
    }
}
