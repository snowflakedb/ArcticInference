//! Serializing the document tree to XML.
//!
//! A small, dependency-free writer: it escapes text and attribute values and
//! indents the tree for readability. Elements with no children are written as
//! self-closing tags.

use crate::node::{Element, Node};
use std::fmt::Write;

use super::{escape_attr as escape_attribute, escape_text};

/// The XML backend. Wraps [`to_xml`] as a [`crate::Backend`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Xml;

impl super::Backend for Xml {
    fn serialize(&self, root: &Element) -> String {
        to_xml(root)
    }
}

/// Serialize a root element to an XML document string (with declaration).
pub fn to_xml(root: &Element) -> String {
    let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    write_element(&mut out, root, 0);
    out
}

/// Serialize a single element (no XML declaration), for tests and fragments.
pub fn element_to_xml(element: &Element) -> String {
    let mut out = String::new();
    write_element(&mut out, element, 0);
    out
}

fn write_element(out: &mut String, el: &Element, depth: usize) {
    let indent = "  ".repeat(depth);
    let _ = write!(out, "{indent}<{}", el.name);
    for (key, value) in &el.attributes {
        let _ = write!(out, " {key}=\"{}\"", escape_attribute(value));
    }

    if el.children.is_empty() {
        out.push_str("/>\n");
        return;
    }

    // A single text child stays on one line: <p>hello</p>.
    if let [Node::Text(text)] = el.children.as_slice() {
        let _ = writeln!(out, ">{}</{}>", escape_text(text), el.name);
        return;
    }

    out.push_str(">\n");
    for child in &el.children {
        write_node(out, child, depth + 1);
    }
    let _ = writeln!(out, "{indent}</{}>", el.name);
}

fn write_node(out: &mut String, node: &Node, depth: usize) {
    match node {
        Node::Text(text) => {
            let indent = "  ".repeat(depth);
            let _ = writeln!(out, "{indent}{}", escape_text(text));
        }
        Node::Element(el) => write_element(out, el, depth),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_closing() {
        let el = Element::new("br");
        assert_eq!(element_to_xml(&el), "<br/>\n");
    }

    #[test]
    fn text_child_inline() {
        let mut el = Element::new("p");
        el.push(Node::text("hello"));
        assert_eq!(element_to_xml(&el), "<p>hello</p>\n");
    }

    #[test]
    fn escaping() {
        let mut el = Element::new("p");
        el.push(Node::text("a < b & c"));
        assert_eq!(element_to_xml(&el), "<p>a &lt; b &amp; c</p>\n");
    }

    #[test]
    fn attributes_escaped() {
        let el = Element::new("ref").attr("target", "a\"b&c");
        assert_eq!(element_to_xml(&el), "<ref target=\"a&quot;b&amp;c\"/>\n");
    }

    #[test]
    fn nested() {
        let mut inner = Element::new("title");
        inner.push(Node::text("Hi"));
        let mut outer = Element::new("section");
        outer.push(Node::Element(inner));
        assert_eq!(
            element_to_xml(&outer),
            "<section>\n  <title>Hi</title>\n</section>\n"
        );
    }
}
