//! The document tree produced by the engine.
//!
//! Deliberately tiny: a node is either text or an element with attributes and
//! children. This maps one-to-one onto XML (see [`crate::backend::xml`]) and, later,
//! onto Markdown. Keeping the tree this plain is a feature — every construct the
//! engine understands lowers into these two shapes and nothing more.

/// A node in the document tree.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    /// A run of literal text.
    Text(String),
    /// An element with a tag name, attributes, and children.
    Element(Element),
}

/// An element: a named node with attributes and child nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct Element {
    pub name: String,
    pub attributes: Vec<(String, String)>,
    pub children: Vec<Node>,
}

impl Element {
    /// Create an empty element with the given tag name.
    pub fn new(name: impl Into<String>) -> Self {
        Element {
            name: name.into(),
            attributes: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Add an attribute, returning `self` for chaining.
    pub fn attr(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.push((key.into(), value.into()));
        self
    }

    /// Append a child node.
    pub fn push(&mut self, node: Node) {
        self.children.push(node);
    }
}

impl Node {
    /// Convenience constructor for a text node.
    pub fn text(s: impl Into<String>) -> Node {
        Node::Text(s.into())
    }
}
