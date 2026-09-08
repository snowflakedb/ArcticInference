//! Native rendering for listings-style packages.
//!
//! Covers `lstlisting` (from the `listings` package), `minted`, and `Verbatim`
//! (from the `fancyvrb` package). All produce a `<verbatim>` node with an
//! optional `language` attribute.
//!
//! The body is read as raw source and its leading argument is parsed in the
//! same raw domain (never by tokenizing past the code), so indentation, blank
//! lines, and backslashes survive intact:
//! - `lstlisting` / `Verbatim`: optional `[language=X,…]` key-val argument.
//! - `minted`: required `{language}` argument.

use super::Package;
use crate::engine::{Engine, Event};
use crate::node::{Element, Node};

pub struct Listings;

impl Package for Listings {
    fn package_names(&self) -> &[&str] {
        &["listings", "minted"]
    }

    fn environments(&self) -> &[&str] {
        &["lstlisting", "minted", "Verbatim"]
    }

    fn environment(&self, name: &str, e: &mut Engine) -> Event {
        // The whole environment body is raw, including any argument on the
        // `\begin` line, so read it all and split the argument off here rather
        // than letting the tokenizer consume past the code.
        let raw = e.read_environment_raw_text(name);
        let (language, body) = if name == "minted" {
            let (_, rest) = strip_optional_argument(&raw);
            let (lang, rest) = strip_braced_argument(rest.trim_start());
            (lang, strip_leading_newline(rest))
        } else {
            let (options, rest) = strip_optional_argument(&raw);
            (language_from_options(options), strip_leading_newline(rest))
        };

        let mut node = Element::new("verbatim");
        if !language.is_empty() {
            node.attributes.push(("language".into(), language));
        }
        node.push(Node::text(restore_reconstructed_lines(body)));
        Event::Blocks(vec![Node::Element(node)])
    }
}

/// Split a leading `[…]` optional argument off raw body text, returning its
/// contents and the remainder. Absent a leading `[`, the options are empty and
/// the whole text is the body.
fn strip_optional_argument(raw: &str) -> (&str, &str) {
    match raw.strip_prefix('[').and_then(|r| r.split_once(']')) {
        Some((options, rest)) => (options, rest),
        None => ("", raw),
    }
}

/// Split a leading `{…}` required argument off raw body text (minted's
/// language), returning its contents and the remainder.
fn strip_braced_argument(raw: &str) -> (String, &str) {
    match raw.strip_prefix('{').and_then(|r| r.split_once('}')) {
        Some((arg, rest)) => (arg.trim().to_string(), rest),
        None => (String::new(), raw),
    }
}

/// Drop a single leading newline (the code begins on the line after the
/// `\begin{…}`, once any argument on that line has been removed).
fn strip_leading_newline(s: &str) -> &str {
    s.strip_prefix("\r\n")
        .or_else(|| s.strip_prefix('\n'))
        .unwrap_or(s)
}

fn restore_reconstructed_lines(body: &str) -> String {
    if body.contains('\n') {
        body.to_string()
    } else {
        body.replace("\\par ", "\n")
    }
}

/// Extract a `language=…` value from a listings-style option string.
fn language_from_options(options: &str) -> String {
    options
        .split(',')
        .find_map(|part| part.trim().strip_prefix("language="))
        .map(|v| v.trim().trim_matches(['{', '}']).to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use crate::latex_to_markdown;

    #[test]
    fn minted_accepts_options_before_language() {
        let markdown = latex_to_markdown(
            "\\begin{document}\\begin{minted}[frame=lines]{python}\nprint(1)\n\\end{minted}\\end{document}",
        );
        assert!(markdown.contains("```python\nprint(1)\n```"), "{markdown}");
        assert!(!markdown.contains("frame=lines"), "{markdown}");
    }
}
