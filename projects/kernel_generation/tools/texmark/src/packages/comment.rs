//! Native handling for the `comment` package.
//!
//! `\usepackage{comment}` provides a `comment` environment whose entire body —
//! everything from `\begin{comment}` to `\end{comment}` — is discarded, exactly
//! like one long `%` comment. Papers lean on it to park draft prose, superseded
//! figures, and half-written LaTeX; none of it should reach the output.
//!
//! When the comment body is present literally in the source — the common case —
//! it is scanned as raw text and dropped whole, so unbalanced braces or a stray
//! `\begin`/`\cite{}` can neither derail the parse nor leak into the Markdown,
//! matching the real package, which never interprets what it swallows. (Only if
//! the body arrives via macro expansion do we fall back to reading tokens —
//! still all discarded, though an `\input`/`\include` there could trigger I/O.)

use super::Package;
use crate::engine::{Engine, Event};

pub struct Comment;

impl Package for Comment {
    fn environments(&self) -> &[&str] {
        &["comment"]
    }

    fn environment(&self, name: &str, e: &mut Engine) -> Event {
        // Swallow the raw body up to (and including) `\end{comment}`; emit nothing.
        let _ = e.read_environment_raw_text(name);
        Event::Inline(vec![])
    }
}

#[cfg(test)]
mod tests {
    use crate::latex_to_markdown;

    #[test]
    fn comment_body_is_discarded() {
        let md = latex_to_markdown(
            "\\documentclass{article}\\usepackage{comment}\\begin{document}\
             Before.\\begin{comment}HIDDEN draft text\\end{comment}After.\\end{document}",
        );
        assert!(md.contains("Before."), "kept text before: {md}");
        assert!(md.contains("After."), "kept text after: {md}");
        assert!(!md.contains("HIDDEN"), "must drop comment body: {md}");
        assert!(!md.contains("Comment"), "no leaked 'Comment' label: {md}");
    }

    #[test]
    fn comment_body_is_not_parsed() {
        // Half-written LaTeX inside a comment (unbalanced brace, empty \cite,
        // stray control words) must never reach the parser or the output.
        let md = latex_to_markdown(
            "\\documentclass{article}\\usepackage{comment}\\begin{document}\
             Keep.\\begin{comment}\\cite{} /name {unbalanced \\section{ghost}\
             \\end{comment}Tail.\\end{document}",
        );
        assert!(
            md.contains("Keep.") && md.contains("Tail."),
            "surrounding text intact: {md}"
        );
        assert!(
            !md.contains("ghost"),
            "commented \\section must not appear: {md}"
        );
        assert!(
            !md.contains("/name"),
            "commented draft must not appear: {md}"
        );
    }
}
