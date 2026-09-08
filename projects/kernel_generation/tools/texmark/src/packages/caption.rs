//! Native handling for the `caption` / `subcaption` packages.
//!
//! Provides `\captionof{<type>}{<text>}` (and its starred, unnumbered form
//! `\captionof*`), which captions arbitrary content — typically an
//! `\includegraphics` inside a `minipage` — as though it were a float of
//! `<type>` (`figure`/`table`/`algorithm`), so it is numbered on that type's
//! counter independently of any surrounding float.
//!
//! It is modeled as a captioned `<float>` so it reuses the engine's existing
//! float numbering (the resolve pass) and caption rendering (the backend),
//! rather than reinventing either. The float is tagged `captionof` so the
//! resolve pass numbers it even when nested inside another float — a real
//! `\captionof` is an explicit caption, unlike a subfigure that must not
//! re-step the counter.

use super::Package;
use crate::engine::{Engine, Event};
use crate::node::{Element, Node};

pub struct Caption;

impl Package for Caption {
    fn package_names(&self) -> &[&str] {
        &["caption", "subcaption"]
    }

    fn commands(&self) -> &[&str] {
        // `\captionof*` arrives as the control word "captionof" followed by a
        // `*` token, so registering the base name is enough; the star is
        // consumed in `command`.
        &["captionof"]
    }

    fn command(&self, _name: &str, e: &mut Engine) -> Event {
        // `\captionof*{type}{text}` is the unnumbered variant.
        let numbered = !e.consume_star();
        let kind = e.grab_argument_text();
        let _ = e.grab_optional(); // optional list-of-figures entry
        let text = e.grab_argument();

        let mut caption = Element::new("caption");
        caption.children = e.render_inline(text);

        let mut float = Element::new("float").attr("kind", normalize_kind(&kind));
        if numbered {
            float = float.attr("captionof", "1");
        }
        float.push(Node::Element(caption));
        Event::Blocks(vec![Node::Element(float)])
    }
}

/// Map a caption type to the float kind the resolve pass counts; unknown types
/// fall back to `figure` (matching the resolve pass's own default).
fn normalize_kind(kind: &str) -> &'static str {
    match kind.trim() {
        "table" => "table",
        "algorithm" => "algorithm",
        _ => "figure",
    }
}

#[cfg(test)]
mod tests {
    use crate::latex_to_markdown;

    #[test]
    fn captionof_figure_is_numbered_inside_a_table_float() {
        // A `\captionof{figure}` inside a `table` float must be numbered on the
        // FIGURE counter, and must not make the enclosing table count as "Table 1".
        let md = latex_to_markdown(
            "\\documentclass{article}\\usepackage{caption}\\begin{document}\
             \\begin{figure}\\caption{First}\\end{figure}\
             \\begin{table}\\captionof{figure}{Second}\\end{table}\
             \\end{document}",
        );
        assert!(
            md.contains("**Figure 1:** First"),
            "native figure numbered: {md}"
        );
        assert!(
            md.contains("**Figure 2:** Second"),
            "captionof continues figure counter: {md}"
        );
        assert!(
            !md.contains("Table 1"),
            "enclosing table must not be numbered: {md}"
        );
    }

    #[test]
    fn captionof_binds_a_following_label() {
        let md = latex_to_markdown(
            "\\documentclass{article}\\usepackage{caption}\\begin{document}\
             \\begin{table}\\captionof{figure}{Cap}\\label{f}\\end{table}\
             See \\ref{f}.\\end{document}",
        );
        assert!(md.contains("**Figure 1:** Cap"), "captionof numbered: {md}");
        // `\ref{f}` resolves to the captionof's number, linking to its anchor.
        assert!(
            md.contains("[1](#f)"),
            "ref resolves to the captionof number: {md}"
        );
    }

    #[test]
    fn starred_captionof_is_unnumbered() {
        let md = latex_to_markdown(
            "\\documentclass{article}\\usepackage{caption}\\begin{document}\
             \\begin{table}\\captionof*{figure}{Bare}\\end{table}\\end{document}",
        );
        assert!(md.contains("Bare"), "text kept: {md}");
        assert!(!md.contains("Figure 1"), "starred form is unnumbered: {md}");
    }
}
