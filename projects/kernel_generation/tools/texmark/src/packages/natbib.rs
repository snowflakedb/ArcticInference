//! Native rendering for the `natbib` citation/bibliography package.
//!
//! Owns the citation commands (`\cite` and friends → `<cite>` markers) and the
//! bibliography machinery. `\bibliography{db}` marks where the reference list
//! goes; a pre-generated `.bbl` (BibTeX's output) is used verbatim when present,
//! otherwise the `.bib` database is formatted natively in `Natbib::finalize` —
//! the equivalent of natbib emitting the bibliography at `\AtEndDocument`.

use super::Package;
use crate::bibtex::{self, Cited};
use crate::engine::{Engine, Event};
use crate::node::{Element, Node};
use crate::state::State;
use crate::tokenizer::{balanced_group, control_word};
use std::collections::{BTreeSet, HashMap};

/// The element name of the not-yet-filled bibliography, created by
/// `\bibliography` and consumed by `Natbib::finalize`.
pub const PLACEHOLDER: &str = "pending-bibliography";

/// How natbib cites and lists references. natbib's default is author-year; its
/// `numbers`/`super` option (and base-LaTeX `\cite`, when natbib isn't loaded)
/// is numeric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CitationMode {
    Numeric,
    AuthorYear,
}

/// The citation mode natbib recorded in [`State::package_state`] when it was
/// loaded. Absent (no natbib, base-LaTeX `\cite`) means numeric.
fn mode(state: &State) -> CitationMode {
    match state.package_state.get("natbib:mode").map(String::as_str) {
        Some("authoryear") => CitationMode::AuthorYear,
        _ => CitationMode::Numeric,
    }
}

pub struct Natbib;

impl Package for Natbib {
    fn commands(&self) -> &[&str] {
        &[
            "cite",
            "citep",
            "citet",
            "citealp",
            "citealt",
            "citeauthor",
            "citeyear",
            "citeyearpar",
            "Citep",
            "Citet",
            "autocite",
            "Autocite",
            "parencite",
            "Parencite",
            "textcite",
            "Textcite",
            "smartcite",
            "footcite",
            "nocite",
            "bibliography",
            "bibliographystyle",
            "addbibresource",
            "printbibliography",
        ]
    }

    fn package_names(&self) -> &[&str] {
        &["natbib", "biblatex"]
    }

    fn configure(&self, options: &str, state: &mut State) {
        // The `numbers`/`numeric`/`super` option selects numeric citations;
        // anything else (including no options) is natbib's author-year default.
        let numeric = options
            .split(',')
            .map(str::trim)
            .any(|o| matches!(o, "numbers" | "numeric" | "super"));
        state.package_state.insert(
            "natbib:mode".into(),
            if numeric { "numeric" } else { "authoryear" }.into(),
        );
    }

    fn command(&self, name: &str, e: &mut Engine) -> Event {
        match name {
            // \nocite{keys} (or {*}) adds to the reference list without printing
            // a citation; the marker is invisible but collected in finalize.
            "nocite" => {
                let keys = e.grab_argument_text();
                Event::Inline(vec![Node::Element(
                    Element::new("nocite").attr("keys", keys),
                )])
            }
            "bibliography" => {
                let dbs = e.grab_argument_text();
                if let Some(bbl) = e.resolver().resolve_bibliography() {
                    e.splice_source(&bbl); // PRIMARY: the author's exact .bbl
                    Event::Inline(vec![])
                } else {
                    // FALLBACK: fill from <db>.bib after the full parse.
                    Event::Blocks(vec![Node::Element(
                        Element::new(PLACEHOLDER).attr("db", dbs),
                    )])
                }
            }
            // biblatex: `\addbibresource{refs.bib}` names the database. Record it
            // so a later `\printbibliography` can fall back to it.
            "addbibresource" => {
                let _ = e.grab_optional();
                let db = e.grab_argument_text();
                let db = db.trim().trim_end_matches(".bib").to_string();
                e.state_mut()
                    .package_state
                    .insert("natbib:bibresource".into(), db);
                Event::Inline(vec![])
            }
            // biblatex: `\printbibliography[opts]` marks where the list goes. It
            // takes only an OPTIONAL argument — grabbing a mandatory one here used
            // to swallow the following `\end{document}`. Emit the list like
            // `\bibliography` (from the .bbl, or the `\addbibresource` .bib).
            "printbibliography" => {
                let _ = e.grab_optional();
                if let Some(bbl) = e.resolver().resolve_bibliography() {
                    e.splice_source(&bbl);
                    Event::Inline(vec![])
                } else {
                    let db = e
                        .state()
                        .package_state
                        .get("natbib:bibresource")
                        .cloned()
                        .unwrap_or_default();
                    Event::Blocks(vec![Node::Element(
                        Element::new(PLACEHOLDER).attr("db", db),
                    )])
                }
            }
            // Presentational no-op: `\bibliographystyle{plain}` selects a style.
            "bibliographystyle" => {
                let _ = e.grab_optional();
                let _ = e.grab_argument();
                Event::Inline(vec![])
            }
            // Everything else this package registers is a citation command.
            // natbib citations take two optional args: `\citep[pre][post]{keys}`.
            // Record the command so finalize can render the author-year form
            // (`\citet` -> "Author (year)", `\citep` -> "(Author, year)", ...).
            _ => {
                let _ = e.grab_optional();
                let _ = e.grab_optional();
                let keys = e.grab_argument_text();
                Event::Inline(vec![Node::Element(
                    Element::new("cite").attr("keys", keys).attr("cmd", name),
                )])
            }
        }
    }

    fn finalize(&self, tree: &mut Element, e: &mut Engine) {
        // First assemble the reference list (from a `.bib` if there was no
        // `.bbl`), then number it and resolve every `\cite*` against it.
        let mut cited = BTreeSet::new();
        let mut all = false;
        collect_keys(tree, &mut cited, &mut all);
        let cited = if all { Cited::All } else { Cited::Keys(cited) };
        fill(tree, e, &cited);
        resolve_citations(tree, e.state());
    }
}

/// Number the reference list and rewrite every `<cite>` against it. The citation
/// style follows the bibliography itself, matching LaTeX: `thebibliography`
/// numbers every `\bibitem`, and natbib's author-year form only applies when the
/// items carry `Author(Year)` labels. So a natbib document whose bibliography is
/// numeric (plain `\bibitem{key}`, no optional label) resolves numerically — the
/// number is always the fallback, so a cite never degrades to a raw key.
fn resolve_citations(tree: &mut Element, state: &State) {
    let mode = effective_mode(state, tree);
    let mut numbers: HashMap<String, usize> = HashMap::new();
    let mut labels: HashMap<String, String> = HashMap::new();
    number_bibliography(tree, mode, &mut numbers, &mut labels);
    rewrite_cites(tree, mode, &numbers, &labels);
}

/// The citation style actually in effect. natbib's `numbers`/`super` option (and
/// base-LaTeX `\cite`) is numeric outright. natbib's author-year default only
/// holds when the bibliography carries `Author(Year)` labels; a plain/numeric
/// bibliography (unlabeled `\bibitem{key}`) is numeric regardless — exactly as
/// LaTeX renders it — rather than failing to find labels that were never there.
fn effective_mode(state: &State, tree: &Element) -> CitationMode {
    match mode(state) {
        CitationMode::AuthorYear if !has_authoryear_labels(tree) => CitationMode::Numeric,
        m => m,
    }
}

/// Whether any `<bibitem>` carries an `Author(Year)`-style label — the signal
/// that the bibliography is author-year rather than numeric.
fn has_authoryear_labels(el: &Element) -> bool {
    if el.name == "bibitem" {
        return attr(el, "label").is_some_and(is_author_year);
    }
    el.children
        .iter()
        .any(|c| matches!(c, Node::Element(e) if has_authoryear_labels(e)))
}

/// A label looks like natbib author-year if it ends with a parenthesized year,
/// e.g. `Aggarwal and Vitter(1988)`.
fn is_author_year(label: &str) -> bool {
    label.trim_end().ends_with(')') && label.contains('(')
}

/// Walk to the reference list, record each entry's number and `[label]`, and
/// stamp the bibitems for the mode: `[1]` numeric, no bracket label author-year.
fn number_bibliography(
    el: &mut Element,
    mode: CitationMode,
    numbers: &mut HashMap<String, usize>,
    labels: &mut HashMap<String, String>,
) {
    if el.name == "bibliography" {
        let mut n = 0;
        for child in &mut el.children {
            if let Node::Element(item) = child
                && item.name == "bibitem"
                && let Some(key) = attr(item, "key").map(str::to_string)
            {
                n += 1;
                numbers.insert(key.clone(), n);
                if let Some(label) = attr(item, "label") {
                    labels.insert(key.clone(), label.to_string());
                }
                // Stamp the entry's link anchor id (`ref-<key>`) so the backend
                // emits a matching `<a id>` that in-text cites can point at. The
                // id is derived from the (unique) cite key, mirroring how a
                // `<label>` anchor is derived from its label id.
                item.attributes.push(("anchor".into(), cite_anchor(&key)));
                match mode {
                    CitationMode::Numeric => item.attributes.push(("number".into(), n.to_string())),
                    CitationMode::AuthorYear => {
                        item.attributes.push(("unlabeled".into(), "1".into()))
                    }
                }
            }
        }
        return;
    }
    for child in &mut el.children {
        if let Node::Element(c) = child {
            number_bibliography(c, mode, numbers, labels);
        }
    }
}

/// Rewrite every `<cite>` in the tree with its resolved text.
fn rewrite_cites(
    el: &mut Element,
    mode: CitationMode,
    numbers: &HashMap<String, usize>,
    labels: &HashMap<String, String>,
) {
    for child in &mut el.children {
        if let Node::Element(c) = child {
            if c.name == "cite" {
                rewrite_cite(c, mode, numbers, labels);
            } else if c.name == "verbatim"
                && attr(c, "language").is_some_and(|language| language == "pseudocode")
            {
                rewrite_pseudocode_cites(c, mode, numbers, labels);
            }
            rewrite_cites(c, mode, numbers, labels);
        }
    }
}

fn rewrite_pseudocode_cites(
    element: &mut Element,
    mode: CitationMode,
    numbers: &HashMap<String, usize>,
    labels: &HashMap<String, String>,
) {
    for child in &mut element.children {
        if let Node::Text(text) = child {
            *text = replace_text_cites(text, mode, numbers, labels);
        }
    }
}

fn replace_text_cites(
    text: &str,
    mode: CitationMode,
    numbers: &HashMap<String, usize>,
    labels: &HashMap<String, String>,
) -> String {
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        let Some((command, command_end)) = control_word(text, i) else {
            let character = text[i..].chars().next().unwrap();
            out.push(character);
            i += character.len_utf8();
            continue;
        };
        if !matches!(
            command,
            "cite" | "citep" | "citet" | "citealp" | "citealt" | "citeauthor" | "citeyear"
        ) {
            out.push('\\');
            i += 1;
            continue;
        }
        let mut argument = command_end;
        for _ in 0..2 {
            argument =
                balanced_group(text, argument, b'[', b']').map_or(argument, |(_, next)| next);
        }
        let Some((keys, next)) = balanced_group(text, argument, b'{', b'}') else {
            out.push('\\');
            i += 1;
            continue;
        };
        let keys = keys
            .split(',')
            .map(str::trim)
            .filter(|key| !key.is_empty())
            .collect::<Vec<_>>();
        let Some(resolved) = citation_text(command, &keys, mode, numbers, labels) else {
            out.push_str(&text[i..next]);
            i = next;
            continue;
        };
        out.push_str(&resolved);
        i = next;
    }
    out
}

fn rewrite_cite(
    el: &mut Element,
    mode: CitationMode,
    numbers: &HashMap<String, usize>,
    labels: &HashMap<String, String>,
) {
    let Some(keys) = attr(el, "keys").map(str::to_string) else {
        return;
    };
    let cmd = attr(el, "cmd").unwrap_or("cite").to_string();
    let keys: Vec<&str> = keys
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if keys.is_empty() {
        return;
    }

    let Some(text) = citation_text(&cmd, &keys, mode, numbers, labels) else {
        return;
    };
    if mode == CitationMode::Numeric {
        let mut nums = keys
            .iter()
            .filter_map(|key| numbers.get(*key).map(|number| (*number, *key)))
            .collect::<Vec<_>>();
        nums.sort_unstable();
        nums.dedup();
        let anchors = nums
            .iter()
            .map(|(_, key)| cite_anchor(key))
            .collect::<Vec<_>>()
            .join(",");
        el.attributes.push(("anchors".into(), anchors));
    }
    el.attributes.push(("resolved".into(), text));
}

fn citation_text(
    cmd: &str,
    keys: &[&str],
    mode: CitationMode,
    numbers: &HashMap<String, usize>,
    labels: &HashMap<String, String>,
) -> Option<String> {
    let text = match mode {
        CitationMode::Numeric => {
            // Collect (number, key) so we can emit both the number text and the
            // per-number link anchor (`ref-<key>`) that points at the entry.
            let mut nums: Vec<(usize, &str)> = Vec::new();
            for key in keys {
                match numbers.get(*key) {
                    Some(&n) => nums.push((n, key)),
                    None => return None,
                }
            }
            nums.sort_unstable();
            nums.dedup();
            let inner = nums
                .iter()
                .map(|(n, _)| n.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{inner}]")
        }
        CitationMode::AuthorYear => {
            let mut entries: Vec<(String, String)> = Vec::new();
            for key in keys {
                match labels.get(*key) {
                    Some(label) => entries.push(parse_label(label)),
                    // A key without an author-year label (a mixed bibliography):
                    // fall back to its ever-present bibliography number rather
                    // than emitting a raw key.
                    None => match numbers.get(*key) {
                        Some(&n) => entries.push((n.to_string(), String::new())),
                        None => return None,
                    },
                }
            }
            format_author_year(cmd, &entries)
        }
    };
    Some(text)
}

/// Split a natbib citation label ("Aggarwal and Vitter(1988)") into its author
/// text and year. A label without a trailing `(year)` becomes all-author.
fn parse_label(label: &str) -> (String, String) {
    if label.ends_with(')')
        && let Some(open) = label.rfind('(')
    {
        let author = label[..open].trim().to_string();
        let year = label[open + 1..label.len() - 1].trim().to_string();
        return (author, year);
    }
    (label.trim().to_string(), String::new())
}

/// Render author-year citation text for a natbib command over its entries.
/// `\citet` → "Author (year)", `\citep` → "(Author, year)", and the `\citealt`/
/// `\citealp`/`\citeauthor`/`\citeyear` variants drop parens/author/year.
fn format_author_year(cmd: &str, entries: &[(String, String)]) -> String {
    let with_year = |a: &str, y: &str| {
        if y.is_empty() {
            a.to_string()
        } else {
            format!("{a} ({y})")
        }
    };
    let comma_year = |a: &str, y: &str| {
        if y.is_empty() {
            a.to_string()
        } else {
            format!("{a}, {y}")
        }
    };

    match cmd {
        // Parenthetical forms: natbib \citep and biblatex \parencite/\autocite/
        // \smartcite/\footcite → "(Author, year)".
        "citep" | "Citep" | "parencite" | "Parencite" | "autocite" | "Autocite" | "smartcite"
        | "footcite" => {
            let inner = entries
                .iter()
                .map(|(a, y)| comma_year(a, y))
                .collect::<Vec<_>>()
                .join("; ");
            format!("({inner})")
        }
        "citealp" => entries
            .iter()
            .map(|(a, y)| comma_year(a, y))
            .collect::<Vec<_>>()
            .join("; "),
        "citealt" => entries
            .iter()
            .map(|(a, y)| format!("{a} {y}").trim().to_string())
            .collect::<Vec<_>>()
            .join("; "),
        "citeauthor" => entries
            .iter()
            .map(|(a, _)| a.clone())
            .collect::<Vec<_>>()
            .join("; "),
        "citeyear" => entries
            .iter()
            .map(|(_, y)| y.clone())
            .collect::<Vec<_>>()
            .join("; "),
        "citeyearpar" => format!(
            "({})",
            entries
                .iter()
                .map(|(_, y)| y.clone())
                .collect::<Vec<_>>()
                .join("; ")
        ),
        _ => entries
            .iter()
            .map(|(a, y)| with_year(a, y))
            .collect::<Vec<_>>()
            .join("; "),
    }
}

/// Gather every key referenced by `<cite>`/`<nocite>` markers. A `*` key
/// (`\nocite{*}`) sets `all`.
fn collect_keys(el: &Element, cited: &mut BTreeSet<String>, all: &mut bool) {
    for child in &el.children {
        if let Node::Element(c) = child {
            if c.name == "cite" || c.name == "nocite" {
                for key in attr(c, "keys").unwrap_or("").split(',') {
                    let key = key.trim();
                    if key == "*" {
                        *all = true;
                    } else if !key.is_empty() {
                        cited.insert(key.to_string());
                    }
                }
            }
            collect_keys(c, cited, all);
        }
    }
}

/// Replace each [`PLACEHOLDER`] with the formatted reference list, generated
/// from its `.bib` database(s) and re-rendered through the engine.
fn fill(el: &mut Element, e: &mut Engine, cited: &Cited) {
    let mut rebuilt = Vec::with_capacity(el.children.len());
    for child in std::mem::take(&mut el.children) {
        match child {
            Node::Element(c) if c.name == PLACEHOLDER => {
                if let Some(nodes) = render_bibliography(&c, e, cited) {
                    rebuilt.extend(nodes);
                }
                // No .bib / no cited entries: drop the placeholder silently.
            }
            Node::Element(mut c) => {
                fill(&mut c, e, cited);
                rebuilt.push(Node::Element(c));
            }
            other => rebuilt.push(other),
        }
    }
    el.children = rebuilt;
}

/// Resolve the placeholder's `.bib` database(s), format the cited entries, and
/// render them to a `<bibliography>` subtree. `None` if nothing resolves.
fn render_bibliography(placeholder: &Element, e: &mut Engine, cited: &Cited) -> Option<Vec<Node>> {
    let dbs = attr(placeholder, "db").unwrap_or("");
    let mut src = String::new();
    for db in dbs.split(',').map(str::trim).filter(|d| !d.is_empty()) {
        if let Some(text) = e.resolver().resolve(&format!("{db}.bib")) {
            src.push_str(&text);
            src.push('\n');
        }
    }
    let entries = bibtex::parse(&src);
    let generated = bibtex::to_thebibliography(&entries, cited)?;
    Some(e.render_fragment(&generated))
}

/// The link anchor id for a reference entry with cite key `key`. Both endpoints
/// — the `<bibitem>` anchor and every in-text `<cite>` link — derive their id
/// from this one function, so the fragment always matches. The `ref-` prefix
/// keeps cite anchors from colliding with same-named `\label` anchors, and the
/// backend runs the result through `slug` (as it does for labels/equations) to
/// make it a GitHub-valid fragment.
fn cite_anchor(key: &str) -> String {
    format!("ref-{key}")
}

fn attr<'a>(el: &'a Element, key: &str) -> Option<&'a str> {
    el.attributes
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

#[cfg(test)]
mod tests {
    use crate::state::Resolver;

    /// A one-file resolver serving an in-memory `.bib` (no `.bbl`).
    struct BibOnly(String);
    impl Resolver for BibOnly {
        fn resolve(&self, name: &str) -> Option<String> {
            (name == "ref.bib").then(|| self.0.clone())
        }
    }

    #[test]
    fn numeric_when_no_natbib() {
        // Base LaTeX \cite with no natbib defaults to numeric.
        let bib = r#"
            @misc{beta, title={B}, author={Beta, Al}, year={2001}}
            @misc{alpha, title={A}, author={Alpha, Bo}, year={2000}}
        "#;
        let src = "\\begin{document}We use \\cite{beta} then \\cite{alpha}.\\bibliography{ref}\\end{document}";
        let md =
            crate::backend::md::to_markdown(&crate::latex_to_tree_with(src, &BibOnly(bib.into())));
        // Sorted list: Alpha=[1], Beta=[2]; in-text cites use those numbers, and
        // each number links to its reference entry's anchor.
        assert!(
            md.contains("We use [[2](#ref-beta)] then [[1](#ref-alpha)]."),
            "cite numbers/links wrong:\n{md}"
        );
        assert!(
            md.contains("- <a id=\"ref-alpha\"></a>[1]")
                && md.contains("- <a id=\"ref-beta\"></a>[2]"),
            "list not numbered/anchored:\n{md}"
        );
    }

    #[test]
    fn author_year_when_natbib_default() {
        let src = r#"\usepackage{natbib}
\begin{document}\citet{a} says it \citep{a,b}.
\begin{thebibliography}{9}
\bibitem[Alpha(2001)]{a} Al Alpha. 2001.
\bibitem[Beta(2002)]{b} Bo Beta. 2002.
\end{thebibliography}\end{document}"#;
        let md = crate::latex_to_markdown(src);
        assert!(
            md.contains("Alpha (2001) says it"),
            "\\citet author-year:\n{md}"
        );
        assert!(
            md.contains("(Alpha, 2001; Beta, 2002)"),
            "\\citep author-year:\n{md}"
        );
    }

    #[test]
    fn numeric_bibliography_under_natbib_default() {
        // natbib is loaded (author-year default), but the bibliography is
        // numeric — plain `\bibitem{key}` with no `[Author(Year)]` labels. The
        // style follows the bibliography, so citations resolve to numbers and the
        // reference list is numbered, instead of degrading to raw keys. This is
        // the common case of a class that `\RequirePackage{natbib}` with a
        // numeric bibliographystyle (e.g. many conference styles).
        let src = r#"\usepackage{natbib}
\begin{document}See \citep{a} then \citet{b} and \citep{a,b}.
\begin{thebibliography}{10}
\bibitem{a} Al Alpha. Foo. 2001.
\bibitem{b} Bo Beta. Bar. 2002.
\end{thebibliography}\end{document}"#;
        let md = crate::latex_to_markdown(src);
        assert!(
            md.contains("See [[1](#ref-a)] then [[2](#ref-b)] and [[1](#ref-a), [2](#ref-b)]."),
            "unlabeled bibliography must cite numerically (and linked), not raw keys:\n{md}"
        );
        assert!(
            md.contains("- <a id=\"ref-a\"></a>[1] Al Alpha")
                && md.contains("- <a id=\"ref-b\"></a>[2] Bo Beta"),
            "reference list must be numbered and anchored:\n{md}"
        );
    }
}
