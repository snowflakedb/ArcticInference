//! The resolution pass — texmark's `.aux`-equivalent second phase.
//!
//! LaTeX runs twice on purpose: the first pass builds the document and records,
//! in the `.aux` file, the number every `\label` resolves to; the second pass
//! substitutes those numbers into `\ref`. A number cannot be known during the
//! forward build because a reference routinely precedes its target
//! (`\cref{fig:x}` before the figure).
//!
//! This module is that second pass for the document's own structure. Given the
//! fully-built tree it:
//!
//! 1. walks the tree in document order, stepping counters for
//!    sections/floats/theorems/equations and binding each `<label>` to the
//!    number of the construct it sits in (a `label → target` table);
//! 2. rewrites every `<ref>`/`\cref`/`\eqref` against that table, and stamps
//!    display prefixes onto captions and theorem blocks.
//!
//! Citations and the reference list are the natbib package's concern, resolved
//! in its `finalize` hook — not here. The tree stays backend-agnostic:
//! resolution is recorded as attributes (`resolved`, `anchor`, `prefix`,
//! `number`) that the Markdown/HTML/XML serializers honor, not baked into text.

use crate::node::{Element, Node};
use crate::state::State;
use crate::tokenizer::{balanced_group, control_word};
use std::collections::HashMap;

/// What a label points at: a human-facing kind ("Figure", "Section", a theorem's
/// printed name) and its assigned number ("3", "2.1").
#[derive(Clone)]
struct Target {
    kind: String,
    number: String,
}

/// Run the resolution pass over the document root, in place: number the
/// document's sections/floats/theorems/equations and resolve `\ref`/`\cref`.
/// Citations and the reference list are owned by the natbib package (resolved in
/// its `finalize` hook), not here.
pub fn resolve(root: &mut Element, state: &State) {
    let mut ctx = Numberer::new(state);
    let mut labels = HashMap::new();
    ctx.walk(root, &mut labels);

    rewrite(root, &labels);
}

// --- label numbering --------------------------------------------------------

/// Carries the counter state across the document-order walk.
struct Numberer<'a> {
    state: &'a State,
    /// Section counters by depth (index 0 = `\section`); deeper levels reset
    /// when a higher level steps.
    sections: Vec<i32>,
    figures: i32,
    tables: i32,
    algorithms: i32,
    equations: i32,
    /// Theorem-family counters, keyed by the (possibly shared) counter name.
    theorems: HashMap<String, i32>,
    /// For `[within]`-scoped theorems: the parent section prefix in effect when
    /// each theorem counter was last reset, so it resets when the parent steps.
    theorem_epoch: HashMap<String, String>,
    /// The most recently numbered construct — what the next `\label` binds to,
    /// mirroring LaTeX's `\@currentlabel`.
    current: Option<Target>,
    /// The enclosing floats, innermost last, as `(kind, is_captionof)`. A caption
    /// is numbered by its nearest float's `kind`; the flags let a `\captionof`
    /// float count while a nested plain subfigure does not.
    float_stack: Vec<(String, bool)>,
    appendix: bool,
}

impl<'a> Numberer<'a> {
    fn new(state: &'a State) -> Self {
        Numberer {
            state,
            sections: Vec::new(),
            figures: 0,
            tables: 0,
            algorithms: 0,
            equations: 0,
            theorems: HashMap::new(),
            theorem_epoch: HashMap::new(),
            current: None,
            float_stack: Vec::new(),
            appendix: false,
        }
    }

    fn walk(&mut self, el: &mut Element, labels: &mut HashMap<String, Target>) {
        match el.name.as_str() {
            "appendix" => {
                self.appendix = true;
                self.sections.clear();
                self.current = None;
            }
            "section" => {
                if attr(el, "starred").is_none()
                    && let Some(level) = attr(el, "level").and_then(|l| l.parse::<i32>().ok())
                    // The class decides which levels are numbered (article numbers
                    // \section..\subsubsection; book/report \chapter..\subsection).
                    && level >= self.state.numbering.top_level
                    && level <= self.state.numbering.secnumdepth
                {
                    let kind = if level <= 0 { "Chapter" } else { "Section" };
                    let mut number = self.step_section(level);
                    if self.appendix {
                        number = appendix_number(&number);
                    }
                    el.attributes.push(("number".into(), number.clone()));
                    self.current = Some(Target {
                        kind: kind.into(),
                        number,
                    });
                } else {
                    // A starred or too-deep section is unnumbered — it has no
                    // number for a `\label` inside it to point at, so clear the
                    // current target rather than let the label bind to the
                    // previous numbered section (a misleading cross-reference).
                    self.current = None;
                }
            }
            "caption" => {
                // A `\caption` steps its enclosing float's counter and becomes
                // "Figure N"/"Table N". Numbering each caption (not the float)
                // in document order means side-by-side panels — one float with
                // several captions — number independently, and each panel's
                // `\label` binds to its own number.
                if attr(el, "prefix").is_none()
                    && self.caption_is_numbered()
                    && let Some((kind, _)) = self.float_stack.last()
                {
                    let kind = kind.clone();
                    let (name, number) = self.step_float(&kind);
                    el.attributes
                        .push(("prefix".into(), format!("{name} {number}")));
                    self.current = Some(Target { kind: name, number });
                }
            }
            "environment" => {
                if let Some(name) = attr(el, "name")
                    && let Some(def) = self.state.theorems.get(name)
                {
                    match &def.counter {
                        Some(counter) => {
                            // `within` is a property of the counter, not the
                            // individual theorem: `\newtheorem{lem}[thm]{Lemma}`
                            // shares `thm`'s counter and inherits its `[section]`
                            // scoping. Look it up on the counter's owning def.
                            let within = self
                                .state
                                .theorems
                                .get(counter)
                                .and_then(|d| d.within.clone());
                            // A `[within]` theorem is prefixed by its parent
                            // section/chapter number ("Theorem 2.1") and resets
                            // whenever that parent steps; otherwise it counts
                            // sequentially across the whole document.
                            let number = match &within {
                                Some(within) => {
                                    let parent = self.parent_prefix(within);
                                    if self.theorem_epoch.get(counter) != Some(&parent) {
                                        self.theorem_epoch.insert(counter.clone(), parent.clone());
                                        self.theorems.insert(counter.clone(), 0);
                                    }
                                    let n = self.theorems.entry(counter.clone()).or_insert(0);
                                    *n += 1;
                                    format!("{parent}.{n}")
                                }
                                None => {
                                    let n = self.theorems.entry(counter.clone()).or_insert(0);
                                    *n += 1;
                                    n.to_string()
                                }
                            };
                            el.attributes
                                .push(("prefix".into(), format!("{} {number}", def.printed)));
                            self.current = Some(Target {
                                kind: def.printed.clone(),
                                number,
                            });
                        }
                        None => {
                            // Unnumbered (starred) theorem: label it, no number.
                            el.attributes.push(("prefix".into(), def.printed.clone()));
                            self.current = None;
                        }
                    }
                }
            }
            "math" => {
                // Only numbered environments step the equation counter (starred
                // forms, inline math, and sub-environments do not). Labels live in
                // the raw TeX now (the engine no longer pre-extracts them); pull
                // them out here to number, and stamp a `labels` attribute so the
                // backend can emit link anchors.
                let tex = el
                    .children
                    .iter()
                    .find_map(|n| match n {
                        Node::Text(t) => Some(t.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                let numbered = attr(el, "env").is_some_and(|e| {
                    matches!(
                        e,
                        "equation" | "align" | "gather" | "multline" | "eqnarray" | "flalign"
                    )
                }) || explicitly_numbers_equation(&tex);
                if numbered {
                    let ids = math_labels(&tex);
                    for id in &ids {
                        self.equations += 1;
                        labels.insert(
                            id.clone(),
                            Target {
                                kind: "Equation".into(),
                                number: self.equations.to_string(),
                            },
                        );
                    }
                    if !ids.is_empty() {
                        el.attributes.push(("labels".into(), ids.join(",")));
                    }
                }
            }
            "label" => {
                if let Some(id) = attr(el, "id")
                    && let Some(current) = &self.current
                {
                    labels.insert(id.to_string(), current.clone());
                }
            }
            _ => {}
        }

        let entering_float = el.name == "float";
        if entering_float {
            let kind = attr(el, "kind").unwrap_or("figure").to_string();
            let is_captionof = attr(el, "captionof").is_some();
            self.float_stack.push((kind, is_captionof));
        }
        for child in &mut el.children {
            if let Node::Element(c) = child {
                self.walk(c, labels);
            }
        }
        if entering_float {
            self.float_stack.pop();
        }
    }

    /// Step the section counter for `level`, reset deeper levels, and return the
    /// dotted number. Counters are indexed from the class's `top_level`, so a
    /// book chapter (`top_level` 0) is "1" and its section "1.1", while an article
    /// section (`top_level` 1) is "1" and its subsection "1.1".
    fn step_section(&mut self, level: i32) -> String {
        let idx = (level - self.state.numbering.top_level).max(0) as usize;
        let depth = idx + 1;
        if self.sections.len() < depth {
            self.sections.resize(depth, 0);
        } else {
            self.sections.truncate(depth);
        }
        self.sections[idx] += 1;
        self.sections
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(".")
    }

    /// The dotted section number of the counter a `[within]` theorem is scoped
    /// to (e.g. `section` → "2", `chapter` → "5", when a book section is "5.2").
    /// The section vec is indexed from the class `top_level`. Before any section
    /// exists, LaTeX's counter is 0, so this returns "0".
    ///
    /// Assumes `within` names a sectioning counter (chapter/section/…), which is
    /// almost always the case; a non-section counter (a custom or list counter)
    /// falls through `section_level`'s default and is treated as subsection-level.
    fn parent_prefix(&self, within: &str) -> String {
        if self.sections.is_empty() {
            return "0".to_string();
        }
        let level = crate::engine::section_level(within);
        let idx = (level - self.state.numbering.top_level).max(0) as usize;
        let end = (idx + 1).min(self.sections.len());
        let number = self.sections[..end]
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(".");
        if self.appendix {
            appendix_number(&number)
        } else {
            number
        }
    }

    /// Step the counter for a float `kind`, returning its printed name and number.
    fn step_float(&mut self, kind: &str) -> (String, String) {
        let (counter, name) = match kind {
            "table" => (&mut self.tables, "Table"),
            "algorithm" => (&mut self.algorithms, "Algorithm"),
            _ => (&mut self.figures, "Figure"),
        };
        *counter += 1;
        (name.to_string(), counter.to_string())
    }

    /// Whether the caption currently being walked should be numbered. A caption
    /// is numbered when its nearest enclosing float is either the outermost
    /// float or a `\captionof` (which is an explicit caption, never a
    /// subfigure). The guard is on *plain* (non-`captionof`) floats: two of them
    /// on the stack means the caption belongs to a nested subfigure, which does
    /// not step the counter.
    fn caption_is_numbered(&self) -> bool {
        match self.float_stack.last() {
            None => false,
            Some((_, true)) => true, // inside a \captionof float
            Some((_, false)) => self.float_stack.iter().filter(|(_, capof)| !capof).count() <= 1,
        }
    }
}

fn appendix_number(number: &str) -> String {
    let mut parts = number.split('.');
    let first = parts
        .next()
        .and_then(|part| part.parse::<u8>().ok())
        .filter(|number| *number > 0)
        .map(|number| char::from(b'A' + (number - 1) % 26).to_string())
        .unwrap_or_default();
    std::iter::once(first.as_str())
        .chain(parts)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(".")
}

// --- rewrite ----------------------------------------------------------------

/// Rewrite `<ref>` nodes against the label table. (`<cite>` is natbib's.)
fn rewrite(el: &mut Element, labels: &HashMap<String, Target>) {
    for child in &mut el.children {
        if let Node::Element(c) = child {
            if c.name == "ref" {
                rewrite_ref(c, labels);
            } else if c.name == "math"
                || (c.name == "verbatim"
                    && attr(c, "language").is_some_and(|language| language == "pseudocode"))
            {
                rewrite_text_refs(c, labels);
            }
            rewrite(c, labels);
        }
    }
}

fn rewrite_text_refs(element: &mut Element, labels: &HashMap<String, Target>) {
    for child in &mut element.children {
        let Node::Text(tex) = child else {
            continue;
        };
        *tex = replace_text_refs(tex, labels);
    }
}

fn replace_text_refs(tex: &str, labels: &HashMap<String, Target>) -> String {
    let mut out = String::with_capacity(tex.len());
    let mut i = 0;
    while i < tex.len() {
        let Some((command, command_end)) = control_word(tex, i) else {
            let character = tex[i..].chars().next().unwrap();
            out.push(character);
            i += character.len_utf8();
            continue;
        };
        if !matches!(command, "ref" | "eqref") {
            out.push('\\');
            i += 1;
            continue;
        }
        let Some((label, next)) = balanced_group(tex, command_end, b'{', b'}') else {
            out.push('\\');
            i += 1;
            continue;
        };
        let Some(target) = labels.get(label.trim()) else {
            out.push_str(&tex[i..next]);
            i = next;
            continue;
        };
        if command == "eqref" {
            out.push('(');
            out.push_str(&target.number);
            out.push(')');
        } else {
            out.push_str(&target.number);
        }
        i = next;
    }
    out
}

fn rewrite_ref(el: &mut Element, labels: &HashMap<String, Target>) {
    let Some(target) = attr(el, "target").map(str::to_string) else {
        return;
    };
    let cmd = attr(el, "cmd").unwrap_or("ref").to_string();

    // A reference may name several targets (`\cref{a,b}`).
    let resolved: Vec<(&str, &Target)> = target
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter_map(|id| labels.get(id).map(|t| (id, t)))
        .collect();

    if resolved.is_empty() {
        return; // leave unresolved; the serializer keeps the raw target link
    }

    let anchor = resolved[0].0.to_string();
    let text = display_ref(&cmd, &resolved);
    el.attributes.push(("resolved".into(), text));
    el.attributes.push(("anchor".into(), anchor));
}

/// Build the visible text for a reference given its command and resolved targets.
fn display_ref(cmd: &str, resolved: &[(&str, &Target)]) -> String {
    let numbers: Vec<&str> = resolved.iter().map(|(_, t)| t.number.as_str()).collect();
    let typed = matches!(cmd, "cref" | "Cref" | "autoref");

    match cmd {
        // \eqref always parenthesizes and never names a type.
        "eqref" => format!("({})", join_and(&numbers)),
        _ if typed => {
            let kinds: Vec<&str> = resolved.iter().map(|(_, t)| t.kind.as_str()).collect();
            let all_same = kinds.windows(2).all(|w| w[0] == w[1]);
            if numbers.len() == 1 {
                format!("{} {}", capitalize(kinds[0]), numbers[0])
            } else if all_same {
                // Same kind: group under the plural ("Figures 1 and 2").
                format!(
                    "{} {}",
                    capitalize(&pluralize(kinds[0])),
                    join_and(&numbers)
                )
            } else {
                // Mixed kinds: name each ("Figure 1 and Table 2").
                let parts: Vec<String> = resolved
                    .iter()
                    .map(|(_, t)| format!("{} {}", capitalize(&t.kind), t.number))
                    .collect();
                let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
                join_and(&refs)
            }
        }
        // \ref / \pageref / \vref: bare number(s).
        _ => join_and(&numbers),
    }
}

// --- small helpers ----------------------------------------------------------

/// Join numbers as an English list: "1", "1 and 2", "1, 2 and 3".
fn join_and(items: &[&str]) -> String {
    match items {
        [] => String::new(),
        [a] => a.to_string(),
        [a, b] => format!("{a} and {b}"),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

fn pluralize(kind: &str) -> String {
    format!("{kind}s")
}

/// Extract the id of every `\label{...}` in a raw math string, in order.
fn math_labels(tex: &str) -> Vec<String> {
    let chars: Vec<char> = tex.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '\\' {
            i += 1;
            continue;
        }
        let start = i + 1;
        let mut j = start;
        while j < chars.len() && chars[j].is_ascii_alphabetic() {
            j += 1;
        }
        if chars[start..j].iter().collect::<String>() == "label" {
            let mut k = j;
            while k < chars.len() && chars[k].is_whitespace() {
                k += 1;
            }
            if k < chars.len() && chars[k] == '{' {
                let gstart = k + 1;
                let mut depth = 0;
                while k < chars.len() {
                    match chars[k] {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    k += 1;
                }
                let id: String = chars[gstart..k].iter().collect();
                if !id.trim().is_empty() {
                    out.push(id.trim().to_string());
                }
                i = k + 1;
                continue;
            }
        }
        i = j.max(i + 1);
    }
    out
}

fn explicitly_numbers_equation(tex: &str) -> bool {
    tex.contains("\\addtocounter {equation}{1}")
        || tex.contains("\\addtocounter{equation}{1}")
        || tex.contains("\\refstepcounter {equation}")
        || tex.contains("\\refstepcounter{equation}")
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

fn attr<'a>(el: &'a Element, key: &str) -> Option<&'a str> {
    el.attributes
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

#[cfg(test)]
mod tests {
    use crate::latex_to_markdown;

    #[test]
    fn cross_references_resolve_to_numbers() {
        let src = r#"
\begin{document}
\section{Intro}\label{sec:intro}
See \cref{sec:method} and \cref{fig:plot}.
\section{Method}\label{sec:method}
\begin{figure}
\caption{A plot.}\label{fig:plot}
\end{figure}
\end{document}
"#;
        let md = latex_to_markdown(src);
        assert!(md.contains("Section 2"), "cref to section 2 missing:\n{md}");
        assert!(md.contains("Figure 1"), "cref to figure 1 missing:\n{md}");
        // The label anchor is emitted so the link lands.
        assert!(
            md.contains("<a id=\"sec:method\">"),
            "anchor missing:\n{md}"
        );
    }

    #[test]
    fn theorems_number_with_shared_counter() {
        let src = r#"
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\begin{document}
\begin{theorem}\label{thm:a}First.\end{theorem}
\begin{lemma}\label{lem:b}Second.\end{lemma}
We cite \cref{thm:a} and \cref{lem:b}.
\end{document}
"#;
        let md = latex_to_markdown(src);
        assert!(
            md.contains("**Theorem 1.**"),
            "theorem 1 prefix missing:\n{md}"
        );
        assert!(md.contains("**Lemma 2.**"), "shared counter missing:\n{md}");
        assert!(
            md.contains("Theorem 1") && md.contains("Lemma 2"),
            "refs wrong:\n{md}"
        );
    }

    #[test]
    fn side_by_side_panels_number_and_label_independently() {
        // One figure float with two \caption commands (side-by-side minipages):
        // each caption steps the figure counter, and each \label binds to its
        // own panel's number — not both to the first.
        let src = r#"
\begin{document}
\begin{figure}
\includegraphics{a.png}\caption{First.}\label{fig:a}
\includegraphics{b.png}\caption{Second.}\label{fig:b}
\end{figure}
See \ref{fig:a} and \ref{fig:b}.
\end{document}
"#;
        let md = latex_to_markdown(src);
        // Each image is paired with the caption from its OWN panel (the second no
        // longer inherits the first caption's text as alt).
        assert!(md.contains("![First.](a.png)"), "panel 1 image alt:\n{md}");
        assert!(
            md.contains("![Second.](b.png)"),
            "panel 2 image alt (not 'First.'):\n{md}"
        );
        // Each label resolves to its own panel's number — two captions, two numbers.
        assert!(md.contains("[1](#fig:a)"), "fig:a -> 1:\n{md}");
        assert!(md.contains("[2](#fig:b)"), "fig:b -> 2 (not 1):\n{md}");
    }
}
