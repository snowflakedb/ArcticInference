//! TeX-math string post-processing for browser math renderers (KaTeX).
//!
//! The document tree stores raw, macro-expanded TeX in `<math>` nodes so it
//! stays format-neutral. Backends that target a KaTeX-style renderer (the
//! Markdown and HTML serializers both do) call [`katex`] to turn that raw TeX
//! into something KaTeX accepts: cross-reference/numbering metadata stripped,
//! the few unsupported font commands rewritten, and alignment environments
//! wrapped in `aligned`/`gathered`. It is deliberately NOT in the engine — this
//! is a rendering concern, not a parsing one — and NOT tied to one backend.

/// Prepare raw math TeX for a KaTeX renderer. `env` is the display-math
/// environment name (`align`, `equation`, ...) or `None` for inline / plain
/// display math. Returns the body without `$…$`/`$$…$$` delimiters.
pub(crate) fn katex(tex: &str, env: Option<&str>) -> String {
    let cleaned = clean(tex);
    match env {
        Some(env) => wrap_alignment(env, cleaned.trim()),
        None => cleaned.trim().to_string(),
    }
}

/// Strip `\label{...}`/`\nonumber`/`\notag` and rewrite KaTeX-invalid font
/// commands (`\textsc`→`\text`, `\textsl`→`\textit`).
fn clean(tex: &str) -> String {
    let chars: Vec<char> = tex.chars().collect();
    let mut out = String::with_capacity(tex.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '\\' {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        let start = i + 1;
        let mut j = start;
        while j < chars.len() && chars[j].is_ascii_alphabetic() {
            j += 1;
        }
        match chars[start..j].iter().collect::<String>().as_str() {
            "label" => i = skip_balanced_group(&chars, j),
            "nonumber" | "notag" => i = j,
            // KaTeX has no `\ensuremath`; it reaches math bodies via expanded
            // macros (`\newcommand{\GeV}{\ensuremath{...}}` used inside `$…$`).
            // Drop the control word and keep its `{...}` group (a no-op in math).
            "ensuremath" => i = j,
            // A `<math>` node is math mode by construction, so evaluate the mode
            // conditionals that reach it via macro expansion (unit macros like
            // `\def\GeV{\ifmmode {\mathrm{...}}\else \textrm{...}\fi}`). Keep the
            // branch that holds in math and clean it recursively; KaTeX cannot
            // parse `\ifmmode`/`\else`/`\fi`.
            "ifmmode" => {
                let (branch, next) = take_branch(&chars, j, true);
                out.push_str(&clean(&branch));
                i = next;
            }
            "ifhmode" | "ifvmode" => {
                let (branch, next) = take_branch(&chars, j, false);
                out.push_str(&clean(&branch));
                i = next;
            }
            // `\mbox`/`\hbox` inside math usually wrap `\ensuremath`-forced math;
            // KaTeX has no `\mbox`, so drop the word and keep the group so the
            // content renders as math.
            "mbox" | "hbox" => i = j,
            // `\protect` (robustness marker) and `\relax` (a no-op boundary) carry
            // no math meaning but reach here inside macro-expanded unit bodies
            // (e.g. `\DeclareRobustCommand`-defined units emit `\protect\ifmmode…`).
            // KaTeX rejects both, so drop the control word and keep going.
            "protect" | "relax" => i = j,
            "textsc" => {
                out.push_str("\\text");
                i = j;
            }
            "textsl" => {
                out.push_str("\\textit");
                i = j;
            }
            // KaTeX has no `\bm` (bold math, from the `bm` package) or `\pmb`
            // (poor-man's bold); both map to the supported `\boldsymbol`. Rewrite
            // the control word and keep the following `{...}` argument.
            "bm" | "pmb" => {
                out.push_str("\\boldsymbol");
                i = j;
            }
            // Line/space-break commands carry no math meaning and KaTeX rejects
            // them; drop the word and an optional `[n]` (e.g. `\linebreak[1]`).
            "linebreak" | "nolinebreak" | "newline" => i = skip_optional_bracket(&chars, j),
            _ => {
                // Control symbol (non-letter) or ordinary word: copy the
                // backslash; the next iteration handles what follows.
                out.push('\\');
                i += 1;
            }
        }
    }
    out
}

/// From an index just past a control word, skip optional whitespace and a single
/// `[...]` optional argument if present; return the next index (unchanged if
/// there is no bracket). Used to consume e.g. the `[1]` of `\linebreak[1]`.
fn skip_optional_bracket(chars: &[char], from: usize) -> usize {
    let mut i = from;
    while i < chars.len() && chars[i].is_whitespace() {
        i += 1;
    }
    if i >= chars.len() || chars[i] != '[' {
        return from;
    }
    while i < chars.len() {
        let closed = chars[i] == ']';
        i += 1;
        if closed {
            return i;
        }
    }
    i
}

/// From an index at (optional whitespace then) `{`, return the index just past
/// the matching `}`; if there is no `{`, returns `from` unchanged.
fn skip_balanced_group(chars: &[char], from: usize) -> usize {
    let mut i = from;
    while i < chars.len() && chars[i].is_whitespace() {
        i += 1;
    }
    if i >= chars.len() || chars[i] != '{' {
        return from;
    }
    let mut depth = 0;
    while i < chars.len() {
        match chars[i] {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    i
}

/// Read the letters of a control word starting at `at` (the index just past a
/// `\`), returning the word and the index just past it. A control symbol (the
/// next char is non-alphabetic) yields an empty word.
fn read_cs_word(chars: &[char], at: usize) -> (String, usize) {
    let mut k = at;
    while k < chars.len() && chars[k].is_ascii_alphabetic() {
        k += 1;
    }
    (chars[at..k].iter().collect(), k)
}

/// Select one branch of a mode conditional whose control word ends at `from`.
/// `keep_true` takes the true branch (from `from` up to the depth-1 `\else`, or
/// the depth-0 `\fi` if there is none); otherwise the else branch (after `\else`
/// up to `\fi`, empty if there is no `\else`). Returns the branch substring and
/// the index just past the matching `\fi`. Nested `\if…`/`\fi` are tracked; a
/// missing `\fi` degrades gracefully (consumes to end).
fn take_branch(chars: &[char], from: usize, keep_true: bool) -> (String, usize) {
    let mut i = from;
    let mut depth = 1usize;
    let mut else_start: Option<usize> = None; // `\` of the depth-1 `\else`
    let mut else_end: Option<usize> = None; // index just past that `\else`
    while i < chars.len() {
        if chars[i] == '\\' {
            let (word, next) = read_cs_word(chars, i + 1);
            if crate::engine::is_conditional_cs(&word) {
                depth += 1;
            } else if word == "fi" {
                depth -= 1;
                if depth == 0 {
                    let branch = select(chars, from, else_start, else_end, i, keep_true);
                    return (branch, next);
                }
            } else if word == "else" && depth == 1 && else_start.is_none() {
                else_start = Some(i);
                else_end = Some(next);
            }
            i = next.max(i + 1);
        } else {
            i += 1;
        }
    }
    let branch = select(chars, from, else_start, else_end, chars.len(), keep_true);
    (branch, chars.len())
}

/// Extract the chosen branch text given the conditional's boundaries.
fn select(
    chars: &[char],
    from: usize,
    else_start: Option<usize>,
    else_end: Option<usize>,
    fi: usize,
    keep_true: bool,
) -> String {
    if keep_true {
        chars[from..else_start.unwrap_or(fi)].iter().collect()
    } else {
        match else_end {
            Some(s) => chars[s..fi].iter().collect(),
            None => String::new(),
        }
    }
}

/// Wrap an alignment-based math environment's body in the KaTeX-compatible
/// `aligned`/`gathered` sub-environment so `&` and `\\` render inside `$$...$$`.
fn wrap_alignment(env: &str, body: &str) -> String {
    let inner = match env.trim_end_matches('*') {
        "align" | "aligned" | "eqnarray" | "flalign" | "split" | "multline" => Some("aligned"),
        "gather" => Some("gathered"),
        _ => None,
    };
    match inner {
        Some(kind) => format!("\\begin{{{kind}}}\n{body}\n\\end{{{kind}}}"),
        None => body.to_string(),
    }
}
