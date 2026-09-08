//! Robustness: the engine must always terminate and must never hide it when it
//! had to truncate. LaTeX macros are Turing-complete, so we cannot *detect*
//! non-termination — instead the engine bounds its work and reports degradation
//! via [`Diagnostics`]. These tests pin both halves of that contract.

use texmark::engine::Engine;
use texmark::state::{NoFiles, State};

/// Parse a self-contained document, returning `(element_count, truncated)`.
/// The mere fact that this returns proves termination — a broken guard would
/// hang the test process.
fn convert(body: &str) -> (usize, bool) {
    let src = format!("\\begin{{document}}{body}\\end{{document}}");
    let mut state = State::new();
    let (tree, diag) = Engine::new(&src, &mut state, &NoFiles).parse();
    (count(&tree), diag.expansion_limit_exceeded)
}

fn count(e: &texmark::node::Element) -> usize {
    1 + e
        .children
        .iter()
        .map(|n| match n {
            texmark::node::Node::Element(c) => count(c),
            texmark::node::Node::Text(_) => 0,
        })
        .sum::<usize>()
}

#[test]
fn pathological_macros_terminate_and_report_truncation() {
    // Each of these would loop forever without a bound. They must (a) return,
    // and (b) set the truncation flag so no caller mistakes the output for whole.
    let cases = [
        "\\def\\x{\\x}\\x",              // direct self-reference
        "\\def\\a{\\b}\\def\\b{\\a}\\a", // mutual recursion
        "\\newcommand{\\r}{\\r\\r}\\r",  // exponential fan-out
        "\\def\\g{10pt\\g}\\g",          // growth divergence (emits + recurses)
    ];
    for body in cases {
        let (_n, truncated) = convert(body);
        assert!(
            truncated,
            "expected truncation to be reported for: {body:?}"
        );
    }
}

#[test]
fn self_referential_input_terminates_and_reports_truncation() {
    // A file that `\input`s itself would splice source forever without a bound.
    // The parse must (a) return and (b) flag truncation.
    struct SelfInput;
    impl texmark::state::Resolver for SelfInput {
        fn resolve(&self, name: &str) -> Option<String> {
            (name == "loop").then(|| "x \\input{loop}".to_string())
        }
    }
    let src = "\\begin{document}\\input{loop}\\end{document}";
    let mut state = State::new();
    let (_tree, diag) = Engine::new(src, &mut state, &SelfInput).parse();
    assert!(
        diag.expansion_limit_exceeded,
        "self-referential \\input must be bounded and reported as truncated"
    );
}

#[test]
fn well_formed_input_reports_clean() {
    let (_n, truncated) = convert("\\newcommand{\\hi}{hello}\\hi\\ world. $a+b$");
    assert!(!truncated, "clean input must not be flagged as truncated");
}

#[test]
fn bounded_recursion_is_not_truncated() {
    // A macro that recurses but is only invoked a fixed number of times is a
    // cycle in the call graph yet terminates — it must NOT be flagged.
    let body = "\\newcommand{\\word}{word }\\word\\word\\word\\word\\word";
    let (_n, truncated) = convert(body);
    assert!(!truncated, "finite repeated use must stay clean");
}

#[test]
fn let_def_alias_cycle_does_not_blank_the_rest_of_the_document() {
    // The classic "save and redefine" idiom: `\let` snapshots \eta's current
    // (native) meaning into \etaa, then \eta is redefined to render that snapshot
    // in math. In TeX this terminates because \let copies the meaning at that
    // instant. A lazy `\etaa`→`\eta` alias instead loops \eta→\etaa→\eta forever,
    // exhausting the *global* expansion budget — which used to make every LATER
    // macro expansion in the document silently return empty. The higher-priority
    // guarantee is that a runaway in ONE construct must not blank the rest: the
    // sentinel after `$\eta$` must still render.
    let src = r"\begin{document}
\let\etaa=\eta
\def\eta{\ensuremath{\etaa}}
Value is $\eta$ here.

\section{Later Section}
\newcommand{\sentinel}{SENTINELWORD}
Body \sentinel end.
\end{document}";
    let md = texmark::latex_to_markdown(src);
    // Goal #1: content AFTER the offending construct is not blanked.
    assert!(
        md.contains("## 1 Later Section"),
        "section heading after a macro cycle must still render: {md}"
    );
    assert!(
        md.contains("SENTINELWORD"),
        "a macro used after a runaway must still expand (not silently blank): {md}"
    );
    // Goal #2: with a faithful `\let` snapshot the cycle never forms, so \eta
    // renders as itself (η in KaTeX) rather than exploding into runaway braces.
    assert!(
        md.contains(r"\eta") && !md.contains("{ { {"),
        "aliased \\eta should render as itself, not a runaway of nested braces: {md}"
    );
}

#[test]
fn direct_macro_cycle_still_reports_truncation_and_continues() {
    // A genuinely non-terminating self-reference must still be bounded AND
    // reported as truncated — the depth cap contains it without silently
    // dropping the guarantee, and the document after it still parses.
    let src = "\\begin{document}\\def\\x{\\x}\\x \\section{After}\\newcommand{\\q}{QWORD}\\q\\end{document}";
    let mut state = State::new();
    let (_tree, diag) = Engine::new(src, &mut state, &NoFiles).parse();
    assert!(
        diag.expansion_limit_exceeded,
        "a true non-terminating cycle must be reported as truncated"
    );
    let md = texmark::latex_to_markdown(src);
    assert!(
        md.contains("## 1 After") && md.contains("QWORD"),
        "the document after a contained runaway must still render: {md}"
    );
}

#[test]
fn cyclic_let_aliases_terminate_without_crashing() {
    // A `\let` to an as-yet-undefined control sequence records an alias to that
    // target's meaning. A self- or mutually-referential alias must NOT recurse
    // natively into a stack overflow (SIGABRT) when the alias is later used — it
    // must terminate and let the rest of the document survive. These are
    // plausible in real papers (`\let\x=\x` is a common no-op guard).
    let cases = [
        // self-alias to an undefined cs
        r"\begin{document}\let\zz=\zz \zz Tail SENTINELA.\end{document}",
        // mutual pair
        r"\begin{document}\let\aa=\bb\let\bb=\aa \aa\bb Tail SENTINELB.\end{document}",
        // three-cycle
        r"\begin{document}\let\pp=\qq\let\qq=\rr\let\rr=\pp \pp Tail SENTINELC.\end{document}",
    ];
    let sentinels = ["SENTINELA", "SENTINELB", "SENTINELC"];
    for (src, sentinel) in cases.iter().zip(sentinels) {
        // Reaching this assertion at all proves termination (a crash would abort
        // the whole test process).
        let md = texmark::latex_to_markdown(src);
        assert!(
            md.contains(sentinel),
            "a cyclic \\let alias must terminate and preserve later content: {md}"
        );
    }
}

#[test]
fn random_inputs_always_terminate() {
    // Property: arbitrary byte soup (heavy on TeX-significant characters) never
    // hangs, panics, or explodes the tree. Deterministic LCG so failures repro.
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed >> 33
    };
    // A mix of raw TeX-significant bytes and whole control words / environments,
    // so the fuzz exercises real dispatch paths (tables, math, defs, includes,
    // accents) in malformed states — not just character soup.
    let atoms: &[&str] = &[
        "\\",
        "{",
        "}",
        "[",
        "]",
        "#",
        "$",
        "%",
        "^",
        "_",
        "&",
        "~",
        " ",
        "\n",
        "\t",
        "a",
        "x",
        "0",
        "\u{00e9}",
        "\u{4e2d}",
        "\\begin{tabular}{c|c}",
        "\\end{tabular}",
        "\\multicolumn{2}{c}{",
        "\\begin{document}",
        "\\end{document}",
        "\\def\\x{",
        "\\x",
        "\\~",
        "\\'",
        "\\c{",
        "\\input{x}",
        "\\newcolumntype{x}",
        "\\begin{itemize}",
        "\\item",
        "$$",
        "\\[",
        "\\csname",
        "\\endcsname",
        "\\if",
        "\\fi",
        "\\\\",
        "\\verb|",
        "\\ref{",
    ];
    for _ in 0..1000 {
        let len = (next() % 400) as usize;
        let mut s = String::new();
        for _ in 0..len {
            s.push_str(atoms[(next() as usize) % atoms.len()]);
        }
        let mut state = State::new();
        // Returning at all is the property under test (termination + no panic).
        let (tree, _diag) = Engine::new(&s, &mut state, &NoFiles).parse();
        assert!(count(&tree) < 10_000_000, "tree size must stay bounded");
    }
}

#[test]
fn full_byte_soup_never_panics() {
    // Every byte value, in random order — catches indexing/UTF-8/slicing panics
    // on genuinely arbitrary (non-LaTeX, non-UTF8-shaped) input.
    let mut seed: u64 = 0x1234_5678_9ABC_DEF0;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    for _ in 0..300 {
        let len = (next() % 800) as usize;
        // Build from arbitrary chars (incl. control chars and non-ASCII scalars).
        let s: String = (0..len)
            .map(|_| char::from_u32(next() % 0x2FFF).unwrap_or('?'))
            .collect();
        let mut state = State::new();
        let (tree, _diag) = Engine::new(&s, &mut state, &NoFiles).parse();
        assert!(count(&tree) < 10_000_000, "tree size must stay bounded");
    }
}
