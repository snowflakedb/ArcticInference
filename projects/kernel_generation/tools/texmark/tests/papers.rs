//! End-to-end tests over small documents and the reference papers.
//!
//! The paper tests are smoke tests: they assert the engine consumes the whole
//! project without panicking and produces a non-trivial tree with the expected
//! top-level structure. Fine-grained behavior is covered by unit tests in the
//! library modules.

use std::path::{Path, PathBuf};
use texmark::engine::Engine;
use texmark::node::Node;
use texmark::state::{Resolver, State};
use texmark::{
    backend::md, latex_to_html, latex_to_markdown, latex_to_tree, latex_to_tree_with, latex_to_xml,
};

#[test]
fn algorithmicx_lowercase_keywords_are_structured() {
    // Modern algorithmicx uses mixed-case keywords (\State/\For/\EndFor), not the
    // old ALL-CAPS \STATE/\FOR. Both must be recognized and rendered readably.
    let md = latex_to_markdown(
        "\\begin{document}\\begin{algorithmic}\n\\State $x \\gets 1$\n\\For{$i$}\n\\State step\n\\EndFor\n\\end{algorithmic}\\end{document}",
    );
    assert!(md.contains("$x \\gets 1$"), "\\State content kept: {md}");
    assert!(md.contains("For $i$"), "\\For condition kept: {md}");
    assert!(md.contains("End for"), "\\EndFor recognized: {md}");
    assert!(
        !md.contains("\\State") && !md.contains("\\For"),
        "keywords must be stripped: {md}"
    );
}

#[test]
fn biblatex_cites_and_printbibliography() {
    // \printbibliography must not swallow \end{document}; biblatex cite commands
    // map to the natbib author-year forms.
    let src = r#"\usepackage{natbib}
\begin{document}\textcite{a} then \parencite{a}. Tail.
\begin{thebibliography}{9}\bibitem[Alpha(2001)]{a} Al. 2001.\end{thebibliography}
\printbibliography
\end{document}"#;
    let md = latex_to_markdown(src);
    assert!(
        md.contains("Alpha (2001) then (Alpha, 2001)."),
        "biblatex cite mapping: {md}"
    );
    assert!(md.contains("Tail."), "body after cites lost: {md}");
    assert!(
        !md.contains("document"),
        "printbibliography must not leak end-document: {md}"
    );
}

#[test]
fn typographic_ligatures_in_prose_only() {
    let md = latex_to_markdown(
        "\\begin{document}He said ``hi'' -- to 1--10, end---stop, don't. \\verb|a--b `c'| $x-y$\\end{document}",
    );
    assert!(md.contains("\u{201C}hi\u{201D}"), "double quotes: {md}");
    assert!(md.contains("\u{2013} to 1\u{2013}10"), "en dash: {md}");
    assert!(md.contains("end\u{2014}stop"), "em dash: {md}");
    assert!(md.contains("don\u{2019}t"), "apostrophe: {md}");
    // Not applied inside \verb code or math:
    assert!(md.contains("`a--b `c'`"), "code must stay literal: {md}");
    assert!(md.contains("$x-y$"), "math must stay literal: {md}");
    // \url is verbatim: its visible text must keep `--`, not become an en-dash.
    let u = latex_to_markdown(r"\begin{document}\url{http://x.io/a--b}\end{document}");
    assert!(
        u.contains("[http://x.io/a--b](http://x.io/a--b)"),
        "url must stay literal: {u}"
    );
}

#[test]
fn presentation_wrappers_render_only_their_content() {
    // Color/box/scale wrappers: the presentation arg must not leak as text, and
    // the content must survive (\scalebox previously ate its content).
    let md = latex_to_markdown(
        r"\begin{document}\textcolor{blue}{A} \colorbox{red}{B} \fcolorbox{k}{w}{C} \raisebox{2pt}{D} \parbox{3cm}{E} \scalebox{0.5}{F} \makebox[2cm]{G} \hyperlink{t}{H} x\rule{1cm}{1pt}y\end{document}",
    );
    for (frag, name) in [
        ("A", "textcolor"),
        ("B", "colorbox"),
        ("C", "fcolorbox"),
        ("D", "raisebox"),
        ("E", "parbox"),
        ("F", "scalebox"),
        ("G", "makebox"),
        ("H", "hyperlink"),
    ] {
        assert!(md.contains(frag), "{name} content lost: {md}");
    }
    assert!(
        !md.contains("blue") && !md.contains("red") && !md.contains("3cm"),
        "presentation arg leaked: {md}"
    );
    assert!(md.contains("xy"), "rule should produce no content: {md}");
}

#[test]
fn inline_verb_reads_raw() {
    // \verb<d>...<d> is verbatim: special chars survive, no tokenizing.
    let md = latex_to_markdown(
        r"\begin{document}code \verb|a_b->c{d}| end and \verb#x|y#.\end{document}",
    );
    assert!(
        md.contains("`a_b->c{d}`"),
        "verb should preserve _ -> {{}}: {md}"
    );
    assert!(
        md.contains("`x|y`"),
        "verb with # delimiter should keep |: {md}"
    );
    // \verb* shows spaces; HTML escapes verbatim metacharacters.
    let html = latex_to_html(r"\begin{document}\verb|p<q>&r|\end{document}");
    assert!(
        html.contains("<code>p&lt;q&gt;&amp;r</code>"),
        "verb html escaping: {html}"
    );
    // `%` is a literal delimiter, not a comment; macro-expanded \verb is captured.
    let md2 = latex_to_markdown(
        r"\newcommand{\v}{\verb|m_n|}\begin{document}\verb%a b%c and \v.\end{document}",
    );
    assert!(
        md2.contains("`a b`c"),
        "%-delimiter must not comment out the line: {md2}"
    );
    assert!(
        md2.contains("`m_n`"),
        "macro-expanded \\verb must be captured, not leaked: {md2}"
    );
}

#[test]
fn unknown_commands_are_reported_not_silently_dropped() {
    // "Never silently fail": an unrecognized command is dropped but recorded in
    // diagnostics; deliberate no-ops (spacing, table rules) are not.
    let src = "\\begin{document}A \\foobar{x} and \\untranslatable, \\hline \\vspace{1pt}.\\end{document}";
    let mut state = State::new();
    let (_tree, diag) = Engine::new(src, &mut state, &texmark::state::NoFiles).parse();
    assert!(
        diag.dropped_commands.contains("foobar"),
        "foobar not reported: {:?}",
        diag.dropped_commands
    );
    assert!(
        diag.dropped_commands.contains("untranslatable"),
        "untranslatable not reported"
    );
    assert!(
        !diag.dropped_commands.contains("hline"),
        "table rule should be a silent no-op"
    );
    assert!(
        !diag.dropped_commands.contains("vspace"),
        "spacing should be a silent no-op"
    );
    assert!(
        !diag.is_clean(),
        "dropped commands must make the run non-clean"
    );
}

#[test]
fn multiple_authors_render_on_one_line() {
    // authblk uses one \author per author; the backends join them into a single
    // list rather than stacking one per line.
    let src = r#"\usepackage{authblk}
\author{Ada Lovelace}
\author{Alan Turing}
\author{Grace Hopper}
\begin{document}\maketitle Body.\end{document}"#;
    let md = latex_to_markdown(src);
    assert!(
        md.contains("*Ada Lovelace, Alan Turing and Grace Hopper*"),
        "authors should be one comma-joined line: {md}"
    );
    let html = latex_to_html(src);
    assert!(
        html.contains("Ada Lovelace, Alan Turing and Grace Hopper"),
        "html author line: {html}"
    );
}

#[test]
fn book_class_numbers_chapters_and_scopes_sections() {
    // The document class drives numbering: book/report number \chapter as the
    // top unit, and sections read "chapter.section" and reset each chapter.
    let src = r#"\documentclass{book}
\begin{document}
\chapter{Intro}\label{ch:i}
\section{Background}\label{sec:b}
\chapter{Method}\label{ch:m}
\section{Setup}\label{sec:s}
Refs: \cref{ch:i}, \cref{sec:b}, \cref{ch:m}, \cref{sec:s}.
\end{document}"#;
    let md = latex_to_markdown(src);
    assert!(md.contains("[Chapter 1](#ch:i)"), "chapter 1 ref: {md}");
    assert!(
        md.contains("[Section 1.1](#sec:b)"),
        "section scoped to chapter: {md}"
    );
    assert!(md.contains("[Chapter 2](#ch:m)"), "chapter 2 ref: {md}");
    assert!(
        md.contains("[Section 2.1](#sec:s)"),
        "section resets per chapter: {md}"
    );
}

#[test]
fn article_class_numbers_sections_at_top() {
    // Default/article: \section is the top numbered unit ("1"), \subsection "1.1".
    let src = r#"\documentclass{article}
\begin{document}
\section{One}\label{a}\subsection{Sub}\label{b}
Refs: \cref{a} and \cref{b}.
\end{document}"#;
    let md = latex_to_markdown(src);
    assert!(md.contains("## 1 One"), "visible section number: {md}");
    assert!(
        md.contains("### 1.1 Sub"),
        "visible subsection number: {md}"
    );
    assert!(md.contains("[Section 1](#a)"), "top section is 1: {md}");
    assert!(md.contains("[Section 1.1](#b)"), "subsection is 1.1: {md}");
}

#[test]
fn appendix_does_not_change_float_numbering() {
    let markdown = texmark::latex_to_markdown(
        r"\begin{document}\section{Main}
          \begin{algorithm}\caption{Main algorithm}\end{algorithm}
          \appendix\section{Details}
          \begin{algorithm}\caption{Appendix algorithm}\end{algorithm}
          \end{document}",
    );
    assert!(markdown.contains("## A Details"), "{markdown}");
    assert!(markdown.contains("Algorithm 2"), "{markdown}");
}

#[test]
fn unnumbered_section_does_not_mislabel_cross_refs() {
    // A section beyond secnumdepth (or starred) has no number; a `\label` in it
    // must not bind to the previous numbered section.
    let src = r#"\documentclass{book}
\begin{document}
\chapter{Intro}\label{ch:i}
\subsubsection{Deep}\label{d}
Refs: \cref{ch:i} then \cref{d}.
\end{document}"#;
    let md = latex_to_markdown(src);
    assert!(md.contains("[Chapter 1](#ch:i)"), "chapter ref: {md}");
    // The subsubsection is unnumbered → its ref stays an unresolved raw link,
    // NOT a misleading "Chapter 1".
    assert!(
        md.contains("[d](#d)"),
        "unnumbered ref should stay raw, not mislabel: {md}"
    );
}

#[test]
fn math_transforms_are_backend_specific_not_baked_into_tree() {
    // Neutrality proof: the engine stores raw TeX (with `\label`, unwrapped);
    // only the KaTeX-targeting backends transform it. XML sees the neutral tree.
    let src =
        r"\begin{document}\begin{align}a &= b \label{eq:x}\end{align} \cref{eq:x}\end{document}";
    let md = latex_to_markdown(src);
    let xml = latex_to_xml(src);

    // Markdown backend: alignment wrapped for KaTeX, `\label` stripped, numbered
    // + anchored, and the cross-ref resolved.
    assert!(
        md.contains("\\begin{aligned}"),
        "md should wrap alignment: {md}"
    );
    assert!(!md.contains("\\label"), "md should strip \\label: {md}");
    assert!(
        md.contains("<a id=\"eq:x\">"),
        "md should anchor the equation: {md}"
    );
    assert!(md.contains("Equation 1"), "cref should resolve: {md}");

    // Neutral tree (XML): raw TeX kept, NOT KaTeX-wrapped, `\label` preserved.
    assert!(
        !xml.contains("aligned"),
        "tree must not bake in KaTeX wrapping: {xml}"
    );
    assert!(
        xml.contains("\\label"),
        "tree should keep raw \\label: {xml}"
    );
    assert!(
        xml.contains("env=\"align\""),
        "math node should carry its env: {xml}"
    );
}

#[test]
fn display_math_with_nested_dollar_in_text_does_not_desync() {
    // A `$...$` sub-formula nested inside `\text{}` (or any group) within display
    // `$$...$$` must not be mistaken for the closing delimiter. Getting this wrong
    // inverts the math-shift state and swallows the rest of the document as raw
    // TeX. The closing delimiter is only recognized at brace depth 0.
    let src = r"\begin{document}Before.
$$
\text{if } x \ge 1 \quad \text{then $y_i$ holds in $\mathbb{R}^n$}
$$
After the math we keep parsing: \textbf{bold} works.\end{document}";
    let md = latex_to_markdown(src);
    // The full display body survives inside one math block...
    assert!(
        md.contains("y_i") && md.contains(r"\mathbb {R}^n"),
        "nested sub-formulas should stay in the math body: {md}"
    );
    // ...and prose AFTER the display is still parsed, not swallowed as raw TeX.
    assert!(
        md.contains("**bold**") && !md.contains(r"\textbf"),
        "content after display math must still transpile: {md}"
    );
}

#[test]
fn display_math_is_not_embedded_in_a_prose_paragraph() {
    let markdown =
        texmark::latex_to_markdown("\\begin{document}Before $$x = 1$$ after.\\end{document}");
    assert!(
        markdown.contains("Before\n\n$$\nx = 1\n$$\n\nafter."),
        "{markdown}"
    );
}

#[test]
fn label_anchors_are_slugged_and_match_their_refs() {
    // A `\label` with spaces (common in math papers, e.g. `\label{spectral norm}`)
    // must become a space-free fragment id, and every `\ref`/`\cref` to it must
    // slug identically so the link still lands. Existing `sec:`/`eq:` labels
    // (which already work) must be preserved unchanged.
    let md = latex_to_markdown(
        r"\begin{document}\section{Intro}\label{spectral norm}
See \cref{spectral norm}, \cref{sec:keep}.
\section{Keep}\label{sec:keep}\end{document}",
    );
    assert!(
        md.contains(r#"<a id="spectral-norm">"#) && md.contains("(#spectral-norm)"),
        "spaced label must slug consistently on anchor and ref: {md}"
    );
    assert!(
        !md.contains("spectral norm"),
        "no space should survive in ids/hrefs: {md}"
    );
    assert!(
        md.contains(r#"<a id="sec:keep">"#) && md.contains("(#sec:keep)"),
        "colon labels must be preserved: {md}"
    );

    // Same consistency in the HTML backend.
    let html = latex_to_html(r"\begin{document}\section{S}\label{a b} x \cref{a b}.\end{document}");
    assert!(
        html.contains(r#"<a id="a-b">"#) && html.contains(r##"href="#a-b""##),
        "html anchor/href must slug and match: {html}"
    );
}

#[test]
fn optional_and_style_args_do_not_leak_as_literal_text() {
    // A cluster of "argument leaks as prose" bugs found across the corpus:
    // \texorpdfstring emits only its TeX (first) arg, a \newtheorem env's
    // optional note becomes a parenthesized title (not literal `[...]`), and
    // \index produces no visible text.
    let tex = latex_to_markdown(
        r"\begin{document}\section{\texorpdfstring{$H\to\gamma$}{H to gamma}}\end{document}",
    );
    // \texorpdfstring keeps its TeX (first) arg and drops the pdf-string; the GFM
    // backend then lowers heading math to text (GitHub does not render `$…$`
    // inside an ATX heading), so `$H\to\gamma$` becomes the Unicode "H→ γ".
    assert!(
        tex.contains('→') && tex.contains('γ') && !tex.contains("H to gamma"),
        "texorpdfstring should keep the TeX form, lowered for the heading: {tex}"
    );

    let thm = latex_to_markdown(
        r"\newtheorem{thm}{Theorem}\begin{document}\begin{thm}[Pythagoras]$a^2=c^2$\end{thm}\end{document}",
    );
    assert!(
        thm.contains("**Theorem 1 (Pythagoras).**") && !thm.contains(r"\[Pythagoras"),
        "theorem optional note should become a parenthesized title: {thm}"
    );

    let idx = latex_to_markdown(r"\begin{document}Word\index{Word} here.\end{document}");
    assert!(
        idx.contains("Word here.") && !idx.contains("WordWord"),
        "\\index argument must not leak into prose: {idx}"
    );
}

#[test]
fn booktabs_specialrule_and_addlinespace_do_not_leak_into_cells() {
    // `\specialrule{h}{a}{b}` (three dimension args) and `\addlinespace[len]` are
    // booktabs rules that can precede a data row; their arguments must be stripped,
    // not flattened into the first cell (which produced e.g. "1pt-1pt0pt Model").
    let md = latex_to_markdown(
        r"\begin{document}\begin{tabular}{lc}\toprule A & B \\ \specialrule{1pt}{-1pt}{0pt} Model & 27.3 \\ \addlinespace Other & 5 \\ \bottomrule\end{tabular}\end{document}",
    );
    assert!(
        md.contains("| Model | 27.3 |") && md.contains("| Other | 5 |"),
        "rule dimensions must not leak into data cells: {md}"
    );
    assert!(
        !md.contains("1pt") && !md.contains("-1pt"),
        "no dimension text should survive: {md}"
    );
}

#[test]
fn newtheorem_within_counter_scopes_and_resets() {
    // `\newtheorem{thm}{Theorem}[section]` numbers theorems "section.n" and
    // resets each section. A theorem sharing that counter (`[thm]`) inherits the
    // scoping (within is a property of the counter, not the individual def), and
    // `\cref` resolves to the scoped number.
    let src = r"\newtheorem{thm}{Theorem}[section]
\newtheorem{lem}[thm]{Lemma}
\begin{document}
\section{First}
\begin{thm}a\end{thm}
\begin{lem}\label{l}b\end{lem}
\section{Second}
\begin{thm}c\end{thm}
See \cref{l}.
\end{document}";
    let md = latex_to_markdown(src);
    assert!(md.contains("**Theorem 1.1."), "first theorem scoped: {md}");
    assert!(
        md.contains("**Lemma 1.2."),
        "shared-counter lemma inherits within-scoping: {md}"
    );
    assert!(
        md.contains("**Theorem 2.1."),
        "counter resets in the next section: {md}"
    );
    assert!(
        md.contains("See [Lemma 1.2](#l)"),
        "cref resolves to the scoped number: {md}"
    );

    // Without [within], numbering stays flat and document-wide.
    let flat = latex_to_markdown(
        r"\newtheorem{thm}{Theorem}\begin{document}\section{A}\begin{thm}x\end{thm}\section{B}\begin{thm}y\end{thm}\end{document}",
    );
    assert!(
        flat.contains("**Theorem 1.") && flat.contains("**Theorem 2."),
        "no-within theorems stay flat: {flat}"
    );
}

#[test]
fn class_renewenvironment_does_not_shadow_native_floats() {
    // A document class often `\renewenvironment{table}{...\@float{table}...}`
    // to tweak spacing. That must NOT shadow the native float handler: doing so
    // leaked `table[t]` as text and, worse, left the caption un-floated so a
    // `\label` after it bound to the enclosing *section* number instead of the
    // table's. The caption is wrapped in `\begin{center}`, a common idiom.
    let src = r"\renewenvironment{table}{\par\@float{table}}{\end@float}
\begin{document}
\section{First}\section{Second}\section{Third}
\section{Results}
\begin{table}[t]
\begin{center}
\caption{Scores}
\label{tab:x}
\begin{tabular}{c}9\end{tabular}
\end{center}
\end{table}
See Table \ref{tab:x}.
\end{document}";
    let md = latex_to_markdown(src);
    assert!(
        !md.contains("table[t]") && !md.contains(r"table\[t\]"),
        "renewenvironment float internals must not leak: {md}"
    );
    assert!(md.contains("**Table 1:"), "float must be numbered: {md}");
    assert!(
        md.contains("See Table [1](#tab:x)"),
        "label must bind to the table number, not the section: {md}"
    );
}

#[test]
fn conditional_skip_cannot_swallow_the_document_body() {
    // A stray/orphaned `\else` in the preamble (e.g. from a `\newif`/`\ifx`
    // conditional in a bundled class/style that texmark can't evaluate) triggers
    // a skip-to-`\fi`. Without a boundary, that skip runs past `\begin{document}`
    // and eats the entire body (observed on the BERT paper: 7-line output). The
    // skip must stop at `\begin{document}` so the body always survives.
    let md = latex_to_markdown(
        r"\else \begin{document}\section{Intro}The body must survive.\end{document}",
    );
    assert!(
        md.contains("The body must survive.") && md.contains("Intro"),
        "conditional runaway must not swallow the document body: {md}"
    );
    // Same guard on the skip_to_else_or_fi path: a text-mode `\ifmmode` (false →
    // skip the true branch) with no `\else`/`\fi` before the body must stop at
    // `\begin{document}`, not eat it.
    let md2 = latex_to_markdown(r"\ifmmode \begin{document}Body kept here.\end{document}");
    assert!(
        md2.contains("Body kept here."),
        "else-or-fi skip must stop at body: {md2}"
    );
}

#[test]
fn ifx_selects_the_matching_macro_branch() {
    let markdown = texmark::latex_to_markdown(
        r"\def\a{x}\def\b{x}\def\c{y}\begin{document}
          \ifx\a\b same\else wrong\fi
          \ifx\a\c wrong\else different\fi
          \end{document}",
    );
    assert!(markdown.contains("same"), "{markdown}");
    assert!(markdown.contains("different"), "{markdown}");
    assert!(!markdown.contains("wrong"), "{markdown}");
}

#[test]
fn split_footnote_commands_form_one_gfm_footnote() {
    let markdown = texmark::latex_to_markdown(
        r"\author{Ada\footnotemark[1]}\begin{document}
          Text.\footnotetext[1]{Equal contribution.}
          \end{document}",
    );
    assert!(markdown.contains("*Ada[^1]*"), "{markdown}");
    assert!(markdown.contains("[^1]: Equal contribution."), "{markdown}");
}

#[test]
fn repeated_footnote_mark_reuses_the_previous_footnote() {
    let markdown = texmark::latex_to_markdown(
        r"\newcommand*\samethanks[1][\value{footnote}]{\footnotemark[#1]}
          \author{Ada\thanks{Equal contribution}, Grace\samethanks}
          \begin{document}Text.\end{document}",
    );
    assert!(markdown.contains("*Ada[^1], Grace[^1]*"), "{markdown}");
    assert_eq!(markdown.matches("[^1]: Equal contribution").count(), 1);
}

#[test]
fn explicit_footnote_number_can_reuse_an_ordinary_footnote() {
    let markdown = texmark::latex_to_markdown(
        r"\begin{document}A\footnote{Shared} B\footnotemark[1].\end{document}",
    );
    assert!(markdown.contains("A[^1] B[^1]"), "{markdown}");
    assert_eq!(markdown.matches("[^1]: Shared").count(), 1, "{markdown}");
}

#[test]
fn unpaired_footnote_text_does_not_replace_an_ordinary_note() {
    let markdown = texmark::latex_to_markdown(
        r"\begin{document}A\footnote{First}. Later\footnotetext{Second}.\end{document}",
    );
    assert!(markdown.contains("[^1]: First"), "{markdown}");
    assert!(markdown.contains("[^2]: Second"), "{markdown}");
}

#[test]
fn ifx_treats_two_undefined_control_sequences_as_equal() {
    let markdown = texmark::latex_to_markdown(
        r"\begin{document}\ifx\undefinedA\undefinedB yes\else no\fi\end{document}",
    );
    assert!(markdown.contains("yes"), "{markdown}");
    assert!(!markdown.contains("no"), "{markdown}");
}

#[test]
fn unmatched_footnote_text_remains_visible() {
    let markdown = texmark::latex_to_markdown(
        r"\begin{document}Text.\footnotetext[1]{Equal contribution.}\end{document}",
    );
    assert!(markdown.contains("Text.[^1]"), "{markdown}");
    assert!(markdown.contains("[^1]: Equal contribution."), "{markdown}");
}

#[test]
fn preamble_footnote_text_does_not_collide_with_document_footnotes() {
    let markdown = texmark::latex_to_markdown(
        r"\author{Ada*}\footnotetext[1]{Equal contribution.}
          \begin{document}Text\footnote{A document note.}\end{document}",
    );
    assert!(markdown.contains("[^1]: Equal contribution."), "{markdown}");
    assert!(markdown.contains("Text[^2]"), "{markdown}");
    assert!(markdown.contains("[^2]: A document note."), "{markdown}");
}

#[test]
fn let_aliases_a_user_macro() {
    // `\let\a=\b` snapshots \b's meaning, so an alias of a unit macro expands
    // identically and never leaks as a raw `\a`. (`=` optional.)
    let def = r"\def\GeV{\ifmmode {\mathrm{Ge\kern -0.1em V}}\else \textrm{GeV}\fi}";
    let eq = latex_to_markdown(&format!(
        r"{def}\let\gev=\GeV\begin{{document}}$5\gev$ and $8\GeV$.\end{{document}}"
    ));
    assert!(
        eq.contains(r"\mathrm")
            && eq.contains("Ge")
            && !eq.contains(r"\gev")
            && !eq.contains(r"\ifmmode"),
        "\\let alias should expand like its target: {eq}"
    );
    let noeq = latex_to_markdown(&format!(
        r"{def}\let\gev\GeV\begin{{document}}$5\gev$.\end{{document}}"
    ));
    assert!(
        noeq.contains(r"\mathrm") && !noeq.contains(r"\gev"),
        "\\let without =: {noeq}"
    );

    // `\let` to a non-user cs must not panic or leak the alias as literal text.
    let prim = latex_to_markdown(r"\let\x\relax\begin{document}a\x b\end{document}");
    assert!(
        prim.contains("a") && prim.contains("b") && !prim.contains(r"\x"),
        "{prim}"
    );
}

#[test]
fn ifmmode_resolves_by_context() {
    // In a math node `\ifmmode T\else E\fi` → T (math mode); no \ifmmode/\else/\fi
    // survives to KaTeX. Nested conditionals and a missing \else both work.
    let m = latex_to_markdown(
        r"\def\GeV{\ifmmode {\mathrm{Ge\kern -0.1em V}}\else \textrm{GeV}\fi}\begin{document}$5\GeV$\end{document}",
    );
    assert!(
        m.contains(r"\mathrm")
            && m.contains("Ge")
            && !m.contains(r"\ifmmode")
            && !m.contains(r"\else")
            && !m.contains(r"\fi"),
        "math \\ifmmode should keep the true branch only: {m}"
    );
    let nested =
        latex_to_markdown(r"\begin{document}$\ifmmode a\ifmmode b\fi c\else d\fi$\end{document}");
    assert!(
        nested.replace(' ', "").contains("abc") && !nested.contains('d'),
        "nested \\ifmmode: {nested}"
    );
    let noelse = latex_to_markdown(r"\begin{document}$\ifmmode x\fi$\end{document}");
    assert!(
        noelse.contains('x') && !noelse.contains(r"\fi"),
        "no-else \\ifmmode: {noelse}"
    );

    // `\iff` (⟺) is not a conditional and has no `\fi`; it must not be counted as
    // a nested opener (which would leak `\else`/`\fi` or eat following content).
    let iff = latex_to_markdown(r"\begin{document}$\ifmmode a \iff b\else c\fi$\end{document}");
    assert!(
        iff.contains(r"\iff")
            && !iff.contains(r"\else")
            && !iff.contains(r"\fi")
            && !iff.contains('c'),
        "\\iff must not disturb conditional nesting: {iff}"
    );

    // In text, `\ifmmode` is false → the else branch renders and no conditional
    // plumbing is reported as a dropped command.
    let src = r"\def\GeV{\ifmmode {\mathrm{GeV}}\else \textrm{GeV}\fi}\begin{document}The \GeV{} scale.\end{document}";
    let t = latex_to_markdown(src);
    assert!(
        t.contains("GeV") && !t.contains(r"\ifmmode"),
        "text \\ifmmode else branch: {t}"
    );
    let mut state = State::new();
    let (_tree, diag) = Engine::new(src, &mut state, &texmark::state::NoFiles).parse();
    for cmd in ["ifmmode", "fi", "else"] {
        assert!(
            !diag.dropped_commands.contains(cmd),
            "{cmd} must not be a dropped command"
        );
    }
}

#[test]
fn mbox_in_math_is_transparent() {
    // `\mbox{\ensuremath{..}}` (idiomatic in unit/result macros) used inside math
    // renders as math: both wrappers drop, keeping their groups.
    let md =
        latex_to_markdown(r"\begin{document}$\mbox{\ensuremath{126.0 \pm 0.4}}$.\end{document}");
    assert!(
        md.contains(r"126.0 \pm 0.4") && !md.contains(r"\mbox"),
        "mbox in math should be transparent: {md}"
    );
}

#[test]
fn robust_ifmmode_unit_macro_renders_in_math() {
    // Unit macros are commonly declared with `\DeclareRobustCommand` and their
    // bodies often carry `\protect`/`\relax` (robustness plumbing). Both the
    // definition form and the plumbing must be handled, or the control sequence
    // leaks raw into `$…$` and KaTeX fails to render the whole span. Sentinel
    // "GeV" has no LaTeX-special characters.
    let robust = latex_to_markdown(
        r"\DeclareRobustCommand{\GeV}{\ifmmode\mathrm{GeV}\else\text{GeV}\fi}\begin{document}$126.0 \GeV$ done.\end{document}",
    );
    // Renders as math with the unit intact, and no KaTeX-invalid leakage.
    assert!(
        robust.contains(r"\mathrm") && robust.contains("GeV"),
        "robust unit macro should render its unit in math: {robust}"
    );
    for leak in [
        r"\GeV",
        r"\ifmmode",
        r"\else",
        r"\fi",
        r"\DeclareRobustCommand",
    ] {
        assert!(!robust.contains(leak), "{leak} must not leak: {robust}");
    }

    // `\protect` / `\relax` inside a macro-expanded math body carry no math
    // meaning and are KaTeX-invalid; they must be stripped, and `\ifmmode` still
    // takes its true (math) branch around them.
    let protect = latex_to_markdown(
        r"\newcommand{\QeV}{\protect\ifmmode\relax\mathrm{QeV}\else\text{QeV}\fi}\begin{document}$5 \QeV$ done.\end{document}",
    );
    assert!(
        protect.contains(r"\mathrm") && protect.contains("QeV"),
        "protect/relax unit macro should render in math: {protect}"
    );
    for leak in [r"\protect", r"\relax", r"\ifmmode", r"\fi"] {
        assert!(!protect.contains(leak), "{leak} must not leak: {protect}");
    }
}

#[test]
fn ensuremath_forces_math_mode() {
    // \ensuremath{X} in text must become inline math, not prose — else math
    // commands in X are dropped (`\ensuremath{H\to\gamma}` → "H"). This is
    // pervasive via house-style macros, e.g.
    // \newcommand{\hgg}{\ensuremath{H\to\gamma\gamma}} used in running text.
    let md = latex_to_markdown(
        r"\newcommand{\hgg}{\ensuremath{H\to\gamma\gamma}}\begin{document}Decay \hgg\ and \ensuremath{\tau^+} seen.\end{document}",
    );
    assert!(
        md.contains(r"$H\to \gamma \gamma$") && md.contains(r"$\tau ^+$"),
        "ensuremath in text should become inline math: {md}"
    );
    // When such a macro is used INSIDE `$…$`, the literal `\ensuremath` reaches
    // the math body via expansion; KaTeX can't parse it, so it must be stripped.
    let inmath = latex_to_markdown(
        r"\newcommand{\hgg}{\ensuremath{H\to\gamma}}\begin{document}$\hgg$ done.\end{document}",
    );
    assert!(
        !inmath.contains(r"\ensuremath") && inmath.contains(r"H\to \gamma"),
        "KaTeX math body must not contain \\ensuremath: {inmath}"
    );
}

#[test]
fn html_backend_needs_no_core_changes() {
    // Adding a backend is purely additive: the HTML serializer reads the same
    // neutral tree the others do.
    let html = latex_to_html(r"\begin{document}\section{S}A \textbf{b} eq $x^2$.\end{document}");
    assert!(
        html.contains("<h2>S</h2>") && html.contains("<strong>b</strong>"),
        "{html}"
    );
    assert!(html.contains("\\(x^2\\)"), "inline math delim: {html}");
}

#[test]
fn html_escapes_math_metacharacters() {
    // `<`/`>`/`&` inside math must be HTML-escaped (the browser decodes them
    // before KaTeX reads the text); the `\(`/`\[` delimiters stay literal.
    let html =
        latex_to_html(r"\begin{document}$a < b$ and \begin{align}x &= y\end{align}\end{document}");
    assert!(
        html.contains("\\(a &lt; b\\)"),
        "inline `<` not escaped: {html}"
    );
    assert!(!html.contains("a < b"), "raw `<` corrupts HTML: {html}");
    assert!(
        html.contains("&amp;=") || html.contains("&amp; ="),
        "align `&` not escaped: {html}"
    );
    assert!(
        html.contains("\\["),
        "display delimiter must stay literal: {html}"
    );
}

#[test]
fn small_document_structure() {
    let src = r#"
\documentclass{article}
\title{Demo}
\author{Test}
\begin{document}
\maketitle
\section{Intro}
Hello \textbf{world}. Math $a+b$.
\begin{itemize}
\item one
\item two
\end{itemize}
\end{document}
"#;
    let tree = latex_to_tree(src);
    assert_eq!(tree.name, "document");
    let names: Vec<&str> = tree
        .children
        .iter()
        .filter_map(|n| match n {
            Node::Element(e) => Some(e.name.as_str()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"title"));
    assert!(names.contains(&"author"));
    assert!(names.contains(&"section"));
}

#[test]
fn toggle_selects_branch() {
    // With the toggle on, the true branch text should appear and not the false.
    let src = r#"
\newtoggle{arxiv}\toggletrue{arxiv}
\begin{document}
\iftoggle{arxiv}{YESVISIBLE}{NOHIDDEN}
\end{document}
"#;
    let xml = latex_to_xml(src);
    assert!(xml.contains("YESVISIBLE"), "true branch missing: {xml}");
    assert!(!xml.contains("NOHIDDEN"), "false branch leaked: {xml}");
}

/// Resolver that reads paper source files relative to a base directory.
struct PaperFiles {
    base: PathBuf,
}

impl Resolver for PaperFiles {
    fn resolve(&self, name: &str) -> Option<String> {
        for candidate in [self.base.join(name), self.base.join(format!("{name}.tex"))] {
            if let Ok(s) = std::fs::read_to_string(&candidate) {
                return Some(s);
            }
        }
        None
    }
}

/// Resolver serving a single in-memory `ref.bib` and no `.bbl`, to exercise the
/// native `.bib` → references fallback (the WASM / no-BibTeX path).
struct BibOnly {
    bib: String,
}

impl Resolver for BibOnly {
    fn resolve(&self, name: &str) -> Option<String> {
        (name == "ref.bib").then(|| self.bib.clone())
    }
    // resolve_bibliography defaults to None: there is no .bbl.
}

#[test]
fn makeatletter_include_avoids_at_macro_loop() {
    // A `\makeatletter` file defining `\@`-macros must tokenize each `\@foo` as
    // one control sequence; otherwise `\def\@onedot{...\@let@token...}` would
    // mis-define the control symbol `\@` into a self-referential loop.
    struct Defs;
    impl Resolver for Defs {
        fn resolve(&self, name: &str) -> Option<String> {
            (name == "defs" || name == "defs.tex").then(|| {
                "\\makeatletter\\def\\@onedot{\\@let@token.}\\def\\eg{e.g\\@onedot}\\makeatother"
                    .to_string()
            })
        }
    }
    let src = "\\begin{document}\\input{defs}For \\eg here.\\end{document}";
    let mut state = State::new();
    let (tree, diag) = Engine::new(src, &mut state, &Defs).parse();
    assert!(
        !diag.expansion_limit_exceeded,
        "\\makeatletter must prevent the \\@ mis-tokenization loop"
    );
    assert!(
        md::to_markdown(&tree).contains("e.g."),
        "\\eg should render"
    );
}

#[test]
fn nested_sty_input_keeps_at_letter() {
    // A package that `\input`s another `.sty` must keep `@` a letter for its own
    // code that follows the nested input. texmark brackets each `.sty` with an
    // implicit `\makeatletter … \makeatother`; a blind trailing `\makeatother`
    // from the *nested* include would reset `@` mid-package, mis-tokenizing every
    // later `\@foo` (e.g. eso-pic → cvpr_eso.sty, which then thrashed the
    // expansion budget and silently dropped the whole document body).
    struct Pkgs;
    impl Resolver for Pkgs {
        fn resolve(&self, name: &str) -> Option<String> {
            match name {
                "outer.sty" => Some(
                    "\\input{inner.sty}\\newcommand\\usething{\\@thing}\\def\\@thing{XYZZY}".into(),
                ),
                "inner.sty" => Some("\\def\\@inner{Y}".into()),
                _ => None,
            }
        }
    }
    let src = "\\usepackage{outer}\\begin{document}\\usething\\end{document}";
    let mut state = State::new();
    let (tree, _diag) = Engine::new(src, &mut state, &Pkgs).parse();
    let md = md::to_markdown(&tree);
    // `\@thing` (defined after the nested \input) expands to "XYZZY" only if `@`
    // stayed a letter; otherwise it mis-tokenizes as `\@`+`thing` → "thing".
    assert!(
        md.contains("XYZZY"),
        "@ must stay a letter after a nested .sty input: {md}"
    );
    assert!(
        !md.contains("thing"),
        "@ mis-tokenized \\@thing into a control symbol: {md}"
    );
}

#[test]
fn native_packages_do_not_load_bundled_implementations() {
    use std::cell::Cell;

    struct Pkgs(Cell<usize>);
    impl Resolver for Pkgs {
        fn resolve(&self, name: &str) -> Option<String> {
            if matches!(name, "natbib.sty" | "fancyhdr.sty") {
                self.0.set(self.0.get() + 1);
            }
            None
        }
    }

    let resolver = Pkgs(Cell::new(0));
    let mut state = State::new();
    let _ = Engine::new(
        "\\usepackage{natbib,fancyhdr}\\begin{document}Paper.\\end{document}",
        &mut state,
        &resolver,
    )
    .parse();
    assert_eq!(resolver.0.get(), 0);
}

#[test]
fn nested_macro_definitions_keep_parameters() {
    let markdown = latex_to_markdown(
        r"\newcommand{\outer}{\newcommand{\inner}[1]{path/##1}}\outer
           \begin{document}\inner{figure}\end{document}",
    );
    assert!(markdown.contains("path/figure"), "{markdown}");
}

#[test]
fn references_inside_math_are_resolved() {
    let markdown = latex_to_markdown(
        r"\begin{document}
           \begin{equation}x=1\label{eq:x}\end{equation}
           $\ref{eq:x} + \eqref{eq:x}$
           \end{document}",
    );
    assert!(markdown.contains("$1 + (1)$"), "{markdown}");
}

#[test]
fn references_inside_pseudocode_are_resolved() {
    let markdown = latex_to_markdown(
        r"\begin{document}
           \begin{equation}x=1\label{eq:x}\end{equation}
           \begin{algorithmic}\State Use equation \eqref{eq:x}.\end{algorithmic}
           \end{document}",
    );
    assert!(markdown.contains("Use equation (1)."), "{markdown}");
}

#[test]
fn citations_inside_pseudocode_are_resolved() {
    let markdown = latex_to_markdown(
        r"\begin{document}
           \begin{algorithmic}\State Update using \cite{paper}.\end{algorithmic}
           \begin{thebibliography}{9}\bibitem{paper}Reference.\end{thebibliography}
           \end{document}",
    );
    assert!(markdown.contains("Update using [1]."), "{markdown}");
}

#[test]
fn pseudocode_keeps_control_flow_and_line_labels() {
    let markdown = texmark::latex_to_markdown(
        r"\begin{document}
          See line \ref{loop:start}.
          \begin{algorithmic}
          \For{$1 \le i \le n$}\label{loop:start}
          \State work
          \EndFor
          \end{algorithmic}
          \end{document}",
    );
    assert!(markdown.contains("For $1 \\le i \\le n$"), "{markdown}");
    assert!(markdown.contains("End for"), "{markdown}");
    assert!(markdown.contains("<a id=\"loop:start\"></a>"), "{markdown}");
}

#[test]
fn mirrored_caption_keeps_its_label_anchor() {
    let markdown = texmark::latex_to_markdown(
        r"\begin{document}\begin{figure}
          \includegraphics{figure.png}
          \caption{A figure.\label{fig:inside}}
          \end{figure}
          See \ref{fig:inside}.
          \end{document}",
    );
    assert!(markdown.contains("<a id=\"fig:inside\"></a>"), "{markdown}");
}

#[test]
fn bibliography_layout_redefinitions_do_not_hide_bibitems() {
    let markdown = latex_to_markdown(
        r"\newenvironment{thebibliography}[1]{Old}{End}
           \begin{document}\cite{x}
           \begin{thebibliography}{9}\bibitem{x}Reference.\end{thebibliography}
           \end{document}",
    );
    assert!(markdown.contains("[1]"), "{markdown}");
    assert!(markdown.contains("Reference."), "{markdown}");
    assert!(!markdown.contains("Old"), "{markdown}");
}

#[test]
fn unbraced_package_inputs_read_the_full_filename_without_reporting_content_loss() {
    struct Pkgs;
    impl Resolver for Pkgs {
        fn resolve(&self, name: &str) -> Option<String> {
            (name == "outer.sty").then(|| "\\input missing-common.tex".to_string())
        }
    }

    let mut state = State::new();
    let (_, diagnostics) = Engine::new(
        "\\usepackage{outer}\\begin{document}Paper.\\end{document}",
        &mut state,
        &Pkgs,
    )
    .parse();
    assert!(diagnostics.unresolved_inputs.is_empty());
}

#[test]
fn edef_expands_body_at_definition_time() {
    // `\edef` freezes the CURRENT expansion of its body. A self-referential
    // `\edef\x{...\x...}` (the LaTeX option-processing idiom) must not store an
    // unexpanded `\x` inside its own body and then recurse forever when used.
    let md = latex_to_markdown("\\def\\x{}\\edef\\x{Q\\x W}\\begin{document}\\x\\end{document}");
    assert!(
        md.contains("QW"),
        "\\edef should expand its body at definition time: {md}"
    );
    assert!(
        !md.contains("QQ"),
        "self-referential \\edef must not recurse: {md}"
    );
}

#[test]
fn edef_body_expansion_does_not_spill_into_the_document() {
    // Definition-time expansion runs behind a read barrier: a trailing macro
    // whose body drops its argument (`\g` below) must grab end-of-input, not
    // reach past the body into the document and silently swallow it.
    let md = latex_to_markdown(
        "\\newcommand\\g[1]{G}\\edef\\x{\\g}\\begin{document}\\x BODYWORD tail.\\end{document}",
    );
    assert!(
        md.contains("BODYWORD tail."),
        "document body after \\edef must survive: {md}"
    );
    assert!(
        md.contains("G"),
        "\\edef body should still expand to G: {md}"
    );
}

#[test]
fn nested_tabular_in_cell_does_not_swallow_following_content() {
    // A nested `tabular` inside a cell has its own `\\` row breaks. Splitting the
    // outer table on those inner `\\` used to leave a cell holding an unclosed
    // `\begin{tabular}`, whose recursive parse then ran off the end of the input
    // and swallowed the rest of the document.
    let md = latex_to_markdown(
        "\\begin{document}\\begin{table}\\begin{tabular}{c|c}\
         \\begin{tabular}{c}A\\\\ B\\end{tabular} & C\\\\ \\end{tabular}\
         \\caption{cap}\\end{table}AFTERWARD\\end{document}",
    );
    assert!(
        md.contains("AFTERWARD"),
        "content after a nested-tabular table must survive: {md}"
    );
}

#[test]
fn multicolumn_pads_spanned_columns_to_keep_alignment() {
    // GFM has no colspan, so a grouped header's `\multicolumn{n}{..}{X}` must emit
    // X plus n-1 empty cells; otherwise the header is short and its labels drift
    // left, misaligning every data column (ResNet/word2vec/VGG multi-level heads).
    let md = latex_to_markdown(
        "\\begin{document}\\begin{tabular}{l|cc|cc}\
         & \\multicolumn{2}{c|}{A} & \\multicolumn{2}{c}{B} \\\\\
         M & P & R & P & R \\\\\
         X & 1 & 2 & 3 & 4 \\\\\
         \\end{tabular}\\end{document}",
    );
    assert!(
        md.contains("|  | A |  | B |  |"),
        "multicolumn header must pad spanned columns: {md}"
    );
    assert!(
        md.contains("| M | P | R | P | R |"),
        "data row must stay aligned: {md}"
    );
}

#[test]
fn tabularnewline_terminates_table_rows() {
    // `\tabularnewline` is the robust alias for `\\` used when a column redefines
    // `\\` (e.g. a `\raggedright` `p{}` column). Rows separated by it must split
    // like `\\`; otherwise the whole table collapses onto one row (GAN's
    // comparison table).
    let md = latex_to_markdown(
        "\\begin{document}\\begin{tabular}{l|c}\
         a & 1 \\tabularnewline{}\
         b & 2 \\tabularnewline{}\
         c & 3 \\tabularnewline{}\
         \\end{tabular}\\end{document}",
    );
    assert!(md.contains("| a | 1 |"), "first row missing: {md}");
    assert!(
        md.contains("| b | 2 |"),
        "\\tabularnewline did not terminate the row: {md}"
    );
    assert!(md.contains("| c | 3 |"), "later row lost: {md}");
}

#[test]
fn newcolumntype_definition_does_not_leak_and_table_renders() {
    // `array`-package `\newcolumntype{<char>}[n]{<spec>}` defines a custom column
    // type used in a tabular colspec (ResNet/GAN/VGG result tables). texmark drops
    // the colspec when building tables, so before the fix the definition's
    // `{char}[n]{spec}` arguments fell through to normal text and leaked (e.g.
    // `x[1]>p1pt`), while a `\newcommand` cell macro filled the data cells. The
    // definition must be consumed (no leak) and the table must still render.
    // The `\newcolumntype` is placed in the document BODY (as in the papers where
    // this leaked): a preamble definition's stray text is dropped regardless, so
    // only a body definition exercises the fix.
    let md = latex_to_markdown(
        "\\begin{document}\
         BEFOREMARK \\newcolumntype{x}[1]{>{\\centering}p{#1pt}} AFTERMARK\
         \\newcommand{\\cellval}[1]{VAL#1}\
         \\begin{tabular}{l|x{42}|c}\
         name & \\cellval{A} & ZZZMARKER \\\\\
         row1 & \\cellval{B} & data \\\\\
         \\end{tabular}\\end{document}",
    );
    // The colspec / column-type definition must not leak as literal text.
    assert!(
        !md.contains("p1pt"),
        "newcolumntype spec leaked into output: {md}"
    );
    assert!(
        !md.contains("centering"),
        "column-type spec leaked into output: {md}"
    );
    assert!(
        !md.contains("x[1]") && !md.contains("x\\[1\\]"),
        "colspec leaked into output: {md}"
    );
    // The table renders as GFM with a header separator row.
    assert!(md.contains("| --- |"), "table did not render as GFM: {md}");
    // The custom cell macro expands in data cells, and plain cells survive.
    assert!(
        md.contains("VALA") && md.contains("VALB"),
        "custom cell macro did not expand: {md}"
    );
    assert!(md.contains("ZZZMARKER"), "plain data cell was lost: {md}");
    // The `\newcolumntype` must be consumed without eating the surrounding text.
    assert!(
        md.contains("BEFOREMARK") && md.contains("AFTERMARK"),
        "text around the definition was lost: {md}"
    );
}

#[test]
fn numeric_citations_when_natbib_numbers_option() {
    let src = r#"\usepackage[numbers]{natbib}
\begin{document}
See \citep{a,b}.
\begin{thebibliography}{9}
\bibitem[Alpha(2001)]{a} Al Alpha. First. 2001.
\bibitem[Beta(2002)]{b} Bo Beta. Second. 2002.
\end{thebibliography}
\end{document}"#;
    let md = latex_to_markdown(src);
    assert!(
        md.contains("See [[1](#ref-a), [2](#ref-b)]."),
        "numeric in-text cite must link each number: {md}"
    );
    assert!(
        md.contains("- <a id=\"ref-a\"></a>[1] Al Alpha"),
        "numeric list label/anchor wrong: {md}"
    );
}

#[test]
fn author_year_citations_when_natbib_default() {
    // natbib without a numeric option → author-year, from the \bibitem[label].
    let src = r#"\usepackage{natbib}
\begin{document}
\citet{a} found it; it is known \citep{a,b}.
\begin{thebibliography}{9}
\bibitem[Alpha(2001)]{a} Al Alpha. First. 2001.
\bibitem[Beta et al.(2002)]{b} Bo Beta. Second. 2002.
\end{thebibliography}
\end{document}"#;
    let md = latex_to_markdown(src);
    assert!(
        md.contains("Alpha (2001) found it"),
        "\\citet author-year wrong: {md}"
    );
    assert!(
        md.contains("(Alpha, 2001; Beta et al., 2002)"),
        "\\citep author-year wrong: {md}"
    );
    // Author-year reference list has no bracketed label (but still carries an
    // anchor so cites can link to it).
    assert!(
        md.contains("Al Alpha. First. 2001."),
        "list should be unlabeled: {md}"
    );
    assert!(
        !md.contains("- [1]"),
        "author-year list must not be numbered: {md}"
    );
}

#[test]
fn numbered_cite_links_to_reference_entry() {
    // A numbered `[N]` in-text cite must be a Markdown link whose `#...` target
    // exactly matches an `<a id="...">` anchor emitted at the reference entry, so
    // clicking the citation jumps to its reference on GitHub. Covers both the
    // single-key and multi-key (`[N, M]`) forms.
    let src = r#"\begin{document}
Foundational \cite{alpha}; combined \cite{alpha,beta}.
\begin{thebibliography}{9}
\bibitem{alpha} Al Alpha. First. 2001.
\bibitem{beta} Bo Beta. Second. 2002.
\end{thebibliography}
\end{document}"#;
    let md = latex_to_markdown(src);

    // The in-text cite is a link, not plain text.
    assert!(
        md.contains("Foundational [[1](#ref-alpha)];"),
        "single cite not linked: {md}"
    );
    assert!(
        md.contains("combined [[1](#ref-alpha), [2](#ref-beta)]."),
        "multi-key cite not linked: {md}"
    );

    // The reference entries emit anchors that exactly match the link targets.
    assert!(
        md.contains("<a id=\"ref-alpha\"></a>[1] Al Alpha"),
        "reference anchor for alpha missing/mismatched: {md}"
    );
    assert!(
        md.contains("<a id=\"ref-beta\"></a>[2] Bo Beta"),
        "reference anchor for beta missing/mismatched: {md}"
    );

    // Every cite link target has a matching anchor id (the link resolves).
    for target in ["ref-alpha", "ref-beta"] {
        assert!(
            md.contains(&format!("(#{target})")) && md.contains(&format!("id=\"{target}\"")),
            "link target #{target} has no matching anchor: {md}"
        );
    }
}

#[test]
fn math_normalizes_katex_invalid_commands() {
    // Standard text-font commands invalid in math are remapped; an undefined
    // macro is left untouched (not special-cased to one paper's intent).
    let src = r#"\begin{document}
Inline $\textsc{mask}(x)$ and $\undefinedmacro{y}$.
\end{document}"#;
    let md = latex_to_markdown(src);
    assert!(!md.contains("\\textsc"), "\\textsc leaked into math: {md}");
    assert!(
        md.contains("\\text {mask}") || md.contains("\\text{mask}"),
        "textsc should map to \\text: {md}"
    );
    assert!(
        md.contains("\\undefinedmacro"),
        "unknown macro must pass through unchanged, not be guessed at: {md}"
    );
}

#[test]
fn katex_normalizes_bm_and_drops_linebreak() {
    // `\bm`/`\pmb` (bold math) aren't in stock KaTeX — map to `\boldsymbol`.
    // Break commands (`\linebreak[n]`) carry no math meaning and KaTeX rejects
    // them — drop the word and its optional argument.
    let md = latex_to_markdown(
        r"\begin{document}$\bm{x} + \pmb{y}$ and $a \linebreak[1] b$.\end{document}",
    );
    assert!(
        md.contains("\\boldsymbol"),
        "\\bm/\\pmb should map to \\boldsymbol: {md}"
    );
    assert!(
        !md.contains("\\bm") && !md.contains("\\pmb"),
        "raw \\bm/\\pmb must be gone: {md}"
    );
    assert!(
        !md.contains("\\linebreak"),
        "\\linebreak must be dropped inside math: {md}"
    );
}

#[test]
fn nonbreaking_tilde_does_not_consume_following_command() {
    // `~` is the active non-breaking space, not the `\~` accent. It must not
    // swallow the command after it (regression: `Fig.~\textbf{A}` lost \textbf).
    let md = latex_to_markdown(r"\begin{document}see~\textbf{Bold} and x~y\end{document}");
    assert!(md.contains("**Bold**"), "~ ate the following command: {md}");
    assert!(
        md.contains("x\u{00A0}y") || md.contains("x y"),
        "~ should be a space: {md}"
    );
}

#[test]
fn text_accents_compose_to_unicode() {
    let src = r#"\begin{document}
Christopher R{\'e}, Erd\H{o}s, na\"ive caf\'e, \c{c}.
\end{document}"#;
    let md = latex_to_markdown(src);
    assert!(md.contains("Christopher Ré"), "acute in group lost: {md}");
    assert!(md.contains("Erdős"), "double acute lost: {md}");
    assert!(md.contains("naïve café"), "umlaut/acute lost: {md}");
    assert!(md.contains('ç'), "cedilla lost: {md}");
}

#[test]
fn tilde_accent_and_nbsp_are_distinct() {
    // `\~` is the tilde accent (Spanish/Portuguese, names); the active `~` is a
    // non-breaking space. They tokenize identically in older designs — verify
    // both are handled correctly in the same document.
    let md = latex_to_markdown(
        r"\begin{document}Pe\~na, S\~ao Paulo, \~{A}. See~\ref{f} x~y.\end{document}",
    );
    assert!(md.contains("Peña"), "tilde accent (bare) lost: {md}");
    assert!(md.contains("São"), "tilde accent lost: {md}");
    assert!(md.contains('Ã'), "tilde accent (braced) lost: {md}");
    assert!(!md.contains("\\~"), "raw tilde accent leaked: {md}");
    // The active `~` stays a non-breaking space and does not eat the next token.
    assert!(
        md.contains("x\u{00A0}y") || md.contains("x y"),
        "nbsp lost: {md}"
    );
}

#[test]
fn core_text_symbols_and_dotless_letters() {
    let md = latex_to_markdown(
        r"\begin{document}\textquestiondown C\'omo? \textexclamdown Hola! \i \j \textbullet\end{document}",
    );
    assert!(
        md.contains('¿') && md.contains('¡'),
        "Spanish punctuation lost: {md}"
    );
    assert!(
        md.contains('ı') && md.contains('ȷ'),
        "dotless i/j lost: {md}"
    );
    assert!(md.contains('•'), "bullet lost: {md}");
}

#[test]
fn missing_input_is_reported_not_silently_dropped() {
    // A `\input`/`\include` the resolver can't find is silent content loss: it
    // must be recorded so the output is not claimed complete. A missing
    // `\usepackage` is NOT reported (texmark bundles few packages — expected).
    struct NoInput;
    impl Resolver for NoInput {
        fn resolve(&self, _: &str) -> Option<String> {
            None
        }
    }
    let src =
        "\\usepackage{amsmath}\\begin{document}Before \\input{chapter2} after.\\end{document}";
    let mut state = State::new();
    let (_tree, diag) = Engine::new(src, &mut state, &NoInput).parse();
    assert!(
        diag.unresolved_inputs.contains("chapter2"),
        "missing \\input must be recorded: {:?}",
        diag.unresolved_inputs
    );
    assert!(
        !diag.unresolved_inputs.iter().any(|n| n.contains("amsmath")),
        "missing \\usepackage must NOT be reported as lost content: {:?}",
        diag.unresolved_inputs
    );
    assert!(
        diag.is_incomplete(),
        "missing content include => output is incomplete"
    );
    assert!(!diag.is_clean(), "missing content include => not clean");
}

#[test]
fn verbatim_in_included_file_preserves_raw_formatting() {
    // A verbatim/lstlisting body reached through `\input` must be read as raw
    // source — indentation and blank lines intact, no `\par` synthesized from
    // blank lines — not reconstructed from a tokenized stream. This is the
    // regression the input-frame-stack refactor exists to fix.
    struct Code;
    impl Resolver for Code {
        fn resolve(&self, name: &str) -> Option<String> {
            (name == "code" || name == "code.tex").then(|| {
                "\\begin{lstlisting}\ndef f(x):\n    return x\n\nprint(f(3))\n\\end{lstlisting}\n"
                    .to_string()
            })
        }
    }
    let src = "\\begin{document}\\input{code}\\end{document}";
    let md = md::to_markdown(&latex_to_tree_with(src, &Code));
    assert!(
        md.contains("def f(x):\n    return x"),
        "indentation/newlines lost: {md}"
    );
    assert!(
        md.contains("\n\nprint(f(3))"),
        "blank line not preserved: {md}"
    );
    assert!(!md.contains("\\par"), "blank line leaked as \\par: {md}");
}

#[test]
fn input_interleaves_before_following_content() {
    // `\input` inserts the file at the point it appears: content after `\input`
    // in the parent must be read *after* the included file, not before it.
    struct Sub;
    impl Resolver for Sub {
        fn resolve(&self, name: &str) -> Option<String> {
            (name == "sub").then(|| "B".to_string())
        }
    }
    let md = md::to_markdown(&latex_to_tree_with(
        "\\begin{document}A\\input{sub}C\\end{document}",
        &Sub,
    ));
    let (a, b, c) = (md.find('A'), md.find('B'), md.find('C'));
    assert!(a < b && b < c, "input ordering wrong (want A<B<C): {md}");
}

#[test]
fn longtable_with_input_rows_and_wrapping_group_renders_as_table() {
    // A real-world shape: a `longtable` wrapped in a `\fontsize` group and a
    // `center`, with caption + repeated-header + footer machinery, whose rows
    // live in an `\input`ed file. All of it must produce one clean pipe table,
    // not run-on inline text.
    struct Rows;
    impl Resolver for Rows {
        fn resolve(&self, name: &str) -> Option<String> {
            (name == "rows" || name == "rows.tex")
                .then(|| "foo & 1 \\\\\nbar & 2 \\\\\n".to_string())
        }
    }
    let src = "\\begin{document}\
        {\\fontsize{6pt}{7pt}\\selectfont\
        \\begin{center}\
        \\begin{longtable}{ll}\
        \\caption{Cap}\\label{t} \\\\\
        \\toprule \\textbf{Name} & \\textbf{N} \\\\ \\midrule\
        \\endfirsthead\
        \\multicolumn{2}{l}{cont} \\\\ \\endhead\
        \\bottomrule \\endfoot\
        \\input{rows}\
        \\end{longtable}\
        \\end{center}}\
        \\end{document}";
    let md = md::to_markdown(&latex_to_tree_with(src, &Rows));
    assert!(
        md.contains("| **Name** | **N** |"),
        "header row missing: {md}"
    );
    assert!(md.contains("| --- | --- |"), "separator missing: {md}");
    assert!(
        md.contains("| foo | 1 |") && md.contains("| bar | 2 |"),
        "data rows missing: {md}"
    );
    assert!(
        !md.contains("Cap"),
        "caption row should be dropped, not inlined: {md}"
    );
    assert!(
        !md.contains("cont"),
        "repeated-header block should be dropped: {md}"
    );
    assert!(!md.contains("6pt"), "\\fontsize args leaked: {md}");
}

#[test]
fn for_loop_generates_content_per_item() {
    // \@for binds its variable to each comma item and emits the body per
    // iteration — a content-generating loop, so the output must contain each.
    struct Defs;
    impl Resolver for Defs {
        fn resolve(&self, name: &str) -> Option<String> {
            (name == "defs").then(|| {
                "\\makeatletter\\gdef\\emitlist{\\@for\\x:=a,b,c\\do{(\\x)}}\\makeatother"
                    .to_string()
            })
        }
    }
    let src = "\\begin{document}\\input{defs}\\emitlist\\end{document}";
    let md = md::to_markdown(&latex_to_tree_with(src, &Defs));
    assert!(
        md.contains("(a)(b)(c)"),
        "\\@for should emit body per item: {md}"
    );
}

#[test]
fn counters_ifcsname_and_forloop() {
    // A miniature of the ICML affiliation machinery: counters, \arabic, a real
    // \ifcsname test, computed-name storage via \expandafter\gdef\csname, and a
    // \forloop over the counters — all must render actual content.
    struct Defs;
    impl Resolver for Defs {
        fn resolve(&self, name: &str) -> Option<String> {
            (name == "defs").then(|| {
                "\\makeatletter\
                 \\newcounter{x}\\stepcounter{x}\\stepcounter{x}\
                 \\expandafter\\gdef\\csname item\\thex\\endcsname{Second}\
                 \\newcommand{\\dump}{\\forloop{i}{1}{\\value{i} < 3}{(\\arabic{i}\
                 \\ifcsname item\\thei\\endcsname:\\csname item\\thei\\endcsname\\fi)}}\
                 \\makeatother"
                    .to_string()
            })
        }
    }
    let src = "\\begin{document}\\input{defs}\\dump\\end{document}";
    let md = md::to_markdown(&latex_to_tree_with(src, &Defs));
    // item2 was stored ("Second"); item1 was not. Loop runs i=1,2.
    assert!(md.contains("(1)"), "i=1 has no stored name: {md}");
    assert!(
        md.contains("(2:Second)"),
        "i=2 resolves its computed name: {md}"
    );
}

#[test]
fn bib_fallback_formats_only_cited_entries() {
    let bib = r#"
        @article{cited1, title={First Paper}, author={Zed, Ada and Young, Bo},
          journal={J of Things}, volume={4}, number={2}, pages={9--12}, year={2020}}
        @inproceedings{uncited, title={Skip Me}, author={Nobody, No}, booktitle={X}, year={1999}}
        @misc{cited2, title={Second Note}, author={Ada Lovelace}, year={2021}}
    "#;
    let src = r#"
\begin{document}
We build on \cite{cited1} and also \cite{cited2}.
\bibliography{ref}
\end{document}
"#;
    let resolver = BibOnly { bib: bib.into() };
    let md = md::to_markdown(&latex_to_tree_with(src, &resolver));

    // Only the two cited entries become reference-list items, sorted by author
    // surname (Lovelace < Zed), and numbered in that order: cited2=[1], cited1=[2].
    assert!(
        md.contains("<a id=\"ref-cited2\"></a>[1]"),
        "cited2 should be [1]:\n{md}"
    );
    assert!(
        md.contains("<a id=\"ref-cited1\"></a>[2]"),
        "cited1 should be [2]:\n{md}"
    );
    assert!(!md.contains("uncited"), "uncited entry leaked:\n{md}");
    let one = md.find("- <a id=\"ref-cited2\"></a>[1]").unwrap();
    let two = md.find("- <a id=\"ref-cited1\"></a>[2]").unwrap();
    assert!(
        one < two,
        "should sort Lovelace ([1]) before Zed ([2]):\n{md}"
    );
    // In-text citations resolve to those numbers, each linked to its entry.
    assert!(
        md.contains("build on [[2](#ref-cited1)] and also [[1](#ref-cited2)]"),
        "cite numbers/links wrong:\n{md}"
    );
    // Formatted body: authors "First Last", journal emphasized, vol(num):pages.
    assert!(md.contains("Ada Zed and Bo Young"), "author format:\n{md}");
    assert!(
        md.contains("*J of Things*, 4(2):9\u{2013}12, 2020."),
        "article body (en-dash page range):\n{md}"
    );
}

#[test]
fn figure_alt_text_renders_math_and_resolved_refs() {
    // A figure caption is mirrored into its image's alt text. That alt must be a
    // readable, plain-text reduction of the RESOLVED caption — the same content
    // the visible caption shows — not a rawer second render. Regression: alt was
    // built from the raw source string, leaking `$…$`/`\times` and dropping the
    // `\ref` number (rendering an empty "Fig. .").
    let src = r"\begin{document}
\begin{figure}
\includegraphics{net.png}
\caption{\label{fig:net}A block over an $N \times N$ grid, as in Fig.~\ref{fig:net}.}
\end{figure}
\end{document}";
    let md = latex_to_markdown(src);
    // Grab the alt text of the `![alt](src)`.
    let alt_start = md.find("![").expect("no image in output");
    let alt_end = md[alt_start..].find("](").expect("malformed image") + alt_start;
    let alt = &md[alt_start + 2..alt_end];

    // Math reads as text, with the delimiters and raw `\times` gone.
    assert!(
        alt.contains("N × N"),
        "math not reduced to readable text: {alt:?}"
    );
    assert!(!alt.contains('$'), "raw math delimiter in alt: {alt:?}");
    assert!(!alt.contains("\\times"), "raw TeX macro in alt: {alt:?}");
    // The cross-reference shows its resolved number, not an empty "Fig. .".
    assert!(
        alt.contains("Fig. 1"),
        "resolved ref number missing from alt: {alt:?}"
    );
    assert!(
        !alt.contains("Fig. ."),
        "empty (dropped) ref in alt: {alt:?}"
    );
    // No leftover Markdown link syntax from the resolved ref.
    assert!(
        !alt.contains("]("),
        "markdown link syntax leaked into alt: {alt:?}"
    );
    // No leftover HTML from the inline `\label`'s `<a id="…"></a>` anchor.
    assert!(!alt.contains('<'), "raw HTML tag leaked into alt: {alt:?}");

    // The visible caption (duplicated) still renders the resolved path intact.
    let dup = latex_to_markdown_with_dup(src);
    assert!(
        dup.contains("$N \\times N$"),
        "visible caption math regressed: {dup}"
    );
    assert!(
        dup.contains("[1](#fig:net)"),
        "visible caption ref regressed: {dup}"
    );
}

/// Convert with `duplicate_captions` on, so the visible caption line is emitted.
fn latex_to_markdown_with_dup(src: &str) -> String {
    let tree = latex_to_tree(src);
    md::to_markdown_with(
        &tree,
        &md::MdOptions {
            duplicate_captions: true,
        },
    )
}

/// Count the elements in a tree, recursively.
fn count_elements(node: &Node) -> usize {
    match node {
        Node::Text(_) => 0,
        Node::Element(e) => 1 + e.children.iter().map(count_elements).sum::<usize>(),
    }
}

fn convert_paper(main: &Path) -> Option<usize> {
    let source = std::fs::read_to_string(main).ok()?;
    let base = main.parent().unwrap_or(Path::new(".")).to_path_buf();
    let resolver = PaperFiles { base };
    let mut state = State::new();
    let (tree, _diag) = Engine::new(&source, &mut state, &resolver).parse();
    Some(count_elements(&Node::Element(tree)))
}

#[test]
fn reference_papers_convert() {
    // (paper id, main .tex file). Skipped individually if not present so the
    // suite still runs in a checkout without the references/ tree.
    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let papers = [
        "references/papers/2205.14135/streaming_attention_neurips_2022.tex",
        "references/papers/2307.08691/flash2.tex",
        "references/papers/2407.08608/fa3_neurips2024.tex",
        "references/papers/2603.05451/arxiv_main.tex",
    ];
    let mut ran = 0;
    for relative_path in papers {
        let main = repository_root.join(relative_path);
        if !main.exists() {
            continue;
        }
        let count =
            convert_paper(&main).unwrap_or_else(|| panic!("failed to read {}", main.display()));
        assert!(
            count > 100,
            "paper {} produced only {count} elements (expected a substantial tree)",
            main.display()
        );
        ran += 1;
    }
    if ran == 0 {
        eprintln!("reference corpus not present; skipping paper conversions");
    }
}
