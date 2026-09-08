//! A tiny, self-contained BibTeX reader and formatter.
//!
//! This module emulates just enough of `bibtex` to render a references section
//! natively — for targets (like WASM) that cannot shell out to `bibtex`/`biber`.
//! It is used only as a fallback: when a paper ships a pre-generated `.bbl`, that
//! exact output is preferred. Formatting targets a clean, simplified `plainnat`
//! (natbib author-year) style — the citation *label* is irrelevant because the
//! Markdown backend keys references by their `\bibitem{key}`, so only entry
//! *bodies* are produced here.
//!
//! Design: parse the messy inputs into precise types once (`EntryKind`,
//! `Name`) so downstream formatting works on parsed data and can never meet an
//! un-parsed value — "parse, don't validate".

use std::collections::{BTreeMap, BTreeSet};

/// The parsed entry type. Kept as an enum (never a re-matched string) so
/// [`format_body`] handles every variant under a compiler-checked `match`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryKind {
    Article,
    InProceedings,
    Book,
    InCollection,
    InBook,
    TechReport,
    PhdThesis,
    MastersThesis,
    Unpublished,
    Misc,
    /// Any type we don't special-case (`@software`, `@online`, ...): rendered
    /// with the generic fallback template.
    Other,
}

impl EntryKind {
    fn from_type(ty: &str) -> EntryKind {
        match ty.to_ascii_lowercase().as_str() {
            "article" => EntryKind::Article,
            "inproceedings" | "conference" => EntryKind::InProceedings,
            "book" => EntryKind::Book,
            "incollection" => EntryKind::InCollection,
            "inbook" => EntryKind::InBook,
            "techreport" => EntryKind::TechReport,
            "phdthesis" => EntryKind::PhdThesis,
            "mastersthesis" => EntryKind::MastersThesis,
            "unpublished" => EntryKind::Unpublished,
            "misc" => EntryKind::Misc,
            _ => EntryKind::Other,
        }
    }
}

/// A parsed author (or editor) name. The "First Last" vs "Last, First"
/// ambiguity is resolved here, once, so the formatter never re-inspects a raw
/// name string.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Name {
    /// A `Last, First` (or `Last, Jr, First`) name — the parts are known.
    Structured {
        first: String,
        last: String,
        suffix: Option<String>,
    },
    /// A name with no comma (`First von Last`) or a braced literal (`{NVIDIA}`):
    /// emitted verbatim, since its display order is already correct.
    Verbatim(String),
    /// The BibTeX `and others` marker.
    EtAl,
}

impl Name {
    /// Render the name in "First Last" display order.
    fn display(&self) -> String {
        match self {
            Name::Structured {
                first,
                last,
                suffix,
            } => {
                let mut s = String::new();
                if !first.is_empty() {
                    s.push_str(first);
                    s.push(' ');
                }
                s.push_str(last);
                if let Some(jr) = suffix {
                    s.push_str(", ");
                    s.push_str(jr);
                }
                s
            }
            Name::Verbatim(raw) => raw.clone(),
            Name::EtAl => "et al.".to_string(),
        }
    }

    /// The last name, for sorting. Comma-form names use their `last` part;
    /// no-comma names ("First Last") use their final word as the surname.
    fn sort_key(&self) -> String {
        match self {
            Name::Structured { last, .. } => last.to_ascii_lowercase(),
            Name::Verbatim(raw) => raw
                .trim_matches(['{', '}'])
                .split_whitespace()
                .last()
                .unwrap_or("")
                .to_ascii_lowercase(),
            Name::EtAl => "\u{10ffff}".to_string(), // sort last
        }
    }
}

/// Which references to emit — replaces a `"*"` sentinel in the key set.
pub enum Cited {
    /// Include every entry (e.g. `\nocite{*}`).
    All,
    /// Include only these citation keys.
    Keys(BTreeSet<String>),
}

impl Cited {
    fn includes(&self, key: &str) -> bool {
        match self {
            Cited::All => true,
            Cited::Keys(keys) => keys.iter().any(|k| k.eq_ignore_ascii_case(key)),
        }
    }
}

/// One parsed `.bib` entry. Fields stay a map because BibTeX fields are
/// open-ended, free-form LaTeX; only the bits that drive formatting (the type,
/// and author names on demand) are lifted into types.
pub struct Entry {
    kind: EntryKind,
    key: String,
    fields: BTreeMap<String, String>,
}

impl Entry {
    fn field(&self, name: &str) -> Option<&str> {
        self.fields.get(name).map(String::as_str)
    }

    fn authors(&self) -> Vec<Name> {
        self.field("author").map(parse_names).unwrap_or_default()
    }
}

// --- parsing ------------------------------------------------------------

/// Parse a `.bib` database into entries. Malformed or unrecognized `@`-blocks
/// (`@comment`, `@preamble`, `@string`) are skipped.
pub fn parse(src: &str) -> Vec<Entry> {
    let chars: Vec<char> = src.chars().collect();
    let mut entries = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '@' {
            i += 1;
            continue;
        }
        i += 1; // past '@'
        let ty = take_while(&chars, &mut i, |c| c.is_ascii_alphabetic());
        skip_ws(&chars, &mut i);
        // Entry body is delimited by { } (or, rarely, parentheses).
        let (open, close) = match chars.get(i) {
            Some('{') => ('{', '}'),
            Some('(') => ('(', ')'),
            _ => continue,
        };
        i += 1;
        let body = take_balanced(&chars, &mut i, open, close);
        match ty.to_ascii_lowercase().as_str() {
            "comment" | "preamble" | "string" => continue,
            _ => {
                if let Some(entry) = parse_entry(&ty, &body) {
                    entries.push(entry);
                }
            }
        }
    }
    entries
}

/// Parse the inside of one `@type{ ... }` block: `key, field = value, ...`.
fn parse_entry(ty: &str, body: &str) -> Option<Entry> {
    let chars: Vec<char> = body.chars().collect();
    let mut i = 0;
    skip_ws(&chars, &mut i);
    let key = take_while(&chars, &mut i, |c| c != ',' && !c.is_whitespace())
        .trim()
        .to_string();
    if key.is_empty() {
        return None;
    }
    let mut fields = BTreeMap::new();
    skip_ws(&chars, &mut i);
    if chars.get(i) == Some(&',') {
        i += 1;
    }
    while i < chars.len() {
        skip_ws(&chars, &mut i);
        let name = take_while(&chars, &mut i, |c| c != '=' && c != ',')
            .trim()
            .to_ascii_lowercase();
        skip_ws(&chars, &mut i);
        if chars.get(i) != Some(&'=') {
            // No value (trailing comma or junk); stop.
            break;
        }
        i += 1; // past '='
        skip_ws(&chars, &mut i);
        let value = take_value(&chars, &mut i);
        if !name.is_empty() {
            fields.insert(name, collapse_ws(&value));
        }
        skip_ws(&chars, &mut i);
        if chars.get(i) == Some(&',') {
            i += 1;
        }
    }
    Some(Entry {
        kind: EntryKind::from_type(ty),
        key,
        fields,
    })
}

/// Read a field value: a `{...}` group, a `"..."` string, or a bare token
/// (number or macro name). Concatenation with `#` is joined.
fn take_value(chars: &[char], i: &mut usize) -> String {
    let mut parts = Vec::new();
    loop {
        skip_ws(chars, i);
        match chars.get(*i) {
            Some('{') => {
                *i += 1;
                parts.push(take_balanced(chars, i, '{', '}'));
            }
            Some('"') => {
                *i += 1;
                parts.push(take_quoted(chars, i));
            }
            Some(c) if c.is_ascii_alphanumeric() => {
                let bare = take_while(chars, i, |c| c.is_ascii_alphanumeric() || c == '_');
                parts.push(month_macro(&bare).unwrap_or(bare));
            }
            _ => break,
        }
        skip_ws(chars, i);
        if chars.get(*i) == Some(&'#') {
            *i += 1; // concatenation; read the next part
            continue;
        }
        break;
    }
    parts.concat()
}

/// Collect characters until the balanced `close` for the already-consumed
/// `open`. The closing delimiter is consumed but not included.
fn take_balanced(chars: &[char], i: &mut usize, open: char, close: char) -> String {
    let mut depth = 1;
    let mut out = String::new();
    while *i < chars.len() {
        let c = chars[*i];
        *i += 1;
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                break;
            }
        }
        out.push(c);
    }
    out
}

/// Collect a `"`-delimited string, honoring `{}` nesting (a `"` inside braces
/// does not close the value). The closing quote is consumed.
fn take_quoted(chars: &[char], i: &mut usize) -> String {
    let mut depth = 0;
    let mut out = String::new();
    while *i < chars.len() {
        let c = chars[*i];
        *i += 1;
        match c {
            '{' => depth += 1,
            '}' => depth -= 1,
            '"' if depth == 0 => break,
            _ => {}
        }
        out.push(c);
    }
    out
}

fn take_while(chars: &[char], i: &mut usize, pred: impl Fn(char) -> bool) -> String {
    let start = *i;
    while *i < chars.len() && pred(chars[*i]) {
        *i += 1;
    }
    chars[start..*i].iter().collect()
}

fn skip_ws(chars: &[char], i: &mut usize) {
    while *i < chars.len() && chars[*i].is_whitespace() {
        *i += 1;
    }
}

/// Collapse internal runs of whitespace to single spaces (author lists and
/// titles are often wrapped across source lines).
fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn month_macro(bare: &str) -> Option<String> {
    let full = match bare.to_ascii_lowercase().as_str() {
        "jan" => "January",
        "feb" => "February",
        "mar" => "March",
        "apr" => "April",
        "may" => "May",
        "jun" => "June",
        "jul" => "July",
        "aug" => "August",
        "sep" => "September",
        "oct" => "October",
        "nov" => "November",
        "dec" => "December",
        _ => return None,
    };
    Some(full.to_string())
}

/// Split an `author`/`editor` field on top-level ` and ` and parse each name.
fn parse_names(raw: &str) -> Vec<Name> {
    split_on_and(raw)
        .into_iter()
        .map(|part| parse_name(part.trim()))
        .collect()
}

/// Split on ` and ` that is not nested inside braces.
fn split_on_and(raw: &str) -> Vec<String> {
    let chars: Vec<char> = raw.chars().collect();
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth = 0;
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '{' => depth += 1,
            '}' => depth -= 1,
            _ => {}
        }
        if depth == 0
            && chars[i] == 'a'
            && matches!(chars.get(i + 1), Some('n'))
            && matches!(chars.get(i + 2), Some('d'))
            && chars
                .get(i.wrapping_sub(1))
                .is_none_or(|c| c.is_whitespace())
            && chars.get(i + 3).is_some_and(|c| c.is_whitespace())
        {
            parts.push(chars[start..i].iter().collect());
            i += 3;
            start = i;
            continue;
        }
        i += 1;
    }
    parts.push(chars[start..].iter().collect());
    parts
}

fn parse_name(raw: &str) -> Name {
    if raw.eq_ignore_ascii_case("others") {
        return Name::EtAl;
    }
    let commas: Vec<&str> = raw.split(',').map(str::trim).collect();
    match commas.as_slice() {
        [last, first] => Name::Structured {
            first: (*first).to_string(),
            last: (*last).to_string(),
            suffix: None,
        },
        [last, jr, first] => Name::Structured {
            first: (*first).to_string(),
            last: (*last).to_string(),
            suffix: Some((*jr).to_string()),
        },
        // No comma (or an unusual shape): keep as written.
        _ => Name::Verbatim(raw.to_string()),
    }
}

// --- formatting ---------------------------------------------------------

/// Format the selected, cited entries as a `thebibliography` environment ready
/// to be re-parsed by the engine. Returns `None` if nothing is selected.
pub fn to_thebibliography(entries: &[Entry], cited: &Cited) -> Option<String> {
    let mut chosen: Vec<&Entry> = entries.iter().filter(|e| cited.includes(&e.key)).collect();
    if chosen.is_empty() {
        return None;
    }
    chosen.sort_by_key(|e| sort_key(e));

    let mut out = format!("\\begin{{thebibliography}}{{{}}}\n", chosen.len());
    for e in chosen {
        out.push_str("\\bibitem{");
        out.push_str(&e.key);
        out.push_str("}\n");
        out.push_str(&format_body(e));
        out.push_str("\n\n");
    }
    out.push_str("\\end{thebibliography}\n");
    Some(out)
}

/// Sort by (first-author last name, year, title) — plainnat's author-sort.
fn sort_key(e: &Entry) -> (String, String, String) {
    let author = e.authors().first().map(Name::sort_key).unwrap_or_default();
    (
        author,
        e.field("year").unwrap_or("").to_string(),
        e.field("title").unwrap_or("").to_ascii_lowercase(),
    )
}

/// Join names in plainnat order: `A`, `A and B`, or `A, B, and C`. A trailing
/// `and others` renders as `... et al.` with no conjunction.
fn format_names(names: &[Name]) -> String {
    let et_al = names.last() == Some(&Name::EtAl);
    let people: Vec<String> = names
        .iter()
        .filter(|n| **n != Name::EtAl)
        .map(Name::display)
        .collect();
    if et_al {
        return format!("{} et al.", people.join(", "));
    }
    match people.as_slice() {
        [] => String::new(),
        [one] => one.clone(),
        [a, b] => format!("{a} and {b}"),
        [rest @ .., last] => format!("{}, and {last}", rest.join(", ")),
    }
}

/// The formatted `\bibitem` body for one entry (plainnat, simplified).
fn format_body(e: &Entry) -> String {
    let mut blocks: Vec<String> = Vec::new();

    let authors = format_names(&e.authors());
    if !authors.is_empty() {
        blocks.push(format!("{}.", authors.trim_end_matches('.')));
    }

    let title = e.field("title").unwrap_or_default();
    let year = e.field("year").unwrap_or_default();

    match e.kind {
        EntryKind::Article => {
            blocks.push(format!("{title}."));
            blocks.push(article_tail(e, year));
        }
        EntryKind::InProceedings => {
            blocks.push(format!("{title}."));
            blocks.push(proceedings_tail(e, year));
        }
        EntryKind::Book => {
            let mut b = format!("\\emph{{{title}}}");
            if let Some(vol) = e.field("volume") {
                b.push_str(&format!(", volume~{vol}"));
            }
            b.push('.');
            blocks.push(b);
            blocks.push(publisher_tail(e, year));
        }
        EntryKind::InCollection | EntryKind::InBook => {
            blocks.push(format!("{title}."));
            if let Some(booktitle) = e.field("booktitle") {
                blocks.push(format!("In \\emph{{{booktitle}}}."));
            }
            blocks.push(publisher_tail(e, year));
        }
        EntryKind::TechReport => {
            blocks.push(format!("{title}."));
            let mut b = String::from("Technical report");
            if let Some(inst) = e.field("institution") {
                b.push_str(&format!(", {inst}"));
            }
            push_year(&mut b, year);
            blocks.push(b);
        }
        EntryKind::PhdThesis | EntryKind::MastersThesis => {
            blocks.push(format!("{title}."));
            let kind = if e.kind == EntryKind::PhdThesis {
                "PhD thesis"
            } else {
                "Master's thesis"
            };
            let mut b = kind.to_string();
            if let Some(school) = e.field("school") {
                b.push_str(&format!(", {school}"));
            }
            push_year(&mut b, year);
            blocks.push(b);
        }
        EntryKind::Unpublished => {
            blocks.push(format!("{title}."));
            let mut b = e.field("note").unwrap_or_default().to_string();
            push_year(&mut b, year);
            blocks.push(b);
        }
        EntryKind::Misc | EntryKind::Other => {
            // Minimal: title then year, plus any note/howpublished.
            let mut b = title.to_string();
            if let Some(how) = e.field("howpublished").or_else(|| e.field("note")) {
                if !b.is_empty() {
                    b.push_str(". ");
                }
                b.push_str(how);
            }
            push_year(&mut b, year);
            blocks.push(b);
        }
    }

    // Trailing DOI / URL blocks, when present.
    if let Some(doi) = e.field("doi") {
        blocks.push(format!("doi: {doi}."));
    }
    if let Some(url) = e.field("url") {
        blocks.push(format!("URL \\url{{{url}}}."));
    }

    blocks
        .into_iter()
        .filter(|b| !b.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\\newblock ")
}

/// `\emph{journal}, vol(num):pages, year.` with parts omitted when absent.
fn article_tail(e: &Entry, year: &str) -> String {
    let mut b = format!("\\emph{{{}}}", e.field("journal").unwrap_or_default());
    if let Some(vol) = e.field("volume") {
        b.push_str(&format!(", {vol}"));
        if let Some(num) = e.field("number") {
            b.push_str(&format!("({num})"));
        }
        if let Some(pages) = e.field("pages") {
            b.push_str(&format!(":{pages}"));
        }
    } else if let Some(pages) = e.field("pages") {
        b.push_str(&format!(", pages {pages}"));
    }
    push_year(&mut b, year);
    b
}

/// `In \emph{booktitle}, pages P, address, year.`
fn proceedings_tail(e: &Entry, year: &str) -> String {
    let mut b = format!("In \\emph{{{}}}", e.field("booktitle").unwrap_or_default());
    if let Some(pages) = e.field("pages") {
        b.push_str(&format!(", pages {pages}"));
    }
    if let Some(addr) = e.field("address") {
        b.push_str(&format!(", {addr}"));
    }
    push_year(&mut b, year);
    b
}

/// `Publisher, year.`
fn publisher_tail(e: &Entry, year: &str) -> String {
    let mut b = e.field("publisher").unwrap_or_default().to_string();
    push_year(&mut b, year);
    b
}

/// Append `, year.` (or just `.`), keeping punctuation clean.
fn push_year(b: &mut String, year: &str) {
    if !year.is_empty() {
        if !b.is_empty() {
            b.push_str(", ");
        }
        b.push_str(year);
    }
    b.push('.');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(list: &[&str]) -> Cited {
        Cited::Keys(list.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn parses_entry_types_and_fields() {
        let src = r#"
            @article{smith2020,
              title = {A study of things},
              author = {Smith, John and Doe, Jane},
              journal = {Journal of Things},
              volume = {12}, number = {3}, pages = {100--110}, year = {2020}
            }
        "#;
        let entries = parse(src);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].kind, EntryKind::Article);
        assert_eq!(entries[0].key, "smith2020");
        assert_eq!(entries[0].field("journal"), Some("Journal of Things"));
    }

    #[test]
    fn formats_article_body() {
        let src = r#"@article{k, title={T}, author={Smith, John and Doe, Jane},
            journal={J}, volume={1}, number={2}, pages={3--4}, year={2020}}"#;
        let out = to_thebibliography(&parse(src), &Cited::All).unwrap();
        assert!(out.contains("\\bibitem{k}"), "{out}");
        assert!(out.contains("John Smith and Jane Doe"), "{out}");
        assert!(out.contains("\\emph{J}, 1(2):3--4, 2020."), "{out}");
    }

    #[test]
    fn filters_to_cited_keys() {
        let src = r#"
            @misc{a, title={A}, year={2001}}
            @misc{b, title={B}, year={2002}}
            @misc{c, title={C}, year={2003}}
        "#;
        let out = to_thebibliography(&parse(src), &keys(&["a", "c"])).unwrap();
        assert!(out.contains("\\bibitem{a}"), "{out}");
        assert!(!out.contains("\\bibitem{b}"), "{out}");
        assert!(out.contains("\\bibitem{c}"), "{out}");
    }

    #[test]
    fn sorts_by_author_last_name() {
        let src = r#"
            @misc{z, title={T}, author={Zeta, Al}, year={2000}}
            @misc{a, title={T}, author={Alpha, Bo}, year={2000}}
        "#;
        let out = to_thebibliography(&parse(src), &Cited::All).unwrap();
        let a = out.find("\\bibitem{a}").unwrap();
        let z = out.find("\\bibitem{z}").unwrap();
        assert!(a < z, "Alpha should sort before Zeta:\n{out}");
    }

    #[test]
    fn sorts_no_comma_names_by_surname() {
        // "First Last" (no comma) must sort by the last word, not the first name.
        let src = r#"
            @misc{lef, title={T}, author={Benjamin Lefaudeux}, year={2022}}
            @misc{che, title={T}, author={Chen, Beidi}, year={2021}}
        "#;
        let out = to_thebibliography(&parse(src), &Cited::All).unwrap();
        let che = out.find("\\bibitem{che}").unwrap();
        let lef = out.find("\\bibitem{lef}").unwrap();
        assert!(che < lef, "Chen should sort before Lefaudeux:\n{out}");
    }

    #[test]
    fn three_authors_use_oxford_and() {
        let src = r#"@misc{k, author={A, X and B, Y and C, Z}, title={T}, year={2020}}"#;
        let out = to_thebibliography(&parse(src), &Cited::All).unwrap();
        assert!(out.contains("X A, Y B, and Z C"), "{out}");
    }

    #[test]
    fn organization_and_no_comma_names_kept_verbatim() {
        let src = r#"@misc{k, author={{NVIDIA}}, title={GPU}, year={2017}}"#;
        let out = to_thebibliography(&parse(src), &Cited::All).unwrap();
        assert!(out.contains("NVIDIA"), "{out}");

        let src2 = r#"@misc{k2, author={Sydney von Arx}, title={T}, year={2020}}"#;
        let out2 = to_thebibliography(&parse(src2), &Cited::All).unwrap();
        assert!(out2.contains("Sydney von Arx"), "{out2}");
    }

    #[test]
    fn and_others_becomes_et_al() {
        let src = r#"@misc{k, author={Smith, John and others}, title={T}, year={2020}}"#;
        let out = to_thebibliography(&parse(src), &Cited::All).unwrap();
        assert!(out.contains("John Smith et al."), "{out}");
    }

    #[test]
    fn skips_comment_and_string_blocks() {
        let src = r#"
            @comment{ this is ignored }
            @string{ acm = "ACM" }
            @book{real, title={Real}, author={Au, Th}, publisher={P}, year={1999}}
        "#;
        let entries = parse(src);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].key, "real");
    }

    #[test]
    fn empty_selection_returns_none() {
        let src = r#"@misc{a, title={A}, year={2001}}"#;
        assert!(to_thebibliography(&parse(src), &keys(&["missing"])).is_none());
    }
}
