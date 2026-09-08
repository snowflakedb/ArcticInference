//! Markdown navigation tools — `markdown_get_toc` / `markdown_grep`.
//!
//! Each tool reads the target file through a shared [`Sandbox`] (so the agent
//! can navigate workspace markdown without us exposing host paths). The parsing
//! lives in [`get_toc`] / [`grep`] as pure functions, trivial
//! to unit-test without a sandbox.
//!
//! Image handling used to live here (the removed `markdown_get_section` embedded
//! inline figures — the fan-out that blew a run's context window). Figures are
//! now viewed on demand: `read` a `.md` shows the literal `![alt](path)`, then
//! the `view` tool fetches that one path. The image machinery moved to
//! [`crate::tool::view`].

use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::exec::sandbox::Sandbox;
use crate::tool::{Tool, ToolOutput};

// ─── shared parser primitives ───────────────────────────────────────────────

/// A markdown ATX heading: `#{1,6}` followed by ` ` or `\t` (or end-of-line).
/// Returns `(level, title)` for valid headings, `None` otherwise.
///
/// Rejects `#include`, `#define`, etc. (no space after `#`) and `7+` hashes
/// (markdown caps headings at level 6). The title is whitespace-trimmed,
/// matching the Python reference's `rest.strip()`.
fn is_heading(line: &str) -> Option<(usize, &str)> {
    if !line.starts_with('#') {
        return None;
    }
    let n = line.bytes().take_while(|&b| b == b'#').count();
    if n > 6 {
        return None;
    }
    // `n` counts leading ASCII `#` bytes, so it is always a char boundary and
    // `get` always yields `Some`.
    let rest = line.get(n..).unwrap_or("");
    match rest.bytes().next() {
        None => Some((n, "")),
        Some(b) if b == b' ' || b == b'\t' => Some((n, rest.trim())),
        Some(_) => None,
    }
}

/// A fenced-code-block delimiter (` ``` ` or `~~~`), possibly indented.
fn is_fence(line: &str) -> bool {
    let s = line.trim_start();
    s.starts_with("```") || s.starts_with("~~~")
}

// ─── markdown_get_toc ───────────────────────────────────────────────────────

/// Returns a navigation summary of a markdown file: the preamble (text
/// before the first heading) followed by every heading with its line
/// number.
///
/// Use `max_depth` to filter when the doc is huge (`0` = all).
#[must_use]
pub fn get_toc(text: &str, max_depth: usize) -> String {
    const PREAMBLE_PEEK: usize = 40;

    let lines: Vec<&str> = text.lines().collect();
    let mut headings: Vec<(usize, usize, &str)> = Vec::new();
    let mut in_fence = false;
    for (i, line) in lines.iter().enumerate() {
        if is_fence(line) {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if let Some((level, _title)) = is_heading(line) {
            headings.push((i.saturating_add(1), level, line));
        }
    }

    let pre_end = headings.first().map_or(lines.len(), |h| h.0.saturating_sub(1));
    let mut out: Vec<String> = Vec::new();

    if pre_end > 0 {
        let peek = pre_end.min(PREAMBLE_PEEK);
        let suffix = if pre_end <= PREAMBLE_PEEK {
            String::new()
        } else {
            format!(" of {pre_end}")
        };
        out.push(format!("--- preamble (lines 1..{peek}{suffix}) ---"));
        for (i, line) in lines.iter().take(peek).enumerate() {
            out.push(format!("{:>5}| {}", i.saturating_add(1), line));
        }
        if pre_end > PREAMBLE_PEEK {
            out.push(format!("     | ... preamble continues to line {pre_end}"));
        }
        out.push(String::new());
    }

    out.push("--- headings ---".to_string());
    let mut shown: usize = 0;
    for (ln, level, raw) in &headings {
        if max_depth > 0 && *level > max_depth {
            continue;
        }
        out.push(format!("L{ln:<6} {raw}"));
        shown = shown.saturating_add(1);
    }
    if shown == 0 {
        out.push("(no headings)".to_string());
    }

    out.join("\n")
}

// ─── markdown_grep ──────────────────────────────────────────────────────────

/// Search a markdown file for `pattern`; returns matches grouped under
/// their enclosing section heading.
///
/// `pattern` is a Rust `regex` crate
/// pattern (PCRE-flavored, no backreferences/lookaround) matched against
/// each line. Prefix with `(?i)` for case-insensitive.
///
/// # Errors
///
/// Returns `Err` if `pattern` does not compile as a `regex` pattern (the message
/// quotes the pattern and the compiler's complaint). A file with no matches is
/// `Ok`, not an error.
pub fn grep(text: &str, path: &str, pattern: &str) -> Result<String, String> {
    const MAX_MATCHES: usize = 200;
    const LINE_TRUNC: usize = 250;

    let rx = regex::Regex::new(pattern).map_err(|e| format!("invalid regex {pattern:?}: {e}"))?;

    let mut sections: std::collections::HashMap<String, Vec<(usize, String)>> = std::collections::HashMap::new();
    let mut order: Vec<String> = Vec::new();
    let mut current = "(before first heading)".to_string();
    let mut in_fence = false;

    for (i, line) in text.lines().enumerate() {
        if is_fence(line) {
            in_fence = !in_fence;
        } else if !in_fence && let Some((_level, title)) = is_heading(line) {
            current = title.to_string();
        }
        if rx.is_match(line) {
            sections
                .entry(current.clone())
                .or_insert_with(|| {
                    order.push(current.clone());
                    Vec::new()
                })
                .push((i.saturating_add(1), line.to_string()));
        }
    }

    if order.is_empty() {
        return Ok(format!("no matches for {pattern:?} in {path}"));
    }

    let total: usize = sections.values().map(std::vec::Vec::len).sum();
    let mut out: Vec<String> = vec![format!("{total} matches in {} section(s) of {path}", order.len())];
    let mut shown = 0;
    for sec in &order {
        // Every entry in `order` was pushed when its `sections` entry was
        // created, so the lookup always hits.
        let Some(ms) = sections.get(sec) else { continue };
        out.push(format!("\n[{}] ({})", sec, ms.len()));
        for (ln, line) in ms {
            if shown >= MAX_MATCHES {
                out.push(format!(
                    "  ... (truncated, {} more matches; refine the pattern)",
                    total.saturating_sub(shown)
                ));
                return Ok(out.join("\n"));
            }
            let l = if line.chars().count() <= LINE_TRUNC {
                line.clone()
            } else {
                let head: String = line.chars().take(LINE_TRUNC).collect();
                format!("{head} …")
            };
            out.push(format!("  L{ln}: {l}"));
            shown = shown.saturating_add(1);
        }
    }

    Ok(out.join("\n"))
}

// ─── Tool impls ─────────────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema)]
pub struct GetTocArgs {
    /// Path to the markdown file, relative to the workspace root (e.g.
    /// `"docs/cuda.md"`).
    pub path: String,
    /// Only show headings up to this nesting level (1 = `#` only,
    /// 2 = `#` + `##`, …). 0 (default) = all levels.
    #[serde(default)]
    pub max_depth: u32,
}

pub struct MarkdownGetToc {
    pub sandbox: Arc<Sandbox>,
}

impl Tool for MarkdownGetToc {
    type Args = GetTocArgs;
    const NAME: &'static str = "markdown_get_toc";
    const DESCRIPTION: &'static str = "Navigation summary of a markdown file: preamble (lines before the first heading, capped \
         at the first 40) followed by every heading with its line number. Use this BEFORE reading \
         a long doc, then `read` the file at the relevant heading's line (via `offset`) — and \
         `view` any `![alt](path)` figures you find there.";

    async fn call(&self, args: GetTocArgs) -> ToolOutput {
        let text = self.sandbox.read(&args.path).map_err(|e| e.to_string())?;
        let max_depth = usize::try_from(args.max_depth)
            .map_err(|_| format!("max_depth {} does not fit this platform's usize", args.max_depth))?;
        Ok(get_toc(&text, max_depth).into())
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct GrepArgs {
    /// Path to the markdown file, relative to the workspace root.
    pub path: String,
    /// Rust `regex` crate pattern matched against each line. Prefix with
    /// `(?i)` for case-insensitive.
    pub pattern: String,
}

pub struct MarkdownGrep {
    pub sandbox: Arc<Sandbox>,
}

impl Tool for MarkdownGrep {
    type Args = GrepArgs;
    const NAME: &'static str = "markdown_grep";
    const DESCRIPTION: &'static str = "Regex-search a markdown file; matches are grouped under their enclosing section \
         heading and capped at 200 matches. Prefer this over the general `grep` on `.md` files — \
         the section context lets you jump directly to the right heading, then `read` at that line.";

    async fn call(&self, args: GrepArgs) -> ToolOutput {
        let text = self.sandbox.read(&args.path).map_err(|e| e.to_string())?;
        grep(&text, &args.path, &args.pattern).map(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
intro line one
intro line two

# First heading

content one
content two

## Sub of first

```python
# this is a comment, not a heading
#include <stdio.h>
def foo(): return 1
```

text after the fence

## Another sub

stuff here

# Second heading

last paragraph
";

    #[test]
    fn is_heading_accepts_atx() {
        assert_eq!(is_heading("# Hello"), Some((1, "Hello")));
        assert_eq!(is_heading("## Sub heading "), Some((2, "Sub heading")));
        assert_eq!(is_heading("###### Deep"), Some((6, "Deep")));
    }

    #[test]
    fn is_heading_rejects_non_headings() {
        assert_eq!(is_heading(""), None);
        assert_eq!(is_heading("plain text"), None);
        assert_eq!(is_heading("#include <stdio.h>"), None, "no space after #");
        assert_eq!(is_heading("#define FOO 1"), None);
        assert_eq!(is_heading("####### too deep"), None);
    }

    #[test]
    fn is_fence_detects_both_styles() {
        assert!(is_fence("```"));
        assert!(is_fence("```python"));
        assert!(is_fence("    ~~~"));
        assert!(!is_fence("plain"));
        assert!(!is_fence("`inline`"));
    }

    #[test]
    fn get_toc_lists_headings_with_line_numbers() {
        let toc = get_toc(SAMPLE, 0);
        assert!(toc.contains("--- preamble (lines 1..3) ---"), "{toc}");
        assert!(toc.contains("    1| intro line one"), "{toc}");
        assert!(toc.contains("# First heading"), "{toc}");
        assert!(toc.contains("## Sub of first"), "{toc}");
        assert!(toc.contains("# Second heading"), "{toc}");
        // Both #-comment and #include are inside a fenced block — must not
        // be confused for headings.
        assert!(!toc.contains("this is a comment"), "{toc}");
        assert!(!toc.contains("#include"), "{toc}");
    }

    #[test]
    fn get_toc_max_depth_filters_subsections() {
        let toc = get_toc(SAMPLE, 1);
        assert!(toc.contains("# First heading"));
        assert!(toc.contains("# Second heading"));
        assert!(!toc.contains("## Sub of first"));
        assert!(!toc.contains("## Another sub"));
    }

    #[test]
    fn get_toc_handles_no_headings() {
        let toc = get_toc("just\nsome\ntext\n", 0);
        assert!(toc.contains("(no headings)"));
        assert!(toc.contains("just"));
    }

    #[test]
    fn grep_groups_by_enclosing_section() {
        let out = grep(SAMPLE, "test.md", "content").unwrap();
        // 'content' appears in the body under "First heading", twice.
        assert!(out.contains("[First heading]"), "{out}");
        assert!(out.contains("L6: content one"), "{out}");
        assert!(out.contains("L7: content two"), "{out}");
    }

    #[test]
    fn grep_matches_inside_fenced_block_keep_their_section() {
        // The `def foo` line is inside a fenced block under "Sub of first".
        // Even though we don't update `current` mid-fence, the previous
        // section is still attributed.
        let out = grep(SAMPLE, "test.md", r"def foo").unwrap();
        assert!(out.contains("[Sub of first]"), "{out}");
    }

    #[test]
    fn grep_no_matches() {
        let out = grep(SAMPLE, "test.md", "definitely-not-here").unwrap();
        assert!(out.contains("no matches"), "{out}");
    }

    #[test]
    fn grep_invalid_pattern_errors() {
        let err = grep(SAMPLE, "test.md", "(unbalanced").unwrap_err();
        assert!(err.contains("invalid regex"), "{err}");
    }
}
