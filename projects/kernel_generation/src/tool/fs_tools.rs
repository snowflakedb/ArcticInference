//! `grep` / `find` / `ls` — general filesystem search, ported from pi.
//!
//! pi shells out to ripgrep/fd; kernelguy bundles neither and can't assume
//! they're on PATH, so these are pure-Rust: the `ignore` crate provides the
//! gitignore-aware walk (ripgrep's own walker; `.require_git(false)` so
//! `.gitignore`/`.ignore` are honored even in the non-git workspace), `globset`
//! the glob matching, and the existing `regex` crate the content search. Path
//! args are resolved through [`Sandbox::host_path`], which rejects `..`/absolute
//! escapes and resolves overlays. NOTE: `host_path` does not canonicalize
//! symlinks, so a symlink *created in the workspace* (via `bash`) whose target
//! is outside can still be followed by these host-side reads — same as the
//! pre-existing `read`/`Sandbox::read`. Confining that is a separate sandbox-
//! hardening task (must whitelist overlay roots), tracked outside this change.
//!
//! Caps mirror pi (whichever hits first wins, with an actionable notice):
//! - `grep`: 100 matches, 500 chars/line, 50 KiB total.
//! - `find`: 1000 results, 50 KiB total.
//! - `ls`: 500 entries, 50 KiB total.
//!
//! These are *reproducible* (re-query with a narrower pattern / higher limit),
//! so they truncate + hint and never spill to a file.
//!
//! Functional-core: the formatters/notice-builders are pure; only the walk and
//! file reads touch the filesystem.

use std::sync::Arc;

use globset::{Glob, GlobBuilder};
use ignore::WalkBuilder;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::exec::sandbox::Sandbox;
use crate::tool::truncate::{GREP_MAX_LINE_LEN, MAX_OUTPUT_BYTES, format_size, truncate_head, truncate_line};
use crate::tool::{Tool, ToolOutput};

const GREP_MATCH_LIMIT: usize = 100;
const FIND_RESULT_LIMIT: usize = 1000;
const LS_ENTRY_LIMIT: usize = 500;

/// Apply the shared byte cap and append a bracketed notice built from the
/// collected reasons. Pure. `raw` already had its rows capped by the caller.
fn finish(raw: &str, mut notices: Vec<String>) -> String {
    let trunc = truncate_head(raw, usize::MAX, MAX_OUTPUT_BYTES);
    if trunc.truncated {
        notices.push(format!("{} limit reached", format_size(MAX_OUTPUT_BYTES)));
    }
    if notices.is_empty() {
        trunc.content
    } else {
        format!("{}\n\n[{}]", trunc.content, notices.join(". "))
    }
}

/// A rel path in forward-slash form (workspace/overlay paths are already `/`).
fn to_posix(p: &std::path::Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

// ─── grep ─────────────────────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema)]
pub struct GrepArgs {
    /// Search pattern. A Rust `regex` by default; set `literal=true` to match it
    /// verbatim.
    pub pattern: String,
    /// Directory or file to search, workspace-relative (default: workspace root).
    #[serde(default)]
    pub path: Option<String>,
    /// Only search files whose relative path matches this glob (e.g. `*.rs`,
    /// `**/*.cu`).
    #[serde(default)]
    pub glob: Option<String>,
    /// Case-insensitive search (default false).
    #[serde(default)]
    pub ignore_case: Option<bool>,
    /// Treat `pattern` as a literal string, not a regex (default false).
    #[serde(default)]
    pub literal: Option<bool>,
    /// Lines of context to show before and after each match (default 0).
    #[serde(default)]
    pub context: Option<usize>,
    /// Max matches to return (default 100).
    #[serde(default)]
    pub limit: Option<usize>,
}

pub struct Grep {
    pub sandbox: Arc<Sandbox>,
}

/// Collect the `(display_path, host_path)` pairs `grep` should search: every
/// non-ignored file under `root` when it is a directory (`.git` pruned, `glob`
/// applied, sorted), or `root` alone when it is a single file.
fn collect_grep_files(
    root: &std::path::Path,
    rel: &str,
    glob: Option<&globset::GlobMatcher>,
) -> Vec<(String, std::path::PathBuf)> {
    let mut files: Vec<(String, std::path::PathBuf)> = Vec::new();
    if root.is_dir() {
        for dent in WalkBuilder::new(root).hidden(false).require_git(false).build() {
            let Ok(dent) = dent else { continue };
            if dent.depth() == 0 {
                continue; // the search root itself
            }
            let p = dent.path();
            if p.components().any(|c| c.as_os_str() == ".git") {
                continue;
            }
            if !dent.file_type().is_some_and(|ft| ft.is_file()) {
                continue;
            }
            let relp = p.strip_prefix(root).unwrap_or(p);
            let display = to_posix(relp);
            if let Some(g) = glob
                && !g.is_match(relp)
            {
                continue;
            }
            files.push((display, p.to_path_buf()));
        }
        files.sort();
    } else {
        let display = root
            .file_name()
            .map_or_else(|| rel.to_string(), |n| n.to_string_lossy().into_owned());
        files.push((display, root.to_path_buf()));
    }
    files
}

impl Tool for Grep {
    type Args = GrepArgs;
    const NAME: &'static str = "grep";
    const DESCRIPTION: &'static str = "Search file contents for a pattern (Rust regex, or literal with literal=true). Returns \
         matching lines as `path:line: text`, capped at 100 matches / 50KB / 500 chars per line, and \
         respects .gitignore. Filter files with `glob` (e.g. `*.cu`); `context` adds surrounding lines. \
         Prefer `markdown_grep` on `.md` files — it groups matches by heading.";

    async fn call(&self, args: GrepArgs) -> ToolOutput {
        let rel = args.path.as_deref().unwrap_or(".");
        let root = self.sandbox.host_path(rel).map_err(|e| e.to_string())?;
        if !root.exists() {
            return Err(format!("path not found: {rel}"));
        }

        // Build the matcher (regex crate; literal → escaped; ignore_case → (?i)).
        let mut pat = if args.literal.unwrap_or(false) {
            regex::escape(&args.pattern)
        } else {
            args.pattern.clone()
        };
        if args.ignore_case.unwrap_or(false) {
            pat = format!("(?i){pat}");
        }
        let re = regex::Regex::new(&pat).map_err(|e| format!("invalid regex {:?}: {e}", args.pattern))?;

        let glob = match &args.glob {
            Some(g) => Some(
                Glob::new(g)
                    .map_err(|e| format!("invalid glob {g:?}: {e}"))?
                    .compile_matcher(),
            ),
            None => None,
        };

        let context = args.context.unwrap_or(0);
        let limit = args.limit.unwrap_or(GREP_MATCH_LIMIT).max(1);

        // Collect (display_path, file_path) pairs to search.
        let files = collect_grep_files(&root, rel, glob.as_ref());

        let mut out_lines: Vec<String> = Vec::new();
        let mut match_count = 0usize;
        let mut match_limit_reached = false;
        let mut lines_truncated = false;
        'outer: for (display, path) in files {
            // Skip binary / non-UTF-8 files (ripgrep does the same).
            let Ok(body) = std::fs::read_to_string(&path) else {
                continue;
            };
            let lines: Vec<&str> = body.split('\n').collect();
            for (i, line) in lines.iter().enumerate() {
                if !re.is_match(line) {
                    continue;
                }
                let ln = i.saturating_add(1);
                let (start, end) = if context > 0 {
                    (
                        ln.saturating_sub(context).max(1),
                        ln.saturating_add(context).min(lines.len()),
                    )
                } else {
                    (ln, ln)
                };
                // Take the window as a slice so the 1-based→0-based shift is
                // checked once here rather than per element.
                let window = lines.get(start.saturating_sub(1)..end).unwrap_or_default();
                for (nth, raw) in window.iter().enumerate() {
                    let c = start.saturating_add(nth);
                    let (text, cut) = truncate_line(raw, GREP_MAX_LINE_LEN);
                    if cut {
                        lines_truncated = true;
                    }
                    if c == ln {
                        out_lines.push(format!("{display}:{c}: {text}"));
                    } else {
                        out_lines.push(format!("{display}-{c}- {text}"));
                    }
                }
                match_count = match_count.saturating_add(1);
                if match_count >= limit {
                    match_limit_reached = true;
                    break 'outer;
                }
            }
        }

        if match_count == 0 {
            return Ok("No matches found".into());
        }

        let mut notices = Vec::new();
        if match_limit_reached {
            notices.push(format!(
                "{limit} matches limit reached. Use limit={} for more, or refine pattern",
                limit.saturating_mul(2)
            ));
        }
        if lines_truncated {
            notices.push(format!(
                "Some lines truncated to {GREP_MAX_LINE_LEN} chars. Use read tool to see full lines"
            ));
        }
        Ok(finish(&out_lines.join("\n"), notices).into())
    }
}

// ─── find ─────────────────────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema)]
pub struct FindArgs {
    /// Glob to match, e.g. `*.cu`, `**/*.json`, `src/**/*.rs`. A pattern with no
    /// `/` matches file names anywhere; a pattern with `/` matches the relative
    /// path.
    pub pattern: String,
    /// Directory to search, workspace-relative (default: workspace root).
    #[serde(default)]
    pub path: Option<String>,
    /// Max results to return (default 1000).
    #[serde(default)]
    pub limit: Option<usize>,
}

pub struct Find {
    pub sandbox: Arc<Sandbox>,
}

impl Tool for Find {
    type Args = FindArgs;
    const NAME: &'static str = "find";
    const DESCRIPTION: &'static str = "Find files/dirs by glob pattern. Returns paths relative to the search directory, \
         capped at 1000 results / 50KB, and respects .gitignore. A pattern without `/` (e.g. `*.cu`) \
         matches names at any depth; a pattern with `/` (e.g. `src/**/*.rs`) matches the relative path.";

    async fn call(&self, args: FindArgs) -> ToolOutput {
        let rel = args.path.as_deref().unwrap_or(".");
        let root = self.sandbox.host_path(rel).map_err(|e| e.to_string())?;
        if !root.exists() {
            return Err(format!("path not found: {rel}"));
        }
        let limit = args.limit.unwrap_or(FIND_RESULT_LIMIT).max(1);

        // A pattern with '/' matches the whole relative path (with `**` crossing
        // dirs); otherwise it matches the basename at any depth (fd semantics).
        let has_sep = args.pattern.contains('/');
        let matcher = if has_sep {
            let mut p = args.pattern.clone();
            if !p.starts_with('/') && !p.starts_with("**/") && p != "**" {
                p = format!("**/{p}");
            }
            GlobBuilder::new(&p)
                .literal_separator(true)
                .build()
                .map_err(|e| format!("invalid glob {:?}: {e}", args.pattern))?
                .compile_matcher()
        } else {
            Glob::new(&args.pattern)
                .map_err(|e| format!("invalid glob {:?}: {e}", args.pattern))?
                .compile_matcher()
        };

        let mut results: Vec<String> = Vec::new();
        let mut limit_reached = false;
        for dent in WalkBuilder::new(&root).hidden(false).require_git(false).build() {
            let Ok(dent) = dent else { continue };
            if dent.depth() == 0 {
                continue;
            }
            let p = dent.path();
            if p.components().any(|c| c.as_os_str() == ".git") {
                continue;
            }
            let relp = p.strip_prefix(&root).unwrap_or(p);
            let hit = if has_sep {
                matcher.is_match(relp)
            } else {
                // Match the basename only.
                p.file_name().is_some_and(|n| matcher.is_match(n))
            };
            if !hit {
                continue;
            }
            results.push(to_posix(relp));
            if results.len() >= limit {
                limit_reached = true;
                break;
            }
        }

        if results.is_empty() {
            return Ok("No files found matching pattern".into());
        }
        results.sort();

        let mut notices = Vec::new();
        if limit_reached {
            notices.push(format!(
                "{limit} results limit reached. Use limit={} for more, or refine pattern",
                limit.saturating_mul(2)
            ));
        }
        Ok(finish(&results.join("\n"), notices).into())
    }
}

// ─── ls ─────────────────────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema)]
pub struct LsArgs {
    /// Directory to list, workspace-relative (default: workspace root).
    #[serde(default)]
    pub path: Option<String>,
    /// Max entries to return (default 500).
    #[serde(default)]
    pub limit: Option<usize>,
}

pub struct Ls {
    pub sandbox: Arc<Sandbox>,
}

impl Tool for Ls {
    type Args = LsArgs;
    const NAME: &'static str = "ls";
    const DESCRIPTION: &'static str = "List a directory's immediate contents, sorted case-insensitively, with `/` appended to \
         directories and dotfiles included. Capped at 500 entries / 50KB. Not recursive — use `find` to \
         search a tree.";

    async fn call(&self, args: LsArgs) -> ToolOutput {
        let rel = args.path.as_deref().unwrap_or(".");
        let dir = self.sandbox.host_path(rel).map_err(|e| e.to_string())?;
        if !dir.exists() {
            return Err(format!("path not found: {rel}"));
        }
        if !dir.is_dir() {
            return Err(format!("not a directory: {rel}"));
        }

        let mut entries: Vec<(String, bool)> = Vec::new(); // (name, is_dir)
        for dent in std::fs::read_dir(&dir).map_err(|e| format!("cannot read directory: {e}"))? {
            let Ok(dent) = dent else { continue };
            let name = dent.file_name().to_string_lossy().into_owned();
            let is_dir = dent.file_type().is_ok_and(|ft| ft.is_dir());
            entries.push((name, is_dir));
        }
        // Case-insensitive alphabetical sort.
        entries.sort_by_key(|(name, _)| name.to_lowercase());

        let mut limit_reached = false;
        let limit = args.limit.unwrap_or(LS_ENTRY_LIMIT).max(1);
        let mut rendered: Vec<String> = Vec::new();
        for (name, is_dir) in entries {
            if rendered.len() >= limit {
                limit_reached = true;
                break;
            }
            rendered.push(if is_dir { format!("{name}/") } else { name });
        }

        if rendered.is_empty() {
            return Ok("(empty directory)".into());
        }

        let mut notices = Vec::new();
        if limit_reached {
            notices.push(format!(
                "{limit} entries limit reached. Use limit={} for more",
                limit.saturating_mul(2)
            ));
        }
        Ok(finish(&rendered.join("\n"), notices).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;
    use serde_json::json;
    use std::fmt::Write as _;

    fn sb() -> Arc<Sandbox> {
        Arc::new(Sandbox::new().expect("sandbox"))
    }

    fn seed(s: &Sandbox) {
        s.write(
            "src/main.rs",
            "fn main() {\n    let x = 42;\n    println!(\"hi\");\n}\n",
        )
        .unwrap();
        s.write("src/lib.rs", "pub fn helper() -> i32 { 42 }\n").unwrap();
        s.write("docs/notes.md", "# Notes\nthe answer is 42\n").unwrap();
        s.write("README.txt", "top level\n").unwrap();
    }

    // ─── grep ──────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn grep_finds_matches_with_relpath_and_line() {
        let s = sb();
        seed(&s);
        let out = Grep { sandbox: s }
            .call_json(json!({ "pattern": "42" }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("src/main.rs:2: "), "{out}");
        assert!(out.contains("src/lib.rs:1: "), "{out}");
        assert!(out.contains("docs/notes.md:2: "), "{out}");
    }

    #[tokio::test]
    async fn grep_glob_filters_files() {
        let s = sb();
        seed(&s);
        let out = Grep { sandbox: s }
            .call_json(json!({ "pattern": "42", "glob": "*.rs" }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("src/main.rs") && out.contains("src/lib.rs"), "{out}");
        assert!(!out.contains("notes.md"), "glob should exclude .md: {out}");
    }

    #[tokio::test]
    async fn grep_no_matches() {
        let s = sb();
        seed(&s);
        let out = Grep { sandbox: s }
            .call_json(json!({ "pattern": "zzzznotpresent" }))
            .await
            .unwrap()
            .as_text();
        assert_eq!(out, "No matches found");
    }

    #[tokio::test]
    async fn grep_ignore_case_and_literal() {
        let s = sb();
        s.write("f.txt", "Hello WORLD\n(a+b)\n").unwrap();
        // ignore_case
        let out = Grep { sandbox: s.clone() }
            .call_json(json!({ "pattern": "world", "ignore_case": true }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("WORLD"), "{out}");
        // literal: "(a+b)" as regex would be invalid-ish; literal matches verbatim
        let out2 = Grep { sandbox: s }
            .call_json(json!({ "pattern": "(a+b)", "literal": true }))
            .await
            .unwrap()
            .as_text();
        assert!(out2.contains("(a+b)"), "{out2}");
    }

    #[tokio::test]
    async fn grep_context_lines() {
        let s = sb();
        s.write("f.txt", "one\ntwo\nMATCH\nfour\nfive\n").unwrap();
        let out = Grep { sandbox: s }
            .call_json(json!({ "pattern": "MATCH", "context": 1 }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("f.txt-2- two"), "context before: {out}");
        assert!(out.contains("f.txt:3: MATCH"), "match line: {out}");
        assert!(out.contains("f.txt-4- four"), "context after: {out}");
    }

    #[tokio::test]
    async fn grep_match_limit_notice() {
        let s = sb();
        let body: String = (0..250).fold(String::new(), |mut acc, i| {
            let _ = writeln!(acc, "hit line {i}");
            acc
        });
        s.write("big.txt", &body).unwrap();
        let out = Grep { sandbox: s }
            .call_json(json!({ "pattern": "hit", "limit": 100 }))
            .await
            .unwrap()
            .as_text();
        assert!(
            out.contains("100 matches limit reached") && out.contains("limit=200"),
            "{out}"
        );
    }

    #[tokio::test]
    async fn grep_long_line_truncated() {
        let s = sb();
        s.write("f.txt", &format!("prefix {}\n", "x".repeat(1000))).unwrap();
        let out = Grep { sandbox: s }
            .call_json(json!({ "pattern": "prefix" }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("... [truncated]"), "{out}");
        assert!(out.contains("Some lines truncated"), "{out}");
    }

    // ─── find ──────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn find_by_extension_glob_matches_any_depth() {
        let s = sb();
        seed(&s);
        let out = Find { sandbox: s }
            .call_json(json!({ "pattern": "*.rs" }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("src/main.rs") && out.contains("src/lib.rs"), "{out}");
        assert!(!out.contains("notes.md"), "{out}");
    }

    #[tokio::test]
    async fn find_path_glob() {
        let s = sb();
        seed(&s);
        let out = Find { sandbox: s }
            .call_json(json!({ "pattern": "src/*.rs" }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("src/main.rs"), "{out}");
    }

    #[tokio::test]
    async fn find_no_match() {
        let s = sb();
        seed(&s);
        let out = Find { sandbox: s }
            .call_json(json!({ "pattern": "*.zzz" }))
            .await
            .unwrap()
            .as_text();
        assert_eq!(out, "No files found matching pattern");
    }

    // ─── ls ──────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn ls_lists_sorted_with_dir_slash() {
        let s = sb();
        seed(&s);
        let out = Ls { sandbox: s }
            .call_json(json!({ "path": "." }))
            .await
            .unwrap()
            .as_text();
        let lines: Vec<&str> = out.lines().collect();
        assert!(lines.contains(&"src/"), "dir gets slash: {out}");
        assert!(lines.contains(&"docs/"), "{out}");
        assert!(lines.contains(&"README.txt"), "{out}");
        // Case-insensitive sort: docs, README, src.
        let pos = |name: &str| lines.iter().position(|l| *l == name).unwrap();
        assert!(
            pos("docs/") < pos("README.txt") && pos("README.txt") < pos("src/"),
            "{out}"
        );
    }

    #[tokio::test]
    async fn ls_subdir_and_dotfiles() {
        let s = sb();
        s.write("d/.hidden", "x").unwrap();
        s.write("d/visible.txt", "y").unwrap();
        let out = Ls { sandbox: s }
            .call_json(json!({ "path": "d" }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains(".hidden"), "dotfiles included: {out}");
        assert!(out.contains("visible.txt"), "{out}");
    }

    #[tokio::test]
    async fn ls_empty_directory() {
        let s = sb();
        std::fs::create_dir_all(s.host_path_for_write("empty").unwrap()).unwrap();
        let out = Ls { sandbox: s }
            .call_json(json!({ "path": "empty" }))
            .await
            .unwrap()
            .as_text();
        assert_eq!(out, "(empty directory)");
    }

    #[tokio::test]
    async fn ls_entry_limit_notice() {
        let s = sb();
        for i in 0..600 {
            s.write(&format!("many/f{i:04}.txt"), "x").unwrap();
        }
        let out = Ls { sandbox: s }
            .call_json(json!({ "path": "many", "limit": 500 }))
            .await
            .unwrap()
            .as_text();
        assert!(
            out.contains("500 entries limit reached") && out.contains("limit=1000"),
            "{out}"
        );
    }

    #[tokio::test]
    async fn ls_missing_path_errors() {
        let s = sb();
        let result = Ls { sandbox: s }.call_json(json!({ "path": "nope" })).await;
        assert!(result.is_err(), "missing dir must error: {result:?}");
    }

    #[tokio::test]
    async fn tools_resolve_through_ro_overlay() {
        // grep/find/ls must see content mounted via add_ro (like mount_docs).
        let upstream = tempfile::TempDir::new().unwrap();
        std::fs::write(upstream.path().join("spec.md"), "the token is 42\n").unwrap();
        let mut s = Sandbox::new().unwrap();
        s.add_ro(upstream.path(), "/workspace/refs");
        let s = Arc::new(s);

        let ls = Ls { sandbox: s.clone() }
            .call_json(json!({ "path": "refs" }))
            .await
            .unwrap()
            .as_text();
        assert!(ls.contains("spec.md"), "ls overlay: {ls}");

        let grep = Grep { sandbox: s }
            .call_json(json!({ "pattern": "token", "path": "refs" }))
            .await
            .unwrap()
            .as_text();
        assert!(grep.contains("spec.md:1: "), "grep overlay: {grep}");
    }
}
