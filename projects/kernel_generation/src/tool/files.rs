//! `read` / `write` — typed workspace file ops.
//!
//! Both bottom out at [`Sandbox::read`] / [`Sandbox::write`], so they
//! automatically respect read-only overlays — a `write` against a file mounted
//! via `mount_docs` (or any `add_ro` overlay) returns an error the model can
//! recover from, rather than silently shadowing the upstream.
//!
//! These give the agent a stable typed surface that doesn't have to go through
//! `bash` (`sed`/`awk`/heredoc gymnastics, prone to shell-quoting errors). The
//! argument JSON is what the agent thinks about.
//!
//! Behavior:
//! - `read` returns text content with 1-based line-number prefixing (on by
//!   default), 1-indexed `offset`, capped at 2000 lines / 50 KiB with a
//!   continuation notice; image paths are redirected to `view`.
//! - `write` always overwrites; parent directories are created on demand, and
//!   the write is serialized under the per-path lock. Precise in-place edits
//!   live in the `edit` tool ([`crate::tool::edit`]).
//!
//! Both respect the same `path` style accepted by [`Sandbox::path`]:
//! workspace-relative (`"src/foo.py"`) or in-sandbox absolute
//! (`"/workspace/src/foo.py"`).

use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::exec::sandbox::Sandbox;
use crate::tool::truncate::{MAX_OUTPUT_BYTES, MAX_OUTPUT_LINES, TruncatedBy, format_size, truncate_head};
use crate::tool::{Tool, ToolOutput};

const MARKDOWN_IMAGE_EXTS: [&str; 5] = ["png", "jpg", "jpeg", "gif", "webp"];

/// True for image extensions `read` can't render as text. Such a path is
/// short-circuited to a note pointing at `view` (which accepts exactly these
/// formats), rather than returning a UTF-8-decode error or a wall of binary.
fn is_image_path(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| MARKDOWN_IMAGE_EXTS.iter().any(|e| ext.eq_ignore_ascii_case(e)))
}

/// Defensively unwrap a text argument that a model serialized as Anthropic
/// "content blocks" instead of a raw string — e.g. `[{"type":"text",
/// "text":"..."}]` or just `[{"text":"..."}]`. When a model double-encodes
/// `write`/`replace` content this way, the literal JSON would otherwise be
/// written to the file, corrupting it and costing a turn to notice and redo.
///
/// Returns the concatenated block text when (and only when) the whole string
/// is unmistakably such an envelope: a JSON array/object whose every element
/// is an object carrying a string `text` (and no non-`text` `type`). Anything
/// else — ordinary code, prose, or a genuine JSON file — returns `None` and is
/// written verbatim.
fn unwrap_content_blocks(s: &str) -> Option<String> {
    fn block_text(v: &Value) -> Option<&str> {
        let obj = v.as_object()?;
        // A text block omits `type` or sets it to "text"; reject image /
        // tool_use / anything unknown so we never silently drop content.
        match obj.get("type") {
            None => {}
            Some(Value::String(t)) if t == "text" => {}
            _ => return None,
        }
        obj.get("text")?.as_str()
    }

    let trimmed = s.trim();
    // Fast path: a content-block envelope is always a JSON array/object, so
    // the overwhelming majority of inputs (source code, prose) bail out here
    // without even attempting a parse.
    if !trimmed.starts_with('[') && !trimmed.starts_with('{') {
        return None;
    }
    let value: Value = serde_json::from_str(trimmed).ok()?;

    match &value {
        Value::Array(items) if !items.is_empty() => {
            let mut out = String::new();
            for item in items {
                out.push_str(block_text(item)?);
            }
            Some(out)
        }
        Value::Object(_) => block_text(&value).map(str::to_string),
        _ => None,
    }
}

// ─── read ───────────────────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema)]
pub struct ReadArgs {
    /// Workspace-relative path of the file to read (e.g. `src/foo.py`).
    pub path: String,
    /// Line number (1-indexed) to start reading at. Default 1 (start of file).
    #[serde(default)]
    pub offset: Option<usize>,
    /// How many lines to read starting from `offset`. Default = read to
    /// end-of-file. Pass a small number to chunk through long files.
    #[serde(default)]
    pub limit: Option<usize>,
    /// Prefix each line with its 1-based line number (default true).
    /// Pass `false` for raw output — useful when copying a small markdown
    /// or config file verbatim into reasoning.
    #[serde(default)]
    pub line_numbers: Option<bool>,
}

pub struct Read {
    pub sandbox: Arc<Sandbox>,
}

impl Tool for Read {
    type Args = ReadArgs;
    const NAME: &'static str = "read";
    const DESCRIPTION: &'static str = "Read a file from the project workspace. Defaults: 1-based line numbers (`   1| ...`), \
         capped at 2000 lines / 50KB. If the file is longer the reply ends with the exact `offset` to \
         continue from (offset is 1-indexed). Use `offset` / `limit` to chunk through long files; pass \
         `line_numbers=false` for raw output. Images aren't text — use `view` to see them.";

    async fn call(&self, args: ReadArgs) -> ToolOutput {
        // Images aren't text — point the model at `view` instead of returning a
        // decode error or binary garbage.
        if is_image_path(&args.path) {
            return Ok(format!(
                "[{} is a binary/image file — a text read isn't meaningful. Use `view` with this path to see it.]",
                args.path
            )
            .into());
        }
        let text = self.sandbox.read(&args.path).map_err(|e| e.to_string())?;
        let all_lines: Vec<&str> = text.split('\n').collect();
        let total = all_lines.len();
        let line_numbers = args.line_numbers.unwrap_or(true);

        // `offset` is 1-indexed (matches the 1-based line-number display, so an
        // echoed `offset=` round-trips). Default = start of file; 0 is treated as 1.
        let offset = args.offset.unwrap_or(1).max(1);
        let start = offset.saturating_sub(1); // 0-indexed array position
        if start >= total {
            return Err(format!("offset {offset} is beyond end of file ({total} lines total)"));
        }

        // Honor an explicit `limit` first (pi semantics); otherwise take to EOF
        // and let truncate_head apply the cap. `start < total` above makes both
        // ranges in-bounds, so neither `get` can miss.
        let (selected, user_limited) = args.limit.map_or_else(
            || (all_lines.get(start..).unwrap_or_default().join("\n"), None),
            |limit| {
                let end = start.saturating_add(limit).min(total);
                (
                    all_lines.get(start..end).unwrap_or_default().join("\n"),
                    Some(end.saturating_sub(start)),
                )
            },
        );

        let trunc = truncate_head(&selected, MAX_OUTPUT_LINES, MAX_OUTPUT_BYTES);

        // Giant first line: no whole-line slice fits the byte cap. Point at a
        // byte-range bash fallback rather than return an empty block.
        if trunc.first_line_exceeds {
            let first_line_bytes = all_lines.get(start).map_or(0, |line| line.len());
            return Ok(format!(
                "[Line {offset} is {}, exceeds {} limit. Use bash: sed -n '{offset}p' {} | head -c {}]",
                format_size(first_line_bytes),
                format_size(MAX_OUTPUT_BYTES),
                args.path,
                MAX_OUTPUT_BYTES
            )
            .into());
        }

        // Prefix the kept lines with their absolute (1-based) line numbers.
        let body = if trunc.content.is_empty() {
            String::new()
        } else if line_numbers {
            trunc
                .content
                .split('\n')
                .enumerate()
                .map(|(i, line)| format!("{:>4}| {line}", offset.saturating_add(i)))
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            trunc.content.clone()
        };

        // Continuation notice — pi's three variants (lines / bytes / user-limit).
        let notice = if trunc.truncated {
            let end_display = offset.saturating_add(trunc.output_lines).saturating_sub(1);
            let next = end_display.saturating_add(1);
            if trunc.truncated_by == Some(TruncatedBy::Lines) {
                format!("\n\n[Showing lines {offset}-{end_display} of {total}. Use offset={next} to continue.]")
            } else {
                format!(
                    "\n\n[Showing lines {offset}-{end_display} of {total} ({} limit). Use offset={next} to continue.]",
                    format_size(MAX_OUTPUT_BYTES)
                )
            }
        } else if let Some(u) = user_limited
            && start.saturating_add(u) < total
        {
            let shown = start.saturating_add(u);
            let remaining = total.saturating_sub(shown);
            let next = shown.saturating_add(1); // 1-indexed next line
            format!("\n\n[{remaining} more lines in file. Use offset={next} to continue.]")
        } else {
            String::new()
        };

        Ok(format!("{body}{notice}").into())
    }
}

// ─── write ──────────────────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema)]
pub struct WriteArgs {
    /// Workspace-relative path of the file to write (e.g. `src/foo.py`).
    /// Parent directories are created on demand.
    pub path: String,
    /// File contents (plain text). Always overwrites if the file exists.
    pub file_text: String,
}

pub struct Write {
    pub sandbox: Arc<Sandbox>,
}

impl Tool for Write {
    type Args = WriteArgs;
    const NAME: &'static str = "write";
    const DESCRIPTION: &'static str = "Write a file in the project workspace, overwriting if it exists. Parent dirs are \
         created. Errors if `path` is covered by a read-only overlay (e.g. files mounted \
         via `mount_docs`).";

    async fn call(&self, args: WriteArgs) -> ToolOutput {
        let content = unwrap_content_blocks(&args.file_text).unwrap_or(args.file_text);
        // Hold the per-path lock across the write so it can't interleave with a
        // concurrent edit/write of the same file.
        let _guard = self.sandbox.lock_path(&args.path).await.map_err(|e| e.to_string())?;
        self.sandbox
            .write(&args.path, &content)
            // Real UTF-8 byte count (Rust `str::len`), not a UTF-16 code-unit
            // count — pi's `.length` over-reports for multibyte content.
            .map(|_| format!("Successfully wrote {} bytes", content.len()).into())
            .map_err(|e| e.to_string())
    }
}

// ─── replace ────────────────────────────────────────────────────────────────

// ─── replace ────────────────────────────────────────────────────────────────
// Removed: `replace` was superseded by the `edit` tool (see `crate::tool::edit`),
// which supports multi-edit, CRLF/BOM handling, and fuzzy matching. `write`'s
// `unwrap_content_blocks` helper stays above (write still uses it).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;
    use serde_json::json;
    use tempfile::TempDir;

    fn fresh_sandbox() -> Arc<Sandbox> {
        Arc::new(Sandbox::new().expect("sandbox"))
    }

    // ─── read ───────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn read_returns_file_with_line_numbers_by_default() {
        let sb = fresh_sandbox();
        sb.write("notes.txt", "alpha\nbeta\ngamma").unwrap();
        let tool = Read { sandbox: sb };
        let out = tool.call_json(json!({ "path": "notes.txt" })).await.unwrap();
        assert_eq!(out.as_text(), "   1| alpha\n   2| beta\n   3| gamma");
    }

    #[tokio::test]
    async fn read_respects_offset_and_limit() {
        let sb = fresh_sandbox();
        sb.write("notes.txt", "a\nb\nc\nd\ne").unwrap();
        let tool = Read { sandbox: sb };

        let out = tool
            .call_json(json!({ "path": "notes.txt", "offset": 1, "limit": 2 }))
            .await
            .unwrap();
        // offset is 1-indexed: offset=1 starts at line 1. Line numbers are
        // absolute. Reading 2 of 5 lines stops early, so a continuation hint follows.
        let text = out.as_text();
        assert!(text.starts_with("   1| a\n   2| b"), "{text}");
        assert!(
            text.contains("3 more lines in file") && text.contains("offset=3"),
            "continuation hint missing: {text}"
        );
    }

    #[tokio::test]
    async fn read_offset_is_one_indexed() {
        // offset=2 (1-indexed) skips exactly the first line.
        let sb = fresh_sandbox();
        sb.write("notes.txt", "a\nb\nc").unwrap();
        let tool = Read { sandbox: sb };
        let out = tool
            .call_json(json!({ "path": "notes.txt", "offset": 2 }))
            .await
            .unwrap();
        assert_eq!(out.as_text(), "   2| b\n   3| c");
    }

    #[tokio::test]
    async fn read_offset_beyond_eof_errors() {
        let sb = fresh_sandbox();
        sb.write("notes.txt", "a\nb\nc").unwrap();
        let tool = Read { sandbox: sb };
        let result = tool.call_json(json!({ "path": "notes.txt", "offset": 99 })).await;
        assert!(result.is_err(), "offset past EOF must error: {result:?}");
        assert!(result.unwrap_err().to_string().contains("beyond end of file"));
    }

    #[tokio::test]
    async fn read_caps_a_long_file_and_hints_how_to_continue() {
        let sb = fresh_sandbox();
        let body: String = (1..=3000).map(|i| format!("line{i}")).collect::<Vec<_>>().join("\n");
        sb.write("big.txt", &body).unwrap();
        let tool = Read { sandbox: sb };
        let out = tool.call_json(json!({ "path": "big.txt" })).await.unwrap();
        let text = out.as_text();
        assert!(
            text.contains("   1| line1"),
            "starts at line 1: {}",
            text.get(..40.min(text.len())).unwrap_or(&text)
        );
        assert!(!text.contains("2001| line2001"), "capped before line 2001");
        assert!(
            text.contains("Showing lines 1-2000 of 3000") && text.contains("offset=2001"),
            "hint should point past the cap: {}",
            text.get(text.len().saturating_sub(120)..).unwrap_or(&text)
        );
    }

    #[tokio::test]
    async fn read_offset_continues_to_end_without_hint() {
        let sb = fresh_sandbox();
        let body: String = (1..=2100).map(|i| format!("line{i}")).collect::<Vec<_>>().join("\n");
        sb.write("big.txt", &body).unwrap();
        let tool = Read { sandbox: sb };
        // Continue from the cap: the tail (lines 2001..2100) fits, so no more hint.
        let out = tool
            .call_json(json!({ "path": "big.txt", "offset": 2000 }))
            .await
            .unwrap();
        let text = out.as_text();
        assert!(text.contains("2100| line2100"), "shows the last line");
        assert!(!text.contains("Use offset="), "no hint once the file ends: {text}");
    }

    #[tokio::test]
    async fn read_giant_single_line_points_to_sed() {
        let sb = fresh_sandbox();
        sb.write("huge.txt", &"x".repeat(200_000)).unwrap();
        let tool = Read { sandbox: sb };
        let out = tool
            .call_json(json!({ "path": "huge.txt", "line_numbers": false }))
            .await
            .unwrap();
        let text = out.as_text();
        // A single over-cap line can't be shown whole; point at a byte-range fallback.
        assert!(text.contains("Line 1 is") && text.contains("exceeds"), "{text}");
        assert!(text.contains("sed -n '1p'"), "{text}");
        assert!(text.contains("head -c 51200"), "{text}");
    }

    #[tokio::test]
    async fn read_line_numbers_false_returns_raw_text() {
        let sb = fresh_sandbox();
        sb.write("notes.txt", "alpha\nbeta\ngamma").unwrap();
        let tool = Read { sandbox: sb };
        let out = tool
            .call_json(json!({ "path": "notes.txt", "line_numbers": false }))
            .await
            .unwrap();
        assert_eq!(out.as_text(), "alpha\nbeta\ngamma");
    }

    #[tokio::test]
    async fn read_markdown_returns_literal_content_no_warning() {
        // .md now reads as plain text — the old markdown_get_section steering
        // warning is gone, and inline images stay as literal `![alt](path)`
        // for the model to `view` on demand.
        let sb = fresh_sandbox();
        sb.write("docs/guide.md", "# Intro\n![fig](fig.png)").unwrap();
        let tool = Read { sandbox: sb };
        let out = tool.call_json(json!({ "path": "docs/guide.md" })).await.unwrap();
        let text = out.as_text();
        assert!(text.contains("   1| # Intro"), "{text}");
        assert!(text.contains("![fig](fig.png)"), "inline image stays literal: {text}");
        assert!(!text.to_lowercase().contains("warning"), "no steering warning: {text}");
    }

    #[tokio::test]
    async fn read_image_path_points_to_view() {
        let sb = fresh_sandbox();
        let tool = Read { sandbox: sb };
        let out = tool.call_json(json!({ "path": "docs/fig.PNG" })).await.unwrap();
        let text = out.as_text();
        assert!(text.contains("`view`"), "should point at view: {text}");
    }

    #[tokio::test]
    async fn read_missing_file_errors() {
        let tool = Read {
            sandbox: fresh_sandbox(),
        };
        let result = tool.call_json(json!({ "path": "no_such.txt" })).await;
        assert!(result.is_err(), "missing file should error: {result:?}");
    }

    // ─── write ──────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn write_creates_file_and_parents() {
        let sb = fresh_sandbox();
        let tool = Write { sandbox: sb.clone() };
        let out = tool
            .call_json(json!({ "path": "deep/nested/dir/foo.txt", "file_text": "hello" }))
            .await
            .unwrap();
        assert_eq!(out.as_text(), "Successfully wrote 5 bytes");
        assert_eq!(sb.read("deep/nested/dir/foo.txt").unwrap(), "hello");
    }

    #[tokio::test]
    async fn write_reports_utf8_byte_count_not_chars() {
        let sb = fresh_sandbox();
        let tool = Write { sandbox: sb.clone() };
        // "héllo" = 6 UTF-8 bytes (é is 2 bytes) / 5 chars. pi's `.length` would
        // wrongly report 5; we count real bytes.
        let out = tool
            .call_json(json!({ "path": "u.txt", "file_text": "héllo" }))
            .await
            .unwrap();
        assert_eq!(out.as_text(), "Successfully wrote 6 bytes");
    }

    #[tokio::test]
    async fn write_overwrites_existing_file() {
        let sb = fresh_sandbox();
        sb.write("foo.txt", "v1").unwrap();
        let tool = Write { sandbox: sb.clone() };
        tool.call_json(json!({ "path": "foo.txt", "file_text": "v2" }))
            .await
            .unwrap();
        assert_eq!(sb.read("foo.txt").unwrap(), "v2");
    }

    #[tokio::test]
    async fn write_to_ro_overlay_errors() {
        let upstream = TempDir::new().unwrap();
        std::fs::write(upstream.path().join("readme.md"), "from upstream").unwrap();
        let mut sb = Sandbox::new().unwrap();
        sb.add_ro(upstream.path(), "/workspace/docs");
        let sb = Arc::new(sb);
        let tool = Write { sandbox: sb };

        let result = tool
            .call_json(json!({ "path": "docs/readme.md", "file_text": "x" }))
            .await;
        assert!(result.is_err(), "write to ro overlay must error: {result:?}");
    }

    // ─── content-block unwrapping ─────────────────────────────────────────────

    #[tokio::test]
    async fn write_unwraps_content_block_array() {
        let sb = fresh_sandbox();
        let tool = Write { sandbox: sb.clone() };
        // A model double-encoded the file content as Anthropic content blocks.
        let payload = r#"[{"type": "text", "text": "import torch\nx = 1\n"}]"#;
        tool.call_json(json!({ "path": "solution.py", "file_text": payload }))
            .await
            .unwrap();
        assert_eq!(sb.read("solution.py").unwrap(), "import torch\nx = 1\n");
    }

    #[tokio::test]
    async fn write_unwraps_bare_text_blocks_and_concatenates() {
        let sb = fresh_sandbox();
        let tool = Write { sandbox: sb.clone() };
        // No `type` key, multiple blocks → concatenated in order.
        let payload = r#"[{"text": "a = 1\n"}, {"text": "b = 2\n"}]"#;
        tool.call_json(json!({ "path": "f.py", "file_text": payload }))
            .await
            .unwrap();
        assert_eq!(sb.read("f.py").unwrap(), "a = 1\nb = 2\n");
    }

    #[tokio::test]
    async fn write_leaves_plain_code_untouched() {
        let sb = fresh_sandbox();
        let tool = Write { sandbox: sb.clone() };
        let code = "import torch\n\nclass Solution: ...\n";
        tool.call_json(json!({ "path": "s.py", "file_text": code }))
            .await
            .unwrap();
        assert_eq!(sb.read("s.py").unwrap(), code);
    }

    #[tokio::test]
    async fn write_leaves_genuine_json_files_untouched() {
        let sb = fresh_sandbox();
        let tool = Write { sandbox: sb.clone() };
        // Real JSON that is NOT a content-block envelope (no string `text`)
        // must round-trip verbatim, not get silently rewritten.
        let arr = r#"[{"id": 1}, {"id": 2}]"#;
        tool.call_json(json!({ "path": "data.json", "file_text": arr }))
            .await
            .unwrap();
        assert_eq!(sb.read("data.json").unwrap(), arr);

        let obj = r#"{"key": "value"}"#;
        tool.call_json(json!({ "path": "cfg.json", "file_text": obj }))
            .await
            .unwrap();
        assert_eq!(sb.read("cfg.json").unwrap(), obj);
    }
}
