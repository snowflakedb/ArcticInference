//! `edit` — precise multi-edit file replacement (replaces the old `replace`).
//!
//! Port of the load-bearing half of pi's `edit` (`edit.ts` + `edit-diff.ts`),
//! dropping the entire diff-rendering layer (the model never sees a diff). What
//! we keep:
//! - an `edits: [{old_text, new_text}]` array, each matched against the
//!   *original* file (not incrementally), overlap-checked, applied in reverse
//!   offset order so earlier offsets stay valid;
//! - CRLF/BOM detect + normalize-to-LF for matching, then restore on write;
//! - a fuzzy fallback (trailing-ws / smart-quote / dash / space folding) when
//!   an exact match misses;
//! - pi's verbatim error strings.
//!
//! Concurrency: the whole read-modify-write is done under the per-path
//! [`Sandbox::lock_path`] guard, so two edits/writes of the same file can't
//! interleave.
//!
//! Error convention (kernelguy's, not pi's): a *recoverable* miss (text not
//! found, not unique, overlap, no-op, empty `old_text`) returns `Ok(<message>)`
//! so the model can adjust and retry within the turn; only genuine I/O
//! (missing/unwritable file) returns `Err`.
//!
//! NOTE on fuzzy matching: pi additionally applies Unicode NFKC here. We omit
//! NFKC (it needs the `unicode-normalization` crate, outside the locked dep
//! set); exact match is always tried first, and the explicit folding below
//! covers the common real-world cases (smart quotes, dashes, NBSP).

use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::exec::sandbox::Sandbox;
use crate::tool::{Tool, ToolOutput};

// ─── line-ending / BOM handling ──────────────────────────────────────────────

/// Detect the dominant line ending: `\r\n` iff the first `\r\n` precedes the
/// first bare `\n`. Mirrors pi's `detectLineEnding`.
fn detect_line_ending(content: &str) -> &'static str {
    let lf = content.find('\n');
    let crlf = content.find("\r\n");
    match (lf, crlf) {
        (None, _) | (Some(_), None) => "\n",
        (Some(lf), Some(crlf)) => {
            if crlf < lf {
                "\r\n"
            } else {
                "\n"
            }
        }
    }
}

fn normalize_to_lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

fn restore_line_endings(text: &str, ending: &str) -> String {
    if ending == "\r\n" {
        text.replace('\n', "\r\n")
    } else {
        text.to_string()
    }
}

/// Strip a leading UTF-8 BOM, returning `(bom, rest)`.
fn strip_bom(content: &str) -> (&str, &str) {
    content
        .strip_prefix('\u{FEFF}')
        .map_or(("", content), |rest| ("\u{FEFF}", rest))
}

// ─── fuzzy normalization ─────────────────────────────────────────────────────

/// Fold a string for fuzzy matching: strip per-line trailing whitespace, and
/// map smart quotes / Unicode dashes / special spaces to their ASCII forms.
/// (pi also NFKC-normalizes first — omitted here; see the module note.)
fn normalize_for_fuzzy_match(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for (i, line) in text.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
        }
        for ch in line.trim_end().chars() {
            let mapped = match ch {
                '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
                '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
                '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}' | '\u{2212}' => '-',
                '\u{00A0}' | '\u{2002}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
                other => other,
            };
            out.push(mapped);
        }
    }
    out
}

// ─── line-span machinery (for fuzzy write-back) ──────────────────────────────

/// Split into lines *keeping* each trailing `\n`. Mirrors pi's
/// `splitLinesWithEndings` (regex `[^\n]*\n|[^\n]+`).
fn split_lines_with_endings(content: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let bytes = content.as_bytes();
    let mut start = 0;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'\n' {
            // A `\n` can never be a UTF-8 continuation byte, so `start..=i` is
            // always a char-boundary range and `get` always yields `Some`. The
            // `""` fallback keeps the line *count* intact for the callers that
            // pair this with `line_spans`.
            out.push(content.get(start..=i).unwrap_or(""));
            start = i.saturating_add(1);
        }
    }
    if start < content.len() {
        out.push(content.get(start..).unwrap_or(""));
    }
    out
}

#[derive(Clone, Copy)]
struct LineSpan {
    start: usize,
    end: usize,
}

fn line_spans(content: &str) -> Vec<LineSpan> {
    let mut offset = 0;
    split_lines_with_endings(content)
        .into_iter()
        .map(|line| {
            let span = LineSpan {
                start: offset,
                end: offset.saturating_add(line.len()),
            };
            offset = span.end;
            span
        })
        .collect()
}

#[derive(Clone)]
struct Replacement {
    match_index: usize,
    match_len: usize,
    new_text: String,
}

/// Byte-line range `[start, end)` a replacement touches. Recoverable error on
/// an out-of-range replacement.
fn replacement_line_range(lines: &[LineSpan], r: &Replacement) -> Result<(usize, usize), String> {
    let rep_start = r.match_index;
    let rep_end = r.match_index.saturating_add(r.match_len);

    let start_line = lines
        .iter()
        .position(|l| rep_start >= l.start && rep_start < l.end)
        .ok_or_else(|| "Replacement range is outside the base content.".to_string())?;

    let mut end_line = start_line;
    while lines.get(end_line).is_some_and(|l| l.end < rep_end) {
        end_line = end_line.saturating_add(1);
    }
    if end_line >= lines.len() {
        return Err("Replacement range is outside the base content.".to_string());
    }
    Ok((start_line, end_line.saturating_add(1)))
}

/// Apply replacements to `content` in reverse offset order (offsets stay valid).
/// `offset` is subtracted from each `match_index` (used when slicing a subrange).
fn apply_replacements(content: &str, replacements: &[Replacement], offset: usize) -> String {
    let mut result = content.to_string();
    for r in replacements.iter().rev() {
        // `match_index` came from `str::find` and `match_len` from the needle's
        // own `len()`, so `at` and `at + match_len` are both char boundaries and
        // the `get`s always yield `Some`.
        let at = r.match_index.saturating_sub(offset);
        let after = at.saturating_add(r.match_len);
        result = format!(
            "{}{}{}",
            result.get(..at).unwrap_or(""),
            r.new_text,
            result.get(after..).unwrap_or("")
        );
    }
    result
}

/// Overlay line-level replacements matched against `base` (a fuzzy-normalized
/// view) onto `original`, keeping unchanged lines' original bytes. Mirrors pi's
/// `applyReplacementsPreservingUnchangedLines`.
fn apply_preserving_unchanged_lines(
    original: &str,
    base: &str,
    replacements: &[Replacement],
) -> Result<String, String> {
    let original_lines = split_lines_with_endings(original);
    let base_lines = line_spans(base);
    if original_lines.len() != base_lines.len() {
        return Err("Cannot preserve unchanged lines because the base content has a different line count.".to_string());
    }

    // Group replacements by the line blocks they touch, merging overlaps.
    let mut sorted: Vec<Replacement> = replacements.to_vec();
    sorted.sort_by_key(|r| r.match_index);
    let mut groups: Vec<(usize, usize, Vec<Replacement>)> = Vec::new();
    for r in sorted {
        let (rs, re) = replacement_line_range(&base_lines, &r)?;
        if let Some(cur) = groups.last_mut()
            && rs < cur.1
        {
            cur.1 = cur.1.max(re);
            cur.2.push(r);
            continue;
        }
        groups.push((rs, re, vec![r]));
    }

    let mut result = String::new();
    let mut orig_idx = 0;
    for (gstart, gend, reps) in groups {
        // Erroring, not skipping: silently copying nothing would return a corrupt
        // edit as `Ok`, matching how the `base_lines` lookups below fail.
        let unchanged = original_lines
            .get(orig_idx..gstart)
            .ok_or_else(|| "Replacement range is outside the base content.".to_string())?;
        for line in unchanged {
            result.push_str(line);
        }
        // `replacement_line_range` already bounds-checked both ends against
        // `base_lines`, and `gend` is an `end_line + 1` so it is never 0. The
        // byte range spans whole `\n`-delimited lines, hence char boundaries.
        let (Some(first), Some(last)) = (base_lines.get(gstart), base_lines.get(gend.saturating_sub(1))) else {
            return Err("Replacement range is outside the base content.".to_string());
        };
        let group = base
            .get(first.start..last.end)
            .ok_or_else(|| "Replacement range is outside the base content.".to_string())?;
        result.push_str(&apply_replacements(group, &reps, first.start));
        orig_idx = gend;
    }
    let tail = original_lines
        .get(orig_idx..)
        .ok_or_else(|| "Replacement range is outside the base content.".to_string())?;
    for line in tail {
        result.push_str(line);
    }
    Ok(result)
}

// ─── matching ────────────────────────────────────────────────────────────────

struct FuzzyMatch {
    index: usize,
    match_len: usize,
    used_fuzzy: bool,
}

/// Find `old_text` in `content`: exact first, then fuzzy (on normalized forms).
/// Returns `None` if neither matches. On a fuzzy hit, the returned index/len are
/// in *normalized* space (the caller passes the normalized content as `content`).
fn fuzzy_find(content: &str, old_text: &str) -> Option<FuzzyMatch> {
    if let Some(idx) = content.find(old_text) {
        return Some(FuzzyMatch {
            index: idx,
            match_len: old_text.len(),
            used_fuzzy: false,
        });
    }
    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old = normalize_for_fuzzy_match(old_text);
    fuzzy_content.find(&fuzzy_old).map(|idx| FuzzyMatch {
        index: idx,
        match_len: fuzzy_old.len(),
        used_fuzzy: true,
    })
}

fn count_occurrences(content: &str, old_text: &str) -> usize {
    let fc = normalize_for_fuzzy_match(content);
    let fo = normalize_for_fuzzy_match(old_text);
    if fo.is_empty() {
        return 0;
    }
    fc.matches(&fo).count()
}

// pi's verbatim error strings.
fn not_found_error(path: &str, i: usize, total: usize) -> String {
    if total == 1 {
        format!(
            "Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
        )
    } else {
        format!(
            "Could not find edits[{i}] in {path}. The oldText must match exactly including all whitespace and newlines."
        )
    }
}

fn duplicate_error(path: &str, i: usize, total: usize, occ: usize) -> String {
    if total == 1 {
        format!(
            "Found {occ} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
        )
    } else {
        format!(
            "Found {occ} occurrences of edits[{i}] in {path}. Each oldText must be unique. Please provide more context to make it unique."
        )
    }
}

fn empty_old_text_error(path: &str, i: usize, total: usize) -> String {
    if total == 1 {
        format!("oldText must not be empty in {path}.")
    } else {
        format!("edits[{i}].oldText must not be empty in {path}.")
    }
}

fn no_change_error(path: &str, total: usize) -> String {
    if total == 1 {
        format!(
            "No changes made to {path}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
        )
    } else {
        format!("No changes made to {path}. The replacements produced identical content.")
    }
}

/// Match every edit against `normalized` (the LF-normalized original), check for
/// duplicates/overlap, and apply. Returns the new (LF) content, or a
/// *recoverable* error message. Mirrors pi's `applyEditsToNormalizedContent`.
fn apply_edits(normalized: &str, edits: &[EditOp], path: &str) -> Result<String, String> {
    let total = edits.len();
    let norm_edits: Vec<(String, String)> = edits
        .iter()
        .map(|e| (normalize_to_lf(&e.old_text), normalize_to_lf(&e.new_text)))
        .collect();

    for (i, (old, _)) in norm_edits.iter().enumerate() {
        if old.is_empty() {
            return Err(empty_old_text_error(path, i, total));
        }
    }

    // If any edit needs fuzzy matching, do all matching + replacement in
    // fuzzy-normalized space, then overlay onto the original to preserve
    // unchanged lines' bytes.
    let used_fuzzy = norm_edits
        .iter()
        .any(|(old, _)| fuzzy_find(normalized, old).is_some_and(|m| m.used_fuzzy));
    let base = if used_fuzzy {
        normalize_for_fuzzy_match(normalized)
    } else {
        normalized.to_string()
    };

    let mut matched: Vec<(usize, Replacement)> = Vec::with_capacity(total);
    for (i, (old, new)) in norm_edits.iter().enumerate() {
        let m = fuzzy_find(&base, old).ok_or_else(|| not_found_error(path, i, total))?;
        let occ = count_occurrences(&base, old);
        if occ > 1 {
            return Err(duplicate_error(path, i, total, occ));
        }
        matched.push((
            i,
            Replacement {
                match_index: m.index,
                match_len: m.match_len,
                new_text: new.clone(),
            },
        ));
    }

    // Overlap check on the matched byte ranges.
    matched.sort_by_key(|(_, r)| r.match_index);
    for w in matched.windows(2) {
        if let [(pi, prev), (ci, cur)] = w
            && prev.match_index.saturating_add(prev.match_len) > cur.match_index
        {
            return Err(format!(
                "edits[{pi}] and edits[{ci}] overlap in {path}. Merge them into one edit or target disjoint regions."
            ));
        }
    }

    let replacements: Vec<Replacement> = matched.into_iter().map(|(_, r)| r).collect();
    let new_content = if used_fuzzy {
        apply_preserving_unchanged_lines(normalized, &base, &replacements)?
    } else {
        apply_replacements(&base, &replacements, 0)
    };

    if new_content == normalized {
        return Err(no_change_error(path, total));
    }
    Ok(new_content)
}

// ─── tool ─────────────────────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema)]
pub struct EditOp {
    /// Exact text for one targeted replacement. Must be unique in the original
    /// file and must not overlap any other edit's `old_text` in the same call.
    pub old_text: String,
    /// Replacement text for this edit.
    pub new_text: String,
}

#[derive(JsonSchema)]
pub struct EditArgs {
    /// Workspace-relative path of the file to edit (e.g. `solution/solution.py`).
    pub path: String,
    /// One or more targeted replacements. Each `old_text` is matched against the
    /// ORIGINAL file (not after earlier edits), so edits must not overlap or
    /// nest — merge nearby changes into one edit. Applied together, atomically.
    pub edits: Vec<EditOp>,
}

// Lenient deserialization (mirrors pi's `prepareEditArguments`): tolerate the
// PARSEABLE malformations the model sometimes emits, while still advertising the
// strict `{path, edits: [{old_text, new_text}]}` schema (JsonSchema is derived
// from the struct above, untouched by this impl). We accept:
//   * `edits` as a JSON array (normal), OR as a JSON-encoded *string* the model
//     double-encoded, OR absent;
//   * a legacy top-level `{old_text, new_text}` single-edit shorthand, folded
//     into `edits`.
// A structurally-invalid tool_use (e.g. `">` for `":`) never reaches here — the
// provider layer turns that into a teaching error tool_result before dispatch.
impl<'de> Deserialize<'de> for EditArgs {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            path: String,
            #[serde(default)]
            edits: Option<RawEdits>,
            #[serde(default)]
            old_text: Option<String>,
            #[serde(default)]
            new_text: Option<String>,
        }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum RawEdits {
            List(Vec<EditOp>),
            Encoded(String),
        }

        let raw = Raw::deserialize(deserializer)?;
        let mut edits = match raw.edits {
            Some(RawEdits::List(list)) => list,
            Some(RawEdits::Encoded(s)) => {
                let s = s.trim();
                if s.is_empty() {
                    Vec::new()
                } else {
                    // A JSON-encoded array, or a single JSON-encoded edit object.
                    serde_json::from_str::<Vec<EditOp>>(s)
                        .or_else(|_| serde_json::from_str::<EditOp>(s).map(|e| vec![e]))
                        .map_err(|e| {
                            serde::de::Error::custom(format!(
                                "`edits` was a string but not valid JSON for an edit array: {e}"
                            ))
                        })?
                }
            }
            None => Vec::new(),
        };
        if let (Some(old_text), Some(new_text)) = (raw.old_text, raw.new_text) {
            edits.push(EditOp { old_text, new_text });
        }
        Ok(Self { path: raw.path, edits })
    }
}

pub struct Edit {
    pub sandbox: Arc<Sandbox>,
}

impl Tool for Edit {
    type Args = EditArgs;
    const NAME: &'static str = "edit";
    const DESCRIPTION: &'static str = "Edit a file via exact text replacement. Provide `edits`: one or more \
         {old_text, new_text} pairs. Each `old_text` must match EXACTLY (whitespace/newlines included) and be \
         UNIQUE in the file; keep it as small as possible while still unique. Make several disjoint changes to \
         one file in a SINGLE call with multiple edits — each is matched against the original, so they must not \
         overlap. A miss (not found / not unique / overlap) returns an error message and leaves the file \
         unchanged, so you can adjust and retry. For a full rewrite, use `write`.";

    async fn call(&self, args: EditArgs) -> ToolOutput {
        if args.edits.is_empty() {
            return Ok("Edit tool input is invalid. edits must contain at least one replacement.".into());
        }
        // Hold the per-path lock across the whole read-modify-write.
        let _guard = self.sandbox.lock_path(&args.path).await.map_err(|e| e.to_string())?;

        // I/O failure (missing/unreadable file) → Err.
        let raw = self
            .sandbox
            .read(&args.path)
            .map_err(|e| format!("Could not edit file: {}. {e}.", args.path))?;

        let (bom, content) = strip_bom(&raw);
        let ending = detect_line_ending(content);
        let normalized = normalize_to_lf(content);

        // Matching/overlap/no-op failures are recoverable → Ok(text).
        let new_content = match apply_edits(&normalized, &args.edits, &args.path) {
            Ok(nc) => nc,
            Err(msg) => return Ok(msg.into()),
        };

        let final_content = format!("{bom}{}", restore_line_endings(&new_content, ending));
        self.sandbox
            .write(&args.path, &final_content)
            .map_err(|e| e.to_string())?;

        Ok(format!("Successfully replaced {} block(s) in {}.", args.edits.len(), args.path).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;
    use serde_json::json;

    fn sb() -> Arc<Sandbox> {
        Arc::new(Sandbox::new().expect("sandbox"))
    }

    fn edit_tool(s: Arc<Sandbox>) -> impl Tool {
        Edit { sandbox: s }
    }

    #[tokio::test]
    async fn single_exact_edit() {
        let s = sb();
        s.write("f.py", "fn old_name() {}").unwrap();
        let out = edit_tool(s.clone())
            .call_json(json!({ "path": "f.py", "edits": [{ "old_text": "old_name", "new_text": "new_name" }] }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("Successfully replaced 1 block(s)"), "{out}");
        assert_eq!(s.read("f.py").unwrap(), "fn new_name() {}");
    }

    // Lenient-parse: the model double-encoded `edits` as a JSON string instead of
    // a real array. We JSON.parse it rather than erroring on the arg shape.
    #[tokio::test]
    async fn edits_accepts_json_encoded_string() {
        let s = sb();
        s.write("f.py", "fn old_name() {}").unwrap();
        let out = edit_tool(s.clone())
            .call_json(json!({
                "path": "f.py",
                "edits": "[{\"old_text\": \"old_name\", \"new_text\": \"new_name\"}]"
            }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("Successfully replaced 1 block(s)"), "{out}");
        assert_eq!(s.read("f.py").unwrap(), "fn new_name() {}");
    }

    // Lenient-parse: legacy single-edit shorthand with top-level {old_text,new_text}
    // and no `edits` array is folded into a one-element edit list.
    #[tokio::test]
    async fn edits_accepts_legacy_top_level_fields() {
        let s = sb();
        s.write("f.py", "fn old_name() {}").unwrap();
        let out = edit_tool(s.clone())
            .call_json(json!({ "path": "f.py", "old_text": "old_name", "new_text": "new_name" }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("Successfully replaced 1 block(s)"), "{out}");
        assert_eq!(s.read("f.py").unwrap(), "fn new_name() {}");
    }

    #[tokio::test]
    async fn multiple_disjoint_edits_matched_against_original() {
        let s = sb();
        s.write("f.py", "a = 1\nb = 2\nc = 3\n").unwrap();
        let out = edit_tool(s.clone())
            .call_json(json!({
                "path": "f.py",
                "edits": [
                    { "old_text": "a = 1", "new_text": "a = 10" },
                    { "old_text": "c = 3", "new_text": "c = 30" }
                ]
            }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("Successfully replaced 2 block(s)"), "{out}");
        assert_eq!(s.read("f.py").unwrap(), "a = 10\nb = 2\nc = 30\n");
    }

    #[tokio::test]
    async fn not_found_is_recoverable_ok_and_leaves_file() {
        let s = sb();
        s.write("f.py", "hello world").unwrap();
        let out = edit_tool(s.clone())
            .call_json(json!({ "path": "f.py", "edits": [{ "old_text": "missing", "new_text": "x" }] }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("Could not find the exact text"), "{out}");
        assert_eq!(s.read("f.py").unwrap(), "hello world", "file untouched on miss");
    }

    #[tokio::test]
    async fn non_unique_match_is_recoverable_and_leaves_file() {
        let s = sb();
        s.write("f.py", "x\nx\nx\n").unwrap();
        let out = edit_tool(s.clone())
            .call_json(json!({ "path": "f.py", "edits": [{ "old_text": "x", "new_text": "y" }] }))
            .await
            .unwrap()
            .as_text();
        assert!(
            out.contains("Found 3 occurrences") && out.contains("must be unique"),
            "{out}"
        );
        assert_eq!(s.read("f.py").unwrap(), "x\nx\nx\n");
    }

    #[tokio::test]
    async fn overlapping_edits_rejected() {
        let s = sb();
        s.write("f.py", "abcdef").unwrap();
        let out = edit_tool(s.clone())
            .call_json(json!({
                "path": "f.py",
                "edits": [
                    { "old_text": "abcd", "new_text": "X" },
                    { "old_text": "cdef", "new_text": "Y" }
                ]
            }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("overlap"), "{out}");
        assert_eq!(s.read("f.py").unwrap(), "abcdef");
    }

    #[tokio::test]
    async fn empty_old_text_rejected() {
        let s = sb();
        s.write("f.py", "content").unwrap();
        let out = edit_tool(s.clone())
            .call_json(json!({ "path": "f.py", "edits": [{ "old_text": "", "new_text": "x" }] }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("must not be empty"), "{out}");
    }

    #[tokio::test]
    async fn no_op_edit_rejected() {
        let s = sb();
        s.write("f.py", "same").unwrap();
        let out = edit_tool(s.clone())
            .call_json(json!({ "path": "f.py", "edits": [{ "old_text": "same", "new_text": "same" }] }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("No changes made"), "{out}");
    }

    #[tokio::test]
    async fn crlf_file_edited_and_endings_restored() {
        let s = sb();
        s.write("f.txt", "one\r\ntwo\r\nthree\r\n").unwrap();
        let out = edit_tool(s.clone())
            .call_json(json!({ "path": "f.txt", "edits": [{ "old_text": "two", "new_text": "TWO" }] }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("Successfully replaced"), "{out}");
        // CRLF endings must be preserved on write.
        assert_eq!(s.read("f.txt").unwrap(), "one\r\nTWO\r\nthree\r\n");
    }

    #[tokio::test]
    async fn fuzzy_matches_smart_quotes() {
        let s = sb();
        // File has a smart apostrophe; the model supplies an ASCII one.
        s.write("f.py", "name = \u{2018}world\u{2019}\n").unwrap();
        let out = edit_tool(s.clone())
            .call_json(
                json!({ "path": "f.py", "edits": [{ "old_text": "name = 'world'", "new_text": "name = 'earth'" }] }),
            )
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("Successfully replaced"), "fuzzy should match: {out}");
        assert!(s.read("f.py").unwrap().contains("earth"), "edit applied");
    }

    #[tokio::test]
    async fn missing_file_is_io_error() {
        let s = sb();
        let result = edit_tool(s)
            .call_json(json!({ "path": "nope.py", "edits": [{ "old_text": "a", "new_text": "b" }] }))
            .await;
        assert!(result.is_err(), "I/O failure must be Err: {result:?}");
    }

    #[tokio::test]
    async fn edit_to_ro_overlay_errors() {
        let upstream = tempfile::TempDir::new().unwrap();
        std::fs::write(upstream.path().join("readme.md"), "hello").unwrap();
        let mut s = Sandbox::new().unwrap();
        s.add_ro(upstream.path(), "/workspace/docs");
        let result = edit_tool(Arc::new(s))
            .call_json(json!({ "path": "docs/readme.md", "edits": [{ "old_text": "hello", "new_text": "x" }] }))
            .await;
        assert!(result.is_err(), "edit on ro overlay must error: {result:?}");
    }

    // ─── pure-function tests ──────────────────────────────────────────────

    #[test]
    fn detect_line_ending_picks_crlf_only_when_dominant() {
        assert_eq!(detect_line_ending("a\r\nb"), "\r\n");
        assert_eq!(detect_line_ending("a\nb"), "\n");
        assert_eq!(detect_line_ending("no newline"), "\n");
    }

    #[test]
    fn split_lines_keeps_endings() {
        assert_eq!(split_lines_with_endings("a\nb"), vec!["a\n", "b"]);
        assert_eq!(split_lines_with_endings("a\n\nb"), vec!["a\n", "\n", "b"]);
        assert_eq!(split_lines_with_endings(""), Vec::<&str>::new());
        assert_eq!(split_lines_with_endings("a\n"), vec!["a\n"]);
    }

    #[test]
    fn fuzzy_normalize_folds_quotes_dashes_and_trailing_ws() {
        assert_eq!(normalize_for_fuzzy_match("a\u{2019}b   \n\u{2014}"), "a'b\n-");
        assert_eq!(normalize_for_fuzzy_match("x\u{00A0}y"), "x y");
    }

    // ─── adversarial: multibyte / CRLF / fuzzy write-back (mutation safety) ───

    #[tokio::test]
    async fn multibyte_exact_edit_no_panic() {
        let s = sb();
        s.write("f.py", "café = 1\nnaïve = 2\n").unwrap();
        let out = edit_tool(s.clone())
            .call_json(json!({ "path": "f.py", "edits": [{ "old_text": "café = 1", "new_text": "café = 100" }] }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("Successfully replaced"), "{out}");
        assert_eq!(s.read("f.py").unwrap(), "café = 100\nnaïve = 2\n");
    }

    #[tokio::test]
    async fn new_text_multibyte_no_panic() {
        let s = sb();
        s.write("f.py", "x = 1\n").unwrap();
        edit_tool(s.clone())
            .call_json(json!({ "path": "f.py", "edits": [{ "old_text": "= 1", "new_text": "= 'πλ'" }] }))
            .await
            .unwrap();
        assert_eq!(s.read("f.py").unwrap(), "x = 'πλ'\n");
    }

    #[tokio::test]
    async fn two_disjoint_edits_same_line() {
        let s = sb();
        s.write("f.txt", "a b c\n").unwrap();
        edit_tool(s.clone())
            .call_json(json!({
                "path": "f.txt",
                "edits": [
                    { "old_text": "a b", "new_text": "A B" },
                    { "old_text": "c", "new_text": "C" }
                ]
            }))
            .await
            .unwrap();
        assert_eq!(s.read("f.txt").unwrap(), "A B C\n");
    }

    #[tokio::test]
    async fn fuzzy_edit_preserves_unchanged_lines_original_bytes() {
        let s = sb();
        // line 0 has smart quotes + trailing spaces that fuzzy-normalization WOULD
        // strip — it must survive byte-for-byte because it's unchanged. line 1 is
        // edited via a FUZZY match (file has smart quotes; old_text is ASCII).
        let original = "header \u{2019}kept\u{2019}   \nvalue = \u{2019}hi\u{2019}\nfooter\n";
        s.write("f.py", original).unwrap();
        let out = edit_tool(s.clone())
            .call_json(json!({
                "path": "f.py",
                "edits": [{ "old_text": "value = 'hi'", "new_text": "value = 'bye'" }]
            }))
            .await
            .unwrap()
            .as_text();
        assert!(out.contains("Successfully replaced"), "fuzzy match should apply: {out}");
        let result = s.read("f.py").unwrap();
        // Unchanged line keeps its ORIGINAL smart quotes + trailing whitespace.
        assert!(
            result.contains("header \u{2019}kept\u{2019}   \n"),
            "unchanged line corrupted: {result:?}"
        );
        // Edited line took the new_text.
        assert!(result.contains("value = 'bye'"), "edit not applied: {result:?}");
        assert!(result.ends_with("footer\n"), "trailing unchanged line lost: {result:?}");
    }

    #[tokio::test]
    async fn crlf_multibyte_preserved() {
        let s = sb();
        s.write("f.txt", "café\r\ntarget\r\nπλ\r\n").unwrap();
        edit_tool(s.clone())
            .call_json(json!({ "path": "f.txt", "edits": [{ "old_text": "target", "new_text": "TARGET" }] }))
            .await
            .unwrap();
        // CRLF restored everywhere; multibyte lines intact.
        assert_eq!(s.read("f.txt").unwrap(), "café\r\nTARGET\r\nπλ\r\n");
    }
}
