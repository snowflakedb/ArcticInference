//! Output truncation helpers for tool results (functional core).
//!
//! Ported from pi's `truncate.ts` + the spill half of `output-accumulator.ts`,
//! keeping the semantics byte-for-byte where it matters. The dual cap is two
//! independent limits — whichever is hit first wins:
//!
//! - line limit ([`MAX_OUTPUT_LINES`] = 2000)
//! - byte limit ([`MAX_OUTPUT_BYTES`] = 50 KiB), counted as real UTF-8 bytes
//!
//! Two shapes: [`truncate_head`] keeps the *beginning* (file reads / grep /
//! find / ls) and never returns a partial line; [`truncate_tail`] keeps the
//! *end* (shell output — errors and final results land last) and may return a
//! partial first line only when a single line is itself over the byte cap.
//! [`truncate_line`] caps one match line for grep.
//!
//! Everything here is pure values-in/values-out except the single isolated
//! side effect, [`spill_full_output`], which writes the un-truncated output to
//! a temp file so non-reproducible tools (bash / `gpu_job` / evaluate) can cite a
//! path instead of discarding the tail. [`clamp_with_spill`] composes the two.

use std::fmt::Write as _;
use std::io::Read;
use std::path::PathBuf;

use crate::domain::convert::usize_to_f64_lossy;

/// Byte cap for a single tool result, 50 KiB = `51_200` bytes.
///
/// Counted as real
/// UTF-8 bytes to match pi's `Buffer.byteLength(_, "utf-8")` and produce a
/// correct "50.0KB" label. Replaces the older `MAX_OUTPUT_CHARS = 50_000`.
pub const MAX_OUTPUT_BYTES: usize = 50 * 1024;

/// Line cap for a single tool result. Matches pi's `DEFAULT_MAX_LINES`.
pub const MAX_OUTPUT_LINES: usize = 2000;

/// Per-line char cap for grep match lines. Matches pi's `GREP_MAX_LINE_LENGTH`.
pub const GREP_MAX_LINE_LEN: usize = 500;

/// Which limit tripped the truncation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncatedBy {
    Lines,
    Bytes,
}

/// Result of a head/tail truncation. Carries enough for the caller to build a
/// continuation/spill notice without re-scanning the source.
#[derive(Debug, Clone)]
pub struct Truncation {
    /// The (possibly truncated) content.
    pub content: String,
    /// Whether any truncation occurred.
    pub truncated: bool,
    /// Which limit was hit, or `None` when not truncated.
    pub truncated_by: Option<TruncatedBy>,
    /// Total number of lines in the original content.
    pub total_lines: usize,
    /// Number of complete lines in the output (the partial line, if any, counts as one).
    pub output_lines: usize,
    /// Tail edge case: the single kept line was itself over the byte cap and
    /// was cut mid-line (only ever set by [`truncate_tail`]).
    pub last_line_partial: bool,
    /// Head edge case: the very first line alone exceeds the byte cap, so
    /// `content` is empty and the caller should point at a `sed`/`head` fallback.
    pub first_line_exceeds: bool,
}

/// Human-readable size, matching pi's `formatSize` (`123B` / `50.0KB` / `1.5MB`).
#[must_use]
pub fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", usize_to_f64_lossy(bytes) / 1024.0)
    } else {
        format!("{:.1}MB", usize_to_f64_lossy(bytes) / (1024.0 * 1024.0))
    }
}

/// Split into logical lines for counting, dropping a single trailing empty
/// line produced by a final `\n`. Mirrors pi's `splitLinesForCounting`.
fn split_lines_for_counting(content: &str) -> Vec<&str> {
    if content.is_empty() {
        return Vec::new();
    }
    let mut lines: Vec<&str> = content.split('\n').collect();
    if content.ends_with('\n') {
        lines.pop();
    }
    lines
}

/// Truncate from the head (keep the first lines/bytes).
///
/// Never returns a partial
/// line. If the first line alone exceeds `max_bytes`, returns empty content with
/// `first_line_exceeds = true`. Mirrors pi's `truncateHead`.
#[must_use]
pub fn truncate_head(content: &str, max_lines: usize, max_bytes: usize) -> Truncation {
    let total_bytes = content.len();
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return Truncation {
            content: content.to_string(),
            truncated: false,
            truncated_by: None,
            total_lines,
            output_lines: total_lines,
            last_line_partial: false,
            first_line_exceeds: false,
        };
    }

    // First line alone over the byte cap: bail with a flag so the caller emits
    // a `sed`/`head` pointer rather than an empty block.
    let first_line_bytes = lines.first().map_or(0, |l| l.len());
    if first_line_bytes > max_bytes {
        return Truncation {
            content: String::new(),
            truncated: true,
            truncated_by: Some(TruncatedBy::Bytes),
            total_lines,
            output_lines: 0,
            last_line_partial: false,
            first_line_exceeds: true,
        };
    }

    // Collect whole lines until either cap trips.
    let mut kept: Vec<&str> = Vec::new();
    let mut bytes_count = 0usize;
    let mut truncated_by = TruncatedBy::Lines;

    for (i, &line) in lines.iter().enumerate().take(max_lines) {
        let line_bytes = line.len().saturating_add(usize::from(i > 0)); // +1 for the joining newline
        if bytes_count.saturating_add(line_bytes) > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            break;
        }
        kept.push(line);
        bytes_count = bytes_count.saturating_add(line_bytes);
    }

    if kept.len() >= max_lines && bytes_count <= max_bytes {
        truncated_by = TruncatedBy::Lines;
    }

    Truncation {
        content: kept.join("\n"),
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        output_lines: kept.len(),
        last_line_partial: false,
        first_line_exceeds: false,
    }
}

/// Truncate from the tail (keep the last lines/bytes).
///
/// May return a partial
/// first line only when the final line of the original is itself over the byte
/// cap (kept as a mid-line slice). Mirrors pi's `truncateTail`.
#[must_use]
pub fn truncate_tail(content: &str, max_lines: usize, max_bytes: usize) -> Truncation {
    let total_bytes = content.len();
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return Truncation {
            content: content.to_string(),
            truncated: false,
            truncated_by: None,
            total_lines,
            output_lines: total_lines,
            last_line_partial: false,
            first_line_exceeds: false,
        };
    }

    let mut kept: Vec<&str> = Vec::new(); // built front-to-back via reverse walk
    let mut owned_partial: Option<String> = None;
    let mut bytes_count = 0usize;
    let mut truncated_by = TruncatedBy::Lines;
    let mut last_line_partial = false;

    for &line in lines.iter().rev() {
        if kept.len() >= max_lines {
            break;
        }
        let line_bytes = line.len().saturating_add(usize::from(!kept.is_empty())); // +1 for the joining newline
        if bytes_count.saturating_add(line_bytes) > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            // Nothing kept yet and this line alone is over cap: keep its end.
            if kept.is_empty() {
                let partial = truncate_string_to_bytes_from_end(line, max_bytes);
                bytes_count = partial.len();
                owned_partial = Some(partial);
                last_line_partial = true;
            }
            break;
        }
        kept.push(line);
        bytes_count = bytes_count.saturating_add(line_bytes);
    }

    let content_out = owned_partial.unwrap_or_else(|| {
        // `kept` was pushed in reverse; restore source order.
        kept.reverse();
        kept.join("\n")
    });
    let output_lines = if last_line_partial { 1 } else { kept.len() };

    if output_lines >= max_lines && bytes_count <= max_bytes {
        truncated_by = TruncatedBy::Lines;
    }

    Truncation {
        content: content_out,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        output_lines,
        last_line_partial,
        first_line_exceeds: false,
    }
}

/// Keep the last `max_bytes` bytes of `s`, snapping forward to a UTF-8 char
/// boundary so the result is always valid. Mirrors pi's
/// `truncateStringToBytesFromEnd`.
fn truncate_string_to_bytes_from_end(s: &str, max_bytes: usize) -> String {
    let bytes = s.as_bytes();
    if bytes.len() <= max_bytes {
        return s.to_string();
    }
    let mut start = bytes.len().saturating_sub(max_bytes);
    // Advance past UTF-8 continuation bytes (0b10xxxxxx) to a char boundary.
    while bytes.get(start).is_some_and(|&b| (b & 0xc0) == 0x80) {
        start = start.saturating_add(1);
    }
    String::from_utf8_lossy(bytes.get(start..).unwrap_or(&[])).into_owned()
}

/// Cap a single line to `max_chars` characters, appending `... [truncated]`
/// when cut.
///
/// Char-based (not byte-based) to match pi + kg markdown. Returns the
/// line and whether it was truncated. Mirrors pi's `truncateLine`.
#[must_use]
pub fn truncate_line(line: &str, max_chars: usize) -> (String, bool) {
    if line.chars().count() <= max_chars {
        return (line.to_string(), false);
    }
    let head: String = line.chars().take(max_chars).collect();
    (format!("{head}... [truncated]"), true)
}

/// 16 hex chars of uniqueness for a spill filename (pi's `randomBytes(8).hex`).
/// Reads 8 bytes from `/dev/urandom`; falls back to a time+pid mix if that is
/// unavailable. Uniqueness, not cryptographic strength, is what matters for a
/// temp filename.
fn random_hex_16() -> String {
    let mut buf = [0u8; 8];
    let ok = std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut buf))
        .is_ok();
    if !ok {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let pid = u128::from(std::process::id());
        let mix = nanos ^ (pid << 96);
        buf.copy_from_slice(&mix.to_le_bytes()[..8]);
    }
    let mut s = String::with_capacity(16);
    for b in buf {
        // Writing to a `String` cannot fail; the `Result` is discarded.
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// THE side effect: write the full, un-truncated output to a temp file so a
/// non-reproducible tool can cite the path.
///
/// Uses `std::env::temp_dir()`, a
/// `<prefix>-<16 hex>.log` name, and does NOT auto-delete — the model may want
/// to `read`/`grep` it later. Mirrors `output-accumulator.ts`'s temp spill.
///
/// # Errors
///
/// Propagates the `std::fs::write` failure if the temp file cannot be created or
/// written (unwritable/missing `TMPDIR`, full filesystem, permissions).
pub fn spill_full_output(prefix: &str, raw: &str) -> std::io::Result<PathBuf> {
    let path = std::env::temp_dir().join(format!("{prefix}-{}.log", random_hex_16()));
    std::fs::write(&path, raw)?;
    Ok(path)
}

/// Composer for non-reproducible tools (`bash`/`gpu_job`/`evaluate`): tail-clamp
/// the output, and if anything was dropped, spill the full output and append a
/// suffix notice citing the path.
///
/// Returns the ready-to-send text. `Err` only on
/// a spill I/O failure. Notice variants mirror pi's `bash.ts formatOutput`.
///
/// # Errors
///
/// Returns `Err` only when the output was truncated *and* writing the spill file
/// failed (see [`spill_full_output`]); the message wraps that I/O error. Output
/// that fits under the caps is always `Ok`.
pub fn clamp_with_spill(raw: &str, prefix: &str) -> Result<String, String> {
    let t = truncate_tail(raw, MAX_OUTPUT_LINES, MAX_OUTPUT_BYTES);
    if !t.truncated {
        return Ok(raw.to_string());
    }

    let path = spill_full_output(prefix, raw).map_err(|e| format!("failed to spill full output: {e}"))?;
    let path = path.display();

    // `output_lines` complete lines were kept from the end; they are the last
    // `output_lines` of `total_lines`.
    let start_line = t.total_lines.saturating_sub(t.output_lines).saturating_add(1);
    let end_line = t.total_lines;

    let notice = if t.last_line_partial {
        // The kept content is the tail of a single over-cap line.
        let last_line_bytes = split_lines_for_counting(raw).last().map_or(0, |l| l.len());
        format!(
            "[Showing last {} of line {} (line is {}). Full output: {}]",
            format_size(t.content.len()),
            end_line,
            format_size(last_line_bytes),
            path
        )
    } else if t.truncated_by == Some(TruncatedBy::Lines) {
        format!(
            "[Showing lines {start_line}-{end_line} of {}. Full output: {path}]",
            t.total_lines
        )
    } else {
        format!(
            "[Showing lines {start_line}-{end_line} of {} ({} limit). Full output: {path}]",
            t.total_lines,
            format_size(MAX_OUTPUT_BYTES)
        )
    };

    Ok(format!("{}\n\n{}", t.content, notice))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Small caps make the intent obvious without giant fixtures.
    const L: usize = 5; // max lines
    const B: usize = 20; // max bytes

    #[test]
    fn head_no_truncation() {
        let t = truncate_head("a\nb\nc", L, B);
        assert!(!t.truncated);
        assert_eq!(t.truncated_by, None);
        assert_eq!(t.content, "a\nb\nc");
        assert_eq!(t.total_lines, 3);
        assert_eq!(t.output_lines, 3);
        assert!(!t.first_line_exceeds);
    }

    #[test]
    fn head_trailing_newline_not_counted() {
        // Trailing "\n" drops the phantom empty line — 3 lines, not 4.
        let t = truncate_head("a\nb\nc\n", L, B);
        assert!(!t.truncated);
        assert_eq!(t.total_lines, 3);
    }

    #[test]
    fn head_truncates_by_lines() {
        let src = "1\n2\n3\n4\n5\n6\n7\n8"; // 8 lines, well under byte cap
        let t = truncate_head(src, L, B);
        assert!(t.truncated);
        assert_eq!(t.truncated_by, Some(TruncatedBy::Lines));
        assert_eq!(t.output_lines, L);
        assert_eq!(t.total_lines, 8);
        assert_eq!(t.content, "1\n2\n3\n4\n5"); // first 5 lines, whole
    }

    #[test]
    fn head_truncates_by_bytes() {
        // 10 two-char lines = plenty of bytes but only 10 lines. Byte cap (20)
        // hits before the line cap would.
        let src = "aa\nbb\ncc\ndd\nee\nff\ngg\nhh\nii\njj";
        let t = truncate_head(src, 1000, B);
        assert!(t.truncated);
        assert_eq!(t.truncated_by, Some(TruncatedBy::Bytes));
        // Whole lines only; never a partial line on the head path.
        assert!(!t.content.contains('\n') || !t.content.ends_with('\n'));
        assert!(t.content.len() <= B);
        assert!(t.output_lines >= 1);
    }

    #[test]
    fn head_giant_first_line_flags_and_empties() {
        let src = format!("{}\nsecond", "x".repeat(B + 10));
        let t = truncate_head(&src, L, B);
        assert!(t.truncated);
        assert!(t.first_line_exceeds);
        assert_eq!(t.content, "");
        assert_eq!(t.output_lines, 0);
        assert_eq!(t.truncated_by, Some(TruncatedBy::Bytes));
    }

    #[test]
    fn tail_no_truncation() {
        let t = truncate_tail("a\nb\nc", L, B);
        assert!(!t.truncated);
        assert_eq!(t.content, "a\nb\nc");
    }

    #[test]
    fn tail_keeps_the_end() {
        let src = "1\n2\n3\n4\n5\n6\n7\n8"; // 8 lines
        let t = truncate_tail(src, L, B);
        assert!(t.truncated);
        assert_eq!(t.truncated_by, Some(TruncatedBy::Lines));
        assert_eq!(t.output_lines, L);
        assert_eq!(t.total_lines, 8);
        assert_eq!(t.content, "4\n5\n6\n7\n8"); // last 5 lines, in order
    }

    #[test]
    fn tail_truncates_by_bytes_keeps_recent() {
        let src = "aa\nbb\ncc\ndd\nee\nff\ngg\nhh\nii\njj";
        let t = truncate_tail(src, 1000, B);
        assert!(t.truncated);
        assert_eq!(t.truncated_by, Some(TruncatedBy::Bytes));
        assert!(!t.last_line_partial);
        assert!(t.content.len() <= B);
        // Keeps the tail, so the last line survives and the first does not.
        assert!(t.content.ends_with("jj"));
        assert!(!t.content.starts_with("aa"));
    }

    #[test]
    fn tail_giant_last_line_partial() {
        // Single line, no newline, longer than the byte cap: kept as a mid-line
        // slice of the END, flagged partial.
        let src = "z".repeat(B + 15);
        let t = truncate_tail(&src, L, B);
        assert!(t.truncated);
        assert!(t.last_line_partial);
        assert_eq!(t.truncated_by, Some(TruncatedBy::Bytes));
        assert_eq!(t.output_lines, 1);
        assert_eq!(t.content.len(), B); // exactly the last B bytes
        assert!(src.ends_with(&t.content)); // it's the END of the line
    }

    #[test]
    fn tail_partial_snaps_to_char_boundary() {
        // Multi-byte chars: cutting at a raw byte offset must not split a char.
        // "é" is 2 bytes; 15 of them = 30 bytes, cap at 5 bytes -> 2 whole chars.
        let src = "é".repeat(15);
        let t = truncate_tail(&src, L, 5);
        assert!(t.last_line_partial);
        assert!(std::str::from_utf8(t.content.as_bytes()).is_ok());
        assert!(t.content.chars().all(|c| c == 'é'));
        assert!(t.content.len() <= 5);
    }

    #[test]
    fn line_under_cap_untouched() {
        let (out, cut) = truncate_line("short line", GREP_MAX_LINE_LEN);
        assert!(!cut);
        assert_eq!(out, "short line");
    }

    #[test]
    fn line_over_cap_truncated() {
        let (out, cut) = truncate_line(&"x".repeat(600), GREP_MAX_LINE_LEN);
        assert!(cut);
        assert!(out.ends_with("... [truncated]"));
        assert_eq!(
            out.chars().take(GREP_MAX_LINE_LEN).filter(|&c| c == 'x').count(),
            GREP_MAX_LINE_LEN
        );
    }

    #[test]
    fn line_cap_is_char_based_not_byte_based() {
        // 10 "é" chars = 20 bytes; cap at 5 CHARS keeps 5 chars + suffix.
        let (out, cut) = truncate_line(&"é".repeat(10), 5);
        assert!(cut);
        assert!(out.starts_with("ééééé... [truncated]"));
    }

    #[test]
    fn format_size_labels() {
        assert_eq!(format_size(512), "512B");
        assert_eq!(format_size(50 * 1024), "50.0KB");
        assert_eq!(format_size(1536), "1.5KB");
        assert_eq!(format_size(3 * 1024 * 1024 / 2), "1.5MB");
    }

    #[test]
    fn spill_writes_full_output_and_returns_path() {
        let raw = "line1\nline2\nline3\n";
        let path = spill_full_output("kg-test", raw).expect("spill");
        assert!(path.exists());
        let name = path.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("kg-test-"));
        assert!(name.ends_with(".log"));
        let back = std::fs::read_to_string(&path).unwrap();
        assert_eq!(back, raw);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn clamp_with_spill_noop_when_small() {
        let out = clamp_with_spill("tiny output", "kg-test").unwrap();
        assert_eq!(out, "tiny output");
        assert!(!out.contains("Full output"));
    }

    #[test]
    fn clamp_with_spill_bytes_variant_notice() {
        // Few lines but over the byte cap → the "(N.NKB limit)" notice variant.
        let raw = format!("{}\n{}\n{}", "a".repeat(30_000), "b".repeat(30_000), "c".repeat(30_000));
        let out = clamp_with_spill(&raw, "kg-test").unwrap();
        assert!(out.contains("Full output:"), "{out}");
        assert!(
            out.contains(&format!("({} limit)", format_size(MAX_OUTPUT_BYTES))),
            "bytes notice: {out}"
        );
        assert!(out.contains("of 3 ("), "totals the 3 lines: {out}");
        let path = out.rsplit("Full output:").next().unwrap().trim_end_matches(']').trim();
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn clamp_with_spill_partial_line_variant_notice() {
        // One line over the byte cap → the "Showing last X of line N (line is Y)" variant.
        let raw = "z".repeat(60_000);
        let out = clamp_with_spill(&raw, "kg-test").unwrap();
        // 60_000 bytes = 58.6KB; kept tail = 50.0KB.
        assert!(
            out.contains("Showing last 50.0KB of line 1 (line is 58.6KB)"),
            "partial notice: {out}"
        );
        assert!(out.contains("Full output:"), "{out}");
        let path = out.rsplit("Full output:").next().unwrap().trim_end_matches(']').trim();
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn clamp_with_spill_lines_variant_literal_numbers() {
        // Over the line cap → "Showing lines A-B of N" with exact numbers.
        let raw: String = (0..MAX_OUTPUT_LINES + 3).fold(String::new(), |mut acc, i| {
            let _ = writeln!(acc, "r{i}");
            acc
        });
        let out = clamp_with_spill(&raw, "kg-test").unwrap();
        let total = MAX_OUTPUT_LINES + 3;
        let start = total - MAX_OUTPUT_LINES + 1; // last MAX_OUTPUT_LINES kept
        assert!(
            out.contains(&format!("Showing lines {start}-{total} of {total}.")),
            "literal lines notice: {}",
            out.get(out.len().saturating_sub(120)..).unwrap_or(out.as_str())
        );
        let path = out.rsplit("Full output:").next().unwrap().trim_end_matches(']').trim();
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn clamp_with_spill_spills_and_notes_when_truncated() {
        // Many lines over the line cap -> tail kept + spill notice citing a path.
        let raw: String = (0..MAX_OUTPUT_LINES + 50).fold(String::new(), |mut acc, i| {
            let _ = writeln!(acc, "row {i}");
            acc
        });
        let out = clamp_with_spill(&raw, "kg-test").unwrap();
        assert!(out.contains("Full output:"));
        assert!(out.contains("Showing lines"));
        // The notice must point at a real file holding the full output.
        let path = out.rsplit("Full output:").next().unwrap().trim_end_matches(']').trim();
        let back = std::fs::read_to_string(path).unwrap();
        assert_eq!(back, raw);
        let _ = std::fs::remove_file(path);
    }
}
