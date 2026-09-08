//! Central backstop clamp at the tool-dispatch boundary (functional core).
//!
//! Every tool result funnels through [`crate::harness::turn::dispatch_tools`],
//! which is the one place pi never closed: pi's truncation is opt-in per tool,
//! so a tool that under-caps (or a new/custom tool with no truncation at all)
//! can inject an oversized `tool_result` and blow the model's input window —
//! exactly the 60-PNG `markdown_get_section` failure this whole change targets.
//!
//! [`clamp_reply`] is the mandatory floor under every tool:
//! - **Text** over a generous central ceiling → re-clamp inline to the normal
//!   50 KiB / 2000-line cap, spill the full block, and cite the path. The
//!   ceiling is set well *above* the per-tool cap (steps 1/2/5) so a correctly
//!   self-capping tool passes through untouched — this only fires on a gross
//!   violation (~100k-token class).
//! - **Images** → per-image byte cap, a per-result image *count* cap, and a
//!   per-result *aggregate* byte cap. Anything beyond the caps is dropped and
//!   summarized in one text placeholder. `View` already returns exactly one
//!   image, so in the normal path this is a no-op too.
//!
//! Pure values-in/values-out except the isolated [`spill_full_output`] call for
//! the oversized-text case.

use crate::ai::protocol::ToolContent;
use crate::tool::protocol::ToolReply;
use crate::tool::truncate::{MAX_OUTPUT_BYTES, MAX_OUTPUT_LINES, format_size, spill_full_output, truncate_head};

/// Backstop ceilings.
///
/// Deliberately looser than the per-tool caps so the clamp
/// is a no-op whenever a tool self-caps correctly; it only catches the
/// pathological (uncapped tool, or a tool that grossly exceeds its own cap).
#[derive(Debug, Clone)]
pub struct Limits {
    /// Central per-block text ceiling (bytes). Above this, a text block is
    /// re-clamped to the normal cap and the full block is spilled. ~400 KiB ≈
    /// ~100k tokens.
    pub text_ceiling_bytes: usize,
    /// Aggregate text ceiling across ALL text blocks in one result. Guards the
    /// case of many individually-sub-ceiling blocks summing to a huge payload.
    pub text_aggregate_ceiling_bytes: usize,
    /// Per-image base64 byte cap. Anthropic rejects inline images over ~5 MB;
    /// this matches pi's per-image `DEFAULT_MAX_BYTES` (4.5 MiB of base64).
    pub image_ceiling_base64_bytes: usize,
    /// Max images kept in a single tool result. No legitimate tool returns many
    /// after the markdown/pdf fan-out is removed; `View` returns exactly one.
    pub max_images: usize,
    /// Aggregate base64 byte budget across all images in one tool result.
    pub total_image_ceiling_base64_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            text_ceiling_bytes: 400 * 1024,
            text_aggregate_ceiling_bytes: 512 * 1024,
            image_ceiling_base64_bytes: 4 * 1024 * 1024 + 512 * 1024, // 4.5 MiB
            max_images: 8,
            total_image_ceiling_base64_bytes: 8 * 1024 * 1024, // 8 MiB
        }
    }
}

/// A clamped reply plus the spill path, if the text backstop fired (for the
/// caller to log/telemetry — the path is already cited inside the reply text).
#[derive(Debug, Clone)]
pub struct Clamped {
    pub reply: ToolReply,
    pub spill: Option<String>,
}

/// Char-boundary byte prefix of `s` (never splits a UTF-8 char).
fn byte_prefix(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end = end.saturating_sub(1);
    }
    // `end` is a char boundary by the walk above, so `get` always yields `Some`;
    // the fallback keeps the expression total without a slicing panic.
    s.get(..end).unwrap_or(s)
}

/// The central backstop. Applied to every `Ok` tool reply at the dispatch
/// boundary. See the module docs for the contract.
#[must_use]
pub fn clamp_reply(reply: ToolReply, limits: &Limits) -> Clamped {
    let blocks = reply.0;
    let total_images = blocks.iter().filter(|b| matches!(b, ToolContent::Image { .. })).count();

    let mut out: Vec<ToolContent> = Vec::with_capacity(blocks.len());
    let mut spill_path: Option<String> = None;
    let mut text_bytes_total = 0usize;
    let mut images_kept = 0usize;
    let mut image_bytes_total = 0usize;
    let mut images_omitted = 0usize;

    for block in blocks {
        match block {
            ToolContent::Text(s) => {
                // Fire on either the per-block ceiling OR the running aggregate
                // across all text blocks (many sub-ceiling blocks can still sum
                // to a huge payload).
                let over = s.len() > limits.text_ceiling_bytes
                    || text_bytes_total.saturating_add(s.len()) > limits.text_aggregate_ceiling_bytes;
                if !over {
                    text_bytes_total = text_bytes_total.saturating_add(s.len());
                    out.push(ToolContent::Text(s));
                    continue;
                }
                // Backstop fired: re-clamp inline to the normal cap, spill full.
                let head = truncate_head(&s, MAX_OUTPUT_LINES, MAX_OUTPUT_BYTES);
                let inline = if head.first_line_exceeds {
                    byte_prefix(&s, MAX_OUTPUT_BYTES).to_string()
                } else {
                    head.content
                };
                text_bytes_total = text_bytes_total.saturating_add(inline.len());
                let note = spill_full_output("kg-clamp", &s).map_or_else(
                    |_| {
                        format!(
                            "[tool result was {}, over the {} ceiling; showing the first {} (full-output spill failed)]",
                            format_size(s.len()),
                            format_size(limits.text_ceiling_bytes),
                            format_size(inline.len())
                        )
                    },
                    |path| {
                        let p = path.display().to_string();
                        let note = format!(
                            "[tool result was {}, over the {} ceiling; showing the first {}. Full output: {}]",
                            format_size(s.len()),
                            format_size(limits.text_ceiling_bytes),
                            format_size(inline.len()),
                            p
                        );
                        spill_path.get_or_insert(p);
                        note
                    },
                );
                out.push(ToolContent::Text(format!("{inline}\n\n{note}")));
            }
            ToolContent::Image {
                media_type,
                data_base64,
            } => {
                let sz = data_base64.len();
                let over_caps = sz > limits.image_ceiling_base64_bytes
                    || images_kept >= limits.max_images
                    || image_bytes_total.saturating_add(sz) > limits.total_image_ceiling_base64_bytes;
                if over_caps {
                    images_omitted = images_omitted.saturating_add(1);
                } else {
                    images_kept = images_kept.saturating_add(1);
                    image_bytes_total = image_bytes_total.saturating_add(sz);
                    out.push(ToolContent::Image {
                        media_type,
                        data_base64,
                    });
                }
            }
        }
    }

    // One aggregated placeholder for any images the caps dropped.
    if images_omitted > 0 {
        out.push(ToolContent::Text(format!(
            "[{images_omitted} of {total_images} image(s) omitted to bound context — \
             each View returns one image; re-view a specific page/region individually]"
        )));
    }

    Clamped {
        reply: ToolReply(out),
        spill: spill_path,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(s: &str) -> ToolContent {
        ToolContent::Text(s.to_string())
    }
    fn img(base64_len: usize) -> ToolContent {
        ToolContent::Image {
            media_type: "image/png".to_string(),
            data_base64: "a".repeat(base64_len),
        }
    }

    #[test]
    fn small_text_passes_through_untouched() {
        let reply = ToolReply(vec![text("hello world")]);
        let c = clamp_reply(reply.clone(), &Limits::default());
        assert_eq!(c.reply, reply);
        assert!(c.spill.is_none());
    }

    #[test]
    fn normally_capped_text_is_a_noop() {
        // A tool that self-capped to ~50 KiB + a notice is well under the
        // central ceiling, so the backstop must not touch it.
        let capped = format!("{}\n\n[Showing lines 1-2000 of 9999. …]", "x".repeat(51_200));
        let reply = ToolReply(vec![text(&capped)]);
        let c = clamp_reply(reply.clone(), &Limits::default());
        assert_eq!(c.reply, reply, "backstop fired on normally-capped output");
        assert!(c.spill.is_none());
    }

    #[test]
    fn aggregate_text_across_blocks_is_bounded() {
        // Each block is under the 400 KiB per-block ceiling, but three of them
        // sum past the 512 KiB aggregate — the backstop must still fire.
        let block = "y".repeat(300 * 1024);
        let reply = ToolReply(vec![text(&block), text(&block), text(&block)]);
        let c = clamp_reply(reply, &Limits::default());
        assert!(c.spill.is_some(), "aggregate text budget should fire");
        let out = c.reply.as_text();
        assert!(
            out.contains("Full output:"),
            "spilled block must cite a path: {}",
            out.get(out.len().saturating_sub(80)..).unwrap_or(out.as_str())
        );
        // Clean up the spill file(s).
        for seg in out.split("Full output:").skip(1) {
            let path = seg.trim_start().split(']').next().unwrap_or("").trim();
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn oversized_text_is_clamped_and_spilled() {
        let big = "line\n".repeat(120_000); // ~600 KiB, over the 400 KiB ceiling
        let reply = ToolReply(vec![text(&big)]);
        let c = clamp_reply(reply, &Limits::default());
        let out = c.reply.as_text();
        assert!(out.len() < big_len_guard(), "inline must be re-clamped small");
        assert!(out.contains("Full output:"), "must cite a spill path: {out:?}");
        assert!(out.contains("over the"), "must note the ceiling");
        let path = c.spill.expect("spill path recorded");
        let back = std::fs::read_to_string(&path).unwrap();
        assert_eq!(back.len(), big.len(), "spill holds the full block");
        let _ = std::fs::remove_file(&path);
    }

    fn big_len_guard() -> usize {
        // inline (≤ ~51.2 KiB) + a short note; comfortably under 60 KiB.
        60 * 1024
    }

    #[test]
    fn single_small_image_kept() {
        let reply = ToolReply(vec![text("caption"), img(10_000)]);
        let c = clamp_reply(reply, &Limits::default());
        let imgs = c
            .reply
            .0
            .iter()
            .filter(|b| matches!(b, ToolContent::Image { .. }))
            .count();
        assert_eq!(imgs, 1);
        assert!(!c.reply.as_text().contains("omitted"));
    }

    #[test]
    fn too_large_single_image_dropped_to_placeholder() {
        let reply = ToolReply(vec![img(5 * 1024 * 1024)]); // > 4.5 MiB per-image cap
        let c = clamp_reply(reply, &Limits::default());
        let imgs = c
            .reply
            .0
            .iter()
            .filter(|b| matches!(b, ToolContent::Image { .. }))
            .count();
        assert_eq!(imgs, 0, "over-cap image must be dropped");
        assert!(c.reply.as_text().contains("1 of 1 image(s) omitted"));
    }

    #[test]
    fn excess_images_beyond_count_cap_are_dropped() {
        let mut blocks = vec![text("many figures")];
        for _ in 0..12 {
            blocks.push(img(1000)); // tiny, so only the COUNT cap bites
        }
        let c = clamp_reply(ToolReply(blocks), &Limits::default());
        let imgs = c
            .reply
            .0
            .iter()
            .filter(|b| matches!(b, ToolContent::Image { .. }))
            .count();
        assert_eq!(imgs, Limits::default().max_images, "kept exactly the count cap");
        assert!(c.reply.as_text().contains("4 of 12 image(s) omitted"));
    }

    #[test]
    fn aggregate_image_bytes_cap_bites_before_count() {
        // Three 4 MiB images: first fits, second pushes total to 8 MiB (== cap,
        // still fits), third would exceed the 8 MiB aggregate -> dropped.
        let four_mib = 4 * 1024 * 1024;
        let reply = ToolReply(vec![img(four_mib), img(four_mib), img(four_mib)]);
        let c = clamp_reply(reply, &Limits::default());
        let imgs = c
            .reply
            .0
            .iter()
            .filter(|b| matches!(b, ToolContent::Image { .. }))
            .count();
        assert_eq!(imgs, 2, "aggregate byte budget kept only two");
        assert!(c.reply.as_text().contains("1 of 3 image(s) omitted"));
    }
}
