//! PDF tools — port of the Python reference's `tool/tools.py::pdf_*`.
//!
//! - [`PdfGetNumPages`]: returns the page count as a text block. Cheap
//!   enough that the agent can call it before deciding how to navigate.
//! - [`render_pdf_page_png`]: renders one zero-indexed page to PNG bytes. The
//!   page-viewing tool `view` (see [`crate::tool::view`]) calls this to render
//!   a PDF page on demand — one page per call (no multi-page fan-out).
//!
//! ## libpdfium provisioning
//!
//! We render via [`pdfium-render`](https://crates.io/crates/pdfium-render),
//! which binds to a Pdfium dynamic library at runtime. The dylib itself is
//! fetched at **build time** by `build.rs` from
//! [`bblanchon/pdfium-binaries`](https://github.com/bblanchon/pdfium-binaries/releases),
//! cached under `$OUT_DIR/pdfium/`, and its absolute path is baked into
//! the binary as the `KERNELGUY_PDFIUM_LIB` env (`env!()` resolves at
//! compile time). No runtime download, no `LD_LIBRARY_PATH` /
//! `DYLD_LIBRARY_PATH` setup, and no third-party companion crate.
//!
//! ## Single cached instance
//!
//! `pdfium-render 0.9`'s [`Pdfium`] is `Sync` with the default
//! `thread_safe` feature (all calls funnel through a global mutex inside
//! the binding), so we keep one shared instance in a [`OnceLock`] for
//! the lifetime of the process — the only cost is a single `dlopen` on
//! the first PDF tool call.
//!
//! ## Blocking pool
//!
//! Pdfium operations (parsing, page-counting, rendering) are blocking
//! C++ calls. We run them on tokio's blocking pool via
//! [`tokio::task::spawn_blocking`] so the harness's async runtime stays
//! free.

use std::io::Cursor;
use std::sync::{Arc, OnceLock};

use image::{DynamicImage, ImageFormat};
use pdfium_render::prelude::{PdfPageRenderRotation, PdfRenderConfig, Pdfium};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::exec::sandbox::Sandbox;
use crate::tool::{Tool, ToolOutput};

/// Absolute path of the prebuilt libpdfium baked in by `build.rs`.
const PDFIUM_LIB_PATH: &str = env!("KERNELGUY_PDFIUM_LIB");

/// Returns a process-wide cached [`Pdfium`]. First call performs the
/// `dlopen`; subsequent calls hand back the same reference. Errors are
/// cached too, so a missing/corrupt dylib doesn't cause repeated failed
/// loads.
fn pdfium() -> Result<&'static Pdfium, String> {
    static CELL: OnceLock<Result<Pdfium, String>> = OnceLock::new();
    CELL.get_or_init(|| {
        let bindings = Pdfium::bind_to_library(PDFIUM_LIB_PATH)
            .map_err(|e| format!("failed to bind libpdfium at {PDFIUM_LIB_PATH}: {e}"))?;
        Ok(Pdfium::new(bindings))
    })
    .as_ref()
    .map_err(std::clone::Clone::clone)
}

// ─── pdf_get_num_pages ──────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema)]
pub struct PdfGetNumPagesArgs {
    /// Workspace-relative path of the PDF file (e.g. `docs/spec.pdf`).
    pub path: String,
}

pub struct PdfGetNumPages {
    pub sandbox: Arc<Sandbox>,
}

impl Tool for PdfGetNumPages {
    type Args = PdfGetNumPagesArgs;
    const NAME: &'static str = "pdf_get_num_pages";
    const DESCRIPTION: &'static str = "Get the total number of pages in a PDF file. Useful before deciding how to read it: \
         a short PDF can be viewed end-to-end, a long one is best navigated by reading page 0 \
         (cover/TOC) and then jumping by page number.";

    async fn call(&self, args: PdfGetNumPagesArgs) -> ToolOutput {
        // `host_path` (not `path`) so the lookup transparently follows
        // any ro overlay (e.g. PDFs mounted under `docs/` via
        // `mount_docs`) to the real host file.
        let host_path = self.sandbox.host_path(&args.path).map_err(|e| e.to_string())?;
        let path = args.path.clone();
        let body = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let pdfium = pdfium()?;
            let document = pdfium
                .load_pdf_from_file(&host_path, None)
                .map_err(|e| format!("failed to open {path}: {e}"))?;
            let n = document.pages().len();
            Ok(format!("The document has {n} pages"))
        })
        .await
        .map_err(|e| format!("pdf task panicked: {e}"))??;
        Ok(body.into())
    }
}

// ─── pdf_view_page ──────────────────────────────────────────────────────────

/// Longest side of the rendered page, in pixels. Mirrors the Python
/// reference's `MAX_DIMENSION = 2000` so transcripts diff cleanly. Each
/// page is scaled by `MAX_DIMENSION / max(width, height)` (in points)
/// before rendering, so portrait and landscape pages both come out at
/// roughly the same total pixel count.
const PDF_VIEW_MAX_DIMENSION: f32 = 2000.0;

/// Render one zero-indexed PDF page to PNG bytes.
///
/// # Errors
///
/// Returns a message describing the first step that failed: the `pdfium` shared
/// library could not be loaded, `host_path` is not an openable/parseable PDF,
/// `page_number` is negative or past the last page, or pdfium failed to fetch,
/// render, or decode the page. Also errors if PNG encoding of the rendered
/// bitmap fails.
pub fn render_pdf_page_png(host_path: &std::path::Path, path: &str, page_number: i64) -> Result<Vec<u8>, String> {
    let pdfium = pdfium()?;
    let document = pdfium
        .load_pdf_from_file(host_path, None)
        .map_err(|e| format!("failed to open {path}: {e}"))?;
    let pages = document.pages();
    let total = i64::from(pages.len());
    // Validate and narrow in one step: pdfium indexes pages with an `i32`, and a
    // page count is a `u16`, so any in-range `page_number` converts — which makes
    // a failed `try_from` just another way of being out of range.
    let Some(index) = i32::try_from(page_number)
        .ok()
        .filter(|_| page_number >= 0 && page_number < total)
    else {
        return Err(format!(
            "page_number {page_number} out of range for {path} ({total} pages, zero-indexed)"
        ));
    };
    let page = pages.get(index).map_err(|e| format!("page {page_number}: {e}"))?;
    // Page size is in PDF points (1/72"). Scale by the ratio of
    // MAX_DIMENSION to the longest side so the resulting bitmap's longest
    // side is ~MAX_DIMENSION pixels, matching the Python reference.
    let w_pts = page.width().value;
    let h_pts = page.height().value;
    let max_pts = w_pts.max(h_pts).max(1.0);
    let scale = PDF_VIEW_MAX_DIMENSION / max_pts;
    let config = PdfRenderConfig::new()
        .scale_page_by_factor(scale)
        .rotate_if_landscape(PdfPageRenderRotation::None, false);
    let bitmap = page
        .render_with_config(&config)
        .map_err(|e| format!("render page {page_number}: {e}"))?;
    let dynamic_image: DynamicImage = bitmap
        .as_image()
        .map_err(|e| format!("decode bitmap for page {page_number}: {e}"))?;
    let mut buf: Vec<u8> = Vec::new();
    dynamic_image
        .write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
        .map_err(|e| format!("encode png for page {page_number}: {e}"))?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;
    use serde_json::json;
    use std::fmt::Write as _;

    /// Build a minimal valid PDF with `num_pages` empty 300x300pt pages.
    /// Hand-assembled so tests don't need a fixture file or a PDF-writer
    /// dep — the format is small enough that this stays readable.
    fn make_minimal_pdf(num_pages: usize) -> Vec<u8> {
        assert!(num_pages >= 1, "PDF must have at least one page");
        let mut out: Vec<u8> = Vec::new();
        let mut offsets: Vec<usize> = Vec::new();

        // Header. The leading `%PDF-1.4` is the version; the `%âãÏÓ`
        // binary marker tells viewers the file should be treated as
        // binary. Pdfium tolerates plain text but the marker is cheap.
        out.extend_from_slice(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n");

        // Object 1: catalog → pages dictionary at object 2.
        offsets.push(out.len());
        out.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

        // Object 2: pages dictionary listing /Kids and /Count.
        let kids: String = (0..num_pages).fold(String::new(), |mut acc, i| {
            let _ = write!(acc, "{} 0 R ", i.saturating_add(3));
            acc
        });
        offsets.push(out.len());
        out.extend_from_slice(
            format!("2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {num_pages} >>\nendobj\n").as_bytes(),
        );

        // Objects 3..: one /Page per object. Empty MediaBox is enough.
        for i in 0..num_pages {
            offsets.push(out.len());
            out.extend_from_slice(
                format!(
                    "{} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] >>\nendobj\n",
                    i.saturating_add(3)
                )
                .as_bytes(),
            );
        }

        // xref table. First entry is the special `0000000000 65535 f`
        // free-list head; the rest are 10-digit-padded offsets.
        let xref_offset = out.len();
        let total_objs = num_pages.saturating_add(3);
        out.extend_from_slice(format!("xref\n0 {total_objs}\n").as_bytes());
        out.extend_from_slice(b"0000000000 65535 f \n");
        for &offset in &offsets {
            out.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }

        // Trailer + startxref + EOF.
        out.extend_from_slice(
            format!("trailer\n<< /Size {total_objs} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n").as_bytes(),
        );
        out
    }

    #[tokio::test]
    async fn returns_page_count_for_three_page_pdf() {
        let sb = Arc::new(Sandbox::new().expect("sandbox"));
        let pdf = make_minimal_pdf(3);
        let host = sb.host_path_for_write("doc.pdf").expect("host path");
        std::fs::write(&host, pdf).expect("write pdf");

        let tool = PdfGetNumPages { sandbox: sb };
        let out = tool.call_json(json!({ "path": "doc.pdf" })).await.unwrap();
        assert_eq!(out.as_text(), "The document has 3 pages");
    }

    #[tokio::test]
    async fn returns_page_count_for_one_page_pdf() {
        let sb = Arc::new(Sandbox::new().expect("sandbox"));
        let pdf = make_minimal_pdf(1);
        let host = sb.host_path_for_write("solo.pdf").expect("host path");
        std::fs::write(&host, pdf).expect("write pdf");

        let tool = PdfGetNumPages { sandbox: sb };
        let out = tool.call_json(json!({ "path": "solo.pdf" })).await.unwrap();
        assert_eq!(out.as_text(), "The document has 1 pages");
    }

    #[tokio::test]
    async fn missing_file_returns_error() {
        let sb = Arc::new(Sandbox::new().expect("sandbox"));
        let tool = PdfGetNumPages { sandbox: sb };
        let result = tool.call_json(json!({ "path": "no_such.pdf" })).await;
        assert!(result.is_err(), "missing file must error: {result:?}");
    }

    #[tokio::test]
    async fn malformed_pdf_returns_error() {
        let sb = Arc::new(Sandbox::new().expect("sandbox"));
        sb.write("garbage.pdf", "this is not a pdf").expect("write garbage");

        let tool = PdfGetNumPages { sandbox: sb };
        let result = tool.call_json(json!({ "path": "garbage.pdf" })).await;
        assert!(result.is_err(), "malformed pdf must error: {result:?}");
    }

    #[tokio::test]
    async fn resolves_through_ro_overlay() {
        // A PDF mounted via `add_ro` (the same mechanism `mount_docs`
        // uses) must be readable through the workspace-relative path —
        // that's the whole point of overlays. Regression guard for the
        // `path()` vs `host_path()` mistake.
        let upstream = tempfile::TempDir::new().unwrap();
        let pdf_bytes = make_minimal_pdf(2);
        std::fs::write(upstream.path().join("spec.pdf"), &pdf_bytes).unwrap();

        let mut sb = Sandbox::new().unwrap();
        sb.add_ro(upstream.path(), "/workspace/refs");
        let sb = Arc::new(sb);

        let tool = PdfGetNumPages { sandbox: sb };
        let out = tool.call_json(json!({ "path": "refs/spec.pdf" })).await.unwrap();
        assert_eq!(out.as_text(), "The document has 2 pages");
    }
}
