//! `view` — render ONE image per call from a path or URL.
//!
//! Unifies all image access behind a single one-image-per-call contract: a
//! workspace-relative file (`docs/fig.png`, `refs/spec.pdf`), an `http(s)://`
//! or `data:` URL, or a PDF page. This is the deliberate structural property
//! that makes the fan-out that killed a 10h run (`markdown_get_section`
//! returning ~60 inline PNGs) impossible: no tool inlines many images.
//!
//! It subsumes the old `pdf_view_page` (fold: `page` selects the PDF page) and
//! the image-embedding half of `markdown_get_section` (now deleted). The image
//! fetch machinery (<data:/http/file/pdf>) lived in `markdown.rs`; it moved here
//! because this is its only remaining consumer.
//!
//! Every fetched image is clamped to fit inline: ≤ 2000 px on the long edge and
//! < 4.5 MiB of base64 (below Anthropic's ~5 MB inline limit), re-encoding
//! PNG→JPEG down a quality ladder and progressively downscaling — pi's
//! `resizeImageInProcess` algorithm. If it still can't fit, one text
//! placeholder is returned instead of a giant block. The central clamp
//! (`tool::clamp`) is the backstop under this; this is the primary shaping.

use std::io::Cursor;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageEncoder, ImageFormat, ImageReader};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::ai::protocol::ToolContent;
use crate::domain::convert::{f32_to_u32_saturating, u32_to_f32_lossy};
use crate::exec::sandbox::Sandbox;
use crate::tool::pdf;
use crate::tool::{Tool, ToolOutput};

/// Longest edge (px) of a returned image. Matches pi's 2000×2000 clamp and the
/// old `pdf_view_page` render size, so dense pages (equations, small text) stay
/// legible.
const VIEW_MAX_DIMENSION: u32 = 2000;

/// Per-image base64 byte ceiling: 4.5 MiB, below Anthropic's ~5 MB inline-image
/// limit. Matches pi's per-image `DEFAULT_MAX_BYTES`.
const VIEW_MAX_BASE64_BYTES: usize = 4 * 1024 * 1024 + 512 * 1024;

/// Process-wide HTTP client for `http(s)://` images. Built once with a fetch
/// timeout so a slow host can't wedge a `view` call. Cloning a `reqwest::Client`
/// is cheap (it's `Arc` internally).
static HTTP: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap_or_default()
});

// ─── fetch ──────────────────────────────────────────────────────────────────

/// Sniff an image's media type from its leading magic bytes — the formats the
/// Anthropic image API accepts (PNG / JPEG / GIF / WebP).
fn sniff_media_type(data: &[u8]) -> Option<&'static str> {
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if data.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WEBP".as_slice()) {
        Some("image/webp")
    } else {
        None
    }
}

/// Decode the part of a `data:` URL after the `data:` prefix
/// (`[<mediatype>][;base64],<payload>`). Base64 payloads are decoded; other
/// encodings hand back raw bytes (the caller's magic-byte sniff rejects them).
fn decode_data_url(rest: &str) -> Result<Vec<u8>, String> {
    let comma = rest
        .find(',')
        .ok_or_else(|| "malformed data: URL (missing comma)".to_string())?;
    // `comma` comes from `find(',')` and `,` is a single ASCII byte, so both
    // `comma` and `comma + 1` are char boundaries and `get` always yields `Some`.
    let meta = rest.get(..comma).unwrap_or("");
    let payload = rest.get(comma.saturating_add(1)..).unwrap_or("");
    if meta.split(';').any(|s| s.eq_ignore_ascii_case("base64")) {
        BASE64
            .decode(payload.as_bytes())
            .map_err(|e| format!("invalid base64 in data: URL: {e}"))
    } else {
        Ok(payload.as_bytes().to_vec())
    }
}

fn is_local_pdf(url: &str) -> bool {
    if url.starts_with("data:") || url.starts_with("http://") || url.starts_with("https://") {
        return false;
    }
    let local = url.strip_prefix("file://").unwrap_or(url);
    std::path::Path::new(local)
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("pdf"))
}

/// Resolve a local (non-URL) image reference to its host path through the
/// sandbox, so it can't escape the workspace. `base_dir` prefixes a relative
/// reference (the markdown file's dir, historically; empty for direct `view`).
fn local_path(url: &str, base_dir: &str, sandbox: &Sandbox) -> Result<std::path::PathBuf, String> {
    let local = url.strip_prefix("file://").unwrap_or(url);
    let rel = if base_dir.is_empty() {
        local.to_string()
    } else {
        format!("{base_dir}/{local}")
    };
    sandbox
        .host_path(&rel)
        .map_err(|e| format!("bad image path {url:?}: {e}"))
}

/// Fetch raw bytes behind an image reference: `data:` inline, `http(s)://` over
/// the network, otherwise a local path resolved through the sandbox.
async fn fetch_bytes(url: &str, base_dir: &str, sandbox: &Sandbox) -> Result<Vec<u8>, String> {
    if let Some(rest) = url.strip_prefix("data:") {
        return decode_data_url(rest);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        let resp = HTTP
            .get(url)
            .send()
            .await
            .map_err(|e| format!("GET {url} failed: {e}"))?
            .error_for_status()
            .map_err(|e| format!("GET {url} failed: {e}"))?;
        let bytes = resp.bytes().await.map_err(|e| format!("reading {url} failed: {e}"))?;
        return Ok(bytes.to_vec());
    }
    let host = local_path(url, base_dir, sandbox)?;
    std::fs::read(&host).map_err(|e| format!("reading image {url:?} failed: {e}"))
}

/// Fetch and validate an image, returning `(media_type, raw bytes)`. A local
/// `.pdf` renders `page` (zero-indexed, default 0) to a PNG; everything else is
/// fetched and its format sniffed. `page` is ignored for non-PDF references.
async fn fetch_image(target: &str, base_dir: &str, sandbox: &Sandbox, page: i64) -> Result<(String, Vec<u8>), String> {
    if is_local_pdf(target) {
        let host = local_path(target, base_dir, sandbox)?;
        let path = target.to_string();
        let data = tokio::task::spawn_blocking(move || pdf::render_pdf_page_png(&host, &path, page))
            .await
            .map_err(|e| format!("pdf task panicked: {e}"))??;
        return Ok(("image/png".to_string(), data));
    }
    let data = fetch_bytes(target, base_dir, sandbox).await?;
    let media_type = sniff_media_type(&data).ok_or_else(|| format!("unsupported image format: {target}"))?;
    Ok((media_type.to_string(), data))
}

// ─── inline-size clamp (pi's resizeImageInProcess) ───────────────────────────

fn image_format_for_media_type(media_type: &str) -> ImageFormat {
    match media_type {
        "image/jpeg" => ImageFormat::Jpeg,
        "image/gif" => ImageFormat::Gif,
        "image/webp" => ImageFormat::WebP,
        _ => ImageFormat::Png,
    }
}

/// base64 length of `n` raw bytes (`ceil(n/3) * 4`), matching pi's estimate.
const fn base64_len(n: usize) -> usize {
    n.div_ceil(3).saturating_mul(4)
}

fn encode_png(img: &DynamicImage) -> Option<Vec<u8>> {
    let mut buf = Vec::new();
    img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png).ok()?;
    Some(buf)
}

fn encode_jpeg(img: &DynamicImage, quality: u8) -> Option<Vec<u8>> {
    let rgb = img.to_rgb8(); // JPEG has no alpha channel
    let mut buf = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality)
        .write_image(rgb.as_raw(), rgb.width(), rgb.height(), image::ExtendedColorType::Rgb8)
        .ok()?;
    Some(buf)
}

/// Clamp an image to fit inline: ≤ [`VIEW_MAX_DIMENSION`] px on the long edge
/// AND < [`VIEW_MAX_BASE64_BYTES`] of base64. Tries PNG then a JPEG quality
/// ladder, downscaling ×0.75 until it fits or hits 1×1. Returns `None` if it
/// can never fit (→ caller emits a text placeholder). Port of pi's
/// `resizeImageInProcess`.
fn shrink_to_inline_limit(media_type: &str, data: Vec<u8>) -> Option<(String, Vec<u8>)> {
    let mut reader = ImageReader::new(Cursor::new(&data));
    reader.set_format(image_format_for_media_type(media_type));
    // Cap decode allocation so a "decompression bomb" (tiny file, enormous
    // dimensions) from a model-supplied URL/path can't OOM the host.
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(512 * 1024 * 1024); // 512 MiB decode ceiling
    reader.limits(limits);
    let Ok(img) = reader.decode() else {
        // Can't decode (e.g. a GIF/WebP — the `image` crate is built without
        // those codecs, or a bomb over the decode limit). If it already fits
        // inline, pass it through untouched (Anthropic accepts it); otherwise
        // we can't shrink it.
        return (base64_len(data.len()) < VIEW_MAX_BASE64_BYTES).then(|| (media_type.to_string(), data));
    };
    let (w, h) = img.dimensions();

    // Already within all limits: pass through untouched.
    if w <= VIEW_MAX_DIMENSION && h <= VIEW_MAX_DIMENSION && base64_len(data.len()) < VIEW_MAX_BASE64_BYTES {
        return Some((media_type.to_string(), data));
    }

    // Fit to the max box first (aspect-preserving).
    let fitted = if w > VIEW_MAX_DIMENSION || h > VIEW_MAX_DIMENSION {
        img.resize(VIEW_MAX_DIMENSION, VIEW_MAX_DIMENSION, FilterType::Lanczos3)
    } else {
        img
    };
    let (base_w, base_h) = fitted.dimensions();

    let qualities = [80u8, 85, 70, 55, 40];
    let (mut cw, mut ch) = (base_w, base_h);
    loop {
        let scaled = if (cw, ch) == (base_w, base_h) {
            fitted.clone()
        } else {
            fitted.resize_exact(cw, ch, FilterType::Lanczos3)
        };

        if let Some(png) = encode_png(&scaled)
            && base64_len(png.len()) < VIEW_MAX_BASE64_BYTES
        {
            return Some(("image/png".to_string(), png));
        }
        for &q in &qualities {
            if let Some(jpg) = encode_jpeg(&scaled, q)
                && base64_len(jpg.len()) < VIEW_MAX_BASE64_BYTES
            {
                return Some(("image/jpeg".to_string(), jpg));
            }
        }

        if cw == 1 && ch == 1 {
            return None;
        }
        let nw = f32_to_u32_saturating((u32_to_f32_lossy(cw) * 0.75).floor()).max(1);
        let nh = f32_to_u32_saturating((u32_to_f32_lossy(ch) * 0.75).floor()).max(1);
        if nw == cw && nh == ch {
            return None;
        }
        cw = nw;
        ch = nh;
    }
}

// ─── tool ─────────────────────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema)]
pub struct ViewArgs {
    /// What to view — exactly one image is returned. Accepts a workspace-
    /// relative file (`docs/fig.png`, `refs/spec.pdf`), an `http(s)://` or
    /// `data:` URL, or a local `.pdf` (use `page` to pick the page).
    pub path: String,
    /// For PDFs: zero-indexed page to render (default 0). Ignored for raster
    /// images. There is no "all pages" — one image per call, by design; call
    /// again with the next page, using `pdf_get_num_pages` to bound the range.
    #[serde(default)]
    pub page: Option<i64>,
}

pub struct View {
    pub sandbox: Arc<Sandbox>,
}

impl Tool for View {
    type Args = ViewArgs;
    const NAME: &'static str = "view";
    const DESCRIPTION: &'static str = "View a single image so you can SEE it (vision input): a workspace image file \
         (png/jpg/jpeg/gif/webp), an http(s)/data: URL, or a PDF page. For a PDF, pass \
         `page` (zero-indexed; use pdf_get_num_pages to bound it) — one page per call. Exactly \
         one image is returned per call; large images are auto-downscaled to fit. This is how you \
         read diagrams/figures in docs: `read` a `.md` shows the literal `![alt](path)`, then \
         `view` that path to see the figure.";

    async fn call(&self, args: ViewArgs) -> ToolOutput {
        let page = args.page.unwrap_or(0);
        let (media_type, data) = fetch_image(&args.path, "", &self.sandbox, page).await?;
        // Decode + Lanczos3 resizes + re-encode are CPU-bound and can run on a
        // large raster; keep them off the async runtime thread (which drives
        // other spawn_local expanders + the gpu_job mpsc).
        let shrunk = tokio::task::spawn_blocking(move || shrink_to_inline_limit(&media_type, data))
            .await
            .map_err(|e| format!("image processing task panicked: {e}"))?;
        match shrunk {
            Some((media_type, bytes)) => Ok(ToolContent::Image {
                media_type,
                data_base64: BASE64.encode(&bytes),
            }
            .into()),
            None => Ok("[Image omitted: could not be resized below the inline image size limit.]".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;
    use serde_json::json;

    fn png_bytes(width: u32, height: u32) -> Vec<u8> {
        let image = image::RgbaImage::from_pixel(width, height, image::Rgba([32, 64, 128, 255]));
        let mut bytes = Vec::new();
        image.write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png).unwrap();
        bytes
    }

    fn make_minimal_pdf() -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        let mut offsets: Vec<usize> = Vec::new();
        out.extend_from_slice(b"%PDF-1.4\n");
        for object in [
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n".as_slice(),
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n".as_slice(),
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] >>\nendobj\n".as_slice(),
        ] {
            offsets.push(out.len());
            out.extend_from_slice(object);
        }
        let xref = out.len();
        out.extend_from_slice(format!("xref\n0 {}\n", offsets.len().saturating_add(1)).as_bytes());
        out.extend_from_slice(b"0000000000 65535 f \n");
        for offset in offsets {
            out.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(format!("trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes());
        out
    }

    #[test]
    fn sniff_detects_supported_formats_only() {
        assert_eq!(sniff_media_type(b"\x89PNG\r\n\x1a\nrest"), Some("image/png"));
        assert_eq!(sniff_media_type(b"\xff\xd8\xff\xe0rest"), Some("image/jpeg"));
        assert_eq!(sniff_media_type(b"GIF89a...."), Some("image/gif"));
        assert_eq!(sniff_media_type(b"GIF87a...."), Some("image/gif"));
        assert_eq!(sniff_media_type(b"RIFF\x00\x00\x00\x00WEBPmore"), Some("image/webp"));
        assert_eq!(sniff_media_type(b"BM....."), None); // BMP is not an accepted inline format
        assert_eq!(sniff_media_type(b""), None);
    }

    #[test]
    fn decode_data_url_base64_and_plain() {
        let bytes = png_bytes(1, 1);
        let enc = BASE64.encode(&bytes);
        assert_eq!(decode_data_url(&format!("image/png;base64,{enc}")).unwrap(), bytes);
        assert_eq!(decode_data_url("text/plain,hello").unwrap(), b"hello");
        assert!(decode_data_url("no-comma-here").is_err());
    }

    #[test]
    fn small_image_passes_through_untouched() {
        let bytes = png_bytes(16, 16);
        let (mt, out) = shrink_to_inline_limit("image/png", bytes.clone()).unwrap();
        assert_eq!(mt, "image/png");
        assert_eq!(out, bytes, "a small image must not be re-encoded");
    }

    #[test]
    fn oversized_dimension_image_is_downscaled() {
        let bytes = png_bytes(4096, 1024);
        let (mt, out) = shrink_to_inline_limit("image/png", bytes).unwrap();
        // Long edge clamped to 2000; aspect preserved (4096x1024 -> 2000x500).
        let (w, h) = png_dimensions_any(&mt, &out);
        assert!(w <= VIEW_MAX_DIMENSION && h <= VIEW_MAX_DIMENSION, "got {w}x{h}");
        assert_eq!((w, h), (2000, 500));
    }

    fn png_dimensions_any(media_type: &str, bytes: &[u8]) -> (u32, u32) {
        let mut r = ImageReader::new(Cursor::new(bytes));
        r.set_format(image_format_for_media_type(media_type));
        r.decode().unwrap().dimensions()
    }

    #[tokio::test]
    async fn view_data_url_returns_one_image_block() {
        let sb = Arc::new(Sandbox::new().unwrap());
        let bytes = png_bytes(8, 8);
        let enc = BASE64.encode(&bytes);
        let tool = View { sandbox: sb };
        let reply = tool
            .call_json(json!({ "path": format!("data:image/png;base64,{enc}") }))
            .await
            .unwrap();
        assert_eq!(reply.0.len(), 1);
        match &reply.0[0] {
            ToolContent::Image {
                media_type,
                data_base64,
            } => {
                assert_eq!(media_type, "image/png");
                assert_eq!(BASE64.decode(data_base64).unwrap(), bytes);
            }
            other @ ToolContent::Text(_) => panic!("expected image, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn view_local_workspace_image() {
        let sb = Arc::new(Sandbox::new().unwrap());
        let bytes = png_bytes(2, 2);
        let host = sb.host_path_for_write("docs/img/fig.png").unwrap();
        std::fs::create_dir_all(host.parent().unwrap()).unwrap();
        std::fs::write(&host, &bytes).unwrap();

        let tool = View { sandbox: sb };
        let reply = tool.call_json(json!({ "path": "docs/img/fig.png" })).await.unwrap();
        assert_eq!(reply.0.len(), 1);
        assert!(matches!(&reply.0[0], ToolContent::Image { .. }));
    }

    #[tokio::test]
    async fn view_pdf_page_renders_png() {
        let sb = Arc::new(Sandbox::new().unwrap());
        let host = sb.host_path_for_write("docs/spec.pdf").unwrap();
        std::fs::create_dir_all(host.parent().unwrap()).unwrap();
        std::fs::write(&host, make_minimal_pdf()).unwrap();

        let tool = View { sandbox: sb };
        let reply = tool
            .call_json(json!({ "path": "docs/spec.pdf", "page": 0 }))
            .await
            .unwrap();
        assert_eq!(reply.0.len(), 1);
        match &reply.0[0] {
            ToolContent::Image {
                media_type,
                data_base64,
            } => {
                assert_eq!(media_type, "image/png");
                assert!(BASE64.decode(data_base64).unwrap().starts_with(b"\x89PNG\r\n\x1a\n"));
            }
            other @ ToolContent::Text(_) => panic!("expected image, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn view_pdf_page_out_of_range_errors() {
        let sb = Arc::new(Sandbox::new().unwrap());
        let host = sb.host_path_for_write("doc.pdf").unwrap();
        std::fs::write(&host, make_minimal_pdf()).unwrap();
        let tool = View { sandbox: sb };
        let result = tool.call_json(json!({ "path": "doc.pdf", "page": 99 })).await;
        assert!(result.is_err(), "out-of-range page must error: {result:?}");
    }

    #[tokio::test]
    async fn view_gif_passes_through() {
        // The `image` crate is built without a GIF decoder, so a GIF that fits
        // inline is passed through as-is (like WebP) — GIF is Anthropic-accepted.
        let sb = Arc::new(Sandbox::new().unwrap());
        let gif = b"GIF89a\x01\x00\x01\x00\x00\x00\x00,".to_vec();
        let enc = BASE64.encode(&gif);
        let tool = View { sandbox: sb };
        let reply = tool
            .call_json(json!({ "path": format!("data:image/gif;base64,{enc}") }))
            .await
            .unwrap();
        match &reply.0[0] {
            ToolContent::Image {
                media_type,
                data_base64,
            } => {
                assert_eq!(media_type, "image/gif");
                assert_eq!(BASE64.decode(data_base64).unwrap(), gif);
            }
            other @ ToolContent::Text(_) => panic!("expected image, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn view_unsupported_format_errors() {
        let sb = Arc::new(Sandbox::new().unwrap());
        let enc = BASE64.encode(b"definitely not an image");
        let tool = View { sandbox: sb };
        let result = tool
            .call_json(json!({ "path": format!("data:application/octet-stream;base64,{enc}") }))
            .await;
        assert!(result.is_err(), "unsupported format must error: {result:?}");
    }

    #[tokio::test]
    async fn view_missing_local_file_errors() {
        let sb = Arc::new(Sandbox::new().unwrap());
        let tool = View { sandbox: sb };
        let result = tool.call_json(json!({ "path": "nope.png" })).await;
        assert!(result.is_err(), "missing file must error: {result:?}");
    }
}
