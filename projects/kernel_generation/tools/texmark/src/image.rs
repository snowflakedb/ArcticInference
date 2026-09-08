//! Making images portable.
//!
//! Portable Markdown may only reference images as a hosted URL or an inlined
//! `data:` URI (no dependence on a local asset directory). The engine emits
//! `<image src="...">` with the original LaTeX path; [`resolve_images`] rewrites
//! each `src` to a portable form using a caller-supplied [`ImageResolver`].
//!
//! The MIME/data-URI helpers here are WASM-safe (base64 via the `base64`
//! crate); the actual file reading lives in whatever `ImageResolver` the
//! caller provides (the CLI reads from disk).

use crate::node::{Element, Node};

/// Turns an original image reference into a portable target (a URL or a data
/// URI), or returns `None` to leave it unchanged.
pub trait ImageResolver {
    fn resolve(&self, src: &str) -> Option<String>;
}

/// Rewrite every `<image>`'s `src` attribute in the tree to the portable form
/// returned by `resolver`. References the resolver leaves as `None` are kept.
///
/// `search_dirs` are the `\graphicspath` directories (see [`graphics_search_dirs`]):
/// if the literal `src` does not resolve, each is prepended and the combined path
/// is `..`/`.`-normalized before retrying — so a figure named `../Figures/fig.png`
/// under `\graphicspath{{Figures/}}` resolves to `Figures/fig.png`, matching how
/// LaTeX searches. Pass `&[]` for none.
pub fn resolve_images(root: &mut Element, resolver: &dyn ImageResolver, search_dirs: &[String]) {
    for child in &mut root.children {
        if let Node::Element(e) = child {
            if e.name == "image" {
                if let Some((_, src)) = e.attributes.iter_mut().find(|(k, _)| k == "src") {
                    // Literal path first, then each `\graphicspath` search dir.
                    let target = resolver.resolve(src).or_else(|| {
                        search_dirs
                            .iter()
                            .find_map(|d| resolver.resolve(&normalize(&format!("{d}{src}"))))
                    });
                    if let Some(target) = target {
                        *src = target;
                    }
                }
            } else {
                resolve_images(e, resolver, search_dirs);
            }
        }
    }
}

/// The `\graphicspath` search directories a parse recorded, for passing to
/// [`resolve_images`]. Kept out of the document tree (it is resolver metadata,
/// not content, so the neutral XML tree stays clean).
pub fn graphics_search_dirs(state: &crate::state::State) -> Vec<String> {
    state
        .package_state
        .get("graphicx:path")
        .map(|v| {
            v.split(',')
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Lexically collapse `.` and `..` segments in a `/`-path (no I/O), so an
/// archive/map lookup matches even when the OS would otherwise resolve `..`.
/// `Figures/../Figures/fig.png` → `Figures/fig.png`; a leading `..` that cannot
/// be collapsed is preserved.
fn normalize(path: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." if matches!(out.last(), Some(&s) if s != "..") => {
                out.pop();
            }
            seg => out.push(seg),
        }
    }
    out.join("/")
}

/// Build a `data:<mime>;base64,<data>` URI.
pub fn data_uri(mime: &str, bytes: &[u8]) -> String {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    format!("data:{mime};base64,{b64}")
}

/// Rasterize the first page of a PDF to PNG bytes, or `None` if it cannot be
/// parsed/rendered. Pure-Rust and WASM-capable via `hayro` when the `pdf` feature
/// is enabled; otherwise it returns `None`. Takes bytes (not a path) to stay
/// I/O-free. Rendered at 300 DPI so text and lines in vector figures stay crisp.
pub fn rasterize_pdf(data: Vec<u8>) -> Option<Vec<u8>> {
    rasterize_pdf_impl(data)
}

#[cfg(feature = "pdf")]
fn rasterize_pdf_impl(data: Vec<u8>) -> Option<Vec<u8>> {
    use hayro::hayro_interpret::InterpreterSettings;
    use hayro::hayro_syntax::Pdf;
    use hayro::{RenderCache, RenderSettings, render};

    // PDF user space is 72 units/inch; scale = target DPI / 72.
    const SCALE: f32 = 300.0 / 72.0;
    let pdf = Pdf::new(data).ok()?;
    let page = pdf.pages().first()?;
    let settings = RenderSettings {
        x_scale: SCALE,
        y_scale: SCALE,
        ..Default::default()
    };
    let pixmap = render(
        page,
        &RenderCache::new(),
        &InterpreterSettings::default(),
        &settings,
    );
    pixmap.into_png().ok()
}

#[cfg(not(feature = "pdf"))]
fn rasterize_pdf_impl(_data: Vec<u8>) -> Option<Vec<u8>> {
    None
}

/// Detect an image MIME type from the leading bytes of its content.
///
/// More reliable than extension-based guessing for files where the extension
/// is absent or wrong — a common LaTeX pattern: `\includegraphics{fig}` with
/// no extension at all.
pub fn sniff_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some("image/png");
    }
    if bytes.starts_with(b"\xff\xd8") {
        return Some("image/jpeg");
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some("image/gif");
    }
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        return Some("image/webp");
    }
    if bytes.starts_with(b"BM") {
        return Some("image/bmp");
    }
    // SVG: UTF-8 text starting with `<svg` (optionally preceded by BOM)
    let text = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes);
    if text.starts_with(b"<svg") || text.starts_with(b"<?xml") {
        return Some("image/svg+xml");
    }
    None
}

/// Guess an image MIME type from a file path's extension. Returns `None` for
/// formats that cannot be embedded as a raster image (e.g. PDF, EPS).
pub fn guess_mime(path: &str) -> Option<&'static str> {
    let ext = path.rsplit('.').next()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "svg" => "image/svg+xml",
        "bmp" => "image/bmp",
        _ => return None,
    })
}

/// Common raster extensions, in preference order, that a bare
/// `\includegraphics{fig}` (no extension) may resolve to.
const RASTER_EXTS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp", "svg"];

/// Candidate raster filenames for an `\includegraphics` src, in order: the
/// literal path when it already names a known raster type, then the same stem
/// with each common raster extension. LaTeX omits the extension, so a bare
/// `figs/fig` must also be tried as `figs/fig.png`, `figs/fig.jpg`, …. A
/// front-end tries each against its own storage (filesystem, archive) and takes
/// the first that exists. Pure (no I/O), so it lives in the WASM-safe library.
pub fn raster_candidates(src: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(RASTER_EXTS.len() + 1);
    if guess_mime(src).is_some() {
        out.push(src.to_string());
    }
    let stem = std::path::Path::new(src).with_extension("");
    for ext in RASTER_EXTS {
        let name = stem.with_extension(ext).to_string_lossy().into_owned();
        if !out.contains(&name) {
            out.push(name);
        }
    }
    out
}

/// The PDF file to rasterize for an `\includegraphics` src: the literal path if
/// it is a `.pdf`, else `<src>.pdf` for an extension-less include (LaTeX would
/// append the extension). `None` if the src already names a non-PDF extension.
pub fn pdf_candidate(src: &str) -> Option<String> {
    if is_pdf(src) {
        Some(src.to_string())
    } else if std::path::Path::new(src).extension().is_none() {
        Some(format!("{src}.pdf"))
    } else {
        None
    }
}

pub fn eps_candidate(src: &str) -> Option<String> {
    if is_eps(src) {
        Some(src.to_string())
    } else if std::path::Path::new(src).extension().is_none() {
        Some(format!("{src}.eps"))
    } else {
        None
    }
}

/// Whether `src` names a PDF figure. Uses the path's file-name extension, so a
/// dotted *directory* (e.g. `v1.0/fig`) is correctly seen as extension-less.
pub fn is_pdf(src: &str) -> bool {
    std::path::Path::new(src)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("pdf"))
}

pub fn is_eps(src: &str) -> bool {
    std::path::Path::new(src)
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("eps"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_resolution() {
        // extension-less include → raster stems + a .pdf fallback
        assert_eq!(
            raster_candidates("figs/fig"),
            [
                "figs/fig.png",
                "figs/fig.jpg",
                "figs/fig.jpeg",
                "figs/fig.gif",
                "figs/fig.webp",
                "figs/fig.svg"
            ]
        );
        assert_eq!(pdf_candidate("figs/fig"), Some("figs/fig.pdf".to_string()));
        // explicit .pdf → literal pdf, no raster literal
        assert_eq!(
            pdf_candidate("figs/fig.pdf"),
            Some("figs/fig.pdf".to_string())
        );
        // explicit raster → literal first, and not treated as pdf
        assert_eq!(raster_candidates("a.png")[0], "a.png");
        assert_eq!(pdf_candidate("a.png"), None);
        // dotted directory, no file extension → still a pdf candidate, not is_pdf
        assert!(!is_pdf("v1.0/fig"));
        assert_eq!(pdf_candidate("v1.0/fig"), Some("v1.0/fig.pdf".to_string()));
        assert_eq!(eps_candidate("v1.0/fig"), Some("v1.0/fig.eps".to_string()));
        assert_eq!(eps_candidate("figure.eps"), Some("figure.eps".to_string()));
    }

    #[cfg(feature = "pdf")]
    #[test]
    fn rasterize_pdf_rejects_non_pdf_gracefully() {
        // Not a PDF → None, never a panic.
        assert!(rasterize_pdf(b"not a pdf at all".to_vec()).is_none());
        assert!(rasterize_pdf(Vec::new()).is_none());
    }

    #[test]
    fn mime_by_extension() {
        assert_eq!(guess_mime("a/b.png"), Some("image/png"));
        assert_eq!(guess_mime("x.JPG"), Some("image/jpeg"));
        assert_eq!(guess_mime("fig.pdf"), None);
    }

    #[test]
    fn sniff_mime_magic_bytes() {
        assert_eq!(sniff_mime(b"\x89PNG\r\n\x1a\nrest"), Some("image/png"));
        assert_eq!(sniff_mime(b"\xff\xd8\xff\xe0rest"), Some("image/jpeg"));
        assert_eq!(sniff_mime(b"GIF89a rest"), Some("image/gif"));
        assert_eq!(sniff_mime(b"GIF87a rest"), Some("image/gif"));
        assert_eq!(
            sniff_mime(b"RIFF\x00\x00\x00\x00WEBPrest"),
            Some("image/webp")
        );
        assert_eq!(sniff_mime(b"BMrest"), Some("image/bmp"));
        assert_eq!(sniff_mime(b"<svg xmlns="), Some("image/svg+xml"));
        assert_eq!(sniff_mime(b"<?xml version"), Some("image/svg+xml"));
        assert_eq!(sniff_mime(b"%PDF-1.4"), None);
        assert_eq!(sniff_mime(b""), None);
    }

    struct Fixed;
    impl ImageResolver for Fixed {
        fn resolve(&self, src: &str) -> Option<String> {
            (src == "keep.png").then(|| "data:image/png;base64,AAAA".to_string())
        }
    }

    #[test]
    fn rewrites_only_resolved_images() {
        let mut root = Element::new("document");
        root.push(Node::Element(Element::new("image").attr("src", "keep.png")));
        root.push(Node::Element(
            Element::new("image").attr("src", "figs/x.pdf"),
        ));
        resolve_images(&mut root, &Fixed, &[]);
        let srcs: Vec<&str> = root
            .children
            .iter()
            .filter_map(|n| match n {
                Node::Element(e) => e
                    .attributes
                    .iter()
                    .find(|(k, _)| k == "src")
                    .map(|(_, v)| v.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(srcs, vec!["data:image/png;base64,AAAA", "figs/x.pdf"]);
    }

    #[test]
    fn graphicspath_dir_resolves_parent_relative_figure() {
        // The resolver only "has" `Figures/foo.png`. An include of
        // `../Figures/foo.png` under `\graphicspath{{Figures/}}` must resolve:
        // `Figures/` + `../Figures/foo.png` normalizes to `Figures/foo.png`.
        struct OnlyFigures;
        impl ImageResolver for OnlyFigures {
            fn resolve(&self, src: &str) -> Option<String> {
                (src == "Figures/foo.png").then(|| "data:image/png;base64,Zm9v".to_string())
            }
        }
        let mut root = Element::new("document");
        root.push(Node::Element(
            Element::new("image").attr("src", "../Figures/foo.png"),
        ));
        resolve_images(&mut root, &OnlyFigures, &["Figures/".to_string()]);
        let src = match &root.children[0] {
            Node::Element(e) => e
                .attributes
                .iter()
                .find(|(k, _)| k == "src")
                .unwrap()
                .1
                .clone(),
            _ => unreachable!(),
        };
        assert_eq!(src, "data:image/png;base64,Zm9v");
    }
}
