//! Command-line interface for texmark.
//!
//! The library is pure and I/O-free (so it runs under WASM); this binary drives
//! it from a filesystem: it reads a LaTeX file, supplies a
//! [`Resolver`] rooted at the file's directory so `\input`/`\bibliography` work,
//! and writes Markdown.
//!
//! Run it with:
//!
//! ```text
//! cargo run -- paper.tex
//! cargo run -- paper.tex --output converted
//! cargo run -- paper.tex --standalone --output paper.md
//! ```
//!
//! Bundles preserve local figure files as-is. With `--standalone`, raster files
//! are inlined as base64 `data:` URIs and PDF figures are rendered to PNG.

use std::cell::RefCell;
use std::num::NonZeroU64;
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;
use std::sync::mpsc;
use std::time::Duration;

use clap::Parser;
use texmark::Backend;
use texmark::backend::md;
use texmark::engine::Engine;
use texmark::image::{self, ImageResolver};
use texmark::state::{Resolver, State};

mod arxiv;

fn info(message: impl std::fmt::Display) {
    eprintln!("info: {message}");
}

fn warn(message: impl std::fmt::Display) {
    eprintln!("warning: {message}");
}

fn err(message: impl std::fmt::Display) {
    eprintln!("error: {message}");
}

/// Convert a LaTeX project to Markdown.
#[derive(Parser)]
#[command(
    name = "texmark",
    about = "Convert LaTeX research papers to Markdown",
    version
)]
struct Cli {
    /// Local LaTeX file to convert.
    #[arg(
        value_name = "INPUT",
        required_unless_present = "arxiv",
        conflicts_with = "arxiv"
    )]
    input: Option<PathBuf>,
    /// Fetch a paper from arXiv by ID or arXiv URL.
    #[arg(long, value_name = "ID_OR_URL", required_unless_present = "input")]
    arxiv: Option<String>,
    /// Override the output directory, or the Markdown file with `--standalone`.
    #[arg(short, long, value_name = "PATH")]
    output: Option<PathBuf>,
    /// Inline images to produce one self-contained Markdown document.
    #[arg(long)]
    standalone: bool,
    /// Stop the command if it runs longer than this many seconds.
    #[arg(long, default_value_t = NonZeroU64::new(120).unwrap(), value_name = "SECONDS")]
    timeout: NonZeroU64,
    /// Omit the YAML frontmatter block (source, title, version, diagnostics)
    /// from Markdown output.
    #[arg(long)]
    no_frontmatter: bool,
    /// Also emit each figure caption as a visible line (default: only as the
    /// image's alt text, avoiding redundancy — best for LLM consumption).
    #[arg(long)]
    duplicate_captions: bool,
}

enum Input {
    File(PathBuf),
    Arxiv(String),
}

enum Destination {
    Directory(PathBuf),
    Standalone(PathBuf),
}

enum DefaultOutput {
    Path(PathBuf),
    Arxiv(String),
}

impl DefaultOutput {
    fn path(&self, title: &str) -> PathBuf {
        match self {
            Self::Path(path) => path.clone(),
            Self::Arxiv(id) => PathBuf::from(paper_name(title).unwrap_or_else(|| id.clone())),
        }
    }
}

impl Destination {
    fn new(default: &Path, options: &Options) -> Self {
        if options.standalone {
            Self::Standalone(
                options
                    .output
                    .clone()
                    .unwrap_or_else(|| default.with_extension("md")),
            )
        } else {
            Self::Directory(
                options
                    .output
                    .clone()
                    .unwrap_or_else(|| default.to_path_buf()),
            )
        }
    }

    fn markdown(&self) -> PathBuf {
        match self {
            Self::Directory(path) => path.join("main.md"),
            Self::Standalone(path) => path.clone(),
        }
    }
}

struct Request {
    input: Input,
    options: Options,
    timeout: Duration,
}

struct Options {
    output: Option<PathBuf>,
    standalone: bool,
    no_frontmatter: bool,
    duplicate_captions: bool,
}

impl Cli {
    fn into_request(self) -> Request {
        let input = match (self.input, self.arxiv) {
            (Some(path), None) => Input::File(path),
            (None, Some(id)) => Input::Arxiv(id),
            _ => unreachable!("clap requires exactly one input"),
        };
        Request {
            input,
            options: Options {
                output: self.output,
                standalone: self.standalone,
                no_frontmatter: self.no_frontmatter,
                duplicate_captions: self.duplicate_captions,
            },
            timeout: Duration::from_secs(self.timeout.get()),
        }
    }
}

pub fn run() -> ExitCode {
    let request = Cli::parse().into_request();
    let timeout = request.timeout;
    run_with_timeout(timeout, move || execute(request))
}

fn execute(request: Request) -> ExitCode {
    match request {
        Request {
            input: Input::File(input),
            options,
            ..
        } => convert_file(input, options),
        Request {
            input: Input::Arxiv(id),
            options,
            ..
        } => convert_arxiv(&id, options),
    }
}

fn run_with_timeout(
    timeout: Duration,
    task: impl FnOnce() -> ExitCode + Send + 'static,
) -> ExitCode {
    let (sender, receiver) = mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let _ = sender.send(task());
    });
    match receiver.recv_timeout(timeout) {
        Ok(exit_code) => exit_code,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            err(format_args!(
                "conversion timed out after {} seconds",
                timeout.as_secs()
            ));
            ExitCode::from(124)
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            err("conversion failed unexpectedly");
            ExitCode::FAILURE
        }
    }
}

fn convert_file(input: PathBuf, options: Options) -> ExitCode {
    let source = match std::fs::read_to_string(&input) {
        Ok(s) => s,
        Err(e) => {
            err(format_args!("cannot read {}: {e}", input.display()));
            return ExitCode::FAILURE;
        }
    };

    let base = input.parent().map(Path::to_path_buf).unwrap_or_default();
    let job = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "texmark".to_string());
    let default_output = input.with_extension("");

    finish(
        &source,
        &FsResolver {
            base: base.clone(),
            job: job.clone(),
        },
        &FsFigureSource { base },
        &input.display().to_string(),
        DefaultOutput::Path(default_output),
        options,
    )
}

fn convert_arxiv(id: &str, options: Options) -> ExitCode {
    let project = match arxiv::Project::fetch(id) {
        Ok(project) => project,
        Err(error) => {
            err(error);
            return ExitCode::FAILURE;
        }
    };
    let source_name = project.source_name();
    let default_output = project.id().replace('/', "-");
    finish(
        project.main_source(),
        &project,
        &project,
        &source_name,
        DefaultOutput::Arxiv(default_output),
        options,
    )
}

fn finish(
    source: &str,
    resolver: &dyn Resolver,
    figure_source: &dyn FigureSource,
    source_name: &str,
    default_output: DefaultOutput,
    options: Options,
) -> ExitCode {
    let mut state = State::new();
    let (mut tree, diagnostics) = Engine::new(source, &mut state, resolver).parse();
    let title = md::title_text(&tree).unwrap_or_default();
    let default_output = default_output.path(&title);
    let destination = Destination::new(&default_output, &options);

    let search_dirs = image::graphics_search_dirs(&state);
    match &destination {
        Destination::Standalone(_) => {
            let resolver = StandaloneImageResolver(figure_source);
            image::resolve_images(&mut tree, &resolver, &search_dirs);
            warn_unresolved_images(&tree, None);
        }
        Destination::Directory(path) => {
            let bundled = DirectoryImageResolver::new(figure_source, path);
            image::resolve_images(&mut tree, &bundled, &search_dirs);
            if let Some(error) = bundled.error() {
                err(error);
                return ExitCode::FAILURE;
            }
            warn_unresolved_images(&tree, Some(path));
        }
    }
    let body = md::Markdown {
        opts: md::MdOptions {
            duplicate_captions: options.duplicate_captions,
        },
    }
    .serialize(&tree);
    let output = if options.no_frontmatter {
        body
    } else {
        let authors = md::author_texts(&tree);
        let frontmatter = md::frontmatter_with_authors(
            &[
                ("source", source_name),
                ("title", &title),
                ("texmark_version", texmark::VERSION),
                (
                    "texmark_truncated",
                    if diagnostics.is_incomplete() {
                        "true"
                    } else {
                        "false"
                    },
                ),
            ],
            &authors,
        );
        format!("{frontmatter}{body}")
    };

    let markdown = destination.markdown();
    if let Err(error) = write_file(&markdown, output.as_bytes()) {
        err(error);
        return ExitCode::FAILURE;
    }

    // Never fail silently: warn about any degradation. Dropped commands are a
    // fidelity loss but the output is still usable, so they only warn; a reached
    // expansion limit means the output is incomplete, so it also exits non-zero.
    if !diagnostics.dropped_commands.is_empty() {
        let list = diagnostics
            .dropped_commands
            .iter()
            .map(|c| format!("\\{c}"))
            .collect::<Vec<_>>()
            .join(", ");
        warn(format_args!(
            "{} unrecognized command(s) dropped from the output: {}",
            diagnostics.dropped_commands.len(),
            list
        ));
    }
    if !diagnostics.unresolved_inputs.is_empty() {
        let list = diagnostics
            .unresolved_inputs
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        warn(format_args!(
            "{} \\input/\\include file(s) could not be resolved — their \
             content is MISSING from the output: {}",
            diagnostics.unresolved_inputs.len(),
            list
        ));
    }
    if diagnostics.expansion_limit_exceeded {
        warn(
            "macro-expansion limit reached — output is INCOMPLETE. \
             The document uses unbounded recursion or package code texmark cannot \
             evaluate; the result above is partial.",
        );
    }
    // Any content loss (a reached expansion limit, or an unresolvable content
    // include) means the output is incomplete — exit non-zero so a pipeline
    // gating on status treats it as such, consistently for both causes.
    if diagnostics.is_incomplete() {
        return ExitCode::from(3);
    }
    ExitCode::SUCCESS
}

/// Resolves `\input`/`\include` names against a base directory, trying the name
/// verbatim and with a `.tex` extension.
struct FsResolver {
    base: PathBuf,
    /// The top-level job name (main file stem), used to locate `<job>.bbl`.
    job: String,
}

impl Resolver for FsResolver {
    fn resolve(&self, name: &str) -> Option<String> {
        let candidates = [self.base.join(name), self.base.join(format!("{name}.tex"))];
        candidates
            .iter()
            .find_map(|p| std::fs::read_to_string(p).ok())
    }

    fn resolve_bibliography(&self) -> Option<String> {
        // BibTeX writes `<job>.bbl`; try that first, then fall back to any
        // `.bbl` sitting in the base directory.
        let by_job = self.base.join(format!("{}.bbl", self.job));
        if let Ok(src) = std::fs::read_to_string(&by_job) {
            return Some(src);
        }
        let mut bbls: Vec<PathBuf> = std::fs::read_dir(&self.base)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "bbl"))
            .collect();
        bbls.sort();
        bbls.first().and_then(|p| std::fs::read_to_string(p).ok())
    }
}

/// Resolves image references to portable Markdown targets: URLs pass through,
struct Figure {
    name: String,
    mime: &'static str,
    bytes: Vec<u8>,
}

trait FigureSource {
    fn read(&self, name: &str) -> Option<Vec<u8>>;

    fn find(&self, source: &str) -> Option<Figure> {
        for name in image::raster_candidates(source) {
            let Some(bytes) = self.read(&name) else {
                continue;
            };
            if let Some(mime) = image::sniff_mime(&bytes) {
                return Some(Figure { name, mime, bytes });
            }
        }
        if let Some(name) = image::pdf_candidate(source)
            && let Some(bytes) = self.read(&name)
            && bytes.starts_with(b"%PDF-")
        {
            return Some(Figure {
                name,
                mime: "application/pdf",
                bytes,
            });
        }
        let name = image::eps_candidate(source)?;
        eps_figure(name.clone(), self.read(&name)?)
    }
}

struct FsFigureSource {
    base: PathBuf,
}

impl FigureSource for FsFigureSource {
    fn read(&self, name: &str) -> Option<Vec<u8>> {
        std::fs::read(self.base.join(name)).ok()
    }
}

fn eps_figure(name: String, bytes: Vec<u8>) -> Option<Figure> {
    if !bytes.starts_with(b"%!") {
        return None;
    }
    let mut interpreter = stet::Interpreter::builder().suppress_output().build();
    let page = match interpreter.render(&bytes, 144.0) {
        Ok(mut pages) => pages.drain(..).next()?,
        Err(error) => {
            warn(format_args!("cannot convert EPS figure {name}: {error}"));
            return None;
        }
    };
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, page.width, page.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .ok()?
            .write_image_data(&page.rgba)
            .ok()?;
    }
    let name = Path::new(&name)
        .with_extension("png")
        .to_string_lossy()
        .into_owned();
    Some(Figure {
        name,
        mime: "image/png",
        bytes,
    })
}

struct StandaloneImageResolver<'a>(&'a dyn FigureSource);

impl ImageResolver for StandaloneImageResolver<'_> {
    fn resolve(&self, source: &str) -> Option<String> {
        if source.starts_with("http://") || source.starts_with("https://") {
            return Some(source.to_string());
        }
        let figure = self.0.find(source)?;
        if figure.mime == "application/pdf" {
            let png = image::rasterize_pdf(figure.bytes)?;
            Some(image::data_uri("image/png", &png))
        } else {
            Some(image::data_uri(figure.mime, &figure.bytes))
        }
    }
}

struct DirectoryImageResolver<'a> {
    source: &'a dyn FigureSource,
    root: &'a Path,
    error: RefCell<Option<String>>,
}

impl<'a> DirectoryImageResolver<'a> {
    fn new(source: &'a dyn FigureSource, root: &'a Path) -> Self {
        Self {
            source,
            root,
            error: RefCell::new(None),
        }
    }

    fn error(&self) -> Option<String> {
        self.error.borrow().clone()
    }

    fn fail(&self, error: String) {
        let mut current = self.error.borrow_mut();
        if current.is_none() {
            *current = Some(error);
        }
    }
}

impl ImageResolver for DirectoryImageResolver<'_> {
    fn resolve(&self, source: &str) -> Option<String> {
        if source.starts_with("http://") || source.starts_with("https://") {
            return Some(source.to_string());
        }
        let figure = self.source.find(source)?;
        let relative = figure_path(&figure.name, figure.mime);
        let destination = self.root.join("figures").join(&relative);
        if let Err(error) = write_file(&destination, &figure.bytes) {
            self.fail(error);
            return None;
        }
        Some(format!("figures/{}", slash_path(&relative)))
    }
}

fn write_file(path: &Path, contents: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    std::fs::write(path, contents)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    info(format_args!("wrote {}", path.display()));
    Ok(())
}

fn figure_path(source: &str, mime: &str) -> PathBuf {
    let mut parts = Path::new(source)
        .components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_os_string()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let has_figure_prefix = parts.first().is_some_and(|part| {
        let part = part.to_string_lossy();
        part.eq_ignore_ascii_case("figs") || part.eq_ignore_ascii_case("figures")
    });
    if parts.len() > 1 && has_figure_prefix {
        parts.remove(0);
    }
    let mut path = parts.into_iter().collect::<PathBuf>();
    if path.as_os_str().is_empty() {
        path.push("figure");
    }
    let extension = match mime {
        "image/jpeg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        "image/svg+xml" => "svg",
        "image/bmp" => "bmp",
        "application/pdf" => "pdf",
        _ => "png",
    };
    let current = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let matches = match mime {
        "image/jpeg" => matches!(current.as_str(), "jpg" | "jpeg"),
        "image/gif" => current == "gif",
        "image/webp" => current == "webp",
        "image/svg+xml" => current == "svg",
        "image/bmp" => current == "bmp",
        "application/pdf" => current == "pdf",
        _ => current == "png",
    };
    if current.is_empty() {
        path.set_extension(extension);
    } else if !matches {
        let name = path
            .file_name()
            .map(|value| value.to_string_lossy())
            .unwrap_or_default();
        path.set_file_name(format!("{name}.{extension}"));
    }
    path
}

fn slash_path(path: &Path) -> String {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_string_lossy()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn paper_name(title: &str) -> Option<String> {
    let mut name = String::new();
    let mut separator = false;
    for character in title.chars() {
        if character.is_alphanumeric() {
            if separator && !name.is_empty() {
                name.push('-');
            }
            separator = false;
            for lowercase in character.to_lowercase() {
                name.push(lowercase);
            }
        } else {
            separator = true;
        }
        if name.chars().count() >= 100 {
            break;
        }
    }
    while name.ends_with('-') {
        name.pop();
    }
    (!name.is_empty()).then_some(name)
}

/// Warn about images that could not be made portable.
fn warn_unresolved_images(tree: &texmark::node::Element, output_root: Option<&Path>) {
    fn walk(e: &texmark::node::Element, output_root: Option<&Path>, unresolved: &mut Vec<String>) {
        for child in &e.children {
            if let texmark::node::Node::Element(el) = child {
                if el.name == "image"
                    && let Some((_, src)) = el.attributes.iter().find(|(k, _)| k == "src")
                    && !src.starts_with("data:")
                    && !is_remote_image(src)
                    && !output_root.is_some_and(|root| root.join(src).is_file())
                {
                    unresolved.push(src.clone());
                }
                walk(el, output_root, unresolved);
            }
        }
    }
    let mut unresolved = Vec::new();
    walk(tree, output_root, &mut unresolved);
    unresolved.sort();
    unresolved.dedup();
    if !unresolved.is_empty() {
        warn(format_args!(
            "{} image(s) left as non-portable references (no raster file found): {}",
            unresolved.len(),
            unresolved.join(", ")
        ));
    }
}

fn is_remote_image(source: &str) -> bool {
    source.starts_with("http://") || source.starts_with("https://")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_input_uses_its_stem_as_the_output_name() {
        let request = Cli::try_parse_from(["texmark", "paper.tex"])
            .unwrap()
            .into_request();
        assert!(matches!(request.input, Input::File(path) if path == Path::new("paper.tex")));
        assert_eq!(
            Destination::new(Path::new("paper"), &request.options).markdown(),
            Path::new("paper/main.md")
        );
    }

    #[test]
    fn output_flag_accepts_short_and_long_forms() {
        for flag in ["-o", "--output"] {
            let request = Cli::try_parse_from(["texmark", "paper.tex", flag, "paper.md"])
                .unwrap()
                .into_request();
            assert_eq!(
                request.options.output.as_deref(),
                Some(Path::new("paper.md"))
            );
        }
    }

    #[test]
    fn arxiv_accepts_an_output_flag() {
        let request = Cli::try_parse_from(["texmark", "--arxiv", "2205.14135v2", "-o", "paper.md"])
            .unwrap()
            .into_request();
        assert!(matches!(request.input, Input::Arxiv(id) if id == "2205.14135v2"));
        assert_eq!(
            request.options.output.as_deref(),
            Some(Path::new("paper.md"))
        );
    }

    #[test]
    fn standalone_is_opt_in() {
        let default = Cli::try_parse_from(["texmark", "paper.tex"])
            .unwrap()
            .into_request();
        let standalone = Cli::try_parse_from(["texmark", "paper.tex", "--standalone"])
            .unwrap()
            .into_request();
        assert!(!default.options.standalone);
        assert!(standalone.options.standalone);
        assert_eq!(
            Destination::new(Path::new("paper"), &standalone.options).markdown(),
            Path::new("paper.md")
        );
    }

    #[test]
    fn timeout_defaults_to_two_minutes_and_can_be_overridden() {
        let default = Cli::try_parse_from(["texmark", "paper.tex"])
            .unwrap()
            .into_request();
        let custom = Cli::try_parse_from(["texmark", "paper.tex", "--timeout", "30"])
            .unwrap()
            .into_request();
        assert_eq!(default.timeout, Duration::from_secs(120));
        assert_eq!(custom.timeout, Duration::from_secs(30));
        assert!(Cli::try_parse_from(["texmark", "paper.tex", "--timeout", "0"]).is_err());
    }

    #[test]
    fn timeout_returns_the_conventional_exit_code() {
        let exit_code = run_with_timeout(Duration::from_millis(1), || {
            std::thread::sleep(Duration::from_millis(20));
            ExitCode::SUCCESS
        });
        assert_eq!(exit_code, ExitCode::from(124));
    }

    #[test]
    fn output_overrides_the_directory_or_markdown_name() {
        let directory = Cli::try_parse_from(["texmark", "paper.tex", "-o", "converted"])
            .unwrap()
            .into_request();
        let standalone =
            Cli::try_parse_from(["texmark", "paper.tex", "--standalone", "-o", "converted.md"])
                .unwrap()
                .into_request();
        assert_eq!(
            Destination::new(Path::new("paper"), &directory.options).markdown(),
            Path::new("converted/main.md")
        );
        assert_eq!(
            Destination::new(Path::new("paper"), &standalone.options).markdown(),
            Path::new("converted.md")
        );
    }

    #[test]
    fn bundled_figure_names_are_safe_and_keep_converted_extensions() {
        assert_eq!(
            figure_path("../plots/loss.pdf", "application/pdf"),
            Path::new("plots/loss.pdf")
        );
        assert_eq!(
            figure_path("plots/loss.jpeg", "image/jpeg"),
            Path::new("plots/loss.jpeg")
        );
        assert_eq!(
            figure_path("figs/loss.png", "image/png"),
            Path::new("loss.png")
        );
        assert_eq!(
            figure_path("Figures/results/loss.png", "image/png"),
            Path::new("results/loss.png")
        );
    }

    #[test]
    fn only_http_urls_are_remote_images() {
        assert!(is_remote_image("https://example.com/plot.pdf"));
        assert!(is_remote_image("http://example.com/plot.pdf"));
        assert!(!is_remote_image("http_plot.pdf"));
    }

    #[test]
    fn eps_figures_are_converted_to_png() {
        let eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\nnewpath 0 0 moveto 10 10 lineto stroke\nshowpage\n";
        let figure = eps_figure("figure.eps".to_string(), eps.to_vec()).unwrap();
        assert_eq!(figure.name, "figure.png");
        assert_eq!(figure.mime, "image/png");
        assert!(figure.bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
    }

    #[test]
    fn arxiv_output_uses_the_paper_name() {
        let options = Options {
            output: None,
            standalone: false,
            no_frontmatter: false,
            duplicate_captions: false,
        };
        let default =
            DefaultOutput::Arxiv("1706.03762v7".to_string()).path("Attention Is All You Need");
        assert_eq!(
            Destination::new(&default, &options).markdown(),
            Path::new("attention-is-all-you-need/main.md")
        );
    }

    #[test]
    fn arxiv_output_falls_back_to_the_id_without_a_title() {
        assert_eq!(
            DefaultOutput::Arxiv("1706.03762v7".to_string()).path(""),
            Path::new("1706.03762v7")
        );
    }
}
