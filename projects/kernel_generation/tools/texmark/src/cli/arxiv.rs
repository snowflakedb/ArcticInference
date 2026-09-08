use std::collections::{BTreeMap, HashSet};
use std::io::Read;
use std::path::{Component, Path};
use std::time::Duration;

use texmark::state::Resolver;

use super::FigureSource;

const FETCH_TIMEOUT: Duration = Duration::from_secs(200);

type Files = BTreeMap<String, Vec<u8>>;

pub struct Project {
    id: String,
    main_name: String,
    main_source: String,
    files: Files,
}

impl Project {
    pub fn fetch(input: &str) -> Result<Self, String> {
        let id = normalize_id(input)?;
        let client = reqwest::blocking::Client::builder()
            .timeout(FETCH_TIMEOUT)
            .user_agent("texmark/0.1")
            .build()
            .map_err(|error| format!("cannot create arXiv client: {error}"))?;
        let url = format!("https://export.arxiv.org/e-print/{id}");
        let response = client
            .get(&url)
            .send()
            .map_err(|error| format!("cannot fetch arXiv paper {id}: {error}"))?
            .error_for_status()
            .map_err(|error| format!("arXiv rejected {id}: {error}"))?;
        let archive = response
            .bytes()
            .map_err(|error| format!("cannot read arXiv paper {id}: {error}"))?;
        Self::from_archive(id, &archive)
    }

    fn from_archive(id: String, archive: &[u8]) -> Result<Self, String> {
        let files = unpack(archive);
        let (main_name, main_source) =
            find_main_tex(&files).ok_or_else(|| format!("arXiv paper {id} has no TeX source"))?;
        Ok(Self {
            id,
            main_name,
            main_source,
            files,
        })
    }

    pub fn main_source(&self) -> &str {
        &self.main_source
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn source_name(&self) -> String {
        format!("https://arxiv.org/abs/{}", self.id)
    }

    fn main_dir(&self) -> &str {
        self.main_name.rsplit_once('/').map_or("", |(dir, _)| dir)
    }

    fn candidates(&self, name: &str, extension: Option<&str>) -> Vec<String> {
        let mut names = vec![name.to_string()];
        if let Some(extension) = extension
            && Path::new(name).extension().is_none()
        {
            names.push(format!("{name}.{extension}"));
        }
        let mut candidates = Vec::new();
        for name in names {
            if let Some(path) = join_archive_path(self.main_dir(), &name) {
                candidates.push(path);
            }
            if let Some(path) = join_archive_path("", &name)
                && !candidates.contains(&path)
            {
                candidates.push(path);
            }
        }
        candidates
    }

    fn bytes(&self, name: &str, extension: Option<&str>) -> Option<&[u8]> {
        self.candidates(name, extension)
            .into_iter()
            .find_map(|candidate| self.files.get(&candidate).map(Vec::as_slice))
    }
}

impl Resolver for Project {
    fn resolve(&self, name: &str) -> Option<String> {
        self.bytes(name, Some("tex"))
            .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
    }

    fn resolve_bibliography(&self) -> Option<String> {
        let job = Path::new(&self.main_name)
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())?;
        if let Some(bytes) = self.bytes(&job, Some("bbl")) {
            return Some(String::from_utf8_lossy(bytes).into_owned());
        }
        self.files
            .iter()
            .find(|(name, _)| name.ends_with(".bbl"))
            .map(|(_, bytes)| String::from_utf8_lossy(bytes).into_owned())
    }
}

impl FigureSource for Project {
    fn read(&self, name: &str) -> Option<Vec<u8>> {
        self.bytes(name, None).map(<[u8]>::to_vec)
    }
}

fn normalize_id(input: &str) -> Result<String, String> {
    let mut value = input.trim();
    if let Some(rest) = value
        .strip_prefix("https://")
        .or_else(|| value.strip_prefix("http://"))
    {
        let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
        if !matches!(host, "arxiv.org" | "www.arxiv.org" | "export.arxiv.org") {
            return Err(format!("not an arXiv URL: {input}"));
        }
        value = path;
    }
    value = value.split(['?', '#']).next().unwrap_or(value);
    for prefix in ["abs/", "pdf/", "src/", "e-print/"] {
        if let Some(rest) = value.strip_prefix(prefix) {
            value = rest;
            break;
        }
    }
    value = value
        .strip_suffix(".pdf")
        .unwrap_or(value)
        .trim_matches('/');
    if value.is_empty()
        || value.len() > 100
        || value.contains("..")
        || !value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || ".-_/".contains(character))
    {
        return Err(format!("invalid arXiv id: {input}"));
    }
    Ok(value.to_string())
}

fn unpack(archive: &[u8]) -> Files {
    let mut decompressed = Vec::new();
    let blob = match flate2::read::GzDecoder::new(archive).read_to_end(&mut decompressed) {
        Ok(_) => decompressed,
        Err(_) => archive.to_vec(),
    };
    let mut files = Files::new();
    let mut tar = tar::Archive::new(&blob[..]);
    if let Ok(entries) = tar.entries() {
        for mut entry in entries.flatten() {
            if !entry.header().entry_type().is_file() {
                continue;
            }
            let Some(name) = entry.path().ok().and_then(|path| clean_archive_path(&path)) else {
                continue;
            };
            let mut bytes = Vec::new();
            if entry.read_to_end(&mut bytes).is_ok() && !bytes.is_empty() {
                files.insert(name, bytes);
            }
        }
    }
    if files.is_empty() {
        files.insert("main.tex".to_string(), blob);
    }
    files
}

fn clean_archive_path(path: &Path) -> Option<String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    (!parts.is_empty()).then(|| parts.join("/"))
}

fn join_archive_path(base: &str, path: &str) -> Option<String> {
    let mut parts: Vec<String> = base
        .split('/')
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect();
    for component in Path::new(path).components() {
        match component {
            Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            Component::CurDir => {}
            Component::ParentDir => {
                parts.pop()?;
            }
            Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    (!parts.is_empty()).then(|| parts.join("/"))
}

#[derive(Default)]
struct Readme {
    toplevel: Vec<String>,
    ignore: HashSet<String>,
}

fn basename_lower(name: &str) -> String {
    name.rsplit('/').next().unwrap_or(name).to_ascii_lowercase()
}

fn parse_readme(files: &Files) -> Readme {
    let mut readme = Readme::default();
    if let Some((_, bytes)) = files
        .iter()
        .find(|(name, _)| basename_lower(name) == "00readme.json")
    {
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes)
            && let Some(sources) = value.get("sources").and_then(|sources| sources.as_array())
        {
            for source in sources {
                let Some(name) = source.get("filename").and_then(|name| name.as_str()) else {
                    continue;
                };
                match source.get("usage").and_then(|usage| usage.as_str()) {
                    Some("toplevel") => readme.toplevel.push(basename_lower(name)),
                    Some("ignore") => {
                        readme.ignore.insert(basename_lower(name));
                    }
                    _ => {}
                }
            }
        }
        return readme;
    }
    if let Some((_, bytes)) = files
        .iter()
        .find(|(name, _)| basename_lower(name).starts_with("00readme"))
    {
        for line in String::from_utf8_lossy(bytes).lines() {
            let mut parts = line.split_whitespace();
            let (Some(name), Some(directive)) = (parts.next(), parts.next()) else {
                continue;
            };
            match directive {
                "toplevelfile" => readme.toplevel.push(basename_lower(name)),
                "ignore" => {
                    readme.ignore.insert(basename_lower(name));
                }
                _ => {}
            }
        }
    }
    readme
}

fn find_main_tex(files: &Files) -> Option<(String, String)> {
    let readme = parse_readme(files);
    let mut tex_files: Vec<(String, String)> = files
        .iter()
        .filter(|(name, _)| name.to_ascii_lowercase().ends_with(".tex"))
        .filter(|(name, _)| !readme.ignore.contains(&basename_lower(name)))
        .map(|(name, bytes)| (name.clone(), String::from_utf8_lossy(bytes).into_owned()))
        .collect();
    tex_files.sort_by(|left, right| left.0.cmp(&right.0));
    for top in &readme.toplevel {
        if let Some(file) = tex_files
            .iter()
            .find(|(name, _)| &basename_lower(name) == top)
        {
            return Some(file.clone());
        }
    }
    let has_class = |source: &str| source.contains("\\documentclass");
    let is_document = |source: &str| has_class(source) && source.contains("\\begin{document}");
    let is_region = |name: &str, source: &str| {
        basename_lower(name) == "_region_.tex"
            || (source.contains("!name(") && source.contains("!offset("))
    };
    let score = |source: &str| {
        let count = |pattern| source.matches(pattern).count();
        (count("\\input") + count("\\include{")) * 2
            + count("\\section") * 2
            + usize::from(source.contains("\\bibliography") || source.contains("thebibliography"))
                * 5
            + usize::from(source.contains("\\maketitle") || source.contains("\\title")) * 3
    };
    let documents: Vec<&(String, String)> = tex_files
        .iter()
        .filter(|(name, source)| is_document(source) && !is_region(name, source))
        .collect();
    if let Some(best) = documents
        .iter()
        .max_by_key(|(name, source)| (score(source), std::cmp::Reverse(name.clone())))
    {
        return Some((*best).clone());
    }
    tex_files
        .iter()
        .find(|(_, source)| is_document(source))
        .or_else(|| tex_files.iter().find(|(_, source)| has_class(source)))
        .or_else(|| tex_files.first())
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(entries: &[(&str, &str)]) -> Files {
        entries
            .iter()
            .map(|(name, source)| (name.to_string(), source.as_bytes().to_vec()))
            .collect()
    }

    const REGION: &str = "\\documentclass{article}\\begin{document}text\\message{ !name(real.tex) !offset(1) }\\end{document}";
    const REAL: &str = "\\documentclass{article}\\title{T}\\begin{document}\\maketitle\\input{intro}\\section{A}\\section{B}\\bibliography{refs}\\end{document}";

    #[test]
    fn normalizes_ids_and_urls() {
        assert_eq!(normalize_id("2205.14135v2").unwrap(), "2205.14135v2");
        assert_eq!(
            normalize_id("https://arxiv.org/pdf/2205.14135v2.pdf").unwrap(),
            "2205.14135v2"
        );
        assert_eq!(
            normalize_id("https://arxiv.org/abs/hep-th/9901001").unwrap(),
            "hep-th/9901001"
        );
        assert!(normalize_id("https://example.com/paper").is_err());
    }

    #[test]
    fn readme_toplevel_wins() {
        let files = files(&[
            (
                "00README.json",
                r#"{"sources":[{"filename":"paper.tex","usage":"toplevel"}]}"#,
            ),
            ("aaa.tex", REAL),
            ("paper.tex", "\\documentclass{x}\\begin{document}paper"),
        ]);
        assert_eq!(find_main_tex(&files).unwrap().0, "paper.tex");
    }

    #[test]
    fn region_file_does_not_beat_real_main() {
        let files = files(&[("_region_.tex", REGION), ("paper.tex", REAL)]);
        assert_eq!(find_main_tex(&files).unwrap().0, "paper.tex");
    }

    #[test]
    fn raw_tex_becomes_main_file() {
        let files = unpack(b"\\documentclass{x}\\begin{document}paper");
        assert!(files.contains_key("main.tex"));
        assert_eq!(find_main_tex(&files).unwrap().0, "main.tex");
    }

    #[test]
    fn resolver_uses_main_directory() {
        let project = Project {
            id: "1".to_string(),
            main_name: "paper/main.tex".to_string(),
            main_source: String::new(),
            files: files(&[("paper/sections/intro.tex", "intro")]),
        };
        assert_eq!(
            Resolver::resolve(&project, "sections/intro").as_deref(),
            Some("intro")
        );
    }
}
