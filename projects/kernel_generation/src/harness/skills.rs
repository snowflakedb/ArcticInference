use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::de::Error as _;
use serde::{Deserialize, Deserializer};

const SKILLS_REL_DIR: &str = "agent_ressources/skills";
const SKILL_MD: &str = "SKILL.md";
const MAX_DEPTH: usize = 6;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub host_dir: PathBuf,
    pub sandbox_dir: String,
}

impl Skill {
    #[must_use]
    pub fn host_skill_md(&self) -> PathBuf {
        self.host_dir.join(SKILL_MD)
    }

    #[must_use]
    pub fn sandbox_skill_md(&self) -> String {
        format!("{}/{}", self.sandbox_dir.trim_end_matches('/'), SKILL_MD)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillLoadError {
    message: String,
}

#[derive(Debug, Deserialize)]
struct SkillFrontmatter {
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    name: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    description: Option<String>,
}

fn deserialize_optional_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    match serde_yaml::Value::deserialize(deserializer)? {
        serde_yaml::Value::Null => Ok(None),
        serde_yaml::Value::String(value) => Ok(Some(value)),
        _ => Err(D::Error::custom("expected string")),
    }
}

impl SkillLoadError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for SkillLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SkillLoadError {}

/// Discover every repo-local `SKILL.md` bundle and map it to its sandbox path.
///
/// # Errors
///
/// Returns [`SkillLoadError`] when `agent_ressources/skills` exists but is not a
/// directory, the traversal hits an IO error, a `SKILL.md` cannot be read, its
/// YAML frontmatter is missing / unterminated / invalid / lacks a non-empty
/// `name` or `description`, or two skills declare the same `name`.
pub fn load_repo_skills(repo_root: &Path, sandbox_skills_root: &str) -> Result<Vec<Skill>, SkillLoadError> {
    let skills_root = repo_root.join(SKILLS_REL_DIR);
    if !skills_root.exists() {
        return Ok(Vec::new());
    }
    if !skills_root.is_dir() {
        return Err(SkillLoadError::new(format!(
            "{}: expected directory",
            skills_root.display()
        )));
    }

    let mut skill_dirs = Vec::new();
    collect_skill_dirs(&skills_root, 0, &mut skill_dirs)
        .map_err(|e| SkillLoadError::new(format!("{}: {e}", skills_root.display())))?;
    skill_dirs.sort();

    if skill_dirs.is_empty() {
        eprintln!(
            "kernelguy: warning: {} exists but contains no SKILL.md files",
            skills_root.display()
        );
        return Ok(Vec::new());
    }

    let sandbox_root = sandbox_skills_root.trim_end_matches('/');
    let mut skills = Vec::new();
    for host_dir in skill_dirs {
        let skill_md = host_dir.join(SKILL_MD);
        let text = std::fs::read_to_string(&skill_md)
            .map_err(|e| SkillLoadError::new(format!("{}: {e}", skill_md.display())))?;
        let (name, description) = parse_frontmatter(&text)
            .map_err(|reason| SkillLoadError::new(format!("{}: {reason}", skill_md.display())))?;
        let rel = host_dir.strip_prefix(&skills_root).map_err(|_| {
            SkillLoadError::new(format!(
                "{}: collected skill dir is not under {}",
                host_dir.display(),
                skills_root.display()
            ))
        })?;
        let rel_sandbox = path_to_slash(rel);
        let sandbox_dir = if rel_sandbox.is_empty() {
            sandbox_root.to_string()
        } else {
            format!("{sandbox_root}/{rel_sandbox}")
        };
        skills.push(Skill {
            name,
            description,
            host_dir,
            sandbox_dir,
        });
    }

    let mut by_name: HashMap<&str, Vec<String>> = HashMap::new();
    for skill in &skills {
        by_name
            .entry(skill.name.as_str())
            .or_default()
            .push(skill.host_dir.display().to_string());
    }
    let mut duplicates: Vec<_> = by_name
        .into_iter()
        .filter_map(|(name, dirs)| (dirs.len() > 1).then_some((name.to_string(), dirs)))
        .collect();
    duplicates.sort_by(|a, b| a.0.cmp(&b.0));
    if !duplicates.is_empty() {
        let details = duplicates
            .into_iter()
            .map(|(name, dirs)| format!("{name}: {}", dirs.join(", ")))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(SkillLoadError::new(format!("duplicate skill names: {details}")));
    }

    Ok(skills)
}

fn collect_skill_dirs(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if depth > MAX_DEPTH {
        return Ok(());
    }
    let skill_md = dir.join(SKILL_MD);
    if skill_md.exists() {
        out.push(dir.to_path_buf());
    }
    if depth == MAX_DEPTH {
        return Ok(());
    }

    let mut children = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        children.push(path);
    }
    children.sort();
    for child in children {
        collect_skill_dirs(&child, depth.saturating_add(1), out)?;
    }
    Ok(())
}

fn parse_frontmatter(text: &str) -> Result<(String, String), String> {
    let mut lines = text.lines();
    if lines.next() != Some("---") {
        return Err("missing YAML frontmatter".to_string());
    }

    let mut yaml = String::new();
    for line in lines {
        if line == "---" {
            let frontmatter: SkillFrontmatter =
                serde_yaml::from_str(&yaml).map_err(|e| format!("invalid YAML frontmatter: {e}"))?;
            let name = required_field(frontmatter.name, "name")?;
            let description = required_field(frontmatter.description, "description")?;
            return Ok((name, description));
        }
        yaml.push_str(line);
        yaml.push('\n');
    }
    Err("unterminated YAML frontmatter".to_string())
}

fn required_field(value: Option<String>, key: &str) -> Result<String, String> {
    let Some(value) = value else {
        return Err(format!("missing required `{key}` frontmatter field"));
    };
    let value = value.trim();
    if value.is_empty() {
        Err(format!("missing required `{key}` frontmatter field"))
    } else {
        Ok(value.to_string())
    }
}

fn path_to_slash(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    fn skill_md(name: &str, description: &str) -> String {
        format!("---\nname: {name}\ndescription: {description}\nignored: yes\n---\n# Body\n")
    }

    #[test]
    fn missing_agents_skills_returns_empty() {
        let tmp = TempDir::new().unwrap();
        assert!(load_repo_skills(tmp.path(), "/workspace/skills").unwrap().is_empty());
    }

    #[test]
    fn empty_existing_root_returns_empty() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join(SKILLS_REL_DIR)).unwrap();
        assert!(load_repo_skills(tmp.path(), "/workspace/skills").unwrap().is_empty());
    }

    #[test]
    fn valid_skill_maps_to_sandbox_path() {
        let tmp = TempDir::new().unwrap();
        write(
            &tmp.path().join("agent_ressources/skills/cuda/SKILL.md"),
            &skill_md("cuda", "CUDA guidance"),
        );
        let skills = load_repo_skills(tmp.path(), "/workspace/skills").unwrap();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "cuda");
        assert_eq!(skills[0].description, "CUDA guidance");
        assert_eq!(skills[0].sandbox_dir, "/workspace/skills/cuda");
        assert_eq!(skills[0].sandbox_skill_md(), "/workspace/skills/cuda/SKILL.md");
    }

    #[test]
    fn yaml_frontmatter_supports_colons_and_quotes() {
        let tmp = TempDir::new().unwrap();
        write(
            &tmp.path().join("agent_ressources/skills/quoted/SKILL.md"),
            "---\nname: 'cuda: inline'\ndescription: \"Use torch: load_inline\"\n---\n# Body\n",
        );
        let skills = load_repo_skills(tmp.path(), "/workspace/skills").unwrap();
        assert_eq!(skills[0].name, "cuda: inline");
        assert_eq!(skills[0].description, "Use torch: load_inline");
    }

    #[test]
    fn yaml_frontmatter_rejects_non_string_fields() {
        let tmp = TempDir::new().unwrap();
        write(
            &tmp.path().join("agent_ressources/skills/bad/SKILL.md"),
            "---\nname: 12\ndescription: Bad\n---\n",
        );
        let err = load_repo_skills(tmp.path(), "/workspace/skills").unwrap_err();
        assert!(err.to_string().contains("invalid YAML frontmatter"));
    }

    #[test]
    fn malformed_frontmatter_errors() {
        let tmp = TempDir::new().unwrap();
        write(&tmp.path().join("agent_ressources/skills/bad/SKILL.md"), "# nope\n");
        let err = load_repo_skills(tmp.path(), "/workspace/skills").unwrap_err();
        assert!(err.to_string().contains("missing YAML frontmatter"));
        assert!(err.to_string().contains("SKILL.md"));
    }

    #[test]
    fn missing_name_errors() {
        let tmp = TempDir::new().unwrap();
        write(
            &tmp.path().join("agent_ressources/skills/bad/SKILL.md"),
            "---\ndescription: Missing name\n---\n",
        );
        let err = load_repo_skills(tmp.path(), "/workspace/skills").unwrap_err();
        assert!(err.to_string().contains("missing required `name`"));
    }

    #[test]
    fn missing_description_errors() {
        let tmp = TempDir::new().unwrap();
        write(
            &tmp.path().join("agent_ressources/skills/bad/SKILL.md"),
            "---\nname: bad\n---\n",
        );
        let err = load_repo_skills(tmp.path(), "/workspace/skills").unwrap_err();
        assert!(err.to_string().contains("missing required `description`"));
    }

    #[test]
    fn duplicate_names_error_with_conflicting_dirs() {
        let tmp = TempDir::new().unwrap();
        write(
            &tmp.path().join("agent_ressources/skills/a/SKILL.md"),
            &skill_md("dupe", "one"),
        );
        write(
            &tmp.path().join("agent_ressources/skills/b/SKILL.md"),
            &skill_md("dupe", "two"),
        );
        let err = load_repo_skills(tmp.path(), "/workspace/skills").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("duplicate skill names"));
        assert!(msg.contains("agent_ressources/skills/a"));
        assert!(msg.contains("agent_ressources/skills/b"));
    }

    #[test]
    fn nested_skills_are_discovered_and_mapped() {
        let tmp = TempDir::new().unwrap();
        write(
            &tmp.path().join("agent_ressources/skills/gpu/cuda/tuning/SKILL.md"),
            &skill_md("tuning", "Tune kernels"),
        );
        let skills = load_repo_skills(tmp.path(), "/workspace/skills").unwrap();
        assert_eq!(skills[0].sandbox_dir, "/workspace/skills/gpu/cuda/tuning");
    }

    #[test]
    fn max_depth_six_is_inclusive() {
        let tmp = TempDir::new().unwrap();
        write(
            &tmp.path().join("agent_ressources/skills/d1/d2/d3/d4/d5/d6/SKILL.md"),
            &skill_md("visible", "At max depth"),
        );
        write(
            &tmp.path().join("agent_ressources/skills/d1/d2/d3/d4/d5/d6/d7/SKILL.md"),
            &skill_md("hidden", "Too deep"),
        );
        let skills = load_repo_skills(tmp.path(), "/workspace/skills").unwrap();
        assert_eq!(
            skills.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            vec!["visible"]
        );
    }

    #[test]
    fn dot_dirs_are_skipped() {
        let tmp = TempDir::new().unwrap();
        write(
            &tmp.path().join("agent_ressources/skills/.hidden/SKILL.md"),
            &skill_md("hidden", "No"),
        );
        assert!(load_repo_skills(tmp.path(), "/workspace/skills").unwrap().is_empty());
    }

    #[test]
    fn actual_repo_load_inline_skill_loads() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let skills = load_repo_skills(repo, "/workspace/skills").unwrap();
        let skill = skills
            .iter()
            .find(|skill| skill.name == "nvidia-torch-cuda-load-inline")
            .expect("repo-local load_inline skill should be discoverable");
        assert_eq!(
            skill.sandbox_skill_md(),
            "/workspace/skills/nvidia-torch-cuda-load-inline/SKILL.md"
        );
        assert!(!skill.description.is_empty());
    }
}
