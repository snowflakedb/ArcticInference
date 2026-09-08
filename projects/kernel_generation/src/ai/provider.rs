//! Provider definitions, loaded from JSON files.
//!
//! There are no providers in the code. Each one is a `.json` file in
//! `~/.kernelguy/providers/`, deserialized into a [`CompletionProvider`]. The files
//! under `src/default_providers/` in the repo are compiled into the binary and copied
//! into that directory on first launch, so a fresh machine works with no setup
//! while the user's copy stays editable — add a model, retarget a host, or swap the
//! auth command without rebuilding.
//!
//! A provider's **name is its filename stem**, so nothing inside the file can
//! disagree with what it is called.

use std::path::{Path, PathBuf};

use crate::ai::CompletionProvider;

/// Files under `src/default_providers/`, embedded so a fresh install can seed
/// itself. `(file name, contents)`.
const DEFAULT_PROVIDERS: &[(&str, &str)] = &[(
    "snowflake_cortex.json",
    include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/default_providers/snowflake_cortex.json"
    )),
)];

/// Why providers could not be loaded.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("$HOME is not set, so ~/.kernelguy/providers cannot be located")]
    NoHome,
    #[error("could not create {}: {source}", path.display())]
    CreateDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not write {}: {source}", path.display())]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not read {}: {source}", path.display())]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{} is not valid provider JSON: {source}", path.display())]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("no providers in {} — remove the directory to re-seed the defaults", dir.display())]
    None { dir: PathBuf },
}

/// A provider plus the name it is known by (its filename stem).
#[derive(Debug, Clone)]
pub struct NamedProvider {
    pub name: String,
    pub provider: CompletionProvider,
}

/// `~/.kernelguy/providers`.
///
/// # Errors
///
/// Returns [`ProviderError::NoHome`] if `$HOME` is unset.
pub fn dir() -> Result<PathBuf, ProviderError> {
    let home = std::env::var_os("HOME").ok_or(ProviderError::NoHome)?;
    Ok(PathBuf::from(home).join(".kernelguy").join("providers"))
}

/// Create the providers directory and write any shipped default not already there.
///
/// Existing files are **never overwritten**, so local edits survive upgrades. The
/// flip side: a model added to a shipped default will not reach a machine that has
/// already launched once — delete that file (or the directory) to pick it up.
///
/// # Errors
///
/// Returns an error if the directory cannot be created or a default cannot be
/// written.
pub fn seed_defaults() -> Result<(), ProviderError> {
    let dir = dir()?;
    std::fs::create_dir_all(&dir).map_err(|source| ProviderError::CreateDir {
        path: dir.clone(),
        source,
    })?;
    for (name, contents) in DEFAULT_PROVIDERS {
        let path = dir.join(name);
        if path.exists() {
            continue;
        }
        std::fs::write(&path, contents).map_err(|source| ProviderError::Write {
            path: path.clone(),
            source,
        })?;
        eprintln!("kernelguy: seeded default provider {}", path.display());
    }
    Ok(())
}

/// Load every `*.json` in the providers directory, sorted by name so resolution
/// order is deterministic.
///
/// # Errors
///
/// Returns an error if the directory cannot be read, a file cannot be read or
/// parsed, or no providers were found. A malformed file is fatal rather than
/// skipped: silently ignoring it would look like the model it declares simply not
/// existing.
pub fn load_all() -> Result<Vec<NamedProvider>, ProviderError> {
    let dir = dir()?;
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|source| ProviderError::Read {
            path: dir.clone(),
            source,
        })?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("json")))
        .collect();
    paths.sort();

    let mut providers = Vec::with_capacity(paths.len());
    for path in paths {
        providers.push(load_one(&path)?);
    }
    if providers.is_empty() {
        return Err(ProviderError::None { dir });
    }
    Ok(providers)
}

fn load_one(path: &Path) -> Result<NamedProvider, ProviderError> {
    let text = std::fs::read_to_string(path).map_err(|source| ProviderError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let provider: CompletionProvider = serde_json::from_str(&text).map_err(|source| ProviderError::Parse {
        path: path.to_path_buf(),
        source,
    })?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unnamed")
        .to_string();
    Ok(NamedProvider { name, provider })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::CompletionProtocol;

    /// The shipped defaults must parse, or a fresh install seeds a file that then
    /// fails to load. They are embedded at compile time but only *parsed* at
    /// runtime, so nothing else would catch a typo in them.
    #[test]
    fn every_shipped_default_parses() {
        for (name, contents) in DEFAULT_PROVIDERS {
            let parsed: Result<CompletionProvider, _> = serde_json::from_str(contents);
            assert!(parsed.is_ok(), "{name} does not parse: {:?}", parsed.err());
        }
    }

    #[test]
    fn the_shipped_cortex_default_serves_both_implemented_protocols() {
        let (_, contents) = DEFAULT_PROVIDERS
            .iter()
            .find(|(n, _)| *n == "snowflake_cortex.json")
            .expect("cortex is shipped");
        let p: CompletionProvider = serde_json::from_str(contents).expect("parses");
        assert!(p.supports(CompletionProtocol::AnthropicMessages));
        assert!(p.supports(CompletionProtocol::OpenaiResponses));
        // No client implements this one, so the file must not claim it.
        assert!(!p.supports(CompletionProtocol::OpenaiChatCompletions));
    }

    /// `X-SNOWFLAKE-APPLICATION` used to be hardcoded into every HTTP client, so
    /// this is the only thing pinning it now that it is provider data. It reaches
    /// the wire through [`CompletionProvider::headers_for`], which is why the
    /// assertion goes through that rather than reading the map directly.
    #[test]
    fn the_shipped_cortex_default_declares_the_snowflake_application_header() {
        let (_, contents) = DEFAULT_PROVIDERS
            .iter()
            .find(|(n, _)| *n == "snowflake_cortex.json")
            .expect("cortex is shipped");
        let p: CompletionProvider = serde_json::from_str(contents).expect("parses");

        // Without this the loop below passes vacuously on an empty endpoint list.
        assert!(!p.endpoints.is_empty(), "cortex must serve at least one endpoint");
        for endpoint in &p.endpoints {
            let headers = p.headers_for(endpoint).expect("shipped headers are valid");
            assert_eq!(
                headers.get("x-snowflake-application").and_then(|v| v.to_str().ok()),
                Some("kernelguy"),
                "{:?} endpoint lost the application header",
                endpoint.protocol
            );
        }
    }

    /// Protocol comes from the enclosing endpoint, so a model cannot be routed one
    /// way and sized another — the property the old slug heuristics lacked.
    #[test]
    fn the_shipped_cortex_default_routes_and_sizes_from_one_place() {
        let (_, contents) = DEFAULT_PROVIDERS.first().expect("at least one default");
        let p: CompletionProvider = serde_json::from_str(contents).expect("parses");

        let (endpoint, model) = p.resolve("claude-opus-5").expect("listed");
        assert_eq!(endpoint.protocol, CompletionProtocol::AnthropicMessages);
        assert_eq!(model.context_size, 1_000_000);

        let (endpoint, model) = p.resolve("openai-gpt-5.5").expect("listed");
        assert_eq!(endpoint.protocol, CompletionProtocol::OpenaiResponses);
        assert_eq!(model.context_size, 400_000);
    }

    #[test]
    fn a_malformed_file_is_an_error_naming_the_path() {
        let dir = std::env::temp_dir().join("kernelguy-provider-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("broken.json");
        let _ = std::fs::write(&path, "{ not json");
        let err = load_one(&path).expect_err("must fail");
        let msg = err.to_string();
        assert!(msg.contains("broken.json"), "{msg}");
        assert!(msg.contains("not valid provider JSON"), "{msg}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_provider_is_named_after_its_file() {
        let dir = std::env::temp_dir().join("kernelguy-provider-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("my_cluster.json");
        let (_, contents) = DEFAULT_PROVIDERS.first().expect("at least one default");
        let _ = std::fs::write(&path, contents);
        let loaded = load_one(&path).expect("parses");
        assert_eq!(loaded.name, "my_cluster");
        let _ = std::fs::remove_file(&path);
    }
}
