//! The runtime half of the model vocabulary: plain data describing *who* serves
//! *which protocols* with *which models*, under *which credential*.
//!
//! Everything here is deserialized — providers are JSON files, never code, so
//! these are ordinary owned types. See [`crate::ai::provider`]. The comptime half
//! is [`ProtocolClient`](crate::ai::ProtocolClient); `main` crosses from this side
//! to that one exactly once, when it picks a client.

use std::collections::BTreeMap;
use std::sync::Arc;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;

use crate::ai::{AuthError, AuthProvider, CommandAuth, StaticAuth};

/// A wire protocol. The runtime tag for the axis that
/// [`ProtocolClient`](crate::ai::ProtocolClient) expresses at compile time.
///
/// **Declaration order is resolution priority.** When a provider lists one model
/// id under several endpoints, [`CompletionProvider::resolve`] picks the
/// lowest-ordered protocol, so reordering these variants silently changes which
/// endpoint answers. `protocol_order_is_resolution_priority` pins it.
///
/// `ValueEnum` so a CLI flag can be typed as this enum directly: clap then
/// enumerates the valid values and rejects anything else, and adding a variant
/// needs no parser kept in sync. Its default rename is kebab-case, matching serde,
/// which `value_enum_and_serde_accept_the_same_spellings` pins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum CompletionProtocol {
    OpenaiResponses,
    AnthropicMessages,
    OpenaiChatCompletions,
}

impl CompletionProtocol {
    /// Path appended to a provider's `base_url` to reach this protocol.
    ///
    /// A convention rather than a per-provider field, which holds for every
    /// provider we target (Cortex, `OpenAI` direct, Anthropic direct, Ollama). It
    /// does *not* hold for Azure (deployment name in the path plus an
    /// `api-version` query) or Bedrock (model in the path); either would need an
    /// explicit per-endpoint path override.
    #[must_use]
    pub const fn path(self) -> &'static str {
        match self {
            Self::OpenaiResponses => "/responses",
            Self::OpenaiChatCompletions => "/chat/completions",
            Self::AnthropicMessages => "/messages",
        }
    }
}

/// A model as named on the wire, plus what the orchestrator needs to size a
/// context.
///
/// `context_size` is `u32` rather than `usize` to match the compaction
/// arithmetic it feeds (`should_compact`, `AvoConfig::context_window_tokens`);
/// mixing in `usize` would force casts that `clippy::cast_possible_truncation`
/// denies.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub id: String,
    pub context_size: u32,
}

/// How to obtain the credential for an endpoint.
///
/// The three variants and their semantics are `.dev/contracts/ai_provider_auth.txt`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum AuthMethod {
    /// A constant used as the key for every request. Convenient for a local
    /// server; note it sits in cleartext in the provider file.
    Key(String),
    /// Name of an environment variable holding the key. Read and cached once at
    /// startup, so changing the variable mid-run has no effect.
    EnvVar(String),
    /// A program whose stdout is the key, spawned directly with these arguments —
    /// no shell, so nothing word-splits or expands them. Cached, re-run when a
    /// request fails, and never run in parallel; see
    /// [`CommandAuth`](crate::ai::CommandAuth).
    Command {
        binary: String,
        #[serde(default)]
        args: Vec<String>,
    },
}

impl AuthMethod {
    /// Turn the description into a live credential source.
    ///
    /// # Errors
    ///
    /// Returns an error if an [`Self::EnvVar`] is unset or empty, or if a
    /// [`Self::Command`] has no binary or fails its first invocation — so a broken
    /// credential fails during startup rather than mid-run.
    pub async fn resolve(&self) -> Result<Arc<dyn AuthProvider>, AuthError> {
        match self {
            Self::Key(key) => Ok(Arc::new(StaticAuth::bearer(key))),
            Self::EnvVar(name) => {
                let key = std::env::var(name).map_err(|_| AuthError::EnvVarUnset { var: name.clone() })?;
                if key.trim().is_empty() {
                    return Err(AuthError::EnvVarEmpty { var: name.clone() });
                }
                Ok(Arc::new(StaticAuth::bearer(key.trim())))
            }
            Self::Command { binary, args } => Ok(Arc::new(CommandAuth::new(binary, args.clone()).await?)),
        }
    }
}

/// Why a configured header cannot be used.
///
/// Separate from [`AuthError`] and deliberately **not** convertible into
/// [`CompletionError`](crate::ai::CompletionError): adding another
/// `From<_> for CompletionError` breaks inference at closures that currently infer
/// their error type uniquely. `main` stringifies this at its one boundary.
#[derive(Debug, thiserror::Error)]
pub enum HeaderError {
    #[error("header `{name}` cannot be configured: {reason}")]
    Reserved { name: &'static str, reason: &'static str },
    #[error("`{name}` is not a valid HTTP header name")]
    InvalidName { name: String },
    #[error("the value configured for header `{name}` is not a valid HTTP header value")]
    InvalidValue { name: String },
}

/// Headers kernelguy or hyper sets itself, and why a configured value cannot stand.
///
/// The first two would be **silently discarded** — both are set per request, and
/// reqwest fills client defaults only into headers the request left vacant. The rest
/// are worse: hyper honours a caller-set value, so these actively corrupt the
/// exchange rather than being ignored. Rejecting at startup beats either outcome.
const RESERVED: [(&str, &str); 6] = [
    ("authorization", "it is taken from the endpoint's `auth`"),
    ("content-type", "it is set per request with the JSON body"),
    (
        "content-length",
        "hyper frames the body and honours a caller-set length, so a wrong value truncates or stalls every request",
    ),
    (
        "transfer-encoding",
        "hyper chooses the framing; overriding it corrupts every request",
    ),
    ("host", "it has to match the URL being requested, and the TLS SNI name"),
    (
        "accept-encoding",
        "no decompression feature is enabled, so a response encoded on request would reach the stream parser raw",
    ),
];

/// One protocol served by a provider: the models reachable over it and the
/// credential it takes.
///
/// Auth sits here rather than on [`CompletionProvider`] because a host can serve
/// two protocols behind different credentials.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionEndpoint {
    pub protocol: CompletionProtocol,
    pub models: Vec<Model>,
    pub auth: AuthMethod,
    /// Headers layered over the provider's, for anything protocol-specific — an
    /// `anthropic-beta` flag belongs on the Messages endpoint alone. Merged by
    /// [`CompletionProvider::headers_for`]; same cleartext caveat as there.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
}

/// A host serving one or more protocols.
///
/// `endpoints` is the single source of truth: "which protocols are supported" and
/// "which models exist" are the same list, so the two cannot drift. That is
/// deliberate — three independent slug heuristics that disagreed with each other
/// are exactly what this type replaces.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionProvider {
    pub base_url: String,
    pub endpoints: Vec<CompletionEndpoint>,
    /// Headers sent to every endpoint of this host — a tenant or application
    /// identifier, say. Per-endpoint entries win; see [`Self::headers_for`].
    ///
    /// These sit in cleartext in the provider file, exactly as [`AuthMethod::Key`]
    /// does, so prefer [`AuthMethod::Command`] or [`AuthMethod::EnvVar`] for a
    /// credential — both keep the secret out of the file entirely.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
}

impl CompletionProvider {
    /// Whether this provider serves `protocol`.
    #[must_use]
    pub fn supports(&self, protocol: CompletionProtocol) -> bool {
        self.endpoints.iter().any(|e| e.protocol == protocol)
    }

    /// Full URL for `protocol`, or `None` if this provider doesn't serve it.
    #[must_use]
    pub fn endpoint_url(&self, protocol: CompletionProtocol) -> Option<String> {
        self.supports(protocol)
            .then(|| format!("{}{}", self.base_url.trim_end_matches('/'), protocol.path()))
    }

    /// Find the endpoint and model for `id`.
    ///
    /// When several endpoints list the same id, the one whose protocol sorts
    /// lowest wins — see [`CompletionProtocol`]. That makes the choice a stated
    /// property of the protocol set rather than of the order endpoints happen to
    /// appear in the file. Use [`Self::resolve_with_protocol`] to pin one.
    #[must_use]
    pub fn resolve(&self, id: &str) -> Option<(&CompletionEndpoint, &Model)> {
        self.endpoints
            .iter()
            .filter_map(|e| e.models.iter().find(|m| m.id == id).map(|m| (e, m)))
            .min_by_key(|(e, _)| e.protocol)
    }

    /// Find `id` specifically on the endpoint serving `protocol`, ignoring
    /// priority. `None` if this provider has no such endpoint, or that endpoint
    /// does not list `id`.
    #[must_use]
    pub fn resolve_with_protocol(
        &self,
        protocol: CompletionProtocol,
        id: &str,
    ) -> Option<(&CompletionEndpoint, &Model)> {
        self.endpoints
            .iter()
            .filter(|e| e.protocol == protocol)
            .find_map(|e| e.models.iter().find(|m| m.id == id).map(|m| (e, m)))
    }

    /// This provider's headers with `endpoint`'s layered on top.
    ///
    /// Names are folded to lowercase before merging, because HTTP compares them
    /// case-insensitively and [`HeaderName`] lowercases: a provider `X-Foo` and an
    /// endpoint `x-foo` are **one** header, and the endpoint wins. Merging the raw
    /// JSON keys instead would keep both and let ASCII ordering pick a winner.
    ///
    /// One value per name — a JSON object cannot express a repeated header, so
    /// values replace rather than append.
    ///
    /// `endpoint` is assumed to belong to this provider; callers get it from
    /// [`Self::resolve`].
    ///
    /// # Errors
    ///
    /// Returns [`HeaderError`] if a name or value is not legal in HTTP, or names a
    /// header set per request downstream — where a configured value would be
    /// silently discarded rather than applied.
    pub fn headers_for(&self, endpoint: &CompletionEndpoint) -> Result<HeaderMap, HeaderError> {
        // Keyed by the folded name so collisions resolve, but carrying the original
        // spelling: an error naming `x foo` when the file says `X Foo` is not
        // greppable.
        let mut merged: BTreeMap<String, (&str, &str)> = BTreeMap::new();
        for (name, value) in self.headers.iter().chain(&endpoint.headers) {
            merged.insert(name.to_ascii_lowercase(), (name.as_str(), value.as_str()));
        }

        // Not `with_capacity`: http's is `try_with_capacity(..).expect(..)`, which
        // panics above MAX_SIZE (1<<15). A provider file with enough distinct names
        // would panic out of a `-> Result` fn in a crate that denies panics, and
        // clippy cannot see it because the expect lives in `http`.
        let mut out = HeaderMap::new();
        for (folded, (name, value)) in merged {
            if let Some(&(name, reason)) = RESERVED.iter().find(|(r, _)| *r == folded.as_str()) {
                return Err(HeaderError::Reserved { name, reason });
            }
            // Folding only lowercases ASCII, which is all `HeaderName` accepts, so
            // validating the folded form accepts exactly the same set.
            let Ok(header) = HeaderName::from_bytes(folded.as_bytes()) else {
                return Err(HeaderError::InvalidName { name: name.to_owned() });
            };
            let Ok(header_value) = HeaderValue::from_str(value) else {
                return Err(HeaderError::InvalidValue { name: name.to_owned() });
            };
            out.insert(header, header_value);
        }
        Ok(out)
    }

    /// Every model id this provider serves, in endpoint order — for error
    /// messages and `--list-models`-style discovery.
    #[must_use]
    pub fn model_ids(&self) -> Vec<&str> {
        self.endpoints
            .iter()
            .flat_map(|e| e.models.iter().map(|m| m.id.as_str()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> CompletionProvider {
        let json = r#"{
            "base_url": "https://example.test/v1/",
            "endpoints": [
                {
                    "protocol": "anthropic-messages",
                    "auth": {"key": "k"},
                    "models": [{"id": "claude-x", "context_size": 1000}]
                },
                {
                    "protocol": "openai-responses",
                    "auth": {"key": "k"},
                    "models": [{"id": "gpt-x", "context_size": 2000}]
                }
            ]
        }"#;
        serde_json::from_str(json).expect("valid config")
    }

    #[test]
    fn endpoint_url_joins_base_and_protocol_path_without_doubling_the_slash() {
        let p = provider();
        assert_eq!(
            p.endpoint_url(CompletionProtocol::AnthropicMessages).as_deref(),
            Some("https://example.test/v1/messages")
        );
    }

    #[test]
    fn an_unserved_protocol_has_no_url() {
        let p = provider();
        assert!(!p.supports(CompletionProtocol::OpenaiChatCompletions));
        assert!(p.endpoint_url(CompletionProtocol::OpenaiChatCompletions).is_none());
    }

    #[test]
    fn resolve_yields_the_endpoint_that_serves_the_model() {
        let p = provider();
        let (endpoint, model) = p.resolve("gpt-x").expect("gpt-x is listed");
        assert_eq!(endpoint.protocol, CompletionProtocol::OpenaiResponses);
        assert_eq!(model.context_size, 2_000);
        // Protocol comes from the enclosing endpoint, so it cannot disagree with
        // the model — the property the old slug heuristics lacked.
        let (endpoint, _) = p.resolve("claude-x").expect("claude-x is listed");
        assert_eq!(endpoint.protocol, CompletionProtocol::AnthropicMessages);
    }

    #[test]
    fn an_unknown_model_does_not_resolve_and_is_listable() {
        let p = provider();
        assert!(p.resolve("deepseek-v4-flash-0731").is_none());
        assert_eq!(p.model_ids(), vec!["claude-x", "gpt-x"]);
    }

    /// The config-facing shape of [`AuthMethod::Command`]: a binary plus an
    /// argument list, so a file never has to encode quoting for a shell that
    /// isn't there. `args` defaults to empty.
    #[test]
    fn a_command_credential_deserializes_as_binary_and_args() {
        let json = r#"{
            "base_url": "https://example.test/v1",
            "endpoints": [{
                "protocol": "anthropic-messages",
                "models": [{"id": "m", "context_size": 1000}],
                "auth": {"command": {"binary": "my-helper", "args": ["token", "--json"]}}
            }, {
                "protocol": "openai-responses",
                "models": [{"id": "n", "context_size": 1000}],
                "auth": {"command": {"binary": "bare-helper"}}
            }]
        }"#;
        let p: CompletionProvider = serde_json::from_str(json).expect("valid config");
        let commands: Vec<_> = p
            .endpoints
            .iter()
            .map(|e| match &e.auth {
                AuthMethod::Command { binary, args } => (binary.clone(), args.clone()),
                _ => (String::new(), vec![]),
            })
            .collect();
        assert_eq!(
            commands,
            vec![
                ("my-helper".to_string(), vec!["token".to_string(), "--json".to_string()]),
                ("bare-helper".to_string(), vec![]),
            ]
        );
    }

    /// Resolution priority, pinned. Reordering the enum silently changes which
    /// endpoint answers for a model listed under several, so this asserts the
    /// intended order outright: Responses, then Messages, then Chat Completions.
    #[test]
    fn protocol_order_is_resolution_priority() {
        let mut all = vec![
            CompletionProtocol::OpenaiChatCompletions,
            CompletionProtocol::AnthropicMessages,
            CompletionProtocol::OpenaiResponses,
        ];
        all.sort();
        assert_eq!(
            all,
            vec![
                CompletionProtocol::OpenaiResponses,
                CompletionProtocol::AnthropicMessages,
                CompletionProtocol::OpenaiChatCompletions,
            ]
        );
    }

    /// `--protocol-override` and a provider file must name protocols identically,
    /// so clap's `ValueEnum` spellings have to match serde's.
    #[test]
    fn value_enum_and_serde_accept_the_same_spellings() {
        use clap::ValueEnum;

        for name in ["openai-responses", "anthropic-messages", "openai-chat-completions"] {
            let via_clap = CompletionProtocol::from_str(name, false).expect("clap accepts it");
            let via_serde: CompletionProtocol = serde_json::from_str(&format!("\"{name}\"")).expect("serde accepts it");
            assert_eq!(via_clap, via_serde, "{name}");
        }
        // Every variant is reachable from the CLI — no variant can be left out.
        assert_eq!(CompletionProtocol::value_variants().len(), 3);
    }

    /// A model listed under two endpoints resolves by protocol priority, not by
    /// the order the endpoints appear in the file — here Messages is listed first
    /// yet Responses wins.
    #[test]
    fn a_duplicated_model_resolves_by_protocol_priority() {
        let json = r#"{
            "base_url": "https://example.test/v1",
            "endpoints": [
                {
                    "protocol": "anthropic-messages",
                    "auth": {"key": "k"},
                    "models": [{"id": "both", "context_size": 111}]
                },
                {
                    "protocol": "openai-responses",
                    "auth": {"key": "k"},
                    "models": [{"id": "both", "context_size": 222}]
                }
            ]
        }"#;
        let p: CompletionProvider = serde_json::from_str(json).expect("valid config");
        let (endpoint, model) = p.resolve("both").expect("listed twice");
        assert_eq!(endpoint.protocol, CompletionProtocol::OpenaiResponses);
        assert_eq!(model.context_size, 222);

        // The override reaches the lower-priority endpoint.
        let (endpoint, model) = p
            .resolve_with_protocol(CompletionProtocol::AnthropicMessages, "both")
            .expect("also listed there");
        assert_eq!(endpoint.protocol, CompletionProtocol::AnthropicMessages);
        assert_eq!(model.context_size, 111);

        // And cannot invent an endpoint the provider does not serve.
        assert!(
            p.resolve_with_protocol(CompletionProtocol::OpenaiChatCompletions, "both")
                .is_none()
        );
    }

    /// Both maps are optional, like `AuthMethod::Command`'s `args`. A provider file
    /// written before headers existed must still load.
    #[test]
    fn headers_default_to_empty_when_omitted() {
        let p = provider();
        assert!(p.headers.is_empty());
        assert!(p.endpoints.iter().all(|e| e.headers.is_empty()));
        let merged = p.headers_for(&p.endpoints[0]).expect("no headers is valid");
        assert!(merged.is_empty());
    }

    /// Provider headers reach every endpoint; an endpoint adds its own on top.
    #[test]
    fn endpoint_headers_layer_over_provider_headers() {
        let json = r#"{
            "base_url": "https://example.test/v1",
            "headers": {"X-Tenant": "acme"},
            "endpoints": [
                {
                    "protocol": "anthropic-messages",
                    "auth": {"key": "k"},
                    "headers": {"anthropic-beta": "ctx-1m"},
                    "models": [{"id": "m", "context_size": 1}]
                },
                {
                    "protocol": "openai-responses",
                    "auth": {"key": "k"},
                    "models": [{"id": "n", "context_size": 1}]
                }
            ]
        }"#;
        let p: CompletionProvider = serde_json::from_str(json).expect("valid config");

        let with_beta = p.headers_for(&p.endpoints[0]).expect("valid");
        assert_eq!(with_beta.get("x-tenant").unwrap(), "acme");
        assert_eq!(with_beta.get("anthropic-beta").unwrap(), "ctx-1m");

        // The endpoint that declares none still inherits the provider's, and does
        // not pick up the sibling's.
        let plain = p.headers_for(&p.endpoints[1]).expect("valid");
        assert_eq!(plain.get("x-tenant").unwrap(), "acme");
        assert!(plain.get("anthropic-beta").is_none());
    }

    /// HTTP header names are case-insensitive, so a differently-cased endpoint key
    /// must *override* the provider's rather than sit beside it. Merging raw JSON
    /// keys would keep both and let ASCII ordering decide.
    #[test]
    fn endpoint_overrides_provider_across_differing_case() {
        let json = r#"{
            "base_url": "https://example.test/v1",
            "headers": {"X-Foo": "from-provider"},
            "endpoints": [{
                "protocol": "anthropic-messages",
                "auth": {"key": "k"},
                "headers": {"x-foo": "from-endpoint"},
                "models": [{"id": "m", "context_size": 1}]
            }]
        }"#;
        let p: CompletionProvider = serde_json::from_str(json).expect("valid config");
        let merged = p.headers_for(&p.endpoints[0]).expect("valid");
        assert_eq!(merged.len(), 1, "one header, not two");
        assert_eq!(merged.get("x-foo").unwrap(), "from-endpoint");
    }

    /// Both are set per request downstream, so accepting them would silently do
    /// nothing. Rejecting at startup is the only honest option.
    #[test]
    fn headers_set_per_request_are_rejected() {
        for (key, expected) in [
            ("Authorization", "endpoint's `auth`"),
            ("Content-Type", "set per request"),
            // Not silently dropped like the two above — hyper honours a caller-set
            // length and would truncate or stall every request.
            ("Content-Length", "truncates or stalls"),
            ("Transfer-Encoding", "corrupts every request"),
            ("Host", "TLS SNI"),
            ("Accept-Encoding", "stream parser raw"),
        ] {
            let json = format!(
                r#"{{
                    "base_url": "https://example.test/v1",
                    "headers": {{"{key}": "whatever"}},
                    "endpoints": [{{
                        "protocol": "anthropic-messages",
                        "auth": {{"key": "k"}},
                        "models": [{{"id": "m", "context_size": 1}}]
                    }}]
                }}"#
            );
            let p: CompletionProvider = serde_json::from_str(&json).expect("valid config");
            let err = p.headers_for(&p.endpoints[0]).expect_err("must reject");
            assert!(err.to_string().contains(expected), "{key}: got {err}");
        }
    }

    /// A malformed name or value is a config error naming the offending key, not a
    /// panic — the provider file is user-authored.
    ///
    /// Each case pins its **own** key and its **own** variant. An earlier version
    /// asserted `contains(key_a) || contains(key_b)` across both iterations, which
    /// passed even when the variants were swapped, and even when the name was
    /// hardcoded — it verified nothing it claimed to.
    #[test]
    fn malformed_names_and_values_error_with_the_key() {
        // A space is illegal in a header name; a newline is illegal in a value.
        let cases = [
            (r#"{"has space": "v"}"#, "has space", "not a valid HTTP header name"),
            ("{\"x-ok\": \"line\\nbreak\"}", "x-ok", "not a valid HTTP header value"),
        ];
        for (bad, key, expected) in cases {
            let json = format!(
                r#"{{
                    "base_url": "https://example.test/v1",
                    "headers": {bad},
                    "endpoints": [{{
                        "protocol": "anthropic-messages",
                        "auth": {{"key": "k"}},
                        "models": [{{"id": "m", "context_size": 1}}]
                    }}]
                }}"#
            );
            let p: CompletionProvider = serde_json::from_str(&json).expect("valid config");
            let err = p.headers_for(&p.endpoints[0]).expect_err("must reject");
            let msg = err.to_string();
            assert!(msg.contains(key), "error should name `{key}`; got {msg}");
            assert!(msg.contains(expected), "wrong variant for `{key}`; got {msg}");
        }
    }

    /// A misspelled key must fail loudly. Without `deny_unknown_fields` a
    /// `#[serde(default)]` field like `headers` is indistinguishable from a typo: the
    /// file parses, the run starts, and nothing is sent — the exact silent no-op that
    /// [`HeaderError::Reserved`] exists to prevent one level down.
    #[test]
    fn a_misspelled_config_key_is_rejected_rather_than_ignored() {
        let cases = [
            // Capitalised `headers` — the mistake this guard is really for.
            (
                r#"{"base_url": "u", "Headers": {"x-a": "1"}, "endpoints": []}"#,
                "Headers",
            ),
            // Singular, at endpoint level.
            (
                r#"{"base_url": "u", "endpoints": [{"protocol": "anthropic-messages",
                   "auth": {"key": "k"}, "models": [], "header": {}}]}"#,
                "header",
            ),
            // Inside the auth command, where a typo would silently drop the args.
            (
                r#"{"base_url": "u", "endpoints": [{"protocol": "anthropic-messages",
                   "auth": {"command": {"binary": "b", "arg": ["token"]}}, "models": []}]}"#,
                "arg",
            ),
        ];
        for (json, bad_key) in cases {
            let parsed: Result<CompletionProvider, _> = serde_json::from_str(json);
            let err = parsed.expect_err("must reject").to_string();
            assert!(err.contains(bad_key), "error should name `{bad_key}`; got {err}");
        }
    }

    #[tokio::test]
    async fn an_unset_env_var_fails_to_resolve() {
        let method = AuthMethod::EnvVar("KERNELGUY_NO_SUCH_VAR_FOR_TEST".to_string());
        assert!(method.resolve().await.is_err());
    }
}
