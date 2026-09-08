use std::time::Duration;

use futures::Stream;
use reqwest::Client;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue, USER_AGENT};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::ai::protocol::{ToolCall, ToolDescriptor, ToolResult};

mod auth;
mod clients;
mod model;
pub mod protocol;
pub mod provider;

pub use auth::{AuthError, AuthProvider, CommandAuth, StaticAuth};
pub use clients::{
    AnthropicContentBlock, AnthropicImageSource, AnthropicMessage, AnthropicMessageRole, AnthropicMessagesClient,
    OpenaiInputContent, OpenaiItem, OpenaiMessage, OpenaiMessageRole, OpenaiResponsesClient,
};
pub use model::{AuthMethod, CompletionEndpoint, CompletionProtocol, CompletionProvider, HeaderError, Model};

/// Whether a response body indicates the OAuth access token expired — the one
/// 401 class that is *recoverable* by refreshing and retrying (vs. a genuine
/// bad-credential 401, which stays `Fatal`). Matches the Snowflake error code
/// `390318` and the human-readable message, case-insensitively.
pub(crate) fn is_expired_token_body(body: &str) -> bool {
    body.contains("390318") || body.to_ascii_lowercase().contains("token expired")
}

/// Classified completion failure. Drives the harness's retry policy: a
/// long-running agent run must not die on a 429 or a 503, but must give up
/// promptly on a 400 (bad request) it will never recover from.
///
/// - [`Self::RateLimited`]: the provider asked us to back off (HTTP 429).
///   `retry_after` carries a server-suggested delay when present; `message`
///   preserves the provider response body for diagnostics.
/// - [`Self::Transient`]: retryable — network blips, 5xx, mid-stream IO /
///   parse hiccups. Retried with exponential backoff.
/// - [`Self::Fatal`]: non-retryable — 4xx other than 429, auth failures,
///   malformed-request errors. Surfaced to the caller immediately.
#[derive(Debug, Clone)]
pub enum CompletionError {
    RateLimited {
        retry_after: Option<Duration>,
        message: String,
    },
    Transient(String),
    Fatal(String),
}

impl std::fmt::Display for CompletionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RateLimited {
                retry_after: Some(d),
                message,
            } => {
                write!(f, "rate limited (retry after {}s): {message}", d.as_secs())
            }
            Self::RateLimited {
                retry_after: None,
                message,
            } => {
                if message.is_empty() {
                    write!(f, "rate limited")
                } else {
                    write!(f, "rate limited: {message}")
                }
            }
            Self::Transient(m) => write!(f, "transient: {m}"),
            Self::Fatal(m) => write!(f, "fatal: {m}"),
        }
    }
}

impl std::error::Error for CompletionError {}

impl CompletionError {
    /// Classify a non-2xx HTTP status into the right error variant.
    /// 429 → rate-limited (honoring `Retry-After`); 5xx → transient;
    /// everything else (4xx) → fatal.
    ///
    /// Exception: a 429 whose body indicates quota/billing exhaustion is
    /// promoted to Fatal immediately — retrying won't help and the default
    /// 12× retry loop would burn several minutes pointlessly.
    pub(crate) fn from_status(status: reqwest::StatusCode, retry_after: Option<Duration>, body: &str) -> Self {
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            if is_quota_exhausted(body) {
                return Self::Fatal(format!("HTTP {status} (quota exhausted): {body}"));
            }
            Self::RateLimited {
                retry_after,
                message: format!("HTTP {status}: {body}"),
            }
        } else if status.is_server_error() {
            Self::Transient(format!("HTTP {status}: {body}"))
        } else {
            Self::Fatal(format!("HTTP {status}: {body}"))
        }
    }

    /// Heuristic: does this error look like a context-window / prompt-too-long
    /// overflow (the request's input exceeded the model's cap), as opposed to a
    /// generic 4xx? Used by the orchestrator to compact-and-retry the turn once
    /// instead of ending the run. Matches on provider error text since neither
    /// Anthropic nor `OpenAI` gives a distinct status code for it.
    ///
    /// TODO: confirm against live provider payloads and tighten — Anthropic
    /// returns 400 `invalid_request_error` with a message like "prompt is too
    /// long: N tokens > M maximum"; `OpenAI` Responses returns 400 with
    /// "...maximum context length is M tokens...". If a provider ever adds a
    /// dedicated code, match that instead of the substring set below.
    #[must_use]
    pub fn is_context_overflow(&self) -> bool {
        const NEEDLES: [&str; 7] = [
            "prompt is too long",
            "maximum context length",
            "context window",
            "context_length_exceeded",
            "too many tokens",
            "input is too long",
            "exceeds the maximum",
        ];
        let msg = match self {
            Self::Fatal(m) => m.as_str(),
            // Overflow is a 400 (Fatal); rate-limit/transient are never overflow.
            _ => return false,
        };
        let lower = msg.to_ascii_lowercase();
        NEEDLES.iter().any(|n| lower.contains(n))
    }
}

/// True if a 429 body signals quota/billing exhaustion rather than a
/// transient rate limit. These won't resolve by waiting, so the caller
/// should promote the error to Fatal instead of retrying.
fn is_quota_exhausted(body: &str) -> bool {
    let lower = body.to_lowercase();
    lower.contains("insufficient_quota")
        || lower.contains("quota exceeded")
        || lower.contains("exceeded your current quota")
        || lower.contains("billing")
        || lower.contains("payment")
}

/// Parse a `Retry-After` header expressed in integer seconds. The HTTP-date
/// form is ignored (returns `None`); our backoff falls back to exponential
/// delay in that case.
pub(crate) fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

/// A note naming a redirect's target, empty for any non-3xx status.
///
/// Redirects are disabled (see [`build_http_client`]), so a 3xx arrives as a
/// response and `from_status` classifies it `Fatal`. A 301/302 body is normally
/// empty, which would leave the operator staring at `HTTP 301 Moved Permanently: `
/// with nothing to act on — and no hint that not following was deliberate. Must be
/// called before the body is consumed, since it reads the headers.
pub(crate) fn redirect_note(status: reqwest::StatusCode, headers: &HeaderMap) -> String {
    if !status.is_redirection() {
        return String::new();
    }
    let target = headers
        .get(reqwest::header::LOCATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("no Location header");
    format!(" (not followed — redirects are disabled; Location: {target}. Point base_url at the final host.)")
}

/// Build the shared `reqwest` client both protocols use: a `kernelguy`
/// user-agent and JSON content-type, plus whatever `configured` the provider file
/// declared (see [`CompletionProvider::headers_for`]).
///
/// `configured` is applied last, so a provider may override the user-agent.
///
/// This function merges whatever it is handed; the reserved-name rule lives in
/// [`CompletionProvider::headers_for`], which is what every production caller goes
/// through. Hand-building a map here bypasses that check — the tests below do
/// exactly that on purpose.
pub(crate) fn build_http_client(configured: &HeaderMap) -> Result<Client, String> {
    http_client_builder(configured)
        .build()
        .map_err(|e| format!("http client build failed: {e}"))
}

/// The builder behind [`build_http_client`], split out so a test can bypass the
/// ambient proxy while still exercising the real header and redirect setup.
///
/// Tests bind a loopback listener, and reqwest honours `http_proxy` with no
/// loopback exemption (hyper-util's matcher has none), so on a host that sets it —
/// a corporate cluster, typically — a test request would go to the proxy instead of
/// the listener. Production must keep honouring the proxy; only the tests add
/// `.no_proxy()`.
fn http_client_builder(configured: &HeaderMap) -> reqwest::ClientBuilder {
    let mut default_headers = HeaderMap::new();
    default_headers.insert(USER_AGENT, HeaderValue::from_static("kernelguy"));
    default_headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    for (name, value) in configured {
        default_headers.insert(name.clone(), value.clone());
    }
    Client::builder()
        .default_headers(default_headers)
        // Redirects are off, which matters now that headers are user-supplied.
        // reqwest's default policy follows up to 10 hops and strips only
        // Authorization / Cookie / Proxy-Authorization / WWW-Authenticate across an
        // origin change — so a configured header holding a secret would be resent
        // to whatever host a 3xx names. A completions endpoint has no reason to
        // redirect; treating one as a response (surfacing the 3xx status) beats
        // forwarding credentials to an unvetted host.
        .redirect(reqwest::redirect::Policy::none())
}

/// Pull the next complete SSE event's `data:` payload out of `buffer`,
/// draining it as we go. Events are `\n\n`-delimited; within an event we
/// concatenate every `data:` line (comments and blank lines are skipped) and
/// return their `\n`-joined payload. Returns `None` when the buffer doesn't
/// yet hold a full event — the caller should pump more bytes and retry.
///
/// Shared by both providers; each interprets the payload itself (`OpenAI`
/// checks for the `[DONE]` sentinel, Anthropic relies on its `message_stop`
/// event), so the framing logic lives in exactly one place.
pub(crate) fn next_sse_data(buffer: &mut String) -> Option<String> {
    while let Some(idx) = buffer.find("\n\n") {
        let raw_event: String = buffer.drain(..idx.saturating_add(2)).collect();
        let mut data_lines: Vec<&str> = Vec::new();
        for line in raw_event.lines() {
            if line.is_empty() || line.starts_with(':') {
                continue;
            }
            if let Some(rest) = line.strip_prefix("data:") {
                data_lines.push(rest.trim_start());
            }
        }
        if data_lines.is_empty() {
            continue;
        }
        return Some(data_lines.join("\n"));
    }
    None
}

/// How much "thinking" / hidden reasoning the model should do before producing
/// visible output. Providers map this to their own knob:
/// - `OpenAI` Responses: `reasoning.effort` (`low` / `medium` / `high`).
///   `XHigh` and `Max` clamp down to `high` since `OpenAI` tops out there.
/// - Anthropic: `output_config.effort` (`low` / `medium` / `high` / `xhigh` / `max`),
///   or omitted entirely for `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingEffort {
    None,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

/// One item of a translated provider stream.
///
/// Protocol-agnostic streaming event. Every [`ProtocolClient`] yields a
/// stream of these; the harness matches on them generically for display and
/// uses the final [`Event::Stop`] to drive the conversation loop.
///
/// Image blocks are deliberately absent — models don't produce images, only
/// consume them. Image attachments live on input messages (provider-specific
/// content blocks) and never appear in a streaming response.
pub enum Event<M> {
    /// Visible response text. Adjacent `Delta` chunks concatenate.
    Text(BlockPhase),
    /// Internal reasoning summary (always assistant-side; visibility depends
    /// on the model + thinking effort).
    Thinking(BlockPhase),
    /// Anything that doesn't fit the above: `tool_use` args, citations,
    /// provider-specific block types. Display can show a placeholder; the
    /// harness only acts on the structured form via `Stop.tool_calls`.
    Opaque(OpaquePhase),
    /// Terminal event of the stream. Carries the aggregated assistant
    /// message (in provider wire format), the extracted tool calls (so the
    /// harness can dispatch without re-parsing), and the stop reason / usage
    /// for display + control flow. Streams MUST yield exactly one `Stop` as
    /// their final non-error item.
    Stop {
        reason: StopReason,
        usage: Usage,
        message: M,
        tool_calls: Vec<ToolCall>,
    },
}

/// Lifecycle of a content block streaming in. `Start` and `End` mark
/// boundaries so the harness can print prefix / suffix decoration; `Delta`
/// carries incremental text content.
pub enum BlockPhase {
    Start,
    Delta(String),
    End,
}

/// Lifecycle of an opaque block. Same shape as [`BlockPhase`] but `Start`
/// carries metadata so display can label it (e.g. tool name for `tool_use`).
pub enum OpaquePhase {
    Start { kind: &'static str, name: Option<String> },
    Delta(String),
    End,
}

/// Why a turn ended. Maps from Anthropic `stop_reason` and `OpenAI`
/// `finish_reason` to a unified set the harness can act on.
#[derive(Debug, Clone)]
pub enum StopReason {
    /// Model finished naturally — no further action needed.
    EndTurn,
    /// Model wants tool results before continuing. The harness must dispatch
    /// `Stop.tool_calls` and feed the results back.
    ToolUse,
    /// Hit the per-turn output token cap.
    MaxTokens,
    /// The client signalled the whole run is complete — the loop should stop,
    /// not nudge for more. Real providers never emit this (an autonomous run
    /// ends on a budget cap or interrupt); it exists for bounded/scripted
    /// clients like the headless self-test, which run a finite script and then
    /// declare completion.
    Done,
    /// Anything else (refusal, content filter, provider-specific).
    Other(String),
}

/// Token accounting for a turn, normalized across providers. Fields are zero
/// when the provider doesn't report them (e.g. `cache_write_tokens` is
/// Anthropic-only; `reasoning_tokens` is OpenAI-only).
#[derive(Debug, Clone, Copy, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_write_tokens: u32,
    pub cache_read_tokens: u32,
    pub reasoning_tokens: u32,
}

/// A streaming completion client for ONE wire protocol, in that protocol's own
/// message format.
///
/// The comptime half of the protocol axis: [`Self::Message`] is an associated
/// type, so the orchestrator monomorphizes over it and journals the transcript
/// verbatim. The runtime half is `CompletionProtocol`, which a
/// `CompletionProvider` uses to name what it serves. `main` crosses from the
/// runtime side to this one exactly once, when it picks a client.
///
/// We deliberately use `async fn` (stable since Rust 1.75 / edition 2024)
/// without an explicit `Send` bound on the returned future. The harness
/// awaits sequentially on a single task; spawning across threads isn't a
/// requirement. Lift the allow if you ever need `Send` here.
#[allow(async_fn_in_trait)]
pub trait ProtocolClient {
    /// The provider's wire-format message type (e.g. `AnthropicMessage`,
    /// `OpenaiMessage`). The harness keeps `Vec<Self::Message>` as the
    /// transcript and pushes `Event::Stop.message` after every turn.
    ///
    /// `Serialize + DeserializeOwned` are required so the AVO orchestrator
    /// can journal the live transcript to disk and reload it verbatim on
    /// `--resume` (including Anthropic thinking-block signatures, which the
    /// API re-validates on replay).
    type Message: Serialize + DeserializeOwned + Clone + Send + 'static;
    type Stream: Stream<Item = Result<Event<Self::Message>, CompletionError>> + Unpin;

    /// Sends the request to the provider and returns a translated event
    /// stream once the HTTP response is back.
    ///
    /// `tools` is the set of tools advertised to the model for this turn, already
    /// projected to descriptors by the registry — a provider describes tools, it
    /// never dispatches them, so it has no reason to see the callable side. Each
    /// provider maps a descriptor onto its native spec shape (Anthropic serializes
    /// it as-is, `OpenAI` Responses renames `input_schema` to `function.parameters`).
    /// Pass `&[]` for plain chat with no tools.
    ///
    /// `max_output_tokens` caps the response length (`max_tokens` on
    /// Anthropic, `max_output_tokens` on `OpenAI` Responses). `thinking` controls
    /// hidden reasoning depth.
    ///
    /// Returns `Err` for pre-stream failures (network errors, non-2xx HTTP),
    /// classified as [`CompletionError`] so the harness can retry the right
    /// ones. Errors that occur mid-stream are surfaced as `Err` items on the
    /// returned stream (also typed).
    async fn complete(
        &self,
        ctx: &[Self::Message],
        tools: &[ToolDescriptor],
        max_output_tokens: usize,
        thinking: ThinkingEffort,
    ) -> Result<Self::Stream, CompletionError>;

    /// Wrap a free-form user prompt as a transcript message in this
    /// provider's shape. `&self` is for ergonomic call syntax; the impl
    /// doesn't need any client state.
    fn user_message(&self, text: String) -> Self::Message;

    /// Convert a batch of [`ToolResult`]s into the message(s) that go on the
    /// wire before the next generation. Anthropic packs all results from one
    /// dispatch into ONE user message with N `tool_result` blocks; `OpenAI`
    /// emits one `function_call_output` item per result.
    fn tool_result_messages(&self, results: Vec<ToolResult>) -> Vec<Self::Message>;

    /// Fold a trailing free-form instruction (e.g. an episode-yield nudge) into
    /// the batch of tool-result messages produced by [`Self::tool_result_messages`],
    /// so it rides the same user turn instead of becoming a *second consecutive*
    /// user message. The default appends it as its own user message — fine for
    /// providers whose wire format is a flat item list (`OpenAI` Responses).
    /// Anthropic overrides this to append a text block onto the single
    /// `tool_result` user message, because the Messages API rejects a user turn
    /// that follows tool results with anything other than the model's response
    /// (the rebuilt transcript would carry consecutive user turns → fatal 400).
    fn append_instruction_to_results(&self, results: &mut Vec<Self::Message>, text: String) {
        results.push(self.user_message(text));
    }

    /// Whether this client sends the standing agent prompt through a dedicated
    /// provider field instead of embedding it in the first user message.
    fn uses_dedicated_instructions(&self) -> bool {
        false
    }

    /// Return the provider messages that are safe to replay as long-lived,
    /// model-visible history. Providers may keep their newest multimodal tool
    /// outputs intact while replacing older heavyweight artifacts with textual
    /// placeholders. The raw archive remains available elsewhere; this is the
    /// durable context used for requests, compaction, and resume.
    fn model_visible_history(&self, messages: &[Self::Message]) -> Vec<Self::Message> {
        messages.to_vec()
    }

    /// Whether `msg` may legally be the FIRST message of a verbatim "recent
    /// tail" placed directly after a freshly synthesized user summary during
    /// compaction. It must neither (a) orphan a tool result whose originating
    /// tool call would be summarized away, nor (b) produce an invalid role
    /// sequence after the summary user turn. Anthropic overrides this to require
    /// an assistant message (a `tool_result` user turn would orphan + double a
    /// user turn); `OpenAI` Responses excludes bare `function_call_output` batches.
    /// Default `true` suits flat-item providers with no pairing constraints.
    fn is_compaction_tail_start(&self, _msg: &Self::Message) -> bool {
        true
    }

    /// Inject a free-standing instruction (a ground-truth anchor, supervisor
    /// directions) into an ongoing transcript, keeping the wire format valid. The
    /// default appends a standalone user message — fine for flat-item providers.
    /// Anthropic overrides this to FOLD the text into the trailing user turn when
    /// the last message is already a user message, since a second consecutive
    /// user turn is a fatal 400 there.
    fn inject_user_note(&self, transcript: &mut Vec<Self::Message>, text: String) {
        transcript.push(self.user_message(text));
    }

    /// Cheap compaction pre-pass: erase OLD tool OUTPUTS in place (keeping the
    /// tool CALLS), protecting the newest `protect_recent_tokens` of messages.
    /// Returns an estimate of the tokens freed. Default no-op (returns 0);
    /// providers that carry heavyweight tool outputs override it. Never changes
    /// the message count, only shrinks output payloads to a placeholder.
    fn prune_old_tool_outputs(&self, _messages: &mut [Self::Message], _protect_recent_tokens: usize) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bind loopback, accept exactly one request, reply `response`, hand back the
    /// raw request bytes. `reqwest::Client` does not expose its default headers, so
    /// reading them off a socket is the only way to pin what actually ships — and it
    /// needs no mock dependency.
    async fn serve_one(response: String) -> (String, tokio::task::JoinHandle<String>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
        // Format the URL from the bound address so it is a literal 127.0.0.1:port and
        // no resolver runs. "localhost" would be a bug here: bound on IPv4, a client
        // resolving ::1 first would be refused.
        let addr = listener.local_addr().expect("local addr");
        let handle = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.expect("accept");
            // Read to end-of-headers rather than trusting a single segment.
            let mut seen = Vec::new();
            loop {
                let mut chunk = [0_u8; 1024];
                let n = sock.read(&mut chunk).await.expect("read");
                if n == 0 {
                    break;
                }
                seen.extend_from_slice(&chunk[..n]);
                if seen.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            let _ = sock.write_all(response.as_bytes()).await;
            String::from_utf8_lossy(&seen).into_owned()
        });
        (format!("http://{addr}/"), handle)
    }

    /// Await a [`serve_one`] task on a deadline, lowercased for assertions.
    ///
    /// The deadline is load-bearing, not defensive padding. `build_http_client` sets
    /// no request or connect timeout, and reqwest honours `http_proxy` while
    /// hyper-util exempts no address — not even loopback. So on a host with a proxy
    /// configured and loopback absent from `no_proxy`, the request goes to the proxy,
    /// `accept()` never returns, and since libtest has no per-test timeout, `cargo
    /// test` itself would never finish. Failing on a deadline keeps that a test
    /// failure rather than a hung suite.
    async fn captured(handle: tokio::task::JoinHandle<String>) -> String {
        tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("listener saw no request — a system proxy? check http_proxy / no_proxy")
            .expect("server task")
            .to_ascii_lowercase()
    }

    #[test]
    fn a_redirect_status_explains_itself_and_names_its_target() {
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::LOCATION,
            HeaderValue::from_static("https://elsewhere.test/v1"),
        );

        let note = redirect_note(reqwest::StatusCode::MOVED_PERMANENTLY, &headers);
        assert!(note.contains("https://elsewhere.test/v1"), "{note}");
        assert!(note.contains("redirects are disabled"), "{note}");

        // A 3xx with no Location still says why it stopped.
        let bare = redirect_note(reqwest::StatusCode::FOUND, &HeaderMap::new());
        assert!(bare.contains("no Location header"), "{bare}");

        // And nothing is appended to an ordinary failure.
        assert!(redirect_note(reqwest::StatusCode::BAD_REQUEST, &headers).is_empty());
    }

    const OK_RESPONSE: &str = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";

    /// Covers `build_http_client`'s merge loop end-to-end. Every other header test
    /// stops at [`CompletionProvider::headers_for`]'s return value, so deleting that
    /// loop used to leave the whole suite green.
    #[tokio::test]
    async fn configured_headers_reach_the_wire_and_override_the_default_user_agent() {
        let (url, server) = serve_one(OK_RESPONSE.to_string()).await;

        let mut configured = HeaderMap::new();
        configured.insert("x-tenant", HeaderValue::from_static("acme"));
        // Not reserved, and the doc comment promises it is overridable.
        configured.insert(USER_AGENT, HeaderValue::from_static("from-config"));

        let client = http_client_builder(&configured)
            .no_proxy()
            .build()
            .expect("client builds");
        let response = client.get(&url).send().await.expect("send");
        // A proxy would answer instead of our listener; catch that here rather than
        // asserting against someone else's response.
        assert!(
            response.status().is_success(),
            "reached something other than the test listener: {}",
            response.status()
        );

        let request = captured(server).await;
        assert!(
            request.contains("x-tenant: acme"),
            "configured header never reached the wire:\n{request}"
        );
        assert!(
            request.contains("user-agent: from-config"),
            "configured header did not override the built-in default:\n{request}"
        );
        assert_eq!(
            request.matches("user-agent:").count(),
            1,
            "the override was appended instead of replacing:\n{request}"
        );
    }

    /// The built-in user-agent, pinned separately. reqwest sets none of its own, so
    /// dropping ours would ship requests with no user-agent at all — and the test
    /// above cannot catch that, since it overrides the value it would check.
    #[tokio::test]
    async fn the_builtin_user_agent_ships_when_nothing_is_configured() {
        let (url, server) = serve_one(OK_RESPONSE.to_string()).await;

        let client = http_client_builder(&HeaderMap::new())
            .no_proxy()
            .build()
            .expect("client builds");
        let response = client.get(&url).send().await.expect("send");
        assert!(
            response.status().is_success(),
            "reached something other than the test listener: {}",
            response.status()
        );

        let request = captured(server).await;
        assert!(
            request.contains("user-agent: kernelguy"),
            "built-in user-agent missing:\n{request}"
        );
    }

    /// Redirects are off, so a 3xx is returned rather than chased. Without this the
    /// `Policy::none()` line could be deleted with the whole suite still green —
    /// and reqwest strips only Authorization / Cookie / Proxy-Authorization /
    /// WWW-Authenticate across an origin change, so a configured secret would ride
    /// along to whatever host the `Location` named.
    #[tokio::test]
    async fn a_redirect_is_returned_rather_than_followed() {
        let (target_url, target) = serve_one(OK_RESPONSE.to_string()).await;
        let (start_url, start) = serve_one(format!(
            "HTTP/1.1 302 Found\r\nLocation: {target_url}\r\nContent-Length: 0\r\n\r\n"
        ))
        .await;

        let mut configured = HeaderMap::new();
        configured.insert("x-secret", HeaderValue::from_static("do-not-forward"));
        let client = http_client_builder(&configured)
            .no_proxy()
            .build()
            .expect("client builds");
        let response = client.get(&start_url).send().await.expect("send");

        assert_eq!(response.status().as_u16(), 302, "the redirect was followed");
        let _ = captured(start).await;
        // Still pending => never accepted. If it had been contacted, x-secret went too.
        assert!(
            tokio::time::timeout(Duration::from_millis(500), target).await.is_err(),
            "the redirect target received a request"
        );
    }

    #[test]
    fn rate_limit_errors_preserve_provider_body_in_display() {
        let err = CompletionError::from_status(
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            Some(Duration::from_secs(7)),
            r#"{"message":"TPM exceeded; image payload too large"}"#,
        );

        assert!(matches!(err, CompletionError::RateLimited { .. }));
        let display = err.to_string();
        assert!(display.contains("retry after 7s"), "{display}");
        assert!(display.contains("HTTP 429 Too Many Requests"), "{display}");
        assert!(display.contains("TPM exceeded"), "{display}");
    }

    #[test]
    fn fatal_http_errors_still_include_provider_body() {
        let err = CompletionError::from_status(
            reqwest::StatusCode::BAD_REQUEST,
            None,
            r#"{"message":"max tokens of 128000 exceeded"}"#,
        );

        assert!(matches!(err, CompletionError::Fatal(_)));
        let display = err.to_string();
        assert!(display.contains("HTTP 400 Bad Request"), "{display}");
        assert!(display.contains("max tokens"), "{display}");
    }

    #[test]
    fn expired_token_body_is_detected_but_other_401s_are_not() {
        // The exact Snowflake shape that killed run_1784322280.
        assert!(is_expired_token_body(
            r#"{"code":"390318","message":"OAuth access token expired. [524564769114007]"}"#
        ));
        // Message match, case-insensitive, without the code.
        assert!(is_expired_token_body("Bearer token expired"));
        assert!(is_expired_token_body("TOKEN EXPIRED"));
        // A genuine bad-credential 401 must NOT be treated as recoverable.
        assert!(!is_expired_token_body(r#"{"message":"invalid authorization token"}"#));
        assert!(!is_expired_token_body("forbidden"));
    }
}
