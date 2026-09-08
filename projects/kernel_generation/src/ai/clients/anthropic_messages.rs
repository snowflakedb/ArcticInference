use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::Stream;
use futures::stream::BoxStream;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, HeaderMap};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ai::protocol::{ToolCall, ToolContent, ToolDescriptor, ToolResult};
use crate::ai::{
    AuthProvider, BlockPhase, CompletionError, Event, OpaquePhase, ProtocolClient, StopReason, ThinkingEffort, Usage,
    build_http_client, is_expired_token_body, next_sse_data, parse_retry_after, redirect_note,
};

/// The `source` of an Anthropic `image` content block.
///
/// Inner `source` payload of an `image` content block. Tagged so it
/// serializes as `{"type": "base64", "media_type": "...", "data": "..."}`
/// or `{"type": "url", "url": "..."}` per Anthropic's API. The previous
/// shape (untagged enum) was buggy: it would serialize as `{"Base64":
/// {...}}`, which the API rejects. No runtime code emitted images
/// before this commit, so the bug was latent.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicMessageRole {
    User,
    Assistant,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AnthropicMessage {
    pub role: AnthropicMessageRole,
    pub content: Vec<AnthropicContentBlock>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicCacheControl {
    Ephemeral {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ttl: Option<String>,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text {
        #[serde(default)]
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        signature: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    Image {
        source: AnthropicImageSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    /// Assistant-originated: a model-issued tool call. `input` is the (possibly
    /// empty) initial object from `content_block_start`; while streaming,
    /// `input_json_delta` events accumulate into `partial_json`, and we parse
    /// it back into `input` on `content_block_stop`. `partial_json` is local
    /// streaming state and never goes on the wire in either direction.
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
        #[serde(default, skip)]
        partial_json: String,
        /// Set (with the raw argument text) when `partial_json` failed to parse
        /// as JSON on `content_block_stop`. Streaming-only, like `partial_json`.
        #[serde(default, skip)]
        parse_error: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    /// User-originated: the response we send back for a previously emitted
    /// `ToolUse`. `is_error: true` tells the model the tool failed.
    ///
    /// `content` is always serialized as an array of inner blocks (which
    /// Anthropic accepts alongside the legacy plain-string form). Only
    /// `Text` and `Image` variants are valid here per the API; we don't
    /// statically prevent other variants since `tool_result_messages` is
    /// the only emitter and it only produces those two.
    ToolResult {
        tool_use_id: String,
        content: Vec<Self>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
enum AnthropicStopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    PauseTurn,
    Refusal,
}

#[expect(clippy::struct_field_names, reason = "field names are the provider's wire keys")]
#[derive(Deserialize, Debug, Default)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
}

#[derive(Deserialize, Debug)]
struct AnthropicMessageStart {
    usage: AnthropicUsage,
}

#[derive(Deserialize, Debug)]
struct AnthropicMessageDelta {
    stop_reason: Option<AnthropicStopReason>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
// Variant names map to the Anthropic wire tags (`text_delta`, `input_json_delta`,
// …) via `rename_all`, so the shared `Delta` suffix is load-bearing, not noise.
#[allow(clippy::enum_variant_names)]
enum AnthropicContentBlockDelta {
    TextDelta { text: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
    InputJsonDelta { partial_json: String },
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicCompletionEvent {
    MessageStart {
        message: AnthropicMessageStart,
    },
    ContentBlockStart {
        index: u32,
        content_block: AnthropicContentBlock,
    },
    ContentBlockDelta {
        index: u32,
        delta: AnthropicContentBlockDelta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: AnthropicMessageDelta,
        usage: AnthropicUsage,
    },
    MessageStop,
    Ping,
}

/// Widen a wire block `index` to a slot index.
///
/// Anthropic sends `index` as a JSON number we deserialize into `u32`. On any
/// target whose `usize` is narrower than that, an oversized index is a malformed
/// event — reported like every other malformed event, never a panic.
fn block_index(index: u32) -> Result<usize, String> {
    usize::try_from(index).map_err(|_| format!("content block index {index} does not fit this target's usize"))
}

/// Per-content-block tag we track inside the stream so each `ContentBlockStop`
/// emits the right standard `End` variant. Filled in on `ContentBlockStart`.
#[derive(Clone, Copy)]
enum BlockSlot {
    Text,
    Thinking,
    Opaque,
}

pub struct AnthropicMessagesClient {
    base_url: String,
    model: String,
    http_client: Client,
    /// When set, the `Authorization` header is sourced from here per request
    /// (rotating OAuth token); when `None`, it lives in the client's default
    /// headers (static PAT-style bearer).
    auth: Option<Arc<dyn AuthProvider>>,
}

impl AnthropicMessagesClient {
    /// Sources the `Authorization` header from an [`AuthProvider`] on every
    /// request, so a rotating credential (e.g. a refreshing Snowflake OAuth
    /// access token) stays current across a run that outlives a single access
    /// token.
    ///
    /// # Errors
    ///
    /// Returns the `reqwest` builder's error message if the shared HTTP client
    /// cannot be constructed (see [`build_http_client`]).
    pub fn with_auth_provider(
        base_url: &str,
        model: &str,
        auth: Arc<dyn AuthProvider>,
        headers: &HeaderMap,
    ) -> Result<Self, String> {
        Ok(Self {
            base_url: base_url.to_string(),
            model: model.to_string(),
            http_client: build_http_client(headers)?,
            auth: Some(auth),
        })
    }
}

/// Lift a [`ToolContent`] block into Anthropic's wire shape. Only Text
/// and Image are valid inside `tool_result.content` per the API; the
/// harness never produces any other variant.
fn tool_content_to_block(c: ToolContent) -> AnthropicContentBlock {
    match c {
        ToolContent::Text(text) => AnthropicContentBlock::Text {
            text,
            cache_control: None,
        },
        ToolContent::Image {
            media_type,
            data_base64,
        } => AnthropicContentBlock::Image {
            source: AnthropicImageSource::Base64 {
                media_type,
                data: data_base64,
            },
            cache_control: None,
        },
    }
}

fn mark_last_user_block_for_cache(messages: &[AnthropicMessage]) -> Vec<AnthropicMessage> {
    let mut cloned: Vec<AnthropicMessage> = messages.to_vec();
    let last_user = cloned.iter_mut().rev().find(|m| m.role == AnthropicMessageRole::User);
    if let Some(msg) = last_user
        && let Some(block) = msg.content.last_mut()
    {
        match block {
            AnthropicContentBlock::Text { cache_control, .. }
            | AnthropicContentBlock::Thinking { cache_control, .. }
            | AnthropicContentBlock::Image { cache_control, .. }
            | AnthropicContentBlock::ToolUse { cache_control, .. }
            | AnthropicContentBlock::ToolResult { cache_control, .. } => {
                *cache_control = Some(AnthropicCacheControl::Ephemeral { ttl: None });
            }
        }
    }
    cloned
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicThinkingConfig {
    Adaptive { display: &'static str },
}

#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
enum AnthropicEffort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

#[derive(Serialize)]
struct AnthropicOutputConfig {
    effort: AnthropicEffort,
}

#[derive(Serialize)]
struct AnthropicCompletionBody<'model, 'ctx> {
    model: &'model str,
    max_tokens: usize,
    stream: bool,
    /// Borrowed straight from the registry's projection — a `ToolDescriptor`'s
    /// field set *is* Anthropic's `tools[i]` shape, so there is no per-request
    /// conversion or schema clone.
    #[serde(skip_serializing_if = "<[ToolDescriptor]>::is_empty")]
    tools: &'ctx [ToolDescriptor],
    messages: &'ctx [AnthropicMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<AnthropicOutputConfig>,
}

/// Max image blocks kept in model-visible history. Snowflake Cortex accepts at
/// most 20 images per request, so we keep the most recent `MAX` image blocks
/// (across ALL tool results, counting individual images — not results) and
/// replace every older image with a text placeholder. 19 leaves a 1-image
/// margin under the hard cap. Images are re-viewable by path (`view`), so
/// eliding is lossless.
const MAX_MODEL_VISIBLE_IMAGES: usize = 19;

fn omitted_image_placeholder(source: &AnthropicImageSource) -> String {
    match source {
        AnthropicImageSource::Base64 { media_type, data } => format!(
            "[omitted old tool image: {media_type}, {} base64 bytes, approx {} decoded bytes — re-view the path if you still need it]",
            data.len(),
            data.len().saturating_mul(3) / 4
        ),
        AnthropicImageSource::Url { url } => {
            format!("[omitted old tool image: {url} — re-view the path if you still need it]")
        }
    }
}

/// Keep the block as-is if it isn't an image, or if the image budget isn't yet
/// spent (decrementing it); otherwise replace the image with a text
/// placeholder. Recurses one level into `ToolResult` content, where tool images
/// actually live. `remaining` is threaded newest→oldest by the caller.
fn keep_or_elide_images(block: &AnthropicContentBlock, remaining: &mut usize) -> AnthropicContentBlock {
    match block {
        AnthropicContentBlock::Image { source, .. } => {
            if *remaining > 0 {
                *remaining = remaining.saturating_sub(1);
                block.clone()
            } else {
                AnthropicContentBlock::Text {
                    text: omitted_image_placeholder(source),
                    cache_control: None,
                }
            }
        }
        AnthropicContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
            cache_control,
        } => {
            let mut inner: Vec<AnthropicContentBlock> = content
                .iter()
                .rev()
                .map(|c| keep_or_elide_images(c, remaining))
                .collect();
            inner.reverse();
            AnthropicContentBlock::ToolResult {
                tool_use_id: tool_use_id.clone(),
                content: inner,
                is_error: *is_error,
                cache_control: cache_control.clone(),
            }
        }
        other => other.clone(),
    }
}

/// Keep every message and result, but retain only the `max_images` most recent
/// image blocks across the whole history; replace all older images with text
/// placeholders. Walks newest→oldest so the freshest images survive, then
/// rebuilds in original order.
fn cap_model_visible_images(messages: &[AnthropicMessage], max_images: usize) -> Vec<AnthropicMessage> {
    let mut remaining = max_images;
    let mut out: Vec<AnthropicMessage> = messages
        .iter()
        .rev()
        .map(|message| {
            let mut content: Vec<AnthropicContentBlock> = message
                .content
                .iter()
                .rev()
                .map(|block| keep_or_elide_images(block, &mut remaining))
                .collect();
            content.reverse();
            AnthropicMessage {
                role: message.role.clone(),
                content,
            }
        })
        .collect();
    out.reverse();
    out
}

impl ProtocolClient for AnthropicMessagesClient {
    type Message = AnthropicMessage;
    type Stream = AnthropicCompletionStream;

    async fn complete(
        &self,
        ctx: &[Self::Message],
        tools: &[ToolDescriptor],
        max_output_tokens: usize,
        thinking: ThinkingEffort,
    ) -> Result<Self::Stream, CompletionError> {
        // Elide images from all but the most recent image-bearing tool results
        // (they're re-viewable by path) so history growth can't push a request
        // over the provider's per-request image / byte limits, then mark the last
        // user block for prompt caching.
        let visible = self.model_visible_history(ctx);
        let messages = mark_last_user_block_for_cache(&visible);

        // Every non-`None` effort uses the identical adaptive/summarized thinking
        // config; only the effort level varies, so map that once and reuse it.
        let effort = match thinking {
            ThinkingEffort::None => None,
            ThinkingEffort::Low => Some(AnthropicEffort::Low),
            ThinkingEffort::Medium => Some(AnthropicEffort::Medium),
            ThinkingEffort::High => Some(AnthropicEffort::High),
            ThinkingEffort::XHigh => Some(AnthropicEffort::XHigh),
            ThinkingEffort::Max => Some(AnthropicEffort::Max),
        };
        let thinking_cfg = effort
            .as_ref()
            .map(|_| AnthropicThinkingConfig::Adaptive { display: "summarized" });
        let output_cfg = effort.map(|effort| AnthropicOutputConfig { effort });

        let body = AnthropicCompletionBody {
            model: &self.model,
            max_tokens: max_output_tokens,
            stream: true,
            tools,
            messages: &messages,
            thinking: thinking_cfg,
            output_config: output_cfg,
        };

        let mut request = self.http_client.post(&self.base_url).json(&body);
        if let Some(auth) = &self.auth {
            request = request.header(AUTHORIZATION, auth.authorization(false).await?);
        }
        let res = request
            .send()
            .await
            .map_err(|e| CompletionError::Transient(format!("request failed: {e}")))?;

        if !res.status().is_success() {
            let status = res.status();
            let retry_after = parse_retry_after(res.headers());
            // Read from the headers before `text()` consumes the response.
            let redirect = redirect_note(status, res.headers());
            let body_text = res
                .text()
                .await
                .unwrap_or_else(|e| format!("(failed to read body: {e})"));
            // A 401 whose body says the OAuth token expired can beat the local
            // expiry check (clock skew / propagation), so `authorization()` handed
            // us a token the server then rejected. Force a refresh and surface a
            // Transient so the harness retries with a fresh token — a single stale
            // 401 must not kill a 10h run (run_1784322280 died here at 25.5x).
            if status == reqwest::StatusCode::UNAUTHORIZED && is_expired_token_body(&body_text) {
                if let Some(auth) = &self.auth {
                    // Renew now; the harness's retry then picks up the fresh token from
                    // the cache. The returned header is not needed here.
                    auth.authorization(true).await?;
                }
                return Err(CompletionError::Transient(format!(
                    "HTTP {status} (OAuth token expired; refreshed, retrying): {body_text}"
                )));
            }
            return Err(CompletionError::from_status(
                status,
                retry_after,
                &format!("{body_text}{redirect}"),
            ));
        }

        Ok(AnthropicCompletionStream {
            bytes: Box::pin(res.bytes_stream()),
            buffer: String::new(),
            content: Vec::new(),
            block_kinds: Vec::new(),
            stop_reason: None,
            usage: AnthropicUsage::default(),
            finalized: false,
        })
    }

    fn user_message(&self, text: String) -> Self::Message {
        AnthropicMessage {
            role: AnthropicMessageRole::User,
            content: vec![AnthropicContentBlock::Text {
                text,
                cache_control: None,
            }],
        }
    }

    fn tool_result_messages(&self, results: Vec<ToolResult>) -> Vec<Self::Message> {
        // All results from one dispatch batch coalesce into a SINGLE user
        // message with N tool_result blocks (mirrors how the assistant emitted
        // its tool_use blocks within one assistant message). OpenAI is the
        // opposite — one message per result.
        if results.is_empty() {
            return Vec::new();
        }
        let blocks: Vec<AnthropicContentBlock> = results
            .into_iter()
            .map(|r| AnthropicContentBlock::ToolResult {
                tool_use_id: r.tool_call_id,
                content: r.content.into_iter().map(tool_content_to_block).collect(),
                is_error: r.is_error.then_some(true),
                cache_control: None,
            })
            .collect();
        vec![AnthropicMessage {
            role: AnthropicMessageRole::User,
            content: blocks,
        }]
    }

    /// Keep every message/result, but retain only the most recent
    /// [`MAX_MODEL_VISIBLE_IMAGES`] image blocks — replacing older tool-result
    /// images with text placeholders. Bounds images per request under the
    /// provider cap (Snowflake Cortex: 20) and stops long runs from accumulating
    /// a huge image payload (the failure `markdown_get_section` triggered).
    /// Lossless — the agent can `view` a path again.
    fn model_visible_history(&self, messages: &[Self::Message]) -> Vec<Self::Message> {
        cap_model_visible_images(messages, MAX_MODEL_VISIBLE_IMAGES)
    }

    /// Fold the instruction into the existing tool-result user message as a
    /// trailing text block (a `tool_result` user turn may carry text blocks
    /// too), so the transcript never grows a second consecutive user turn after
    /// tool results. Falls back to a standalone user message if there are no
    /// results to attach to.
    fn append_instruction_to_results(&self, results: &mut Vec<Self::Message>, text: String) {
        match results.last_mut() {
            Some(msg) => msg.content.push(AnthropicContentBlock::Text {
                text,
                cache_control: None,
            }),
            None => results.push(self.user_message(text)),
        }
    }

    /// Safe compaction tail start = an assistant message. A `tool_result` user
    /// turn would orphan its `tool_use` (summarized away) AND double a user turn
    /// after the summary; a plain user turn would double a user turn. So only an
    /// assistant turn is safe to place right after the synthesized summary.
    fn is_compaction_tail_start(&self, msg: &Self::Message) -> bool {
        matches!(msg.role, AnthropicMessageRole::Assistant)
    }

    /// Fold the note into the trailing user turn as a text block when the last
    /// message is a user turn (a second consecutive user turn is a fatal 400 on
    /// the Messages API); otherwise append a standalone user message.
    fn inject_user_note(&self, transcript: &mut Vec<Self::Message>, text: String) {
        match transcript.last_mut() {
            Some(msg) if msg.role == AnthropicMessageRole::User => {
                msg.content.push(AnthropicContentBlock::Text {
                    text,
                    cache_control: None,
                });
            }
            _ => transcript.push(self.user_message(text)),
        }
    }

    /// Replace OLD `tool_result` output blocks with a short placeholder (keeping
    /// the assistant `tool_use` CALLS intact), protecting the newest
    /// `protect_recent_tokens`. Only commits if it would reclaim a worthwhile
    /// amount, so it never churns the transcript for a trivial gain.
    fn prune_old_tool_outputs(&self, messages: &mut [Self::Message], protect_recent_tokens: usize) -> u32 {
        const PRUNED_TOOL_OUTPUT: &str =
            "[older tool output pruned to reclaim context — re-run the tool or `view` the path to see it again]";
        const PRUNE_MIN_GAIN_TOKENS: usize = 20_000;
        const MIN_BLOCK_BYTES: usize = 200;
        let est = |m: &AnthropicMessage| serde_json::to_string(m).map_or(0, |s| s.len());
        // Protected suffix: newest messages summing to protect_recent_tokens.
        let mut protect_from = messages.len();
        let mut acc = 0usize;
        for (i, msg) in messages.iter().enumerate().rev() {
            acc = acc.saturating_add(est(msg) / 4);
            protect_from = i;
            if acc >= protect_recent_tokens {
                break;
            }
        }
        // Pass 1: find prunable tool_result blocks in the old prefix + tally gain.
        let mut candidates: Vec<(usize, usize)> = Vec::new();
        let mut freed_bytes = 0usize;
        for (mi, msg) in messages.get(..protect_from).unwrap_or_default().iter().enumerate() {
            if msg.role != AnthropicMessageRole::User {
                continue;
            }
            for (bi, block) in msg.content.iter().enumerate() {
                if let AnthropicContentBlock::ToolResult { content, .. } = block {
                    let before: usize = content
                        .iter()
                        .map(|b| serde_json::to_string(b).map_or(0, |s| s.len()))
                        .sum();
                    if before < MIN_BLOCK_BYTES {
                        continue;
                    }
                    candidates.push((mi, bi));
                    freed_bytes = freed_bytes.saturating_add(before.saturating_sub(PRUNED_TOOL_OUTPUT.len()));
                }
            }
        }
        if freed_bytes / 4 < PRUNE_MIN_GAIN_TOKENS {
            return 0;
        }
        // Pass 2: apply.
        for (mi, bi) in candidates {
            if let Some(AnthropicContentBlock::ToolResult { content, .. }) =
                messages.get_mut(mi).and_then(|msg| msg.content.get_mut(bi))
            {
                *content = vec![AnthropicContentBlock::Text {
                    text: PRUNED_TOOL_OUTPUT.to_string(),
                    cache_control: None,
                }];
            }
        }
        u32::try_from(freed_bytes / 4).unwrap_or(u32::MAX)
    }
}

/// Streams Anthropic SSE bytes, parses raw provider events, accumulates an
/// assistant message in-place, and yields provider-agnostic
/// [`Event<AnthropicMessage>`] values for display + control flow.
///
/// The stream MUST yield exactly one [`Event::Stop`] as its last non-error
/// item; all transcript-pushing decisions in the harness key off that event.
pub struct AnthropicCompletionStream {
    bytes: BoxStream<'static, reqwest::Result<Bytes>>,
    buffer: String,
    /// Aggregated content blocks. Becomes `Event::Stop.message.content` once
    /// `MessageStop` arrives.
    content: Vec<AnthropicContentBlock>,
    /// Per-block standard kind, parallel-indexed with `content`. Used at
    /// `ContentBlockStop` time to emit the right `End` variant.
    block_kinds: Vec<Option<BlockSlot>>,
    stop_reason: Option<AnthropicStopReason>,
    usage: AnthropicUsage,
    /// Once `Event::Stop` has been emitted, the stream is done; subsequent
    /// polls return `Ready(None)` even before the underlying byte stream
    /// closes.
    finalized: bool,
}

impl AnthropicCompletionStream {
    /// Pull the next complete SSE event from `buffer` and parse it. Returns
    /// `None` when the buffer doesn't contain a full event yet — the caller
    /// should pull more bytes. Comment-only / `data:`-less chunks are
    /// skipped silently.
    fn pop_event(buffer: &mut String) -> Option<Result<AnthropicCompletionEvent, String>> {
        let data = next_sse_data(buffer)?;
        Some(
            serde_json::from_str::<AnthropicCompletionEvent>(&data)
                .map_err(|e| format!("parse error: {e}\nraw: {data}")),
        )
    }

    /// Apply one provider event to the in-progress aggregate AND return the
    /// matching standardized `Event` (or `None` for events that have no
    /// observable effect, like `MessageStart` / `Ping` / `SignatureDelta`).
    ///
    /// `MessageStop` is the terminal event: it drains `self.content` /
    /// `self.usage` to construct the final `Event::Stop` and flips
    /// `self.finalized`.
    fn process(&mut self, ev: AnthropicCompletionEvent) -> Result<Option<Event<AnthropicMessage>>, String> {
        match ev {
            AnthropicCompletionEvent::MessageStart { message } => {
                self.usage = message.usage;
                Ok(None)
            }
            AnthropicCompletionEvent::Ping => Ok(None),

            AnthropicCompletionEvent::ContentBlockStart { index, content_block } => {
                self.on_content_block_start(index, content_block)
            }

            AnthropicCompletionEvent::ContentBlockDelta { index, delta } => self.on_content_block_delta(index, delta),

            AnthropicCompletionEvent::ContentBlockStop { index } => self.on_content_block_stop(index),

            AnthropicCompletionEvent::MessageDelta { delta, usage } => {
                self.stop_reason = delta.stop_reason;
                self.usage.output_tokens = usage.output_tokens;
                self.usage.cache_creation_input_tokens = usage.cache_creation_input_tokens;
                self.usage.cache_read_input_tokens = usage.cache_read_input_tokens;
                Ok(None)
            }

            AnthropicCompletionEvent::MessageStop => Ok(Some(self.on_message_stop())),
        }
    }

    /// `content_block_start`: open a new aggregate block at `index`, remember its
    /// [`BlockSlot`] so `ContentBlockStop` can emit the matching `End`, and return
    /// the standardized `Start` event (or `None` for block kinds we only record).
    fn on_content_block_start(
        &mut self,
        index: u32,
        content_block: AnthropicContentBlock,
    ) -> Result<Option<Event<AnthropicMessage>>, String> {
        let expected = self.content.len();
        if expected != block_index(index)? {
            return Err(format!("content_block_start: expected index {expected}, got {index}"));
        }
        let standard = match &content_block {
            AnthropicContentBlock::Text { .. } => {
                self.block_kinds.push(Some(BlockSlot::Text));
                Some(Event::Text(BlockPhase::Start))
            }
            AnthropicContentBlock::Thinking { .. } => {
                self.block_kinds.push(Some(BlockSlot::Thinking));
                Some(Event::Thinking(BlockPhase::Start))
            }
            AnthropicContentBlock::ToolUse { name, .. } => {
                let name = name.clone();
                self.block_kinds.push(Some(BlockSlot::Opaque));
                Some(Event::Opaque(OpaquePhase::Start {
                    kind: "tool_use",
                    name: Some(name),
                }))
            }
            // Image / ToolResult shouldn't appear in streamed assistant
            // output; if they do we record them but emit no event.
            _ => {
                self.block_kinds.push(None);
                None
            }
        };
        self.content.push(content_block);
        Ok(standard)
    }

    /// `content_block_delta`: fold the delta into the block already open at
    /// `index` and return the matching standardized `Delta` event.
    fn on_content_block_delta(
        &mut self,
        index: u32,
        delta: AnthropicContentBlockDelta,
    ) -> Result<Option<Event<AnthropicMessage>>, String> {
        let block = self
            .content
            .get_mut(block_index(index)?)
            .ok_or_else(|| format!("content_block_delta: no block at index {index}"))?;
        let standard = match (block, delta) {
            (AnthropicContentBlock::Text { text, .. }, AnthropicContentBlockDelta::TextDelta { text: d }) => {
                text.push_str(&d);
                Some(Event::Text(BlockPhase::Delta(d)))
            }
            (
                AnthropicContentBlock::Thinking { thinking, .. },
                AnthropicContentBlockDelta::ThinkingDelta { thinking: d },
            ) => {
                thinking.push_str(&d);
                Some(Event::Thinking(BlockPhase::Delta(d)))
            }
            (
                AnthropicContentBlock::Thinking { signature, .. },
                AnthropicContentBlockDelta::SignatureDelta { signature: s },
            ) => {
                // Anthropic verifies replayed thinking blocks against
                // this signature on subsequent turns; dropping it
                // results in HTTP 400 "Invalid signature in thinking
                // block". Not a streaming-display event though.
                //
                // ASSIGNMENT IS DELIBERATE — do NOT "fix" this to append.
                // The whole signature arrives in a single `signature_delta`.
                // That is undocumented, but it is what the API does (Leo,
                // 2026-08-06). pi appends; that is not evidence to the
                // contrary, just harmless there. Appending here would
                // corrupt the value the moment a block saw two deltas.
                *signature = s;
                None
            }
            (
                AnthropicContentBlock::ToolUse { partial_json, .. },
                AnthropicContentBlockDelta::InputJsonDelta { partial_json: d },
            ) => {
                partial_json.push_str(&d);
                Some(Event::Opaque(OpaquePhase::Delta(d)))
            }
            (block, delta) => {
                return Err(format!(
                    "content_block_delta: delta {delta:?} doesn't fit block {block:?} at {index}",
                ));
            }
        };
        Ok(standard)
    }

    /// `content_block_stop`: finish the block at `index` and return the matching
    /// standardized `End` event.
    fn on_content_block_stop(&mut self, index: u32) -> Result<Option<Event<AnthropicMessage>>, String> {
        // tool_use: parse the accumulated input_json_delta buffer into
        // `input` so consumers see a structured Value, not a raw string.
        if let Some(AnthropicContentBlock::ToolUse {
            input,
            partial_json,
            parse_error,
            ..
        }) = self.content.get_mut(block_index(index)?)
        {
            if !partial_json.is_empty() {
                match serde_json::from_str(partial_json) {
                    Ok(value) => *input = value,
                    // A malformed tool_use argument (e.g. the model typed
                    // `">` where JSON needs `":`) is NOT a stream error:
                    // returning Err here surfaces as a Transient failure and
                    // blind-retries — then kills — the whole run. Instead
                    // record the error + raw text and let the turn finalize
                    // normally; the harness answers this call with a teaching
                    // error tool_result so the model can recover next turn.
                    Err(e) => {
                        *parse_error = Some(format!("{e} (raw arguments: {partial_json})"));
                        *input = Value::Object(serde_json::Map::new());
                    }
                }
            }
            partial_json.clear();
        }
        let standard = match self.block_kinds.get(block_index(index)?).copied().flatten() {
            Some(BlockSlot::Text) => Some(Event::Text(BlockPhase::End)),
            Some(BlockSlot::Thinking) => Some(Event::Thinking(BlockPhase::End)),
            Some(BlockSlot::Opaque) => Some(Event::Opaque(OpaquePhase::End)),
            None => None,
        };
        Ok(standard)
    }

    /// `message_stop`: the terminal event. Drains `self.content` / `self.usage`
    /// into the final `Event::Stop` and flips `self.finalized`.
    fn on_message_stop(&mut self) -> Event<AnthropicMessage> {
        let reason = match self.stop_reason.take() {
            Some(AnthropicStopReason::EndTurn) => StopReason::EndTurn,
            Some(AnthropicStopReason::ToolUse) => StopReason::ToolUse,
            Some(AnthropicStopReason::MaxTokens) => StopReason::MaxTokens,
            Some(other) => StopReason::Other(format!("{other:?}")),
            None => StopReason::Other("missing".to_string()),
        };
        let usage = Usage {
            input_tokens: self.usage.input_tokens,
            output_tokens: self.usage.output_tokens,
            cache_write_tokens: self.usage.cache_creation_input_tokens,
            cache_read_tokens: self.usage.cache_read_input_tokens,
            reasoning_tokens: 0,
        };
        let content = std::mem::take(&mut self.content);
        let tool_calls: Vec<ToolCall> = content
            .iter()
            .filter_map(|b| match b {
                AnthropicContentBlock::ToolUse {
                    id,
                    name,
                    input,
                    parse_error,
                    ..
                } => Some(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                    parse_error: parse_error.clone(),
                }),
                _ => None,
            })
            .collect();
        let message = AnthropicMessage {
            role: AnthropicMessageRole::Assistant,
            content,
        };
        self.finalized = true;
        Event::Stop {
            reason,
            usage,
            message,
            tool_calls,
        }
    }
}

impl Stream for AnthropicCompletionStream {
    type Item = Result<Event<AnthropicMessage>, CompletionError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            if this.finalized {
                return Poll::Ready(None);
            }

            // Try to advance the SSE buffer. If we get an event, process it
            // and either yield a standard event or loop to the next one (for
            // events that translate to nothing observable). Mid-stream parse
            // / IO errors are transient: retrying re-requests the whole turn.
            if let Some(parse) = Self::pop_event(&mut this.buffer) {
                let provider_event = match parse {
                    Ok(e) => e,
                    Err(e) => return Poll::Ready(Some(Err(CompletionError::Transient(e)))),
                };
                match this.process(provider_event) {
                    Ok(Some(std_ev)) => return Poll::Ready(Some(Ok(std_ev))),
                    Ok(None) => continue,
                    Err(e) => return Poll::Ready(Some(Err(CompletionError::Transient(e)))),
                }
            }

            // Need more bytes from the socket.
            match this.bytes.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => match std::str::from_utf8(&chunk) {
                    Ok(s) => this.buffer.push_str(s),
                    Err(e) => return Poll::Ready(Some(Err(CompletionError::Transient(e.to_string())))),
                },
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(CompletionError::Transient(e.to_string())))),
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{Tool, ToolOutput};
    use schemars::JsonSchema;
    use serde::Deserialize;
    use serde_json::json;

    #[derive(Deserialize, JsonSchema)]
    struct PingArgs {
        /// Bytes of payload to echo back.
        #[allow(dead_code)]
        payload: String,
    }

    struct Ping;
    impl Tool for Ping {
        type Args = PingArgs;
        const NAME: &'static str = "ping";
        const DESCRIPTION: &'static str = "Round-trip test.";

        async fn call(&self, _args: PingArgs) -> ToolOutput {
            Ok(String::new().into())
        }
    }

    #[test]
    fn tool_definition_wire_format() {
        let value = serde_json::to_value(Ping.descriptor()).unwrap();

        // Exact pin: this is the object the model actually receives at `tools[i]`,
        // captured before `AnthropicToolDefinition` was replaced by serializing
        // `ToolDescriptor` directly. Any drift here is a change to the prompt.
        assert_eq!(
            value,
            json!({
                "name": "ping",
                "description": "Round-trip test.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "payload": { "type": "string", "description": "Bytes of payload to echo back." }
                    },
                    "required": ["payload"]
                }
            })
        );
    }

    #[test]
    fn tool_result_messages_emit_array_form_with_text_and_image() {
        // Single ToolResult mixing a text block and an image block must
        // serialize as ONE user message containing one tool_result block
        // whose `content` is an array of `{type:text}` and `{type:image}`
        // sub-blocks (the array form Anthropic accepts alongside the
        // legacy plain-string form). This is the wire shape pdf_view_page
        // depends on.
        let client_messages = AnthropicMessagesClient {
            base_url: String::new(),
            model: String::new(),
            http_client: Client::new(),
            auth: None,
        }
        .tool_result_messages(vec![ToolResult {
            tool_call_id: "tcid_1".to_string(),
            content: vec![
                ToolContent::Text("page 0 rendered".to_string()),
                ToolContent::image_png("AAAA".to_string()),
            ],
            is_error: false,
        }]);

        let value = serde_json::to_value(&client_messages).unwrap();
        let messages = value.as_array().expect("array");
        assert_eq!(messages.len(), 1, "one user message per dispatch batch");
        let msg = &messages[0];
        assert_eq!(msg["role"], "user");
        let content = msg["content"].as_array().expect("content array");
        assert_eq!(content.len(), 1, "one tool_result block per ToolResult");
        let tr = &content[0];
        assert_eq!(tr["type"], "tool_result");
        assert_eq!(tr["tool_use_id"], "tcid_1");
        let inner = tr["content"].as_array().expect("inner array form");
        assert_eq!(inner.len(), 2, "one text + one image block");
        assert_eq!(inner[0]["type"], "text");
        assert_eq!(inner[0]["text"], "page 0 rendered");
        assert_eq!(inner[1]["type"], "image");
        assert_eq!(inner[1]["source"]["type"], "base64");
        assert_eq!(inner[1]["source"]["media_type"], "image/png");
        assert_eq!(inner[1]["source"]["data"], "AAAA");
    }

    #[test]
    fn tool_result_messages_emit_is_error_only_when_set() {
        let client = AnthropicMessagesClient {
            base_url: String::new(),
            model: String::new(),
            http_client: Client::new(),
            auth: None,
        };
        let ok = client.tool_result_messages(vec![ToolResult {
            tool_call_id: "a".to_string(),
            content: vec![ToolContent::Text("ok".to_string())],
            is_error: false,
        }]);
        let err = client.tool_result_messages(vec![ToolResult {
            tool_call_id: "b".to_string(),
            content: vec![ToolContent::Text("boom".to_string())],
            is_error: true,
        }]);
        let ok_v = serde_json::to_value(&ok).unwrap();
        let err_v = serde_json::to_value(&err).unwrap();
        assert!(
            ok_v[0]["content"][0].get("is_error").is_none(),
            "is_error must be omitted when false: {ok_v}"
        );
        assert_eq!(err_v[0]["content"][0]["is_error"], true);
    }

    /// The AVO `--resume` journal serializes the live transcript and reloads
    /// it verbatim. Anthropic re-validates the cryptographic `signature` on
    /// any replayed thinking block (HTTP 400 "Invalid signature in thinking
    /// block" otherwise), and `tool_use` `input` must survive too. This pins
    /// the Serialize+Deserialize round-trip for an assistant turn carrying
    /// all three block kinds.
    #[test]
    fn message_round_trips_thinking_signature_and_tool_use() {
        let original = AnthropicMessage {
            role: AnthropicMessageRole::Assistant,
            content: vec![
                AnthropicContentBlock::Thinking {
                    thinking: "let me reason about this".to_string(),
                    signature: "SIG_abc123==".to_string(),
                    cache_control: None,
                },
                AnthropicContentBlock::Text {
                    text: "Here is my answer.".to_string(),
                    cache_control: None,
                },
                AnthropicContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "evaluate".to_string(),
                    input: json!({ "stage": "full" }),
                    partial_json: String::new(),
                    parse_error: None,
                    cache_control: None,
                },
            ],
        };

        let wire = serde_json::to_string(&original).expect("serialize");
        let restored: AnthropicMessage = serde_json::from_str(&wire).expect("deserialize");

        assert_eq!(restored.role, AnthropicMessageRole::Assistant);
        match &restored.content[0] {
            AnthropicContentBlock::Thinking {
                thinking, signature, ..
            } => {
                assert_eq!(thinking, "let me reason about this");
                assert_eq!(
                    signature, "SIG_abc123==",
                    "thinking signature must survive the journal round-trip"
                );
            }
            other => panic!("expected Thinking block, got {other:?}"),
        }
        match &restored.content[2] {
            AnthropicContentBlock::ToolUse {
                name,
                input,
                partial_json,
                ..
            } => {
                assert_eq!(name, "evaluate");
                assert_eq!(input, &json!({ "stage": "full" }));
                assert!(
                    partial_json.is_empty(),
                    "streaming-only field resets to empty on reload"
                );
            }
            other => panic!("expected ToolUse block, got {other:?}"),
        }
    }

    #[test]
    fn model_visible_history_caps_images_newest_first() {
        // One image per result across 3 results; cap at 2 keeps the 2 newest
        // images and elides the oldest — every RESULT is retained either way.
        let img_msg = |data: &str| AnthropicMessage {
            role: AnthropicMessageRole::User,
            content: vec![AnthropicContentBlock::ToolResult {
                tool_use_id: "t".to_string(),
                content: vec![AnthropicContentBlock::Image {
                    source: AnthropicImageSource::Base64 {
                        media_type: "image/png".to_string(),
                        data: data.to_string(),
                    },
                    cache_control: None,
                }],
                is_error: None,
                cache_control: None,
            }],
        };
        let messages = vec![img_msg("OLD"), img_msg("NEWER"), img_msg("NEWEST")];
        let out = cap_model_visible_images(&messages, 2);
        assert_eq!(out.len(), 3, "all results retained");

        let inner = |m: &AnthropicMessage| match &m.content[0] {
            AnthropicContentBlock::ToolResult { content, .. } => content[0].clone(),
            other => panic!("expected tool_result, got {other:?}"),
        };
        assert!(
            matches!(inner(&out[0]), AnthropicContentBlock::Text { text, .. } if text.contains("omitted old tool image")),
            "oldest image elided"
        );
        assert!(
            matches!(inner(&out[1]), AnthropicContentBlock::Image { .. }),
            "2nd-newest kept"
        );
        assert!(
            matches!(inner(&out[2]), AnthropicContentBlock::Image { .. }),
            "newest kept"
        );
    }

    #[test]
    fn model_visible_history_counts_images_within_one_result() {
        // A single result carrying 3 images; cap at 2 keeps the 2 most recent
        // images IN THAT RESULT and elides the first — proves per-image counting.
        let img = |data: &str| AnthropicContentBlock::Image {
            source: AnthropicImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: data.to_string(),
            },
            cache_control: None,
        };
        let messages = vec![AnthropicMessage {
            role: AnthropicMessageRole::User,
            content: vec![AnthropicContentBlock::ToolResult {
                tool_use_id: "t".to_string(),
                content: vec![img("A"), img("B"), img("C")],
                is_error: None,
                cache_control: None,
            }],
        }];
        let out = cap_model_visible_images(&messages, 2);
        let content = match &out[0].content[0] {
            AnthropicContentBlock::ToolResult { content, .. } => content,
            other => panic!("expected tool_result, got {other:?}"),
        };
        // First (oldest) image elided; the two most recent kept, in order.
        assert!(matches!(&content[0], AnthropicContentBlock::Text { text, .. } if text.contains("omitted")));
        assert!(matches!(&content[1], AnthropicContentBlock::Image { .. }));
        assert!(matches!(&content[2], AnthropicContentBlock::Image { .. }));
    }

    #[test]
    fn model_visible_history_noop_when_under_budget() {
        // Text-only tool results are passed through unchanged; images under the
        // budget are kept.
        let text_msg = AnthropicMessage {
            role: AnthropicMessageRole::User,
            content: vec![AnthropicContentBlock::ToolResult {
                tool_use_id: "t".to_string(),
                content: vec![AnthropicContentBlock::Text {
                    text: "plain result".to_string(),
                    cache_control: None,
                }],
                is_error: None,
                cache_control: None,
            }],
        };
        let out = cap_model_visible_images(std::slice::from_ref(&text_msg), MAX_MODEL_VISIBLE_IMAGES);
        assert!(matches!(
            &out[0].content[0],
            AnthropicContentBlock::ToolResult { content, .. }
                if matches!(&content[0], AnthropicContentBlock::Text { text, .. } if text == "plain result")
        ));
    }

    fn bare_client() -> AnthropicMessagesClient {
        AnthropicMessagesClient {
            base_url: String::new(),
            model: String::new(),
            http_client: Client::new(),
            auth: None,
        }
    }

    fn user_tool_result() -> AnthropicMessage {
        AnthropicMessage {
            role: AnthropicMessageRole::User,
            content: vec![AnthropicContentBlock::ToolResult {
                tool_use_id: "t".to_string(),
                content: vec![AnthropicContentBlock::Text {
                    text: "result".to_string(),
                    cache_control: None,
                }],
                is_error: None,
                cache_control: None,
            }],
        }
    }

    fn assistant_text() -> AnthropicMessage {
        AnthropicMessage {
            role: AnthropicMessageRole::Assistant,
            content: vec![AnthropicContentBlock::Text {
                text: "thinking out loud".to_string(),
                cache_control: None,
            }],
        }
    }

    #[test]
    fn only_assistant_messages_are_safe_compaction_tail_starts() {
        let client = bare_client();
        // An assistant turn is a safe tail start; a tool_result user turn is not
        // (it would orphan its tool_use and double a user turn after the summary).
        assert!(client.is_compaction_tail_start(&assistant_text()));
        assert!(!client.is_compaction_tail_start(&user_tool_result()));
        assert!(!client.is_compaction_tail_start(&client.user_message("hi".to_string())));
    }

    #[test]
    fn inject_user_note_folds_into_trailing_user_turn() {
        // Injecting after a user (tool_result) turn must NOT create a second
        // consecutive user turn (a fatal 400); it folds in as a text block.
        let client = bare_client();
        let mut transcript = vec![assistant_text(), user_tool_result()];
        let before_len = transcript.len();
        client.inject_user_note(&mut transcript, "GROUND TRUTH: best 1.5x".to_string());
        assert_eq!(transcript.len(), before_len, "must fold, not append a new turn");
        let last = transcript.last().unwrap();
        assert_eq!(last.role, AnthropicMessageRole::User);
        assert_eq!(last.content.len(), 2, "original tool_result block + folded note");
        assert!(matches!(
            &last.content[1],
            AnthropicContentBlock::Text { text, .. } if text.contains("GROUND TRUTH")
        ));
    }

    #[test]
    fn inject_user_note_appends_after_assistant_turn() {
        // After an assistant turn, a standalone user turn is valid, so append one.
        let client = bare_client();
        let mut transcript = vec![assistant_text()];
        client.inject_user_note(&mut transcript, "note".to_string());
        assert_eq!(transcript.len(), 2);
        assert_eq!(transcript[1].role, AnthropicMessageRole::User);
    }
}
