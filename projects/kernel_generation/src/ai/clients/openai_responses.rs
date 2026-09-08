use std::collections::{BTreeMap, VecDeque, btree_map};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use bytes::Bytes;
use futures::Stream;
use futures::stream::BoxStream;
use image::GenericImageView;
use image::ImageFormat;
use image::ImageReader;
use image::imageops::FilterType;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, HeaderMap};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ai::protocol::{ToolCall, ToolContent, ToolDescriptor, ToolResult};
use crate::ai::{
    AuthProvider, BlockPhase, CompletionError, Event, OpaquePhase, ProtocolClient, StopReason, ThinkingEffort, Usage,
    build_http_client, is_expired_token_body, next_sse_data, parse_retry_after, redirect_note,
};

const TOOL_IMAGE_MAX_DIMENSION: u32 = 2048;
const DEFAULT_IMAGE_DETAIL: &str = "high";
const MODEL_VISIBLE_RECENT_TOOL_IMAGE_OUTPUTS: usize = 2;

/// A batch of `OpenAI` Responses items making up one transcript turn.
///
/// One journaled `OpenAI` Responses transcript turn. The harness wants one
/// provider message per assistant turn, while the Responses API wants an input
/// array of items, so this type is intentionally a small batch wrapper.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenaiMessage {
    pub items: Vec<OpenaiItem>,
}

impl OpenaiMessage {
    fn one(item: OpenaiItem) -> Self {
        Self { items: vec![item] }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenaiItem {
    Message {
        role: OpenaiMessageRole,
        content: Vec<OpenaiInputContent>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: OpenaiFunctionCallOutput,
    },
    Reasoning {
        #[serde(default, skip_serializing)]
        id: String,
        #[serde(default)]
        summary: Vec<Value>,
        #[serde(default, skip_serializing_if = "should_skip_reasoning_content")]
        content: Option<Vec<Value>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
}

#[expect(clippy::ref_option, reason = "signature dictated by serde's skip_serializing_if")]
fn should_skip_reasoning_content(content: &Option<Vec<Value>>) -> bool {
    content.as_ref().is_none_or(|items| {
        items.iter().any(|item| {
            item.get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| kind == "reasoning_text")
        })
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum OpenaiFunctionCallOutput {
    Text(String),
    Content(Vec<OpenaiInputContent>),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OpenaiMessageRole {
    User,
    Developer,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum OpenaiInputContent {
    InputText {
        text: String,
    },
    InputImage {
        image_url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    OutputText {
        text: String,
    },
}

impl OpenaiInputContent {
    fn from_wire_value(value: &Value) -> Option<Self> {
        let kind = value.get("type")?.as_str()?;
        match kind {
            "input_text" => Some(Self::InputText {
                text: value.get("text")?.as_str()?.to_string(),
            }),
            "output_text" => Some(Self::OutputText {
                text: value.get("text")?.as_str()?.to_string(),
            }),
            "input_image" => Some(Self::InputImage {
                image_url: value.get("image_url")?.as_str()?.to_string(),
                detail: value.get("detail").and_then(Value::as_str).map(str::to_string),
            }),
            _ => None,
        }
    }
}

fn content_text(content: &[OpenaiInputContent]) -> String {
    content
        .iter()
        .filter_map(|part| match part {
            OpenaiInputContent::InputText { text } | OpenaiInputContent::OutputText { text } => Some(text.as_str()),
            OpenaiInputContent::InputImage { .. } => None,
        })
        .collect()
}

#[derive(Debug, Serialize)]
struct ResponsesRequest<'model> {
    model: &'model str,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<&'model str>,
    input: Vec<OpenaiItem>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ResponsesTool>,
    tool_choice: &'static str,
    parallel_tool_calls: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ResponsesReasoning>,
    store: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    include: Vec<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<&'model str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<&'model str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<ResponsesTextControls>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    client_metadata: BTreeMap<String, String>,
    max_output_tokens: usize,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct ResponsesTool {
    #[serde(rename = "type")]
    kind: &'static str,
    name: String,
    description: String,
    parameters: Value,
    strict: bool,
}

impl ResponsesTool {
    /// Responses renames `input_schema` to `parameters` and adds its own
    /// `type`/`strict`, so unlike Anthropic it keeps a wrapper. This is also where
    /// any future provider-specific per-tool flag belongs.
    fn from_descriptor(descriptor: &ToolDescriptor) -> Self {
        Self {
            kind: "function",
            name: descriptor.name.to_string(),
            description: descriptor.description.to_string(),
            parameters: descriptor.input_schema.clone(),
            strict: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
struct ResponsesReasoning {
    effort: ResponsesReasoningEffort,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<ResponsesReasoningSummary>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum ResponsesReasoningSummary {
    Auto,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
struct ResponsesTextControls {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum ResponsesReasoningEffort {
    Low,
    Medium,
    High,
}

impl From<ThinkingEffort> for ResponsesReasoningEffort {
    fn from(effort: ThinkingEffort) -> Self {
        match effort {
            ThinkingEffort::None | ThinkingEffort::Low => Self::Low,
            ThinkingEffort::Medium => Self::Medium,
            ThinkingEffort::High | ThinkingEffort::XHigh | ThinkingEffort::Max => Self::High,
        }
    }
}

fn reasoning(thinking: ThinkingEffort) -> Option<ResponsesReasoning> {
    (thinking != ThinkingEffort::None).then(|| ResponsesReasoning {
        effort: thinking.into(),
        summary: Some(ResponsesReasoningSummary::Auto),
    })
}

#[derive(Debug, Default)]
struct RequestDiagnostics {
    input_items: usize,
    tool_defs: usize,
    function_outputs: usize,
    images: usize,
    message_images: usize,
    function_output_images: usize,
    data_url_bytes: usize,
    image_bytes: usize,
    request_json_bytes: usize,
    message_json_bytes: usize,
    function_output_json_bytes: usize,
    function_call_json_bytes: usize,
    reasoning_json_bytes: usize,
}

impl RequestDiagnostics {
    fn new(input: &[OpenaiItem], tools: &[ResponsesTool]) -> Self {
        let mut diag = Self {
            input_items: input.len(),
            tool_defs: tools.len(),
            ..Self::default()
        };
        for item in input {
            let item_json_bytes = serde_json::to_vec(item).map_or(0, |bytes| bytes.len());
            match item {
                OpenaiItem::FunctionCallOutput { output, .. } => {
                    diag.function_outputs = diag.function_outputs.saturating_add(1);
                    diag.function_output_json_bytes = diag.function_output_json_bytes.saturating_add(item_json_bytes);
                    diag.count_function_output_images(output);
                }
                OpenaiItem::Message { content, .. } => {
                    diag.message_json_bytes = diag.message_json_bytes.saturating_add(item_json_bytes);
                    for part in content {
                        diag.count_image_part(part, ImageLocation::Message);
                    }
                }
                OpenaiItem::FunctionCall { .. } => {
                    diag.function_call_json_bytes = diag.function_call_json_bytes.saturating_add(item_json_bytes);
                }
                OpenaiItem::Reasoning { .. } => {
                    diag.reasoning_json_bytes = diag.reasoning_json_bytes.saturating_add(item_json_bytes);
                }
            }
        }
        diag.request_json_bytes = serde_json::to_vec(input).map_or(0, |bytes| bytes.len());
        diag
    }

    fn count_function_output_images(&mut self, output: &OpenaiFunctionCallOutput) {
        if let OpenaiFunctionCallOutput::Content(content) = output {
            for part in content {
                self.count_image_part(part, ImageLocation::FunctionOutput);
            }
        }
    }

    fn count_image_part(&mut self, part: &OpenaiInputContent, location: ImageLocation) {
        let OpenaiInputContent::InputImage { image_url, .. } = part else {
            return;
        };
        self.images = self.images.saturating_add(1);
        match location {
            ImageLocation::Message => self.message_images = self.message_images.saturating_add(1),
            ImageLocation::FunctionOutput => {
                self.function_output_images = self.function_output_images.saturating_add(1);
            }
        }
        self.data_url_bytes = self.data_url_bytes.saturating_add(image_url.len());
        if let Some((_, data)) = image_url.rsplit_once(',') {
            self.image_bytes = self.image_bytes.saturating_add(data.len().saturating_mul(3) / 4);
        }
    }
}

#[derive(Clone, Copy)]
enum ImageLocation {
    Message,
    FunctionOutput,
}

impl std::fmt::Display for RequestDiagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "responses request diagnostics: input_items={}, function_outputs={}, tool_defs={}, images={} (messages={}, function_outputs={}), data_url_bytes={}, approx_image_bytes={}, request_json_bytes={}, json_bytes_by_item={{message:{}, function_call:{}, function_call_output:{}, reasoning:{}}}",
            self.input_items,
            self.function_outputs,
            self.tool_defs,
            self.images,
            self.message_images,
            self.function_output_images,
            self.data_url_bytes,
            self.image_bytes,
            self.request_json_bytes,
            self.message_json_bytes,
            self.function_call_json_bytes,
            self.function_output_json_bytes,
            self.reasoning_json_bytes
        )
    }
}

pub struct OpenaiResponsesClient {
    base_url: String,
    model: String,
    instructions: Option<String>,
    parallel_tool_calls: bool,
    prompt_cache_key: Option<String>,
    client_metadata: BTreeMap<String, String>,
    http_client: Client,
    auth: Option<Arc<dyn AuthProvider>>,
}

impl OpenaiResponsesClient {
    /// Build a client that sources its `Authorization` header from `auth` on every
    /// request, so a rotating credential stays current across a long run.
    ///
    /// # Errors
    ///
    /// Returns the `reqwest` builder's error message if the shared HTTP client
    /// cannot be constructed (see [`build_http_client`]).
    pub fn with_auth_provider(
        base_url: impl Into<String>,
        model: impl Into<String>,
        auth: Arc<dyn AuthProvider>,
        headers: &HeaderMap,
    ) -> Result<Self, String> {
        Ok(Self {
            base_url: base_url.into(),
            model: model.into(),
            instructions: None,
            parallel_tool_calls: true,
            prompt_cache_key: None,
            client_metadata: BTreeMap::from([("client".to_string(), "kernelguy".to_string())]),
            http_client: build_http_client(headers)?,
            auth: Some(auth),
        })
    }

    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    #[must_use]
    pub const fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = enabled;
        self
    }

    #[must_use]
    pub fn with_prompt_cache_key(mut self, prompt_cache_key: impl Into<String>) -> Self {
        self.prompt_cache_key = Some(prompt_cache_key.into());
        self
    }

    #[must_use]
    pub fn without_prompt_cache_key(mut self) -> Self {
        self.prompt_cache_key = None;
        self
    }

    #[must_use]
    pub fn with_client_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.client_metadata.insert(key.into(), value.into());
        self
    }
}

impl ProtocolClient for OpenaiResponsesClient {
    type Message = OpenaiMessage;
    type Stream = ResponsesStream;

    async fn complete(
        &self,
        ctx: &[Self::Message],
        tools: &[ToolDescriptor],
        max_output_tokens: usize,
        thinking: ThinkingEffort,
    ) -> Result<Self::Stream, CompletionError> {
        let model_ctx = self.model_visible_history(ctx);
        let input = flatten_messages(&model_ctx);
        let tools: Vec<ResponsesTool> = tools.iter().map(ResponsesTool::from_descriptor).collect();
        let diagnostics = RequestDiagnostics::new(&input, &tools);
        let body = ResponsesRequest {
            model: &self.model,
            instructions: self.instructions.as_deref(),
            input,
            tools,
            tool_choice: "auto",
            parallel_tool_calls: self.parallel_tool_calls,
            reasoning: reasoning(thinking),
            store: false,
            include: if thinking == ThinkingEffort::None {
                Vec::new()
            } else {
                vec!["reasoning.encrypted_content"]
            },
            service_tier: None,
            prompt_cache_key: self.prompt_cache_key.as_deref(),
            text: None,
            client_metadata: self.client_metadata.clone(),
            max_output_tokens,
            stream: true,
        };

        let mut request = self.http_client.post(&self.base_url).json(&body);
        if let Some(auth) = &self.auth {
            request = request.header(AUTHORIZATION, auth.authorization(false).await?);
        }
        let response = request
            .send()
            .await
            .map_err(|e| CompletionError::Transient(format!("request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let retry_after = parse_retry_after(response.headers());
            // Read from the headers before `text()` consumes the response.
            let redirect = redirect_note(status, response.headers());
            let body = response
                .text()
                .await
                .unwrap_or_else(|e| format!("(failed to read body: {e})"));
            // Recoverable OAuth-expired 401: force a refresh + retry (see the
            // matching path in the Anthropic client) rather than dying Fatal.
            if status == reqwest::StatusCode::UNAUTHORIZED && is_expired_token_body(&body) {
                if let Some(auth) = &self.auth {
                    // Renew now; the harness's retry then picks up the fresh token from
                    // the cache. The returned header is not needed here.
                    auth.authorization(true).await?;
                }
                return Err(CompletionError::Transient(format!(
                    "HTTP {status} (OAuth token expired; refreshed, retrying): {body}"
                )));
            }
            return Err(CompletionError::from_status(
                status,
                retry_after,
                &format!("{body}{redirect}\n{diagnostics}"),
            ));
        }

        Ok(ResponsesStream::new(Box::pin(response.bytes_stream())))
    }

    fn user_message(&self, text: String) -> Self::Message {
        OpenaiMessage::one(OpenaiItem::Message {
            role: OpenaiMessageRole::User,
            content: vec![OpenaiInputContent::InputText { text }],
        })
    }

    fn tool_result_messages(&self, results: Vec<ToolResult>) -> Vec<Self::Message> {
        results
            .into_iter()
            .map(|result| OpenaiMessage::one(tool_result_item(result)))
            .collect()
    }

    fn uses_dedicated_instructions(&self) -> bool {
        self.instructions.is_some()
    }

    fn model_visible_history(&self, messages: &[Self::Message]) -> Vec<Self::Message> {
        sanitize_old_tool_images(messages, MODEL_VISIBLE_RECENT_TOOL_IMAGE_OUTPUTS)
    }

    /// Safe compaction tail start = any turn that is NOT a bare
    /// `function_call_output` batch. Starting the tail on a tool output would
    /// orphan it from the `function_call` that produced it (summarized away).
    /// Assistant turns (which carry the `function_call` and its reasoning
    /// together) and user turns are fine — Responses is a flat item list, so
    /// consecutive user items after the summary are legal.
    fn is_compaction_tail_start(&self, msg: &Self::Message) -> bool {
        !msg.items.is_empty()
            && !msg
                .items
                .iter()
                .all(|i| matches!(i, OpenaiItem::FunctionCallOutput { .. }))
    }

    /// Replace OLD `function_call_output` payloads with a short placeholder
    /// (keeping the `function_call` items intact), protecting the newest
    /// `protect_recent_tokens`. Only commits if it reclaims a worthwhile amount.
    fn prune_old_tool_outputs(&self, messages: &mut [Self::Message], protect_recent_tokens: usize) -> u32 {
        const PRUNED_TOOL_OUTPUT: &str =
            "[older tool output pruned to reclaim context — re-run the tool or `view` the path to see it again]";
        const PRUNE_MIN_GAIN_TOKENS: usize = 20_000;
        const MIN_BLOCK_BYTES: usize = 200;
        let est = |m: &OpenaiMessage| serde_json::to_string(m).map_or(0, |s| s.len());
        let mut protect_from = messages.len();
        let mut acc = 0usize;
        for (i, msg) in messages.iter().enumerate().rev() {
            acc = acc.saturating_add(est(msg) / 4);
            protect_from = i;
            if acc >= protect_recent_tokens {
                break;
            }
        }
        // Pass 1: find prunable function_call_output items + tally gain.
        let mut candidates: Vec<(usize, usize)> = Vec::new();
        let mut freed_bytes = 0usize;
        for (mi, msg) in messages.get(..protect_from).unwrap_or_default().iter().enumerate() {
            for (ii, item) in msg.items.iter().enumerate() {
                if let OpenaiItem::FunctionCallOutput { output, .. } = item {
                    let before = serde_json::to_string(output).map_or(0, |s| s.len());
                    if before < MIN_BLOCK_BYTES {
                        continue;
                    }
                    candidates.push((mi, ii));
                    freed_bytes = freed_bytes.saturating_add(before.saturating_sub(PRUNED_TOOL_OUTPUT.len()));
                }
            }
        }
        if freed_bytes / 4 < PRUNE_MIN_GAIN_TOKENS {
            return 0;
        }
        // Pass 2: apply.
        for (mi, ii) in candidates {
            if let Some(OpenaiItem::FunctionCallOutput { output, .. }) =
                messages.get_mut(mi).and_then(|msg| msg.items.get_mut(ii))
            {
                *output = OpenaiFunctionCallOutput::Text(PRUNED_TOOL_OUTPUT.to_string());
            }
        }
        u32::try_from(freed_bytes / 4).unwrap_or(u32::MAX)
    }
}

fn flatten_messages(messages: &[OpenaiMessage]) -> Vec<OpenaiItem> {
    messages
        .iter()
        .flat_map(|message| message.items.iter().cloned())
        .collect()
}

fn sanitize_old_tool_images(messages: &[OpenaiMessage], keep_recent_image_outputs: usize) -> Vec<OpenaiMessage> {
    let mut remaining = keep_recent_image_outputs;
    let mut keep_output_images = vec![false; messages.len()];
    // Zip the flags with the messages (equal lengths by construction) so the
    // "is this index in range" question can't be asked, let alone answered wrong.
    for (keep, message) in keep_output_images.iter_mut().zip(messages).rev() {
        if !message_has_function_output_image(message) {
            continue;
        }
        if remaining > 0 {
            *keep = true;
            remaining = remaining.saturating_sub(1);
        }
    }

    messages
        .iter()
        .zip(keep_output_images)
        .map(|(message, keep_images)| sanitize_message_tool_images(message, keep_images))
        .collect()
}

fn message_has_function_output_image(message: &OpenaiMessage) -> bool {
    message.items.iter().any(|item| match item {
        OpenaiItem::FunctionCallOutput { output, .. } => function_output_has_image(output),
        OpenaiItem::Message { .. } | OpenaiItem::FunctionCall { .. } | OpenaiItem::Reasoning { .. } => false,
    })
}

fn function_output_has_image(output: &OpenaiFunctionCallOutput) -> bool {
    match output {
        OpenaiFunctionCallOutput::Text(_) => false,
        OpenaiFunctionCallOutput::Content(content) => content
            .iter()
            .any(|part| matches!(part, OpenaiInputContent::InputImage { .. })),
    }
}

fn sanitize_message_tool_images(message: &OpenaiMessage, keep_images: bool) -> OpenaiMessage {
    if keep_images {
        return message.clone();
    }
    OpenaiMessage {
        items: message
            .items
            .iter()
            .map(|item| match item {
                OpenaiItem::FunctionCallOutput { call_id, output } => OpenaiItem::FunctionCallOutput {
                    call_id: call_id.clone(),
                    output: sanitize_function_output_images(output),
                },
                OpenaiItem::Message { .. } | OpenaiItem::FunctionCall { .. } | OpenaiItem::Reasoning { .. } => {
                    item.clone()
                }
            })
            .collect(),
    }
}

fn sanitize_function_output_images(output: &OpenaiFunctionCallOutput) -> OpenaiFunctionCallOutput {
    match output {
        OpenaiFunctionCallOutput::Text(_) => output.clone(),
        OpenaiFunctionCallOutput::Content(content) => OpenaiFunctionCallOutput::Content(
            content
                .iter()
                .map(|part| match part {
                    OpenaiInputContent::InputImage { image_url, .. } => OpenaiInputContent::InputText {
                        text: omitted_image_placeholder(image_url),
                    },
                    OpenaiInputContent::InputText { .. } | OpenaiInputContent::OutputText { .. } => part.clone(),
                })
                .collect(),
        ),
    }
}

fn omitted_image_placeholder(image_url: &str) -> String {
    let media_type = image_url
        .strip_prefix("data:")
        .and_then(|rest| rest.split_once(';').map(|(media_type, _)| media_type))
        .unwrap_or("unknown media type");
    let decoded_bytes = image_url
        .rsplit_once(',')
        .map_or(0, |(_, data)| data.len().saturating_mul(3) / 4);
    format!(
        "[omitted old tool image: {media_type}, {} data-url bytes, approx {} decoded bytes]",
        image_url.len(),
        decoded_bytes
    )
}

fn tool_result_item(result: ToolResult) -> OpenaiItem {
    let mut content = Vec::new();
    if result.is_error {
        content.push(OpenaiInputContent::InputText {
            text: "[tool error]".to_string(),
        });
    }
    for block in result.content {
        match block {
            ToolContent::Text(text) => content.push(OpenaiInputContent::InputText { text }),
            ToolContent::Image {
                media_type,
                data_base64,
            } => {
                let (media_type, data_base64) = compact_image(&media_type, data_base64);
                content.push(OpenaiInputContent::InputImage {
                    image_url: format!("data:{media_type};base64,{data_base64}"),
                    detail: Some(DEFAULT_IMAGE_DETAIL.to_string()),
                });
            }
        }
    }
    // Exactly one part collapses to the scalar `output` form; the array-of-one
    // conversion is the length check, so there is nothing left to re-check.
    let output = match <[OpenaiInputContent; 1]>::try_from(content) {
        Ok([OpenaiInputContent::InputText { text }]) => OpenaiFunctionCallOutput::Text(text),
        Ok([other]) => OpenaiFunctionCallOutput::Content(vec![other]),
        Err(content) => OpenaiFunctionCallOutput::Content(content),
    };
    OpenaiItem::FunctionCallOutput {
        call_id: result.tool_call_id,
        output,
    }
}

fn compact_image(media_type: &str, data_base64: String) -> (String, String) {
    let Ok(data) = BASE64.decode(data_base64.as_bytes()) else {
        return (media_type.to_string(), data_base64);
    };
    let mut reader = ImageReader::new(std::io::Cursor::new(&data));
    reader.set_format(image_format(media_type));
    let Ok(image) = reader.decode() else {
        return (media_type.to_string(), BASE64.encode(data));
    };
    let (width, height) = image.dimensions();
    if width <= TOOL_IMAGE_MAX_DIMENSION && height <= TOOL_IMAGE_MAX_DIMENSION {
        return (media_type.to_string(), BASE64.encode(data));
    }

    let resized = image.resize(TOOL_IMAGE_MAX_DIMENSION, TOOL_IMAGE_MAX_DIMENSION, FilterType::Triangle);
    let mut out = Vec::new();
    if resized
        .write_to(&mut std::io::Cursor::new(&mut out), image_format(media_type))
        .is_err()
    {
        return (media_type.to_string(), BASE64.encode(data));
    }
    (media_type.to_string(), BASE64.encode(out))
}

fn image_format(media_type: &str) -> ImageFormat {
    match media_type {
        "image/jpeg" => ImageFormat::Jpeg,
        _ => ImageFormat::Png,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum CurrentBlock {
    #[default]
    None,
    Text,
    Thinking,
    FunctionCall(u32),
}

#[derive(Debug, Default, Clone)]
struct FunctionCallState {
    call_id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Default, Clone)]
struct MessageState {
    role: Option<OpenaiMessageRole>,
    content: Vec<OpenaiInputContent>,
}

#[derive(Debug, Clone)]
enum OutputState {
    Message(MessageState),
    FunctionCall(FunctionCallState),
    Reasoning(OpenaiItem),
}

impl OutputState {
    /// The wire item type this slot holds, for conflict diagnostics.
    const fn kind(&self) -> &'static str {
        match self {
            Self::Message(_) => "message",
            Self::FunctionCall(_) => "function_call",
            Self::Reasoning(_) => "reasoning",
        }
    }
}

/// A stream that reused one `output_index` for two different item types.
///
/// `output_index` indexes the response's output array, so one index maps to
/// exactly one item; a type change means the stream contradicted itself. This is
/// [`CompletionError::Transient`] rather than `Fatal` so the retry layer
/// re-issues the turn — a single anomalous event should not end a long run.
fn output_index_conflict(output_index: u32, want: &str, held: &OutputState) -> CompletionError {
    CompletionError::Transient(format!(
        "stream reused output_index {output_index} for {want}, but that slot already holds {}; \
         refusing to overwrite it",
        held.kind()
    ))
}

pub struct ResponsesStream {
    bytes: BoxStream<'static, Result<Bytes, reqwest::Error>>,
    buffer: String,
    pending: VecDeque<Event<OpenaiMessage>>,
    text: String,
    output: BTreeMap<u32, OutputState>,
    usage: Usage,
    current_block: CurrentBlock,
    stop_reason: StopReason,
    done: bool,
    finalized: bool,
}

impl ResponsesStream {
    fn new(bytes: BoxStream<'static, Result<Bytes, reqwest::Error>>) -> Self {
        Self {
            bytes,
            buffer: String::new(),
            pending: VecDeque::new(),
            text: String::new(),
            output: BTreeMap::new(),
            usage: Usage::default(),
            current_block: CurrentBlock::None,
            stop_reason: StopReason::Other("missing".to_string()),
            done: false,
            finalized: false,
        }
    }

    fn close_current_block(&mut self) {
        match std::mem::take(&mut self.current_block) {
            CurrentBlock::Text => self.pending.push_back(Event::Text(BlockPhase::End)),
            CurrentBlock::Thinking => self.pending.push_back(Event::Thinking(BlockPhase::End)),
            CurrentBlock::FunctionCall(_) => self.pending.push_back(Event::Opaque(OpaquePhase::End)),
            CurrentBlock::None => {}
        }
    }

    fn handle_event(&mut self, event: StreamEvent) -> Result<(), CompletionError> {
        match event {
            StreamEvent::OutputTextDelta { delta } => self.push_text(delta),
            StreamEvent::ReasoningSummaryTextDelta { delta, .. } | StreamEvent::ReasoningTextDelta { delta, .. } => {
                self.push_thinking(delta);
            }
            StreamEvent::ReasoningSummaryPartAdded { .. } => self.start_thinking(),
            StreamEvent::FunctionCallArgumentsDelta { output_index, delta } => {
                self.push_arguments(output_index, delta)?;
            }
            StreamEvent::OutputItemAdded { output_index, item } => self.record_item_added(output_index, item)?,
            StreamEvent::OutputItemDone { output_index, item } => self.record_item_done(output_index, item)?,
            StreamEvent::Completed { response } => {
                self.close_current_block();
                self.usage = response.usage.unwrap_or_default().into();
                self.stop_reason = StopReason::EndTurn;
                self.done = true;
                self.enqueue_stop();
            }
            StreamEvent::Incomplete { response } => {
                self.close_current_block();
                let incomplete_reason = response.incomplete_reason();
                if incomplete_reason.as_deref() == Some("max_output_tokens") && self.has_partial_output() {
                    self.usage = response.usage.unwrap_or_default().into();
                    self.stop_reason = StopReason::MaxTokens;
                    self.done = true;
                    self.enqueue_stop();
                } else {
                    return Err(classify_incomplete_response(response));
                }
            }
            StreamEvent::Failed { response } => return Err(classify_stream_failure(response)),
            StreamEvent::Other => {}
        }
        Ok(())
    }

    fn push_text(&mut self, delta: String) {
        if self.current_block != CurrentBlock::Text {
            self.close_current_block();
            self.pending.push_back(Event::Text(BlockPhase::Start));
            self.current_block = CurrentBlock::Text;
        }
        self.text.push_str(&delta);
        self.pending.push_back(Event::Text(BlockPhase::Delta(delta)));
    }

    fn start_thinking(&mut self) {
        if self.current_block != CurrentBlock::Thinking {
            self.close_current_block();
            self.pending.push_back(Event::Thinking(BlockPhase::Start));
            self.current_block = CurrentBlock::Thinking;
        }
    }

    fn push_thinking(&mut self, delta: String) {
        self.start_thinking();
        if !delta.is_empty() {
            self.pending.push_back(Event::Thinking(BlockPhase::Delta(delta)));
        }
    }

    /// # Errors
    ///
    /// Propagates an `output_index` type conflict — see [`Self::with_call_mut`].
    fn push_arguments(&mut self, output_index: u32, delta: String) -> Result<(), CompletionError> {
        let name = self.with_call_mut(output_index, |call| {
            call.arguments.push_str(&delta);
            (!call.name.is_empty()).then(|| call.name.clone())
        })?;
        if self.current_block != CurrentBlock::FunctionCall(output_index) {
            self.close_current_block();
            self.pending.push_back(Event::Opaque(OpaquePhase::Start {
                kind: "function_call",
                name,
            }));
            self.current_block = CurrentBlock::FunctionCall(output_index);
        }
        if !delta.is_empty() {
            self.pending.push_back(Event::Opaque(OpaquePhase::Delta(delta)));
        }
        Ok(())
    }

    /// # Errors
    ///
    /// Propagates an `output_index` type conflict — see [`Self::with_call_mut`].
    fn record_item_added(&mut self, output_index: u32, item: OutputItem) -> Result<(), CompletionError> {
        match item {
            OutputItem::FunctionCall { call_id, name, .. } => {
                self.with_call_mut(output_index, |call| {
                    if let Some(call_id) = call_id {
                        call.call_id = call_id;
                    }
                    if let Some(name) = name {
                        call.name = name;
                    }
                })?;
            }
            OutputItem::Message { role, .. } => {
                if let Some(role) = role {
                    self.with_message_mut(output_index, |message| message.role = Some(role))?;
                }
            }
            OutputItem::Reasoning { .. } | OutputItem::Other => {}
        }
        Ok(())
    }

    fn record_item_done(&mut self, output_index: u32, item: OutputItem) -> Result<(), CompletionError> {
        match item {
            OutputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                // Outer `?` catches an output_index type conflict; inner catches
                // malformed arguments JSON.
                self.with_call_mut(output_index, |call| {
                    if let Some(call_id) = call_id {
                        call.call_id = call_id;
                    }
                    if let Some(name) = name {
                        call.name = name;
                    }
                    if let Some(arguments) = arguments {
                        call.arguments = arguments;
                    }
                    if !call.arguments.is_empty() {
                        let _: Value = serde_json::from_str(&call.arguments).map_err(|e| {
                            CompletionError::Fatal(format!("invalid function call arguments for {}: {e}", call.name))
                        })?;
                    }
                    Ok::<(), CompletionError>(())
                })??;
            }
            OutputItem::Message { role, content } => {
                // Snapshot the finalized text BEFORE handing `content` to the slot,
                // so the reconciliation below still sees it without holding a
                // borrow of `self.output` across the `push_text` calls.
                let new_text = content.as_deref().map(content_text);
                self.with_message_mut(output_index, |message| {
                    message.role = role.or(message.role);
                    if let Some(content) = content {
                        message.content = content;
                    }
                })?;
                if let Some(new_text) = new_text.filter(|text| !text.is_empty()) {
                    if let Some(suffix) = new_text.strip_prefix(&self.text) {
                        if !suffix.is_empty() {
                            self.push_text(suffix.to_string());
                        }
                    } else if self.text != new_text {
                        self.push_text(new_text);
                    }
                }
            }
            OutputItem::Reasoning {
                id,
                summary,
                content,
                encrypted_content,
            } => {
                // Reasoning arrives whole at `done`, so this replaces rather than
                // accumulates — but only into a vacant or already-Reasoning slot.
                // Overwriting a message or function call here would discard its
                // accumulated content, and overwriting reasoning with reasoning is
                // the normal added-then-done path.
                let state = OutputState::Reasoning(OpenaiItem::Reasoning {
                    id: id.unwrap_or_default(),
                    summary,
                    content,
                    encrypted_content,
                });
                match self.output.entry(output_index) {
                    btree_map::Entry::Vacant(slot) => {
                        slot.insert(state);
                    }
                    btree_map::Entry::Occupied(mut slot) if matches!(slot.get(), OutputState::Reasoning(_)) => {
                        slot.insert(state);
                    }
                    btree_map::Entry::Occupied(slot) => {
                        return Err(output_index_conflict(output_index, "reasoning", slot.get()));
                    }
                }
            }
            OutputItem::Other => {}
        }
        Ok(())
    }

    /// Run `f` against the function-call state for `output_index`, creating the
    /// slot if it is absent. Takes a closure so the variant check and the payload
    /// access are one expression, leaving no impossible match arm behind.
    ///
    /// # Errors
    ///
    /// [`CompletionError::Transient`] if the slot holds a different item type —
    /// overwriting would discard the accumulated `call_id`/`arguments`, after which
    /// [`Self::ordered_calls`] drops the empty husk and the tool call vanishes.
    fn with_call_mut<T>(
        &mut self,
        output_index: u32,
        f: impl FnOnce(&mut FunctionCallState) -> T,
    ) -> Result<T, CompletionError> {
        let entry = self
            .output
            .entry(output_index)
            .or_insert_with(|| OutputState::FunctionCall(FunctionCallState::default()));
        match entry {
            OutputState::FunctionCall(call) => Ok(f(call)),
            held => Err(output_index_conflict(output_index, "function_call", held)),
        }
    }

    /// Run `f` against the message state for `output_index`, creating the slot if
    /// it is absent. See [`Self::with_call_mut`] for why this takes a closure.
    ///
    /// # Errors
    ///
    /// [`CompletionError::Transient`] if the slot already holds a different item
    /// type — see [`output_index_conflict`].
    fn with_message_mut<T>(
        &mut self,
        output_index: u32,
        f: impl FnOnce(&mut MessageState) -> T,
    ) -> Result<T, CompletionError> {
        let entry = self
            .output
            .entry(output_index)
            .or_insert_with(|| OutputState::Message(MessageState::default()));
        match entry {
            OutputState::Message(message) => Ok(f(message)),
            held => Err(output_index_conflict(output_index, "message", held)),
        }
    }

    fn enqueue_stop(&mut self) {
        if self.finalized {
            return;
        }
        self.finalized = true;
        let tool_calls = self.tool_calls();
        let reason = if tool_calls.is_empty() {
            self.stop_reason.clone()
        } else {
            StopReason::ToolUse
        };
        self.pending.push_back(Event::Stop {
            reason,
            usage: self.usage,
            message: OpenaiMessage {
                items: self.output_items(),
            },
            tool_calls,
        });
    }

    fn output_items(&self) -> Vec<OpenaiItem> {
        let mut items = Vec::new();
        for item in self.output.values() {
            match item {
                OutputState::Reasoning(reasoning) => items.push(reasoning.clone()),
                OutputState::Message(message) => {
                    if !message.content.is_empty() {
                        items.push(OpenaiItem::Message {
                            role: message.role.unwrap_or(OpenaiMessageRole::Assistant),
                            content: message.content.clone(),
                        });
                    }
                }
                OutputState::FunctionCall(call) => {
                    if !call.call_id.is_empty() || !call.name.is_empty() {
                        items.push(OpenaiItem::FunctionCall {
                            call_id: call.call_id.clone(),
                            name: call.name.clone(),
                            arguments: call.arguments.clone(),
                        });
                    }
                }
            }
        }
        if !self.text.is_empty() && !self.messages_contain_streamed_text() {
            items.push(OpenaiItem::Message {
                role: OpenaiMessageRole::Assistant,
                content: vec![OpenaiInputContent::OutputText {
                    text: self.text.clone(),
                }],
            });
        }
        items
    }

    fn messages_contain_streamed_text(&self) -> bool {
        let final_text = self
            .output
            .values()
            .filter_map(|item| match item {
                OutputState::Message(message) => Some(content_text(&message.content)),
                OutputState::FunctionCall(_) | OutputState::Reasoning(_) => None,
            })
            .collect::<String>();
        !final_text.is_empty() && final_text == self.text
    }

    fn ordered_calls(&self) -> Vec<&FunctionCallState> {
        self.output
            .values()
            .filter_map(|item| match item {
                OutputState::FunctionCall(call) => Some(call),
                OutputState::Message(_) | OutputState::Reasoning(_) => None,
            })
            .filter(|call| !call.call_id.is_empty() || !call.name.is_empty())
            .collect()
    }

    fn tool_calls(&self) -> Vec<ToolCall> {
        self.ordered_calls()
            .into_iter()
            .map(|call| ToolCall {
                id: call.call_id.clone(),
                name: call.name.clone(),
                input: serde_json::from_str(&call.arguments)
                    .unwrap_or_else(|_| Value::Object(serde_json::Map::default())),
                parse_error: None,
            })
            .collect()
    }

    fn has_partial_output(&self) -> bool {
        !self.text.is_empty() || !self.output_items().is_empty() || !self.tool_calls().is_empty()
    }
}

impl Stream for ResponsesStream {
    type Item = Result<Event<OpenaiMessage>, CompletionError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(event) = self.pending.pop_front() {
            return Poll::Ready(Some(Ok(event)));
        }
        if self.done {
            return Poll::Ready(None);
        }

        loop {
            while let Some(data) = next_sse_data(&mut self.buffer) {
                if data.trim() == "[DONE]" {
                    continue;
                }
                let event = match serde_json::from_str::<StreamEvent>(&data) {
                    Ok(event) => event,
                    Err(e) => {
                        self.done = true;
                        return Poll::Ready(Some(Err(CompletionError::Transient(format!(
                            "bad Responses stream JSON: {e}; data={data}"
                        )))));
                    }
                };
                if let Err(e) = self.handle_event(event) {
                    self.done = true;
                    return Poll::Ready(Some(Err(e)));
                }
                if let Some(event) = self.pending.pop_front() {
                    return Poll::Ready(Some(Ok(event)));
                }
            }

            match self.bytes.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    self.buffer.push_str(&String::from_utf8_lossy(&chunk));
                }
                Poll::Ready(Some(Err(e))) => {
                    self.done = true;
                    return Poll::Ready(Some(Err(CompletionError::Transient(format!("stream error: {e}")))));
                }
                Poll::Ready(None) => {
                    self.done = true;
                    if self.finalized {
                        return Poll::Ready(None);
                    }
                    return Poll::Ready(Some(Err(CompletionError::Transient(
                        "Responses stream ended before response.completed".to_string(),
                    ))));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum StreamEvent {
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta { delta: String },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta { output_index: u32, delta: String },
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta { delta: String },
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta { delta: String },
    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded {},
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded { output_index: u32, item: OutputItem },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone { output_index: u32, item: OutputItem },
    #[serde(rename = "response.completed")]
    Completed { response: ResponseEnvelope },
    #[serde(rename = "response.incomplete")]
    Incomplete { response: ResponseEnvelope },
    #[serde(rename = "response.failed")]
    Failed { response: ResponseEnvelope },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputItem {
    Message {
        #[serde(default)]
        role: Option<OpenaiMessageRole>,
        #[serde(default, deserialize_with = "deserialize_content_items")]
        content: Option<Vec<OpenaiInputContent>>,
    },
    Reasoning {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        summary: Vec<Value>,
        #[serde(default)]
        content: Option<Vec<Value>>,
        #[serde(default)]
        encrypted_content: Option<String>,
    },
    FunctionCall {
        #[serde(default)]
        call_id: Option<String>,
        #[serde(default)]
        name: Option<String>,
        #[serde(default, deserialize_with = "deserialize_arguments")]
        arguments: Option<String>,
    },
    #[serde(other)]
    Other,
}

fn deserialize_content_items<'de, D>(deserializer: D) -> Result<Option<Vec<OpenaiInputContent>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let Some(values) = Option::<Vec<Value>>::deserialize(deserializer)? else {
        return Ok(None);
    };
    Ok(Some(
        values.iter().filter_map(OpenaiInputContent::from_wire_value).collect(),
    ))
}

fn deserialize_arguments<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Arguments {
        String(String),
        Snowflake {
            #[serde(rename = "OfString")]
            of_string: String,
        },
    }

    Ok(match Option::<Arguments>::deserialize(deserializer)? {
        Some(Arguments::String(value)) => Some(value),
        Some(Arguments::Snowflake { of_string }) => Some(of_string),
        None => None,
    })
}

#[derive(Debug, Default, Deserialize)]
struct ResponseEnvelope {
    #[serde(default)]
    usage: Option<ResponsesUsage>,
    #[serde(default)]
    error: Option<ResponseError>,
    #[serde(default)]
    incomplete_details: Option<IncompleteDetails>,
}

impl ResponseEnvelope {
    fn incomplete_reason(&self) -> Option<String> {
        self.incomplete_details
            .as_ref()
            .and_then(|details| details.reason.clone())
    }
}

#[derive(Debug, Deserialize)]
struct IncompleteDetails {
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseError {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default, rename = "type")]
    kind: Option<String>,
}

fn classify_stream_failure(response: ResponseEnvelope) -> CompletionError {
    let Some(error) = response.error else {
        return CompletionError::Transient("Responses stream failed without an error body".to_string());
    };
    let code = error.code.unwrap_or_default();
    let kind = error.kind.unwrap_or_default();
    let message = error.message.unwrap_or_else(|| "Responses stream failed".to_string());
    let lower = format!("{code} {kind} {message}").to_ascii_lowercase();

    if lower.contains("rate_limit") || lower.contains("rate limit") || lower.contains("too many requests") {
        CompletionError::RateLimited {
            retry_after: None,
            message,
        }
    } else if lower.contains("overload")
        || lower.contains("server_error")
        || lower.contains("service_unavailable")
        || lower.contains("temporar")
        || lower.contains("timeout")
    {
        CompletionError::Transient(message)
    } else {
        CompletionError::Fatal(message)
    }
}

fn classify_incomplete_response(response: ResponseEnvelope) -> CompletionError {
    let reason = response
        .incomplete_details
        .and_then(|details| details.reason)
        .unwrap_or_else(|| "unknown".to_string());
    let message = format!("Incomplete response returned, reason: {reason}");
    if reason == "max_output_tokens" {
        CompletionError::Transient(message)
    } else {
        CompletionError::Fatal(message)
    }
}

#[derive(Debug, Default, Deserialize)]
struct ResponsesUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    input_tokens_details: InputTokenDetails,
    #[serde(default)]
    output_tokens_details: OutputTokenDetails,
}

#[derive(Debug, Default, Deserialize)]
struct InputTokenDetails {
    #[serde(default)]
    cached_tokens: u32,
}

#[derive(Debug, Default, Deserialize)]
struct OutputTokenDetails {
    #[serde(default)]
    reasoning_tokens: u32,
}

impl From<ResponsesUsage> for Usage {
    fn from(usage: ResponsesUsage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cache_write_tokens: 0,
            cache_read_tokens: usage.input_tokens_details.cached_tokens,
            reasoning_tokens: usage.output_tokens_details.reasoning_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;
    use futures::StreamExt;
    use schemars::JsonSchema;
    use serde::Deserialize;

    struct DummyAuth;

    impl AuthProvider for DummyAuth {
        fn authorization(
            &self,
            _force_refresh: bool,
        ) -> futures::future::BoxFuture<'_, Result<String, CompletionError>> {
            Box::pin(async { Ok("Bearer test".to_string()) })
        }
    }

    #[derive(Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct EchoArgs {
        text: String,
    }

    struct Echo;

    impl Tool for Echo {
        type Args = EchoArgs;
        const NAME: &'static str = "echo";
        const DESCRIPTION: &'static str = "Echo text";

        async fn call(&self, _args: Self::Args) -> crate::tool::ToolOutput {
            Ok("ok".into())
        }
    }

    #[derive(Deserialize, JsonSchema)]
    #[serde(rename_all = "lowercase")]
    enum StageLike {
        Correctness,
        Full,
    }

    #[derive(Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct RefSiblingArgs {
        /// This description becomes a sibling next to `$ref` in schemars output.
        stage: StageLike,
    }

    struct RefSiblingTool;

    impl Tool for RefSiblingTool {
        type Args = RefSiblingArgs;
        const NAME: &'static str = "ref_sibling";
        const DESCRIPTION: &'static str = "Has a described enum field.";

        async fn call(&self, _args: Self::Args) -> crate::tool::ToolOutput {
            Ok("ok".into())
        }
    }

    fn client() -> OpenaiResponsesClient {
        OpenaiResponsesClient {
            base_url: "http://example.test/responses".to_string(),
            model: "openai-gpt-5.5".to_string(),
            instructions: None,
            parallel_tool_calls: true,
            prompt_cache_key: None,
            client_metadata: BTreeMap::from([("client".to_string(), "kernelguy".to_string())]),
            http_client: Client::new(),
            auth: Some(Arc::new(DummyAuth)),
        }
    }

    #[test]
    fn request_serializes_responses_not_chat() {
        let input = flatten_messages(&[client().user_message("hi".to_string())]);
        let body = ResponsesRequest {
            model: "openai-gpt-5.5",
            instructions: Some("system prompt"),
            input,
            tools: Vec::new(),
            tool_choice: "auto",
            parallel_tool_calls: false,
            reasoning: reasoning(ThinkingEffort::High),
            store: false,
            include: vec!["reasoning.encrypted_content"],
            service_tier: None,
            prompt_cache_key: Some("kernelguy"),
            text: None,
            client_metadata: BTreeMap::from([("client".to_string(), "kernelguy".to_string())]),
            max_output_tokens: 64,
            stream: true,
        };
        let value = serde_json::to_value(body).unwrap();

        assert_eq!(value["instructions"], "system prompt");
        assert_eq!(value["input"][0]["type"], "message");
        assert_eq!(value["parallel_tool_calls"], false);
        assert_eq!(value["reasoning"]["effort"], "high");
        assert_eq!(value["reasoning"]["summary"], "auto");
        assert_eq!(value["store"], false);
        assert_eq!(value["include"][0], "reasoning.encrypted_content");
        assert_eq!(value["prompt_cache_key"], "kernelguy");
        assert_eq!(value["client_metadata"]["client"], "kernelguy");
        assert!(value.get("service_tier").is_none());
        assert!(value.get("text").is_none());
        assert!(value.get("messages").is_none());
        assert!(value.get("reasoning_effort").is_none());
        assert!(value.get("max_completion_tokens").is_none());
    }

    #[test]
    fn client_defaults_do_not_use_global_prompt_cache_key() {
        assert!(client().prompt_cache_key.is_none());
        assert_eq!(
            client().with_prompt_cache_key("run-1").prompt_cache_key.as_deref(),
            Some("run-1")
        );
    }

    #[test]
    fn client_reports_dedicated_instructions_only_when_configured() {
        assert!(!client().uses_dedicated_instructions());
        assert!(client().with_instructions("system").uses_dedicated_instructions());
    }

    #[test]
    fn tool_definitions_are_responses_functions() {
        let value = serde_json::to_value(ResponsesTool::from_descriptor(&Echo.descriptor())).unwrap();

        assert_eq!(value["type"], "function");
        assert_eq!(value["name"], "echo");
        assert_eq!(value["description"], "Echo text");
        assert_eq!(value["strict"], false);
        assert_eq!(value["parameters"]["type"], "object");
    }

    #[test]
    fn tool_definitions_are_non_strict_like_codex_for_ref_sibling_schemas() {
        // Built from the real `Tool` impl, not a hand-written descriptor: the point
        // is that schemars' `$ref`-with-sibling-`description` output survives into
        // the request body, which a literal we authored ourselves wouldn't prove.
        let value = serde_json::to_value(ResponsesTool::from_descriptor(&RefSiblingTool.descriptor())).unwrap();

        assert_eq!(value["strict"], false);
        assert!(value["parameters"]["properties"]["stage"].get("$ref").is_some());
        assert!(value["parameters"]["properties"]["stage"].get("description").is_some());
    }

    #[test]
    fn tool_results_are_function_call_outputs() {
        let messages = client().tool_result_messages(vec![ToolResult {
            tool_call_id: "call_123".to_string(),
            content: vec![ToolContent::Text("done".to_string())],
            is_error: false,
        }]);
        let value = serde_json::to_value(&messages).unwrap();

        assert_eq!(value[0]["items"][0]["type"], "function_call_output");
        assert_eq!(value[0]["items"][0]["call_id"], "call_123");
        assert_eq!(value[0]["items"][0]["output"], "done");
    }

    #[test]
    fn tool_images_are_codex_style_content_items_resized_to_2048() {
        let bytes = png_bytes(4096, 2048);
        let messages = client().tool_result_messages(vec![ToolResult {
            tool_call_id: "call_img".to_string(),
            content: vec![ToolContent::image_png(BASE64.encode(&bytes))],
            is_error: false,
        }]);
        let value = serde_json::to_value(&messages).unwrap();
        let image = &value[0]["items"][0]["output"][0];

        assert_eq!(image["type"], "input_image");
        assert_eq!(image["detail"], "high");
        let data_base64 = image["image_url"]
            .as_str()
            .unwrap()
            .strip_prefix("data:image/png;base64,")
            .unwrap();
        let decoded = BASE64.decode(data_base64).unwrap();

        assert_eq!(png_dimensions(&decoded), (2048, 1024));
    }

    #[tokio::test]
    async fn stream_parses_text_and_final_message() {
        let mut stream = stream_from_sse(concat!(
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"pong\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":4,\"input_tokens_details\":{\"cached_tokens\":1},\"output_tokens_details\":{\"reasoning_tokens\":2}}}}\n\n",
        ));

        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            Event::Text(BlockPhase::Start)
        ));
        assert!(matches!(stream.next().await.unwrap().unwrap(), Event::Text(BlockPhase::Delta(s)) if s == "pong"));
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            Event::Text(BlockPhase::End)
        ));
        match stream.next().await.unwrap().unwrap() {
            Event::Stop {
                reason,
                usage,
                message,
                tool_calls,
                ..
            } => {
                assert!(matches!(reason, StopReason::EndTurn));
                assert_eq!(usage.input_tokens, 3);
                assert_eq!(usage.cache_read_tokens, 1);
                assert_eq!(usage.reasoning_tokens, 2);
                assert!(tool_calls.is_empty());
                assert_eq!(message.items.len(), 1);
                assert_eq!(
                    serde_json::to_value(message).unwrap()["items"][0]["content"][0]["text"],
                    "pong"
                );
            }
            _ => panic!("expected stop"),
        }
    }

    #[tokio::test]
    async fn stream_preserves_final_message_when_no_text_delta_arrives() {
        let mut stream = stream_from_sse(concat!(
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"final only\"}]}}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n",
        ));

        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            Event::Text(BlockPhase::Start)
        ));
        assert!(
            matches!(stream.next().await.unwrap().unwrap(), Event::Text(BlockPhase::Delta(text)) if text == "final only")
        );
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            Event::Text(BlockPhase::End)
        ));
        match stream.next().await.unwrap().unwrap() {
            Event::Stop { message, .. } => {
                let value = serde_json::to_value(message).unwrap();
                assert_eq!(value["items"][0]["type"], "message");
                assert_eq!(value["items"][0]["content"][0]["text"], "final only");
            }
            _ => panic!("expected stop"),
        }
    }

    #[tokio::test]
    async fn stream_final_message_only_emits_suffix_after_text_delta() {
        let mut stream = stream_from_sse(concat!(
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n",
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello world\"}]}}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n",
        ));

        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            Event::Text(BlockPhase::Start)
        ));
        assert!(
            matches!(stream.next().await.unwrap().unwrap(), Event::Text(BlockPhase::Delta(text)) if text == "hello")
        );
        assert!(
            matches!(stream.next().await.unwrap().unwrap(), Event::Text(BlockPhase::Delta(text)) if text == " world")
        );
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            Event::Text(BlockPhase::End)
        ));
        match stream.next().await.unwrap().unwrap() {
            Event::Stop { message, .. } => {
                let value = serde_json::to_value(message).unwrap();
                assert_eq!(value["items"][0]["content"][0]["text"], "hello world");
            }
            _ => panic!("expected stop"),
        }
    }

    #[tokio::test]
    async fn stream_preserves_output_index_order() {
        let mut stream = stream_from_sse(concat!(
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_a\",\"name\":\"echo\",\"arguments\":\"{}\"}}\n\n",
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"before tool\"}]}}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n",
        ));

        while let Some(event) = stream.next().await {
            if let Event::Stop { message, .. } = event.unwrap() {
                let value = serde_json::to_value(message).unwrap();
                assert_eq!(value["items"][0]["type"], "message");
                assert_eq!(value["items"][1]["type"], "function_call");
                return;
            }
        }
        panic!("expected stop");
    }

    #[tokio::test]
    async fn stream_ignores_unknown_final_message_content_items() {
        let mut stream = stream_from_sse(concat!(
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"refusal\",\"refusal\":\"no\"},{\"type\":\"output_text\",\"text\":\"visible\",\"annotations\":[]}]}}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n",
        ));

        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            Event::Text(BlockPhase::Start)
        ));
        assert!(
            matches!(stream.next().await.unwrap().unwrap(), Event::Text(BlockPhase::Delta(text)) if text == "visible")
        );
    }

    #[tokio::test]
    async fn stream_parses_function_call() {
        let mut stream = stream_from_sse(concat!(
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_a\",\"name\":\"echo\",\"arguments\":\"\"}}\n\n",
            "event: response.function_call_arguments.delta\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"text\\\":\\\"hi\\\"}\"}\n\n",
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_a\",\"name\":\"echo\",\"arguments\":\"{\\\"text\\\":\\\"hi\\\"}\"}}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n",
        ));

        assert!(
            matches!(stream.next().await.unwrap().unwrap(), Event::Opaque(OpaquePhase::Start { name, .. }) if name.as_deref() == Some("echo"))
        );
        assert!(
            matches!(stream.next().await.unwrap().unwrap(), Event::Opaque(OpaquePhase::Delta(s)) if s.contains("hi"))
        );
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            Event::Opaque(OpaquePhase::End)
        ));
        match stream.next().await.unwrap().unwrap() {
            Event::Stop {
                reason,
                tool_calls,
                message,
                ..
            } => {
                assert!(matches!(reason, StopReason::ToolUse));
                assert_eq!(tool_calls.len(), 1);
                assert_eq!(tool_calls[0].id, "call_a");
                assert_eq!(tool_calls[0].name, "echo");
                assert_eq!(tool_calls[0].input["text"], "hi");
                assert_eq!(message.items.len(), 1);
                assert_eq!(
                    serde_json::to_value(message).unwrap()["items"][0]["type"],
                    "function_call"
                );
            }
            _ => panic!("expected stop"),
        }
    }

    #[tokio::test]
    async fn stream_preserves_reasoning_encrypted_content() {
        let mut stream = stream_from_sse(concat!(
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[],\"content\":[{\"type\":\"reasoning_text\",\"text\":\"private\"}],\"encrypted_content\":\"ciphertext\"}}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"done\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n",
        ));

        while let Some(event) = stream.next().await {
            if let Event::Stop { message, .. } = event.unwrap() {
                let value = serde_json::to_value(message).unwrap();
                assert_eq!(value["items"][0]["type"], "reasoning");
                assert_eq!(value["items"][0]["encrypted_content"], "ciphertext");
                assert!(value["items"][0].get("content").is_none());
                assert_eq!(value["items"][1]["type"], "message");
                return;
            }
        }
        panic!("expected stop");
    }

    /// Regression: a stream that reuses one `output_index` for a second item type
    /// must NOT silently overwrite the slot. The old code did, which threw away
    /// the accumulated `call_id`/`arguments`; `ordered_calls` then filtered the
    /// empty husk out and the tool call vanished from the turn with no trace.
    #[tokio::test]
    async fn stream_rejects_output_index_reused_for_a_different_item_type() {
        let mut stream = stream_from_sse(concat!(
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"bash\"}}\n\n",
            "event: response.function_call_arguments.delta\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}\n\n",
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[],\"encrypted_content\":\"ciphertext\"}}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n",
        ));

        // Drain the events the function call legitimately produced, then expect the
        // conflict rather than a silently-clobbered slot. Collecting the error
        // explicitly (rather than unwrapping each item) means that WITHOUT the guard
        // this fails on the assertion below — naming the actual defect — instead of
        // panicking on the terminal `None`.
        let mut error = None;
        while let Some(item) = stream.next().await {
            if let Err(e) = item {
                error = Some(e);
                break;
            }
        }
        let Some(CompletionError::Transient(message)) = &error else {
            panic!("expected a Transient output_index conflict so the turn is retried, got {error:?}");
        };
        assert!(message.contains("output_index 0"), "{message}");
        assert!(message.contains("reasoning"), "{message}");
        assert!(message.contains("function_call"), "{message}");
    }

    /// The reverse, and the more damaging direction: a `message` item landing on an
    /// index that already holds an in-flight function call. Overwriting there
    /// discarded the accumulated `call_id`/`name`/`arguments`, after which
    /// `ordered_calls` filtered the empty husk out and the tool call vanished from
    /// the turn — so the orchestrator saw a model that had simply not called a tool.
    /// Covers `with_message_mut`; the sibling test covers `record_item_done`'s
    /// reasoning arm.
    #[tokio::test]
    async fn stream_rejects_message_overwriting_an_in_flight_function_call() {
        let mut stream = stream_from_sse(concat!(
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"bash\"}}\n\n",
            "event: response.function_call_arguments.delta\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}\n\n",
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"hi\"}]}}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n",
        ));

        let mut error = None;
        while let Some(item) = stream.next().await {
            if let Err(e) = item {
                error = Some(e);
                break;
            }
        }
        let Some(CompletionError::Transient(message)) = &error else {
            panic!("expected a Transient output_index conflict so the turn is retried, got {error:?}");
        };
        assert!(message.contains("output_index 0"), "{message}");
        assert!(message.contains("message"), "{message}");
        assert!(message.contains("function_call"), "{message}");
    }

    /// The guard is specific to *type changes*: reasoning arriving at `added` and
    /// again at `done` on the same index is the normal path and must still replace.
    #[tokio::test]
    async fn stream_allows_reasoning_replaced_on_the_same_output_index() {
        let mut stream = stream_from_sse(concat!(
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[],\"encrypted_content\":\"first\"}}\n\n",
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[],\"encrypted_content\":\"second\"}}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n",
        ));

        while let Some(event) = stream.next().await {
            if let Event::Stop { message, .. } = event.expect("no error for same-type replace") {
                let value = serde_json::to_value(message).unwrap();
                assert_eq!(value["items"][0]["encrypted_content"], "second");
                return;
            }
        }
        panic!("expected stop");
    }

    #[tokio::test]
    async fn stream_parses_reasoning_summary_deltas_as_thinking() {
        let mut stream = stream_from_sse(concat!(
            "event: response.reasoning_summary_part.added\n",
            "data: {\"type\":\"response.reasoning_summary_part.added\",\"summary_index\":0}\n\n",
            "event: response.reasoning_summary_text.delta\n",
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"summary_index\":0,\"delta\":\"checking\"}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"answer\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n",
        ));

        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            Event::Thinking(BlockPhase::Start)
        ));
        assert!(
            matches!(stream.next().await.unwrap().unwrap(), Event::Thinking(BlockPhase::Delta(text)) if text == "checking")
        );
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            Event::Thinking(BlockPhase::End)
        ));
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            Event::Text(BlockPhase::Start)
        ));
    }

    #[tokio::test]
    async fn stream_parses_reasoning_text_deltas_as_thinking() {
        let mut stream = stream_from_sse(concat!(
            "event: response.reasoning_text.delta\n",
            "data: {\"type\":\"response.reasoning_text.delta\",\"content_index\":0,\"delta\":\"raw thought\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n",
        ));

        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            Event::Thinking(BlockPhase::Start)
        ));
        assert!(
            matches!(stream.next().await.unwrap().unwrap(), Event::Thinking(BlockPhase::Delta(text)) if text == "raw thought")
        );
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            Event::Thinking(BlockPhase::End)
        ));
    }

    #[tokio::test]
    async fn stream_failed_classifies_errors() {
        let mut rate_limited = stream_from_sse(
            "event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"error\":{\"code\":\"rate_limit_exceeded\",\"message\":\"slow down\"}}}\n\n",
        );
        assert!(matches!(
            next_error(&mut rate_limited).await,
            CompletionError::RateLimited { message, .. } if message == "slow down"
        ));

        let mut overloaded = stream_from_sse(
            "event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"error\":{\"code\":\"server_error\",\"message\":\"try later\"}}}\n\n",
        );
        assert!(matches!(
            next_error(&mut overloaded).await,
            CompletionError::Transient(message) if message == "try later"
        ));

        let mut invalid = stream_from_sse(
            "event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"error\":{\"code\":\"invalid_request_error\",\"message\":\"bad schema\"}}}\n\n",
        );
        assert!(matches!(
            next_error(&mut invalid).await,
            CompletionError::Fatal(message) if message == "bad schema"
        ));
    }

    #[tokio::test]
    async fn stream_incomplete_is_error_with_reason() {
        let mut stream = stream_from_sse(
            "event: response.incomplete\ndata: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"content_filter\"}}}\n\n",
        );

        assert!(matches!(
            next_error(&mut stream).await,
            CompletionError::Fatal(message) if message.contains("content_filter")
        ));
    }

    #[tokio::test]
    async fn stream_incomplete_max_tokens_with_output_stops() {
        let mut stream = stream_from_sse(concat!(
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n",
            "event: response.incomplete\n",
            "data: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"usage\":{}}}\n\n",
        ));

        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            Event::Text(BlockPhase::Start)
        ));
        assert!(
            matches!(stream.next().await.unwrap().unwrap(), Event::Text(BlockPhase::Delta(text)) if text == "partial")
        );
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            Event::Text(BlockPhase::End)
        ));
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            Event::Stop {
                reason: StopReason::MaxTokens,
                ..
            }
        ));
    }

    #[test]
    fn model_visible_history_replaces_old_tool_images_only() {
        let messages = vec![
            tool_image_message("old_call", "image/png", "AAAA"),
            tool_image_message("newer_call", "image/jpeg", "BBBB"),
            tool_image_message("newest_call", "image/webp", "CCCC"),
        ];

        let sanitized = sanitize_old_tool_images(&messages, 2);

        assert!(function_output_text(&sanitized[0]).contains("omitted old tool image: image/png"));
        assert!(first_function_output_has_image(&sanitized[1]));
        assert!(first_function_output_has_image(&sanitized[2]));
    }

    #[test]
    fn diagnostics_count_function_output_images() {
        let messages = vec![tool_image_message("call", "image/png", "AAAA")];
        let input = flatten_messages(&messages);
        let diagnostics = RequestDiagnostics::new(&input, &[]);

        assert_eq!(diagnostics.images, 1);
        assert_eq!(diagnostics.message_images, 0);
        assert_eq!(diagnostics.function_output_images, 1);
        assert!(diagnostics.data_url_bytes > 0);
        assert!(diagnostics.function_output_json_bytes > 0);
    }

    async fn next_error(stream: &mut ResponsesStream) -> CompletionError {
        match stream.next().await.expect("stream item") {
            Ok(_) => panic!("expected error"),
            Err(error) => error,
        }
    }

    fn stream_from_sse(body: &'static str) -> ResponsesStream {
        ResponsesStream::new(Box::pin(futures::stream::iter(vec![Ok(Bytes::from_static(
            body.as_bytes(),
        ))])))
    }

    fn tool_image_message(call_id: &str, media_type: &str, data_base64: &str) -> OpenaiMessage {
        OpenaiMessage::one(OpenaiItem::FunctionCallOutput {
            call_id: call_id.to_string(),
            output: OpenaiFunctionCallOutput::Content(vec![OpenaiInputContent::InputImage {
                image_url: format!("data:{media_type};base64,{data_base64}"),
                detail: Some(DEFAULT_IMAGE_DETAIL.to_string()),
            }]),
        })
    }

    fn first_function_output_has_image(message: &OpenaiMessage) -> bool {
        match &message.items[0] {
            OpenaiItem::FunctionCallOutput { output, .. } => function_output_has_image(output),
            _ => false,
        }
    }

    fn function_output_text(message: &OpenaiMessage) -> String {
        match &message.items[0] {
            OpenaiItem::FunctionCallOutput {
                output: OpenaiFunctionCallOutput::Content(content),
                ..
            } => content_text(content),
            _ => String::new(),
        }
    }

    fn png_bytes(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::new(width, height);
        let mut out = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut out), ImageFormat::Png)
            .unwrap();
        out
    }

    fn png_dimensions(bytes: &[u8]) -> (u32, u32) {
        let reader = ImageReader::new(std::io::Cursor::new(bytes))
            .with_guessed_format()
            .unwrap();
        let img = reader.decode().unwrap();
        img.dimensions()
    }

    #[test]
    fn bare_function_call_output_is_not_a_safe_compaction_tail_start() {
        let client = client();
        // A tool-output-only turn would orphan its function_call if it started the
        // tail after the summary; assistant/user turns are fine (flat item list).
        let output = OpenaiMessage::one(OpenaiItem::FunctionCallOutput {
            call_id: "c1".to_string(),
            output: OpenaiFunctionCallOutput::Text("result".to_string()),
        });
        let assistant = OpenaiMessage::one(OpenaiItem::Message {
            role: OpenaiMessageRole::Assistant,
            content: vec![OpenaiInputContent::OutputText {
                text: "done".to_string(),
            }],
        });
        let user = client.user_message("hi".to_string());
        assert!(!client.is_compaction_tail_start(&output));
        assert!(client.is_compaction_tail_start(&assistant));
        assert!(client.is_compaction_tail_start(&user));
    }
}
