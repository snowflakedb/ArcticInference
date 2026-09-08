//! The tool *transport* vocabulary: the types that cross between a tool call and
//! the wire.
//!
//! They live in the wire layer because this is the code that serializes them. A
//! provider never dispatches a tool — it only describes one
//! ([`ToolDescriptor`]), reads back what the model asked for ([`ToolCall`]), and
//! sends the answer ([`ToolResult`] / [`ToolContent`]).
//!
//! The tool-*authoring* vocabulary — `ToolReply` and `ToolOutput`, i.e. what you
//! return from `Tool::call` — deliberately stays in [`crate::tool::protocol`]. No
//! provider mentions either.

use serde::Serialize;
use serde_json::Value;

/// One block of a tool's reply. Most tools return a single [`ToolContent::Text`]
/// block; image-returning tools (e.g. `view`) return an
/// [`ToolContent::Image`] block.
///
/// Protocol mapping:
/// - Anthropic: each variant maps to one `tool_result.content[]` block —
///   `text` → `{ type: "text", text }`, `image` → `{ type: "image", source: {...} }`.
/// - `OpenAI` Responses: tool outputs become `function_call_output` items.
///   Text-only results use a string `output`; image-bearing results use content
///   items with compacted data-URL `input_image` parts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolContent {
    Text(String),
    /// Inline image. `media_type` is an HTTP-style content type
    /// (`image/png`, `image/jpeg`, …); `data_base64` is standard
    /// (RFC 4648) base64 with no surrounding `data:` prefix.
    Image {
        media_type: String,
        data_base64: String,
    },
}

impl ToolContent {
    /// Convenience constructor for the most common case.
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    /// Convenience constructor for a base64-encoded PNG.
    #[must_use]
    pub fn image_png(data_base64: String) -> Self {
        Self::Image {
            media_type: "image/png".to_string(),
            data_base64,
        }
    }
}

/// Everything a provider needs to advertise one tool to the model, and nothing else.
///
/// Produced by `Tool::descriptor` and projected once per registry into a
/// `ToolBox`, so building one is not on a per-request path.
///
/// The field set *is* Anthropic's wire shape, so this serializes straight into
/// `tools[]` with no adapter. `OpenAI` Responses renames `input_schema` to
/// `parameters` and adds its own `type`/`strict` fields, so it keeps a thin
/// wrapper — which is also where any future provider-specific per-tool flag
/// (`cache_control`, `defer_loading`, …) belongs.
///
/// `name` and `description` are `&'static str` because `Tool::NAME` and
/// `Tool::DESCRIPTION` are associated *consts* — a tool's identity is fixed at
/// compile time, and this states that rather than laundering it through an
/// allocation. `name` additionally keys the registry's dispatch map.
///
/// A tool wanting a description computed from instance state cannot get there by
/// overriding `Tool::descriptor` alone: the field type is fixed here, not chosen
/// per implementor, so an override still has to produce a `&'static str`. Making
/// that work means changing this field (to `Cow<'static, str>`, most likely) — the
/// deliberate cost of the first tool that needs it, rather than something paid up
/// front for a case that does not yet exist.
#[derive(Debug, Clone, Serialize)]
pub struct ToolDescriptor {
    pub name: &'static str,
    pub description: &'static str,
    pub input_schema: Value,
}

/// One model-issued tool invocation extracted from an aggregated assistant
/// turn. The harness builds these from provider-specific `tool_use` / `tool_call`
/// blocks and passes them to `ToolBox::dispatch`.
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: Value,
    /// Set when the provider streamed `tool_use` arguments that were not valid
    /// JSON. When present the call is answered with a teaching error `tool_result`
    /// instead of being dispatched, so a malformed argument is a recoverable turn
    /// rather than a run-fatal stream error. `ToolBox::dispatch` enforces this, so
    /// no caller can skip it.
    pub parse_error: Option<String>,
}

/// Outcome of dispatching a [`ToolCall`], in the shape `ProtocolClient`'s
/// `tool_result_messages` consumes to produce provider-specific reply
/// messages.
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: Vec<ToolContent>,
    pub is_error: bool,
}
