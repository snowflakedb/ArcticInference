//! The tool-authoring vocabulary and the [`Tool`] trait.
//!
//! [`ToolReply`] / [`ToolOutput`] live here because they are what a tool *returns*
//! — no provider mentions either. The transport half ([`ToolContent`],
//! [`ToolCall`](crate::ai::protocol::ToolCall),
//! [`ToolResult`](crate::ai::protocol::ToolResult), and
//! [`ToolDescriptor`](crate::ai::protocol::ToolDescriptor)) lives in
//! [`crate::ai::protocol`], below the providers that serialize it.
//!
//! The registry that erases a `Tool` into something callable with raw JSON is
//! [`ToolBox`](super::ToolBox) in the parent [`mod`](super) module.

use std::future::Future;
use std::pin::Pin;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::ai::protocol::{ToolContent, ToolDescriptor};

/// What a [`Tool`] returns inside `Ok(...)`.
///
/// Wraps `Vec<ToolContent>` so tools can write `Ok("ok".into())` for the
/// common "single text block" case and `Ok(blocks.into())` when they
/// actually produce multiple content blocks. The wrapper unwraps to
/// `Vec<ToolContent>` at the harness boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolReply(pub Vec<ToolContent>);

impl From<String> for ToolReply {
    fn from(s: String) -> Self {
        Self(vec![ToolContent::Text(s)])
    }
}

impl From<&str> for ToolReply {
    fn from(s: &str) -> Self {
        Self(vec![ToolContent::Text(s.to_string())])
    }
}

impl From<ToolContent> for ToolReply {
    fn from(c: ToolContent) -> Self {
        Self(vec![c])
    }
}

impl From<Vec<ToolContent>> for ToolReply {
    fn from(v: Vec<ToolContent>) -> Self {
        Self(v)
    }
}

impl From<ToolReply> for Vec<ToolContent> {
    fn from(r: ToolReply) -> Self {
        r.0
    }
}

impl ToolReply {
    /// Concatenate every [`ToolContent::Text`] block into one string,
    /// dropping image blocks. Useful in tests asserting a tool's
    /// observable text output without caring about other content.
    #[must_use]
    pub fn as_text(&self) -> String {
        self.0
            .iter()
            .filter_map(|c| match c {
                ToolContent::Text(s) => Some(s.as_str()),
                ToolContent::Image { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

/// Result of a tool invocation.
///
/// The `Ok` payload is appended back into the conversation as the user-side
/// `tool_result` (Anthropic) / `tool` + follow-up `user` message (`OpenAI`).
/// `Err` carries a string the harness forwards to the model as an error tool
/// result so it can recover.
///
/// This is [`Tool::call`]'s result specifically — a tool that ran and reported a
/// problem. It stays a `String` because a tool's own failures are its own
/// business; the *framework's* failures are typed, see [`ToolError`].
pub type ToolOutput = Result<ToolReply, String>;

/// Everything that can go wrong reaching or running a tool.
///
/// The distinction that matters is **whether the tool body executed**.
/// [`Self::UnknownTool`], [`Self::UnparseableArguments`] and
/// [`Self::InvalidArguments`] all mean it never did — the model malformed its
/// request — while [`Self::Failed`] means the tool ran and reported a problem.
/// [`Self::never_ran`] is that predicate.
///
/// Only [`Tool::call_json`] can produce `InvalidArguments`, because it is the one
/// place untyped model JSON becomes a typed `Args`. [`Tool::call`] cannot fail
/// that way — its arguments are already typed.
///
/// Every `Display` here is **prompt content**: the harness turns it into the
/// `tool_result` body the model reads, so the wording is behaviour, not
/// diagnostics. `dispatch_error_messages_are_pinned` in [`super`] asserts each one
/// byte for byte.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// The model asked for a tool that was never registered.
    #[error("unknown tool: {name}")]
    UnknownTool { name: String },

    /// The provider streamed `tool_use` arguments that weren't valid JSON at all, so
    /// there was nothing to deserialize. Recorded on
    /// [`ToolCall::parse_error`](crate::ai::protocol::ToolCall::parse_error) and
    /// converted here by [`ToolBox::dispatch`](super::ToolBox::dispatch).
    #[error(
        "Your `{tool}` arguments were not valid JSON: {detail}. A common cause is \
         writing `\">` where JSON needs `\":\"` between a field name and its value, \
         e.g. \"old_text\": \"...\". Re-issue the call with valid JSON."
    )]
    UnparseableArguments { tool: String, detail: String },

    /// Valid JSON, wrong shape for this tool's `Args`. `raw` is echoed back so the
    /// error is teachable rather than an opaque "invalid args".
    #[error(
        "invalid arguments for `{tool}`: {source}. Received: {raw}. \
         Re-issue the call with arguments matching the tool's schema."
    )]
    InvalidArguments {
        tool: &'static str,
        raw: Value,
        #[source]
        source: serde_json::Error,
    },

    /// The tool ran and returned an error. Constructed only via [`From<String>`],
    /// which is what lets every existing tool keep returning `Err(String)` from
    /// [`Tool::call`] untouched. Keep it that way until some caller actually wants
    /// to branch on a specific tool's failure, or this becomes the only variant
    /// anyone uses.
    #[error("{0}")]
    Failed(String),
}

impl ToolError {
    /// Whether the tool body never executed — i.e. the model's request was
    /// malformed, rather than the workspace being uncooperative.
    ///
    /// This is the reason these are typed rather than strings: "the model has sent
    /// unusable arguments five turns running" is a different pathology from "this
    /// tool keeps failing", and they warrant different responses. Nothing consumes
    /// it yet — the harness still renders both as an error `tool_result`.
    #[must_use]
    pub const fn never_ran(&self) -> bool {
        !matches!(self, Self::Failed(_))
    }
}

impl From<String> for ToolError {
    fn from(message: String) -> Self {
        Self::Failed(message)
    }
}

/// Boxed, erased tool future. `'static` (not borrowing `&self`) because the
/// registry's closures own their tool via `Arc`, which leaves the future free to
/// be spawned or run concurrently.
pub type ToolFuture = Pin<Box<dyn Future<Output = Result<ToolReply, ToolError>> + Send>>;

/// User-facing tool trait. Implement this on a (usually unit-like) struct to
/// register a tool with the agent.
///
/// ```ignore
/// struct Read { sandbox: Arc<Sandbox> }
///
/// #[derive(Deserialize, JsonSchema)]
/// struct ReadArgs { /// Path relative to the workspace.
///                   path: String }
///
/// impl Tool for Read {
///     type Args = ReadArgs;
///     const NAME: &'static str = "read";
///     const DESCRIPTION: &'static str = "Read a file from the workspace.";
///
///     async fn call(&self, args: ReadArgs) -> ToolOutput {
///         let text = self.sandbox.read(&args.path).map_err(|e| e.to_string())?;
///         Ok(text.into())
///     }
/// }
/// ```
pub trait Tool: Send + Sync + 'static {
    /// JSON-deserializable, schema-deriving struct describing the tool's input.
    type Args: JsonSchema + DeserializeOwned + Send + 'static;

    /// Tool name as shown to the model. Must be unique within a registry.
    const NAME: &'static str;
    /// Free-form description shown to the model alongside the input schema.
    const DESCRIPTION: &'static str;

    /// How this tool is advertised to the model. The default derives everything
    /// from the type — name and description from the consts, schema from
    /// [`Self::Args`] — so the type stays the single source of truth and a tool
    /// cannot drift from its own schema.
    ///
    /// Override only when the descriptor depends on instance state (a description
    /// quoting a configured budget, say) — but note that
    /// [`ToolDescriptor`]'s fields are `&'static str`, so an override alone cannot
    /// return a computed string. Making that work means widening that field type
    /// first; see its docs.
    ///
    /// Deriving the schema calls [`schema_for!`](schemars::schema_for), which is
    /// real work: call this **once per registry**, not once per request.
    /// `ToolBox::register` stores the result for exactly that reason.
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            name: Self::NAME,
            description: Self::DESCRIPTION,
            input_schema: super::build_input_schema::<Self::Args>(),
        }
    }

    /// Run the tool. The future must be `Send` so dispatchers can spawn across
    /// threads; this is enforced by the bound on the return type rather than
    /// the `async fn` form.
    fn call(&self, args: Self::Args) -> impl Future<Output = ToolOutput> + Send;

    /// Untyped entry point: deserialize raw model JSON into [`Self::Args`], then
    /// run [`Self::call`].
    ///
    /// This is the **only** place untyped input becomes typed, which is why it is
    /// also the only place [`ToolError::InvalidArguments`] can originate. The
    /// registry's dispatch closure and the per-tool unit tests both go through
    /// here, so a malformed-argument error reads the same however the tool was
    /// reached. Not intended to be overridden.
    fn call_json(&self, args: Value) -> impl Future<Output = Result<ToolReply, ToolError>> + Send {
        async move {
            // `args` is cloned into the deserializer so the original can be quoted
            // back in the error; that makes any tool's arg error teachable rather
            // than an opaque "invalid args". One clone, negligible next to the
            // tool's actual work.
            let typed: Self::Args =
                serde_json::from_value(args.clone()).map_err(|source| ToolError::InvalidArguments {
                    tool: Self::NAME,
                    raw: args,
                    source,
                })?;
            self.call(typed).await.map_err(ToolError::from)
        }
    }
}
