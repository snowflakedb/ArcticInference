//! Protocol-agnostic tool abstraction.
//!
//! [`Tool`] is the only trait. Implement it on a per-tool struct: it carries the
//! tool's identity as `const NAME` / `const DESCRIPTION`, ties to a typed
//! [`Tool::Args`] struct that derives [`schemars::JsonSchema`] +
//! [`serde::Deserialize`], and exposes an `async fn call`. No string
//! allocation, no docstring parsing — the type system is the source of truth.
//!
//! Erasure is a **struct, not a second trait**. [`ToolBox`] is the registry: it
//! holds the [`ToolDescriptor`]s the providers serialize, plus one boxed closure
//! per tool ([`ToolFn`]) that turns raw model JSON into a typed call.
//! [`Tool::call`] is the only part of the trait that cannot be made
//! dyn-compatible — its signature depends on `Self::Args` and it returns an
//! opaque future — so a closure is the honest encoding of "erased callable",
//! and it lets the future be `'static` rather than borrowing the registry.
//!
//! The tool-authoring vocabulary ([`ToolReply`], [`ToolOutput`]) lives in
//! [`protocol`]; the transport vocabulary the providers serialize
//! ([`ToolContent`](crate::ai::protocol::ToolContent),
//! [`ToolCall`](crate::ai::protocol::ToolCall),
//! [`ToolResult`](crate::ai::protocol::ToolResult),
//! [`ToolDescriptor`](crate::ai::protocol::ToolDescriptor)) lives in
//! [`crate::ai::protocol`], below them.

use std::collections::HashMap;
use std::sync::Arc;

use schemars::{JsonSchema, schema_for};
use serde_json::Value;

use crate::ai::protocol::{ToolCall, ToolDescriptor};

pub mod bash;
pub mod clamp;
pub mod command_guard;
pub mod edit;
pub mod evaluate;
pub mod files;
pub mod fs_tools;
pub mod markdown;
pub mod pdf;
pub mod protocol;
pub mod restore;
pub mod run;
pub mod search;
pub mod time_left;
pub mod todo;
pub mod truncate;
pub mod view;

pub use protocol::{Tool, ToolError, ToolFuture, ToolOutput, ToolReply};

/// Erased tool invocation: raw model JSON in, [`ToolOutput`] out. Owns its tool
/// via `Arc`, which is what makes the returned [`ToolFuture`] `'static`.
pub type ToolFn = Arc<dyn Fn(Value) -> ToolFuture + Send + Sync>;

/// The agent's tool registry: what to advertise, and how to run it.
///
/// Two containers because order and lookup are different concerns. Descriptors
/// keep **insertion order**, which is the order tools are advertised to the model
/// and therefore prompt-visible; dispatch only needs name → callable.
#[derive(Default)]
pub struct ToolBox {
    descriptors: Vec<ToolDescriptor>,
    calls: HashMap<&'static str, ToolFn>,
}

impl ToolBox {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tool. Its descriptor is built **once, here** — deriving a JSON
    /// schema is real work, and the same set is advertised on every turn.
    ///
    /// Re-registering a name replaces that entry in place rather than advertising
    /// the tool twice with only one of them reachable, so a collision degrades to
    /// "the last one wins" instead of an inconsistent registry. `build_tools`
    /// registers a fixed compiled-in set, so a collision there is a programming
    /// error rather than something a run can hit.
    pub fn register<T: Tool>(&mut self, tool: T) {
        let descriptor = tool.descriptor();
        let name = descriptor.name;

        let tool = Arc::new(tool);
        let call: ToolFn = Arc::new(move |args| {
            let tool = Arc::clone(&tool);
            Box::pin(async move { tool.call_json(args).await })
        });

        match self.descriptors.iter_mut().find(|d| d.name == name) {
            Some(existing) => *existing = descriptor,
            None => self.descriptors.push(descriptor),
        }
        self.calls.insert(name, call);
    }

    /// What the providers serialize into `tools[]`. A borrow, so advertising the
    /// toolset on a request costs nothing.
    #[must_use]
    pub fn descriptors(&self) -> &[ToolDescriptor] {
        &self.descriptors
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.descriptors.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    /// Run one model-issued call. Every error here is recoverable and gets
    /// forwarded to the model as an error `tool_result` so it can retry.
    ///
    /// Takes the whole [`ToolCall`] so the `parse_error` rule cannot be skipped
    /// by a caller: a call whose arguments never parsed as JSON is answered with
    /// a teaching error instead of being dispatched.
    ///
    /// # Errors
    ///
    /// - [`ToolError::UnparseableArguments`] when `call.parse_error` is set — the
    ///   provider streamed `tool_use` arguments that were not valid JSON, so
    ///   there is nothing to dispatch.
    /// - [`ToolError::UnknownTool`] when `call.name` was never registered.
    /// - Whatever the tool itself returns: [`ToolError::InvalidArguments`] if the
    ///   model's JSON does not deserialize into the tool's `Args`, or
    ///   [`ToolError::Failed`] if the tool ran and reported a problem.
    pub async fn dispatch(&self, call: &ToolCall) -> Result<ToolReply, ToolError> {
        if let Some(detail) = &call.parse_error {
            // The provider streamed tool_use arguments that weren't valid JSON, so
            // there is no real call to run. Answering (rather than failing the
            // stream) keeps a malformed argument a recoverable turn. Applies to
            // ANY tool's bad args.
            return Err(ToolError::UnparseableArguments {
                tool: call.name.clone(),
                detail: detail.clone(),
            });
        }
        match self.calls.get(call.name.as_str()) {
            Some(run) => run(call.input.clone()).await,
            None => Err(ToolError::UnknownTool {
                name: call.name.clone(),
            }),
        }
    }
}

/// Build the input schema for a typed argument struct, normalizing the output
/// for what LLM APIs accept at `tools[].input_schema` / `tools[].function.parameters`.
///
/// We strip the obvious noise schemars adds at the top level (`$schema`,
/// `title`) since neither provider needs it. Nested `$defs` / `definitions`
/// stay because referenced subschemas rely on them.
///
/// `pub(crate)` so [`Tool::descriptor`]'s default body — which lives in the
/// sibling [`protocol`] module — can derive the schema from `Self::Args`, keeping
/// the type the single source of truth for every tool.
pub(crate) fn build_input_schema<A: JsonSchema>() -> Value {
    let schema = schema_for!(A);
    let mut value = schema.to_value();
    if let Value::Object(map) = &mut value {
        map.remove("$schema");
        map.remove("title");
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::Deserialize;
    use serde_json::json;

    #[derive(Deserialize, JsonSchema)]
    struct EchoArgs {
        /// The text to echo back.
        text: String,
        /// How many times to repeat it. Defaults to 1.
        #[serde(default)]
        count: Option<u32>,
    }

    struct Echo;

    impl Tool for Echo {
        type Args = EchoArgs;
        const NAME: &'static str = "echo";
        const DESCRIPTION: &'static str = "Echo the input text, optionally repeated.";

        async fn call(&self, args: EchoArgs) -> ToolOutput {
            let n = usize::try_from(args.count.unwrap_or(1)).unwrap_or(usize::MAX);
            Ok(args.text.repeat(n).into())
        }
    }

    #[test]
    fn schema_matches_args_struct() {
        let descriptor = Echo.descriptor();
        let schema = &descriptor.input_schema;

        assert_eq!(descriptor.name, "echo");
        assert_eq!(descriptor.description, "Echo the input text, optionally repeated.");
        assert_eq!(schema["type"], json!("object"));
        assert_eq!(schema["properties"]["text"]["type"], json!("string"));
        let required = schema["required"].as_array().expect("required is an array");
        assert!(required.iter().any(|v| v == "text"));
        assert!(!required.iter().any(|v| v == "count"));
        assert!(!schema.as_object().unwrap().contains_key("$schema"));
        assert!(!schema.as_object().unwrap().contains_key("title"));
    }

    /// The registry must advertise exactly what the tool declares — `register`
    /// stores the descriptor once, so this pins that it stores the real one.
    #[test]
    fn registry_advertises_the_tools_own_descriptor() {
        let mut toolbox = ToolBox::new();
        toolbox.register(Echo);

        let advertised = toolbox.descriptors();
        let declared = Echo.descriptor();
        assert_eq!(advertised.len(), 1);
        assert_eq!(advertised[0].name, declared.name);
        assert_eq!(advertised[0].description, declared.description);
        assert_eq!(advertised[0].input_schema, declared.input_schema);
    }

    /// Registration order is prompt-visible (it's the order tools are advertised),
    /// so it must be insertion order and not the `HashMap`'s.
    #[test]
    fn registry_preserves_registration_order() {
        struct Zzz;
        impl Tool for Zzz {
            type Args = EchoArgs;
            const NAME: &'static str = "zzz";
            const DESCRIPTION: &'static str = "Sorts last alphabetically.";
            async fn call(&self, _args: EchoArgs) -> ToolOutput {
                Ok(String::new().into())
            }
        }

        let mut toolbox = ToolBox::new();
        toolbox.register(Zzz);
        toolbox.register(Echo);

        let names: Vec<&str> = toolbox.descriptors().iter().map(|d| d.name).collect();
        assert_eq!(names, vec!["zzz", "echo"]);
    }

    /// A duplicate name replaces in place rather than advertising the tool twice
    /// with only one of them reachable.
    #[test]
    fn re_registering_a_name_replaces_it_in_place() {
        let mut toolbox = ToolBox::new();
        toolbox.register(Echo);
        toolbox.register(Echo);

        assert_eq!(toolbox.len(), 1);
        assert_eq!(toolbox.descriptors()[0].name, "echo");
    }

    /// ...and it replaces the *callable* too, not only the descriptor.
    ///
    /// The test above cannot catch this: it registers the same type twice, so a
    /// stale `calls` entry is indistinguishable from a replaced one. Registering a
    /// **different** tool under the same name is what separates them. If
    /// `register`'s `insert` ever became `or_insert`, the descriptor `Vec` would
    /// still update and every other test would still pass, while the model was
    /// advertised one tool and served another — the exact desync the in-place
    /// replacement exists to prevent.
    #[tokio::test]
    async fn re_registering_replaces_the_callable_not_just_the_descriptor() {
        struct ShoutingEcho;
        impl Tool for ShoutingEcho {
            type Args = EchoArgs;
            // Deliberately collides with `Echo`.
            const NAME: &'static str = "echo";
            const DESCRIPTION: &'static str = "Echo the input text, shouting.";

            async fn call(&self, args: EchoArgs) -> ToolOutput {
                Ok(args.text.to_uppercase().into())
            }
        }

        let mut toolbox = ToolBox::new();
        toolbox.register(Echo);
        toolbox.register(ShoutingEcho);

        assert_eq!(toolbox.len(), 1, "one name, one advertised entry");
        assert_eq!(
            toolbox.descriptors()[0].description,
            "Echo the input text, shouting.",
            "the descriptor is the second registration's"
        );

        let reply = toolbox
            .dispatch(&call_of("echo", json!({ "text": "hi" })))
            .await
            .expect("dispatch should reach the replacement");
        assert_eq!(
            reply.as_text(),
            "HI",
            "the tool that RUNS must be the same registration that is advertised; \
             a stale `calls` entry would return \"hi\" from the first `Echo`"
        );
    }

    #[test]
    fn schema_shape_dump() {
        // Run with: `cargo test --bin kernelguy schema_shape_dump -- --nocapture`.
        let descriptor = Echo.descriptor();
        let pretty = serde_json::to_string_pretty(&descriptor.input_schema).unwrap();
        println!("\n{} input_schema:\n{}", descriptor.name, pretty);
    }

    fn call_of(name: &str, input: Value) -> ToolCall {
        ToolCall {
            id: "t0".to_string(),
            name: name.to_string(),
            input,
            parse_error: None,
        }
    }

    #[tokio::test]
    async fn dispatch_deserializes_typed_args() {
        let mut toolbox = ToolBox::new();
        toolbox.register(Echo);

        let result = toolbox
            .dispatch(&call_of("echo", json!({ "text": "ab", "count": 3 })))
            .await;
        assert_eq!(result.unwrap().as_text(), "ababab");

        let result_default = toolbox.dispatch(&call_of("echo", json!({ "text": "hi" }))).await;
        assert_eq!(result_default.unwrap().as_text(), "hi");
    }

    /// Which variant each failure produces. Asserted on the variant, not its text,
    /// so rewording a message can't quietly change what this covers — the text is
    /// pinned separately by `tool_error_display_is_prompt_content`.
    #[tokio::test]
    async fn dispatch_produces_the_right_error_variant() {
        let mut toolbox = ToolBox::new();
        toolbox.register(Echo);

        // Never registered.
        let err = toolbox.dispatch(&call_of("nope", json!({}))).await.unwrap_err();
        assert!(
            matches!(&err, ToolError::UnknownTool { name } if name == "nope"),
            "{err:?}"
        );

        // Valid JSON, wrong shape: `text` is required.
        let err = toolbox
            .dispatch(&call_of("echo", json!({ "count": 2 })))
            .await
            .unwrap_err();
        assert!(
            matches!(&err, ToolError::InvalidArguments { tool, .. } if *tool == "echo"),
            "{err:?}"
        );

        // `parse_error` short-circuits: the tool never runs, so this must NOT come
        // back as `InvalidArguments` from a deserialize attempt. Enforcing it inside
        // `dispatch` rather than in the caller is the point of taking the whole
        // `ToolCall`.
        let mut call = call_of("echo", json!({}));
        call.parse_error = Some("expected `:` at line 1 column 12".to_string());
        let err = toolbox.dispatch(&call).await.unwrap_err();
        assert!(
            matches!(&err, ToolError::UnparseableArguments { tool, .. } if tool == "echo"),
            "{err:?}"
        );

        // A tool that ran and failed is the one variant that isn't the framework's.
        let err = ToolError::from("disk on fire".to_string());
        assert!(matches!(&err, ToolError::Failed(m) if m == "disk on fire"), "{err:?}");
    }

    /// The whole reason these are typed: three of the four variants mean the tool
    /// body never executed, which is a different pathology from a tool that ran and
    /// failed, and unrecoverable from a `String`.
    #[test]
    fn never_ran_separates_protocol_failures_from_tool_failures() {
        let protocol = [
            ToolError::UnknownTool { name: "x".into() },
            ToolError::UnparseableArguments {
                tool: "x".into(),
                detail: "y".into(),
            },
            ToolError::InvalidArguments {
                tool: "x",
                raw: json!({}),
                source: serde_json::from_str::<u8>("{}").unwrap_err(),
            },
        ];
        for err in &protocol {
            assert!(err.never_ran(), "{err:?} means the tool body never ran");
        }
        assert!(!ToolError::Failed("boom".into()).never_ran());
    }

    /// Exact pins on the text of every dispatch error. These strings are handed
    /// straight to the model as the `tool_result` body, so they are prompt content,
    /// not diagnostics: rewording one changes model behaviour. Captured before the
    /// errors were typed, so the typed rewrite reproduces them byte for byte.
    ///
    /// Separate from the variant test on purpose — that one covers the logic, this
    /// one covers the prompt. Neither substitutes for the other.
    #[tokio::test]
    async fn tool_error_display_is_prompt_content() {
        let mut toolbox = ToolBox::new();
        toolbox.register(Echo);

        let unknown = toolbox.dispatch(&call_of("nope", json!({}))).await.unwrap_err();
        assert_eq!(unknown.to_string(), "unknown tool: nope");

        let bad_shape = toolbox
            .dispatch(&call_of("echo", json!({ "count": 2 })))
            .await
            .unwrap_err();
        assert_eq!(
            bad_shape.to_string(),
            "invalid arguments for `echo`: missing field `text`. Received: {\"count\":2}. \
             Re-issue the call with arguments matching the tool's schema."
        );

        let mut unparseable = call_of("echo", json!({}));
        unparseable.parse_error = Some("expected `:` at line 1 column 12".to_string());
        let unparseable = toolbox.dispatch(&unparseable).await.unwrap_err();
        assert_eq!(
            unparseable.to_string(),
            "Your `echo` arguments were not valid JSON: expected `:` at line 1 column 12. \
             A common cause is writing `\">` where JSON needs `\":\"` between a field name \
             and its value, e.g. \"old_text\": \"...\". Re-issue the call with valid JSON."
        );
    }

    /// Compile-time check: `Echo::NAME` is `&'static str`, can be matched
    /// in const contexts.
    #[test]
    fn name_is_static_str() {
        const NAME: &str = Echo::NAME;
        assert_eq!(NAME, "echo");
    }
}
