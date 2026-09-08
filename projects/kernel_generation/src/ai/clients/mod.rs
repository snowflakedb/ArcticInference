//! Per-protocol wire implementations of the
//! [`ProtocolClient`](super::ProtocolClient) trait.
//!
//! One module per *protocol*, not per vendor: `anthropic_messages` and
//! `openai_responses` are two wire formats, and the same host can serve both
//! (Snowflake Cortex does). A third, `openai_chat_completions`, exists in
//! [`CompletionProtocol`](super::CompletionProtocol) with no client yet.
//!
//! The protocol modules are private; this re-export list is the crate's entire
//! view of them, and widening it exports types that are currently internal.

mod anthropic_messages;
mod openai_responses;

pub use anthropic_messages::{
    AnthropicContentBlock, AnthropicImageSource, AnthropicMessage, AnthropicMessageRole, AnthropicMessagesClient,
};
pub use openai_responses::{OpenaiInputContent, OpenaiItem, OpenaiMessage, OpenaiMessageRole, OpenaiResponsesClient};
