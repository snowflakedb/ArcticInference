# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import time
from typing import (AsyncGenerator, AsyncIterator, Callable, Dict, Final, List,
                    Optional)
from typing import Sequence as GenericSequence
from typing import Union

from fastapi import Request

from vllm.entrypoints.chat_utils import (ChatTemplateContentFormatOption,
                                         ConversationMessage)
from vllm.entrypoints.openai.protocol import (
    ChatCompletionLogProb, ChatCompletionLogProbs,
    ChatCompletionLogProbsContent, ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaFunctionCall, DeltaMessage,
    DeltaToolCall, ErrorResponse, FunctionCall, FunctionDefinition,
    PromptTokenUsageInfo, RequestResponseMetadata, ToolCall, UsageInfo)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager
from vllm.entrypoints.openai.tool_parsers.mistral_tool_parser import (
    MistralToolCall)
from vllm.logger import init_logger
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.transformers_utils.tokenizers import (maybe_serialize_tool_calls,
                                                truncate_tool_call_ids)
from vllm.entrypoints.chat_utils import (ChatTemplateContentFormatOption,
                                         ConversationMessage,
                                         apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         parse_chat_messages_futures,
                                         resolve_chat_template_content_format)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
logger = init_logger(__name__)


class AdaptiveOpenAIServingChat(OpenAIServingChat):
    async def get_completion_prompt(
        self,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Union[str, List[int]]:
        """
        Check to see if we should check for unstreamed tool arguments tokens.
        This is only applicable when auto tool parsing is enabled, the delta
        is a tool call with arguments.
        """
        (lora_request,
            prompt_adapter_request,
        ) = self._maybe_get_adapters(request)

        model_name = self.models.model_name(lora_request)

        tokenizer = await self.engine_client.get_tokenizer(lora_request)
        tool_dicts = None if request.tools is None else [
                        tool.model_dump() for tool in request.tools
                    ]

        resolved_content_format = resolve_chat_template_content_format(
            self.chat_template,
            tool_dicts,
            self.chat_template_content_format,
            tokenizer,
        )
        conversation, mm_data_future = parse_chat_messages_futures(
            request.messages,
            self.model_config,
            tokenizer,
            content_format=resolved_content_format,
        )
        _chat_template_kwargs: Dict[str, Any] = dict(
            chat_template=self.chat_template,
            add_generation_prompt=request.add_generation_prompt,
            continue_final_message=request.continue_final_message,
            tools=tool_dicts,
            documents=request.documents,
        )
        _chat_template_kwargs.update(request.chat_template_kwargs or {})
        request_prompt: Union[str, List[int]]
        if isinstance(tokenizer, MistralTokenizer):
            request_prompt = apply_mistral_chat_template(
                tokenizer,
                messages=messages,
                **_chat_template_kwargs,
            )
        else:
            request_prompt = apply_hf_chat_template(
                tokenizer,
                conversation=conversation,
                **_chat_template_kwargs,
            )
        return request_prompt

