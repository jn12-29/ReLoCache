"""External reusable-content inputs, independent of scientific methods."""

from typing import Annotated, Literal, Required

from openai.types.chat import (
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageToolCallParam,
)
from openai.types.responses import ResponseInputTextParam, ResponseOutputTextParam
from pydantic import AfterValidator, BaseModel, Discriminator, Field, Tag, model_validator
from typing_extensions import TypedDict
from vllm.entrypoints.chat_utils import (
    ChatCompletionContentPartPromptEmbedsParam,
    ChatCompletionMessageParam,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponseInputOutputItem, ResponsesRequest

from relocache.protocol import PromptEmbedsPart, ReusableContentPart, TokenPart


def validate_content_source(part):
    if not part["id"]:
        raise ValueError("Content ID must not be empty")
    if "input_ids" in part and "text" in part:
        raise ValueError("Supply at most one of input_ids or text")
    return part


ContentReference = Annotated[ReusableContentPart, AfterValidator(validate_content_source)]


def validate_token_part(part):
    if not part["input_ids"] or any(token < 0 for token in part["input_ids"]):
        raise ValueError("Token blocks require non-empty, non-negative token IDs")
    return part


class BlockPrompt(TypedDict):
    blocks: Annotated[
        list[
            Annotated[TokenPart, AfterValidator(validate_token_part)]
            | PromptEmbedsPart
            | ContentReference
        ],
        Field(min_length=1),
    ]


class ReusableCompletionRequest(CompletionRequest):
    prompt: (
        BlockPrompt
        | list[Annotated[int, Field(ge=0)]]
        | list[list[Annotated[int, Field(ge=0)]]]
        | str
        | list[str]
        | None
    ) = None

    @model_validator(mode="after")
    def block_options(self):
        if isinstance(self.prompt, dict):
            if self.prompt_embeds is not None:
                raise ValueError("Use prompt_embeds blocks inside prompt.blocks")
            if self.truncate_prompt_tokens is not None:
                raise ValueError("Token content blocks execute exactly; truncation is unsupported")
            if self.echo or self.prompt_logprobs is not None:
                raise ValueError("Echo and prompt_logprobs are unsupported with content blocks")
        return self


class ReusableChatMessage(TypedDict, total=False):
    role: Required[str]
    content: Required[
        list[
            ChatCompletionContentPartTextParam
            | ChatCompletionContentPartPromptEmbedsParam
            | ContentReference
        ]
    ]
    name: str
    tool_call_id: str | None
    tool_calls: list[ChatCompletionMessageToolCallParam] | None
    reasoning: str | None
    tools: list[ChatCompletionFunctionToolParam] | None
    task: str | None


class ReusableResponseMessage(TypedDict, total=False):
    role: Required[Literal["user", "assistant", "system", "developer"]]
    content: Required[
        list[
            ResponseInputTextParam
            | ResponseOutputTextParam
            | ChatCompletionContentPartPromptEmbedsParam
            | ContentReference
        ]
    ]
    type: Literal["message"]
    id: str
    status: Literal["in_progress", "completed", "incomplete"]
    phase: Literal["commentary", "final_answer"] | None


def message_kind(message, *, types=("reusable_content",)):
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list) and any(
        isinstance(part, dict) and part.get("type") in types for part in content
    ):
        return "reusable"
    return "native"


def response_message_kind(message):
    return message_kind(message, types=("reusable_content", "prompt_embeds"))


class ReusableChatRequest(ChatCompletionRequest):
    messages: list[
        Annotated[
            Annotated[ReusableChatMessage, Tag("reusable")]
            | Annotated[ChatCompletionMessageParam, Tag("native")],
            Discriminator(message_kind),
        ]
    ]


class ReusableResponsesRequest(ResponsesRequest):
    input: (
        str
        | list[
            Annotated[
                Annotated[ReusableResponseMessage, Tag("reusable")]
                | Annotated[ResponseInputOutputItem, Tag("native")],
                Discriminator(response_message_kind),
            ]
        ]
    )


class CompileInput(BaseModel):
    content_id: str = Field(min_length=1)
    input_ids: list[int] | None = None
    text: str | None = None
    persist: bool = False

    @model_validator(mode="after")
    def source(self):
        if self.input_ids is not None and self.text is not None:
            raise ValueError("Supply at most one of input_ids or text")
        return self


class ContentInput(CompileInput):
    @model_validator(mode="after")
    def require_source(self):
        if self.input_ids is None and self.text is None:
            raise ValueError("Registration requires input_ids or text")
        return self
