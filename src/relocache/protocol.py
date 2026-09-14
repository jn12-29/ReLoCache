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
from vllm.entrypoints.openai.responses.protocol import ResponseInputOutputItem, ResponsesRequest


class ReusableContentPart(TypedDict, total=False):
    type: Required[Literal["reusable_content"]]
    id: Required[Annotated[str, Field(min_length=1)]]
    input_ids: list[int]
    text: str


def validate_content_source(part):
    if "input_ids" in part and "text" in part:
        raise ValueError("Supply at most one of input_ids or text")
    return part


ContentReference = Annotated[ReusableContentPart, AfterValidator(validate_content_source)]


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


class ContentInput(BaseModel):
    content_id: str = Field(min_length=1)
    input_ids: list[int] | None = None
    text: str | None = None
    persist: bool = False

    @model_validator(mode="after")
    def source(self):
        if (self.input_ids is None) == (self.text is None):
            raise ValueError("Supply exactly one of input_ids or text")
        return self
