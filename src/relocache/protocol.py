"""Public content types usable without an inference backend."""

from typing import Literal, NotRequired, TypedDict


class TokenPart(TypedDict):
    type: Literal["tokens"]
    input_ids: list[int]


class PromptEmbedsPart(TypedDict):
    type: Literal["prompt_embeds"]
    data: str


class ReusableContentPart(TypedDict):
    type: Literal["reusable_content"]
    id: str
    input_ids: NotRequired[list[int]]
    text: NotRequired[str]


ContentBlock = TokenPart | PromptEmbedsPart | ReusableContentPart


class BlockPrompt(TypedDict):
    blocks: list[ContentBlock]


class ContentInfo(TypedDict):
    content_id: str
    input_ids: list[int]
    logical_length: int
    source_registered: bool
    compilation: NotRequired[dict]
    metrics: NotRequired[dict]
