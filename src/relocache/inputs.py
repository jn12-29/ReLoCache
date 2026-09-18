"""Compose logical Reader inputs independently of backend storage coordinates."""

from dataclasses import dataclass, field
from typing import Any

from .content import Content, ContentError, wrapped_token_ids


@dataclass
class ReaderInput:
    input_ids: list[int] = field(default_factory=list)
    contents: list[Content] = field(default_factory=list)
    spans: list[tuple[int, int, int]] = field(default_factory=list)
    embeddings: list[Any] = field(default_factory=list)
    embedding_positions: list[tuple[int, int]] = field(default_factory=list)

    def retained_spans(self, left, right):
        spans = []
        for start, length, index in self.spans:
            end = start + length
            if start < left < end or start < right < end:
                raise ContentError("Truncation cannot split reusable content")
            if left <= start and end <= right:
                spans.append((start - left, length, index))
        return spans


def assemble_input(blocks, *, pre_length=0, post_length=0):
    """Compose token lists, embedding tensors, and resolved content snapshots."""
    result = ReaderInput()
    content_indices = {}
    for block in blocks:
        if isinstance(block, Content):
            key = id(block)
            if key not in content_indices:
                content_indices[key] = len(result.contents)
                result.contents.append(block)
            ids = wrapped_token_ids(block.input_ids, pre_length, post_length)
            result.spans.append((len(result.input_ids), len(ids), content_indices[key]))
            result.input_ids.extend(ids)
        elif isinstance(block, list):
            result.input_ids.extend(block)
        else:
            length = block.shape[0]
            result.embeddings.append(block)
            result.embedding_positions.append((len(result.input_ids), length))
            result.input_ids.extend([0] * length)
    return result
