# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve reusable references while retaining canonical model token IDs."""

import asyncio
import time
from dataclasses import replace
from uuid import uuid4

from vllm.exceptions import VLLMValidationError
from vllm.renderers.embed_utils import safe_load_prompt_embeds_async
from vllm.renderers.hf import _build_mixed_prompt_embeds
from vllm.v1.core.pic import ReusableInput


def has_reusable_content(messages):
    return any(
        isinstance(message.get("content"), list)
        and any(
            isinstance(part, dict) and part.get("type") == "reusable_content"
            for part in message["content"]
        )
        for message in messages
    )


def prepare_reusable_messages(messages):
    markers = {}
    result = []
    for message in messages:
        if not isinstance(message.get("content"), list):
            result.append(message)
            continue
        parts = []
        for part in message["content"]:
            if part.get("type") in ("reusable_content", "prompt_embeds"):
                marker = f"<|reusable_{uuid4().hex}|>"
                markers[marker] = part
                part = {"type": "text", "text": marker}
            if part.get("type") in ("text", "input_text", "output_text"):
                if parts and parts[-1].get("type") == "text":
                    parts[-1]["text"] += part["text"]
                else:
                    parts.append({"type": "text", "text": part["text"]})
            else:
                parts.append(part)
        result.append({**message, "content": parts})
    return result, markers


def iter_reusable_spans(text, markers):
    cursor = 0
    for start, marker in sorted((text.index(key), key) for key in markers):
        if cursor < start:
            yield text[cursor:start], None
        yield "", markers[marker]
        cursor = start + len(marker)
    if cursor < len(text):
        yield text[cursor:], None


async def render_reusable_chat(
    renderer,
    messages,
    chat_params,
    tok_params,
    resolver,
    *,
    prompt_extras=None,
    skip_mm_cache=False,
):
    if not has_reusable_content(messages):
        return None
    messages, markers = prepare_reusable_messages(messages)
    references = {}
    for part in markers.values():
        if part["type"] == "reusable_content" and (
            part["id"] not in references or "text" in part or "input_ids" in part
        ):
            references[part["id"]] = part
    resolved = await asyncio.gather(*(resolver(part) for part in references.values()))
    contents = dict(zip(references, resolved, strict=True))
    content_indices = {name: index for index, name in enumerate(contents)}
    params = replace(
        chat_params,
        chat_template_kwargs={
            **chat_params.chat_template_kwargs,
            "tokenize": False,
        },
    )
    conversation, prompt = await renderer.render_messages_async(messages, params)
    if prompt.get("multi_modal_data"):
        raise NotImplementedError("Reusable rendering currently supports text inputs")
    tokenizer = renderer.get_tokenizer()
    ids, spans, embeddings, embed_positions = [], [], [], []
    decoded_embeds = {}
    for text, part in iter_reusable_spans(prompt["prompt"], markers):
        if part is None:
            ids.extend(
                tokenizer.encode(
                    text.lower() if tok_params.do_lower_case else text,
                    add_special_tokens=False,
                )
            )
        elif part["type"] == "reusable_content":
            content = contents[part["id"]]
            spans.append((len(ids), len(content["input_ids"]), content_indices[part["id"]]))
            ids.extend(content["input_ids"])
        else:
            data = part["data"]
            if data not in decoded_embeds:
                decoded_embeds[data] = await safe_load_prompt_embeds_async(
                    renderer.model_config, data.encode()
                )
            tensor = decoded_embeds[data]
            embeddings.append(tensor)
            embed_positions.append((len(ids), tensor.shape[0]))
            ids.extend([0] * tensor.shape[0])
    truncation = tok_params._truncation_slice(tokenizer, len(ids))
    if truncation is not None:
        left, right, _ = truncation.indices(len(ids))
        retained = []
        for start, length, index in spans:
            end = start + length
            if start < left < end or start < right < end:
                raise VLLMValidationError(
                    "Truncation cannot split reusable content",
                    parameter=tok_params.truncate_prompt_tokens_param,
                )
            if left <= start and end <= right:
                retained.append((start - left, length, index))
        spans = retained
    prompt = renderer._build_tokens_prompt(ids, prompt)
    if embeddings:
        prompt["prompt_embeds"], prompt["prompt_is_token_ids"] = _build_mixed_prompt_embeds(
            ids, embeddings, embed_positions
        )
    prompt = await renderer.tokenize_prompt_async(prompt, tok_params)
    renderer._apply_prompt_extras([prompt], prompt_extras or {})
    engine_input = await renderer.process_for_engine_async(
        prompt, time.time(), skip_mm_cache=skip_mm_cache
    )
    engine_input["reusable_input"] = ReusableInput(resolved, spans)
    return conversation, engine_input
