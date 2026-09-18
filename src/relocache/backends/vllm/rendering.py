# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve reusable references while retaining canonical model token IDs."""

from dataclasses import replace
from uuid import uuid4

from vllm.renderers.embed_utils import safe_load_prompt_embeds_async
from vllm.renderers.hf import _build_mixed_prompt_embeds
from vllm.v1.core.pic import ReusableInput

from relocache.content import ContentError
from relocache.inputs import assemble_input


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


async def resolve_blocks(parts, renderer, service):
    blocks, decoded_embeds = [], {}
    for part in parts:
        if part["type"] == "tokens":
            blocks.append(part["input_ids"])
        elif part["type"] == "reusable_content":
            blocks.append(await service.resolve(part))
        else:
            data = part["data"]
            if data not in decoded_embeds:
                decoded_embeds[data] = await safe_load_prompt_embeds_async(
                    renderer.model_config, data.encode()
                )
            blocks.append(decoded_embeds[data])
    return assemble_input(blocks, pre_length=service.pre_length, post_length=service.post_length)


async def render_reader_input(
    renderer, prepared, tok_params, service, *, prompt_extras=None, skip_mm_cache=False
):
    truncation = tok_params._truncation_slice(renderer.tokenizer, len(prepared.input_ids))
    left, right, _ = (truncation or slice(None)).indices(len(prepared.input_ids))
    spans = prepared.retained_spans(left, right)
    if spans and spans[-1][0] + spans[-1][1] == right - left:
        raise ContentError("Reader input must end with Dynamic Content")
    prompt = {"prompt_token_ids": prepared.input_ids}
    if prepared.embeddings:
        prompt["prompt_embeds"], prompt["prompt_is_token_ids"] = _build_mixed_prompt_embeds(
            prepared.input_ids, prepared.embeddings, prepared.embedding_positions
        )
    (engine_input,) = await renderer.render_cmpl_async(
        [prompt], tok_params, prompt_extras=prompt_extras, skip_mm_cache=skip_mm_cache
    )
    if spans:
        engine_input["reusable_input"] = ReusableInput(
            [service.native_content(content) for content in prepared.contents], spans
        )
    return engine_input


async def render_reusable_completion(renderer, request, *, service, skip_mm_cache=False):
    if not isinstance(request.prompt, dict):
        return None
    prepared = await resolve_blocks(request.prompt["blocks"], renderer, service)
    return [
        await render_reader_input(
            renderer,
            prepared,
            request.build_tok_params(renderer.model_config),
            service,
            prompt_extras={"cache_salt": request.cache_salt},
            skip_mm_cache=skip_mm_cache,
        )
    ]


async def render_reusable_chat(
    renderer,
    messages,
    chat_params,
    tok_params,
    *,
    service,
    prompt_extras=None,
    skip_mm_cache=False,
):
    if not has_reusable_content(messages):
        return None
    messages, markers = prepare_reusable_messages(messages)
    params = replace(
        chat_params,
        chat_template_kwargs={**chat_params.chat_template_kwargs, "tokenize": False},
    )
    conversation, prompt = await renderer.render_messages_async(messages, params)
    if prompt.get("multi_modal_data"):
        raise ContentError("Reusable rendering currently supports text inputs")
    tokenizer = renderer.get_tokenizer()
    parts = []
    for text, part in iter_reusable_spans(prompt["prompt"], markers):
        if part is None:
            part = {
                "type": "tokens",
                "input_ids": tokenizer.encode(
                    text.lower() if tok_params.do_lower_case else text, add_special_tokens=False
                ),
            }
        parts.append(part)
    prepared = await resolve_blocks(parts, renderer, service)
    engine_input = await render_reader_input(
        renderer,
        prepared,
        tok_params,
        service,
        prompt_extras=prompt_extras,
        skip_mm_cache=skip_mm_cache,
    )
    return conversation, engine_input
