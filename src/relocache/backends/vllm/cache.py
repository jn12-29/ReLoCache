# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable-content hashing and request coordinates, without cache ownership."""

from bisect import bisect_right
from itertools import accumulate
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.request import Request


from relocache.layout import (
    PICContentLayout,
    PICLayout,
    PICOccurrence,
    dynamic_intervals,
    map_dynamic_intervals,
    merge_segments,
)


def dynamic_content_ranges(request: "Request") -> list[tuple[int, int]]:
    """Canonical intervals remaining after reusable occurrences are removed."""
    return dynamic_intervals(
        len(request.canonical_prompt_token_ids),
        [(span.start, span.length) for span in request.reusable_spans],
    )


def pack_prompt_embeds(request: "Request"):
    """Give the native worker storage-ordered inputs; hash logical inputs unchanged."""
    embeds, mask = request.prompt_embeds, request.prompt_is_token_ids
    layout = request.pic_layout
    if layout is None or embeds is None:
        return embeds, mask
    mask = cast("list[bool]", mask)
    packed = embeds.new_zeros((layout.storage_length, embeds.shape[1]))
    packed_mask = [True] * layout.storage_length
    for storage, logical, length in layout.dynamic_content_ranges:
        packed[storage : storage + length] = embeds[logical : logical + length]
        packed_mask[storage : storage + length] = mask[logical : logical + length]
    return packed, packed_mask


def hash_reusable_request(
    request: "Request", block_size: int, hash_function: "Callable[[Any], bytes]"
) -> list["BlockHash"]:
    """Hash packed Dynamic Content by canonical dependencies, not storage order."""
    from vllm.v1.core.kv_cache_utils import (
        generate_block_hash_extra_keys,
        hash_block_tokens,
    )

    request._pic_hash_block_size = block_size
    prompt = request._pic_dynamic_content_prompt_token_ids
    hashes = request._pic_dynamic_content_hashes
    old_count = len(hashes)
    num_tokens = len(prompt) + request.num_output_tokens
    if num_tokens < (old_count + 1) * block_size:
        return []
    canonical_length = len(cast("list[int]", request.canonical_prompt_token_ids))
    ranges = dynamic_content_ranges(request)
    ranges.append((canonical_length, canonical_length + request.num_output_tokens))

    range_ends = list(accumulate(end - start for start, end in ranges))

    def canonical_position(index: int) -> int:
        range_index = bisect_right(range_ends, index)
        prior_end = range_ends[range_index - 1] if range_index else 0
        return ranges[range_index][0] + index - prior_end

    for offset in range(old_count * block_size, num_tokens - block_size + 1, block_size):
        end = offset + block_size
        canonical_end = canonical_position(end - 1) + 1
        previous_end = canonical_position(offset - 1) + 1 if offset else 0
        extra, _ = generate_block_hash_extra_keys(request, previous_end, canonical_end, 0)
        crossed = tuple(
            (
                span.start,
                span.length,
                span.content["compiled_id"],
            )
            for span in request.reusable_spans
            if previous_end <= span.start < canonical_end
        )
        if crossed:
            extra = (*(extra or ()), ("reusable", crossed))
        block_tokens = prompt[offset:end]
        if end > len(prompt):
            block_tokens.extend(
                request.output_token_ids[max(0, offset - len(prompt)) : end - len(prompt)]
            )
        hashes.append(
            hash_block_tokens(hash_function, hashes[-1] if hashes else None, block_tokens, extra)
        )
    # Before admission, only the semantic hashes exist. Storage ordering is
    # installed immediately before the native computed-prefix lookup.
    return hashes[old_count:] if request.pic_layout is not None else []


def prepare_reusable_request(request: "Request", manager: "KVCacheManager") -> None:
    """Place cache hits before native execution, then let native admission own them."""
    block_size = request._pic_hash_block_size
    last_span = request.reusable_spans[-1]
    if last_span.start + last_span.length == len(
        cast("list[int]", request.canonical_prompt_token_ids)
    ):
        raise NotImplementedError(
            "Terminal reusable content requires the Writer's final observation"
        )
    group_ids = list(range(len(manager.kv_cache_config.kv_cache_groups)))
    dynamic_content_hashes = request._pic_dynamic_content_hashes
    dynamic_content_ids = request._pic_dynamic_content_prompt_token_ids
    # Ordinary prefix lookup runs before the independent reusable lookups.
    _, cached_dynamic_content_tokens, _ = manager.coordinator.find_longest_cache_hit(
        dynamic_content_hashes,
        max(
            0,
            min(
                len(dynamic_content_ids),
                len(dynamic_content_ids) + request.num_output_tokens - 1,
            ),
        ),
    )
    cached_dynamic_content_blocks = cached_dynamic_content_tokens // block_size
    gpu_pages = []
    remote_pages = []
    page_occurrences: dict[BlockHash, list[tuple[int, int]]] = {}
    for occurrence_index, span in enumerate(request.reusable_spans):
        # Shared FullAttention block geometry: all layers use the same padded
        # occurrence capacity, while runtime clips each layer to its own R.
        hashes = span.content["groups"][0]["block_hashes"]
        for page, block_hash in enumerate(hashes):
            if block_hash in page_occurrences:
                page_occurrences[block_hash].append((occurrence_index, page))
                continue
            page_occurrences[block_hash] = [(occurrence_index, page)]
            if manager.block_pool.get_cached_block(block_hash, group_ids) is None:
                remote_pages.append(block_hash)
            else:
                gpu_pages.append(block_hash)

    pages = gpu_pages + remote_pages
    contents: list[PICContentLayout] = []
    content_indices = {}
    occurrences = []
    for span in request.reusable_spans:
        content = span.content
        compiled_id = content["compiled_id"]
        if compiled_id not in content_indices:
            content_indices[compiled_id] = len(contents)
            contents.append(
                PICContentLayout(
                    content["logical_length"],
                    {
                        name: layer["entry_positions"]
                        for group in content["groups"]
                        for name, layer in group["layers"].items()
                        if layer["entry_positions"] is not None
                    },
                )
            )
        occurrences.append(PICOccurrence(span.start, content_indices[compiled_id], []))
    storage_hashes = dynamic_content_hashes[:cached_dynamic_content_blocks]
    for page_index, block_hash in enumerate(pages):
        storage_start = (cached_dynamic_content_blocks + page_index) * block_size
        for occurrence_index, entry_page in page_occurrences[block_hash]:
            occurrences[occurrence_index].segments.append(
                (storage_start, entry_page * block_size, block_size)
            )
        storage_hashes.append(block_hash)
    for occurrence in occurrences:
        occurrence.segments = merge_segments(occurrence.segments)
    storage_hashes.extend(dynamic_content_hashes[cached_dynamic_content_blocks:])

    prefix_slots = len(pages) * block_size
    mapped_dynamic_content = map_dynamic_intervals(
        dynamic_content_ranges(request), cached_dynamic_content_tokens, prefix_slots
    )

    storage_ids = (
        dynamic_content_ids[:cached_dynamic_content_tokens]
        + [0] * prefix_slots
        + dynamic_content_ids[cached_dynamic_content_tokens:]
    )
    request.pic_layout = PICLayout(
        len(cast("list[int]", request.canonical_prompt_token_ids)),
        len(storage_ids),
        mapped_dynamic_content,
        occurrences,
        contents,
        [name for group in manager.kv_cache_config.kv_cache_groups for name in group.layer_names],
    )
    request.prompt_token_ids = storage_ids
    request.num_prompt_tokens = len(storage_ids)
    request._all_token_ids[:] = storage_ids
    request._all_token_ids.extend(request.output_token_ids)
    request.block_hashes = storage_hashes
