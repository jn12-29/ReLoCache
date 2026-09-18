# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logical PIC traversal and fused pre-RoPE key rotation."""

import triton
import triton.language as tl


@triton.jit
def logical_visible_end(ranges, range_count, last_query):
    lo, hi = 0, range_count
    while lo < hi:
        mid = (lo + hi) // 2
        start = tl.load(ranges + mid * 4 + 2)
        before = start <= last_query
        lo = tl.where(before, mid + 1, lo)
        hi = tl.where(before, hi, mid)
    last_range = lo - 1
    end = tl.load(ranges + last_range * 4)
    start = tl.load(ranges + last_range * 4 + 2)
    entries = tl.load(ranges + last_range * 4 + 3)
    previous = tl.load(ranges + (last_range - 1) * 4, mask=last_range > 0, other=0)
    return tl.where(entries == 0, tl.minimum(end, previous + last_query - start + 1), end).to(
        tl.int32
    )


@triton.jit
def map_entries(ranges, range_count, seq_offset, tile_mask, TILE_SIZE: tl.constexpr):
    # Binary search the ordered range ends for each live entry.
    # Alignment PAD has no entry ordinal and incurs no K/V load.
    lo = tl.full((TILE_SIZE,), 0, tl.int32)
    hi = tl.full((TILE_SIZE,), range_count, tl.int32)
    span = range_count
    while span > 0:
        mid = (lo + hi) // 2
        end = tl.load(ranges + mid * 4, mask=mid < range_count, other=0)
        before = (mid < range_count) & (end <= seq_offset)
        lo = tl.where(before, mid + 1, lo)
        hi = tl.where(before, hi, mid)
        span = span // 2
    previous = tl.load(ranges + (lo - 1) * 4, mask=tile_mask & (lo > 0), other=0)
    offset = seq_offset - previous
    first = tl.load(ranges + lo * 4 + 1, mask=tile_mask, other=0)
    start = tl.load(ranges + lo * 4 + 2, mask=tile_mask, other=0)
    entries = tl.load(ranges + lo * 4 + 3, mask=tile_mask, other=0)
    has_entry_positions = entries != 0
    local_positions = tl.load(
        entries.to(tl.pointer_type(tl.float32)) + offset,
        mask=tile_mask & has_entry_positions,
        other=0,
    )
    logical_k = start + tl.where(has_entry_positions, local_positions, offset)
    cache_offset = first + offset
    return cache_offset, logical_k


@triton.jit
def rotate_keys(
    K,
    logical_k,
    offs_d,
    tile_mask,
    dtype: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    tensors,
    config: tl.constexpr,
):
    rope_inv_freq_ptr, cos_sin_cache_ptr = tensors
    ROTARY_DIM: tl.constexpr = config[0]
    ROPE_IS_NEOX_STYLE: tl.constexpr = config[1]
    ROPE_ATTENTION_SCALING: tl.constexpr = config[2]
    if ROPE_IS_NEOX_STYLE:
        partner_dim = (offs_d + ROTARY_DIM // 2) % ROTARY_DIM
        frequency_index = offs_d % (ROTARY_DIM // 2)
        sign = tl.where(offs_d < ROTARY_DIM // 2, -1.0, 1.0)
    else:
        partner_dim = offs_d ^ 1
        frequency_index = offs_d // 2
        sign = tl.where(offs_d % 2 == 0, -1.0, 1.0)
    # Reuse the loaded K tile in registers, as MiniPIC reuses its two
    # half-head tiles. No second KV load or in-place cache rotation.
    partner = tl.gather(
        K.to(tl.float32),
        tl.broadcast_to(partner_dim[:, None], (HEAD_SIZE_PADDED, TILE_SIZE)),
        axis=0,
    )
    if cos_sin_cache_ptr is not None:
        position = logical_k.to(tl.int32)
        offset = position[None, :] * ROTARY_DIM + frequency_index[:, None]
        cosine = tl.load(
            cos_sin_cache_ptr + offset,
            mask=(offs_d < ROTARY_DIM)[:, None] & tile_mask[None, :],
            other=0,
        ).to(tl.float32)
        sine = tl.load(
            cos_sin_cache_ptr + offset + ROTARY_DIM // 2,
            mask=(offs_d < ROTARY_DIM)[:, None] & tile_mask[None, :],
            other=0,
        ).to(tl.float32)
    else:
        frequency = tl.load(
            rope_inv_freq_ptr + frequency_index,
            mask=offs_d < ROTARY_DIM,
            other=0.0,
        )
        angle = frequency[:, None] * logical_k[None, :]
        cosine = tl.cos(angle) * ROPE_ATTENTION_SCALING
        sine = tl.sin(angle) * ROPE_ATTENTION_SCALING
    rotated = K.to(tl.float32) * cosine + partner * sign[:, None] * sine
    K = tl.where((offs_d < ROTARY_DIM)[:, None], rotated, K).to(dtype)
    return K
