"""PIC metadata and kernel arguments on the native Triton attention backend."""

from dataclasses import dataclass
from typing import Any

import torch
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)


@dataclass
class PICAttentionMetadata(TritonAttentionMetadata):
    # Stable per-sequence GPU pointer table; zero selects ordinary attention.
    # Each target is int64[R, 4]: cumulative entry end, physical slot start,
    # canonical start, local float32 position pointer (zero for contiguous positions).
    range_end: Any = None
    range_map: Any = None
    key_transform: Any = None
    logical_ranges: torch.Tensor | None = None
    logical_range_counts: torch.Tensor | None = None
    query_positions: torch.Tensor | None = None
    rope_inv_freq: torch.Tensor | None = None
    cos_sin_cache: torch.Tensor | None = None
    rope_is_neox_style: bool = True
    rope_attention_scaling: float = 1.0


class PICAttentionMetadataBuilder(TritonAttentionMetadataBuilder):
    def build(self, *args, **kwargs):
        return PICAttentionMetadata(**vars(super().build(*args, **kwargs)))


class PICAttentionImpl(TritonAttentionImpl):
    def get_kernel_kwargs(self, attn_metadata):
        return {
            "range_end": attn_metadata.range_end,
            "range_map": attn_metadata.range_map,
            "key_transform": attn_metadata.key_transform,
            "logical_ranges": attn_metadata.logical_ranges,
            "logical_range_counts": attn_metadata.logical_range_counts,
            "query_positions": attn_metadata.query_positions,
            "key_transform_tensors": (attn_metadata.rope_inv_freq, attn_metadata.cos_sin_cache),
            "key_transform_config": (
                2 * attn_metadata.rope_inv_freq.numel()
                if attn_metadata.rope_inv_freq is not None
                else 0,
                attn_metadata.rope_is_neox_style,
                attn_metadata.rope_attention_scaling,
            ),
        }


class PICAttentionBackend(TritonAttentionBackend):
    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return PICAttentionImpl

    @staticmethod
    def get_builder_cls():
        return PICAttentionMetadataBuilder
