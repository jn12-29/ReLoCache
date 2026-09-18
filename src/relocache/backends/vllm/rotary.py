"""Query-only RoPE through vLLM's out-of-tree CustomOp registry."""

from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.llama3_rope import Llama3RotaryEmbedding


class PositionFreeKeys:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = get_current_vllm_config_or_none()
        self.position_free_kv = bool(
            config and config.additional_config.get("position_free_kv", False)
        )
        self.attention_scaling = 1.0

    def _compute_inv_freq(self, base):
        frequencies = super()._compute_inv_freq(base)
        self.register_buffer("inv_freq", frequencies, persistent=False)
        return frequencies

    def forward(self, positions, query, key=None):
        query, rotated_key = super().forward(
            positions, query, None if self.position_free_kv else key
        )
        return query, key if self.position_free_kv else rotated_key


class PICRotaryEmbedding(PositionFreeKeys, RotaryEmbedding):
    pass


class PICLlama3RotaryEmbedding(PositionFreeKeys, Llama3RotaryEmbedding):
    pass


RotaryEmbedding.register_oot(PICRotaryEmbedding)
Llama3RotaryEmbedding.register_oot(PICLlama3RotaryEmbedding)
# Preserve native CustomOp enable/disable configuration names.
PICRotaryEmbedding.name = RotaryEmbedding.name
PICLlama3RotaryEmbedding.name = Llama3RotaryEmbedding.name
