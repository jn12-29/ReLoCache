"""Native vLLM serving and execution integration."""


def register():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    from . import rotary  # noqa: F401

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "relocache.backends.vllm.attention.PICAttentionBackend",
    )
