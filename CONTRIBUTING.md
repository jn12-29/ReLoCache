# Contributing

Report bugs and discuss changes in the
[ReLoCache issue tracker](https://github.com/jn12-29/ReLoCache/issues), including
issues involving the ReLoCache-vLLM integration. Provide both Git revisions,
model and Writer configuration, GPU/software versions, a minimal request, and
the observed result. Remove credentials and private input from shared logs.

## Development environment

```bash
git submodule update --init --recursive
uv sync --python 3.12 --locked --extra server --extra webchat --extra demo
uv run --no-sync ruff check src tests examples
CUDA_VISIBLE_DEVICES='' uv run --no-sync pytest -q
node --test tests/test_webchat_stream.mjs tests/test_webchat_conversation.mjs
```

The CPU command skips CUDA tests. To run attention checks on an available GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync pytest -q tests/test_pic_attention.py
```

Use a device allocated to you. Changes affecting model execution require an
appropriate GPU/service check; the existing validation scope is documented in
[validation notes](docs/decoupling-validation.md). Do not infer method quality or
throughput from a successful smoke test.

## Ownership

PIC semantics and tests belong in ReLoCache. Keep native allocation, scheduling,
offload, and attention loops in vLLM; prefer its extension points. Keep training,
dataset policy, and scoring outside the service package.

`third_party/vllm` is a separate repository. Follow its contribution instructions,
commit changes there first, and then update the parent submodule reference.
Describe behavior, test results, and any remaining limitations in the PR. Include
AI assistance attribution when used. Contributions are licensed under Apache-2.0.

## Package checks

```bash
uv build
```

The wheel and source distribution contain the client/service package, browser
assets, and license notices. The source distribution includes documentation and
examples but not the vLLM submodule; use a recursive Git checkout for serving.
Do not upload model weights, private datasets, adapters, caches, or run outputs.
