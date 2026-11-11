# ReLoCache

Reusable KV caching for LLM serving: compile content once with a Writer, then
reuse its KV at different positions in Reader requests.

ReLoCache implements position-independent caching (PIC), content management,
compilation, logical position mapping, and fused pre-RoPE attention. The current
backend is a pinned [ReLoCache-vLLM](https://github.com/jn12-29/ReLoCache-vLLM)
fork. Native vLLM owns KV pages, scheduling, prefix caching, and offloading.

PIC content is compiled independently of the Reader's preceding context. It
changes the computation compared with full causal prefill; evaluate answer
quality for your model and task. It is not a numerically equivalent replacement
for full prefill.

## Install

Serving requires Linux, an NVIDIA GPU supported by the pinned vLLM build, and
[uv](https://docs.astral.sh/uv/getting-started/installation/). Python 3.12 is the
validated development environment.

```bash
git clone --recurse-submodules https://github.com/jn12-29/ReLoCache.git
cd ReLoCache
uv sync --python 3.12 --locked --extra server --extra webchat
uv run --no-sync relocache serve --help
```

For an existing checkout, run `git submodule update --init --recursive` first.
The `server` extra uses the local fork through `tool.uv.sources`; an unrelated
PyPI vLLM installation cannot replace it. Installation downloads precompiled
extensions for the pinned upstream commit; keep the build settings in
`pyproject.toml`. Model weights and trained adapters are not included.

For an HTTP client only, install the base package from the checkout in your own
environment (`uv pip install .`). It requires neither vLLM nor Torch. The
`webchat` extra provides a standalone browser UI and proxy; `demo` adds artifact
conversion and scoring dependencies.

## First request

The checked-in configuration uses the public `Qwen/Qwen3-0.6B` model and a base
Writer, with no trained adapter. Model files are downloaded on first use; set
`model` to a local model directory for offline serving.

```bash
uv run --no-sync relocache serve \
  --config examples/quickstart/deployment.json \
  --cuda-visible-devices 0 --host 127.0.0.1 --port 8000
```

Choose an available GPU and adjust the memory budget for your hardware. Wait
for `curl --fail http://127.0.0.1:8000/health`, then use another terminal:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/reusable_contents/register \
  -H 'Content-Type: application/json' \
  -d '{"content_id":"document","text":"The archive access code is ORCHID-42."}'

curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"reader",
    "messages":[{"role":"user","content":[
      {"type":"text","text":"Read the document:\n"},
      {"type":"reusable_content","id":"document"},
      {"type":"text","text":"\nWhat is the archive access code?"}
    ]}],
    "chat_template_kwargs":{"enable_thinking":false},
    "temperature":0,
    "max_tokens":64
  }'
```

The first reference automatically compiles missing content. Later requests can
reuse it across different prefixes and positions. Explicit compilation, cache
status, exact token blocks, streaming, and the Python client are documented in
the [usage guide (中文)](docs/user-guide.md).

## Web Chat

For the built-in long-document demo, start the API server with the larger
context configuration:

```bash
uv run --no-sync relocache serve \
  --config examples/webchat/deployment.json \
  --cuda-visible-devices 0 --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/demo`. The page compares Full Prefill and PIC, streams
answers, and reports TTFT and cache statistics. Full Prefill retains native
prefix caching. The default document and UI are in Chinese; you can replace
the document and questions. The demo uses a base Writer; supply your own
matching standard PEFT `writer_adapter` to evaluate a trained Writer.

For a separate UI process:

```bash
uv run --no-sync relocache webchat \
  --api-base http://127.0.0.1:8000 --host 127.0.0.1 --port 8080
```

Visit `http://127.0.0.1:8080/demo`. Ctrl-C stops each process separately. The
optional cache reset button requires the API server to start with
`VLLM_SERVER_DEV_MODE=1`; the reset endpoint is a development feature.

## Capabilities and boundaries

- Ordered Dynamic Content and Reusable Content, including repeated references.
- Base or LoRA Writer; Writer Pre/Post soft tokens and separate Reader embeddings.
- Completions, Chat Completions, Responses, SSE, and a lightweight async client.
- Native GPU caching, CPU offload, and filesystem KV persistence with optional
  source metadata persistence.
- Request-level compilation, reuse, loading, and prefill statistics.

Qwen3 execution, numerical attention checks, DP, chunked prefill, CUDA Graph,
and filesystem reload have historical validation recorded in
[validation notes](docs/decoupling-validation.md). These are execution checks,
not a formal quality or throughput benchmark. Object-store end-to-end serving,
other attention backends, compression, and selective recomputation are not
covered by those checks. The Reader input currently ends with Dynamic Content.

SemPIC training, datasets, prompt conventions, and scoring remain outside the
service. `examples/sempic` provides an evaluation client and an optional converter
for the documented research artifact format; neither is needed for quickstart.

## Development and documentation

- [Usage guide (中文)](docs/user-guide.md): deployment, API, soft tokens, and evaluation.
- [Integration boundary](docs/integration.md): PIC semantics and backend contracts.
- [vLLM patch boundary](docs/vllm-patch.md): cumulative execution changes.
- [Roadmap](docs/roadmap.md), [provenance](SOURCE.md), and [contributing](CONTRIBUTING.md).

Both repositories use `main`. Commit fork changes before updating the parent
submodule reference. Report ReLoCache integration issues in
[this repository](https://github.com/jn12-29/ReLoCache/issues).

## License

ReLoCache is licensed under [Apache-2.0](LICENSE). The vLLM fork retains its
upstream Apache-2.0 license and attribution. See [NOTICE](NOTICE) and the bundled
[browser dependency licenses](src/relocache/webchat_assets/THIRD_PARTY_LICENSES.txt).
Model weights, adapters, and datasets have their own licenses.
