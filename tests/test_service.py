"""Protect content identity and native compilation through HTTP."""

import asyncio
import base64
import json
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
import torch
from fastapi import FastAPI
from pydantic import ValidationError
from vllm.config import KVTransferConfig, LoRAConfig
from vllm.lora.peft_helper import PEFTHelper

from relocache.backends.vllm.api import ReLoCacheEndpoints
from relocache.backends.vllm.config import engine_options
from relocache.backends.vllm.content import ContentService
from relocache.backends.vllm.protocol import ReusableChatRequest, ReusableResponsesRequest
from relocache.client import Client
from relocache.content import ContentError, ContentStore


class Engine:
    """Replace engine execution; content registration must not make an IPC call."""

    def __init__(self):
        self.engine_core = self
        self.vllm_config = SimpleNamespace(
            kv_transfer_config=None, cache_config=SimpleNamespace(block_size=16)
        )
        self.pending = []
        self.requests = []
        self.request_ids = []

    async def call_utility_async(self, method, *args, all_results=False):
        if args[0] == "has_requests":
            return [self.pending.pop(0) if self.pending else False]
        if args[0] == "forget_compile":
            return None
        assert args[0] == "compile_status"
        assert args[1][0] in self.request_ids
        return [
            {
                "pending": self.pending.pop(0) if self.pending else False,
                "compiled": True,
                "persisted": args[1][1],
            }
        ]

    async def generate(self, prompt, sampling, request_id, **kwargs):
        self.requests.append((prompt, sampling, kwargs))
        self.request_ids.append(request_id)
        yield SimpleNamespace(
            kv_transfer_params={"relocache": {"compile_request_id": request_id + "-native"}},
            metrics=None,
            num_cached_tokens=0,
            outputs=[
                SimpleNamespace(
                    index=0, text="answer", token_ids=[7], finish_reason="stop", stop_reason=None
                )
            ],
        )


@pytest.mark.parametrize(
    "rank,capacity,expected",
    [(1, None, 1), (2, None, 8), (4, None, 8), (12, None, 16), (4, 32, 32)],
)
def test_writer_rank_maps_to_native_capacity_and_preserves_explicit_config(
    tmp_path, rank, capacity, expected
):
    adapter = {"r": rank, "lora_alpha": 16, "target_modules": ["q_proj"]}
    (tmp_path / "adapter_config.json").write_text(json.dumps(adapter))
    engine = {} if capacity is None else {"max_lora_rank": capacity}
    options = engine_options(
        {"model": "/model", "method_id": "writer", "writer_adapter": tmp_path, "engine": engine}
    )
    native = LoRAConfig(max_lora_rank=options["max_lora_rank"])
    PEFTHelper.from_dict(adapter).validate_legal(native)
    assert native.max_lora_rank == expected
    assert engine == ({} if capacity is None else {"max_lora_rank": capacity})


@pytest.mark.parametrize("rank,capacity", [(16, 8), (513, None), (4, 4)])
def test_writer_rejects_insufficient_or_unsupported_native_capacity(tmp_path, rank, capacity):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": rank}))
    with pytest.raises(ValueError):
        options = engine_options(
            {
                "model": "/model",
                "method_id": "writer",
                "writer_adapter": tmp_path,
                "engine": {} if capacity is None else {"max_lora_rank": capacity},
            }
        )
        LoRAConfig(max_lora_rank=options["max_lora_rank"])


def test_responses_preserve_structured_parts_in_assistant_history():
    from vllm.entrypoints.openai.responses.utils import construct_input_messages

    messages = [
        {"role": "assistant", "content": [{"type": "prompt_embeds", "data": "encoded"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Context: ", "annotations": []},
                {"type": "reusable_content", "id": "document"},
                {"type": "output_text", "text": " End.", "annotations": []},
            ],
        },
    ]
    request = ReusableResponsesRequest(model="reader", input=messages)
    from relocache.backends.vllm.rendering import convert_response_input

    assert (
        construct_input_messages(request_input=request.input, item_converter=convert_response_input)
        == request.input
    )
    assert request.model_dump(mode="json")["input"] == messages


@pytest.mark.parametrize("persist_metadata", [False, True])
def test_catalog_reopens_and_isolates_writer_identity(tmp_path, persist_metadata):
    directory = tmp_path if persist_metadata else None
    store = ContentStore("writer-a", metadata_dir=directory)
    first = store.register("document", [1, 2, 3])
    assert store.register("alias", [1, 2, 3]).input_ids == first.input_ids
    store.register("document", [4, 5])
    assert first.input_ids == (1, 2, 3)
    assert store.resolve("document").input_ids == (4, 5)
    store = ContentStore("writer-a", metadata_dir=directory)
    if persist_metadata:
        assert store.resolve("alias").input_ids == (1, 2, 3)
        assert store.resolve("document").input_ids == (4, 5)
    else:
        with pytest.raises(ContentError, match="Unknown content"):
            store.resolve("alias")
        assert not list(tmp_path.iterdir())
    other = ContentStore("writer-b", metadata_dir=directory)
    with pytest.raises(ContentError, match="Unknown content"):
        other.resolve("alias")


@pytest.mark.parametrize("storage", [None, "cpu", "fs"])
@pytest.mark.parametrize("writer_soft", [False, True])
def test_http_compilation_uses_writer_lora_and_requires_persistent_kv(storage, writer_soft):
    async def check():
        engine = Engine()
        if storage is not None:
            engine.vllm_config.kv_transfer_config = KVTransferConfig(
                kv_connector="OffloadingConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "spec_name": "TieringOffloadingSpec",
                    "cpu_bytes_to_use": 4096,
                    "secondary_tiers": [{"type": "fs", "root_dir": "/kv"}]
                    if storage == "fs"
                    else [],
                },
            )
        writer = {"lora_name": "writer", "lora_int_id": 1, "lora_path": "/adapter"}
        config = {"method_id": "writer", "writer": writer}
        if writer_soft:
            engine.model_config = SimpleNamespace(
                enable_prompt_embeds=True,
                dtype=torch.float32,
                get_hidden_size=lambda: 4,
            )
            buffer = BytesIO()
            torch.save(torch.full((2, 4), 0.5), buffer)
            config["writer_soft_tokens"] = {
                "post": {"data": base64.b64encode(buffer.getvalue()).decode()},
            }
        store = ContentService(engine, config)
        app = FastAPI()
        app.state.relocache = store
        ReLoCacheEndpoints().attach_router(app)
        async with Client("http://service") as client:
            await client.http.aclose()
            client.http = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://service"
            )
            result = await client.compile("doc", input_ids=[21, 22])
            assert result["logical_length"] == (4 if writer_soft else 2)
            prompt, _sampling, kwargs = engine.requests[0]
            assert kwargs["lora_request"].lora_name == "writer"
            assert prompt["prompt_token_ids"][:2] == [21, 22]
            from relocache.backends.vllm.transport import decode_input

            source = decode_input(prompt["extension"])["contents"][0]
            assert source["input_ids"] == [21, 22]
            if writer_soft:
                tensor = source["writer_soft_tokens"]["post"]
                torch.testing.assert_close(tensor, torch.full((2, 4), 0.5))
            if storage == "fs":
                engine.pending = [True, False]
                await client.compile("persisted", input_ids=[31, 32], persist=True)
                assert not engine.pending
            else:
                with pytest.raises(httpx.HTTPStatusError) as error:
                    await client.compile("persisted", input_ids=[31, 32], persist=True)
                assert error.value.response.status_code == 400
                assert len(engine.requests) == 1
                with pytest.raises(ContentError):
                    await store.resolve({"id": "persisted"})

    asyncio.run(check())


@pytest.mark.parametrize("shortcuts", [False, True])
def test_connector_settings_survive_pre_rope_configuration(tmp_path, shortcuts):
    transfer = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": 4096,
            "eviction_policy": "arc",
            "key_encoding": "rope",
            "secondary_tiers": [{"type": "obj", "bucket_name": "existing"}],
        },
    }
    config = {
        "model": "/model",
        "method_id": "writer",
        "engine": {
            "kv_transfer_config": transfer,
        },
    }
    if shortcuts:
        config.update(cpu_bytes_to_use=8192, kv_dir=str(tmp_path))
    options = engine_options(config)
    extra = options["kv_transfer_config"]["kv_connector_extra_config"]
    assert extra["key_encoding"] == "pre_rope"
    assert extra["eviction_policy"] == "arc"
    assert extra["cpu_bytes_to_use"] == (8192 if shortcuts else 4096)
    assert extra["secondary_tiers"][0] == {"type": "obj", "bucket_name": "existing"}
    if shortcuts:
        assert extra["secondary_tiers"][1] == {"type": "fs", "root_dir": str(tmp_path)}
    assert transfer["kv_connector_extra_config"]["key_encoding"] == "rope"
    assert len(transfer["kv_connector_extra_config"]["secondary_tiers"]) == 1


@pytest.mark.parametrize(
    "request_class,field,text_type",
    [
        (ReusableChatRequest, "messages", "text"),
        (ReusableResponsesRequest, "input", "input_text"),
    ],
)
def test_reusable_schema_validates_references_without_changing_native_inputs(
    request_class,
    field,
    text_type,
):
    native = request_class.__bases__[0]
    ordinary = [{"role": "user", "content": [{"type": text_type, "text": "Question"}]}]
    assert request_class(**{field: ordinary}).model_dump(mode="json", include={field}) == native(
        **{field: ordinary}
    ).model_dump(mode="json", include={field})
    content = [
        {"type": text_type, "text": "Context: "},
        {"type": "prompt_embeds", "data": "encoded"},
        {"type": "reusable_content", "id": "document", "input_ids": [1, 2]},
    ]
    message = {"role": "user", "content": content}
    request = request_class(**{field: [message]})
    assert getattr(request, field) == [message]
    for reference in (
        {"type": "reusable_content"},
        {"type": "reusable_content", "id": "document", "input_ids": "invalid"},
        {"type": "reusable_content", "id": "document", "text": "text", "input_ids": [1]},
    ):
        with pytest.raises(ValidationError):
            request_class(**{field: [{"role": "user", "content": [reference]}]})


def test_responses_keep_native_tool_and_assistant_conversion():
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.entrypoints.openai.responses.utils import construct_input_messages

    items = [{"type": "function_call_output", "call_id": "call", "output": "done"}]
    assert ReusableResponsesRequest(input=items).model_dump(
        mode="json", include={"input"}
    ) == ResponsesRequest(input=items).model_dump(mode="json", include={"input"})
    messages = construct_input_messages(
        request_input=[
            {
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Answer"},
                    {"type": "refusal", "refusal": "No"},
                ],
            }
        ]
    )
    assert messages == [{"role": "assistant", "content": "Answer"}]


def test_logical_input_preserves_tokens_wrapping_and_source_snapshots():
    from relocache.inputs import assemble_input

    store = ContentStore("writer")
    source = store.register("doc", [21, 22])
    embedding = torch.full((2, 4), 0.25)
    updated = store.register("doc", [31])
    prepared = assemble_input(
        [[11], source, embedding, source, updated, [12]], pre_length=1, post_length=2
    )
    assert prepared.input_ids == [
        11,
        0,
        21,
        22,
        0,
        0,
        0,
        0,
        0,
        21,
        22,
        0,
        0,
        0,
        31,
        0,
        0,
        12,
    ]
    assert prepared.spans == [(1, 5, 0), (8, 5, 0), (13, 4, 1)]
    assert prepared.embedding_positions == [(6, 2)]
    assert prepared.contents[0].input_ids == (21, 22)
    assert prepared.contents[1].input_ids == (31,)
    assert prepared.retained_spans(6, 18) == [(2, 5, 0), (7, 4, 1)]
    with pytest.raises(ContentError, match="split"):
        prepared.retained_spans(2, 18)


def test_token_content_request_retains_native_sampling_and_rejects_ambiguous_inputs():
    from relocache.backends.vllm.protocol import ReusableCompletionRequest

    blocks = [
        {"type": "tokens", "input_ids": [1, 2]},
        {"type": "reusable_content", "id": "doc"},
        {"type": "prompt_embeds", "data": "encoded"},
        {"type": "tokens", "input_ids": [3]},
    ]
    request = ReusableCompletionRequest(
        prompt={"blocks": blocks}, max_tokens=8, temperature=0.2, seed=42
    )
    assert request.prompt == {"blocks": blocks}
    sampling = request.to_sampling_params(8, {})
    assert sampling.max_tokens == 8 and sampling.temperature == 0.2 and sampling.seed == 42
    for options in ({"prompt_embeds": "encoded"}, {"truncate_prompt_tokens": 2}):
        with pytest.raises(ValidationError):
            ReusableCompletionRequest(prompt={"blocks": blocks}, **options)
    for invalid in ([], [{"type": "tokens", "input_ids": [-1]}]):
        with pytest.raises(ValidationError):
            ReusableCompletionRequest(prompt={"blocks": invalid})


def test_compile_registered_id_and_content_errors_through_http():
    async def check():
        engine = Engine()
        app = FastAPI()
        app.state.relocache = ContentService(engine, {"method_id": "writer"})
        ReLoCacheEndpoints().attach_router(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://service"
        ) as client:
            registered = await client.post(
                "/v1/reusable_contents/register", json={"content_id": "doc", "input_ids": [21, 22]}
            )
            compiled = await client.post(
                "/v1/reusable_contents/compile", json={"content_id": "doc"}
            )
            assert compiled.status_code == 200
            assert compiled.json()["compilation"]["compiled"]
            assert compiled.json()["input_ids"] == registered.json()["input_ids"]
            assert engine.requests[0][0]["prompt_token_ids"] == [21, 22] + [0] * 15
            missing = await client.post(
                "/v1/reusable_contents/compile", json={"content_id": "missing"}
            )
            assert missing.status_code == 404
            assert missing.json()["error"]["code"] == "unknown_content"
            empty = await client.post(
                "/v1/reusable_contents/register", json={"content_id": "empty", "input_ids": []}
            )
            assert empty.status_code == 400
            assert empty.json()["error"]["code"] == "invalid_content"
            assert len(engine.requests) == 1

    asyncio.run(check())


def test_chat_and_token_blocks_share_native_tokens_embeddings_and_pic_positions(tmp_path):
    from functools import partial

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from vllm.config import ModelConfig, VllmConfig
    from vllm.renderers import ChatParams, TokenizeParams
    from vllm.renderers.hf import HfRenderer
    from vllm.renderers.online_renderer import OnlineRenderer

    from relocache.backends.vllm.protocol import ReusableCompletionRequest
    from relocache.backends.vllm.rendering import render_reusable_chat, render_reusable_completion

    tokenizer_impl = Tokenizer(WordLevel({"<unk>": 0, "Context": 1, "Answer": 2}))
    tokenizer_impl.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_impl, unk_token="<unk>")
    tokenizer.save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 3,
                "max_position_embeddings": 128,
            }
        )
    )
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            dtype="float32",
            max_model_len=128,
            enable_prompt_embeds=True,
        )
    )
    renderer = HfRenderer(config, tokenizer=tokenizer)

    def encode(tensor):
        buffer = BytesIO()
        torch.save(tensor, buffer)
        return {"data": base64.b64encode(buffer.getvalue()).decode()}

    service = ContentService(
        SimpleNamespace(model_config=config.model_config),
        {
            "method_id": "writer",
            "writer_soft_tokens": {
                "pre": encode(torch.ones(1, 4)),
                "post": encode(torch.ones(2, 4)),
            },
        },
    )
    soft = {"type": "prompt_embeds", **encode(torch.full((2, 4), 0.5))}
    reference = {"type": "reusable_content", "id": "doc", "input_ids": [1, 2]}
    template = "{{ messages[0]['content'] }}"

    async def check():
        online = OnlineRenderer(
            config.model_config,
            renderer,
            request_logger=None,
            chat_template=template,
            chat_template_content_format="string",
        )
        online.completion_input_renderer = partial(render_reusable_completion, service=service)
        request = ReusableCompletionRequest(
            prompt={
                "blocks": [
                    {"type": "tokens", "input_ids": [1]},
                    reference,
                    soft,
                    {"type": "reusable_content", "id": "doc"},
                    {"type": "tokens", "input_ids": [2]},
                ]
            },
            max_tokens=8,
        )
        (exact,) = await online.render_completion(request)
        _, chat = await render_reusable_chat(
            renderer,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Context "},
                        reference,
                        soft,
                        {"type": "reusable_content", "id": "doc"},
                        {"type": "text", "text": " Answer"},
                    ],
                }
            ],
            ChatParams(chat_template=template, chat_template_content_format="string"),
            TokenizeParams(max_total_tokens=128, max_output_tokens=8),
            service=service,
        )
        assert (
            exact["prompt_token_ids"]
            == chat["prompt_token_ids"]
            == [
                1,
                0,
                1,
                2,
                0,
                0,
                0,
                0,
                0,
                1,
                2,
                0,
                0,
                2,
            ]
        )
        assert (
            exact["extension"].data["spans"]
            == chat["extension"].data["spans"]
            == [
                (1, 5, 0),
                (8, 5, 0),
            ]
        )
        assert (
            exact["is_token_ids"] == chat["is_token_ids"] == [True] * 6 + [False] * 2 + [True] * 6
        )
        torch.testing.assert_close(exact["prompt_embeds"], chat["prompt_embeds"])
        torch.testing.assert_close(exact["prompt_embeds"][6:8], torch.full((2, 4), 0.5))
        ordinary = await online.render_completion(ReusableCompletionRequest(prompt=[1, 2]))
        assert ordinary[0]["prompt_token_ids"] == [1, 2]
        with pytest.raises(ContentError, match="end with Dynamic Content"):
            await online.render_completion(
                ReusableCompletionRequest(prompt={"blocks": [reference]})
            )

    try:
        asyncio.run(check())
    finally:
        renderer.shutdown()
