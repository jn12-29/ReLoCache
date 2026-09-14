"""Protect content identity and native compilation through HTTP."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from vllm.config import KVTransferConfig

from relocache.api import router
from relocache.client import Client
from relocache.config import engine_options
from relocache.content import ContentStore
from relocache.protocol import ReusableChatRequest, ReusableResponsesRequest


class Engine:
    """Replace engine execution; content registration must not make an IPC call."""

    def __init__(self):
        self.engine_core = self
        self.vllm_config = SimpleNamespace(kv_transfer_config=None)
        self.pending = []
        self.requests = []

    async def call_utility_async(self, method, *args):
        assert method == "reusable_storage_pending"
        return self.pending.pop(0) if self.pending else False

    async def generate(self, prompt, sampling, request_id, **kwargs):
        self.requests.append((prompt, sampling, kwargs))
        yield SimpleNamespace(
            metrics=None,
            num_cached_tokens=0,
            outputs=[
                SimpleNamespace(
                    index=0, text="answer", token_ids=[7], finish_reason="stop", stop_reason=None
                )
            ],
        )


def test_responses_preserve_structured_parts_in_assistant_history():
    from vllm.entrypoints.openai.responses.utils import construct_input_messages

    from relocache.protocol import ReusableResponsesRequest

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
    assert construct_input_messages(request_input=request.input) == request.input
    assert request.model_dump(mode="json")["input"] == messages


@pytest.mark.parametrize("persist_metadata", [False, True])
def test_catalog_reopens_and_isolates_writer_identity(tmp_path, persist_metadata):
    async def check():
        engine = Engine()
        config = {"method_id": "writer-a", "writer": None}
        if persist_metadata:
            config["metadata_dir"] = str(tmp_path)
        store = ContentStore(engine, config)
        first = await store.register({"content_id": "document", "input_ids": [1, 2, 3]})
        alias = await store.register({"content_id": "alias", "input_ids": [1, 2, 3]})
        assert alias["input_ids"] == first["input_ids"]
        assert set(first) == {"content_id", "input_ids", "method_id", "writer"}
        assert (await store.resolve({"id": "alias"}))["input_ids"] == [1, 2, 3]
        store = ContentStore(engine, config)
        if persist_metadata:
            assert (await store.resolve({"id": "alias"}))["input_ids"] == [1, 2, 3]
            assert (await store.resolve({"id": "document"}))["input_ids"] == [1, 2, 3]
        else:
            with pytest.raises(KeyError):
                await store.resolve({"id": "alias"})
            assert not list(tmp_path.iterdir())
        other = ContentStore(engine, {**config, "method_id": "writer-b"})
        with pytest.raises(KeyError):
            await other.resolve({"id": "document"})
        second = await other.register({"content_id": "document", "input_ids": [1, 2, 3]})
        assert second["method_id"] != first["method_id"]
        assert not engine.requests

    asyncio.run(check())


@pytest.mark.parametrize("storage", [None, "cpu", "fs"])
def test_http_compilation_uses_writer_lora_and_requires_persistent_kv(storage):
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
        store = ContentStore(engine, {"method_id": "writer", "writer": writer})
        app = FastAPI()
        app.state.relocache = store
        app.include_router(router)
        async with Client("http://service") as client:
            await client.http.aclose()
            client.http = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://service"
            )
            await client.compile("doc", input_ids=[21, 22])
            prompt, sampling, kwargs = engine.requests[0]
            assert kwargs["lora_request"].lora_name == "writer"
            assert prompt["prompt_token_ids"] == [21, 22]
            assert sampling.extra_args["pic_compile"]["input_ids"] == [21, 22]
            if storage == "fs":
                engine.pending = [True, False]
                await client.compile("persisted", input_ids=[31, 32], persist=True)
                assert not engine.pending
            else:
                with pytest.raises(httpx.HTTPStatusError) as error:
                    await client.compile("persisted", input_ids=[31, 32], persist=True)
                assert error.value.response.status_code == 400
                assert len(engine.requests) == 1
                with pytest.raises(KeyError):
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
