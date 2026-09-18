"""Bind source snapshots to the deployed Writer and native compilation."""

import asyncio
from uuid import uuid4

from vllm.renderers.embed_utils import safe_load_prompt_embeds

from relocache.content import ContentError, ContentStore

from .compiler import compile_inputs, compile_prompt, source_signature
from .transport import encode_input


class ContentService:
    def __init__(self, engine, config):
        self.engine = engine
        self.config = config
        self.store = ContentStore(config["method_id"], metadata_dir=config.get("metadata_dir"))
        self.writer_soft_tokens = {
            name: safe_load_prompt_embeds(engine.model_config, part["data"].encode())
            for name, part in config.get("writer_soft_tokens", {}).items()
        }
        self.pre_length = len(self.writer_soft_tokens.get("pre", ()))
        self.post_length = len(self.writer_soft_tokens.get("post", ()))

    def native_content(self, content):
        return {
            "content_id": content.content_id,
            "input_ids": list(content.input_ids),
            "method_id": self.config["method_id"],
            "writer": self.config.get("writer"),
            "writer_soft_tokens": self.writer_soft_tokens,
        }

    def public_content(self, content):
        return {
            "content_id": content.content_id,
            "input_ids": list(content.input_ids),
            "logical_length": self.pre_length + len(content.input_ids) + self.post_length,
            "source_registered": True,
        }

    async def register(self, payload):
        ids = payload.get("input_ids")
        if ids is None:
            ids = self.engine.renderer.get_tokenizer().encode(
                payload["text"], add_special_tokens=False
            )
        return self.store.register(payload["content_id"], ids)

    async def resolve(self, part):
        if "input_ids" in part or "text" in part:
            return await self.register({**part, "content_id": part["id"]})
        return self.store.resolve(part["id"])

    async def compile(self, payload):
        if payload.get("persist", False):
            transfer = self.engine.vllm_config.kv_transfer_config
            extra = transfer.kv_connector_extra_config if transfer is not None else {}
            if not (
                transfer is not None
                and transfer.kv_connector == "OffloadingConnector"
                and extra.get("spec_name") == "TieringOffloadingSpec"
                and any(tier["type"] in ("fs", "obj") for tier in extra.get("secondary_tiers", []))
            ):
                raise ContentError(
                    "persist requires a filesystem or object-store KV tier",
                    code="persistence_unavailable",
                )
        content = await self.resolve({**payload, "id": payload["content_id"]})
        native = self.native_content(content)
        _, sampling, lora = compile_inputs(native)
        prompt = compile_prompt(native, self.engine.vllm_config.cache_config.block_size)
        request_id = f"compile:{uuid4().hex}"
        prompt["extension"] = encode_input([native], compile=True, operation_id=request_id)
        persist = payload.get("persist", False)
        try:
            async for output in self.engine.generate(
                prompt, sampling, request_id, lora_request=lora
            ):
                if output.outputs and output.outputs[0].finish_reason == "error":
                    raise ContentError(
                        str(output.outputs[0].stop_reason), code="compilation_failed"
                    )
            metrics = output.kv_transfer_params["relocache"]
            while True:
                results = await self.engine.engine_core.call_utility_async(
                    "extension_call", "compile_status", [request_id, persist]
                )
                states = [r for r in (results if isinstance(results, list) else [results]) if r]
                state = states[0]
                if not state["pending"]:
                    break
                await asyncio.sleep(0.05)
            if persist and not state["persisted"]:
                raise ContentError(
                    "Compiled KV is absent from the configured persistent tiers",
                    code="persistence_failed",
                )
            return {**self.public_content(content), "compilation": state, "metrics": metrics}
        finally:
            await self.engine.engine_core.call_utility_async(
                "extension_call", "forget_compile", [request_id]
            )

    async def status(self, content_id):
        content = self.store.resolve(content_id)
        native = self.native_content(content)
        signature = source_signature(native)
        native.pop("writer_soft_tokens")
        native["signature"] = signature
        inspection_id = "inspect:" + uuid4().hex
        try:
            while True:
                result = await self.engine.engine_core.call_utility_async(
                    "extension_call", "content_status", [native, inspection_id]
                )
                replicas = result if isinstance(result, list) else [result]
                if not any(
                    v in ("RETRY", "HIT_PENDING")
                    for r in replicas
                    for values in r["tiers"].values()
                    for v in values
                ):
                    return {**self.public_content(content), "replicas": replicas}
                await asyncio.sleep(0.05)
        finally:
            await self.engine.engine_core.call_utility_async(
                "extension_call", "finish_inspection", [inspection_id]
            )

    def deployment(self):
        config = self.engine.vllm_config
        transfer = config.kv_transfer_config
        return {
            "model": config.model_config.model,
            "method_id": self.config["method_id"],
            "writer": self.config.get("writer"),
            "writer_pre_tokens": self.pre_length,
            "writer_post_tokens": self.post_length,
            "auto_compile": self.config["auto_compile"],
            "block_size": config.cache_config.block_size,
            "data_parallel_size": config.parallel_config.data_parallel_size,
            "tensor_parallel_size": config.parallel_config.tensor_parallel_size,
            "storage": transfer.kv_connector_extra_config if transfer else None,
        }

    async def flush(self):
        """Drain all native DP engines, including pending connector writes."""
        while await self.engine.engine_core.call_utility_async("storage_pending"):
            await asyncio.sleep(0.1)
