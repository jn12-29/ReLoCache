"""Bind source snapshots to the deployed Writer and native compilation."""

import asyncio
from uuid import uuid4

from vllm.inputs import tokens_input
from vllm.renderers.embed_utils import safe_load_prompt_embeds

from relocache.content import ContentError, ContentStore

from .compiler import compile_inputs


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
        ids, sampling, lora = compile_inputs(self.native_content(content))
        async for _ in self.engine.generate(
            tokens_input(ids), sampling, f"compile:{uuid4().hex}", lora_request=lora
        ):
            pass
        if payload.get("persist", False):
            await self.flush()
        return content

    async def flush(self):
        """Drain all native DP engines, including pending connector writes."""
        while await self.engine.engine_core.call_utility_async("reusable_storage_pending"):
            await asyncio.sleep(0.1)
