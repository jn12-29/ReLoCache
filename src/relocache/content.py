"""Service-owned source catalog and explicit native compilation."""

import asyncio
import json
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from vllm.inputs import tokens_input
from vllm.renderers.embed_utils import safe_load_prompt_embeds
from vllm.v1.engine.pic_compiler import compile_inputs


class ContentStore:
    def __init__(self, engine, config):
        self.engine = engine
        self.config = config
        self.contents = {}
        self.writer_soft_tokens = {
            name: safe_load_prompt_embeds(engine.model_config, part["data"].encode())
            for name, part in config.get("writer_soft_tokens", {}).items()
        }
        directory = config.get("metadata_dir")
        self.metadata_dir = (
            Path(directory) / f"writer-{quote(config['method_id'], safe='')}"
            if directory is not None
            else None
        )
        if self.metadata_dir is not None:
            self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def metadata_path(self, content_id):
        return self.metadata_dir / f"content-{quote(content_id, safe='')}.json"

    async def register(self, payload):
        ids = payload.get("input_ids")
        if ids is None:
            ids = self.engine.renderer.get_tokenizer().encode(
                payload["text"],
                add_special_tokens=False,
            )
        if not ids:
            raise ValueError("Reusable content must contain at least one token")
        existing = self.contents.get(payload["content_id"])
        if existing is not None and existing["input_ids"] == ids:
            return existing
        content = self.prepare(payload["content_id"], ids)
        if self.metadata_dir is not None:
            self.metadata_path(payload["content_id"]).write_text(
                json.dumps({"content_id": payload["content_id"], "input_ids": ids}) + "\n"
            )
        return content

    def prepare(self, content_id, ids):
        content = {
            "content_id": content_id,
            "input_ids": ids,
            "method_id": self.config["method_id"],
            "writer": self.config.get("writer"),
        }
        if self.writer_soft_tokens:
            content["writer_soft_tokens"] = self.writer_soft_tokens
        self.contents[content_id] = content
        return content

    async def resolve(self, part):
        content_id = part["id"]
        if "input_ids" in part or "text" in part:
            return await self.register({**part, "content_id": content_id})
        if content_id in self.contents:
            return self.contents[content_id]
        if self.metadata_dir is not None:
            try:
                metadata = json.loads(self.metadata_path(content_id).read_text())
            except FileNotFoundError:
                raise KeyError(content_id) from None
            return self.prepare(content_id, metadata["input_ids"])
        raise KeyError(content_id)

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
                raise ValueError("persist requires a filesystem or object-store KV tier")
        content = await self.register(payload)
        ids, sampling, lora = compile_inputs(content)
        async for _ in self.engine.generate(
            tokens_input(ids),
            sampling,
            f"compile:{uuid4().hex}",
            lora_request=lora,
        ):
            pass
        if payload.get("persist", False):
            await self.flush()
        return content

    async def flush(self):
        """Drain all native DP engines, including pending connector writes."""
        while await self.engine.engine_core.call_utility_async("reusable_storage_pending"):
            await asyncio.sleep(0.1)
