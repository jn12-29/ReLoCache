"""PIC coordination on the native scheduler, allocator and offload lifecycle."""

import time
from dataclasses import asdict

from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from relocache.compilation import CompilationTracker
from relocache.layout import reused_token_count

from .cache import pack_prompt_embeds, prepare_reusable_request
from .compiler import prepare_content
from .coordination import defer_missing_content
from .request import PICRequest, pic_hasher
from .storage import finish_inspection, inspect_tiers
from .transport import decode_input, encode_input


class PICSchedulerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compiles = CompilationTracker()
        self.compilations = {}

    @property
    def offload(self):
        return self.connector.connector_scheduler if self.connector is not None else None

    def create_request(self, request, block_hasher):
        if request.extension is not None:
            data = decode_input(request.extension)
            contents = [
                prepare_content(
                    c, self.vllm_config.model_config, self.kv_cache_config.kv_cache_groups
                )
                for c in data["contents"]
            ]
            request.extension = encode_input(
                contents,
                data["spans"],
                compile=data["compile"],
                internal=data["internal"],
                operation_id=data["operation_id"],
            )
        hasher = pic_hasher(
            block_hasher,
            self.hash_block_size,
            get_hash_fn_by_name(self.vllm_config.cache_config.prefix_caching_hash_algo),
        )
        return PICRequest.from_engine_core_request(request, hasher)

    def _get_local_prefix_cache_hit(self, request):
        if request.reusable_spans and not (
            self.offload and self.offload.has_request_transfers(request.request_id)
        ):
            prepare_reusable_request(request, self.kv_cache_manager)
            request.install_layout()
        return super()._get_local_prefix_cache_hit(request)

    def defer_request(self, request, num_cached_tokens):
        if not request.reusable_spans:
            return False
        pending = defer_missing_content(self, request, num_cached_tokens)
        if (
            pending or num_cached_tokens < request.required_cached_tokens
        ) and request.wait_started is None:
            request.wait_started = time.monotonic()
        return pending

    def add_request(self, request):
        super().add_request(request)
        if request.pic_compile is not None:
            key = request.pic_compile["compiled_id"]
            self.compiles.start(key, request.request_id)
            if not request.internal:
                self.compilations[request.operation_id] = (request.request_id, request.pic_compile)

    def _free_request(self, request, delay_free_blocks=False):
        if request.pic_compile is not None:
            key = request.pic_compile["compiled_id"]
            compile_seconds = self.compiles.finish(key, request.request_id)
            if str(request.get_finished_reason()) == "error":
                for reader in self.requests.values():
                    if any(span.content["compiled_id"] == key for span in reader.reusable_spans):
                        self.admission_errors[reader.request_id] = "reusable_compilation_failed"
        kv_params, ec_params = super()._free_request(request, delay_free_blocks)
        if request.reusable_spans or request.pic_compile is not None:
            metrics = dict(request.pic_metrics)
            metrics["loads"] = request.cache_transfer_stats
            if request.pic_compile is not None:
                metrics["compile_seconds"] = compile_seconds
                metrics["compiled_id"] = request.pic_compile["compiled_id"]
                metrics["compile_request_id"] = request.request_id
            if request.pic_layout is not None:
                layout = request.pic_layout
                metrics["reused_tokens"] = reused_token_count(layout)
                metrics["reused_pages"] = len(
                    {
                        h
                        for span in request.reusable_spans
                        for h in span.content["groups"][0]["block_hashes"]
                    }
                )
            kv_params = {**(kv_params or {}), "relocache": metrics}
        return kv_params, ec_params

    def _update_after_schedule(self, output):
        for data in output.scheduled_new_reqs:
            request = self.requests[data.req_id]
            data.prompt_embeds, data.prompt_is_token_ids = pack_prompt_embeds(request)
            if request.pic_layout is not None:
                data.worker_metadata = asdict(request.pic_layout)
        for rid, count in output.num_scheduled_tokens.items():
            request = self.requests[rid]
            if request.wait_started is not None:
                request.pic_metrics["wait_seconds"] += time.monotonic() - request.wait_started
                request.wait_started = None
            request.pic_metrics["prefill_tokens_executed"] += max(
                0, min(count, request.num_prompt_tokens - request.num_computed_tokens)
            )
        super()._update_after_schedule(output)

    def inspect_content(self, content, request_id):
        from vllm.v1.kv_offload.base import make_offload_key

        entries = [
            (group_id, h) for group_id, g in enumerate(content["groups"]) for h in g["block_hashes"]
        ]
        gpu = sum(
            self.kv_cache_manager.block_pool.get_cached_block(h, [g]) is not None
            for g, h in entries
        )
        tiers = {}
        if self.offload:
            tiers = inspect_tiers(
                self.offload.manager, [make_offload_key(h, g) for g, h in entries], request_id
            )
        return {"pages": len(entries), "gpu_pages": gpu, "tiers": tiers}

    def finish_inspection(self, request_id):
        if self.offload:
            finish_inspection(self.offload.manager, request_id)

    def forget_compile(self, operation_id):
        operation = self.compilations.pop(operation_id, None)
        if operation is not None:
            self.finish_inspection(operation[0])

    def compile_status(self, operation_id, persist):
        operation = self.compilations.get(operation_id)
        if operation is None:
            return None
        rank = self.parallel_config.data_parallel_index
        request_id, content = operation
        pending = persist and (
            request_id in self.requests
            or bool(self.offload and self.offload.has_request_work(request_id))
        )
        if pending:
            return {"rank": rank, "pending": True}
        state = self.inspect_content(content, request_id) if persist else None
        if state and any(
            v in ("RETRY", "HIT_PENDING") for values in state["tiers"].values() for v in values
        ):
            return {"rank": rank, "pending": True}
        durable = (
            [
                values
                for name, values in state["tiers"].items()
                if name.split(":")[0] in ("fs", "obj")
            ]
            if state
            else []
        )
        return {
            "rank": rank,
            "pending": False,
            "compiled": True,
            "persisted": all(all(v == "HIT" for v in values) for values in durable)
            if durable
            else (False if persist else None),
            "cache": state,
        }

    def content_status(self, payload, inspection_id):
        content = prepare_content(
            payload, self.vllm_config.model_config, self.kv_cache_config.kv_cache_groups
        )
        return {
            "rank": self.parallel_config.data_parallel_index,
            "compiled_id": content["compiled_id"],
            "compiling": content["compiled_id"] in self.compiles.active,
            **self.inspect_content(content, inspection_id),
        }


class PICScheduler(PICSchedulerMixin, Scheduler):
    pass


class PICAsyncScheduler(PICSchedulerMixin, AsyncScheduler):
    pass
