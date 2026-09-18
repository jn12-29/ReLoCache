"""Compile only confirmed misses; wait for native promotions and shared Writers."""


def defer_missing_content(scheduler, request, num_computed_tokens: int) -> bool:
    """Enqueue all confirmed missing documents on this Reader's native engine."""
    from vllm.v1.kv_offload.base import LookupResult, make_offload_key

    from relocache.backends.vllm.compiler import compile_inputs, compile_prompt
    from relocache.compilation import compilation_action

    from .request import PICRequest
    from .transport import input_metadata

    if num_computed_tokens >= request.pic_layout.precomputed_length:
        return False
    auto_compile = scheduler.vllm_config.additional_config["relocache"]["auto_compile"]
    offload = scheduler.offload
    pool = scheduler.kv_cache_manager.block_pool
    contents = {span.content["compiled_id"]: span.content for span in request.reusable_spans}
    pending = False
    for compiled_id, content in contents.items():
        if compiled_id in scheduler.compiles.active:
            pending = True
            continue
        missing = False
        for group_id, group in enumerate(content["groups"]):
            for block_hash in group["block_hashes"]:
                if pool.get_cached_block(block_hash, [group_id]) is not None:
                    continue
                # Native lookup also starts SSD promotion. Pending loads are
                # hits in progress, not permission to recompile the document.
                result = (
                    offload.lookup_key(request.request_id, make_offload_key(block_hash, group_id))
                    if offload
                    else LookupResult.MISS
                )
                if result in (LookupResult.HIT_PENDING, LookupResult.RETRY):
                    pending = True
                if result is LookupResult.MISS:
                    missing = True
                    break
            if missing:
                break
        action = compilation_action(missing=missing, pending=pending, auto_compile=auto_compile)
        if action in ("ready", "wait"):
            continue
        if action == "fail":
            scheduler.admission_errors[request.request_id] = "reusable_cache_miss"
            return True
        _, params, lora = compile_inputs(content)
        params.extra_args = {"relocache": input_metadata([content], compile=True, internal=True)}
        prompt = compile_prompt(
            {**content, "writer_soft_tokens": scheduler.writer_soft_tokens},
            content["groups"][0]["block_size"],
        )
        child = PICRequest(
            f"pic:{compiled_id}",
            prompt["prompt_token_ids"],
            params,
            None,
            lora_request=lora,
            prompt_embeds=prompt.get("prompt_embeds"),
            prompt_is_token_ids=prompt.get("is_token_ids"),
            block_hasher=request._block_hasher,
            arrival_time=request.arrival_time,
            priority=request.priority,
        )
        scheduler.add_request(child)
        request.pic_metrics["auto_compiles"] += 1
        pending = True
    return pending
