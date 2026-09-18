"""Translate a deployment description into explicit vLLM serving options."""

import json
from copy import deepcopy
from pathlib import Path
from typing import get_args


def engine_options(config):
    options = deepcopy(config.get("engine", {}))
    options.update(
        model=config["model"],
        revision=config.get("revision"),
        tokenizer=config.get("tokenizer", config["model"]),
        served_model_name=[config.get("served_model_name", "reader")],
        attention_backend="CUSTOM",
        enable_prefix_caching=True,
        enable_per_request_metrics=True,
        disable_log_stats=False,
    )
    options["scheduler_cls"] = (
        "relocache.backends.vllm.scheduler.PICScheduler"
        if options.get("async_scheduling") is False
        else "relocache.backends.vllm.scheduler.PICAsyncScheduler"
    )
    options.setdefault("api_server_count", 1)
    if config.get("writer_soft_tokens"):
        options["enable_prompt_embeds"] = True
    writer_path = config.get("writer_adapter")
    writer = None
    if writer_path:
        from vllm.config.lora import MaxLoRARanks

        writer_path = Path(writer_path).resolve()
        adapter = json.loads((writer_path / "adapter_config.json").read_text())
        rank = adapter["r"]
        if "max_lora_rank" not in options:
            capacities = get_args(MaxLoRARanks)
            if rank > max(capacities):
                raise ValueError(f"Writer rank {rank} exceeds vLLM's maximum {max(capacities)}")
            options["max_lora_rank"] = min(capacity for capacity in capacities if capacity >= rank)
        if options["max_lora_rank"] < rank:
            raise ValueError(
                f"max_lora_rank={options['max_lora_rank']} is smaller than Writer rank {rank}"
            )
        options["enable_lora"] = True
        writer = {"lora_name": config["method_id"], "lora_int_id": 1, "lora_path": str(writer_path)}
    options["additional_config"] = {
        **options.get("additional_config", {}),
        "position_free_kv": True,
        "online_renderer_cls": "relocache.backends.vllm.rendering.PICOnlineRenderer",
        "model_state_cls": "relocache.backends.vllm.worker.PICModelStateAdapter",
        "relocache": {
            "auto_compile": config.get("auto_compile", True),
            "method_id": config["method_id"],
            "writer": writer,
            "writer_soft_tokens": config.get("writer_soft_tokens", {}),
            "metadata_dir": (
                str(Path(config["metadata_dir"]).resolve())
                if config.get("metadata_dir") is not None
                else None
            ),
        },
    }
    cpu_bytes = config.get("cpu_bytes_to_use")
    kv_dir = config.get("kv_dir")
    if cpu_bytes is not None or kv_dir is not None:
        transfer = options.get("kv_transfer_config") or {}
        options["kv_transfer_config"] = transfer
        transfer.setdefault("kv_connector", "OffloadingConnector")
        transfer.setdefault("kv_role", "kv_both")
        extra = transfer.setdefault("kv_connector_extra_config", {})
        extra.setdefault("spec_name", "TieringOffloadingSpec")
        extra.setdefault("block_size", options.get("block_size", 16))
        if cpu_bytes is not None:
            extra["cpu_bytes_to_use"] = cpu_bytes
        if kv_dir is not None and "cpu_bytes_to_use" not in extra:
            raise ValueError("kv_dir requires an explicit cpu_bytes_to_use staging budget")
        if kv_dir is not None:
            extra.setdefault("secondary_tiers", []).append(
                {"type": "fs", "root_dir": str(Path(kv_dir).resolve())}
            )
    if options.get("kv_transfer_config") is not None:
        extra = options["kv_transfer_config"].setdefault("kv_connector_extra_config", {})
        for tier in extra.get("secondary_tiers", []):
            if tier["type"] == "fs":
                tier["root_dir"] = str(Path(tier["root_dir"]) / "pre_rope")
            elif tier["type"] == "obj":
                prefix = tier.get("prefix", "").rstrip("/")
                tier["prefix"] = f"{prefix}/pre_rope" if prefix else "pre_rope"
    return options
