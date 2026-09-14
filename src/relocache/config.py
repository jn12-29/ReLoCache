"""Translate a deployment description into explicit vLLM serving options."""

import json
from copy import deepcopy
from pathlib import Path


def engine_options(config):
    options = deepcopy(config.get("engine", {}))
    options.update(
        model=config["model"],
        revision=config.get("revision"),
        tokenizer=config.get("tokenizer", config["model"]),
        served_model_name=[config.get("served_model_name", "reader")],
        attention_backend="TRITON_ATTN",
        enable_prefix_caching=True,
        enable_per_request_metrics=True,
        disable_log_stats=False,
    )
    options.setdefault("api_server_count", 1)
    writer_path = config.get("writer_adapter")
    writer = None
    if writer_path:
        writer_path = Path(writer_path).resolve()
        adapter = json.loads((writer_path / "adapter_config.json").read_text())
        options.update(enable_lora=True, max_lora_rank=adapter["r"])
        writer = {"lora_name": config["method_id"], "lora_int_id": 1, "lora_path": str(writer_path)}
    options["additional_config"] = {
        **options.get("additional_config", {}),
        "position_free_kv": True,
        "pic": {"auto_compile": config.get("auto_compile", True)},
        "relocache": {
            "method_id": config["method_id"],
            "writer": writer,
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
        options["kv_transfer_config"].setdefault("kv_connector_extra_config", {})[
            "key_encoding"
        ] = "pre_rope"
    return options
