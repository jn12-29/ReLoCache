"""Export a SemPIC ICLR artifact and a real canonical dataset subset for the demo."""

import argparse
import base64
import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path

import torch
from peft import LoraConfig
from safetensors.torch import load_file, save_file


def export_artifact(artifact, output):
    metadata = json.loads((artifact / "method.json").read_text())
    if metadata["reduction"]["mode"] != "identity":
        raise ValueError("The native demo requires identity reduction")
    tensors = load_file(artifact / "method.safetensors")
    identity = hashlib.sha256()
    identity.update(
        json.dumps(
            {
                key: metadata.get(key)
                for key in (
                    "model_name_or_path",
                    "model_revision",
                    "writer",
                    "reduction",
                )
            },
            sort_keys=True,
        ).encode()
    )
    for name, tensor in sorted(tensors.items()):
        if name.startswith("soft_tokens."):
            continue
        identity.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        identity.update(tensor.view(torch.uint8).numpy().tobytes())
    writer = metadata["writer"]
    adapter = None
    if writer is not None:
        adapter = output / "writer-adapter"
        adapter.mkdir()
        LoraConfig(
            r=writer["rank"],
            lora_alpha=writer["alpha"],
            lora_dropout=writer["dropout"],
            target_modules=writer["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=True,
        ).save_pretrained(adapter)
        save_file(
            {
                name.removeprefix("model.").replace(".default.", "."): tensor
                for name, tensor in tensors.items()
                if ".lora_" in name
            },
            adapter / "adapter_model.safetensors",
        )
    soft = {}
    for name in ("pre", "post"):
        count = metadata["soft_tokens"][f"{name}_count"]
        if count:
            buffer = BytesIO()
            torch.save(tensors[f"soft_tokens.{name}"], buffer)
            soft[name] = {"count": count, "data": base64.b64encode(buffer.getvalue()).decode()}
    (output / "reader-soft-tokens.json").write_text(json.dumps(soft) + "\n")
    return metadata, identity.hexdigest(), adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--artifact", required=True, help="Path relative to source-repo")
    parser.add_argument(
        "--data", required=True, help="Canonical JSONL path relative to source-repo"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=1)
    args = parser.parse_args()
    source = args.source_repo.resolve()
    artifact = (source / args.artifact).resolve()
    data = (source / args.data).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata, method_id, adapter = export_artifact(artifact, output)
    counts = Counter()
    with data.open() as stream, (output / "inputs.jsonl").open("w") as destination:
        for line in stream:
            row = json.loads(line)
            if counts[row["dataset"]] >= args.samples_per_dataset:
                continue
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[row["dataset"]] += 1
    config = {
        "model": str((source / metadata["model_name_or_path"]).resolve()),
        "tokenizer": str((source / metadata["tokenizer_name_or_path"]).resolve()),
        "revision": metadata.get("model_revision"),
        "method_id": method_id,
        "writer_adapter": str(adapter) if adapter is not None else None,
        "served_model_name": "reader",
        "auto_compile": True,
        "engine": {
            "dtype": "bfloat16",
            "max_model_len": 16640,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 2048,
            "block_size": 16,
            "gpu_memory_utilization": 0.80,
            "enable_prompt_embeds": bool(
                metadata["soft_tokens"]["pre_count"] or metadata["soft_tokens"]["post_count"]
            ),
        },
    }
    (output / "deployment.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "source.json").write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "data": str(data),
                "metadata": metadata,
                "samples_per_dataset": dict(counts),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"bundle": str(output), "datasets": dict(counts), "method_id": method_id}))


if __name__ == "__main__":
    main()
