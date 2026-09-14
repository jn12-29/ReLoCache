"""Compare real LoRA PIC generation with ordinary Full Prefill on the same source text."""

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from time import perf_counter

from rouge_score.rouge_scorer import RougeScorer

from relocache.client import completed_requests
from relocache.server import Server

ROUGE = RougeScorer(["rougeL"], use_stemmer=True)
MAX_TOKENS = {
    "hotpotqa": 48,
    "2wikimqa": 48,
    "musique": 48,
    "ruler_mv": 128,
    "ruler_mq": 128,
    "ruler_vt": 128,
}


def score(prediction, row):
    answers = row["answers"]
    if not answers:
        return 0.0
    if row["dataset"].startswith("ruler_"):
        return sum(answer.lower() in prediction.lower() for answer in answers) / len(answers)
    return max(ROUGE.score(answer, prediction)["rougeL"].fmeasure for answer in answers)


def full_messages(row):
    contents = {item["content_id"]: item["text"] for item in row["reusable_contents"]}
    messages = []
    for message in row["messages"]:
        if not isinstance(message.get("content"), list):
            messages.append(message)
            continue
        text = "".join(
            contents[part["id"]] if part["type"] == "reusable_content" else part["text"]
            for part in message["content"]
        )
        messages.append({**message, "content": text})
    return messages


def pic_messages(row, soft):
    contents = {item["content_id"]: item["text"] for item in row["reusable_contents"]}
    messages = []
    occurrence = 0
    for message in row["messages"]:
        if not isinstance(message.get("content"), list):
            messages.append(message)
            continue
        parts = []
        for part in message["content"]:
            if part["type"] != "reusable_content":
                parts.append(part)
                continue
            enabled = (
                row["occurrence_soft_tokens"][occurrence]
                if "occurrence_soft_tokens" in row
                else (True, True)
            )
            if "pre" in soft and enabled[0]:
                parts.append({"type": "prompt_embeds", "data": soft["pre"]["data"]})
            parts.append({**part, "text": contents[part["id"]]})
            if "post" in soft and enabled[1]:
                parts.append({"type": "prompt_embeds", "data": soft["post"]["data"]})
            occurrence += 1
        messages.append({**message, "content": parts})
    return messages


async def command_output(*command):
    process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE)
    stdout, _ = await process.communicate()
    if process.returncode:
        raise RuntimeError(f"Command failed: {command}")
    return stdout.decode()


async def evaluate(args):
    bundle, output = args.bundle.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((bundle / "deployment.json").read_text())
    config["engine"].update(
        data_parallel_size=args.data_parallel_size, tensor_parallel_size=args.tensor_parallel_size
    )
    if args.cpu_bytes_to_use is not None:
        config["cpu_bytes_to_use"] = args.cpu_bytes_to_use
    if args.kv_dir is not None:
        config["kv_dir"] = str(args.kv_dir.resolve())
    rows = [json.loads(line) for line in (bundle / "inputs.jsonl").read_text().splitlines()]
    soft = json.loads((bundle / "reader-soft-tokens.json").read_text())
    run = {
        "start_time": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "command": list(sys.argv),
        "config": config,
        "bundle": str(bundle),
        "api_base": args.api_base,
        "concurrency": args.concurrency,
        "seed": args.seed,
        "max_tokens": MAX_TOKENS,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source": json.loads((bundle / "source.json").read_text()),
        "code": {
            name: {
                "commit": (await command_output("git", "-C", path, "rev-parse", "HEAD")).strip(),
                "status": await command_output("git", "-C", path, "status", "--short"),
            }
            for name, path in (("relocache", "."), ("vllm", "third_party/vllm"))
        },
    }
    if args.api_base is None:
        # Inspect devices immediately before the managed server launch.
        run["gpu_snapshot"] = await command_output("nvidia-smi")
        print(run["gpu_snapshot"], flush=True)
    (output / "config.json").write_text(json.dumps(run, indent=2) + "\n")
    server = Server(config, output, api_base=args.api_base)
    records = []
    started = perf_counter()
    async with server as client:

        async def generate(row):
            sampling = {
                "temperature": 0,
                "seed": args.seed,
                "max_tokens": MAX_TOKENS[row["dataset"]],
            }
            payload = {
                "model": config["served_model_name"],
                "chat_template_kwargs": row.get("chat_template_kwargs", {}),
                "return_token_ids": True,
                "stream": False,
                **sampling,
            }
            if "chat_template" in row:
                payload["chat_template"] = row["chat_template"]
            full = await client.post(
                "/v1/chat/completions", {**payload, "messages": full_messages(row)}
            )
            pic = await client.post(
                "/v1/chat/completions", {**payload, "messages": pic_messages(row, soft)}
            )
            result = {"id": row["id"], "dataset": row["dataset"], "answers": row["answers"]}
            for name, response in (("full_prefill", full), ("pic", pic)):
                choice = response["choices"][0]
                prediction = choice["message"]["content"] or ""
                result[name] = {**response, "score": score(prediction, row)}
            return result

        with (output / "predictions.jsonl").open("w") as stream:
            async for result in completed_requests(generate, rows, args.concurrency):
                records.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                print(
                    f"{len(records)}/{len(rows)} {result['dataset']}: "
                    f"Full={result['full_prefill']['score']:.4f} PIC={result['pic']['score']:.4f}",
                    flush=True,
                )
        await client.flush()
    summary = {
        "samples": len(records),
        "completed": len(records) == len(rows),
        "elapsed_seconds": perf_counter() - started,
        "startup_seconds": server.startup_seconds,
        "quality_evaluation": True,
        "timing_scope": "Non-streaming HTTP latency with connection reuse; includes on-demand "
        "compilation and native cache lookup/load. This is not a controlled latency benchmark.",
        "results": {
            name: {
                "macro_score": fmean(record[name]["score"] for record in records),
                "datasets": {
                    dataset: {
                        "samples": sum(record["dataset"] == dataset for record in records),
                        "macro_score": fmean(
                            record[name]["score"]
                            for record in records
                            if record["dataset"] == dataset
                        ),
                        "metric": "string_match_all"
                        if dataset.startswith("ruler_")
                        else "rouge_l_fmeasure",
                    }
                    for dataset in sorted({record["dataset"] for record in records})
                },
            }
            for name in ("full_prefill", "pic")
        },
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-bytes-to-use", type=int)
    parser.add_argument("--kv-dir", type=Path)
    args = parser.parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
