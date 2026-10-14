# ReLoCache 使用指南：PIC 推理与评测

本文面向使用 ReLoCache 执行 PIC 推理、接入训练产物或开展评测的使用者。环境准备、配置、请求、评测数据格式和结果解释均在本文中给出，无需阅读源码或其他项目文档。以下命令从 **ReLoCache 仓库根目录**执行。

当前实现使用仓库固定版本的 vLLM fork。已完成 Qwen3-0.6B、Qwen3-8B 的相关执行验证；这不表示任意模型、attention 配置和并行组合均已验证。本文最小示例使用本机已有的 `../models/qwen3-0.6b`，服务模型名为 `reader`。

## 1. 先理解输入与方法

PIC 将文档独立编译成可复用的 KV，Reader 在当前请求中引用这些 KV，并计算问题等动态输入。文档可以跨请求、跨位置复用，同一请求也可以多次引用。独立编译的文档不包含 Reader 前文产生的状态，因此不能把 PIC 当成与 Full Prefill 数值等价的缓存优化；需要评测具体方法的质量。

| 名称 | 使用者提供什么 | 执行方式 |
| --- | --- | --- |
| Reusable Content | 文档文本或 token IDs，再用内容 ID 引用 | Writer 独立编译，Reader 复用 KV |
| Dynamic Content | 指令、问题、普通 tokens、Reader embeddings | 在 Reader 当前上下文中执行，可命中原生前缀缓存 |
| Writer | 基础模型，可选 LoRA adapter 和编译用 soft tokens | 生成文档 KV |
| Reader | 有序输入与生成参数 | 读取文档 KV，生成答案 |

方法通过配置和输入组合表达，API 没有 `method="sempic"` 这样的算法开关：

| 用法 | 部署中的 `writer_adapter` | 部署中的 `writer_soft_tokens` | Reader 输入 |
| --- | --- | --- | --- |
| Full Prefill / full recompute | 不用于该普通请求 | 不用于该普通请求 | 直接提交完整文本或 tokens，不含内容引用 |
| Base PIC / no recompute | 省略 | 省略 | 内容引用 + 动态输入 |
| SemPIC | 标准 PEFT Writer 路径 | 省略 | 内容引用，以及训练约定要求的 Reader soft tokens |
| KV Packet | 省略 | Writer Pre/Post embeddings | 内容引用 + 动态输入 |
| Joint | 标准 PEFT Writer 路径 | Writer Pre/Post embeddings | 内容引用 + 动态输入；Reader soft tokens 按训练约定提供 |

Writer soft tokens 进入文档编译并影响缓存身份；Reader soft tokens 是请求中的动态输入，服务不会从 Writer 配置自动插入。不要将仅用于 Reader 的训练产物直接放进 `writer_soft_tokens`。

## 2. 环境准备

需要 Linux、可用的 NVIDIA GPU/驱动、`uv`、Git，以及模型文件。本机约定只使用空闲的 GPU 4–7，不中断其他任务；模型从 `../models` 读取。新环境、缓存、产物均保存在本仓库，不修改旧研究仓库或环境。

首次获取仓库时，需要有两个私有仓库的读取权限：

```bash
git clone --recurse-submodules git@github.com:jn12-29/ReLoCache.git
cd ReLoCache
```

已在仓库内则从以下命令开始；`submodule update` 使用主仓库固定的提交，不跟随 fork 最新分支：

```bash
git submodule update --init --recursive
uv sync --locked --extra server --extra demo
uv run --no-sync relocache serve --help
```

项目要求 Python 3.12 或以上；`uv` 管理项目 `.venv`。`server` 安装本地 vLLM fork 和 Torch，`demo` 安装产物转换及评分依赖。项目已配置预编译 vLLM 扩展及其基线提交，安装可能需要下载对应 wheel；不要用 PyPI 的独立 vLLM 覆盖本地 fork。已有可用环境直接使用后文的 `uv run --no-sync` 即可。

只运行 HTTP 客户端的环境可安装 ReLoCache 基础包，无需 `server` / `demo` extras；本文的 tokenizer、soft-token 导出和评测例子使用上述完整环境。

## 3. 最小部署与第一次 PIC 推理

### 3.1 创建部署配置

```bash
mkdir -p artifacts/quickstart
cat > artifacts/quickstart/deployment.json <<'JSON'
{
  "model": "../models/qwen3-0.6b",
  "tokenizer": "../models/qwen3-0.6b",
  "served_model_name": "reader",
  "method_id": "qwen3-0.6b-base-pic-v1",
  "auto_compile": true,
  "metadata_dir": "artifacts/quickstart/metadata",
  "engine": {
    "dtype": "bfloat16",
    "max_model_len": 4096,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 512,
    "block_size": 16,
    "gpu_memory_utilization": 0.5
  }
}
JSON
```

`method_id` 标识本次部署的 Writer 语义。更换模型权重、revision、Writer adapter 或编译方法时使用新 ID，避免把不同 Writer 的缓存混用。服务会将内容 tokens 和 Writer soft-token 值等加入缓存身份，但不会替使用者推断 adapter 文件内容是否发生变化。`served_model_name` 是生成请求中 `model` 的值，与模型文件路径不同。

路径相对于启动时的工作目录；跨目录部署建议使用绝对路径。`metadata_dir` 仅保存来源 tokens，重启后可以继续解析内容 ID；它不保存 KV。

### 3.2 启动服务

先查看设备状态，确认 GPU 4 空闲；否则将下面的 `4` 换成另一张空闲的 4–7 号卡：

```bash
nvidia-smi
uv run --no-sync relocache serve \
  --config artifacts/quickstart/deployment.json \
  --cuda-visible-devices 4 \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --host 127.0.0.1 \
  --port 8000
```

该命令前台运行。另开终端，仍在仓库根目录执行：

```bash
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/relocache
```

health 成功、模型列表包含 `reader` 后再发送生成请求。结束时在服务终端按 Ctrl-C。

### 3.3 注册内容并引用

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/reusable_contents/register \
  -H 'Content-Type: application/json' \
  -d '{"content_id":"demo-doc","text":"The access code for the archive is ORCHID-42."}'

curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"reader",
    "messages":[{"role":"user","content":[
      {"type":"text","text":"Read the document:\n"},
      {"type":"reusable_content","id":"demo-doc"},
      {"type":"text","text":"\nWhat is the access code? Reply with the code only."}
    ]}],
    "chat_template_kwargs":{"enable_thinking":false},
    "temperature":0,
    "max_tokens":64
  }'
```

`enable_thinking=false` 用于本例 Qwen3 模板；换模型时按该模型模板设置。答案位于 `choices[0].message.content`。首次请求若缺少文档 KV，会自动提交 Writer；后续请求可以复用。短模型、基础 Writer 的回答质量需由实际输出判断。

查看当前状态：

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/reusable_contents/demo-doc/status
```

返回的 `replicas` 列出各 DP 副本的 `compiled_id`、`compiling`、`pages`、`gpu_pages` 和各存储层状态。注册成功不等于编译完成；编译完成也不保证 KV 永久驻留。

## 4. 内容管理与 HTTP 接口

| 方法 | 路径 | 请求与作用 |
| --- | --- | --- |
| POST | `/v1/reusable_contents/register` | `content_id` + `text` 或 `input_ids`，注册/更新来源，不编译 |
| POST | `/v1/reusable_contents/compile` | `content_id`，可附来源，`persist` 默认 false；等待本次编译完成 |
| GET | `/v1/reusable_contents/{content_id}` | 返回来源 `input_ids`、`logical_length`、`source_registered` |
| GET | `/v1/reusable_contents/{content_id}/status` | 查询各 DP 副本当前驻留；查询不会触发编译或 KV 晋升 |
| POST | `/v1/reusable_contents/flush` | 空 JSON 请求，等待所有 DP 副本的请求和待完成写入排空 |
| GET | `/v1/relocache` | 部署摘要：模型、Writer、Pre/Post 长度、自动编译、TP/DP、存储 |
| POST | `/v1/completions` | 原生 prompt 或本指南的 `prompt.blocks` |
| POST | `/v1/chat/completions` | 原生 messages，可包含内容引用和 Reader embeddings |
| POST | `/v1/responses` | 原生 input，可包含内容引用和 Reader embeddings |

来源只能提供 `text`、`input_ids` 之一，且不能是空内容。文本注册使用部署 tokenizer，`add_special_tokens=False`；精确控制 tokens 时直接提供 `input_ids`。示例 ID 使用简单字符串。

显式编译已注册文档：

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/reusable_contents/compile \
  -H 'Content-Type: application/json' \
  -d '{"content_id":"demo-doc","persist":false}'
```

也可以附带 `text` 或 `input_ids`，一次完成注册与编译。返回包含内容信息、`compilation` 和 `metrics`。`compilation.rank` 表示执行编译的 DP 副本；一次显式编译不表示所有副本都已预热。

内容引用也允许内联注册：`{"type":"reusable_content","id":"demo-doc","text":"..."}`。同一 ID 更新来源只影响后续解析；已经解析的引用绑定旧的不可变快照。每个请求按块顺序解析。相同内容可在一个请求中重复引用，各引用保留自己的逻辑位置并共享可复用页。

`flush` 不清缓存，也不会将未编译的来源自动编译。持续有其他请求时它可能持续等待；单文档持久化应使用 `compile(persist=true)`。

## 5. 输入格式与 Python 调用

### 5.1 精确 token 输入

`POST /v1/completions` 的 `prompt.blocks` 支持：

| 块类型 | JSON 字段 | 含义 |
| --- | --- | --- |
| `tokens` | `input_ids: list[int]` | 非空、非负 token IDs，原样执行 |
| `reusable_content` | `id`，可附 `text` 或 `input_ids` | 文档引用 |
| `prompt_embeds` | `data: string` | base64 编码的 `torch.save` tensor，作为 Reader 动态输入 |

服务按顺序计算位置，不需要调用方提交 spans 或 offsets。每次引用占用 Writer Pre + document + Post 的完整逻辑长度。

以下例子自己分词并同时调用 Full Prefill/PIC；完整 prompt 的特殊 tokens 和模板在真实评测中应由调用方按训练约定准备。此例使用第 3 节的 Base Writer 配置，不含 soft tokens：

```bash
uv run --no-sync python - <<'PY'
import asyncio
from transformers import AutoTokenizer
from relocache.client import Client

async def main():
    tokenizer = AutoTokenizer.from_pretrained("../models/qwen3-0.6b")
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    prefix = encode("Read the document:\n")
    document = encode("The access code for the archive is ORCHID-42.")
    suffix = encode("\nWhat is the access code?\nAnswer:")
    async with Client("http://127.0.0.1:8000") as client:
        await client.register("token-doc", input_ids=document)
        compiled = await client.compile("token-doc")
        generation = {"model": "reader", "max_tokens": 32, "temperature": 0}
        full = await client.post("/v1/completions", {
            **generation, "prompt": prefix + document + suffix,
        })
        pic = await client.complete([
            {"type": "tokens", "input_ids": prefix},
            {"type": "reusable_content", "id": "token-doc"},
            {"type": "tokens", "input_ids": suffix},
        ], **generation)
        print("Compilation:", compiled["compilation"])
        print("Full Prefill:", full["choices"][0]["text"])
        print("PIC:", pic["choices"][0]["text"])
        print("PIC metrics:", pic.get("kv_transfer_params"))

asyncio.run(main())
PY
```

`prompt.blocks` 不做解码后重新分词，也不自动插入特殊 tokens。它不支持 `truncate_prompt_tokens`、`echo`、`prompt_logprobs` 或顶层 `prompt_embeds`；embeddings 放在块内。当前 Reader 输入必须以 Dynamic Content 结束。Chat 截断可去掉完整引用，但不能截断引用内部。PIC 结构化渲染当前支持文本及上述 embeddings，不能与图像等多模态内容混合。

### 5.2 Responses

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"reader",
    "input":[{"role":"user","content":[
      {"type":"input_text","text":"Read this document:\n"},
      {"type":"reusable_content","id":"demo-doc"},
      {"type":"input_text","text":"\nReturn the access code only."}
    ]}],
    "max_output_tokens":128
  }'
```

Responses 返回原生 `output` items，不使用 Chat 的 `choices` 字段。普通 Completion 文本/tokens、普通 Chat messages、普通 Responses input 仍沿用原生协议。自定义 `reusable_content` 属于 ReLoCache 扩展，可直接用 HTTP 或本项目客户端发送。

### 5.3 流式消费与取消

Completion/Chat 使用 `stream=true`；需要最终复用统计时，同时传 `stream_options: {"include_usage": true}`。Python 客户端的 `complete_stream` 默认包含该设置：

```bash
uv run --no-sync python - <<'PY'
import asyncio
from contextlib import aclosing
from transformers import AutoTokenizer
from relocache.client import Client

async def main():
    tokenizer = AutoTokenizer.from_pretrained("../models/qwen3-0.6b")
    question = tokenizer.encode("\nReturn the access code:\n", add_special_tokens=False)
    async with Client("http://127.0.0.1:8000") as client:
        blocks = [
            {"type": "reusable_content", "id": "demo-doc"},
            {"type": "tokens", "input_ids": question},
        ]
        async with aclosing(client.complete_stream(
            blocks, model="reader", max_tokens=64, temperature=0,
        )) as events:
            async for event in events:
                for choice in event.get("choices", []):
                    print(choice.get("text", ""), end="", flush=True)
                if event.get("kv_transfer_params"):
                    print("\nMetrics:", event["kv_transfer_params"])
        print()

asyncio.run(main())
PY
```

提前 `break` 时，`aclosing` 会关闭 HTTP 流，由原生请求取消流程终止执行。客户端还提供 `register`、`compile`、`status`、`deployment`、`flush`；Chat/Responses 使用 `client.post(path, payload)`。`Client.post` 在响应字典中增加客户端测得的 `http_latency_seconds`，它不是服务端原生字段。

## 6. 接入 LoRA Writer 与 soft tokens

### 6.1 LoRA Writer

准备与基础模型匹配的标准 PEFT 目录，包含 `adapter_config.json` 和 adapter 权重。在部署 JSON 顶层增加：

```json
{
  "writer_adapter": "/absolute/path/to/writer-adapter",
  "method_id": "qwen3-writer-checkpoint-216"
}
```

这是对完整部署文件的字段修改，不是单独的部署文件。重启服务使其生效。框架从 adapter 配置读取 rank，并启用原生 LoRA；Writer 编译使用该 adapter，普通 Reader 请求使用基础模型。无需在生成请求中重复传 adapter。

### 6.2 soft-token 编码与放置

每份 tensor 的形状为 `[soft_token_count, model_hidden_size]`，行数决定长度。先将训练得到的 tensor 导出为单独的 `.pt` 文件，再按下面方式生成传输表示；把路径换成实际文件：

```bash
uv run --no-sync python - <<'PY'
import base64
import json
from io import BytesIO
from pathlib import Path
import torch

encoded = {}
for name in ("pre", "post"):
    tensor = torch.load(f"artifacts/{name}.pt", map_location="cpu", weights_only=True)
    buffer = BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    encoded[name] = {"data": base64.b64encode(buffer.getvalue()).decode()}
Path("artifacts/soft-tokens.json").write_text(json.dumps(encoded) + "\n")
PY
```

只有一侧时删除循环中另一侧名称。`data` 是编码内容，不是文件路径。

**用于 Writer**：将生成 JSON 对象放进部署的 `writer_soft_tokens`：

```json
{
  "writer_soft_tokens": {
    "pre": {"data": "<encoded tensor>"},
    "post": {"data": "<encoded tensor>"}
  }
}
```

服务自动启用 prompt embeddings。Writer 编译 `[Pre, document, Post]`，保存三部分 KV；页对齐 padding 和内部 observation 不属于可引用内容。不配置 adapter 对应 KV Packet，同时配置 adapter 对应 Joint。

**用于 Reader**：在部署的 `engine` 中设置 `"enable_prompt_embeds": true`，在每次需要的位置插入 `{"type":"prompt_embeds","data":"..."}`。例如块顺序为 `[tokens, pre embedding, reusable_content, post embedding, tokens]`。Chat/Responses 的 content 列表使用相同的 embedding part。Reader embeddings 随当前上下文计算，不写入独立文档的编译身份。

## 7. KV 持久化与多 GPU

### 7.1 文件系统 KV

在部署 JSON 顶层加入下面字段，停止旧服务后用新配置重启：

```json
{
  "cpu_bytes_to_use": 1073741824,
  "kv_dir": "artifacts/quickstart/kv",
  "metadata_dir": "artifacts/quickstart/metadata"
}
```

本例给 CPU staging 分配 1 GiB。实际容量按文档规模和并发调整。`kv_dir` 要求 CPU staging 预算；框架配置原生 Tiering OffloadingConnector，实际文件存储路径会增加 `pre_rope` 子目录。

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/reusable_contents/compile \
  -H 'Content-Type: application/json' \
  -d '{"content_id":"demo-doc","persist":true}'
```

返回 `compilation.persisted=true` 表示此次完成时已检查配置的持久层存在这些页。仅 CPU offload 不满足持久化条件。重启后，保持模型、Writer 身份、页配置和存储路径一致；来源从 `metadata_dir` 恢复，KV 按需由原生存储层加载。若不保留元数据，也可以重新注册相同来源后引用。

`auto_compile=false` 可用于要求只读取已有 KV 的部署；真实缺失返回 `reusable_cache_miss`，不会偷偷重新编译。该设置不禁止显式 compile。

原生 `engine.kv_transfer_config` 也可配置对象存储，但当前项目没有执行对象存储端到端验证；本文给出的持久化运行步骤使用文件系统。

### 7.2 DP / TP

单张卡可容纳模型时，可以通过 DP 提高并发；单张卡容不下时使用 TP。设备数按 DP × TP 配置，例如确认 4–7 均空闲后：

```bash
uv run --no-sync relocache serve \
  --config artifacts/quickstart/deployment.json \
  --cuda-visible-devices 4,5,6,7 \
  --data-parallel-size 4 \
  --tensor-parallel-size 1 \
  --port 8000
```

DP 副本各有 GPU 缓存；一个副本预热不意味着另一个已预热。自动 Writer 在 Reader 所在 engine 执行，状态查询返回所有副本。先用 DP=TP=1 建立可解释的缓存实验，再扩展并行规模。已有多卡验证主要为 DP、TP=1；具体 TP 组合需在目标模型上验证。

## 8. 使用现有脚本完成质量评测

现有 `examples/sempic/evaluate.py` 对每个样本先发 Full Prefill，再发 PIC，保存两份响应并评分。它可以自动启动服务或连接已运行服务，默认是**质量评测流程**。

### 8.1 评测 bundle 格式

一个 bundle 目录包含四个文件：

| 文件 | 内容 |
| --- | --- |
| `deployment.json` | 上文的完整部署配置；脚本读取 `engine` 和 `served_model_name` |
| `reader-soft-tokens.json` | Reader `pre` / `post` 的编码对象；不用时为 `{}` |
| `inputs.jsonl` | 每行一个样本，格式如下 |
| `source.json` | 数据和模型产物来源说明，JSON 对象；原样记入实验配置 |

样本示例（写入 JSONL 时为单行）：

```json
{
  "id": "example-1",
  "dataset": "hotpotqa",
  "answers": ["ORCHID-42"],
  "reusable_contents": [
    {"content_id": "example-1-doc", "text": "The access code for the archive is ORCHID-42."}
  ],
  "messages": [
    {"role": "user", "content": [
      {"type": "text", "text": "Read the document:\n"},
      {"type": "reusable_content", "id": "example-1-doc"},
      {"type": "text", "text": "\nWhat is the access code? Reply with the code only."}
    ]}
  ],
  "chat_template_kwargs": {"enable_thinking": false}
}
```

`messages` 中每个引用 ID 必须出现在该行 `reusable_contents`。普通 message content 可以是字符串；列表内容使用 text 和 reusable_content。脚本按引用顺序插入 Reader soft tokens，可选 `occurrence_soft_tokens: [[true, false], ...]` 为每次引用控制 Pre/Post 是否启用，省略时两侧均启用（前提是 bundle 提供该侧 tensor）。跨样本推荐使用不同内容 ID。

脚本当前支持的数据集键与评分如下。上面的手写样本借用 `hotpotqa` 键选择评分/长度配置，不是真实 HotpotQA 样本：

| `dataset` | 最大生成 tokens | 评分 |
| --- | --- | --- |
| `hotpotqa`、`2wikimqa`、`musique` | 48 | 取各参考答案中最高 ROUGE-L F1，启用 stemming |
| `ruler_mv`、`ruler_mq`、`ruler_vt` | 128 | 参考答案中，忽略大小写后作为子串出现在预测中的比例 |

RULER 输出中的 metric 名为 `string_match_all`，实际按上述命中比例计算。没有参考答案时得分为 0。新数据集键、其他评分或生成长度需要修改评测脚本，不是服务配置项。脚本可选传入每行的 `chat_template`；这需要在部署 `engine` 中设置 `"trust_request_chat_template": true` 并重启。通常优先使用模型原生模板或部署级 `engine.chat_template`。

### 8.2 创建并运行一个完整的小样本评测

以下命令直接生成 bundle，无需已有 SemPIC checkpoint。它验证工作流，不构成方法质量结论：

```bash
uv run --no-sync python - <<'PY'
import json
from pathlib import Path

bundle = Path("artifacts/quickstart-eval")
bundle.mkdir(exist_ok=False)
config = json.loads(Path("artifacts/quickstart/deployment.json").read_text())
(bundle / "deployment.json").write_text(json.dumps(config, indent=2) + "\n")
(bundle / "reader-soft-tokens.json").write_text("{}\n")
(bundle / "source.json").write_text(json.dumps({
    "description": "Hand-written workflow example; not a HotpotQA benchmark sample",
}) + "\n")
row = {
    "id": "example-1", "dataset": "hotpotqa", "answers": ["ORCHID-42"],
    "reusable_contents": [{
        "content_id": "example-1-doc",
        "text": "The access code for the archive is ORCHID-42.",
    }],
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Read the document:\n"},
        {"type": "reusable_content", "id": "example-1-doc"},
        {"type": "text", "text": "\nWhat is the access code? Reply with the code only."},
    ]}],
    "chat_template_kwargs": {"enable_thinking": False},
}
(bundle / "inputs.jsonl").write_text(json.dumps(row) + "\n")
PY

uv run --no-sync python examples/sempic/evaluate.py \
  --bundle artifacts/quickstart-eval \
  --output runs/quickstart-eval-01 \
  --api-base http://127.0.0.1:8000 \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --concurrency 1 \
  --seed 42
```

连接已有服务时，bundle 和 CLI 不会修改正在运行的部署。运行前用 `/v1/relocache` 确认模型、Writer、TP/DP 和 soft-token 配置相符。

若希望脚本自行启动并关闭服务，先停止占用目标 GPU 的本例服务，确认 GPU 空闲，再使用：

```bash
uv run --no-sync python examples/sempic/evaluate.py \
  --bundle artifacts/quickstart-eval \
  --output runs/quickstart-eval-02 \
  --cuda-visible-devices 4 \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --concurrency 1 \
  --seed 42
```

bundle 生成目录和每次 `--output` 必须是新目录。自动启动时服务使用动态端口；脚本打印 GPU 状态，但不会替使用者选择空闲卡。

输出文件：

| 文件 | 如何使用 |
| --- | --- |
| `predictions.jsonl` | 每行含样本 ID、参考答案、Full Prefill/PIC 完整响应、分数及 HTTP 延时 |
| `metrics.json` | `completed`、样本量、按数据集和整体汇总的分数 |
| `config.json` | 部署、CLI、种子、并发、来源、两个仓库 commit/status |
| `server.json`、`server.log`、`server-process.json` | 仅自动启动服务时生成，记录配置、日志和进程 |

整体 `macro_score` 实际是**所有样本分数的算术平均**，不是各数据集均值的等权平均。`elapsed_seconds` 覆盖进入服务上下文、全部请求及收尾；自动启动模式也包含启动和关闭。评分与保存预测由评测脚本负责，服务自身不读取 answers 或 dataset。

### 8.3 从现有 SemPIC ICLR 产物导出

已有 ICLR 产物时，可使用现有转换脚本。源仓库只读，输出目录必须不存在：

```bash
uv run --no-sync python examples/sempic/prepare.py \
  --source-repo ../sempic_iclr \
  --artifact runs/all-layer-supervision/training/qwen3-8b/P16/artifacts/step-00000216 \
  --data data/six-datasets-128/models/qwen3-8b/validation.jsonl \
  --output artifacts/sempic-eval-new \
  --samples-per-dataset 1
```

这里给出的是本机既有产物路径；使用自己的训练结果时替换 `--source-repo`、`--artifact` 和 `--data`。产物包含 `method.json`、`method.safetensors`，数据为上面的 canonical JSONL。脚本只接受 `reduction.mode="identity"`，导出标准 PEFT Writer、**Reader** soft tokens、部署文件、样本及来源说明。它不负责把任意 KV Packet/Joint checkpoint 转成 Writer soft tokens。

随后用 8.2 节的评测命令替换 `--bundle`，按 Qwen3-8B 的容量选择设备和并行配置。导出配置默认 `max_model_len=16640`、显存比例 0.8，应与可用设备容量相符。已有 `artifacts/qwen3-8b-P16` 时也可直接作为 bundle 使用，但 `artifacts/` 不随 Git 分发，不能假设新克隆已有该目录。

### 8.4 方法比较的输入约定

当前 demo 的 Full Prefill 将文档展开成普通消息，PIC 将相同文档作为引用。它们共享任务文本、模板和采样设置，但整体分词与分段分词可能不同；demo 不要求 token IDs 完全一致。

若研究要求 canonical token 对齐，应使用 5.1 节的精确 token 接口：分别分词各段，Full Prefill 提交拼接后的 IDs，PIC 提交相同动态 IDs 与文档引用。训练约定要求的特殊 tokens、Reader embeddings 和 Writer 包装由评测侧组织；不能用展开原始文档文本代替所有方法的正式 baseline 定义。

服务已经支持 Base/LoRA Writer、Writer/Reader soft tokens，但当前 demo 不是完整的多方法评测套件。SemPIC 全方法评测接入、正式质量/性能评测仍待完成；EPIC、CacheBlend、KV 压缩与第二后端暂未实现。

## 9. 性能评测与统计解释

现有质量脚本每样本按 Full→PIC 顺序执行，不重置缓存，不测流式 TTFT，也不隔离自动编译。`http_latency_seconds` 包含非流式请求的完整等待和生成，不能据此直接得出 PIC 加速比。

### 9.1 先确定缓存初态

| 实验 | 准备方式 | 测量包含什么 |
| --- | --- | --- |
| 首次使用 | 新内容、无已有 KV，`auto_compile=true` | 注册后 Reader 自动编译、等待、生成 |
| GPU 热缓存 | 先完成相同 Reader 请求，再在缓存仍驻留时测量 | 命中后的生成；动态前缀也可能被复用 |
| 文件 KV 冷加载 | 先持久化，再清 GPU/CPU 缓存，`auto_compile=false` | 文件→CPU→GPU 加载、Reader 执行与生成 |
| 稳态服务 | 固定流量、并发和缓存容量，经过预热后测量 | 实际命中、淘汰、加载与调度的组合 |

可控的文件 KV 冷加载步骤（先使用 DP=TP=1）：

1. 按第 7 节配置文件存储，将顶层 `auto_compile` 设为 false。
2. 启动命令前加 `VLLM_SERVER_DEV_MODE=1`，使原生 reset 接口可用。
3. 注册待测全部内容，对每份执行 `compile(persist=true)`；准备结束后执行下面的 flush 和 reset。
4. 确认 reset 返回 `{"success":true}` 后开始计时和发送 Reader。准备、编译、flush、reset 均不计入 Reader 延时。

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/reusable_contents/flush \
  -H 'Content-Type: application/json' -d '{}'
curl --fail-with-body -X POST \
  'http://127.0.0.1:8000/reset_prefix_cache?reset_external=true'
```

这是当前原生文件分层存储的 GPU/CPU reset，保留文件 KV。只有实验服务空闲时操作；HTTP 200 不等于 reset 成功，检查响应中的 `success`。返回 false 时说明仍有块或传输占用，等待其完成再操作。这里不使用强制重置运行中请求的选项。

运行前 reset 一次后允许自然复用，与逐请求 reset 是两种不同实验。文件 KV 冷加载也不等于操作系统 page cache 已冷，更不能直接称为磁盘物理冷读。记录实际初态；不要通过中断他人任务或全局清系统缓存准备实验。

热缓存实验记录预热请求数，并用 `/status` 和响应统计确认实际复用。Full Prefill baseline 也可能命中动态前缀缓存，需记录其是否预热。若重复完整问题，测到的收益可能同时来自原生前缀缓存和文档 PIC 复用。

### 9.2 指标读取

Completion、Chat、Responses 响应使用 `kv_transfer_params.relocache`。Completion/Chat 流式响应在最终 usage chunk 提供该字段；显式 compile 则直接返回 `metrics`。

| 字段 | 口径 |
| --- | --- |
| `referenced_tokens`、`occurrences` | 含 Writer 包装的逻辑引用 tokens 与引用次数；重复引用重复计数 |
| `reused_tokens`、`reused_pages` | 实际复用的去重 KV 行/页；有效行数不含页 padding |
| `auto_compiles` | 本 Reader 发起的 Writer 数；等待其他 Reader 的 Writer 不计入 |
| `wait_seconds` | Reader 从延后准入到执行的时间，含编译、查找/加载及调度等待 |
| `prefill_tokens_executed` | 实际调度 prefill 行数；Reader 排除复用页，Writer 包括 padding/observation，抢占后的重复执行会计入 |
| `compile_seconds` | 显式 Writer 在调度器中的耗时，不含后续持久化等待 |
| `loads.cpu_loaded_pages`、`cpu_load_jobs`、`cpu_load_seconds` | 请求 CPU→GPU 搬运的页数、任务数、任务耗时之和 |
| `loads.<tier>:<index>` | 如 `fs:0`，该层→CPU 的 `loaded_pages`、`load_jobs`、`load_seconds` |

例如 58-token 文档引用两次，逻辑引用为 116 tokens，去重 KV 复用可以只有 58 rows。相同请求的热缓存还可能复用问题前缀，使 Reader prefill 进一步减少。加载统计可能包含动态内容页；文件经 CPU 到 GPU 的一页会分别出现在两段搬运统计中。

编译、等待和加载存在重叠，不能相加当作端到端延时。客户端 TTFT 应从发送流式请求计时，到第一个非空生成文本事件；完整 HTTP 延时则到响应完成。引擎 prefill 时间不含全部排队，不等于 TTFT。

正式实验至少保存：模型/Writer 身份、两个仓库版本、模板与 soft-token 放置、数据与样本顺序、采样及生成长度、GPU/TP/DP、并发、缓存容量与初态、reset 范围、预热与重复次数、逐请求结果和统计。跨方法或冷/热状态不默认要求逐 token 完全相等，应按预先确定的质量指标比较。

## 10. 常见错误与处理

| 现象 / 错误 | 含义与操作 |
| --- | --- |
| `unknown_content`（404） | ID 没有注册且元数据无法恢复；先注册或在引用中提供来源 |
| `invalid_content`（400） | 空来源、引用被截断、输入末尾只有引用等；按提示修正输入 |
| `persistence_unavailable`（400） | `persist=true` 但没有文件/对象 KV 层；按第 7 节配置 |
| `persistence_failed` | 编译后未能确认配置持久层存在 KV；查看服务日志和存储配置 |
| `reusable_cache_miss` | 自动编译关闭且真实缺页；显式编译，或启用自动编译并重启 |
| `reusable_compilation_failed` | Reader 依赖的 Writer 失败；检查 LoRA、长度、显存与服务日志 |
| schema 校验错误 | `text` 与 `input_ids` 同时提供、块类型或生成字段不合法；按本指南格式发送 |
| embeddings 被拒绝 | Reader embeddings 需要 `engine.enable_prompt_embeds=true`，tensor 应为二维且 hidden size 匹配模型 |
| 文本超长 / 显存不足 | 将文档、Writer/Reader soft tokens、模板与输出一起计入长度；Writer 还有页对齐/observation 开销。按模型限制调整 `max_model_len`、并发或设备配置 |
| 重启后引用找不到 | `metadata_dir` 未配置或路径/Writer ID 改变；重新注册来源。文件 KV 与内容目录是两个独立部分 |
| 指标显示再次编译 | 检查是否落到另一个 DP 副本、缓存被淘汰、来源或 Writer 身份变化，以及存储层是否配置 |

缓存缺失是请求级错误，不应把一次失败解释为服务整体退出。HTTP 和 SSE 错误封装可能不同；保留原始错误消息与 `code`/`finish_reason` 以定位。服务没有对外上传/下载原始 KV tensor、删除内容或驱逐指定文档页的专用接口。

## 11. Web Chat 演示

ReLoCache API server 启动后，浏览器访问 `http://127.0.0.1:8000/demo`。页面默认填入虚构的长篇企业运营手册与第一个问题：24 节制度、12 个核算案例和 144 条项目台账，约 1.9 万字符、Qwen3 tokenizer 下 13,902 tokens。也可切换 ReLoCache 技术说明或自行编辑原文。长文建议使用至少 32K 上下文，为模板、问题、对话历史和输出保留空间。页面支持连续对话、停止生成、修改文档与生成设置，以及以下模式：

- **PIC 复用**：文档以 `reusable_content` 传入，使用当前部署的 Writer。首次请求可自动编译；也可先点击「提前编译 PIC」。
- **Full Prefill**：将相同文档作为普通文本提交，保留原生 prefix caching。
- **并排对比**：同时发送两个独立对话请求，分别呈现回答与指标。它们共享引擎资源，可能相互影响排队和缓存，不能作为隔离性能基准。

修改场景、文档、System prompt 或推理模式会开始新对话。「新对话」只清空页面历史，不清空引擎缓存。刷新页面恢复预置文本。提前编译不请求持久化，且只在返回的 DP rank 上完成；后续请求落到其他副本时仍可能自动编译。

页面使用浏览器 `performance.now()` 计时：TTFT 从发起请求前到首个非空内容或推理片段，不计 role-only、空 delta 或 usage 事件；总耗时到 SSE `[DONE]`。输出 tokens 采用服务端最终 usage，tokens/s 为输出 tokens 除以总耗时，属于端到端速率。推理文本单独折叠展示。若流中断、服务返回错误或手动停止，会保留已显示内容并标记状态，不把本轮加入后续历史。KV 指标来自最终 usage chunk 的 `kv_transfer_params.relocache`，可展开查看原始统计。

也可独立启动 Web Chat，通过同源流式代理连接已有服务，前端无需配置 CORS，也不需要单独安装 Node：

```bash
# A separate webchat installation needs only the webchat extra, without vLLM.
uv sync --extra webchat
uv run --no-sync relocache webchat \
  --api-base http://127.0.0.1:8000 \
  --host 0.0.0.0 --port 8080
```

项目已有完整推理环境时，直接运行第二条命令，无需重新 sync。浏览器访问 `http://<服务器地址>:8080/demo`。`--api-base` 为 API 根地址，不包含 `/v1`。若上游启用了 API key，通过环境变量 `RELOCACHE_API_KEY` 提供，代理在服务器端附加该凭据。未连接上游时仍能浏览预置文本，启动推理服务后点击左下角的重新连接按钮。

只需展示当前基础能力时，可使用 Base Writer 和无 soft tokens 的部署配置。LoRA 与 Writer soft tokens 仍由推理服务部署配置决定，页面不会根据场景替换 Writer。

展示跨上下文复用时，先点击「提前编译 PIC」，再用「并排对比」提问。修改生成设置中的 System prompt 会开始新对话并改变文档的前置上下文，已编译文档仍可复用；Full Prefill 的原生前缀命中则取决于实际前缀。不要通过反复提交相同问题把原生前缀缓存收益解释为 PIC 独有收益。

当前本机演示选用 `../sempic_open/train_outputs/qwen_3_4b/native/sempic/20260920_035603_two_stage_reference_stage2/lora`：Qwen3-4B-Instruct-2507、rank 8 / alpha 16，先完成 3 个 epoch 的 SRD，再完成 2 个 epoch 的 native 训练。其训练与 ReLoCache 评测已完成，不需要额外 PacketWrapper 或 LinkToken。Adapter 只读复制到 `artifacts/webchat-qwen3-4b-two-stage/writer-adapter`，来源记录保存在相邻的 `source.json`；旧仓库不修改。启动配置见 `examples/webchat/deployment.json`，在根目录执行：

```bash
uv run --no-sync relocache serve \
  --config examples/webchat/deployment.json \
  --cuda-visible-devices 1 --host 127.0.0.1 --port 8010
```

此处 GPU 1 来自本次用户允许使用其他空闲 GPU 的授权；后续启动前应重新确认设备空闲并按当前设备授权调整。页面通过前述 `webchat` 命令连接 `http://127.0.0.1:8010`。LoRA 仅用于 Writer 编译；Reader 使用基础模型。对其他机器，先准备匹配的标准 PEFT adapter 并修改配置路径，不依赖当前目录中的研究 checkpoint 布局。

前端为直接随包分发的 HTML/CSS/ES modules，无构建步骤或 CDN 请求。Markdown 展示使用随包附带的 marked 18.1.1 与 DOMPurify 3.4.17；版本及授权文本见 `src/relocache/webchat_assets/THIRD_PARTY_LICENSES.txt`。
