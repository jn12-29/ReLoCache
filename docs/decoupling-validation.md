# PIC 解耦验证

日期：2026-09-18。以下为历史执行记录，原始 runs 目录未随开源仓库分发；公开版本保留验证条件与摘要。本页记录 PIC 核心、vLLM 接入、生命周期与请求统计拆分的已执行验证。SemPIC 方法集成、正式质量/性能评测与后续方法扩展仍按[路线图](roadmap.md)推进。

## 开源准备检查（2026-11-11）

- `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q`：59 项通过，7 项 CUDA 测试跳过。
- `node --test tests/test_webchat_stream.mjs tests/test_webchat_conversation.mjs`：9 项通过。
- 主仓库 Ruff 检查和格式检查通过；构建 wheel 与 sdist，许可证、第三方声明和浏览器资源均已打包，无模型或实验产物。
- wheel 在独立 Python 3.12 环境安装，未安装 Torch/vLLM；客户端、CLI、Web Chat 可用，3 项 Web Chat 测试通过。
- `uv lock --check --offline` 和文档安装命令的 `uv sync --locked ... --dry-run` 通过。公开索引改为 PyPI；仅更新已撤回版本的构建工具 setuptools-scm 及其 vcs-versioning 依赖，运行时版本不变。

本轮调整公开文档、示例、包元数据和修改标注，没有修改推理计算逻辑；没有新建完整 GPU 服务环境，也没有重跑 GPU 数值或方法质量/性能实验。下文的 GPU 结果仍属于历史验证。

## 实现边界

PIC 内容与 Writer 身份、编译协调、逻辑位置和 fused-RoPE 实现属于 ReLoCache；页哈希、请求/调度子类、IPC、worker buffers 属于 `backends.vllm`。相对 `upstream-base`，fork 扩展执行接口，继续复用原生 KV 管理、offload、attention 主循环；完整累计差异见 [vLLM 修改边界](vllm-patch.md)。PIC 测试由主仓库维护。

阻止导入 `relocache` 后，原生 EngineCore、DefaultModelState、Request 与普通请求哈希仍可使用。核心布局、身份与轻量客户端不依赖 vLLM。attention 扩展函数在同一 Triton kernel 内融合执行，不新增 KV 搬运或单独的旋转 kernel。

## 当前最小侵入接入验证

在该轮验证时，相对 `upstream-base` 累计修改 **22 个源码文件，+314/-44**；开源准备另增加了修改标注与 fork 入口文档，当前清单见 [修改边界](vllm-patch.md)。EngineCore 请求结构、InputProcessor、NewRequestData、FileMapper、worker input_batch/model_runner/ModelState interface 均恢复原生实现。新增 OnlineRenderer 类选择；PIC attention metadata 和位置映射位于 ReLoCache。

- `CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m pytest -q`：**63 项通过**，包含 6 项 fused attention 数值/图执行测试，以及请求 slot 重排、dummy batch 的独立位置映射测试。
- GPU 6–7，Qwen3-0.6B BF16，DP=2/TP=1、LoRA Writer、Pre/Post soft tokens、Reader embeddings、chunk size 64、CUDA graph：显式/自动 Writer、重复引用、两副本状态、Completion/Chat/Responses（含 assistant）、SSE、错误请求与 flush 全部通过。普通输入与结构化 token 输入继续使用原生执行。含启动总耗时 49.01 秒。
- GPU 7 文件系统生命周期：关闭自动编译时，三个 API 的缓存缺失均为请求级错误；并发生成期间，单个 Writer 持久化等待约 0.558 秒返回，其他请求仍在执行。重启从新配置的 `pre_rope` 子目录加载 8 页，`auto_compiles=0`、Reader 实际 prefill 13 rows，原生 fs→CPU→GPU 统计保留。
- Ruff、fork 适用 pre-commit hooks、mypy 3.10/3.12、离线 lock 检查通过。未运行对象存储服务或正式 SemPIC 质量评测。测试及说明仅提交到主仓库。

命令与完整产物位于本地运行目录（`runs/pic-minimal-integration-20260918/`），包含配置、代码快照/patch、PID、设备映射和日志。主要入口：

```bash
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES=6,7 .venv/bin/python runs/pic-minimal-integration-20260918/verify.py
CUDA_VISIBLE_DEVICES=7 .venv/bin/python runs/pic-minimal-integration-20260918/lifecycle.py
```

HTTP 结果（`runs/pic-minimal-integration-20260918/results.json`）和生命周期结果（`runs/pic-minimal-integration-20260918/lifecycle/results.json`）均有完成标记。运行进程已退出，未中断其他任务。

本轮最初一次 pytest 未限定 CUDA 可见设备，不能算作指定 GPU 的验收；其中两个测试仍按旧类继承顺序寻找原生 schema，修改断言对象后通过。最终全套测试在卡 7 完整重跑，后续 GPU 启动均检查并限定为卡 6/7。fork 首次 lint 修正了一处行宽，之后适用检查通过。

### 位置映射的性能取舍

采用外置映射，恢复三个原生 worker 文件。对比使用相同 storage/query/slot 输入，先检查数值一致，再用 CUDA graph 测量位置准备阶段；该测量不包含 Python launch 和服务调度。GPU 7，int64 positions，结果为微秒：

| batch / query rows | 融合准备 | 原生准备 + 外置映射 | 增加 |
| --- | ---: | ---: | ---: |
| 1 / 1 | 1.672 | 3.087 | 1.415 |
| 4 / 1 | 1.888 | 3.458 | 1.571 |
| 32 / 1 | 2.001 | 3.691 | 1.690 |
| 4 / 64 | 1.933 | 3.410 | 1.477 |
| 4 / 256 | 1.969 | 3.510 | 1.540 |

另用同一 Qwen3-0.6B 服务、GPU 7、DP=TP=1，比较 batch=1/4 的 128-token decode 和 chunked prefill。每项 2 次预热、8 次测量；prefill 使用不同 cache salt。下表为整批 HTTP 请求的中位耗时（毫秒），包括调度和传输，并非模型单步 GPU 耗时：

| 工作负载 | 融合 | 外置 | 固定 CPU affinity 后融合 | 固定 CPU affinity 后外置 |
| --- | ---: | ---: | ---: | ---: |
| decode，batch=1 | 458.2 | 423.6 | 451.6 | 371.2 |
| decode，batch=4 | 369.7 | 508.6 | 504.8 | 375.0 |
| chunked prefill，batch=1 | 103.0 | 68.2 | 100.7 | 66.9 |
| chunked prefill，batch=4 | 111.2 | 69.4 | 75.2 | 111.4 |

首轮 batch=4 decode 外置较慢，因而补做相同 CPU affinity（56–63、120–127）的对照；差异方向没有保持一致。端到端结果波动较大，不能据此宣称性能提升或完全无损。选择外置实现的已知代价是每步一次小 kernel，隔离测量增加约 1.4–1.7 微秒；收益是直接使用原生 ModelState 生命周期，不修改 storage positions 和 slot mapping 的准备流程。旋转仍融合在 attention 内，不增加 KV materialization。

详细测量条件与原始样本见 `position-bench.json`、`serving-{fused,external}/measurement.json` 和 `serving-{fused,external}-pinned/measurement.json`。这些只比较位置映射接入方式，不作为正式方法性能结果。

## 前轮接入收缩验证（已有记录）

该轮 fork 相对 `upstream-base` 修改 28 个源码文件，增加 410 行、删除 46 行。RoPE 通过 ReLoCache CustomOp 插件接入，存储查询组合原生各层接口；原生 rotary、secondary-tier 基类、filesystem/object-store manager 源码与基线一致。

- 主仓库 CPU 测试 54 项通过，6 项 CUDA kernel 测试跳过；包含标准/Llama 3 RoPE 在普通模式与 Query-only 模式下的数值检查。
- 原生 Responses utils 测试 40 项通过，保留普通 assistant/tool 转换行为。
- GPU 6–7、Qwen3-0.6B BF16、DP=2/TP=1 服务验证通过，沿用下面的 LoRA、soft-token、chunked-prefill 和 CUDA Graph 配置。两个副本状态分别返回 rank 0、1；另覆盖 assistant 历史中的内容引用和 embeddings。含启动总耗时约 28.07 秒。
- GPU 7 文件系统生命周期验证通过：独立持久化约 0.299 秒返回，其他生成仍在执行；重启后原生加载 8 页，`auto_compiles=0`，Reader 实际执行 13 个 prefill rows。
- Ruff、fork 适用 pre-commit hooks、mypy 3.10/3.12 和离线 lock 检查通过。没有重跑未改变的 attention 数学测试与原生调度/offload 全套；其已有验证列于下文。

当前验证命令与产物：

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync pytest -q
CUDA_VISIBLE_DEVICES='' uv run --no-sync pytest -q --noconftest \
  third_party/vllm/tests/entrypoints/openai/responses/test_responses_utils.py
CUDA_VISIBLE_DEVICES=6,7 uv run --no-sync python runs/pic-fork-reduction-20260918/verify.py
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python runs/pic-fork-reduction-20260918/lifecycle.py
```

配置、代码 patch、PID、GPU 映射、日志及响应保存在本地运行目录（`runs/pic-fork-reduction-20260918/`）。HTTP 结果（`runs/pic-fork-reduction-20260918/results.json`）与生命周期结果（`runs/pic-fork-reduction-20260918/lifecycle/results.json`）均有完成标记。

HTTP 重试记录分别保留在 `attempt-1-scheduler-signature`、`attempt-2-dp-index`、`attempt-3-request-schema`：调度输出处理改用已有回调；状态标识修正为不会被 dense 模型重置的 `data_parallel_index`；测试请求补齐 `output_text.annotations`。最后一项只修改验证脚本。Responses 单元测试目录的 conftest 依赖未安装的 `pytest_asyncio`；该纯单元测试文件不依赖目录 fixtures，使用 `--noconftest` 运行，未新增依赖。

## 测试归属

Fork 中新增的共享 mmap unlink、必需缓存 promotion 等待测试现位于主仓库 `tests/test_vllm_offload.py`；两项迁移后的测试均通过：

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync pytest -q tests/test_vllm_offload.py
```

该轮 Fork 的测试和说明文件与 `upstream-base` 一致；开源准备后另有 fork 入口说明。此次仅调整测试与文档归属，运行源码未变，未重复执行 GPU 验证。

## 基础执行验证（已有记录）

| 范围 | 结果 |
| --- | --- |
| 主仓库 CPU 服务、哈希、布局、IPC、worker 测试 | 53 项通过；6 项 CUDA 测试在此命令中跳过。 |
| 原生 tiering offload | 55 项通过。 |
| 原生 offloading connector scheduler | 212 项通过，包含原生同步/异步调度及缓存生命周期。 |
| GPU 7 attention | 6 项通过：BF16 dense reference、连续/显式位置、部分 rotary dimension、旋转布局、缓存表及 CUDA Graph。 |
| GPU 6–7 HTTP | Qwen3-0.6B BF16，DP=2、TP=1，合成 rank-8 LoRA Writer，Writer/Reader soft tokens，chunked prefill 和 CUDA Graph；全部请求成功。 |
| GPU 7 生命周期 | 自动编译关闭；请求级缺失错误、并发期间独立持久化、重启后的文件系统→CPU→GPU 加载均通过。 |

HTTP 覆盖 completions 精确 token blocks、普通 token baseline、两个 DP 副本、按 ID 编译、重复引用、Chat、Responses、SSE 客户端、部署摘要、状态查询和 flush。相同缓存状态的 Chat/completions tokens、logprobs 一致，流式文本与非流式一致。

生命周期验证先注册但不编译，三个生成接口均返回含 `reusable_cache_miss` 的请求级错误，随后 health 与正常生成仍成功。持久化编译在约 0.240 秒返回，此时另一条 1500-token 生成仍在执行。重启后 GPU/CPU 中均无该内容，文件系统有 8 页；Reader 加载这些页，`auto_compiles=0`，实际执行 13 个 prefill rows。

复用统计实例：58-token 内容引用两次，`referenced_tokens=116`、`reused_tokens=58`、`reused_pages=4`。首次 Reader 执行 17 个 prefill rows，同输入热缓存只执行 1 个。它们验证计算和存储复用口径，不是吞吐基准。冷、热缓存可能有低精度数值差异；接口等价按相同缓存状态比较，不要求跨缓存状态逐 token 相等。

Ruff、fork 全部适用的 pre-commit hooks、Python 3.10/3.12 mypy、`uv lock --check --offline` 均通过。本轮 GPU 进程已退出并释放其占用，未中断其他任务。

## 命令与记录

从主仓库运行：

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync pytest -q
CUDA_VISIBLE_DEVICES=7 uv run --no-sync pytest -q tests/test_pic_attention.py
CUDA_VISIBLE_DEVICES=6,7 uv run --no-sync python runs/pic-decoupling-20260918/verify.py
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python runs/pic-decoupling-20260918/lifecycle.py
```

原生测试使用同一项目环境，以 fork 为工作目录运行：

```bash
CUDA_VISIBLE_DEVICES='' uv run --project ../.. --no-sync pytest -q \
  --confcutdir=tests/v1/kv_offload \
  tests/v1/kv_offload/tiering/test_tiering_offloading.py

env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u http_proxy -u https_proxy HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  uv run --project ../.. --no-sync pytest -x -q \
  --confcutdir=tests/v1/kv_connector/unit/offloading_connector \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py
```

运行配置、启动 PID、GPU 映射、代码状态、服务日志和响应保存在本地运行目录（`runs/pic-decoupling-20260918/`）。主要结果：HTTP（`runs/pic-decoupling-20260918/results.json`）、生命周期（`runs/pic-decoupling-20260918/lifecycle/results.json`）、attention（`runs/pic-decoupling-20260918/kernel/pytest.log`）。运行目录为本地实验产物，不提交 Git；本文保留验证摘要。

首次 HTTP 验证发现 utility RPC 可变参数签名不符合原生参数转换规则，改为固定列表参数后通过；失败日志保留在 `attempt-1-rpc-signature`。随后用稳定操作 ID 将外部编译等待与原生改写的请求 ID 分离，并重跑 HTTP 与生命周期。原生调度测试最初因环境 SOCKS 代理缺少依赖而失败；临时清除该命令的代理变量、使用已缓存配置离线运行后 212 项通过，未修改依赖或全局环境。

本轮没有运行对象存储或正式 SemPIC 质量评测。此前已验证的执行结果保持为历史记录，不扩大为这些尚未执行的验证结论。
