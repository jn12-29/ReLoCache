# ReLoCache

ReLoCache 面向基于 PIC 的高效推理服务，负责 PIC 相关语义与核心组件的实现。当前实现基于 `sempic_iclr` Native PIC，接入 vLLM；未来支持多后端。

首次使用请阅读 [PIC 推理与评测使用指南](docs/user-guide.md)：从环境安装、部署配置到内容复用、soft tokens、质量评测和缓存性能实验，提供完整步骤及可运行示例，无需阅读源码。

交互演示支持长文问答、流式回答、TTFT、原生前缀缓存与 PIC 复用统计，以及清空 KV 缓存。启动步骤见下方 [Web Chat 演示](#web-chat-演示)，完整说明见[使用指南](docs/user-guide.md#11-web-chat-演示)。

Writer 编译支持 Pre/Post soft tokens：部署配置加入 `writer_soft_tokens`，不启用 LoRA 时对应 KV Packet，配合 `writer_adapter` 时对应 Joint。显式与自动编译共用原生混合输入，保留 Pre/document/Post KV，并按完整长度推进 Reader 位置。配置格式见 [接入说明](docs/integration.md)。GPU 6 上的 Qwen3-0.6B 执行验证已通过，包含合成 rank-8 LoRA、chunked prefill、重复引用和缓存复用；[验证结果](runs/writer-soft-tokens-20260914/results.json)不构成方法质量或性能评测。

独立环境已安装，服务逻辑已迁移，已修正元数据持久化、KV 持久化检查、connector 配置合并和请求 schema 的迁移偏差。Qwen3-8B P16 demo 已在 GPU 4–7、DP=4、TP=1 下完成六个数据集各一个样本的 Full Prefill/PIC 对比，共 12 次请求全部成功；结果见 [metrics.json](runs/sempic-demo-20260914/metrics.json)。本次验证包含 LoRA Writer 自动编译和 Reader soft tokens，不构成正式质量或性能评测。

PIC 核心与 vLLM 的当前拆分已完成。内容与 Writer 身份、编译协调、位置映射和 fused-RoPE 实现位于 ReLoCache；原生页管理、调度、offload 和 attention 主循环通过窄接口接入。输入、缓存生命周期与可观测性功能已同步补齐，见 [接入说明](docs/integration.md)和 [TODO 与未来计划](docs/roadmap.md)。EPIC、CacheBlend 等重算方法和压缩方法仍暂缓，后续规划保留。

GPU 6–7 的 Qwen3-0.6B、DP=2、TP=1 验证覆盖 LoRA Writer、Writer/Reader soft tokens、精确 token 输入、Chat/Responses、流式客户端与请求统计。文件系统 KV 验证覆盖请求级 cache miss、并发生成期间独立完成持久化，以及重启后的原生加载。验证范围和指标见 [拆分验证记录](docs/decoupling-validation.md)。

## Web Chat 演示

以下命令均在 ReLoCache 根目录、已有项目环境中执行。演示使用 Qwen3-4B-Instruct-2507、32K 上下文和已训练的 rank-8 LoRA Writer；Reader 使用基础模型。默认文档为约 1.9 万字符、13,902 tokens 的企业运营手册。

本机模型位于 `../models/Qwen3-4B-Instruct-2507`，Writer 已复制到 `artifacts/webchat-qwen3-4b-two-stage/writer-adapter`。`artifacts/` 不提交 Git；若该目录尚未准备，从只读训练产物复制这两个标准 PEFT 文件：

```bash
mkdir -p artifacts/webchat-qwen3-4b-two-stage/writer-adapter
cp \
  ../sempic_open/train_outputs/qwen_3_4b/native/sempic/20260920_035603_two_stage_reference_stage2/lora/adapter_config.json \
  ../sempic_open/train_outputs/qwen_3_4b/native/sempic/20260920_035603_two_stage_reference_stage2/lora/adapter_model.safetensors \
  artifacts/webchat-qwen3-4b-two-stage/writer-adapter/
```

**终端一：启动推理服务。** 先确认 GPU 空闲并符合当前设备授权；`1` 是本次演示使用的设备，可按实际情况替换。

```bash
VLLM_SERVER_DEV_MODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
uv run --no-sync relocache serve \
  --config examples/webchat/deployment.json \
  --cuda-visible-devices 1 \
  --host 127.0.0.1 --port 8010
```

`VLLM_SERVER_DEV_MODE=1` 开放原生缓存重置接口，供页面的「清空 KV 缓存」按钮使用。配置已开启原生 `cached_tokens` 统计。等待服务启动后检查：

```bash
curl --fail http://127.0.0.1:8010/health
```

**终端二：启动 Web Chat。**

```bash
uv run --no-sync relocache webchat \
  --api-base http://127.0.0.1:8010 \
  --host 0.0.0.0 --port 8080
```

浏览器访问 `http://<服务器地址>:8080/demo`；在服务器本机访问 `http://127.0.0.1:8080/demo`。页面通过同源代理访问推理 API；只在本机使用时，也可省略终端二，直接打开 `http://127.0.0.1:8010/demo`。

建议先点击「提前编译 PIC」，再比较固定前置上下文下的连续问答与切换前置上下文后的问答。Full Prefill 保留原生 prefix caching，不保证每轮 PIC 都更快。「清空 KV 缓存」同时使两者的 GPU 缓存失效，成功后清空页面历史和编译状态、保留文档；不会释放显存池。两路请求顺序执行、交替先后，编译耗时与 Reader TTFT 分别观察。

停止时在两个终端分别按 Ctrl-C；仅停止 Web Chat 不会停止推理服务或释放其 GPU 占用。

## API 服务启动

在 ReLoCache 根目录使用已有环境，等待 GPU 4–7 空闲后执行：

```bash
uv run --no-sync relocache serve \
  --config artifacts/qwen3-8b-P16/deployment.json \
  --cuda-visible-devices 4,5,6,7 \
  --data-parallel-size 4 \
  --tensor-parallel-size 1 \
  --host 127.0.0.1 \
  --port 8000
```

该配置来自已导出的 Qwen3-8B P16 demo，包含模型、Writer LoRA 路径和 vLLM 参数。CLI 加载 ReLoCache endpoint 和 CustomOp plugins，并通过 vLLM 原生 `ServeSubcommand` 启动服务。参数说明可用 `uv run --no-sync relocache serve --help` 查看。

启动完成后检查：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

服务在前台运行，Ctrl-C 停止。生成使用 `/v1/completions`、`/v1/chat/completions` 或 `/v1/responses`；内容注册与编译使用 `/v1/reusable_contents/*`。

内容元数据默认保存在内存；部署配置可选 `metadata_dir`，以 JSON 保存内容来源供重启后解析引用。`compile` 的 `persist=true` 要求配置文件或对象存储 KV 层，并等待原生写入完成；仅配置 CPU offload 不满足持久化要求。配置示意和接口边界见 [docs/integration.md](docs/integration.md)。

## 职责

内容统一分为 **Reusable Content** 和 **Dynamic Content**：前者由 Writer 独立编译为 PIC，供 Reader 跨位置引用；后者随当前上下文执行，也可通过原生前缀缓存复用。输入可按顺序组合两类内容，例如 `[Dynamic Content: 指令] [Reusable Content: 文档] [Dynamic Content: 问题]`。

- **SemPIC**：主要负责训练和评测语义，包括训练产物导出、评测输入与评分。
- **ReLoCache**：PIC 核心组件与高效服务，包括缓存编译、内容复用、相关推理实现及后端接入。
- **推理后端**：提供原生执行基础，通过插件或扩展点调用 ReLoCache 的组件，尽量减少源码修改，复用原生能力并保持性能。

允许 vLLM fork 依赖并调用 ReLoCache。优先使用现有插件和扩展点，将后端细节留在接入模块；必要的 vLLM 修改以可提交上游的窄接口或执行能力为目标，保留原生功能与性能。避免包安装循环依赖，不要求上游 vLLM 强制安装 ReLoCache。

当前仅接入 vLLM。PIC 组件及其测试归 ReLoCache，fork 保留通用执行扩展；具体模块与接口见 [docs/integration.md](docs/integration.md)，相对原始 vLLM 的累计修改见 [docs/vllm-patch.md](docs/vllm-patch.md)。

## 仓库

`third_party/vllm/` 是独立 Git 仓库，以 submodule 管理；父仓库不跟踪其中的源码文件。Native PIC 改动基于真实上游 `08b3e67b669d` 提交，该提交作为 shallow 历史边界并标记为 `upstream-base`，其后集中保存当前 fork 改动。父仓库固定对应的 fork 版本。修改时先提交 fork，再更新父仓库的 submodule 引用。

两个仓库的开发分支均为 `main`，`origin` 分别指向私有 GitHub 仓库 `jn12-29/ReLoCache` 和 `jn12-29/vllm-relocache`；submodule URL 指向后者。

```bash
git status
git submodule status
git -C third_party/vllm status
git -C third_party/vllm diff upstream-base..main
```

初始化来源及环境约束见 [SOURCE.md](SOURCE.md)，接入边界见 [docs/integration.md](docs/integration.md)。
