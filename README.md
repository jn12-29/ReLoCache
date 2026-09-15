# ReLoCache

ReLoCache 面向基于 PIC 的高效推理服务，负责 PIC 相关语义与核心组件的实现。当前实现基于 `sempic_iclr` Native PIC，接入 vLLM；未来支持多后端。

Writer 编译支持 Pre/Post soft tokens：部署配置加入 `writer_soft_tokens`，不启用 LoRA 时对应未压缩 KV Packet，配合 `writer_adapter` 时对应 Joint。显式与自动编译共用原生混合输入，保留 Pre/document/Post KV，并按完整长度推进 Reader 位置。配置格式见 [接入说明](docs/integration.md)。GPU 6 上的 Qwen3-0.6B 执行验证已通过，包含合成 rank-8 LoRA、chunked prefill、重复引用和缓存复用；[验证结果](runs/writer-soft-tokens-20260914/results.json)不构成方法质量或性能评测。

独立环境已安装，服务逻辑已迁移，已修正元数据持久化、KV 持久化检查、connector 配置合并和请求 schema 的迁移偏差。Qwen3-8B P16 demo 已在 GPU 4–7、DP=4、TP=1 下完成六个数据集各一个样本的 Full Prefill/PIC 对比，共 12 次请求全部成功；结果见 [metrics.json](runs/sempic-demo-20260914/metrics.json)。本次验证包含 LoRA Writer 自动编译和 Reader soft tokens，不构成正式质量或性能评测。

后续工作、实施顺序与完成标准见 [TODO 与未来计划](docs/roadmap.md)。

## 启动

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

该配置来自已导出的 Qwen3-8B P16 demo，包含模型、Writer LoRA 路径和 vLLM 参数。CLI 加载 ReLoCache endpoint plugin，并通过 vLLM 原生 `ServeSubcommand` 启动服务。参数说明可用 `uv run --no-sync relocache serve --help` 查看。

启动完成后检查：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

服务在前台运行，Ctrl-C 停止。生成使用 `/v1/completions`、`/v1/chat/completions` 或 `/v1/responses`；内容注册与编译使用 `/v1/reusable_contents/*`。

内容元数据默认保存在内存；部署配置可选 `metadata_dir`，以 JSON 保存内容来源供重启后解析引用。`compile` 的 `persist=true` 要求配置文件或对象存储 KV 层，并等待原生写入完成；仅配置 CPU offload 不满足持久化要求。配置示意和接口边界见 [docs/integration.md](docs/integration.md)。

## 职责

- **SemPIC**：主要负责训练和评测语义，包括训练产物导出、评测输入与评分。
- **ReLoCache**：PIC 核心组件与高效服务，包括缓存编译、内容复用、相关推理实现及后端接入。
- **推理后端**：提供原生执行基础，通过插件或扩展点调用 ReLoCache 的组件，尽量减少源码修改，复用原生能力并保持性能。

允许 vLLM fork 依赖并调用 ReLoCache。优先使用现有插件和扩展点，将后端细节留在接入模块；必要的 vLLM 修改以可提交上游的窄接口或执行能力为目标，保留原生功能与性能。避免包安装循环依赖，不要求上游 vLLM 强制安装 ReLoCache。

这是目标边界。当前仅接入 vLLM，PIC 调度、位置映射和 fused-RoPE 等实现仍在 fork 中，尚未按此边界完成收敛。具体接入约定见 [docs/integration.md](docs/integration.md)。

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
