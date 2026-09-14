# ReLoCache

基于 `sempic_iclr` Native PIC 实现的 vLLM 内容复用与推理服务。

独立环境已安装，服务逻辑已迁移，已修正元数据持久化、KV 持久化检查、connector 配置合并和请求 schema 的迁移偏差。Qwen3-8B P16 demo 已在 GPU 4–7、DP=4、TP=1 下完成六个数据集各一个样本的 Full Prefill/PIC 对比，共 12 次请求全部成功；结果见 [metrics.json](runs/sempic-demo-20260914/metrics.json)。本次验证包含 LoRA Writer 自动编译和 Reader soft tokens，不构成正式质量或性能评测。

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

- **SemPIC**：模型方法、训练、Writer 编译语义与科研实现、方法导出、数据集、prompt 和评测语义。
- **ReLoCache**：服务启停、内容接入与元数据持久化、请求渲染、编译与推理调用、客户端和服务指标。
- **vLLM fork**：请求执行、原生缓存、缺失内容调度、offload、位置映射和 fused-RoPE attention。

依赖方向为 `SemPIC → ReLoCache → vLLM fork`。当前只支持 vLLM。

## 仓库

`third_party/vllm/` 是独立 Git 仓库，以 submodule 管理；父仓库不跟踪其中的源码文件。Native PIC 改动基于上游 `08b3e67b669d` 提交，父仓库固定对应的 fork 版本。修改时先提交 fork，再更新父仓库的 submodule 引用。

两个仓库的开发分支均为 `main`。当前未配置发布远程；submodule URL 指向本机的 fork。发布后，将 `.gitmodules` 中的 URL 更新为 fork 的远程地址并运行 `git submodule sync`。

```bash
git status
git submodule status
git -C third_party/vllm status
```

初始化来源及环境约束见 [SOURCE.md](SOURCE.md)，接入边界见 [docs/integration.md](docs/integration.md)。
