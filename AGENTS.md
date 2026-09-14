# 项目约定

- 旧 `sempic_iclr`、`sempic_open`、`ref_repo` 及其环境只读，不干预已有任务。修改、验证及生成文件仅在新仓库和独立环境中进行。
- 基于当前源码拆分，遵循当前文档和用户决定，不照搬旧实现的所有选择。
- ReLoCache 负责服务与内容管理；SemPIC 负责方法、Writer 编译语义、数据集、prompt 和评测语义。
- 直接调用 vLLM，不做多后端抽象；fork 不反向依赖 ReLoCache 或 SemPIC。
- 保留 vLLM 原生 KV 管理、调度、缓存查找和 offload 语义。
- KV Packet 在 Writer 编译时输入 soft tokens；Joint 再启用 LoRA Writer。复用表示，明确 Writer/Reader 执行阶段。
- `third_party/vllm` 保持独立 Git 子仓库，遵循其开发规则；先提交 fork，再更新主仓库引用。
- 对`third_party/vllm` 进行最小化、最简化、最充分复用原有功能和逻辑的修改，并且保持原有功能与性能，不破坏原有代码结构，保持高性能。
- 先读已有源码，优先复用 vLLM、FlashInfer、Triton 和参考实现；正确性与实际效率同时满足，不保留旧 API 兼容层。
- 使用项目级 `uv`，维护 `pyproject.toml`、`uv.lock`、`.venv`，不依赖全局 Python 或 root 权限。
- 模型读取 `../models`；仅使用空闲的 GPU 4–7，不中断已有任务。
- 自动维护新 Git 仓库，允许 commit。
