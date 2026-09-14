# 项目约定

- 旧 `sempic_iclr`、`sempic_open`、`ref_repo` 及其环境只读，不干预已有任务。修改、验证及生成文件仅在新仓库和独立环境中进行。
- 基于当前源码拆分，遵循当前文档和用户决定，不照搬旧实现的所有选择。
- ReLoCache 定位为位置无关缓存的语义层库，负责内容表示、缓存身份、编译与复用语义；服务是其接入方式之一。SemPIC 负责方法、Writer 编译策略、数据集、prompt 和评测语义。
- 当前接入 vLLM，未来支持多后端；后端细节集中在接入模块，按实际需求扩展，不预建通用后端框架。
- 允许 vLLM fork 依赖并调用 ReLoCache，优先通过插件和现有扩展点接入；避免包安装循环依赖，上游 vLLM 不必强制安装 ReLoCache。
- 保留 vLLM 原生 KV 管理、调度、缓存查找和 offload 语义。
- KV Packet 在 Writer 编译时输入 soft tokens；Joint 再启用 LoRA Writer。复用表示，明确 Writer/Reader 执行阶段。
- `third_party/vllm` 保持独立 Git 子仓库，遵循其开发规则；先提交 fork，再更新主仓库引用。
- 最小化 vLLM 侵入，保留原生结构、功能与性能；复用逻辑尽量由 ReLoCache 承载，必要的引擎修改应形成可独立解释、可提交上游 PR 的窄接口或执行能力。减少私有实现耦合，不以复制原生逻辑换取更少的修改文件。
- 先读已有源码，优先复用 vLLM、FlashInfer、Triton 和参考实现；正确性与实际效率同时满足，不保留旧 API 兼容层。
- 使用项目级 `uv`，维护 `pyproject.toml`、`uv.lock`、`.venv`，不依赖全局 Python 或 root 权限。
- 模型读取 `../models`；仅使用空闲的 GPU 4–7，不中断已有任务。
- 自动维护新 Git 仓库，允许 commit。
