# 项目约定

- 不修改服务之外的研究仓库、已有环境或运行中的任务；验证产物写入独立目录。
- 基于当前源码拆分，遵循当前文档和用户决定，不照搬旧实现的所有选择。
- ReLoCache 面向基于 PIC 的高效推理服务，负责 PIC 相关语义与核心组件的实现；SemPIC 主要负责训练和评测语义。
- 当前接入 vLLM，未来支持多后端；后端细节集中在接入模块，按实际需求扩展，不预建通用后端框架。
- 允许 vLLM fork 依赖并调用 ReLoCache，优先通过插件和现有扩展点接入；避免包安装循环依赖，上游 vLLM 不必强制安装 ReLoCache。
- 保留 vLLM 原生 KV 管理、调度、缓存查找和 offload 语义。
- KV Packet 在 Writer 编译时输入 soft tokens；Joint 再启用 LoRA Writer。复用表示，明确 Writer/Reader 执行阶段。
- `third_party/vllm` 保持独立 Git 子仓库，遵循其开发规则；先提交 fork，再更新主仓库引用。
- 推理后端通过插件或扩展点调用 ReLoCache 的组件，尽量减少源码修改，复用原生能力并保持性能；必要的引擎修改应形成可独立解释、可提交上游 PR 的窄接口或执行能力。减少私有实现耦合，不以复制原生逻辑换取更少的修改文件。
- 先读已有源码，优先复用 vLLM、FlashInfer、Triton 和参考实现；正确性与实际效率同时满足，不保留旧 API 兼容层。
- 使用项目级 `uv`，维护 `pyproject.toml`、`uv.lock`、`.venv`，不依赖全局 Python 或 root 权限。
- 模型路径与 GPU 由部署环境指定；仅使用分配给自己的空闲设备，不中断已有任务。
- 自动维护新 Git 仓库，允许 commit。
