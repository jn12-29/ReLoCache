# ReLoCache

基于 `sempic_iclr` Native PIC 实现的 vLLM 内容复用与推理服务。

当前仅完成仓库初始化：vLLM 源码基线已经提取，服务逻辑尚未迁移，尚未创建运行环境。

## 职责

- **SemPIC**：模型方法、训练、Writer 编译语义与科研实现、方法导出、数据集、prompt 和评测语义。
- **ReLoCache**：服务启停、内容接入与元数据持久化、请求渲染、编译与推理调用、客户端和服务指标。
- **vLLM fork**：请求执行、原生缓存、缺失内容调度、offload、位置映射和 fused-RoPE attention。

依赖方向为 `SemPIC → ReLoCache → vLLM fork`。当前只支持 vLLM。

## 仓库

`third_party/vllm/` 是独立 Git 仓库，以 submodule 固定版本；父仓库不跟踪其中的源码文件。其提交历史保留真实上游基线，再提交当前 Native PIC 改动。

两个仓库的开发分支均为 `main`。当前未配置发布远程；submodule URL 指向本机的 fork。发布后，将 `.gitmodules` 中的 URL 更新为 fork 的远程地址并运行 `git submodule sync`。

```bash
git status
git submodule status
git -C third_party/vllm status
```

修改 fork 时先在子仓库提交，再在父仓库提交 submodule 版本更新。

初始化来源及环境约束见 [SOURCE.md](SOURCE.md)，接入边界见 [docs/integration.md](docs/integration.md)。
