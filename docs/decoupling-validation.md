# PIC 解耦验证

日期：2026-09-18。当前 PIC 核心、vLLM 接入、生命周期与请求统计拆分已完成。SemPIC 方法集成、正式质量/性能评测与后续方法扩展仍按[路线图](roadmap.md)推进。

## 实现边界

PIC 内容与 Writer 身份、编译协调、逻辑位置和 fused-RoPE 实现属于 ReLoCache；页哈希、请求/调度子类、IPC、worker buffers 属于 `backends.vllm`。fork 删除原 `core/pic.py`、`worker/gpu/pic.py` 及 PIC 测试，保留通用执行接口和原生 KV 管理、offload、attention 主循环。PIC 测试迁入主仓库。

阻止导入 `relocache` 后，原生 EngineCore、DefaultModelState、Request 与普通请求哈希仍可使用。核心布局、身份与轻量客户端不依赖 vLLM。attention 扩展函数在同一 Triton kernel 内融合执行，不新增 KV 搬运或单独的旋转 kernel。

## 已执行验证

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

运行配置、启动 PID、GPU 映射、代码状态、服务日志和响应保存在 [运行目录](../runs/pic-decoupling-20260918/)。主要结果：[HTTP](../runs/pic-decoupling-20260918/results.json)、[生命周期](../runs/pic-decoupling-20260918/lifecycle/results.json)、[attention](../runs/pic-decoupling-20260918/kernel/pytest.log)。运行目录为本地实验产物，不提交 Git；本文保留验证摘要。

首次 HTTP 验证发现 utility RPC 可变参数签名不符合原生参数转换规则，改为固定列表参数后通过；失败日志保留在 `attempt-1-rpc-signature`。随后用稳定操作 ID 将外部编译等待与原生改写的请求 ID 分离，并重跑 HTTP 与生命周期。原生调度测试最初因环境 SOCKS 代理缺少依赖而失败；临时清除该命令的代理变量、使用已缓存配置离线运行后 212 项通过，未修改依赖或全局环境。

本轮没有运行对象存储或正式 SemPIC 质量评测。此前已验证的执行结果保持为历史记录，不扩大为这些尚未执行的验证结论。
