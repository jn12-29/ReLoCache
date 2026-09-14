# 评测约定

- 数据集、artifact 转换、prompt 策略、评分和聚合留在 demo，不进入服务包。
- 生成统一经 ReLoCache 调用 vLLM API server，不用进程内 `LLM` 或手工数据分片。
- 在允许的空闲 GPU 上优先原生 DP；单卡放不下用 TP，可组合 TP × DP。设备和并行数由 CLI 控制。
- Full Prefill 提交普通 messages，PIC 提交内容引用；评测不要求 token IDs 对齐。
- 默认质量评测；已有 timing 不代表正式延时评测。延时实验记录缓存初态、reset 范围、并发、TP/DP、生成配置和指标口径。
- SSD 冷缓存：配置文件 KV 存储，启动时设置 `VLLM_SERVER_DEV_MODE=1`、`pic.auto_compile=false`；编译并等待 `/v1/reusable_contents/flush`，服务空闲后调用 `POST /reset_prefix_cache?reset_external=true` 并确认成功。清空 GPU/CPU 缓存，保留文件 KV；测量期间不逐请求预编译。
- Reader 计时排除准备、编译、flush 和 reset。运行前 reset 后允许自然复用，逐请求 reset 属于另一实验。
- 区分完整 HTTP 响应延时与流式 TTFT；引擎 prefill 时间不含排队，不能当作端到端 TTFT。
