# vLLM 累计修改边界

比较范围固定为 `upstream-base`（`08b3e67b669d18d5bbd14e456b8bb4794916e402`）到当前 fork；不以中间提交作为比较基线。

```bash
git -C third_party/vllm diff upstream-base HEAD
```

当前累计差异包含 22 个源码文件，增加 314 行、删除 44 行。相关回归测试和说明统一放在 ReLoCache 主仓库的 `tests/`、`docs/` 中；fork 的测试和文档与 `upstream-base` 保持一致。

这是当前实现的清单，不代表最小必要修改。已有扩展点的覆盖范围、可收回的改动及实施取舍见[最小侵入接入评估](vllm-minimal-integration.md)。

PIC 服务仍依赖该 fork。ReLoCache 使用原生 endpoint/general plugins、CustomOp 注册、Scheduler 子类和调度输出处理；没有复制原生调度器、KV 管理器或模型执行主循环。

## 当前保留的源码修改

| 范围 | 相对原始版本的行为 | 当前用途 |
| --- | --- | --- |
| 输入渲染 | OnlineRenderer 类选择和 Responses item 转换方法。 | 自有 renderer 在结构化输入丢失信息前接管；仍使用原生校验、工具/推理解析、历史和生成流程。 |
| Request | 内部请求标记、必需缓存量、加载统计。 | 内部 Writer 不产生外部响应；缓存准入和搬运使用明确的请求状态。 |
| Engine 与 Scheduler | 自定义 Request 构造、缓存准入等待与错误、scheduler 方法调用；utility 可显式返回所有 DP 结果。 | 在页哈希前建立请求语义，在缓存准入时协调 Writer/加载；服务自行汇总各副本结果。 |
| Worker | ModelState 类选择。 | 自有 ModelState 通过已有生命周期准备逻辑位置及 attention metadata；原生位置准备、slot mapping 和模型执行主循环保持不变。 |
| Triton attention | Backend kernel 参数扩展；区间遍历、逻辑因果 mask、Key 变换和分段归约。 | 自有 backend 通过原生 registry 接入；原生 kernel 接收变换函数、tensor tuple 和 constexpr 配置，不识别 RoPE 参数。分块、KV 加载、softmax 和归约仍为原生实现。 |
| Offload | 页顺序变化处理、必需加载的容量等待、公开按请求查询加载/未完成工作、请求级加载统计。 | 页面排列变化时同步 offload keys；区分真实缺失与加载等待；按请求等待持久化并计量原生搬运。 |
| 存储生命周期 | 共享 mmap 在全部映射建立后 unlink。 | 进程退出后回收共享存储；这是独立的资源生命周期修复。 |
| 生成输出 | 传回请求失败原因；Completion/Chat 最终 usage chunk 携带 KV 元数据。 | 缓存缺失保持请求级错误，流式调用可以获得与非流式相同的复用统计。 |

## ReLoCache 负责的部分

- `rotary.py` 通过原生 CustomOp registry 注册 Query-only RoPE。原始 rotary 源码没有修改；标准与 Llama 3 的频率计算复用原生实现。
- `transport.py` 使用 `SamplingParams.extra_args` 传纯值元数据。Writer soft tokens 在每个 EngineCore 启动时加载；混合 embeddings 使用原生 tensor 字段。`EngineCoreRequest`、`InputProcessor`、`NewRequestData` 与基线一致。
- `scheduler.py` 在原生调度完成后填充 worker 布局和按存储顺序排列的 embeddings；worker 参数只保留布局，不重复传源内容。原生 Request 没有为 embedding 排列新增方法。
- `PICOnlineRenderer` 组合原生 BaseRenderer，共享 tokenizer、缓存与线程池；ReLoCache 请求类型在原生 serving 创建 SamplingParams 时绑定渲染元数据。没有复制原生 Completion/Chat/Responses 处理流程。
- `positions.py` 在 ModelState 准备模型输入时映射逻辑位置，保留稳定 GPU buffer。`input_batch.py`、`model_runner.py`、ModelState interface 与基线一致。额外一次 kernel 的测量和取舍见验证记录。
- `attention.py` 定义 PIC metadata 和外部 backend，RoPE 配置与参数装配不再放进原生 Triton backend。
- `config.py` 为 filesystem `root_dir` 和 object-store `prefix` 添加 `pre_rope` 路径分量；原生 `FileMapper` 与基线一致。
- `storage.py` 组合公开的各层 `lookup`、`on_schedule_end` 和 `on_request_finished`，实现驻留查询及异步 existence lookup 的提交和清理。文件系统、对象存储及 secondary-tier 基类源码没有为状态查询增加接口。
- 内容注册、编译协调、来源持久化、状态查询、DP 汇总和 flush 策略属于服务。原生 DP client 不根据 PIC 方法名选择聚合规则，Responses 不识别 PIC 内容类型。

这些修改尚未进入上游。保留扩展接口不意味着所有模型、attention backend 或存储配置均已验证；实际覆盖范围见[验证记录](decoupling-validation.md)。
