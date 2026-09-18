# vLLM 累计修改边界

比较范围固定为 `upstream-base`（`08b3e67b669d18d5bbd14e456b8bb4794916e402`）到当前 fork；不以中间提交作为比较基线。

```bash
git -C third_party/vllm diff upstream-base HEAD
```

当前累计差异为 31 个文件：28 个源码、2 个测试、1 个说明；增加 468 行、删除 47 行。

PIC 服务仍依赖该 fork。ReLoCache 使用原生 endpoint/general plugins、CustomOp 注册、Scheduler 子类和调度输出处理；没有复制原生调度器、KV 管理器或模型执行主循环。

## 当前保留的源码修改

| 范围 | 相对原始版本的行为 | 保留原因 |
| --- | --- | --- |
| 输入渲染 | Completion、Chat 与 Responses item 转换回调。 | 结构化内容必须在原生 tokenization、消息转换丢失信息之前接管；普通输入继续调用原生处理。 |
| Request 与 IPC | 扩展元数据、typed tensors、内部请求标记、必需缓存量、加载统计及 worker 元数据。 | 跨进程传递布局与 Writer 输入，沿用原生 Tensor IPC；内部 Writer 不产生外部响应。 |
| Engine 与 Scheduler | 自定义 Request 构造、缓存准入等待与错误、scheduler 方法调用；utility 可显式返回所有 DP 结果。 | 在页哈希前建立请求语义，在缓存准入时协调 Writer/加载；服务自行汇总各副本结果。 |
| Worker | ModelState 类选择、存储位置到逻辑位置映射。 | 共享物理页的多个引用可以处于不同逻辑位置；原生 position preparation 中需要传入映射。 |
| Triton attention | 区间遍历、物理索引与逻辑位置分离、Key 变换回调、逻辑因果 mask 和分段归约。 | 在同一个 attention kernel 内使用共享页并按引用位置旋转 Key。分块、KV 加载、softmax 和归约主流程仍为原生实现。 |
| Offload | 页顺序变化处理、必需加载的容量等待、公开按请求查询加载/未完成工作、请求级加载统计。 | 页面排列变化时同步 offload keys；区分真实缺失与加载等待；按请求等待持久化并计量原生搬运。 |
| 存储表示与生命周期 | 持久化命名空间记录 Key 编码；共享 mmap 在全部映射建立后 unlink。 | 区分不同 KV 表示；进程退出后回收共享存储。mmap 修改是独立的资源生命周期修复。 |
| 生成输出 | 传回请求失败原因；Completion/Chat 最终 usage chunk 携带 KV 元数据。 | 缓存缺失保持请求级错误，流式调用可以获得与非流式相同的复用统计。 |

## ReLoCache 负责的部分

- `rotary.py` 通过原生 CustomOp registry 注册 Query-only RoPE。原始 rotary 源码没有修改；标准与 Llama 3 的频率计算复用原生实现。
- `scheduler.py` 在原生调度完成后填充 worker 布局和按存储顺序排列的 embeddings；原生 Request 没有为 embedding 排列新增方法。
- `storage.py` 组合公开的各层 `lookup`、`on_schedule_end` 和 `on_request_finished`，实现驻留查询及异步 existence lookup 的提交和清理。文件系统、对象存储及 secondary-tier 基类源码没有为状态查询增加接口。
- 内容注册、编译协调、来源持久化、状态查询、DP 汇总和 flush 策略属于服务。原生 DP client 不根据 PIC 方法名选择聚合规则，Responses 不识别 PIC 内容类型。

这些修改尚未进入上游。保留扩展接口不意味着所有模型、attention backend 或存储配置均已验证；实际覆盖范围见[验证记录](decoupling-validation.md)。
