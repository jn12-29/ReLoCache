/** Decode SSE across arbitrary transport and UTF-8 boundaries. */
export async function* sseEvents(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  function dataOf(frame) {
    return frame.split(/\r?\n/).filter(line => line.startsWith("data:"))
      .map(line => line.slice(5).replace(/^ /, "")).join("\n");
  }
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = /\r?\n\r?\n/.exec(buffer))) {
        const data = dataOf(buffer.slice(0, boundary.index));
        buffer = buffer.slice(boundary.index + boundary[0].length);
        if (!data) continue;
        yield data;
        if (data === "[DONE]") return;
      }
      if (done) {
        const data = dataOf(buffer);
        if (data) yield data;
        return;
      }
    }
  } finally {
    await reader.cancel();
    reader.releaseLock();
  }
}

/** TTFT starts before fetch and ignores role-only, usage, and empty deltas. */
export async function readCompletion(response, onUpdate, { started, clock = () => performance.now() }) {
  if (!response.ok) {
    const text = await response.text();
    let message;
    try { const data = JSON.parse(text); message = data.error?.message ?? data.detail; } catch { /* Keep the HTTP body. */ }
    throw new Error(typeof message === "string" ? message : text || `HTTP ${response.status}`);
  }
  const result = { text: "", reasoning: "", ttft: null, duration: null, usage: null, pic: null, timing: null, finishReason: null };
  for await (const data of sseEvents(response.body)) {
    if (data === "[DONE]") {
      result.duration = clock() - started;
      onUpdate(result);
      return result;
    }
    const event = JSON.parse(data);
    if (event.error) throw new Error(event.error.message ?? JSON.stringify(event.error));
    if (event.usage) result.usage = event.usage;
    if (event.metrics) result.timing = event.metrics;
    if (event.kv_transfer_params?.relocache) result.pic = event.kv_transfer_params.relocache;
    const choice = event.choices?.[0];
    if (choice?.finish_reason === "error") throw new Error(String(choice.stop_reason ?? "推理请求失败"));
    const content = choice?.delta?.content ?? "";
    const reasoning = choice?.delta?.reasoning ?? choice?.delta?.reasoning_content ?? "";
    if ((content || reasoning) && result.ttft === null) result.ttft = clock() - started;
    result.text += content;
    result.reasoning += reasoning;
    if (choice?.finish_reason) result.finishReason = choice.finish_reason;
    onUpdate(result);
  }
  throw new Error("流式连接在 [DONE] 前结束，回答可能不完整。");
}
