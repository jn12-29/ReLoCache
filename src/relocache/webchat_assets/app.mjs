import { presets } from "./presets.mjs";
import { readCompletion } from "./stream.mjs";
import { marked } from "./marked.mjs";
import DOMPurify from "./purify.mjs";

const $ = id => document.getElementById(id);
const state = { mode: "pic", busy: false, model: null, histories: { pic: [], full: [] }, controller: null };
let contentId;
let toastTimer;

function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 4500);
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function resetSession({ sourceChanged = false } = {}) {
  if (sourceChanged || !contentId) contentId = `demo-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  state.histories = { pic: [], full: [] };
  for (const kind of ["pic", "full"]) {
    const empty = element("div", "empty-state");
    empty.append(element("div", "empty-glyph", "[ ↗ ]"));
    empty.append(element("h2", "", kind === "pic" ? "一份文档，不止一次对话" : "从普通推理开始"));
    empty.append(element("p", "", presets[$("scenario").value].description));
    $(`messages-${kind}`).replaceChildren(empty);
    $(`state-${kind}`).textContent = "就绪";
  }
  if (sourceChanged) $("compile-status").textContent = "";
  $("document-size").textContent = `${$("document").value.length.toLocaleString()} 字符`;
}

function loadPreset() {
  const preset = presets[$("scenario").value];
  $("document").value = preset.text;
  $("question").value = preset.questions[0];
  $("suggestions").replaceChildren(...preset.questions.map(question => {
    const button = element("button", "", question);
    button.type = "button";
    button.addEventListener("click", () => {
      $("question").value = question;
      if (state.model && !state.busy) $("chat-form").requestSubmit();
      else $("question").focus();
    });
    return button;
  }));
  resetSession({ sourceChanged: true });
}

function setBusy(busy) {
  state.busy = busy;
  for (const id of ["scenario", "document", "restore", "system", "new-chat", "max-tokens", "temperature"]) $(id).disabled = busy;
  document.querySelectorAll("[data-mode], #suggestions button").forEach(button => { button.disabled = busy; });
  $("send").disabled = busy || !state.model;
  $("compile").disabled = busy || !state.model || !$("document").value.trim();
  $("stop").hidden = !busy;
  $("send").hidden = busy;
}

async function requestJSON(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message ?? JSON.stringify(data.detail ?? data));
  return data;
}

async function connect() {
  $("connection").textContent = "正在连接推理服务";
  $("reconnect").disabled = true;
  try {
    const models = await requestJSON("/v1/models");
    if (!models.data?.length) throw new Error("服务未返回可用模型");
    state.model = models.data[0].id;
    $("model-name").textContent = state.model;
    $("connection").textContent = "推理服务已连接";
    $("connection-dot").className = "connection-dot online";
    try {
      const deployment = await requestJSON("/v1/relocache");
      $("model-name").textContent = deployment.model.split("/").filter(Boolean).at(-1);
      $("writer-name").textContent = deployment.writer ? "LoRA Writer" : "Base Writer";
      $("writer-name").title = deployment.method_id;
      $("model-name").title = `Writer: ${deployment.method_id} · DP ${deployment.data_parallel_size} / TP ${deployment.tensor_parallel_size}`;
      if (!deployment.auto_compile) $("session-note").textContent = "自动编译已关闭，请先点击「提前编译 PIC」。新对话保留引擎缓存。";
    } catch { /* Ordinary OpenAI-compatible servers can still run Full Prefill. */ }
  } catch (error) {
    state.model = null;
    $("connection").textContent = "推理服务未连接 · 点击 ↻ 重试";
    $("connection").title = error.message;
    $("connection-dot").className = "connection-dot offline";
    $("model-name").textContent = "模型未连接";
    $("writer-name").textContent = "";
  } finally {
    $("reconnect").disabled = false;
    setBusy(state.busy);
  }
}

function buildMessages(kind, question) {
  const messages = [];
  if ($("system").value.trim()) messages.push({ role: "system", content: $("system").value.trim() });
  for (const turn of [...state.histories[kind], { question }]) {
    if (messages.every(message => message.role !== "user") && $("document").value.trim()) {
      const document = $("document").value;
      const prefix = "请阅读以下参考资料：\n\n";
      const suffix = `\n\n参考资料结束。\n问题：${turn.question}`;
      messages.push({ role: "user", content: kind === "pic" ? [
        { type: "text", text: prefix },
        { type: "reusable_content", id: contentId, text: document },
        { type: "text", text: suffix },
      ] : prefix + document + suffix });
    } else messages.push({ role: "user", content: turn.question });
    if (turn.answer !== undefined) messages.push({ role: "assistant", content: turn.answer });
  }
  return messages;
}

function addTurn(kind, question) {
  const container = $(`messages-${kind}`);
  container.querySelector(".empty-state")?.remove();
  const user = element("article", "message user");
  user.append(element("div", "message-label", "YOU"), element("div", "message-body", question));
  const assistant = element("article", "message assistant");
  assistant.append(element("div", "message-label", kind === "pic" ? "RELOCACHE / PIC" : "FULL PREFILL"));
  const reasoning = element("details", "reasoning");
  reasoning.hidden = true;
  reasoning.append(element("summary", "", "推理过程"), element("pre"));
  const body = element("div", "message-body pending-text", "正在等待首个生成片段……");
  const metrics = element("div", "metrics");
  const values = {};
  for (const [key, title] of [["ttft", "TTFT"], ["duration", "总耗时"], ["tokens", "输出 tokens"], ["rate", "tokens/s · 端到端"]]) {
    const metric = element("div", "metric");
    values[key] = element("strong", "", "—");
    metric.append(element("small", "", title), values[key]);
    metrics.append(metric);
  }
  const detail = element("details", "metric-detail");
  detail.hidden = true;
  detail.append(element("summary"), element("pre"));
  assistant.append(reasoning, body, metrics, detail);
  container.append(user, assistant);

  return {
    update(result, elapsed) {
      const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;
      body.classList.toggle("pending-text", !result.text);
      if (result.text) body.innerHTML = DOMPurify.sanitize(marked.parse(result.text));
      else body.textContent = result.reasoning ? "正在推理……" : "正在等待首个生成片段……";
      reasoning.hidden = !result.reasoning;
      reasoning.querySelector("pre").textContent = result.reasoning;
      values.ttft.textContent = result.ttft === null ? "—" : `${Math.round(result.ttft)} ms`;
      values.duration.textContent = `${((result.duration ?? elapsed) / 1000).toFixed(2)} s`;
      values.tokens.textContent = result.usage?.completion_tokens ?? "—";
      values.rate.textContent = result.duration && result.usage?.completion_tokens
        ? (result.usage.completion_tokens / (result.duration / 1000)).toFixed(1) : "—";
      if (result.pic) {
        detail.hidden = false;
        const pic = result.pic;
        detail.querySelector("summary").textContent = `KV 复用 ${pic.reused_tokens ?? 0} tokens / ${pic.reused_pages ?? 0} 页 · 自动编译 ${pic.auto_compiles ?? 0} · Reader prefill ${pic.prefill_tokens_executed ?? "—"} tokens`;
        detail.querySelector("pre").textContent = JSON.stringify(pic, null, 2);
      }
      if (nearBottom) container.scrollTop = container.scrollHeight;
    },
    error(message) {
      assistant.classList.add("error");
      body.classList.remove("pending-text");
      if (body.classList.contains("has-output")) body.append(element("p", "", message));
      else body.textContent = message;
    },
    markOutput() { body.classList.add("has-output"); },
  };
}

async function generate(kind, question, signal) {
  const messages = buildMessages(kind, question);
  const turn = addTurn(kind, question);
  const started = performance.now();
  let current = { text: "", reasoning: "", ttft: null, duration: null, usage: null, pic: null };
  const timer = setInterval(() => turn.update(current, performance.now() - started), 100);
  $(`state-${kind}`).textContent = "生成中";
  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST", signal, headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: state.model, messages, stream: true, stream_options: { include_usage: true },
        max_tokens: Number($("max-tokens").value), temperature: Number($("temperature").value), seed: 42,
      }),
    });
    const result = await readCompletion(response, update => {
      current = update;
      if (current.text) turn.markOutput();
      turn.update(current, performance.now() - started);
    }, { started });
    state.histories[kind].push({ question, answer: result.text });
    $(`state-${kind}`).textContent = result.finishReason === "length" ? "达到输出上限" : "已完成";
    if (!result.text) toast("本次没有生成回答正文，可提高输出上限后重试。");
  } catch (error) {
    current.duration = performance.now() - started;
    turn.update(current, current.duration);
    turn.error(error.name === "AbortError" ? "已停止生成。本轮不加入后续对话上下文。" : error.message);
    $(`state-${kind}`).textContent = error.name === "AbortError" ? "已停止" : "请求失败";
  } finally {
    clearInterval(timer);
  }
}

$("chat-form").addEventListener("submit", async event => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question || state.busy || !state.model) return;
  if (!$("max-tokens").reportValidity() || !$("temperature").reportValidity()) return;
  const kinds = state.mode === "compare" ? ["pic", "full"] : [state.mode];
  state.controller = new AbortController();
  setBusy(true);
  $("question").value = "";
  try { await Promise.all(kinds.map(kind => generate(kind, question, state.controller.signal))); }
  finally { state.controller = null; setBusy(false); $("question").focus(); }
});

$("question").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("chat-form").requestSubmit();
  }
});
$("stop").addEventListener("click", () => state.controller?.abort());
$("new-chat").addEventListener("click", () => { resetSession(); toast("已开始新对话，引擎缓存保留。"); });
$("reconnect").addEventListener("click", connect);
$("scenario").addEventListener("change", loadPreset);
$("restore").addEventListener("click", loadPreset);
$("document").addEventListener("input", () => { resetSession({ sourceChanged: true }); setBusy(false); });
$("system").addEventListener("input", () => resetSession());
document.querySelectorAll("[data-mode]").forEach(button => button.addEventListener("click", () => {
  if (state.mode === button.dataset.mode) return;
  state.mode = button.dataset.mode;
  for (const option of document.querySelectorAll("[data-mode]")) {
    option.classList.toggle("selected", option === button);
    option.setAttribute("aria-pressed", String(option === button));
  }
  $("panel-pic").hidden = state.mode === "full";
  $("panel-full").hidden = state.mode === "pic";
  resetSession();
  toast("已切换模式并开始新对话，引擎缓存保留。");
}));
$("compile").addEventListener("click", async () => {
  state.controller = new AbortController();
  setBusy(true);
  $("compile-status").textContent = "正在独立编译文档……";
  const started = performance.now();
  try {
    const result = await requestJSON("/v1/reusable_contents/compile", {
      method: "POST", signal: state.controller.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content_id: contentId, text: $("document").value, persist: false }),
    });
    $("compile-status").textContent = `编译完成 · ${result.logical_length} tokens · ${((performance.now() - started) / 1000).toFixed(2)} s · DP rank ${result.compilation.rank}`;
  } catch (error) {
    $("compile-status").textContent = error.name === "AbortError" ? "编译请求已取消。" : `编译失败：${error.message}`;
  } finally { state.controller = null; setBusy(false); }
});

presets.forEach((preset, index) => {
  const option = element("option", "", preset.name);
  option.value = index;
  $("scenario").append(option);
});
loadPreset();
connect();
