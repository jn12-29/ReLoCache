import assert from "node:assert/strict";
import test from "node:test";
import { readCompletion, sseEvents } from "../src/relocache/webchat_assets/stream.mjs";

function response(frames, { bytewise = false } = {}) {
  const bytes = new TextEncoder().encode(frames);
  return new Response(new ReadableStream({
    start(controller) {
      if (bytewise) for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
      else controller.enqueue(bytes);
      controller.close();
    },
  }));
}

const frame = data => `data: ${JSON.stringify(data)}\r\n\r\n`;

test("SSE preserves split UTF-8, CRLF, comments, and multiline data", async () => {
  const stream = response(': keepalive\r\n\r\ndata: {"text":\r\ndata: "中文"}\r\n\r\ndata: [DONE]\r\n\r\n', { bytewise: true });
  const events = [];
  for await (const data of sseEvents(stream.body)) events.push(data);
  assert.deepEqual(events, ['{"text":\n"中文"}', "[DONE]"]);
});

test("TTFT ignores empty/role deltas and keeps final usage and PIC metrics", async () => {
  const events = [
    { choices: [{ delta: { role: "assistant", content: "" } }] },
    { choices: [{ delta: { content: "你好" } }] },
    { choices: [{ delta: { content: "。" }, finish_reason: "stop" }] },
    { choices: [], usage: { prompt_tokens: 160, completion_tokens: 3, prompt_tokens_details: { cached_tokens: 128 } }, metrics: { queue_time: 0.01 }, kv_transfer_params: { relocache: { reused_tokens: 120 } } },
  ];
  const ticks = [145, 220];
  const result = await readCompletion(response(events.map(frame).join("") + "data: [DONE]\n\n", { bytewise: true }), () => {}, { started: 100, clock: () => ticks.shift() });
  assert.equal(result.ttft, 45);
  assert.equal(result.duration, 120);
  assert.equal(result.text, "你好。");
  assert.equal(result.usage.completion_tokens, 3);
  assert.equal(result.usage.prompt_tokens_details.cached_tokens, 128);
  assert.equal(result.timing.queue_time, 0.01);
  assert.equal(result.pic.reused_tokens, 120);
  assert.equal(result.finishReason, "stop");
});

test("reasoning is timed as generated text and kept separate from the answer", async () => {
  const events = [{ choices: [{ delta: { reasoning: "分析" } }] }, { choices: [{ delta: { content: "回答" } }] }];
  const ticks = [20, 40];
  const result = await readCompletion(response(events.map(frame).join("") + "data: [DONE]\n\n"), () => {}, { started: 10, clock: () => ticks.shift() });
  assert.equal(result.ttft, 10);
  assert.equal(result.reasoning, "分析");
  assert.equal(result.text, "回答");
});

test("HTTP errors, SSE errors, and premature EOF stay failures", async () => {
  const options = { started: 0, clock: () => 1 };
  await assert.rejects(readCompletion(new Response('{"error":{"message":"unavailable"}}', { status: 503 }), () => {}, options), /unavailable/);
  await assert.rejects(readCompletion(response(frame({ error: { message: "reusable_cache_miss" } })), () => {}, options), /reusable_cache_miss/);
  await assert.rejects(readCompletion(response(frame({ choices: [{ delta: { content: "partial" } }] })), () => {}, options), /\[DONE\]/);
});

test("cancelling a request closes its SSE reader", async () => {
  let cancelled = false;
  const body = new ReadableStream({
    start(controller) { controller.enqueue(new TextEncoder().encode("data: first\n\n")); },
    cancel() { cancelled = true; },
  });
  for await (const data of sseEvents(body)) { assert.equal(data, "first"); break; }
  assert.equal(cancelled, true);
});
