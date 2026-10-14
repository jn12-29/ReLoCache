import assert from "node:assert/strict";
import test from "node:test";
import { contexts, readerMessages, cacheObservation } from "../src/relocache/webchat_assets/conversation.mjs";

const input = {
  system: "Answer from the document.", context: contexts[0].text,
  document: "A reusable document.", contentId: "document", history: [], question: "What does it say?",
};

test("PIC and Full Prefill share visible context, document, and question", () => {
  const full = readerMessages({ ...input, kind: "full" });
  const pic = readerMessages({ ...input, kind: "pic" });
  assert.deepEqual(full[0], pic[0]);
  assert.equal(full[1].content, pic[1].content.map(part => part.text).join(""));
});

test("switching context changes the causal prefix while preserving the PIC source", () => {
  const first = readerMessages({ ...input, kind: "pic" });
  const next = readerMessages({ ...input, kind: "pic", context: contexts[1].text });
  assert.notEqual(first[1].content[0].text, next[1].content[0].text);
  assert.deepEqual(first[1].content[1], next[1].content[1]);
  const repeat = readerMessages({ ...input, kind: "pic" });
  assert.deepEqual(repeat, first);
});

test("continued dialogue keeps the old prefix so native caching can work", () => {
  const first = readerMessages({ ...input, kind: "full" });
  const next = readerMessages({ ...input, kind: "full", history: [{ question: input.question, answer: "The first answer." }], question: "And then?" });
  assert.deepEqual(next.slice(0, 2), first);
  assert.deepEqual(next.slice(2), [{ role: "assistant", content: "The first answer." }, { role: "user", content: "And then?" }]);
});

test("native prefix hits stay visible and unavailable counters are not zero", () => {
  assert.equal(cacheObservation({ usage: { prompt_tokens: 1000, prompt_tokens_details: { cached_tokens: 960 } } }, "full"), "原生前缀命中 960 / 1000 tokens · 非缓存输入 40 tokens");
  assert.match(cacheObservation({ usage: { prompt_tokens: 1000 } }, "full"), /未报告/);
  assert.match(cacheObservation({ pic: { reused_tokens: 900, prefill_tokens_executed: 50 }, usage: { prompt_tokens: 1000, prompt_tokens_details: { cached_tokens: 944 } } }, "pic"), /PIC 文档复用 900 tokens · Reader prefill 50 tokens/);
});
