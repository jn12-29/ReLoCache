/** Visible business context precedes the same reusable document in both paths. */
export const contexts = [
  {
    name: "行政助理 · 出差咨询",
    text: "你是华东交付团队的行政助理。员工即将出差，请从参考资料中提取与问题直接相关的费用标准，用简短的条目回答。",
  },
  {
    name: "财务审核 · 项目核对",
    text: "你是公司财务共享中心的审核专员。本次咨询来自一个新项目，需要核对制度原文。请给出适用标准、必要条件和相应章节，避免推测未提供的信息。",
  },
  {
    name: "新员工 · 首次报销",
    text: "你正在帮助一位刚入职、第一次提交差旅申请的新员工。他还不熟悉城市分类、住宿与餐补的区别。请用通俗语言解释适用条款，并保留关键数值。",
  },
];

export function readerMessages({ kind, system, context, document, contentId, history, question }) {
  const messages = [];
  if (system.trim()) messages.push({ role: "system", content: system.trim() });
  for (const [index, turn] of [...history, { question }].entries()) {
    if (index === 0) {
      const prefix = `${context.trim()}\n\n请阅读以下参考资料：\n\n`;
      const suffix = `\n\n参考资料结束。\n问题：${turn.question}`;
      messages.push({ role: "user", content: kind === "pic" && document.trim() ? [
        { type: "text", text: prefix },
        { type: "reusable_content", id: contentId, text: document },
        { type: "text", text: suffix },
      ] : prefix + document + suffix });
    } else messages.push({ role: "user", content: turn.question });
    if (turn.answer !== undefined) messages.push({ role: "assistant", content: turn.answer });
  }
  return messages;
}

export function cacheObservation(result, kind) {
  const cached = result.usage?.prompt_tokens_details?.cached_tokens;
  if (kind === "pic") {
    if (!result.pic) return "等待服务端 PIC 统计";
    return `PIC 文档复用 ${result.pic.reused_tokens ?? 0} tokens · Reader prefill ${result.pic.prefill_tokens_executed} tokens`;
  }
  if (cached === undefined || cached === null) return "原生前缀命中：未报告（服务需开启 enable_prompt_tokens_details）";
  return `原生前缀命中 ${cached} / ${result.usage.prompt_tokens} tokens · 非缓存输入 ${result.usage.prompt_tokens - cached} tokens`;
}
