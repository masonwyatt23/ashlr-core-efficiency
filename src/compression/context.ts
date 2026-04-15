/**
 * 3-tier context compression — extracted from ashlrcode/src/agent/context.ts.
 *
 * Tier 1: autoCompact — summarize older messages via LLMSummarizer
 * Tier 2: snipCompact — truncate tool results > 2000 chars
 * Tier 3: contextCollapse — remove short/duplicate older messages
 *
 * No coupling to ashlrcode's ProviderRouter — everything flows through
 * the LLMSummarizer interface in ../types, so ashlr-plugin can reuse this
 * module against any streaming-capable LLM endpoint.
 */

import type { LLMSummarizer, Message } from "../types/index.ts";
import { estimateTokensFromMessages } from "../tokens/index.ts";

export interface ContextConfig {
  /** Max tokens before triggering compaction (default: 100000) */
  maxContextTokens: number;
  /** Tokens to reserve for the response (default: 8192) */
  reserveTokens: number;
  /** Number of recent messages to keep at full fidelity (default: 10) */
  recentMessageCount: number;
}

export const DEFAULT_CONFIG: ContextConfig = {
  maxContextTokens: 100_000,
  reserveTokens: 8192,
  recentMessageCount: 10,
};

/**
 * Tier 3: contextCollapse — remove redundant messages from older history.
 * - Remove short assistant messages (< 10 chars)
 * - Deduplicate consecutive tool results with similar content
 * - Keep last 5 messages at full fidelity
 */
export function contextCollapse(messages: Message[]): Message[] {
  if (messages.length <= 5) return messages;

  const keepRecent = 5;
  const older = messages.slice(0, -keepRecent);
  const recent = messages.slice(-keepRecent);

  const collapsed: Message[] = [];
  let lastToolResultHash = "";

  let skipNext = false;
  for (const msg of older) {
    if (skipNext) { skipNext = false; continue; }

    if (msg.role === "assistant" && typeof msg.content === "string" && msg.content.trim().length < 10) {
      skipNext = true;
      continue;
    }

    if (msg.role === "user" && Array.isArray(msg.content)) {
      const toolResults = msg.content.filter((b) => b.type === "tool_result");
      if (toolResults.length > 0) {
        const hash = toolResults.map((b) => (b.type === "tool_result" ? b.content.slice(0, 200) : "")).join("|");
        if (hash === lastToolResultHash) continue;
        lastToolResultHash = hash;
      }
    }

    collapsed.push(msg);
  }

  return [...collapsed, ...recent];
}

/**
 * Check if context needs compaction.
 *
 * @param actualTokensUsed - If provided, uses the real token count from the
 *   last API response instead of the chars/4 estimate.
 */
export function needsCompaction(
  messages: Message[],
  systemPromptTokens: number,
  config: Partial<ContextConfig> = {},
  actualTokensUsed?: number,
): boolean {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const messageTokens = actualTokensUsed ?? estimateTokensFromMessages(messages);
  return messageTokens + systemPromptTokens > cfg.maxContextTokens - cfg.reserveTokens;
}

/**
 * Tier 1: autoCompact — summarize older messages.
 * Splits messages at the last N, summarizes everything before that window,
 * and prepends the summary as a synthetic user/assistant exchange.
 */
export async function autoCompact(
  messages: Message[],
  summarizer: LLMSummarizer,
  config: Partial<ContextConfig> = {},
): Promise<Message[]> {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  if (messages.length <= cfg.recentMessageCount) return messages;

  const splitIndex = messages.length - cfg.recentMessageCount;
  const olderMessages = messages.slice(0, splitIndex);
  const recentMessages = messages.slice(splitIndex);

  const summary = await summarizeMessages(olderMessages, summarizer);

  return [
    {
      role: "user",
      content: `[Context Summary — earlier conversation was compacted to save tokens]\n\n${summary}`,
    },
    {
      role: "assistant",
      content: "Understood. I have the context from our earlier conversation. Let me continue from where we left off.",
    },
    ...recentMessages,
  ];
}

/**
 * Tier 2: snipCompact — remove verbose tool results and stale messages.
 */
export function snipCompact(messages: Message[]): Message[] {
  return messages.map((msg) => {
    if (typeof msg.content === "string") return msg;

    const trimmedBlocks = msg.content.map((block) => {
      if (block.type === "tool_result" && block.content.length > 2000) {
        const truncated =
          block.content.slice(0, 800) +
          "\n\n[... truncated ...]\n\n" +
          block.content.slice(-800);
        return { ...block, content: truncated };
      }
      return block;
    });

    return { ...msg, content: trimmedBlocks };
  });
}

/** Internal: ask the LLM to produce a compact summary of an older conversation slice. */
async function summarizeMessages(messages: Message[], summarizer: LLMSummarizer): Promise<string> {
  const conversationText = messages
    .map((msg) => {
      const role = msg.role;
      const content =
        typeof msg.content === "string"
          ? msg.content
          : msg.content
              .map((b) => {
                if (b.type === "text") return b.text;
                if (b.type === "tool_use")
                  return `[Tool: ${b.name}(${JSON.stringify(b.input).slice(0, 200)})]`;
                if (b.type === "tool_result")
                  return `[Result: ${b.content.slice(0, 300)}]`;
                return "";
              })
              .join("\n");
      return `${role}: ${content}`;
    })
    .join("\n\n");

  let summary = "";
  const stream = summarizer.stream({
    systemPrompt:
      "Summarize the following conversation concisely. Preserve key decisions, file paths mentioned, code changes made, and important context. Be thorough but compact. Output only the summary, no preamble.",
    messages: [
      {
        role: "user",
        content: `Summarize this conversation:\n\n${conversationText.slice(0, 50000)}`,
      },
    ],
    tools: [],
  });

  for await (const event of stream) {
    if (event.type === "text_delta" && event.text) {
      summary += event.text;
    }
  }

  return summary || "[Unable to generate summary]";
}
