/**
 * snip-compact.ts — demonstrates @ashlr/core-efficiency/compression
 *
 * Subpath: @ashlr/core-efficiency/compression
 *
 * Shows how snipCompact truncates large tool results to a head+tail window,
 * contextCollapse drops short/duplicate turns, and autoCompact calls an LLM
 * summarizer when the conversation approaches a token budget.
 *
 * Run (no API key required for snipCompact / contextCollapse):
 *   bun run examples/compression/snip-compact.ts
 *
 * Windows note: no shell-outs; runs identically on macOS / Linux / Windows.
 */
import {
  snipCompact,
  contextCollapse,
  PromptPriority,
} from "../../src/compression/index.ts";
import type { Message } from "../../src/types/index.ts";

// Simulate a conversation with a large tool result.
const bigToolOutput = "x".repeat(8000);

const messages: Message[] = [
  { role: "user", content: "Read package.json" },
  {
    role: "assistant",
    content: [
      {
        type: "tool_result",
        tool_use_id: "tool_1",
        content: bigToolOutput,
      },
    ],
  },
  { role: "user", content: "ok" },           // short — candidate for collapse
  { role: "user", content: "ok" },           // duplicate
  { role: "user", content: "Now summarize the file you just read." },
];

// 1. Truncate tool results that exceed 2 KB.
const snipped = snipCompact(messages, { maxBytes: 2048 });
const toolResult = snipped[1];
const resultContent =
  Array.isArray(toolResult?.content) && toolResult.content[0]?.type === "tool_result"
    ? (toolResult.content[0] as { content: string }).content
    : "";
console.log(`snipCompact: tool result reduced to ${resultContent.length} chars (was ${bigToolOutput.length})`);

// 2. Drop short/duplicate turns.
const collapsed = contextCollapse(snipped);
console.log(`contextCollapse: ${messages.length} messages → ${collapsed.length}`);

// 3. Show PromptPriority values (numeric stability guarantee).
console.log(`PromptPriority.Core=${PromptPriority.Core} PromptPriority.High=${PromptPriority.High}`);
