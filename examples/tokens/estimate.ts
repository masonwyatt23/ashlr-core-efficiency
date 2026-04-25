/**
 * estimate.ts — demonstrates @ashlr/core-efficiency/tokens
 *
 * Subpath: @ashlr/core-efficiency/tokens
 *
 * Estimates token counts for a plain string and for a multi-turn message
 * array that includes tool_use and tool_result blocks.
 *
 * Run:
 *   bun run examples/tokens/estimate.ts
 *
 * Windows note: no shell-outs; runs identically on macOS / Linux / Windows.
 */
import {
  estimateTokensFromString,
  estimateTokensFromMessages,
} from "../../src/tokens/index.ts";
import type { Message } from "../../src/types/index.ts";

const phrase = "The quick brown fox jumps over the lazy dog.";
console.log(`"${phrase}"\n  -> ~${estimateTokensFromString(phrase)} tokens`);

const messages: Message[] = [
  { role: "user", content: "Read package.json and summarize it." },
  {
    role: "assistant",
    content: [
      { type: "tool_use", id: "tool_1", name: "ashlr__read", input: { path: "package.json" } },
    ],
  },
  {
    role: "user",
    content: [
      {
        type: "tool_result",
        tool_use_id: "tool_1",
        content: '{"name":"@ashlr/core-efficiency","version":"0.2.0"}',
      },
    ],
  },
  { role: "assistant", content: "The package is @ashlr/core-efficiency at version 0.2.0." },
];

const total = estimateTokensFromMessages(messages);
console.log(`\n4-turn conversation (with tool_use/tool_result): ~${total} tokens`);
