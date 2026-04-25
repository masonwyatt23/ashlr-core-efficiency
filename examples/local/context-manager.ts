/**
 * context-manager.ts — demonstrates @ashlr/core-efficiency/local
 *
 * Subpath: @ashlr/core-efficiency/local
 *
 * Shows how LocalContextManager trims a growing conversation to fit inside
 * a small-context local model's window (e.g. a 4K-token Ollama model).
 *
 * Run:
 *   bun run examples/local/context-manager.ts
 *
 * Windows note: no shell-outs; runs identically on macOS / Linux / Windows.
 */
import { LocalContextManager } from "../../src/local/index.ts";
import type { Message } from "../../src/types/index.ts";

// Simulate a conversation that has grown beyond a 512-token window.
const allMessages: Message[] = Array.from({ length: 20 }, (_, i) => ({
  role: i % 2 === 0 ? "user" : "assistant",
  content: `Turn ${i + 1}: ${"word ".repeat(30)}`,   // ~30 tokens each
}));

const mgr = new LocalContextManager({ contextWindow: 512 });
const fitted = mgr.fit(allMessages);

console.log(
  `LocalContextManager: ${allMessages.length} messages → ${fitted.length} ` +
    `(window=512 tokens, oldest turns dropped)`,
);
