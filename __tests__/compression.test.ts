/**
 * Tests for src/compression/ — contextCollapse, needsCompaction, snipCompact, autoCompact.
 */

import { describe, expect, test } from "bun:test";
import {
  contextCollapse,
  needsCompaction,
  snipCompact,
  autoCompact,
  DEFAULT_CONFIG,
} from "../src/compression/context.ts";
import { PromptPriority } from "../src/compression/priority.ts";
import type { LLMSummarizer, Message, StreamEvent } from "../src/types/index.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function textMsg(role: "user" | "assistant", text: string): Message {
  return { role, content: text };
}

function toolResultMsg(content: string): Message {
  return {
    role: "user",
    content: [{ type: "tool_result", tool_use_id: "tu_1", content }],
  };
}

/** Stub LLMSummarizer that echoes back a fixed string. */
function makeSummarizer(response: string): LLMSummarizer {
  return {
    async *stream(_req) {
      yield { type: "text_delta", text: response } as StreamEvent;
    },
  };
}

// ---------------------------------------------------------------------------
// contextCollapse
// ---------------------------------------------------------------------------

describe("contextCollapse", () => {
  test("passthrough when ≤5 messages", () => {
    const msgs = [textMsg("user", "a"), textMsg("assistant", "b")];
    expect(contextCollapse(msgs)).toEqual(msgs);
  });

  test("removes short assistant messages (< 10 chars)", () => {
    const msgs: Message[] = [
      textMsg("user", "question one"),
      textMsg("assistant", "ok"), // short — should be removed along with skipNext
      textMsg("user", "question two"),
      textMsg("user", "q3"),
      textMsg("user", "q4"),
      textMsg("user", "q5"),
      textMsg("user", "q6"), // recent tail
    ];
    const result = contextCollapse(msgs);
    // "ok" assistant message and the user message after it are skipped
    const assistantContents = result
      .filter((m) => m.role === "assistant")
      .map((m) => m.content);
    expect(assistantContents).not.toContain("ok");
  });

  test("deduplicates consecutive identical tool results", () => {
    // Need enough messages that both dups land in the "older" slice (not the recent-5 tail)
    const dup = toolResultMsg("same result content here");
    const msgs: Message[] = [
      dup,
      dup, // duplicate — should be removed by dedup logic
      textMsg("user", "q3"),
      textMsg("user", "q4"),
      textMsg("user", "q5"),
      textMsg("user", "q6"),
      textMsg("user", "q7"),
      textMsg("user", "q8"),
      textMsg("user", "q9"),
      textMsg("user", "q10"),
    ];
    const result = contextCollapse(msgs);
    // Older slice (first 5) had two identical tool-results; only 1 should survive
    const olderSlice = result.slice(0, result.length - 5);
    const toolResults = olderSlice.filter(
      (m) => Array.isArray(m.content) && m.content.some((b) => b.type === "tool_result"),
    );
    expect(toolResults.length).toBe(1);
  });

  test("always keeps last 5 messages intact", () => {
    const msgs: Message[] = Array.from({ length: 10 }, (_, i) =>
      textMsg("user", `message ${i}`),
    );
    const result = contextCollapse(msgs);
    const last5 = msgs.slice(-5);
    expect(result.slice(-5)).toEqual(last5);
  });

  test("empty array returns empty", () => {
    expect(contextCollapse([])).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// needsCompaction
// ---------------------------------------------------------------------------

describe("needsCompaction", () => {
  test("returns false when well within limits", () => {
    const msgs = [textMsg("user", "hello")];
    expect(needsCompaction(msgs, 0, { maxContextTokens: 100_000, reserveTokens: 8192 })).toBe(false);
  });

  test("returns true when tokens exceed limit - reserve", () => {
    // 1 char = 0.25 tokens heuristic; need to exceed 100K - 8K = 92K tokens
    // Use actualTokensUsed override to avoid needing huge strings
    expect(needsCompaction([], 90_000, {}, 5_000)).toBe(true); // 90K + 5K = 95K > 91.8K
  });

  test("uses actualTokensUsed when provided", () => {
    const msgs = [textMsg("user", "x")]; // heuristic = 1 token
    // actualTokensUsed=50000 + system=50000 = 100000; limit=100000, reserve=8192 → 100000 > 91808
    expect(needsCompaction(msgs, 50_000, {}, 50_000)).toBe(true);
  });

  test("uses default config when none provided", () => {
    expect(needsCompaction([], DEFAULT_CONFIG.maxContextTokens, {})).toBe(true);
  });

  test("falls back to heuristic when actualTokensUsed absent", () => {
    const msgs = [textMsg("user", "a".repeat(400_000))]; // 100K tokens heuristic
    expect(needsCompaction(msgs, 0, { maxContextTokens: 90_000 })).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// snipCompact
// ---------------------------------------------------------------------------

describe("snipCompact", () => {
  test("passthrough messages with string content", () => {
    const msgs = [textMsg("user", "hello")];
    expect(snipCompact(msgs)).toEqual(msgs);
  });

  test("truncates tool_result blocks longer than 2000 chars", () => {
    const longContent = "x".repeat(3000);
    const msgs: Message[] = [
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "tu_1", content: longContent }],
      },
    ];
    const result = snipCompact(msgs);
    const block = (result[0]!.content as any[])[0];
    expect(block.content.length).toBeLessThan(longContent.length);
    expect(block.content).toContain("[... truncated ...]");
    // Head and tail are preserved
    expect(block.content.startsWith("x".repeat(800))).toBe(true);
    expect(block.content.endsWith("x".repeat(800))).toBe(true);
  });

  test("does not touch tool_result blocks under 2000 chars", () => {
    const shortContent = "y".repeat(500);
    const msgs: Message[] = [
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "tu_1", content: shortContent }],
      },
    ];
    const result = snipCompact(msgs);
    const block = (result[0]!.content as any[])[0];
    expect(block.content).toBe(shortContent);
  });

  test("does not truncate exactly-2000-char content", () => {
    const content = "z".repeat(2000);
    const msgs: Message[] = [
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "tu_1", content }],
      },
    ];
    const result = snipCompact(msgs);
    const block = (result[0]!.content as any[])[0];
    expect(block.content).toBe(content);
  });

  test("empty messages array returns empty", () => {
    expect(snipCompact([])).toEqual([]);
  });

  test("non-tool_result blocks are unchanged", () => {
    const msgs: Message[] = [
      {
        role: "assistant",
        content: [{ type: "text", text: "normal text" }],
      },
    ];
    expect(snipCompact(msgs)).toEqual(msgs);
  });
});

// ---------------------------------------------------------------------------
// autoCompact
// ---------------------------------------------------------------------------

describe("autoCompact", () => {
  test("passthrough when ≤ recentMessageCount messages", async () => {
    const msgs: Message[] = Array.from({ length: 5 }, (_, i) =>
      textMsg("user", `msg ${i}`),
    );
    const summ = makeSummarizer("summary");
    const result = await autoCompact(msgs, summ, { recentMessageCount: 10 });
    expect(result).toEqual(msgs);
  });

  test("prepends summary exchange when messages exceed window", async () => {
    const msgs: Message[] = Array.from({ length: 15 }, (_, i) =>
      textMsg(i % 2 === 0 ? "user" : "assistant", `msg ${i}`),
    );
    const summ = makeSummarizer("here is the summary");
    const result = await autoCompact(msgs, summ, { recentMessageCount: 5 });
    // Result: [user-summary, assistant-ack, ...recent 5]
    expect(result.length).toBe(7);
    expect(result[0]!.role).toBe("user");
    expect(typeof result[0]!.content).toBe("string");
    expect((result[0]!.content as string)).toContain("here is the summary");
    expect(result[1]!.role).toBe("assistant");
    // Last 5 messages are preserved verbatim
    expect(result.slice(2)).toEqual(msgs.slice(-5));
  });

  test("falls back to [Unable to generate summary] when summarizer yields nothing", async () => {
    const msgs: Message[] = Array.from({ length: 15 }, (_, i) =>
      textMsg("user", `msg ${i}`),
    );
    const emptySumm: LLMSummarizer = {
      async *stream(_req) {
        // yields nothing
      },
    };
    const result = await autoCompact(msgs, emptySumm, { recentMessageCount: 5 });
    expect((result[0]!.content as string)).toContain("[Unable to generate summary]");
  });
});

// ---------------------------------------------------------------------------
// PromptPriority
// ---------------------------------------------------------------------------

describe("PromptPriority", () => {
  test("Core is lowest (0)", () => {
    expect(PromptPriority.Core).toBe(0);
  });

  test("Undercover is highest (95)", () => {
    expect(PromptPriority.Undercover).toBe(95);
  });

  test("values are numerically ordered low→high", () => {
    const ordered = [
      PromptPriority.Core,
      PromptPriority.PlanMode,
      PromptPriority.Tools,
      PromptPriority.Permissions,
      PromptPriority.Genome,
      PromptPriority.Knowledge,
      PromptPriority.Git,
      PromptPriority.Memory,
      PromptPriority.Default,
      PromptPriority.BuddyInfluence,
      PromptPriority.ModelPatches,
      PromptPriority.Undercover,
    ];
    for (let i = 1; i < ordered.length; i++) {
      expect(ordered[i]!).toBeGreaterThan(ordered[i - 1]!);
    }
  });
});
