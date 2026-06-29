/**
 * Integration test: compression tier ladder ordering + tier selection hardening.
 *
 * Asserts:
 *   1. Higher tiers compress more than lower ones (monotonic savings).
 *   2. Each tier's output is semantically valid — no broken truncation, sentinel present.
 *   3. selectCompressionTier picks the correct tier for a given token budget.
 *   4. Savings across the full ladder are strictly non-increasing (or equal when a
 *      tier genuinely cannot help a given input).
 *
 * Uses realistic inputs: a mix of large tool results, short assistant messages,
 * duplicate tool results, and normal conversation turns — the kind of context a
 * live fleet session accumulates after several dozen tool calls.
 */

import { describe, expect, test } from "bun:test";
import {
  autoCompact,
  contextCollapse,
  needsCompaction,
  selectCompressionTier,
  snipCompact,
  DEFAULT_CONFIG,
} from "../src/compression/context.ts";
import { estimateTokensFromMessages } from "../src/tokens/index.ts";
import type { LLMSummarizer, Message, StreamEvent } from "../src/types/index.ts";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** Large tool result (~6 KB) — simulates a file read or grep output. */
function largeTool(id: string, content: string): Message {
  return {
    role: "user",
    content: [{ type: "tool_result", tool_use_id: id, content }],
  };
}

function textMsg(role: "user" | "assistant", text: string): Message {
  return { role, content: text };
}

function makeSummarizer(response: string): LLMSummarizer {
  return {
    async *stream(_req): AsyncGenerator<StreamEvent> {
      yield { type: "text_delta", text: response };
    },
  };
}

/**
 * Build a realistic conversation that stresses every tier:
 *   - 8 large tool results (6 000 chars each) → snipCompact reduces these
 *   - 4 duplicate tool results → contextCollapse deduplicates
 *   - 6 short assistant messages (< 10 chars) → contextCollapse removes
 *   - 10 normal conversation turns → kept by all tiers
 */
function buildRealisticConversation(): Message[] {
  const msgs: Message[] = [];

  // Short assistant messages that contextCollapse should strip
  for (let i = 0; i < 6; i++) {
    msgs.push(textMsg("user", `question ${i}`));
    msgs.push(textMsg("assistant", "ok")); // < 10 chars → collapse target
  }

  // Duplicate tool results (same hash) → contextCollapse strips second+
  const dupContent = "duplicate genome result line\n".repeat(20);
  msgs.push(largeTool("dup_1", dupContent));
  msgs.push(largeTool("dup_1", dupContent)); // duplicate
  msgs.push(largeTool("dup_1", dupContent)); // duplicate

  // Large unique tool results → snipCompact truncates each from 6000 → ~1617 chars
  for (let i = 0; i < 8; i++) {
    msgs.push(
      largeTool(
        `tr_${i}`,
        `FILE_READ_RESULT_${i}\n` + `line ${i}: ${"abcdefghij".repeat(590)}\n`,
      ),
    );
  }

  // Normal conversation turns at the end (kept by all tiers)
  for (let i = 0; i < 10; i++) {
    msgs.push(textMsg(i % 2 === 0 ? "user" : "assistant", `normal turn ${i} with some real content here`));
  }

  return msgs;
}

// ---------------------------------------------------------------------------
// Helper: token counts after each tier
// ---------------------------------------------------------------------------

interface TierTokenCounts {
  original: number;
  afterT3: number; // contextCollapse
  afterT2: number; // snipCompact
  afterT2T3: number; // snipCompact then contextCollapse (combined)
}

function measureTiers(msgs: Message[]): TierTokenCounts {
  const original = estimateTokensFromMessages(msgs);
  const afterT3 = estimateTokensFromMessages(contextCollapse(msgs));
  const afterT2 = estimateTokensFromMessages(snipCompact(msgs));
  // T2 then T3 represents applying both structural tiers in sequence
  const afterT2T3 = estimateTokensFromMessages(contextCollapse(snipCompact(msgs)));
  return { original, afterT3, afterT2, afterT2T3 };
}

// ---------------------------------------------------------------------------
// 1. Monotonic savings: higher tiers remove more tokens than lower ones
// ---------------------------------------------------------------------------

describe("tier ordering — monotonic savings", () => {
  test("T3 (contextCollapse) reduces tokens vs original on a realistic conversation", () => {
    const msgs = buildRealisticConversation();
    const { original, afterT3 } = measureTiers(msgs);
    expect(afterT3).toBeLessThan(original);
  });

  test("T2 (snipCompact) reduces tokens vs original on a realistic conversation", () => {
    const msgs = buildRealisticConversation();
    const { original, afterT2 } = measureTiers(msgs);
    expect(afterT2).toBeLessThan(original);
  });

  test("T2 compresses more than T3 alone on large-tool-result-heavy input", () => {
    // Input has 8 large tool results — snipCompact should outperform contextCollapse
    const msgs = buildRealisticConversation();
    const { afterT3, afterT2 } = measureTiers(msgs);
    // snipCompact targets the large tool results that contextCollapse ignores
    expect(afterT2).toBeLessThan(afterT3);
  });

  test("combined T2+T3 compresses at least as much as T2 alone", () => {
    const msgs = buildRealisticConversation();
    const { afterT2, afterT2T3 } = measureTiers(msgs);
    expect(afterT2T3).toBeLessThanOrEqual(afterT2);
  });

  test("combined T2+T3 compresses more than T3 alone on mixed input", () => {
    const msgs = buildRealisticConversation();
    const { afterT3, afterT2T3 } = measureTiers(msgs);
    expect(afterT2T3).toBeLessThan(afterT3);
  });

  test("savings are non-negative across all tiers (no tier makes things worse)", () => {
    const msgs = buildRealisticConversation();
    const { original, afterT3, afterT2, afterT2T3 } = measureTiers(msgs);
    expect(afterT3).toBeLessThanOrEqual(original);
    expect(afterT2).toBeLessThanOrEqual(original);
    expect(afterT2T3).toBeLessThanOrEqual(original);
  });
});

// ---------------------------------------------------------------------------
// 2. Semantic validity
// ---------------------------------------------------------------------------

describe("tier outputs — semantic validity", () => {
  test("snipCompact: truncated block contains sentinel and non-empty head+tail", () => {
    const longContent = "HEADER_LINE\n" + "body ".repeat(1000) + "\nFOOTER_LINE";
    const msgs: Message[] = [
      { role: "user", content: [{ type: "tool_result", tool_use_id: "x", content: longContent }] },
    ];
    const result = snipCompact(msgs);
    const block = (result[0]!.content as any[])[0];

    expect(block.content).toContain("[... truncated ...]");
    // Head is preserved
    expect(block.content.startsWith("HEADER_LINE")).toBe(true);
    // Tail is preserved (last 800 chars of original ends with FOOTER_LINE)
    expect(block.content).toContain("FOOTER_LINE");
    // Result is shorter than original
    expect(block.content.length).toBeLessThan(longContent.length);
  });

  test("snipCompact: multiple tool results in one message — each truncated independently", () => {
    const msgs: Message[] = [
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "a", content: "A".repeat(3000) },
          { type: "tool_result", tool_use_id: "b", content: "B".repeat(3000) },
          { type: "tool_result", tool_use_id: "c", content: "C".repeat(100) }, // short — untouched
        ],
      },
    ];
    const result = snipCompact(msgs);
    const blocks = result[0]!.content as any[];
    expect(blocks[0].content).toContain("[... truncated ...]");
    expect(blocks[1].content).toContain("[... truncated ...]");
    expect(blocks[2].content).toBe("C".repeat(100)); // short block unchanged
  });

  test("contextCollapse: output is a valid Message[] (no undefined entries)", () => {
    const msgs = buildRealisticConversation();
    const result = contextCollapse(msgs);
    expect(result.length).toBeGreaterThan(0);
    for (const msg of result) {
      expect(msg).toBeDefined();
      expect(["user", "assistant", "tool"]).toContain(msg.role);
      expect(msg.content).toBeDefined();
    }
  });

  test("contextCollapse: recent 5 messages are always structurally intact", () => {
    const msgs = buildRealisticConversation();
    const result = contextCollapse(msgs);
    const original_last5 = msgs.slice(-5);
    const result_last5 = result.slice(-5);
    expect(result_last5).toEqual(original_last5);
  });

  test("snipCompact: preserves message count (no messages dropped)", () => {
    const msgs = buildRealisticConversation();
    expect(snipCompact(msgs).length).toBe(msgs.length);
  });

  test("snipCompact: non-tool-result blocks are byte-identical after pass", () => {
    const textBlock = { type: "text" as const, text: "important analysis here" };
    const msgs: Message[] = [{ role: "assistant", content: [textBlock] }];
    const result = snipCompact(msgs);
    expect((result[0]!.content as any[])[0]).toEqual(textBlock);
  });
});

// ---------------------------------------------------------------------------
// 3. Tier selection — selectCompressionTier picks right tier for budget
// ---------------------------------------------------------------------------

describe("selectCompressionTier", () => {
  test("returns 4 when treeCompact alone fits within budget (cheapest tier)", () => {
    // Small conversation — treeCompact easily fits
    const msgs: Message[] = Array.from({ length: 8 }, (_, i) =>
      textMsg(i % 2 === 0 ? "user" : "assistant", `turn ${i} text`),
    );
    // Very large budget — any tier fits; should pick cheapest (4)
    const tier = selectCompressionTier(msgs, 0, {
      maxContextTokens: 1_000_000,
      reserveTokens: 8192,
    });
    expect(tier).toBe(4);
  });

  test("returns 2 when snipCompact fits but contextCollapse alone does not", () => {
    // Build a message set where contextCollapse doesn't help much (no short msgs or dups)
    // but snipCompact cuts significantly (large tool results)
    const msgs: Message[] = [];
    for (let i = 0; i < 10; i++) {
      msgs.push(largeTool(`t${i}`, "X".repeat(5000)));
    }
    // Make budget: after snipCompact, 10 messages × (800+sentinel+800) ≈ 10×1640 chars ≈ 4100 tokens
    // After contextCollapse: no short/dup messages → barely changes → same ~12500 tokens
    // Set budget just above snipped size but below original
    const snippedTokens = estimateTokensFromMessages(snipCompact(msgs));
    const tier = selectCompressionTier(msgs, 0, {
      maxContextTokens: snippedTokens + 1000, // snipCompact fits
      reserveTokens: 0,
    });
    expect(tier).toBe(2);
  });

  test("returns 1 when neither T3 nor T2 fits within budget", () => {
    // Huge messages, tiny budget
    const msgs: Message[] = [];
    for (let i = 0; i < 20; i++) {
      msgs.push(largeTool(`t${i}`, "X".repeat(5000)));
    }
    // Budget so tight even snipCompact won't fit
    const tier = selectCompressionTier(msgs, 0, {
      maxContextTokens: 100, // impossibly small
      reserveTokens: 0,
    });
    expect(tier).toBe(1);
  });

  test("tier 1 is returned when system tokens alone consume the budget", () => {
    const msgs = [textMsg("user", "hello")];
    const tier = selectCompressionTier(msgs, 200_000, {
      maxContextTokens: 100_000,
      reserveTokens: 0,
    });
    expect(tier).toBe(1);
  });

  test("uses DEFAULT_CONFIG when no config provided", () => {
    const msgs: Message[] = Array.from({ length: 3 }, (_, i) =>
      textMsg("user", `msg ${i}`),
    );
    // With default 100K limit and tiny messages, should fit in tier 4 (cheapest)
    const tier = selectCompressionTier(msgs, 0);
    expect(tier).toBe(4);
  });

  test("tier selection is consistent with needsCompaction", () => {
    // If needsCompaction says no compaction needed, tier 4 should be sufficient (cheapest)
    const msgs = [textMsg("user", "small message")];
    const systemTokens = 100;
    const config = { maxContextTokens: 100_000, reserveTokens: 8192 };
    expect(needsCompaction(msgs, systemTokens, config)).toBe(false);
    // Tier 4 (treeCompact) is the new lightest/cheapest tier
    expect(selectCompressionTier(msgs, systemTokens, config)).toBe(4);
  });
});

// ---------------------------------------------------------------------------
// 4. Full ladder feed — input through T3 → T2 → T1 in sequence
// ---------------------------------------------------------------------------

describe("full tier ladder — end-to-end", () => {
  test("each tier in the ladder produces a smaller or equal token count than the previous", async () => {
    const msgs = buildRealisticConversation();
    const originalTokens = estimateTokensFromMessages(msgs);

    // T3
    const afterT3 = contextCollapse(msgs);
    const t3Tokens = estimateTokensFromMessages(afterT3);
    expect(t3Tokens).toBeLessThanOrEqual(originalTokens);

    // T2 applied to T3 output
    const afterT2 = snipCompact(afterT3);
    const t2Tokens = estimateTokensFromMessages(afterT2);
    expect(t2Tokens).toBeLessThanOrEqual(t3Tokens);

    // T1 applied to T2+T3 output
    const summ = makeSummarizer("Fleet session summary: tool calls executed, files modified, context preserved.");
    const afterT1 = await autoCompact(afterT2, summ, { recentMessageCount: 5 });
    const t1Tokens = estimateTokensFromMessages(afterT1);
    expect(t1Tokens).toBeLessThanOrEqual(t2Tokens);
  });

  test("T1 output is semantically valid — summary exchange prepended, recent messages intact", async () => {
    const msgs = buildRealisticConversation();
    const recentMessageCount = 5;
    const summaryText = "Concise summary of prior work.";
    const summ = makeSummarizer(summaryText);

    const result = await autoCompact(msgs, summ, { recentMessageCount });

    // First two messages are the synthetic summary exchange
    expect(result[0]!.role).toBe("user");
    expect(result[0]!.content).toContain(summaryText);
    expect(result[1]!.role).toBe("assistant");

    // Last N messages are verbatim
    const originalTail = msgs.slice(-recentMessageCount);
    expect(result.slice(-recentMessageCount)).toEqual(originalTail);
  });

  test("T1 → T2 cascade: applying snipCompact after autoCompact still valid", async () => {
    const msgs = buildRealisticConversation();
    const summ = makeSummarizer("summary");
    const afterT1 = await autoCompact(msgs, summ, { recentMessageCount: 5 });
    // Applying snipCompact on top of autoCompact output should not throw
    const afterT1T2 = snipCompact(afterT1);
    expect(afterT1T2.length).toBeGreaterThan(0);
    expect(estimateTokensFromMessages(afterT1T2)).toBeLessThanOrEqual(
      estimateTokensFromMessages(afterT1),
    );
  });

  test("savings across tier ladder are strictly non-increasing", async () => {
    const msgs = buildRealisticConversation();
    const original = estimateTokensFromMessages(msgs);
    const t3 = estimateTokensFromMessages(contextCollapse(msgs));
    const t2 = estimateTokensFromMessages(snipCompact(contextCollapse(msgs)));
    const summ = makeSummarizer("compact summary of all prior work in this fleet session");
    const t1 = estimateTokensFromMessages(
      await autoCompact(snipCompact(contextCollapse(msgs)), summ, { recentMessageCount: 3 }),
    );

    // Each successive tier should not increase the count
    expect(t3).toBeLessThanOrEqual(original);
    expect(t2).toBeLessThanOrEqual(t3);
    expect(t1).toBeLessThanOrEqual(t2);
  });
});
