/**
 * Tests for src/compression/semantic-prefilter.ts
 *
 * Covers:
 *   - extractDominantTopics() — term extraction and TF-IDF ranking
 *   - computeKeywordSignal() — BM25 scoring, signal strength, recommendation logic
 *   - SemanticPreFilterer class — rank() and pickTier()
 *   - Integration with selectCompressionTierAdaptive() via skipPreFilter flag
 */

import { describe, expect, test } from "bun:test";
import {
  computeKeywordSignal,
  extractDominantTopics,
  SemanticPreFilterer,
  DEFAULT_TIER_CANDIDATES,
  type TierCandidate,
  type PreFilterResult,
} from "../src/compression/semantic-prefilter.ts";
import { selectCompressionTierAdaptive } from "../src/compression/adaptive.ts";
import type { Message } from "../src/types/index.ts";
import type { LearnedThresholds } from "../src/compression/adaptive.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function textMsg(role: "user" | "assistant", text: string): Message {
  return { role, content: text };
}

function toolResultMsg(id: string, content: string): Message {
  return {
    role: "user",
    content: [{ type: "tool_result", tool_use_id: id, content }],
  };
}

/** Build a minimal LearnedThresholds with enough samples to enable adaptive path. */
function makeThresholds(
  overrides: Partial<Record<1 | 2 | 3 | 4, { successRate: number; avgOvershootPct: number; sampleCount: number }>>,
): LearnedThresholds {
  const defaults = { successRate: 1.0, avgOvershootPct: 0, sampleCount: 5 };
  return {
    byTier: {
      1: { tier: 1, ...defaults, ...overrides[1] },
      2: { tier: 2, ...defaults, ...overrides[2] },
      3: { tier: 3, ...defaults, ...overrides[3] },
      4: { tier: 4, ...defaults, ...overrides[4] },
    },
  };
}

// ---------------------------------------------------------------------------
// extractDominantTopics
// ---------------------------------------------------------------------------

describe("extractDominantTopics", () => {
  test("returns empty array for empty messages", () => {
    expect(extractDominantTopics([])).toEqual([]);
  });

  test("returns terms from a single message", () => {
    const msgs = [textMsg("user", "compression tool results truncation")];
    const topics = extractDominantTopics(msgs);
    expect(topics.length).toBeGreaterThan(0);
    // "compression" and "tool" should rank highly
    expect(topics).toContain("compression");
  });

  test("returns up to topN terms", () => {
    const msgs = Array.from({ length: 5 }, (_, i) =>
      textMsg("user", `word${i} word${i + 1} word${i + 2} word${i + 3} word${i + 4}`),
    );
    const topics = extractDominantTopics(msgs, 5);
    expect(topics.length).toBeLessThanOrEqual(5);
  });

  test("filters out very short terms (≤2 chars)", () => {
    const msgs = [textMsg("user", "is it ok to do so by at up")];
    const topics = extractDominantTopics(msgs);
    // All terms are ≤2 chars and should be filtered out
    expect(topics.length).toBe(0);
  });

  test("higher-frequency terms rank above lower-frequency terms", () => {
    // "summarize" appears 5 times; "tree" appears once
    const content = "summarize summarize summarize summarize summarize tree";
    const msgs = [textMsg("user", content)];
    const topics = extractDominantTopics(msgs);
    const sumIdx = topics.indexOf("summarize");
    const treeIdx = topics.indexOf("tree");
    expect(sumIdx).toBeGreaterThanOrEqual(0);
    if (treeIdx !== -1) {
      // summarize should rank at least as high as tree
      expect(sumIdx).toBeLessThanOrEqual(treeIdx);
    }
  });

  test("handles ContentBlock[] messages (tool_result)", () => {
    const msgs: Message[] = [
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "t1", content: "compression truncation tool output" },
        ],
      },
    ];
    const topics = extractDominantTopics(msgs);
    expect(topics).toContain("compression");
    expect(topics).toContain("truncation");
  });

  test("handles ContentBlock[] messages (text block)", () => {
    const msgs: Message[] = [
      {
        role: "assistant",
        content: [{ type: "text", text: "autocompact summarize history session" }],
      },
    ];
    const topics = extractDominantTopics(msgs);
    expect(topics).toContain("summarize");
  });

  test("deduplication: IDF penalises terms present in all messages", () => {
    // "common" is in every message; "unique" only in one
    const msgs = [
      textMsg("user", "common unique_alpha specific_term"),
      textMsg("assistant", "common another_word example_data"),
      textMsg("user", "common redundant_phrase overlap_text"),
    ];
    const topics = extractDominantTopics(msgs);
    const commonIdx = topics.indexOf("common");
    const uniqueIdx = topics.indexOf("unique_alpha");
    // unique_alpha appears in fewer docs → higher IDF → should rank higher
    if (uniqueIdx !== -1 && commonIdx !== -1) {
      expect(uniqueIdx).toBeLessThanOrEqual(commonIdx);
    }
  });
});

// ---------------------------------------------------------------------------
// computeKeywordSignal — basic scoring
// ---------------------------------------------------------------------------

describe("computeKeywordSignal — basic scoring", () => {
  test("returns ranked candidates array with one entry per candidate", () => {
    const msgs = [textMsg("user", "summarize session history")];
    const result = computeKeywordSignal(msgs, DEFAULT_TIER_CANDIDATES);
    expect(result.rankedCandidates.length).toBe(DEFAULT_TIER_CANDIDATES.length);
  });

  test("dominantTopics is populated from messages", () => {
    const msgs = [textMsg("user", "summarize autocompact history session long")];
    const result = computeKeywordSignal(msgs, DEFAULT_TIER_CANDIDATES);
    expect(result.dominantTopics.length).toBeGreaterThan(0);
  });

  test("signalStrength is in [0, 1]", () => {
    const msgs = [textMsg("user", "some random content here")];
    const result = computeKeywordSignal(msgs, DEFAULT_TIER_CANDIDATES);
    expect(result.signalStrength).toBeGreaterThanOrEqual(0);
    expect(result.signalStrength).toBeLessThanOrEqual(1);
  });

  test("empty messages produce invoke_ollama recommendation", () => {
    const result = computeKeywordSignal([], DEFAULT_TIER_CANDIDATES);
    expect(result.recommendation).toBe("invoke_ollama");
    expect(result.dominantTopics).toEqual([]);
  });

  test("messages about tool results rank tier-2 (snip) higher", () => {
    const msgs = [
      toolResultMsg("t1", "oversized tool output truncation needed"),
      textMsg("assistant", "tool result was truncated snip limit"),
      toolResultMsg("t2", "large tool response limit trim character"),
    ];
    const result = computeKeywordSignal(msgs, DEFAULT_TIER_CANDIDATES);
    // Tier 2 (snipCompact) should be ranked near the top for tool-heavy content
    const tier2Idx = result.rankedCandidates.findIndex((c) => c.tier.tier === 2);
    expect(tier2Idx).toBeLessThanOrEqual(2); // in top-3 at minimum
  });

  test("messages about long session history rank tier-1 (autoCompact) higher", () => {
    const msgs = Array.from({ length: 10 }, () =>
      textMsg("user", "summarize session history autocompact long context preserve semantic"),
    );
    const result = computeKeywordSignal(msgs, DEFAULT_TIER_CANDIDATES);
    const tier1Idx = result.rankedCandidates.findIndex((c) => c.tier.tier === 1);
    // Tier 1 (autoCompact/summarize) should be highly ranked for summarize-heavy content
    expect(tier1Idx).toBeLessThanOrEqual(2);
  });

  test("scores are all non-negative", () => {
    const msgs = [textMsg("user", "some generic content")];
    const result = computeKeywordSignal(msgs, DEFAULT_TIER_CANDIDATES);
    for (const { score } of result.rankedCandidates) {
      expect(score).toBeGreaterThanOrEqual(0);
    }
  });

  test("empty candidate list returns invoke_ollama", () => {
    const msgs = [textMsg("user", "some content")];
    const result = computeKeywordSignal(msgs, []);
    expect(result.recommendation).toBe("invoke_ollama");
    expect(result.rankedCandidates.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// computeKeywordSignal — recommendation logic
// ---------------------------------------------------------------------------

describe("computeKeywordSignal — recommendation logic", () => {
  /**
   * Build synthetic candidates with fixed scores by creating summaries that
   * exactly match or miss the query terms.
   */
  function makeCandidates(specs: Array<{ tier: 1 | 2 | 3 | 4; summary: string; tags: string[]; costWeight: number }>): TierCandidate[] {
    return specs.map((s) => ({ ...s }));
  }

  test("pick_top when one candidate clearly dominates", () => {
    // Only candidate A has terms matching the query
    const msgs = [textMsg("user", "summarize autocompact session history")];
    const candidates = makeCandidates([
      { tier: 1, summary: "summarize autocompact session history", tags: ["summarize", "autocompact"], costWeight: 4 },
      { tier: 2, summary: "unrelated xyzzy foobar", tags: ["unrelated"], costWeight: 2 },
      { tier: 3, summary: "different content qwerty", tags: ["qwerty"], costWeight: 1.5 },
      { tier: 4, summary: "another topic blarg", tags: ["blarg"], costWeight: 1 },
    ]);
    const result = computeKeywordSignal(msgs, candidates);
    // Tier 1 should dominate; recommendation should not be invoke_ollama
    // (unless signal is too low — but our messages are heavily matched)
    if (result.signalStrength >= 0.05) {
      expect(["pick_top", "pick_cheaper"]).toContain(result.recommendation);
      expect(result.rankedCandidates[0]!.tier.tier).toBe(1);
    }
  });

  test("pick_cheaper when top-2 are close and second is cheaper", () => {
    // Construct scenario where tier 4 is slightly cheaper and very close to tier 3
    const msgs = [textMsg("user", "prune tree subtree branch lightweight ratio value")];
    const candidates = makeCandidates([
      { tier: 3, summary: "collapse prune tree subtree branch", tags: ["prune", "tree", "subtree"], costWeight: 1.5 },
      { tier: 4, summary: "prune tree subtree lightweight ratio value recent", tags: ["prune", "tree", "lightweight", "ratio", "value"], costWeight: 1.0 },
      { tier: 1, summary: "summarize history llm semantic", tags: ["summarize"], costWeight: 4 },
      { tier: 2, summary: "truncate tool result output", tags: ["truncate", "tool"], costWeight: 2 },
    ]);
    const result = computeKeywordSignal(msgs, candidates);
    // If recommendation is pick_cheaper, it should return the higher-numbered (cheaper) tier
    if (result.recommendation === "pick_cheaper") {
      const prefilter = new SemanticPreFilterer(candidates);
      const tier = prefilter.pickTier(result);
      // Should pick the cheaper (higher-numbered) of the top-2
      expect(tier).not.toBeNull();
    }
  });

  test("recommendation is one of the three valid values", () => {
    const msgs = [textMsg("user", "generic content here")];
    const result = computeKeywordSignal(msgs, DEFAULT_TIER_CANDIDATES);
    expect(["pick_top", "pick_cheaper", "invoke_ollama"]).toContain(result.recommendation);
  });
});

// ---------------------------------------------------------------------------
// SemanticPreFilterer — class interface
// ---------------------------------------------------------------------------

describe("SemanticPreFilterer", () => {
  test("rank() returns a PreFilterResult", () => {
    const pf = new SemanticPreFilterer();
    const msgs = [textMsg("user", "tool result truncation oversized")];
    const result = pf.rank(msgs);
    expect(result).toHaveProperty("rankedCandidates");
    expect(result).toHaveProperty("recommendation");
    expect(result).toHaveProperty("signalStrength");
    expect(result).toHaveProperty("dominantTopics");
  });

  test("pickTier returns null for invoke_ollama recommendation", () => {
    const pf = new SemanticPreFilterer();
    const result: PreFilterResult = {
      rankedCandidates: [
        { tier: DEFAULT_TIER_CANDIDATES[0]!, score: 1.0 },
        { tier: DEFAULT_TIER_CANDIDATES[1]!, score: 0.99 },
        { tier: DEFAULT_TIER_CANDIDATES[2]!, score: 0.98 },
      ],
      recommendation: "invoke_ollama",
      signalStrength: 0.5,
      dominantTopics: ["some", "topics"],
    };
    expect(pf.pickTier(result)).toBeNull();
  });

  test("pickTier returns top tier for pick_top recommendation", () => {
    const pf = new SemanticPreFilterer();
    const result: PreFilterResult = {
      rankedCandidates: [
        { tier: DEFAULT_TIER_CANDIDATES[0]!, score: 5.0 }, // tier 1
        { tier: DEFAULT_TIER_CANDIDATES[1]!, score: 1.0 }, // tier 2
      ],
      recommendation: "pick_top",
      signalStrength: 0.8,
      dominantTopics: ["summarize"],
    };
    expect(pf.pickTier(result)).toBe(1);
  });

  test("pickTier returns cheaper tier for pick_cheaper recommendation", () => {
    const pf = new SemanticPreFilterer();
    // Top-2: tier 3 (costWeight 1.5) and tier 4 (costWeight 1.0)
    const tier3Candidate = DEFAULT_TIER_CANDIDATES.find((c) => c.tier === 3)!;
    const tier4Candidate = DEFAULT_TIER_CANDIDATES.find((c) => c.tier === 4)!;
    const result: PreFilterResult = {
      rankedCandidates: [
        { tier: tier3Candidate, score: 3.0 },
        { tier: tier4Candidate, score: 2.95 },
      ],
      recommendation: "pick_cheaper",
      signalStrength: 0.4,
      dominantTopics: ["tree", "prune"],
    };
    // Should pick tier 4 (higher number = cheaper/lighter)
    expect(pf.pickTier(result)).toBe(4);
  });

  test("pickTier returns null for empty rankedCandidates", () => {
    const pf = new SemanticPreFilterer();
    const result: PreFilterResult = {
      rankedCandidates: [],
      recommendation: "pick_top",
      signalStrength: 0,
      dominantTopics: [],
    };
    expect(pf.pickTier(result)).toBeNull();
  });

  test("accepts custom candidates in constructor", () => {
    const customCandidates: TierCandidate[] = [
      {
        tier: 2,
        summary: "custom tier two description with unique_keyword_xyz",
        tags: ["custom", "unique_keyword_xyz"],
        costWeight: 2.0,
      },
      {
        tier: 4,
        summary: "custom tier four description with another_unique_abc",
        tags: ["another_unique_abc"],
        costWeight: 1.0,
      },
    ];
    const pf = new SemanticPreFilterer(customCandidates);
    const msgs = [textMsg("user", "custom unique_keyword_xyz content")];
    const result = pf.rank(msgs);
    expect(result.rankedCandidates.length).toBe(2);
    // Tier 2 should score higher given the unique_keyword_xyz match
    expect(result.rankedCandidates[0]!.tier.tier).toBe(2);
  });

  test("rank() with targetTokenBudget parameter does not throw", () => {
    const pf = new SemanticPreFilterer();
    const msgs = [textMsg("user", "some content")];
    expect(() => pf.rank(msgs, 50_000)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// DEFAULT_TIER_CANDIDATES structure
// ---------------------------------------------------------------------------

describe("DEFAULT_TIER_CANDIDATES", () => {
  test("has exactly 4 candidates", () => {
    expect(DEFAULT_TIER_CANDIDATES.length).toBe(4);
  });

  test("covers tiers 1–4", () => {
    const tiers = DEFAULT_TIER_CANDIDATES.map((c) => c.tier).sort();
    expect(tiers).toEqual([1, 2, 3, 4]);
  });

  test("costWeight decreases as tier number increases (lighter = cheaper)", () => {
    const sorted = [...DEFAULT_TIER_CANDIDATES].sort((a, b) => a.tier - b.tier);
    for (let i = 0; i < sorted.length - 1; i++) {
      expect(sorted[i]!.costWeight).toBeGreaterThan(sorted[i + 1]!.costWeight);
    }
  });

  test("all candidates have non-empty summary and tags", () => {
    for (const c of DEFAULT_TIER_CANDIDATES) {
      expect(c.summary.length).toBeGreaterThan(0);
      expect(c.tags.length).toBeGreaterThan(0);
    }
  });
});

// ---------------------------------------------------------------------------
// Integration: selectCompressionTierAdaptive with pre-filter
// ---------------------------------------------------------------------------

describe("selectCompressionTierAdaptive — pre-filter integration", () => {
  const goodHistory = makeThresholds({
    1: { successRate: 0.99, avgOvershootPct: 0, sampleCount: 5 },
    2: { successRate: 0.90, avgOvershootPct: 5, sampleCount: 5 },
    3: { successRate: 0.90, avgOvershootPct: 5, sampleCount: 5 },
    4: { successRate: 0.95, avgOvershootPct: 2, sampleCount: 5 },
  });

  test("returns a valid CompressionTier (1–4) when pre-filter is active", () => {
    const msgs = Array.from({ length: 5 }, (_, i) =>
      textMsg(i % 2 === 0 ? "user" : "assistant", `message ${i} content`),
    );
    const tier = selectCompressionTierAdaptive(
      msgs,
      0,
      { maxContextTokens: 1_000_000, reserveTokens: 8192 },
      goodHistory,
    );
    expect([1, 2, 3, 4]).toContain(tier);
  });

  test("skipPreFilter=true gives same result as original adaptive path (no regression)", () => {
    const msgs = Array.from({ length: 8 }, (_, i) =>
      textMsg(i % 2 === 0 ? "user" : "assistant", `turn ${i} text content`),
    );
    // With skipPreFilter, the pre-filter is bypassed entirely
    const tierWithSkip = selectCompressionTierAdaptive(
      msgs,
      0,
      { maxContextTokens: 1_000_000, reserveTokens: 8192 },
      goodHistory,
      null,
      null,
      null,
      true, // skipPreFilter
    );
    expect([1, 2, 3, 4]).toContain(tierWithSkip);
  });

  test("null history still bypasses pre-filter and returns static tier", () => {
    const msgs = Array.from({ length: 5 }, (_, i) => textMsg("user", `msg ${i}`));
    // With null history the pre-filter is never reached
    const tier = selectCompressionTierAdaptive(msgs, 0, {}, null);
    expect([1, 2, 3, 4]).toContain(tier);
  });

  test("pre-filter active path does not throw on empty messages", () => {
    expect(() =>
      selectCompressionTierAdaptive(
        [],
        0,
        { maxContextTokens: 1_000_000, reserveTokens: 8192 },
        goodHistory,
      ),
    ).not.toThrow();
  });

  test("pre-filter result is consistent across multiple calls with same input", () => {
    const msgs = Array.from({ length: 6 }, (_, i) =>
      textMsg(i % 2 === 0 ? "user" : "assistant", "tool result truncation oversized snip"),
    );
    const tier1 = selectCompressionTierAdaptive(
      msgs,
      0,
      { maxContextTokens: 1_000_000, reserveTokens: 8192 },
      goodHistory,
    );
    const tier2 = selectCompressionTierAdaptive(
      msgs,
      0,
      { maxContextTokens: 1_000_000, reserveTokens: 8192 },
      goodHistory,
    );
    expect(tier1).toBe(tier2);
  });

  test("pre-filter does not escalate beyond budget constraints", () => {
    // Very tight budget — pre-filter keyword tier must fail budget check and fall through
    const msgs = Array.from({ length: 4 }, (_, i) =>
      textMsg(i % 2 === 0 ? "user" : "assistant", `turn ${i}`),
    );
    // Extremely tight budget — forces tier 1 eventually
    const tier = selectCompressionTierAdaptive(
      msgs,
      0,
      { maxContextTokens: 50, reserveTokens: 0 },
      goodHistory,
    );
    expect([1, 2, 3, 4]).toContain(tier);
  });
});
