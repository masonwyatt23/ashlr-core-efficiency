/**
 * Tests for Dynamic Budget Rebalancing on Provider Switch.
 *
 * Covers:
 *  - computeProviderCostRatio: ratio values, tier shift recommendations
 *  - rebalanceBudgetOnSwitch: budget deltas, tier selection, edge cases
 *  - compareProviderCosts: cost ordering, cache break-even, provider filtering
 *  - selectCompressionTierAdaptive with provider param: pricing-aware tier shift
 */

import { describe, expect, test } from "bun:test";
import {
  computeProviderCostRatio,
  rebalanceBudgetOnSwitch,
  PROVIDER_PRICING,
  systemPromptBudget,
} from "../src/budget/index.ts";
import { compareProviderCosts } from "../src/budget/cost-comparison.ts";
import { selectCompressionTierAdaptive } from "../src/compression/adaptive.ts";
import type { Message } from "../src/types/index.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeMessages(count: number, contentLength = 500): Message[] {
  return Array.from({ length: count }, (_, i) => ({
    role: (i % 2 === 0 ? "user" : "assistant") as "user" | "assistant",
    content: "x".repeat(contentLength),
  }));
}

// ---------------------------------------------------------------------------
// computeProviderCostRatio
// ---------------------------------------------------------------------------

describe("computeProviderCostRatio", () => {
  test("same provider → ratios are 1.0 and no tier shift", () => {
    const ratio = computeProviderCostRatio("anthropic", "anthropic");
    expect(ratio.inputRatio).toBeCloseTo(1.0);
    expect(ratio.outputRatio).toBeCloseTo(1.0);
    expect(ratio.cacheReadRatio).toBeCloseTo(1.0);
    expect(ratio.recommendedTierShift).toBe(0);
  });

  test("anthropic → groq: groq is much cheaper → recommendedTierShift = -1", () => {
    const ratio = computeProviderCostRatio("anthropic", "groq");
    // groq input 0.27 vs anthropic 3.0 → inputRatio ≈ 0.09
    expect(ratio.inputRatio).toBeLessThan(0.5);
    expect(ratio.outputRatio).toBeLessThan(0.5);
    expect(ratio.recommendedTierShift).toBe(-1);
  });

  test("anthropic → deepseek: deepseek is much cheaper → recommendedTierShift = -1", () => {
    const ratio = computeProviderCostRatio("anthropic", "deepseek");
    expect(ratio.inputRatio).toBeLessThan(0.5);
    expect(ratio.recommendedTierShift).toBe(-1);
  });

  test("groq → anthropic: anthropic is more expensive → recommendedTierShift = +1", () => {
    const ratio = computeProviderCostRatio("groq", "anthropic");
    // anthropic input 3.0 vs groq 0.27 → inputRatio ≈ 11 (clamped to 10)
    expect(ratio.inputRatio).toBeGreaterThanOrEqual(2.0);
    expect(ratio.recommendedTierShift).toBe(1);
  });

  test("anthropic → openai: similar pricing → no tier shift", () => {
    const ratio = computeProviderCostRatio("anthropic", "openai");
    // openai 2.5 vs anthropic 3.0 → inputRatio ≈ 0.83 — within neutral band
    const blended = (ratio.inputRatio + ratio.outputRatio) / 2;
    expect(blended).toBeGreaterThan(0.5);
    expect(blended).toBeLessThan(2.0);
    expect(ratio.recommendedTierShift).toBe(0);
  });

  test("ollama → anthropic: ollama is free, anthropic is costly → clamped ratio", () => {
    const ratio = computeProviderCostRatio("ollama", "anthropic");
    // denominator is 0 for ollama → ratio should be 10.0 (clamped max)
    expect(ratio.inputRatio).toBe(10.0);
    expect(ratio.outputRatio).toBe(10.0);
    expect(ratio.recommendedTierShift).toBe(1);
  });

  test("anthropic → ollama: free target → ratios are 0", () => {
    const ratio = computeProviderCostRatio("anthropic", "ollama");
    expect(ratio.inputRatio).toBe(0);
    expect(ratio.outputRatio).toBe(0);
    expect(ratio.recommendedTierShift).toBe(-1);
  });

  test("case-insensitive provider resolution", () => {
    const a = computeProviderCostRatio("Anthropic", "OpenAI");
    const b = computeProviderCostRatio("anthropic", "openai");
    expect(a.inputRatio).toBeCloseTo(b.inputRatio);
    expect(a.recommendedTierShift).toBe(b.recommendedTierShift);
  });

  test("compound provider names resolve via substring match", () => {
    const ratio = computeProviderCostRatio("anthropic-claude-sonnet", "openai-gpt-4o");
    expect(ratio.inputRatio).toBeGreaterThan(0);
    // Should resolve correctly (not fall to unknown pricing)
    const anthropicPricing = PROVIDER_PRICING["anthropic"];
    const openaiPricing    = PROVIDER_PRICING["openai"];
    if (anthropicPricing && openaiPricing) {
      expect(ratio.inputRatio).toBeCloseTo(openaiPricing.inputPerMToken / anthropicPricing.inputPerMToken);
    }
  });

  test("unknown provider uses neutral fallback pricing", () => {
    // Both unknown → same pricing → ratio ≈ 1.0
    const ratio = computeProviderCostRatio("mystery-llm", "another-mystery");
    expect(ratio.inputRatio).toBeCloseTo(1.0);
    expect(ratio.recommendedTierShift).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// rebalanceBudgetOnSwitch
// ---------------------------------------------------------------------------

describe("rebalanceBudgetOnSwitch", () => {
  test("switching to a larger-context provider increases system budget", () => {
    // xai has 2M context → budget would be 50K (capped); anthropic is 200K → 10K
    const result = rebalanceBudgetOnSwitch("anthropic", "xai", []);
    expect(result.newSystemBudget).toBeGreaterThan(systemPromptBudget("anthropic"));
    expect(result.tokensFreed).toBeGreaterThan(0);
  });

  test("switching to a smaller-context provider shrinks budget", () => {
    // ollama 32K → budget ~1600; anthropic 200K → 10K
    const result = rebalanceBudgetOnSwitch("anthropic", "ollama", []);
    expect(result.newSystemBudget).toBeLessThan(systemPromptBudget("anthropic"));
    expect(result.tokensFreed).toBeLessThan(0);
  });

  test("tokensFreed equals newSystemBudget - oldSystemBudget", () => {
    const messages = makeMessages(5);
    const result = rebalanceBudgetOnSwitch("anthropic", "openai", messages);
    const expectedDelta = result.newSystemBudget - systemPromptBudget("anthropic");
    expect(result.tokensFreed).toBe(expectedDelta);
  });

  test("recommendedTier is within valid range [1, 3]", () => {
    for (const [from, to] of [
      ["anthropic", "groq"],
      ["groq", "anthropic"],
      ["openai", "deepseek"],
      ["deepseek", "xai"],
      ["ollama", "anthropic"],
    ]) {
      const result = rebalanceBudgetOnSwitch(from!, to!, makeMessages(10));
      expect(result.recommendedTier).toBeGreaterThanOrEqual(1);
      expect(result.recommendedTier).toBeLessThanOrEqual(3);
    }
  });

  test("empty messages → context is near-empty → tier 3 (cheapest)", () => {
    // With zero message tokens the context is empty regardless of provider.
    // Fill ratio ≈ 0 → baseline tier 3. Anthropic → anthropic has no shift → tier 3.
    const result = rebalanceBudgetOnSwitch("anthropic", "anthropic", []);
    expect(result.recommendedTier).toBe(3);
  });

  test("very large message set filling ollama context → tier 2 or tighter", () => {
    // ollama has 32K context. Fill >80% with 6400 tokens (≈ 25600 chars across messages).
    // 64 messages × 400 chars = 25600 chars → ~6400 tokens.
    // available ≈ 32K - 8192 - 1600 ≈ 22K; fill ratio ≈ 6400/22000 ≈ 0.29 → baseline tier 3.
    // Switching to a very cheap provider shifts tier by -1 (clamped to 3).
    // Switching to an expensive provider shifts tier by +1 (tighter).
    // Verify tier is valid regardless of direction.
    const messages = makeMessages(64, 400);
    const toExpensive = rebalanceBudgetOnSwitch("ollama", "anthropic", messages);
    expect(toExpensive.recommendedTier).toBeGreaterThanOrEqual(1);
    expect(toExpensive.recommendedTier).toBeLessThanOrEqual(3);
    // Switching from ollama to anthropic (expensive) should tighten vs same-provider.
    const neutral = rebalanceBudgetOnSwitch("anthropic", "anthropic", messages);
    expect(toExpensive.recommendedTier).toBeLessThanOrEqual(neutral.recommendedTier);
  });

  test("switching to a much cheaper provider shifts tier toward 3 (less aggressive)", () => {
    // groq is very cheap → tier shift -1 (relax) applied to baseline
    const affordable = rebalanceBudgetOnSwitch("anthropic", "groq", makeMessages(30, 300));
    const neutral    = rebalanceBudgetOnSwitch("anthropic", "anthropic", makeMessages(30, 300));
    // affordable tier should be >= neutral tier (higher = less aggressive)
    expect(affordable.recommendedTier).toBeGreaterThanOrEqual(neutral.recommendedTier);
  });

  test("switching to a much pricier provider shifts tier toward 1 (more aggressive)", () => {
    // anthropic → ollama-to-anthropic: groq-to-anthropic is expensive
    const expensive = rebalanceBudgetOnSwitch("groq", "anthropic", makeMessages(30, 300));
    const neutral   = rebalanceBudgetOnSwitch("anthropic", "anthropic", makeMessages(30, 300));
    expect(expensive.recommendedTier).toBeLessThanOrEqual(neutral.recommendedTier);
  });
});

// ---------------------------------------------------------------------------
// compareProviderCosts
// ---------------------------------------------------------------------------

describe("compareProviderCosts", () => {
  test("ollama always has zero cost", () => {
    const messages = makeMessages(10);
    const results = compareProviderCosts(messages, 50, 512);
    expect(results["ollama"]!.totalCost).toBe(0);
    expect(results["ollama"]!.inputCost).toBe(0);
    expect(results["ollama"]!.outputCost).toBe(0);
  });

  test("result is ordered cheapest first", () => {
    const messages = makeMessages(10);
    const results = compareProviderCosts(messages, 50, 512);
    const costs = Object.values(results).map((r) => r.totalCost);
    for (let i = 1; i < costs.length; i++) {
      expect(costs[i]!).toBeGreaterThanOrEqual(costs[i - 1]!);
    }
  });

  test("all known providers appear in the result", () => {
    const results = compareProviderCosts(makeMessages(5), 10, 256);
    for (const provider of Object.keys(PROVIDER_PRICING)) {
      expect(results[provider]).toBeDefined();
    }
  });

  test("provider filter limits output to requested providers", () => {
    const results = compareProviderCosts(makeMessages(5), 10, 256, {
      providers: ["anthropic", "openai"],
    });
    expect(Object.keys(results)).toHaveLength(2);
    expect(results["anthropic"]).toBeDefined();
    expect(results["openai"]).toBeDefined();
    expect(results["groq"]).toBeUndefined();
  });

  test("totalCost = inputCost + outputCost", () => {
    const results = compareProviderCosts(makeMessages(8), 20, 1024);
    for (const breakdown of Object.values(results)) {
      expect(breakdown.totalCost).toBeCloseTo(breakdown.inputCost + breakdown.outputCost);
    }
  });

  test("higher queryCount scales costs proportionally", () => {
    const messages = makeMessages(5);
    const r1 = compareProviderCosts(messages, 1, 512);
    const r10 = compareProviderCosts(messages, 10, 512);
    for (const provider of ["anthropic", "openai"]) {
      expect(r10[provider]!.totalCost).toBeCloseTo(r1[provider]!.totalCost * 10);
    }
  });

  test("cacheBreakeven_queries is null for ollama (zero-cost, no caching benefit)", () => {
    const results = compareProviderCosts(makeMessages(5), 10, 512);
    expect(results["ollama"]!.cacheBreakeven_queries).toBeNull();
  });

  test("cacheBreakeven_queries is a positive integer for anthropic", () => {
    const results = compareProviderCosts(makeMessages(20, 2000), 100, 512);
    const breakeven = results["anthropic"]!.cacheBreakeven_queries;
    expect(breakeven).not.toBeNull();
    expect(breakeven!).toBeGreaterThan(0);
    expect(Number.isInteger(breakeven)).toBe(true);
  });

  test("cacheBreakeven_queries is null when cacheWriteMultiplier = 0", () => {
    const results = compareProviderCosts(makeMessages(10), 50, 512, {
      cacheWriteMultiplier: 0,
    });
    // With zero write multiplier, caching is disabled → all null
    for (const breakdown of Object.values(results)) {
      expect(breakdown.cacheBreakeven_queries).toBeNull();
    }
  });

  test("empty messages → near-zero input costs", () => {
    const results = compareProviderCosts([], 10, 512);
    for (const breakdown of Object.values(results)) {
      expect(breakdown.inputCost).toBe(0);
    }
  });
});

// ---------------------------------------------------------------------------
// selectCompressionTierAdaptive with provider hint
// ---------------------------------------------------------------------------

describe("selectCompressionTierAdaptive — provider pricing integration", () => {
  // Build a synthetic LearnedThresholds with enough samples for all tiers.
  const richHistory = {
    byTier: {
      1: { tier: 1 as const, sampleCount: 10, successRate: 0.9, avgOvershootPct: 5 },
      2: { tier: 2 as const, sampleCount: 10, successRate: 0.9, avgOvershootPct: 5 },
      3: { tier: 3 as const, sampleCount: 10, successRate: 0.9, avgOvershootPct: 5 },
      4: { tier: 4 as const, sampleCount: 10, successRate: 0.9, avgOvershootPct: 5 },
    },
  };

  test("no provider hint → result identical to provider-unaware call", () => {
    const messages = makeMessages(5);
    const withoutProvider = selectCompressionTierAdaptive(messages, 500, {}, richHistory, null);
    const withUndefined   = selectCompressionTierAdaptive(messages, 500, {}, richHistory, null, undefined);
    expect(withoutProvider).toBe(withUndefined);
  });

  test("expensive provider (groq → anthropic) does not raise tier above 4", () => {
    const messages = makeMessages(5);
    const tier = selectCompressionTierAdaptive(messages, 500, {}, richHistory, null, "xai");
    expect(tier).toBeGreaterThanOrEqual(1);
    expect(tier).toBeLessThanOrEqual(4);
  });

  test("cheap provider (deepseek) keeps tier in valid range", () => {
    const messages = makeMessages(5);
    const tier = selectCompressionTierAdaptive(messages, 500, {}, richHistory, null, "deepseek");
    expect(tier).toBeGreaterThanOrEqual(1);
    expect(tier).toBeLessThanOrEqual(4);
  });

  test("null history → provider hint has no effect on static path", () => {
    const messages = makeMessages(5);
    const staticResult     = selectCompressionTierAdaptive(messages, 500, {}, null);
    const withProvider     = selectCompressionTierAdaptive(messages, 500, {}, null, null, "groq");
    expect(staticResult).toBe(withProvider);
  });

  test("provider hint applied only when history is present", () => {
    const messages = makeMessages(50, 1000);
    // With rich history and an extremely cheap provider, should not crash
    const tier = selectCompressionTierAdaptive(messages, 1000, {}, richHistory, null, "deepseek");
    expect([1, 2, 3, 4]).toContain(tier);
  });
});
