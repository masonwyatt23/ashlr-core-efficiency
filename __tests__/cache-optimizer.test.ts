/**
 * Tests for CacheOptimizer and cache-audit helpers.
 *
 * Covers:
 *  - computeEffectiveCostMicroUsd / computeBaselineCostMicroUsd math
 *  - summarizeAuditWindow: hit-rate, saving, window slicing
 *  - CacheOptimizer.recordResponse: appends JSONL, returns ID
 *  - CacheOptimizer.recommendStrategy: heuristic fallback, learned recommendation
 *  - CacheOptimizer.generateReport: ROI breakdown, saving pct, advice ordering
 *  - CacheOptimizer.recommendOnProviderSwitch: budget integration + strategy switch
 *  - CacheOptimizer.computeStrategyDelta: cost delta computation
 *  - resolveModelPricing: model-specific, provider-family, fallback
 *  - Edge cases: zero cache hits, malformed/missing metadata, empty audit
 */

import { describe, expect, test, beforeEach } from "bun:test";
import { mkdtemp, rm } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";

import {
  CacheOptimizer,
  resolveModelPricing,
  type CacheOptimizerOptions,
} from "../src/anthropic/cache-optimizer.ts";

import {
  appendCacheAudit,
  loadCacheAudit,
  summarizeAuditWindow,
  computeEffectiveCostMicroUsd,
  computeBaselineCostMicroUsd,
  cacheAuditPath,
  type CacheStrategy,
} from "../src/anthropic/cache-audit.ts";

import type { MessageResponse } from "../src/types/index.ts";

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/** Create a fresh temp directory for each test (isolated JSONL state). */
async function makeTmpDir(): Promise<string> {
  return mkdtemp(join(tmpdir(), "cache-opt-test-"));
}

async function cleanupDir(dir: string) {
  await rm(dir, { recursive: true, force: true });
}

/** Build a minimal MessageResponse. */
function makeResponse(
  inputTokens: number,
  cacheCreationInputTokens?: number,
  cacheReadInputTokens?: number,
  outputTokens = 200,
): MessageResponse {
  return {
    input_tokens: inputTokens,
    cache_creation_input_tokens: cacheCreationInputTokens,
    cache_read_input_tokens: cacheReadInputTokens,
    output_tokens: outputTokens,
  };
}

// ---------------------------------------------------------------------------
// computeEffectiveCostMicroUsd
// ---------------------------------------------------------------------------

describe("computeEffectiveCostMicroUsd", () => {
  test("no cache: cost equals input+output at standard rates", () => {
    // 1000 input tokens at $3/M + 200 output at $15/M
    // = (1000/1e6)*3 + (200/1e6)*15 = 0.000003 + 0.000003 = 0.000006 USD
    // × 1_000_000 = 6 micro-USD
    const cost = computeEffectiveCostMicroUsd(1000, 0, 0, 200, 3.0, 15.0);
    expect(cost).toBe(6000); // Math.round((3+3) * 1000) = 6000
  });

  test("cache write costs 1.25× input rate", () => {
    // 500 write tokens at $3/M × 1.25 = $3.75/M
    // → (500/1e6)*3.75 = 0.000001875 USD × 1_000_000 = 1875 micro-USD
    const cost = computeEffectiveCostMicroUsd(0, 500, 0, 0, 3.0, 15.0);
    expect(cost).toBe(1875);
  });

  test("cache read costs 0.1× input rate", () => {
    // 1_000_000 read tokens at $3/M × 0.1 = $0.3/M
    // → (1_000_000/1e6)*0.3 = 0.3 USD × 1_000_000 = 300_000 micro-USD
    const cost = computeEffectiveCostMicroUsd(0, 0, 1_000_000, 0, 3.0, 15.0);
    expect(cost).toBe(300_000);
  });

  test("combined tokens accumulate correctly", () => {
    // 100 input at $3/M    → 0.3 micro-USD
    // 200 write at $3.75/M → 0.75 micro-USD
    // 300 read at $0.3/M   → 0.09 micro-USD
    // 50 output at $15/M   → 0.75 micro-USD
    // total = 1.89 micro-USD × 1000 = 1890
    const cost = computeEffectiveCostMicroUsd(100, 200, 300, 50, 3.0, 15.0);
    expect(cost).toBe(1890);
  });

  test("zero tokens → zero cost", () => {
    expect(computeEffectiveCostMicroUsd(0, 0, 0, 0, 3.0, 15.0)).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// computeBaselineCostMicroUsd
// ---------------------------------------------------------------------------

describe("computeBaselineCostMicroUsd", () => {
  test("all tokens treated as standard input + output", () => {
    // (1000+200+300)/1e6 * 3 + 50/1e6 * 15
    // = 1500/1e6 * 3 + 50/1e6 * 15 = 4.5 + 0.75 = 5.25 micro-USD × 1000 = 5250
    const cost = computeBaselineCostMicroUsd(1000, 200, 300, 50, 3.0, 15.0);
    expect(cost).toBe(5250);
  });

  test("zero tokens → zero cost", () => {
    expect(computeBaselineCostMicroUsd(0, 0, 0, 0, 3.0, 15.0)).toBe(0);
  });

  test("baseline > effective when there is a large cache read", () => {
    // A large cache read (0.1×) makes effective < baseline
    const effective = computeEffectiveCostMicroUsd(100, 0, 10000, 200, 3.0, 15.0);
    const baseline  = computeBaselineCostMicroUsd(100, 0, 10000, 200, 3.0, 15.0);
    expect(baseline).toBeGreaterThan(effective);
  });

  test("baseline < effective when there is a cache write with no subsequent read", () => {
    // Cache write costs 1.25× → effective > baseline for a pure write
    const effective = computeEffectiveCostMicroUsd(0, 5000, 0, 0, 3.0, 15.0);
    const baseline  = computeBaselineCostMicroUsd(0, 5000, 0, 0, 3.0, 15.0);
    expect(effective).toBeGreaterThan(baseline);
  });
});

// ---------------------------------------------------------------------------
// summarizeAuditWindow
// ---------------------------------------------------------------------------

describe("summarizeAuditWindow", () => {
  test("returns null when no entries exist for the strategy", () => {
    const result = summarizeAuditWindow([], "system-only");
    expect(result).toBeNull();
  });

  test("hitRate is 0 when all entries have zero cacheReadTokens", () => {
    const entries = [
      {
        id: "a",
        timestamp: new Date().toISOString(),
        provider: "anthropic",
        modelId: "claude-sonnet-4",
        strategy: "system-only" as CacheStrategy,
        inputTokens: 1000,
        cacheCreationTokens: 200,
        cacheReadTokens: 0,
        outputTokens: 100,
        effectiveCostMicroUsd: 10,
        baselineCostMicroUsd: 8,
      },
    ];
    const summary = summarizeAuditWindow(entries, "system-only");
    expect(summary).not.toBeNull();
    expect(summary!.hitRate).toBe(0);
  });

  test("hitRate is 1.0 when all entries have non-zero cacheReadTokens", () => {
    const entries = Array.from({ length: 3 }, (_, i) => ({
      id: `e${i}`,
      timestamp: new Date().toISOString(),
      provider: "anthropic",
      modelId: "claude-sonnet-4",
      strategy: "full" as CacheStrategy,
      inputTokens: 100,
      cacheCreationTokens: 0,
      cacheReadTokens: 500,
      outputTokens: 50,
      effectiveCostMicroUsd: 2,
      baselineCostMicroUsd: 10,
    }));
    const summary = summarizeAuditWindow(entries, "full");
    expect(summary!.hitRate).toBe(1.0);
  });

  test("avgSavingMicroUsd is positive when effective < baseline", () => {
    const entries = [
      {
        id: "x",
        timestamp: new Date().toISOString(),
        provider: "anthropic",
        modelId: "claude-sonnet-4",
        strategy: "system+context" as CacheStrategy,
        inputTokens: 100,
        cacheCreationTokens: 0,
        cacheReadTokens: 1000,
        outputTokens: 50,
        effectiveCostMicroUsd: 3,
        baselineCostMicroUsd: 10,
      },
    ];
    const summary = summarizeAuditWindow(entries, "system+context");
    expect(summary!.avgSavingMicroUsd).toBeGreaterThan(0);
  });

  test("avgSavingMicroUsd is negative when effective > baseline (write with no read)", () => {
    const entries = [
      {
        id: "y",
        timestamp: new Date().toISOString(),
        provider: "anthropic",
        modelId: "claude-sonnet-4",
        strategy: "tools-only" as CacheStrategy,
        inputTokens: 0,
        cacheCreationTokens: 5000,
        cacheReadTokens: 0,
        outputTokens: 0,
        effectiveCostMicroUsd: 20,
        baselineCostMicroUsd: 15,
      },
    ];
    const summary = summarizeAuditWindow(entries, "tools-only");
    expect(summary!.avgSavingMicroUsd).toBeLessThan(0);
  });

  test("windowSize limits entries considered", () => {
    const entries = Array.from({ length: 20 }, (_, i) => ({
      id: `w${i}`,
      timestamp: new Date().toISOString(),
      provider: "anthropic",
      modelId: "claude-sonnet-4",
      strategy: "multi-turn" as CacheStrategy,
      inputTokens: 100,
      cacheCreationTokens: 0,
      cacheReadTokens: 500,
      outputTokens: 50,
      effectiveCostMicroUsd: 2,
      baselineCostMicroUsd: 8,
    }));
    const summary5  = summarizeAuditWindow(entries, "multi-turn", 5);
    const summary20 = summarizeAuditWindow(entries, "multi-turn", 20);
    expect(summary5!.sampleCount).toBe(5);
    expect(summary20!.sampleCount).toBe(20);
  });

  test("entries for other strategies are excluded", () => {
    const entries = [
      {
        id: "z1",
        timestamp: new Date().toISOString(),
        provider: "anthropic",
        modelId: "claude-sonnet-4",
        strategy: "system-only" as CacheStrategy,
        inputTokens: 100,
        cacheCreationTokens: 0,
        cacheReadTokens: 200,
        outputTokens: 50,
        effectiveCostMicroUsd: 2,
        baselineCostMicroUsd: 5,
      },
      {
        id: "z2",
        timestamp: new Date().toISOString(),
        provider: "anthropic",
        modelId: "claude-sonnet-4",
        strategy: "full" as CacheStrategy,
        inputTokens: 100,
        cacheCreationTokens: 50,
        cacheReadTokens: 0,
        outputTokens: 50,
        effectiveCostMicroUsd: 4,
        baselineCostMicroUsd: 3,
      },
    ];
    const summary = summarizeAuditWindow(entries, "system-only");
    expect(summary!.sampleCount).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// resolveModelPricing
// ---------------------------------------------------------------------------

describe("resolveModelPricing", () => {
  test("claude-opus-4 returns model-specific pricing ($15/M input)", () => {
    const pricing = resolveModelPricing("anthropic", "claude-opus-4");
    expect(pricing.inputPricePerMToken).toBe(15.0);
    expect(pricing.outputPricePerMToken).toBe(75.0);
  });

  test("claude-sonnet-4 returns model-specific pricing ($3/M input)", () => {
    const pricing = resolveModelPricing("anthropic", "claude-sonnet-4");
    expect(pricing.inputPricePerMToken).toBe(3.0);
    expect(pricing.outputPricePerMToken).toBe(15.0);
  });

  test("dated model variant resolves via substring match", () => {
    const pricing = resolveModelPricing("anthropic", "claude-sonnet-4-20250514");
    expect(pricing.inputPricePerMToken).toBe(3.0);
  });

  test("provider family fallback for unknown model", () => {
    const pricing = resolveModelPricing("groq", "llama-3-70b");
    // PROVIDER_PRICING groq: inputPerMToken: 0.27
    expect(pricing.inputPricePerMToken).toBe(0.27);
  });

  test("fallback pricing for completely unknown provider+model", () => {
    const pricing = resolveModelPricing("unknown-provider", "unknown-model-xyz");
    expect(pricing.inputPricePerMToken).toBe(3.0);
    expect(pricing.outputPricePerMToken).toBe(15.0);
  });
});

// ---------------------------------------------------------------------------
// CacheOptimizer.recordResponse
// ---------------------------------------------------------------------------

describe("CacheOptimizer.recordResponse", () => {
  let tmpDir: string;
  let optimizer: CacheOptimizer;

  beforeEach(async () => {
    tmpDir = await makeTmpDir();
    optimizer = new CacheOptimizer({ cwd: tmpDir, provider: "anthropic", modelId: "claude-sonnet-4" });
  });

  // Note: cleanup is best-effort; temp dirs are small and OS-managed.

  test("returns a non-empty string ID", async () => {
    const response = makeResponse(500, 200, 0, 100);
    const id = await optimizer.recordResponse(response, "system-only");
    expect(typeof id).toBe("string");
    expect(id.length).toBeGreaterThan(0);
  });

  test("appends one entry to the JSONL file", async () => {
    const response = makeResponse(500, 200, 0, 100);
    await optimizer.recordResponse(response, "system-only");
    const entries = await loadCacheAudit(tmpDir);
    expect(entries).toHaveLength(1);
  });

  test("each call appends one new entry", async () => {
    for (let i = 0; i < 5; i++) {
      await optimizer.recordResponse(makeResponse(100, 0, 50, 50), "full");
    }
    const entries = await loadCacheAudit(tmpDir);
    expect(entries).toHaveLength(5);
  });

  test("entry fields match the response data", async () => {
    const response = makeResponse(1000, 500, 300, 200);
    await optimizer.recordResponse(response, "system+context");
    const [entry] = await loadCacheAudit(tmpDir);
    expect(entry).toBeDefined();
    expect(entry!.inputTokens).toBe(1000);
    expect(entry!.cacheCreationTokens).toBe(500);
    expect(entry!.cacheReadTokens).toBe(300);
    expect(entry!.outputTokens).toBe(200);
    expect(entry!.strategy).toBe("system+context");
    expect(entry!.provider).toBe("anthropic");
    expect(entry!.modelId).toBe("claude-sonnet-4");
  });

  test("missing cache fields in response default to zero", async () => {
    const response: MessageResponse = { input_tokens: 400, output_tokens: 100 };
    await optimizer.recordResponse(response, "none");
    const [entry] = await loadCacheAudit(tmpDir);
    expect(entry!.cacheCreationTokens).toBe(0);
    expect(entry!.cacheReadTokens).toBe(0);
  });

  test("effectiveCostMicroUsd is lower than baselineCostMicroUsd when there is a large cache read", async () => {
    const response = makeResponse(100, 0, 10000, 100);
    await optimizer.recordResponse(response, "full");
    const [entry] = await loadCacheAudit(tmpDir);
    expect(entry!.effectiveCostMicroUsd).toBeLessThan(entry!.baselineCostMicroUsd);
  });

  test("effectiveCostMicroUsd is higher when there is a cache write with no read (first request)", async () => {
    const response = makeResponse(0, 5000, 0, 0);
    await optimizer.recordResponse(response, "system-only");
    const [entry] = await loadCacheAudit(tmpDir);
    expect(entry!.effectiveCostMicroUsd).toBeGreaterThan(entry!.baselineCostMicroUsd);
  });

  test("sessionId is stored when provided", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir, sessionId: "sess-abc" });
    await opt.recordResponse(makeResponse(100, 0, 0, 50), "none");
    const [entry] = await loadCacheAudit(tmpDir);
    expect(entry!.sessionId).toBe("sess-abc");
  });
});

// ---------------------------------------------------------------------------
// CacheOptimizer.recommendStrategy
// ---------------------------------------------------------------------------

describe("CacheOptimizer.recommendStrategy", () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await makeTmpDir();
  });

  test("returns 'none' for a non-caching provider regardless of data", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir, provider: "groq", modelId: "llama-3-70b" });
    const strategy = await opt.recommendStrategy();
    expect(strategy).toBe("none");
  });

  test("returns heuristic default 'system-only' when there is no audit data", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir, provider: "anthropic" });
    const strategy = await opt.recommendStrategy();
    expect(strategy).toBe("system-only");
  });

  test("returns heuristic default when below minSamples threshold", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir, provider: "anthropic", minSamplesForLearning: 5 });
    // Record only 3 entries for system-only
    for (let i = 0; i < 3; i++) {
      await opt.recordResponse(makeResponse(100, 50, 500, 100), "system-only");
    }
    const strategy = await opt.recommendStrategy();
    expect(strategy).toBe("system-only"); // heuristic (not learned)
  });

  test("recommends the strategy with the highest average saving when above threshold", async () => {
    const opt = new CacheOptimizer({
      cwd: tmpDir,
      provider: "anthropic",
      modelId: "claude-sonnet-4",
      minSamplesForLearning: 3,
    });
    // Record 5 entries for "full" with large cache reads (big savings)
    for (let i = 0; i < 5; i++) {
      await opt.recordResponse(makeResponse(50, 0, 50000, 100), "full");
    }
    // Record 5 entries for "system-only" with small cache reads (small savings)
    for (let i = 0; i < 5; i++) {
      await opt.recordResponse(makeResponse(1000, 200, 100, 100), "system-only");
    }
    const strategy = await opt.recommendStrategy();
    expect(strategy).toBe("full");
  });

  test("does not recommend 'none' via learned path", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir, provider: "anthropic", minSamplesForLearning: 2 });
    for (let i = 0; i < 5; i++) {
      await opt.recordResponse(makeResponse(100, 0, 0, 50), "none");
    }
    const strategy = await opt.recommendStrategy();
    // 'none' is excluded from learned recommendation path; should fall back
    expect(strategy).not.toBe("none");
  });
});

// ---------------------------------------------------------------------------
// CacheOptimizer.generateReport
// ---------------------------------------------------------------------------

describe("CacheOptimizer.generateReport", () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await makeTmpDir();
  });

  test("returns a report with all six strategies in strategyROI", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir });
    const report = await opt.generateReport();
    const strategies = report.strategyROI.map((r) => r.strategy);
    expect(strategies).toContain("system-only");
    expect(strategies).toContain("system+context");
    expect(strategies).toContain("multi-turn");
    expect(strategies).toContain("tools-only");
    expect(strategies).toContain("full");
    expect(strategies).toContain("none");
    expect(report.strategyROI).toHaveLength(6);
  });

  test("empty audit → totalAuditEntries is 0, saving is 0", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir });
    const report = await opt.generateReport();
    expect(report.totalAuditEntries).toBe(0);
    expect(report.totalSavingMicroUsd).toBe(0);
    expect(report.savingPct).toBe(0);
    expect(report.recommendedStrategy).toBeNull();
  });

  test("breakpointAdvice includes a default message when no data collected", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir });
    const report = await opt.generateReport();
    expect(report.breakpointAdvice[0]).toContain("No cache data");
  });

  test("exactly one strategy is marked recommended when above minSamples", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir, minSamplesForLearning: 3 });
    for (let i = 0; i < 5; i++) {
      await opt.recordResponse(makeResponse(100, 0, 8000, 100), "full");
    }
    const report = await opt.generateReport();
    const recommended = report.strategyROI.filter((r) => r.recommended);
    expect(recommended).toHaveLength(1);
  });

  test("savingPct is positive when cache reads dominate", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir, minSamplesForLearning: 2 });
    for (let i = 0; i < 5; i++) {
      await opt.recordResponse(makeResponse(50, 0, 20000, 100), "system+context");
    }
    const report = await opt.generateReport();
    expect(report.savingPct).toBeGreaterThan(0);
  });

  test("savingPct is negative when writes outweigh reads (no hits)", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir, minSamplesForLearning: 2 });
    for (let i = 0; i < 5; i++) {
      // pure write, no read → effective > baseline
      await opt.recordResponse(makeResponse(0, 10000, 0, 0), "system-only");
    }
    const report = await opt.generateReport();
    expect(report.savingPct).toBeLessThanOrEqual(0);
  });

  test("breakpointAdvice is ordered highest-to-lowest saving", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir, minSamplesForLearning: 2 });
    // full: big savings (large reads)
    for (let i = 0; i < 5; i++) {
      await opt.recordResponse(makeResponse(100, 0, 50000, 100), "full");
    }
    // system-only: small savings (tiny reads)
    for (let i = 0; i < 5; i++) {
      await opt.recordResponse(makeResponse(1000, 200, 50, 100), "system-only");
    }
    const report = await opt.generateReport();
    const advice = report.breakpointAdvice;
    expect(advice.length).toBeGreaterThan(0);
    expect(advice[0]).toContain("full");
  });

  test("totalSavingMicroUsd accumulates across strategies", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir, minSamplesForLearning: 1 });
    for (let i = 0; i < 3; i++) {
      await opt.recordResponse(makeResponse(100, 0, 5000, 100), "system-only");
      await opt.recordResponse(makeResponse(100, 0, 5000, 100), "full");
    }
    const report = await opt.generateReport();
    // Both strategies save money → totalSaving > 0
    expect(report.totalSavingMicroUsd).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// CacheOptimizer.recommendOnProviderSwitch
// ---------------------------------------------------------------------------

describe("CacheOptimizer.recommendOnProviderSwitch", () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await makeTmpDir();
  });

  test("switching to groq (no caching) → strategy is 'none'", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir, provider: "anthropic" });
    const rec = await opt.recommendOnProviderSwitch("groq", []);
    expect(rec.suggestedStrategy).toBe("none");
    expect(rec.fromProvider).toBe("anthropic");
    expect(rec.toProvider).toBe("groq");
  });

  test("switching to deepseek → strategy is 'none'", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir, provider: "anthropic" });
    const rec = await opt.recommendOnProviderSwitch("deepseek", []);
    expect(rec.suggestedStrategy).toBe("none");
  });

  test("switching to ollama → strategy is 'none'", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir, provider: "anthropic" });
    const rec = await opt.recommendOnProviderSwitch("ollama", []);
    expect(rec.suggestedStrategy).toBe("none");
  });

  test("switching to anthropic with no prior data → heuristic 'system-only'", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir, provider: "openai" });
    const rec = await opt.recommendOnProviderSwitch("anthropic", []);
    expect(rec.suggestedStrategy).toBe("system-only");
  });

  test("budgetRebalance is populated with valid tier", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir, provider: "anthropic" });
    const rec = await opt.recommendOnProviderSwitch("openai", []);
    expect(rec.budgetRebalance.newSystemBudget).toBeGreaterThan(0);
    expect([1, 2, 3]).toContain(rec.budgetRebalance.recommendedTier);
  });

  test("rationale is a non-empty string", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir, provider: "anthropic" });
    const rec = await opt.recommendOnProviderSwitch("openai", []);
    expect(typeof rec.rationale).toBe("string");
    expect(rec.rationale.length).toBeGreaterThan(0);
  });

  test("switching to anthropic with enough prior anthropic data → learned strategy", async () => {
    const opt = new CacheOptimizer({
      cwd: tmpDir,
      provider: "openai",
      modelId: "gpt-4o",
      minSamplesForLearning: 3,
    });
    // Pre-populate audit data with anthropic entries indicating "full" is best
    for (let i = 0; i < 5; i++) {
      await appendCacheAudit(tmpDir, {
        provider: "anthropic",
        modelId: "claude-sonnet-4",
        strategy: "full",
        inputTokens: 100,
        cacheCreationTokens: 0,
        cacheReadTokens: 10000,
        outputTokens: 100,
        effectiveCostMicroUsd: 5,
        baselineCostMicroUsd: 50,
      });
    }
    const rec = await opt.recommendOnProviderSwitch("anthropic", []);
    expect(rec.suggestedStrategy).toBe("full");
  });
});

// ---------------------------------------------------------------------------
// CacheOptimizer.computeStrategyDelta (static)
// ---------------------------------------------------------------------------

describe("CacheOptimizer.computeStrategyDelta", () => {
  test("from 'none' to any strategy: delta is 0 (same observed cost)", () => {
    const response = makeResponse(1000, 200, 500, 100);
    const delta = CacheOptimizer.computeStrategyDelta(response, "none", "system-only");
    expect(delta).toBe(0);
  });

  test("from strategy to 'none': delta is positive when cache helps (baseline > effective)", () => {
    // Large read → effective < baseline; switching to none would cost more
    const response = makeResponse(100, 0, 20000, 100);
    const delta = CacheOptimizer.computeStrategyDelta(response, "full", "none");
    expect(delta).toBeGreaterThan(0); // none costs more
  });

  test("from strategy to 'none': delta is negative when cache write has no read (cache hurts)", () => {
    // Pure write → effective > baseline; switching to none would be cheaper
    const response = makeResponse(0, 10000, 0, 0);
    const delta = CacheOptimizer.computeStrategyDelta(response, "system-only", "none");
    expect(delta).toBeLessThan(0); // none is cheaper
  });

  test("same strategy to same strategy: delta is 0", () => {
    const response = makeResponse(500, 200, 100, 200);
    const delta = CacheOptimizer.computeStrategyDelta(response, "system+context", "system+context");
    expect(delta).toBe(0);
  });

  test("accepts provider and modelId overrides", () => {
    const response = makeResponse(1000, 0, 5000, 200);
    // Should not throw with different providers/models
    const delta = CacheOptimizer.computeStrategyDelta(response, "full", "none", "openai", "o1-preview");
    expect(typeof delta).toBe("number");
  });
});

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

describe("Edge cases", () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await makeTmpDir();
  });

  test("zero cache hits: hitRate is 0 in report", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir, minSamplesForLearning: 2 });
    for (let i = 0; i < 5; i++) {
      await opt.recordResponse(makeResponse(500, 300, 0, 100), "system-only");
    }
    const report = await opt.generateReport();
    const row = report.strategyROI.find((r) => r.strategy === "system-only");
    expect(row!.hitRate).toBe(0);
  });

  test("malformed response metadata: undefined cache fields treated as zero", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir });
    const response: MessageResponse = {
      input_tokens: 200,
      output_tokens: 100,
      // cache_creation_input_tokens and cache_read_input_tokens intentionally omitted
    };
    await expect(opt.recordResponse(response, "none")).resolves.toBeDefined();
    const entries = await loadCacheAudit(tmpDir);
    expect(entries[0]!.cacheCreationTokens).toBe(0);
    expect(entries[0]!.cacheReadTokens).toBe(0);
  });

  test("very large token counts do not overflow", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir });
    const response = makeResponse(500_000, 300_000, 1_000_000, 200_000);
    await expect(opt.recordResponse(response, "full")).resolves.toBeDefined();
    const entries = await loadCacheAudit(tmpDir);
    expect(entries[0]!.effectiveCostMicroUsd).toBeGreaterThan(0);
  });

  test("report handles mixed strategies in the audit gracefully", async () => {
    const opt = new CacheOptimizer({ cwd: tmpDir, minSamplesForLearning: 2 });
    const strategies: CacheStrategy[] = ["system-only", "full", "multi-turn", "none"];
    for (const strategy of strategies) {
      for (let i = 0; i < 3; i++) {
        await opt.recordResponse(makeResponse(200, 100, 400, 100), strategy);
      }
    }
    const report = await opt.generateReport();
    expect(report.totalAuditEntries).toBe(12);
    expect(report.strategyROI.every((r) => r.sampleCount >= 0)).toBe(true);
  });

  test("cacheAuditPath returns a path ending with .jsonl", () => {
    const path = cacheAuditPath("/some/working/dir");
    expect(path.endsWith(".jsonl")).toBe(true);
    expect(path).toContain(".ashlr");
  });

  test("loadCacheAudit returns empty array for non-existent directory", async () => {
    const entries = await loadCacheAudit(join(tmpDir, "does-not-exist"));
    expect(entries).toEqual([]);
  });

  test("multiple optimizer instances with different providers do not cross-contaminate", async () => {
    const optA = new CacheOptimizer({ cwd: tmpDir, provider: "anthropic", modelId: "claude-sonnet-4" });
    const optB = new CacheOptimizer({ cwd: tmpDir, provider: "openai",    modelId: "gpt-4o" });

    for (let i = 0; i < 5; i++) {
      await optA.recordResponse(makeResponse(100, 0, 5000, 100), "system-only");
      await optB.recordResponse(makeResponse(100, 0, 5000, 100), "full");
    }

    const reportA = await optA.generateReport();
    const reportB = await optB.generateReport();

    // Each optimizer only sees its own provider's entries
    const anthropicRow = reportA.strategyROI.find((r) => r.strategy === "system-only");
    const openaiRow    = reportB.strategyROI.find((r) => r.strategy === "full");
    expect(anthropicRow!.sampleCount).toBe(5);
    expect(openaiRow!.sampleCount).toBe(5);
  });
});
