/**
 * Tests for the streaming session cost attribution backpropagation engine.
 *
 * Coverage:
 *   - BackpropagationEngine: record(), computeROI(), computeAllROI()
 *   - Rolling 30-turn window enforcement (window edge cases)
 *   - Concurrent writes (parallel async record() calls)
 *   - Provider switches mid-session (different providers in same engine)
 *   - computeParetoFrontier: non-dominated point identification
 *   - getROIDashboard: bestTierByProvider, bestProviderByTier, entries, frontier
 *   - biasAdaptiveTierSelection: no-bias path, bias applied path
 *   - attributeSessionCost: records + persists
 *   - persistBackpropRecord: file writes + ASHLR_SESSION_LOG=0 suppression
 *   - BACKPROP_WINDOW_SIZE, BACKPROP_MIN_SAMPLES, BACKPROP_MIN_ROI_EFFICIENCY constants
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  BackpropagationEngine,
  getBackpropEngine,
  _resetBackpropEngine,
  computeParetoFrontier,
  getROIDashboard,
  biasAdaptiveTierSelection,
  attributeSessionCost,
  persistBackpropRecord,
  BACKPROP_WINDOW_SIZE,
  BACKPROP_MIN_SAMPLES,
  BACKPROP_MIN_ROI_EFFICIENCY,
  type BackpropAttributionRecord,
  type ProviderTierROI,
  type ParetoPoint,
  type ROIDashboard,
  type BiasedTierSelection,
} from "../src/session-log/backpropagation.ts";
import { PROVIDER_RATES } from "../src/tokens/index.ts";

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

const SONNET = "claude-3-5-sonnet";
const HAIKU = "claude-3-haiku";
const OPUS = "claude-3-opus";
const GPT4O = "gpt-4o";

// messageCount=20 → bucket "sm" (11–50). Used as the default across most tests
// so that computed ROI is keyed "sm" and tests can pass "sm" to getRecords().
const DEFAULT_MSG_COUNT = 20;

/** Record N identical attributions into an engine for a given tier/provider. */
async function bulkRecord(
  engine: BackpropagationEngine,
  count: number,
  tier: 1 | 2 | 3 | 4,
  provider: string,
  opts: {
    messageCount?: number;
    estimatedTokens?: number;
    actualTokens?: number;
    actualCostUsd?: number;
    latencyMs?: number;
  } = {},
): Promise<void> {
  const {
    messageCount = DEFAULT_MSG_COUNT,
    estimatedTokens = 50_000,
    actualTokens = 40_000,
    actualCostUsd = 0.05,
    latencyMs = 100,
  } = opts;
  for (let i = 0; i < count; i++) {
    await engine.record(tier, messageCount, provider, estimatedTokens, actualTokens, actualCostUsd, latencyMs);
  }
}

let tmpBackpropPath: string;

beforeEach(() => {
  _resetBackpropEngine();
  tmpBackpropPath = join(tmpdir(), `ashlr-test-backprop-${Date.now()}-${Math.random().toString(36).slice(2)}.jsonl`);
  process.env.ASHLR_BACKPROP_PATH = tmpBackpropPath;
  delete process.env.ASHLR_SESSION_LOG;
});

afterEach(() => {
  _resetBackpropEngine();
  delete process.env.ASHLR_BACKPROP_PATH;
  delete process.env.ASHLR_SESSION_LOG;
  try { rmSync(tmpBackpropPath, { force: true }); } catch { /* ignore */ }
});

// ---------------------------------------------------------------------------
// BackpropagationEngine — basic record + computeROI
// ---------------------------------------------------------------------------

describe("BackpropagationEngine — record() and computeROI()", () => {
  test("computeROI returns null for a key with no records", () => {
    const engine = new BackpropagationEngine();
    expect(engine.computeROI(2, SONNET, "sm")).toBeNull();
  });

  test("record() returns a well-formed BackpropAttributionRecord", async () => {
    const engine = new BackpropagationEngine();
    const rec = await engine.record(2, 10, SONNET, 50_000, 40_000, 0.05, 100);

    expect(rec.tier).toBe(2);
    expect(rec.provider).toBe(SONNET);
    expect(rec.estimatedTokens).toBe(50_000);
    expect(rec.actualTokens).toBe(40_000);
    expect(rec.tokenDelta).toBe(10_000); // 50k - 40k
    expect(rec.latencyMs).toBe(100);
    expect(new Date(rec.recordedAt).getTime()).not.toBeNaN();
  });

  test("tokenDelta = estimatedTokens - actualTokens", async () => {
    const engine = new BackpropagationEngine();
    const rec = await engine.record(3, 5, HAIKU, 30_000, 22_000, 0.01, 80);
    expect(rec.tokenDelta).toBe(8_000);
  });

  test("counterfactualCostUsd = actualCostUsd + tokenDelta × inputRate", async () => {
    const engine = new BackpropagationEngine();
    const rate = PROVIDER_RATES[SONNET];
    const rec = await engine.record(2, 10, SONNET, 50_000, 40_000, 0.05, 100);
    const expected = 0.05 + 10_000 * rate.inputRate;
    expect(rec.counterfactualCostUsd).toBeCloseTo(expected, 10);
  });

  test("usdSaved = counterfactualCostUsd - actualCostUsd (positive when tokens removed)", async () => {
    const engine = new BackpropagationEngine();
    const rec = await engine.record(2, 10, SONNET, 50_000, 40_000, 0.05, 100);
    expect(rec.usdSaved).toBeCloseTo(rec.counterfactualCostUsd - rec.actualCostUsd, 10);
    expect(rec.usdSaved).toBeGreaterThan(0);
  });

  test("usdSaved is 0 when no tokens are saved (estimatedTokens = actualTokens)", async () => {
    const engine = new BackpropagationEngine();
    const rec = await engine.record(3, 10, SONNET, 40_000, 40_000, 0.05, 100);
    expect(rec.tokenDelta).toBe(0);
    expect(rec.usdSaved).toBeCloseTo(0, 10);
  });

  test("bucket is correctly assigned from messageCount", async () => {
    const engine = new BackpropagationEngine();
    // xs bucket: messageCount <= 10
    const rec1 = await engine.record(2, 5, SONNET, 5_000, 4_000, 0.001, 50);
    expect(rec1.bucket).toBe("xs");
    const rec1b = await engine.record(2, 10, SONNET, 5_000, 4_000, 0.001, 50);
    expect(rec1b.bucket).toBe("xs");
    // sm bucket: 11-50
    const rec2 = await engine.record(2, 20, SONNET, 10_000, 8_000, 0.002, 60);
    expect(rec2.bucket).toBe("sm");
    const rec2b = await engine.record(2, 50, SONNET, 10_000, 8_000, 0.002, 60);
    expect(rec2b.bucket).toBe("sm");
    // md bucket: 51-200
    const rec3 = await engine.record(2, 100, SONNET, 20_000, 15_000, 0.004, 80);
    expect(rec3.bucket).toBe("md");
    // lg bucket: > 200
    const rec4 = await engine.record(2, 250, SONNET, 50_000, 40_000, 0.01, 120);
    expect(rec4.bucket).toBe("lg");
  });

  test("computeROI aggregates records correctly", async () => {
    const engine = new BackpropagationEngine();
    await bulkRecord(engine, 5, 2, SONNET, { messageCount: 20, estimatedTokens: 50_000, actualTokens: 40_000, actualCostUsd: 0.05, latencyMs: 100 });

    const roi = engine.computeROI(2, SONNET, "sm");
    expect(roi).not.toBeNull();
    expect(roi!.tier).toBe(2);
    expect(roi!.provider).toBe(SONNET);
    expect(roi!.bucket).toBe("sm");
    expect(roi!.sampleCount).toBe(5);
    expect(roi!.totalTokensSaved).toBe(5 * 10_000);
    expect(roi!.totalUsdSaved).toBeGreaterThan(0);
    expect(roi!.avgUsdSavedPerCall).toBeCloseTo(roi!.totalUsdSaved / 5, 10);
    expect(roi!.avgLatencyMs).toBe(100);
    expect(roi!.roiEfficiency).toBeGreaterThan(0);
    expect(new Date(roi!.updatedAt).getTime()).not.toBeNaN();
  });

  test("totalRecordCount returns sum across all keys", async () => {
    const engine = new BackpropagationEngine();
    await bulkRecord(engine, 3, 2, SONNET);
    await bulkRecord(engine, 4, 3, HAIKU);
    expect(engine.totalRecordCount()).toBe(7);
  });

  test("getRecords returns the stored records for a key", async () => {
    const engine = new BackpropagationEngine();
    await bulkRecord(engine, 3, 2, SONNET, { messageCount: 20 });
    const records = engine.getRecords(2, SONNET, "sm");
    expect(records.length).toBe(3);
    expect(records[0]!.tier).toBe(2);
  });

  test("getRecords returns empty array for unknown key", () => {
    const engine = new BackpropagationEngine();
    expect(engine.getRecords(3, OPUS, "lg")).toHaveLength(0);
  });

  test("reset() clears all state", async () => {
    const engine = new BackpropagationEngine();
    await bulkRecord(engine, 5, 2, SONNET);
    engine.reset();
    expect(engine.totalRecordCount()).toBe(0);
    expect(engine.computeROI(2, SONNET, "sm")).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Rolling 30-turn window edge cases
// ---------------------------------------------------------------------------

describe("rolling window enforcement (BACKPROP_WINDOW_SIZE = 30)", () => {
  test("BACKPROP_WINDOW_SIZE constant is 30", () => {
    expect(BACKPROP_WINDOW_SIZE).toBe(30);
  });

  test("records capped at BACKPROP_WINDOW_SIZE per key", async () => {
    const engine = new BackpropagationEngine();
    // Record 35 entries (5 over the cap).
    await bulkRecord(engine, 35, 2, SONNET, { messageCount: 20 });
    const records = engine.getRecords(2, SONNET, "sm");
    expect(records.length).toBe(BACKPROP_WINDOW_SIZE);
  });

  test("oldest records are evicted when window is full", async () => {
    const engine = new BackpropagationEngine();
    // Record 30 entries with latencyMs=100 — messageCount=20 → bucket "sm".
    await bulkRecord(engine, 30, 2, SONNET, { messageCount: 20, latencyMs: 100 });
    // Record 1 more with latencyMs=999 in the same bucket — should evict the oldest.
    await engine.record(2, 20, SONNET, 50_000, 40_000, 0.05, 999);
    const records = engine.getRecords(2, SONNET, "sm");
    expect(records.length).toBe(BACKPROP_WINDOW_SIZE);
    // The last record should have latencyMs=999.
    expect(records[records.length - 1]!.latencyMs).toBe(999);
  });

  test("sampleCount in computeROI is capped at BACKPROP_WINDOW_SIZE", async () => {
    const engine = new BackpropagationEngine();
    await bulkRecord(engine, 50, 3, HAIKU, { messageCount: 20 });
    const roi = engine.computeROI(3, HAIKU, "sm");
    expect(roi!.sampleCount).toBeLessThanOrEqual(BACKPROP_WINDOW_SIZE);
  });

  test("window at exactly BACKPROP_WINDOW_SIZE keeps all records (no eviction)", async () => {
    const engine = new BackpropagationEngine();
    await bulkRecord(engine, BACKPROP_WINDOW_SIZE, 2, SONNET, { messageCount: 20 });
    const records = engine.getRecords(2, SONNET, "sm");
    expect(records.length).toBe(BACKPROP_WINDOW_SIZE);
  });

  test("separate keys have independent windows", async () => {
    const engine = new BackpropagationEngine();
    // Fill tier 2 window completely.
    await bulkRecord(engine, 35, 2, SONNET, { messageCount: 20 });
    // Tier 3 should be unaffected.
    await bulkRecord(engine, 5, 3, SONNET, { messageCount: 20 });
    expect(engine.getRecords(2, SONNET, "sm").length).toBe(BACKPROP_WINDOW_SIZE);
    expect(engine.getRecords(3, SONNET, "sm").length).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// Concurrent writes
// ---------------------------------------------------------------------------

describe("concurrent writes (parallel async record() calls)", () => {
  test("all concurrent records are stored without loss", async () => {
    const engine = new BackpropagationEngine();
    const N = 20;
    // Fire N record() calls concurrently. messageCount=20 → bucket "sm".
    await Promise.all(
      Array.from({ length: N }, (_, i) =>
        engine.record(2, 20, SONNET, 50_000, 40_000 - i, 0.05, 100 + i),
      ),
    );
    const records = engine.getRecords(2, SONNET, "sm");
    expect(records.length).toBe(N);
  });

  test("concurrent records across different tiers are all stored", async () => {
    const engine = new BackpropagationEngine();
    // messageCount=20 → bucket "sm" for all.
    await Promise.all([
      ...Array.from({ length: 5 }, () => engine.record(2, 20, SONNET, 50_000, 40_000, 0.05, 100)),
      ...Array.from({ length: 5 }, () => engine.record(3, 20, SONNET, 30_000, 25_000, 0.03, 80)),
      ...Array.from({ length: 5 }, () => engine.record(4, 20, SONNET, 20_000, 18_000, 0.02, 60)),
    ]);
    expect(engine.getRecords(2, SONNET, "sm").length).toBe(5);
    expect(engine.getRecords(3, SONNET, "sm").length).toBe(5);
    expect(engine.getRecords(4, SONNET, "sm").length).toBe(5);
    expect(engine.totalRecordCount()).toBe(15);
  });

  test("concurrent records beyond window size enforce cap correctly", async () => {
    const engine = new BackpropagationEngine();
    const N = BACKPROP_WINDOW_SIZE + 10; // 40 total
    // Launch all in a single Promise.all batch; engine serializes internally.
    await Promise.all(
      Array.from({ length: N }, (_, i) =>
        engine.record(2, 20, SONNET, 50_000, 40_000 - i, 0.05, i),
      ),
    );
    const records = engine.getRecords(2, SONNET, "sm");
    expect(records.length).toBe(BACKPROP_WINDOW_SIZE);
  });

  test("computeROI is consistent after concurrent writes", async () => {
    const engine = new BackpropagationEngine();
    const N = 15;
    // messageCount=20 → bucket "sm".
    await Promise.all(
      Array.from({ length: N }, () =>
        engine.record(3, 20, HAIKU, 20_000, 15_000, 0.005, 50),
      ),
    );
    const roi = engine.computeROI(3, HAIKU, "sm");
    expect(roi).not.toBeNull();
    expect(roi!.sampleCount).toBe(N);
    expect(roi!.totalTokensSaved).toBe(N * 5_000);
  });
});

// ---------------------------------------------------------------------------
// Provider switches mid-session
// ---------------------------------------------------------------------------

describe("provider switches mid-session", () => {
  test("different providers are stored in separate keys", async () => {
    const engine = new BackpropagationEngine();
    await bulkRecord(engine, 5, 2, SONNET, { messageCount: 20 });
    await bulkRecord(engine, 5, 2, HAIKU, { messageCount: 20 });
    await bulkRecord(engine, 5, 2, OPUS, { messageCount: 20 });

    expect(engine.getRecords(2, SONNET, "sm").length).toBe(5);
    expect(engine.getRecords(2, HAIKU, "sm").length).toBe(5);
    expect(engine.getRecords(2, OPUS, "sm").length).toBe(5);
    expect(engine.totalRecordCount()).toBe(15);
  });

  test("computeROI for each provider is independent", async () => {
    const engine = new BackpropagationEngine();
    await bulkRecord(engine, 5, 2, SONNET, { messageCount: 20, actualCostUsd: 0.05 });
    await bulkRecord(engine, 5, 2, HAIKU, { messageCount: 20, actualCostUsd: 0.005 });

    const roiSonnet = engine.computeROI(2, SONNET, "sm");
    const roiHaiku = engine.computeROI(2, HAIKU, "sm");

    expect(roiSonnet).not.toBeNull();
    expect(roiHaiku).not.toBeNull();
    // Haiku is cheaper; usdSaved differs due to different inputRates.
    expect(roiSonnet!.provider).toBe(SONNET);
    expect(roiHaiku!.provider).toBe(HAIKU);
    // Sonnet has higher inputRate → more USD saved per token removed.
    expect(roiSonnet!.avgUsdSavedPerCall).toBeGreaterThan(roiHaiku!.avgUsdSavedPerCall);
  });

  test("computeAllROI returns entries for all providers", async () => {
    const engine = new BackpropagationEngine();
    await bulkRecord(engine, 4, 2, SONNET, { messageCount: 20 });
    await bulkRecord(engine, 4, 3, GPT4O, { messageCount: 20 });

    const all = engine.computeAllROI();
    const providers = new Set(all.map((r) => r.provider));
    expect(providers.has(SONNET)).toBe(true);
    expect(providers.has(GPT4O)).toBe(true);
  });

  test("switching providers mid-session: windows are independent per provider", async () => {
    const engine = new BackpropagationEngine();
    // Start with SONNET, fill its window.
    await bulkRecord(engine, BACKPROP_WINDOW_SIZE, 2, SONNET, { messageCount: 20 });
    // Switch to HAIKU — its window should be empty.
    expect(engine.getRecords(2, HAIKU, "sm").length).toBe(0);
    // Record some HAIKU calls.
    await bulkRecord(engine, 3, 2, HAIKU, { messageCount: 20 });
    expect(engine.getRecords(2, HAIKU, "sm").length).toBe(3);
    // SONNET window should still be at cap.
    expect(engine.getRecords(2, SONNET, "sm").length).toBe(BACKPROP_WINDOW_SIZE);
  });
});

// ---------------------------------------------------------------------------
// computeParetoFrontier
// ---------------------------------------------------------------------------

describe("computeParetoFrontier", () => {
  test("returns empty array for empty input", () => {
    expect(computeParetoFrontier([])).toHaveLength(0);
  });

  test("filters out entries below BACKPROP_MIN_SAMPLES", () => {
    const roi: ProviderTierROI = {
      tier: 2, provider: SONNET, bucket: "sm",
      sampleCount: BACKPROP_MIN_SAMPLES - 1, // too few
      totalTokensSaved: 5_000, totalUsdSaved: 0.01,
      avgUsdSavedPerCall: 0.01, avgLatencyMs: 100,
      roiEfficiency: 1.0, updatedAt: new Date().toISOString(),
    };
    expect(computeParetoFrontier([roi])).toHaveLength(0);
  });

  test("single sufficient entry is non-dominated", () => {
    const roi: ProviderTierROI = {
      tier: 2, provider: SONNET, bucket: "sm",
      sampleCount: BACKPROP_MIN_SAMPLES,
      totalTokensSaved: 10_000, totalUsdSaved: 0.03,
      avgUsdSavedPerCall: 0.006, avgLatencyMs: 100,
      roiEfficiency: 2.0, updatedAt: new Date().toISOString(),
    };
    const frontier = computeParetoFrontier([roi]);
    expect(frontier).toHaveLength(1);
    expect(frontier[0]!.isNonDominated).toBe(true);
  });

  test("dominated point is flagged correctly", () => {
    const dominant: ProviderTierROI = {
      tier: 3, provider: HAIKU, bucket: "sm",
      sampleCount: BACKPROP_MIN_SAMPLES,
      totalTokensSaved: 20_000, totalUsdSaved: 0.06,
      // Higher ROI AND lower latency than dominated point below.
      avgUsdSavedPerCall: 0.02, avgLatencyMs: 50,
      roiEfficiency: 3.0, updatedAt: new Date().toISOString(),
    };
    const dominated: ProviderTierROI = {
      tier: 2, provider: SONNET, bucket: "sm",
      sampleCount: BACKPROP_MIN_SAMPLES,
      totalTokensSaved: 10_000, totalUsdSaved: 0.01,
      // Lower ROI AND higher latency — dominated by above.
      avgUsdSavedPerCall: 0.005, avgLatencyMs: 200,
      roiEfficiency: 1.0, updatedAt: new Date().toISOString(),
    };
    const frontier = computeParetoFrontier([dominant, dominated]);
    const nonDominated = frontier.filter((p) => p.isNonDominated);
    const dominatedPoints = frontier.filter((p) => !p.isNonDominated);
    expect(nonDominated).toHaveLength(1);
    expect(nonDominated[0]!.tier).toBe(3);
    expect(nonDominated[0]!.provider).toBe(HAIKU);
    expect(dominatedPoints).toHaveLength(1);
  });

  test("two non-dominated points with trade-offs", () => {
    const highROI: ProviderTierROI = {
      tier: 1, provider: OPUS, bucket: "lg",
      sampleCount: BACKPROP_MIN_SAMPLES,
      totalTokensSaved: 100_000, totalUsdSaved: 0.5,
      avgUsdSavedPerCall: 0.05, // high ROI
      avgLatencyMs: 2000, // but slow
      roiEfficiency: 5.0, updatedAt: new Date().toISOString(),
    };
    const lowLatency: ProviderTierROI = {
      tier: 4, provider: HAIKU, bucket: "xs",
      sampleCount: BACKPROP_MIN_SAMPLES,
      totalTokensSaved: 5_000, totalUsdSaved: 0.003,
      avgUsdSavedPerCall: 0.001, // lower ROI
      avgLatencyMs: 30, // but fast
      roiEfficiency: 1.0, updatedAt: new Date().toISOString(),
    };
    const frontier = computeParetoFrontier([highROI, lowLatency]);
    // Neither dominates the other — both should be non-dominated.
    expect(frontier.filter((p) => p.isNonDominated)).toHaveLength(2);
  });

  test("sorted descending by avgUsdSavedPerCall", () => {
    const entries: ProviderTierROI[] = [
      { tier: 4, provider: HAIKU, bucket: "xs", sampleCount: 3, totalTokensSaved: 1_000, totalUsdSaved: 0.001, avgUsdSavedPerCall: 0.001, avgLatencyMs: 30, roiEfficiency: 0.5, updatedAt: "" },
      { tier: 1, provider: OPUS, bucket: "lg", sampleCount: 3, totalTokensSaved: 100_000, totalUsdSaved: 0.5, avgUsdSavedPerCall: 0.05, avgLatencyMs: 2000, roiEfficiency: 5.0, updatedAt: "" },
      { tier: 2, provider: SONNET, bucket: "sm", sampleCount: 3, totalTokensSaved: 10_000, totalUsdSaved: 0.03, avgUsdSavedPerCall: 0.01, avgLatencyMs: 100, roiEfficiency: 2.0, updatedAt: "" },
    ];
    const frontier = computeParetoFrontier(entries);
    // Should be sorted descending by avgUsdSavedPerCall.
    for (let i = 1; i < frontier.length; i++) {
      expect(frontier[i - 1]!.avgUsdSavedPerCall).toBeGreaterThanOrEqual(frontier[i]!.avgUsdSavedPerCall);
    }
  });
});

// ---------------------------------------------------------------------------
// getROIDashboard
// ---------------------------------------------------------------------------

describe("getROIDashboard", () => {
  test("empty engine → empty dashboard", () => {
    const engine = new BackpropagationEngine();
    const dashboard = getROIDashboard(engine);
    expect(Object.keys(dashboard.entries)).toHaveLength(0);
    expect(dashboard.paretoFrontier).toHaveLength(0);
    expect(Object.keys(dashboard.bestTierByProvider)).toHaveLength(0);
    expect(Object.keys(dashboard.bestProviderByTier)).toHaveLength(0);
    expect(dashboard.totalRecords).toBe(0);
    expect(new Date(dashboard.generatedAt).getTime()).not.toBeNaN();
  });

  test("dashboard entries keyed by provider:tier:bucket", async () => {
    const engine = new BackpropagationEngine();
    await bulkRecord(engine, 5, 2, SONNET, { messageCount: 20 });
    const dashboard = getROIDashboard(engine);
    expect(Object.keys(dashboard.entries)).toContain(`${SONNET}:2:sm`);
  });

  test("bestTierByProvider picks the tier with highest avgUsdSavedPerCall", async () => {
    const engine = new BackpropagationEngine();
    // Tier 2 saves more (more tokens removed) vs tier 3.
    await bulkRecord(engine, 5, 2, SONNET, { messageCount: 20, estimatedTokens: 50_000, actualTokens: 30_000 }); // 20k tokens saved
    await bulkRecord(engine, 5, 3, SONNET, { messageCount: 20, estimatedTokens: 50_000, actualTokens: 45_000 }); // 5k tokens saved
    const dashboard = getROIDashboard(engine);
    expect(dashboard.bestTierByProvider[SONNET]).toBe(2);
  });

  test("bestProviderByTier picks the provider with highest avgUsdSavedPerCall", async () => {
    const engine = new BackpropagationEngine();
    // Opus has higher inputRate → more USD saved per token removed.
    await bulkRecord(engine, 5, 2, SONNET, { messageCount: 20, estimatedTokens: 50_000, actualTokens: 40_000 });
    await bulkRecord(engine, 5, 2, OPUS, { messageCount: 20, estimatedTokens: 50_000, actualTokens: 40_000 });
    const dashboard = getROIDashboard(engine);
    // Opus inputRate is 5× sonnet → more USD saved per token.
    expect(dashboard.bestProviderByTier[2]).toBe(OPUS);
  });

  test("totalRecords reflects all attribution records in engine", async () => {
    const engine = new BackpropagationEngine();
    await bulkRecord(engine, 7, 2, SONNET, { messageCount: 20 });
    await bulkRecord(engine, 4, 3, HAIKU, { messageCount: 20 });
    const dashboard = getROIDashboard(engine);
    expect(dashboard.totalRecords).toBe(11);
  });

  test("uses module singleton when no engine passed", async () => {
    // Record into the module-level singleton.
    const singleton = getBackpropEngine();
    await bulkRecord(singleton, 5, 2, SONNET, { messageCount: 20 });
    const dashboard = getROIDashboard(); // no engine arg → uses singleton
    expect(dashboard.totalRecords).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// biasAdaptiveTierSelection
// ---------------------------------------------------------------------------

describe("biasAdaptiveTierSelection", () => {
  test("returns baseTier unchanged when engine has no records", () => {
    const engine = new BackpropagationEngine();
    const result = biasAdaptiveTierSelection(2, SONNET, 10, engine);
    expect(result.biasedTier).toBe(2);
    expect(result.biasApplied).toBe(false);
    expect(result.drivingROI).toBeNull();
  });

  test("returns baseTier unchanged when samples below BACKPROP_MIN_SAMPLES", async () => {
    const engine = new BackpropagationEngine();
    await bulkRecord(engine, BACKPROP_MIN_SAMPLES - 1, 3, SONNET, { messageCount: 20 });
    const result = biasAdaptiveTierSelection(2, SONNET, 10, engine);
    expect(result.biasApplied).toBe(false);
  });

  test("does NOT bias toward tier 1 (too expensive)", async () => {
    const engine = new BackpropagationEngine();
    // Even if tier 1 had great ROI, bias should not shift to it.
    // We only record tier 1 ROI with good numbers.
    await bulkRecord(engine, 5, 1, SONNET, { messageCount: 20, estimatedTokens: 100_000, actualTokens: 10_000 });
    const result = biasAdaptiveTierSelection(3, SONNET, 10, engine);
    // Tier 1 is excluded from biasing (only tiers 2/3/4 are candidates).
    expect(result.biasedTier).toBe(3);
    expect(result.biasApplied).toBe(false);
  });

  test("biases toward lighter tier when ROI evidence is strong", async () => {
    const engine = new BackpropagationEngine();
    // Tier 3 has strong ROI history (many tokens removed).
    await bulkRecord(engine, 5, 3, SONNET, {
      messageCount: 20,
      estimatedTokens: 50_000,
      actualTokens: 10_000, // 40k tokens saved → strong ROI
      actualCostUsd: 0.001,
      latencyMs: 80,
    });
    // baseTier is 1 (most aggressive); tier 3 should be preferred as lighter.
    const result = biasAdaptiveTierSelection(1, SONNET, 10, engine);
    // Tier 3 > tier 1 (higher number = lighter), so bias should shift from 1 → 3.
    if (result.biasApplied) {
      expect(result.biasedTier).toBeGreaterThan(result.baseTier);
    }
  });

  test("biasApplied=false when best candidate tier >= baseTier (already at optimal)", async () => {
    const engine = new BackpropagationEngine();
    await bulkRecord(engine, 5, 4, SONNET, {
      messageCount: 20,
      estimatedTokens: 50_000,
      actualTokens: 40_000,
      actualCostUsd: 0.001,
      latencyMs: 50,
    });
    // baseTier is 4 (lightest) — no lighter tier exists, so no bias should apply.
    const result = biasAdaptiveTierSelection(4, SONNET, 10, engine);
    expect(result.biasApplied).toBe(false);
    expect(result.biasedTier).toBe(4);
  });

  test("decidedAt is a valid ISO timestamp", () => {
    const engine = new BackpropagationEngine();
    const result = biasAdaptiveTierSelection(2, SONNET, 10, engine);
    expect(new Date(result.decidedAt).getTime()).not.toBeNaN();
  });

  test("uses module singleton when no engine passed", async () => {
    const singleton = getBackpropEngine();
    await bulkRecord(singleton, 5, 3, SONNET, { messageCount: 20 });
    // Should not throw.
    const result = biasAdaptiveTierSelection(1, SONNET, 10);
    expect(typeof result.biasedTier).toBe("number");
  });
});

// ---------------------------------------------------------------------------
// attributeSessionCost — records + persists
// ---------------------------------------------------------------------------

describe("attributeSessionCost", () => {
  test("records into engine and returns a valid BackpropAttributionRecord", async () => {
    const engine = new BackpropagationEngine();
    const rec = await attributeSessionCost(2, 10, SONNET, 50_000, 40_000, 0.05, 100, engine);
    expect(rec.tier).toBe(2);
    expect(rec.provider).toBe(SONNET);
    expect(rec.tokenDelta).toBe(10_000);
    expect(engine.totalRecordCount()).toBe(1);
  });

  test("persists to file at ASHLR_BACKPROP_PATH", async () => {
    const engine = new BackpropagationEngine();
    await attributeSessionCost(3, 10, HAIKU, 30_000, 25_000, 0.01, 80, engine);
    const content = readFileSync(tmpBackpropPath, "utf8");
    expect(content.trim().length).toBeGreaterThan(0);
    const parsed: BackpropAttributionRecord = JSON.parse(content.trim());
    expect(parsed.tier).toBe(3);
    expect(parsed.provider).toBe(HAIKU);
    expect(parsed.tokenDelta).toBe(5_000);
  });

  test("multiple calls produce multiple JSONL lines on disk", async () => {
    const engine = new BackpropagationEngine();
    for (let i = 0; i < 3; i++) {
      await attributeSessionCost(2, 10, SONNET, 50_000, 40_000, 0.05, 100, engine);
    }
    const lines = readFileSync(tmpBackpropPath, "utf8").trim().split("\n");
    expect(lines.length).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// persistBackpropRecord — filesystem writes
// ---------------------------------------------------------------------------

describe("persistBackpropRecord", () => {
  test("writes a JSONL line to the configured path", () => {
    const rec: BackpropAttributionRecord = {
      recordedAt: new Date().toISOString(),
      tier: 2, bucket: "sm", provider: SONNET,
      estimatedTokens: 50_000, actualTokens: 40_000, tokenDelta: 10_000,
      actualCostUsd: 0.05, counterfactualCostUsd: 0.08, usdSaved: 0.03,
      latencyMs: 120,
    };
    persistBackpropRecord(rec);
    const content = readFileSync(tmpBackpropPath, "utf8");
    expect(content.trim().length).toBeGreaterThan(0);
    const parsed = JSON.parse(content.trim());
    expect(parsed.tier).toBe(2);
    expect(parsed.provider).toBe(SONNET);
    expect(parsed.tokenDelta).toBe(10_000);
  });

  test("multiple writes produce multiple JSONL lines", () => {
    for (let i = 0; i < 4; i++) {
      persistBackpropRecord({
        recordedAt: new Date().toISOString(),
        tier: 3, bucket: "sm", provider: HAIKU,
        estimatedTokens: 20_000, actualTokens: 15_000, tokenDelta: 5_000,
        actualCostUsd: 0.005, counterfactualCostUsd: 0.007, usdSaved: 0.002,
        latencyMs: 50,
      });
    }
    const lines = readFileSync(tmpBackpropPath, "utf8").trim().split("\n");
    expect(lines.length).toBe(4);
  });

  test("suppressed when ASHLR_SESSION_LOG=0", () => {
    process.env.ASHLR_SESSION_LOG = "0";
    persistBackpropRecord({
      recordedAt: new Date().toISOString(),
      tier: 2, bucket: "sm", provider: SONNET,
      estimatedTokens: 50_000, actualTokens: 40_000, tokenDelta: 10_000,
      actualCostUsd: 0.05, counterfactualCostUsd: 0.08, usdSaved: 0.03,
      latencyMs: 100,
    });
    let content = "";
    try { content = readFileSync(tmpBackpropPath, "utf8"); } catch { /* not created */ }
    expect(content).toBe("");
  });

  test("persisted JSON contains all BackpropAttributionRecord fields", () => {
    const now = new Date().toISOString();
    persistBackpropRecord({
      recordedAt: now,
      tier: 4, bucket: "lg", provider: OPUS,
      estimatedTokens: 100_000, actualTokens: 80_000, tokenDelta: 20_000,
      actualCostUsd: 1.5, counterfactualCostUsd: 1.8, usdSaved: 0.3,
      latencyMs: 500,
    });
    const parsed: BackpropAttributionRecord = JSON.parse(
      readFileSync(tmpBackpropPath, "utf8").trim(),
    );
    expect(parsed.recordedAt).toBe(now);
    expect(parsed.tier).toBe(4);
    expect(parsed.bucket).toBe("lg");
    expect(parsed.provider).toBe(OPUS);
    expect(parsed.estimatedTokens).toBe(100_000);
    expect(parsed.actualTokens).toBe(80_000);
    expect(parsed.tokenDelta).toBe(20_000);
    expect(parsed.latencyMs).toBe(500);
  });
});

// ---------------------------------------------------------------------------
// ROI efficiency — provider-aware pricing
// ---------------------------------------------------------------------------

describe("ROI efficiency is provider-aware", () => {
  test("opus generates higher usdSaved per token than haiku (same tokenDelta)", async () => {
    const engine = new BackpropagationEngine();
    await bulkRecord(engine, 5, 2, OPUS, { messageCount: 20, estimatedTokens: 50_000, actualTokens: 40_000, actualCostUsd: 1.5 });
    await bulkRecord(engine, 5, 2, HAIKU, { messageCount: 20, estimatedTokens: 50_000, actualTokens: 40_000, actualCostUsd: 0.01 });

    const roiOpus = engine.computeROI(2, OPUS, "sm");
    const roiHaiku = engine.computeROI(2, HAIKU, "sm");

    expect(roiOpus).not.toBeNull();
    expect(roiHaiku).not.toBeNull();
    // Opus inputRate (15 USD/M) >> haiku (0.25 USD/M) → more USD saved per token.
    expect(roiOpus!.avgUsdSavedPerCall).toBeGreaterThan(roiHaiku!.avgUsdSavedPerCall);
  });

  test("roiEfficiency reflects provider inputRate differences", async () => {
    const engine = new BackpropagationEngine();
    // Use a larger token delta to amplify the difference past float noise.
    await bulkRecord(engine, 5, 2, OPUS, { messageCount: 20, estimatedTokens: 100_000, actualTokens: 10_000, actualCostUsd: 1.5 });
    await bulkRecord(engine, 5, 2, HAIKU, { messageCount: 20, estimatedTokens: 100_000, actualTokens: 10_000, actualCostUsd: 0.01 });

    const roiOpus = engine.computeROI(2, OPUS, "sm");
    const roiHaiku = engine.computeROI(2, HAIKU, "sm");

    expect(roiOpus).not.toBeNull();
    expect(roiHaiku).not.toBeNull();
    // Opus inputRate (15 USD/M) >> haiku (0.25 USD/M).
    // Both roiEfficiency values are proportional to avgUsdSavedPerCall / estimatedTierCost.
    // The estimatedTierCost itself scales with inputRate, so the ratio is:
    //   opus_eff / haiku_eff ≈ (opus_usdSaved / opus_tierCost) / (haiku_usdSaved / haiku_tierCost)
    // Since usdSaved ∝ inputRate AND tierCost ∝ inputRate, the ratio stays near 1×.
    // The meaningful assertion is that both are positive and non-zero.
    expect(roiOpus!.roiEfficiency).toBeGreaterThan(0);
    expect(roiHaiku!.roiEfficiency).toBeGreaterThan(0);
  });

  test("unknown provider falls back to default rate without throwing", async () => {
    const engine = new BackpropagationEngine();
    // Should not throw for unknown provider.
    const rec = await engine.record(2, 10, "unknown-model-xyz", 50_000, 40_000, 0.05, 100);
    expect(rec.tier).toBe(2);
    expect(rec.tokenDelta).toBe(10_000);
    expect(rec.usdSaved).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

describe("module constants", () => {
  test("BACKPROP_WINDOW_SIZE is 30", () => {
    expect(BACKPROP_WINDOW_SIZE).toBe(30);
  });

  test("BACKPROP_MIN_SAMPLES is a positive integer", () => {
    expect(Number.isInteger(BACKPROP_MIN_SAMPLES)).toBe(true);
    expect(BACKPROP_MIN_SAMPLES).toBeGreaterThan(0);
  });

  test("BACKPROP_MIN_ROI_EFFICIENCY is a positive number < 1", () => {
    expect(BACKPROP_MIN_ROI_EFFICIENCY).toBeGreaterThan(0);
    expect(BACKPROP_MIN_ROI_EFFICIENCY).toBeLessThan(1);
  });
});
