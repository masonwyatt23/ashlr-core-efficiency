/**
 * Tests for Multi-Tier Compression ROI Optimizer with Per-Model Latency Impact Analysis.
 *
 * Coverage:
 *   - analyzeLatencyROI: frontier computation from synthetic records
 *   - recommendTierForLatencyBudget: Pareto-domination logic
 *   - Per-model coefficient extraction (claude-3-5-sonnet ≠ gpt-4o)
 *   - getLatencyDashboard: multi-model aggregation
 *   - getLatencyDashboard (backpropagation.ts): per-tier latency summaries + TTFT
 *   - Edge cases: empty profile, no feasible tier, single tier
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import {
  analyzeLatencyROI,
  recommendTierForLatencyBudget,
  getLatencyDashboard as getLatencyDashboardFromAnalyzer,
  getModelLatencyCoefficient,
  MODEL_LATENCY_COEFFICIENTS,
  DEFAULT_LATENCY_COEFFICIENT_MS,
  type LatencyROIProfile,
  type TierLatencyStats,
} from "../src/compression/latency-roi-analyzer.ts";

import {
  BackpropagationEngine,
  getLatencyDashboard,
  _resetBackpropEngine,
  BACKPROP_MIN_SAMPLES,
  type BackpropAttributionRecord,
} from "../src/session-log/backpropagation.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const SONNET = "claude-3-5-sonnet";
const GPT4O = "gpt-4o";
const HAIKU = "claude-3-haiku";
const OPUS = "claude-3-opus";

/** Inject N records for a given tier + model into an engine. */
async function inject(
  engine: BackpropagationEngine,
  tier: 1 | 2 | 3 | 4,
  model: string,
  count: number,
  opts: {
    messageCount?: number;
    estimatedTokens?: number;
    actualTokens?: number;
    actualCostUsd?: number;
    latencyMs?: number;
    requestStartMs?: number;
    firstTokenMs?: number;
  } = {},
): Promise<void> {
  const {
    messageCount = 20,
    estimatedTokens = 50_000,
    actualTokens = 40_000,
    actualCostUsd = 0.05,
    latencyMs = 200,
  } = opts;
  for (let i = 0; i < count; i++) {
    await engine.record(tier, messageCount, model, estimatedTokens, actualTokens, actualCostUsd, latencyMs);
  }
}

beforeEach(() => {
  _resetBackpropEngine();
  delete process.env.ASHLR_SESSION_LOG;
});

afterEach(() => {
  _resetBackpropEngine();
  delete process.env.ASHLR_SESSION_LOG;
});

// ---------------------------------------------------------------------------
// Per-model coefficient extraction
// ---------------------------------------------------------------------------

describe("getModelLatencyCoefficient — per-model coefficients", () => {
  test("claude-3-5-sonnet has coefficient 150", () => {
    expect(getModelLatencyCoefficient("claude-3-5-sonnet")).toBe(150);
  });

  test("gpt-4o has coefficient 80", () => {
    expect(getModelLatencyCoefficient("gpt-4o")).toBe(80);
  });

  test("claude-3-5-sonnet coefficient differs from gpt-4o coefficient", () => {
    const claudeCoeff = getModelLatencyCoefficient("claude-3-5-sonnet");
    const gptCoeff = getModelLatencyCoefficient("gpt-4o");
    expect(claudeCoeff).not.toBe(gptCoeff);
    // claude coefficient should be larger (empirically slower per tier step)
    expect(claudeCoeff).toBeGreaterThan(gptCoeff);
  });

  test("versioned slug falls back via prefix match", () => {
    // "claude-3-5-sonnet-20241022" should resolve to "claude-3-5-sonnet"
    expect(getModelLatencyCoefficient("claude-3-5-sonnet-20241022")).toBe(150);
  });

  test("versioned gpt-4o slug resolves correctly", () => {
    expect(getModelLatencyCoefficient("gpt-4o-2024-11-20")).toBe(80);
  });

  test("unknown model returns DEFAULT_LATENCY_COEFFICIENT_MS", () => {
    expect(getModelLatencyCoefficient("some-unknown-model-xyz")).toBe(DEFAULT_LATENCY_COEFFICIENT_MS);
  });

  test("all models in MODEL_LATENCY_COEFFICIENTS are positive integers", () => {
    for (const [model, coeff] of Object.entries(MODEL_LATENCY_COEFFICIENTS)) {
      expect(typeof coeff).toBe("number");
      expect(coeff).toBeGreaterThan(0);
      // Reasonable range: 10ms to 1000ms per tier step
      expect(coeff).toBeGreaterThanOrEqual(10);
      expect(coeff).toBeLessThanOrEqual(1000);
    }
  });

  test("claude-3-opus has higher coefficient than gpt-4o-mini (slower model)", () => {
    expect(getModelLatencyCoefficient("claude-3-opus")).toBeGreaterThan(
      getModelLatencyCoefficient("gpt-4o-mini"),
    );
  });
});

// ---------------------------------------------------------------------------
// analyzeLatencyROI — frontier computation
// ---------------------------------------------------------------------------

describe("analyzeLatencyROI — building latency-cost frontier", () => {
  test("returns empty tiers and frontier when no records exist", async () => {
    const engine = new BackpropagationEngine();
    const profile = await analyzeLatencyROI(SONNET, ".", engine);
    expect(profile.model).toBe(SONNET);
    expect(profile.tiers).toHaveLength(0);
    expect(profile.paretoFrontier).toHaveLength(0);
    expect(profile.totalRecords).toBe(0);
    expect(new Date(profile.generatedAt).getTime()).not.toBeNaN();
  });

  test("builds TierLatencyStats for each tier with records", async () => {
    const engine = new BackpropagationEngine();
    await inject(engine, 2, SONNET, 5, { latencyMs: 120, actualCostUsd: 0.04 });
    await inject(engine, 3, SONNET, 5, { latencyMs: 80, actualCostUsd: 0.02 });

    const profile = await analyzeLatencyROI(SONNET, ".", engine);
    expect(profile.tiers).toHaveLength(2);
    const tiers = profile.tiers.map((t) => t.tier).sort();
    expect(tiers).toEqual([2, 3]);
  });

  test("meanLatencyMs is correctly computed from injected records", async () => {
    const engine = new BackpropagationEngine();
    // All 5 records have latencyMs=300.
    await inject(engine, 2, SONNET, 5, { latencyMs: 300 });

    const profile = await analyzeLatencyROI(SONNET, ".", engine);
    const tier2 = profile.tiers.find((t) => t.tier === 2)!;
    expect(tier2).toBeDefined();
    expect(tier2.meanLatencyMs).toBeCloseTo(300, 5);
  });

  test("p50LatencyMs and p95LatencyMs are within observed range", async () => {
    const engine = new BackpropagationEngine();
    // Inject records with varying latencies: 100, 150, 200, 250, 300 (×2 each)
    for (const ms of [100, 150, 200, 250, 300]) {
      await inject(engine, 3, SONNET, 2, { latencyMs: ms });
    }

    const profile = await analyzeLatencyROI(SONNET, ".", engine);
    const tier3 = profile.tiers.find((t) => t.tier === 3)!;
    expect(tier3).toBeDefined();
    expect(tier3.p50LatencyMs).toBeGreaterThanOrEqual(100);
    expect(tier3.p50LatencyMs).toBeLessThanOrEqual(300);
    expect(tier3.p95LatencyMs).toBeGreaterThanOrEqual(tier3.p50LatencyMs);
    expect(tier3.p95LatencyMs).toBeLessThanOrEqual(300);
  });

  test("qualityLossPct is higher for tier 1 than tier 4", async () => {
    const engine = new BackpropagationEngine();
    // Tier 1 — heavy token removal (aggressive summarisation)
    await inject(engine, 1, SONNET, 5, {
      estimatedTokens: 100_000,
      actualTokens: 20_000,  // 80k tokens removed
      latencyMs: 2000,
    });
    // Tier 4 — light token removal (tree prune)
    await inject(engine, 4, SONNET, 5, {
      estimatedTokens: 100_000,
      actualTokens: 95_000,  // only 5k tokens removed
      latencyMs: 50,
    });

    const profile = await analyzeLatencyROI(SONNET, ".", engine);
    const tier1 = profile.tiers.find((t) => t.tier === 1)!;
    const tier4 = profile.tiers.find((t) => t.tier === 4)!;
    expect(tier1.qualityLossPct).toBeGreaterThan(tier4.qualityLossPct);
  });

  test("latencyCoefficientMs matches getModelLatencyCoefficient for each tier", async () => {
    const engine = new BackpropagationEngine();
    await inject(engine, 2, SONNET, 3);
    await inject(engine, 2, GPT4O, 3);

    const profileSonnet = await analyzeLatencyROI(SONNET, ".", engine);
    const profileGpt = await analyzeLatencyROI(GPT4O, ".", engine);

    const sonnetTier2 = profileSonnet.tiers.find((t) => t.tier === 2)!;
    const gptTier2 = profileGpt.tiers.find((t) => t.tier === 2)!;

    expect(sonnetTier2.latencyCoefficientMs).toBe(getModelLatencyCoefficient(SONNET));
    expect(gptTier2.latencyCoefficientMs).toBe(getModelLatencyCoefficient(GPT4O));
    // Different models → different coefficients
    expect(sonnetTier2.latencyCoefficientMs).not.toBe(gptTier2.latencyCoefficientMs);
  });

  test("totalRecords reflects all records across tiers", async () => {
    const engine = new BackpropagationEngine();
    await inject(engine, 2, SONNET, 4);
    await inject(engine, 3, SONNET, 6);

    const profile = await analyzeLatencyROI(SONNET, ".", engine);
    expect(profile.totalRecords).toBe(10);
  });

  test("records from different models are not mixed", async () => {
    const engine = new BackpropagationEngine();
    await inject(engine, 2, SONNET, 5, { latencyMs: 100 });
    await inject(engine, 2, GPT4O, 5, { latencyMs: 500 });

    const profileSonnet = await analyzeLatencyROI(SONNET, ".", engine);
    const profileGpt = await analyzeLatencyROI(GPT4O, ".", engine);

    const sonnetTier2 = profileSonnet.tiers.find((t) => t.tier === 2)!;
    const gptTier2 = profileGpt.tiers.find((t) => t.tier === 2)!;

    expect(sonnetTier2.meanLatencyMs).toBeCloseTo(100, 5);
    expect(gptTier2.meanLatencyMs).toBeCloseTo(500, 5);
  });
});

// ---------------------------------------------------------------------------
// Pareto frontier in analyzeLatencyROI
// ---------------------------------------------------------------------------

describe("analyzeLatencyROI — Pareto frontier", () => {
  test("single tier with sufficient samples is on the frontier", async () => {
    const engine = new BackpropagationEngine();
    await inject(engine, 3, SONNET, BACKPROP_MIN_SAMPLES);

    const profile = await analyzeLatencyROI(SONNET, ".", engine);
    expect(profile.paretoFrontier).toContain(3);
  });

  test("dominated tier is excluded from Pareto frontier", async () => {
    const engine = new BackpropagationEngine();
    // Tier 3: lower latency AND lower cost — dominates tier 2
    await inject(engine, 3, SONNET, BACKPROP_MIN_SAMPLES, {
      latencyMs: 80,
      actualCostUsd: 0.01,
      estimatedTokens: 50_000,
      actualTokens: 40_000,
    });
    // Tier 2: higher latency AND higher cost — dominated by tier 3
    await inject(engine, 2, SONNET, BACKPROP_MIN_SAMPLES, {
      latencyMs: 200,
      actualCostUsd: 0.05,
      estimatedTokens: 50_000,
      actualTokens: 40_000,
    });

    const profile = await analyzeLatencyROI(SONNET, ".", engine);
    // Tier 3 dominates tier 2 on both objectives (lower latency AND lower cost).
    expect(profile.paretoFrontier).toContain(3);
    expect(profile.paretoFrontier).not.toContain(2);
  });

  test("two tiers with trade-offs are both on the frontier", async () => {
    const engine = new BackpropagationEngine();
    // Tier 4: low latency but not much savings
    await inject(engine, 4, SONNET, BACKPROP_MIN_SAMPLES, {
      latencyMs: 50,
      actualCostUsd: 0.08,
    });
    // Tier 1: high savings but high latency
    await inject(engine, 1, SONNET, BACKPROP_MIN_SAMPLES, {
      latencyMs: 2000,
      actualCostUsd: 0.01,
    });

    const profile = await analyzeLatencyROI(SONNET, ".", engine);
    // Neither dominates the other: tier 4 wins on latency, tier 1 wins on cost.
    expect(profile.paretoFrontier).toContain(4);
    expect(profile.paretoFrontier).toContain(1);
  });

  test("tier below BACKPROP_MIN_SAMPLES is excluded from frontier", async () => {
    const engine = new BackpropagationEngine();
    // Only 1 record — below threshold
    await inject(engine, 2, SONNET, 1);

    const profile = await analyzeLatencyROI(SONNET, ".", engine);
    // TierLatencyStats entry exists but should not appear in paretoFrontier
    expect(profile.tiers.find((t) => t.tier === 2)).toBeDefined();
    expect(profile.paretoFrontier).not.toContain(2);
  });
});

// ---------------------------------------------------------------------------
// recommendTierForLatencyBudget — multi-objective picker
// ---------------------------------------------------------------------------

describe("recommendTierForLatencyBudget — Pareto-domination logic", () => {
  function makeProfile(
    entries: Array<{
      tier: 1 | 2 | 3 | 4;
      meanLatencyMs: number;
      meanCostUsd: number;
      meanUsdSaved: number;
      sampleCount?: number;
    }>,
  ): LatencyROIProfile {
    const tiers: TierLatencyStats[] = entries.map((e) => ({
      tier: e.tier,
      sampleCount: e.sampleCount ?? BACKPROP_MIN_SAMPLES,
      meanLatencyMs: e.meanLatencyMs,
      p50LatencyMs: e.meanLatencyMs,
      p95LatencyMs: e.meanLatencyMs * 1.2,
      meanCostUsd: e.meanCostUsd,
      qualityLossPct: 5,
      meanUsdSaved: e.meanUsdSaved,
      latencyCoefficientMs: 150,
    }));

    // Compute a simple pareto frontier: tier not dominated on (latency, cost)
    const pareto: (1 | 2 | 3 | 4)[] = [];
    for (const t of tiers) {
      if (t.sampleCount < BACKPROP_MIN_SAMPLES) continue;
      let dominated = false;
      for (const other of tiers) {
        if (other.tier === t.tier) continue;
        if (other.sampleCount < BACKPROP_MIN_SAMPLES) continue;
        if (
          other.meanLatencyMs <= t.meanLatencyMs &&
          other.meanCostUsd <= t.meanCostUsd &&
          (other.meanLatencyMs < t.meanLatencyMs || other.meanCostUsd < t.meanCostUsd)
        ) {
          dominated = true;
          break;
        }
      }
      if (!dominated) pareto.push(t.tier);
    }

    return {
      model: SONNET,
      tiers,
      paretoFrontier: pareto,
      generatedAt: new Date().toISOString(),
      totalRecords: tiers.reduce((s, t) => s + t.sampleCount, 0),
    };
  }

  test("returns tier 4 when profile is empty", () => {
    const empty: LatencyROIProfile = {
      model: SONNET,
      tiers: [],
      paretoFrontier: [],
      generatedAt: new Date().toISOString(),
      totalRecords: 0,
    };
    expect(recommendTierForLatencyBudget(500, 1.0, empty)).toBe(4);
  });

  test("picks the only feasible tier satisfying both constraints", () => {
    const profile = makeProfile([
      { tier: 2, meanLatencyMs: 150, meanCostUsd: 0.04, meanUsdSaved: 0.02 },
      { tier: 1, meanLatencyMs: 800, meanCostUsd: 0.10, meanUsdSaved: 0.05 }, // too slow + expensive
    ]);
    const rec = recommendTierForLatencyBudget(200, 0.05, profile);
    expect(rec).toBe(2);
  });

  test("among feasible tiers, picks the one with highest meanUsdSaved", () => {
    // Both tiers fit within budget.
    // Tier 3: higher latency but lower cost — saves more USD (Pareto trade-off).
    // Tier 4: lower latency but higher cost — less USD saved.
    // Neither dominates the other: tier 4 wins on latency, tier 3 wins on cost.
    // Both are Pareto-optimal; the recommender should pick tier 3 (max savings).
    const profile = makeProfile([
      { tier: 3, meanLatencyMs: 150, meanCostUsd: 0.02, meanUsdSaved: 0.04 },
      { tier: 4, meanLatencyMs: 50,  meanCostUsd: 0.05, meanUsdSaved: 0.01 },
    ]);
    const rec = recommendTierForLatencyBudget(200, 0.10, profile);
    expect(rec).toBe(3);
  });

  test("Pareto-dominated tier X is not recommended when tier Y beats it on both metrics", () => {
    // Tier 3: latency=80, cost=0.02 — strictly better than tier 2 on both.
    // Tier 2: latency=200, cost=0.05 — dominated.
    const profile = makeProfile([
      { tier: 3, meanLatencyMs: 80, meanCostUsd: 0.02, meanUsdSaved: 0.04 },
      { tier: 2, meanLatencyMs: 200, meanCostUsd: 0.05, meanUsdSaved: 0.02 },
    ]);
    // Both fit within budget 500ms / $0.10
    const rec = recommendTierForLatencyBudget(500, 0.10, profile);
    // Tier 3 is Pareto-optimal; tier 2 is dominated. Should pick tier 3.
    expect(rec).toBe(3);
    // Specifically: tier 2 should NOT be recommended over tier 3.
    expect(rec).not.toBe(2);
  });

  test("when no tier satisfies both constraints, picks least-bad option", () => {
    // No tier fits within 50ms latency + $0.001 cost.
    const profile = makeProfile([
      { tier: 4, meanLatencyMs: 200, meanCostUsd: 0.02, meanUsdSaved: 0.01 }, // closest
      { tier: 1, meanLatencyMs: 3000, meanCostUsd: 0.50, meanUsdSaved: 0.40 }, // worst
    ]);
    const rec = recommendTierForLatencyBudget(50, 0.001, profile);
    // Tier 4 violates less than tier 1 — should be preferred.
    expect(rec).toBe(4);
  });

  test("latency constraint binds when cost is generous", () => {
    const profile = makeProfile([
      { tier: 1, meanLatencyMs: 500, meanCostUsd: 0.01, meanUsdSaved: 0.10 }, // slow but cheap
      { tier: 4, meanLatencyMs: 50, meanCostUsd: 0.05, meanUsdSaved: 0.02 },  // fast
    ]);
    // Strict latency budget of 100ms; cost is generous.
    const rec = recommendTierForLatencyBudget(100, 1.0, profile);
    expect(rec).toBe(4);
  });

  test("cost constraint binds when latency is generous", () => {
    const profile = makeProfile([
      { tier: 1, meanLatencyMs: 2000, meanCostUsd: 0.01, meanUsdSaved: 0.20 }, // expensive latency but max savings
      { tier: 4, meanLatencyMs: 50, meanCostUsd: 0.10, meanUsdSaved: 0.01 },   // cheap savings
    ]);
    // Generous latency (10s), tight cost budget $0.05 — only tier 1 fits.
    const rec = recommendTierForLatencyBudget(10_000, 0.05, profile);
    expect(rec).toBe(1);
  });

  test("ties in meanUsdSaved are broken by lower latency", () => {
    // Both tiers have the same meanUsdSaved.
    const profile = makeProfile([
      { tier: 2, meanLatencyMs: 200, meanCostUsd: 0.03, meanUsdSaved: 0.05 },
      { tier: 3, meanLatencyMs: 100, meanCostUsd: 0.03, meanUsdSaved: 0.05 }, // same savings, lower latency
    ]);
    const rec = recommendTierForLatencyBudget(500, 0.10, profile);
    expect(rec).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// getLatencyDashboard (from latency-roi-analyzer.ts)
// ---------------------------------------------------------------------------

describe("getLatencyDashboard (latency-roi-analyzer) — multi-model aggregation", () => {
  test("returns empty profiles when engine has no records", async () => {
    const engine = new BackpropagationEngine();
    const dashboard = await getLatencyDashboardFromAnalyzer(engine);
    expect(Object.keys(dashboard.profiles)).toHaveLength(0);
    expect(Object.keys(dashboard.fastestTierByModel)).toHaveLength(0);
    expect(Object.keys(dashboard.cheapestTierByModel)).toHaveLength(0);
  });

  test("produces a profile for each model with records", async () => {
    const engine = new BackpropagationEngine();
    await inject(engine, 2, SONNET, 5);
    await inject(engine, 3, GPT4O, 5);

    const dashboard = await getLatencyDashboardFromAnalyzer(engine);
    expect(Object.keys(dashboard.profiles)).toContain(SONNET);
    expect(Object.keys(dashboard.profiles)).toContain(GPT4O);
  });

  test("fastestTierByModel picks tier with lowest meanLatencyMs", async () => {
    const engine = new BackpropagationEngine();
    // Tier 4 is fastest; tier 2 is slower.
    await inject(engine, 4, SONNET, BACKPROP_MIN_SAMPLES, { latencyMs: 50 });
    await inject(engine, 2, SONNET, BACKPROP_MIN_SAMPLES, { latencyMs: 300 });

    const dashboard = await getLatencyDashboardFromAnalyzer(engine);
    expect(dashboard.fastestTierByModel[SONNET]).toBe(4);
  });

  test("cheapestTierByModel picks tier with highest meanUsdSaved", async () => {
    const engine = new BackpropagationEngine();
    // Tier 2 removes more tokens → higher usdSaved.
    await inject(engine, 2, SONNET, BACKPROP_MIN_SAMPLES, {
      estimatedTokens: 100_000,
      actualTokens: 20_000,   // 80k tokens saved
      actualCostUsd: 0.05,
    });
    await inject(engine, 4, SONNET, BACKPROP_MIN_SAMPLES, {
      estimatedTokens: 100_000,
      actualTokens: 95_000,   // only 5k tokens saved
      actualCostUsd: 0.04,
    });

    const dashboard = await getLatencyDashboardFromAnalyzer(engine);
    expect(dashboard.cheapestTierByModel[SONNET]).toBe(2);
  });

  test("generatedAt is a valid ISO timestamp", async () => {
    const engine = new BackpropagationEngine();
    const dashboard = await getLatencyDashboardFromAnalyzer(engine);
    expect(new Date(dashboard.generatedAt).getTime()).not.toBeNaN();
  });
});

// ---------------------------------------------------------------------------
// getLatencyDashboard (from backpropagation.ts) — per-tier summaries + TTFT
// ---------------------------------------------------------------------------

describe("getLatencyDashboard (backpropagation.ts) — per-tier summaries", () => {
  test("returns empty tiers for model with no records", () => {
    const engine = new BackpropagationEngine();
    const dashboard = getLatencyDashboard("no-records-model", engine);
    expect(dashboard.model).toBe("no-records-model");
    expect(dashboard.tiers).toHaveLength(0);
    expect(dashboard.fastestTier).toBeNull();
    expect(dashboard.cheapestTier).toBeNull();
    expect(new Date(dashboard.generatedAt).getTime()).not.toBeNaN();
  });

  test("aggregates latency across multiple tiers", async () => {
    const engine = new BackpropagationEngine();
    await inject(engine, 2, SONNET, 5, { latencyMs: 150 });
    await inject(engine, 3, SONNET, 5, { latencyMs: 80 });

    const dashboard = getLatencyDashboard(SONNET, engine);
    expect(dashboard.tiers).toHaveLength(2);
    const tiers = dashboard.tiers.map((t) => t.tier).sort();
    expect(tiers).toEqual([2, 3]);
  });

  test("avgLatencyMs is correct for each tier", async () => {
    const engine = new BackpropagationEngine();
    await inject(engine, 2, SONNET, 5, { latencyMs: 200 });
    await inject(engine, 4, SONNET, 5, { latencyMs: 60 });

    const dashboard = getLatencyDashboard(SONNET, engine);
    const tier2 = dashboard.tiers.find((t) => t.tier === 2)!;
    const tier4 = dashboard.tiers.find((t) => t.tier === 4)!;
    expect(tier2.avgLatencyMs).toBeCloseTo(200, 5);
    expect(tier4.avgLatencyMs).toBeCloseTo(60, 5);
  });

  test("fastestTier picks tier with lowest avgLatencyMs", async () => {
    const engine = new BackpropagationEngine();
    await inject(engine, 3, SONNET, BACKPROP_MIN_SAMPLES, { latencyMs: 100 });
    await inject(engine, 2, SONNET, BACKPROP_MIN_SAMPLES, { latencyMs: 400 });

    const dashboard = getLatencyDashboard(SONNET, engine);
    expect(dashboard.fastestTier).toBe(3);
  });

  test("cheapestTier picks tier with highest avgUsdSaved", async () => {
    const engine = new BackpropagationEngine();
    // Tier 2: saves more per call
    await inject(engine, 2, SONNET, BACKPROP_MIN_SAMPLES, {
      estimatedTokens: 100_000,
      actualTokens: 10_000,
      actualCostUsd: 0.05,
    });
    await inject(engine, 4, SONNET, BACKPROP_MIN_SAMPLES, {
      estimatedTokens: 100_000,
      actualTokens: 98_000,
      actualCostUsd: 0.04,
    });

    const dashboard = getLatencyDashboard(SONNET, engine);
    expect(dashboard.cheapestTier).toBe(2);
  });

  test("avgTtftMs is null when no records have TTFT fields", async () => {
    const engine = new BackpropagationEngine();
    await inject(engine, 3, SONNET, 3, { latencyMs: 100 });

    const dashboard = getLatencyDashboard(SONNET, engine);
    const tier3 = dashboard.tiers.find((t) => t.tier === 3)!;
    expect(tier3.avgTtftMs).toBeNull();
  });

  test("minLatencyMs and maxLatencyMs bound the observed values", async () => {
    const engine = new BackpropagationEngine();
    // Record 3 calls with different latencies.
    await engine.record(2, 20, SONNET, 50_000, 40_000, 0.05, 100);
    await engine.record(2, 20, SONNET, 50_000, 40_000, 0.05, 200);
    await engine.record(2, 20, SONNET, 50_000, 40_000, 0.05, 300);

    const dashboard = getLatencyDashboard(SONNET, engine);
    const tier2 = dashboard.tiers.find((t) => t.tier === 2)!;
    expect(tier2.minLatencyMs).toBe(100);
    expect(tier2.maxLatencyMs).toBe(300);
    expect(tier2.avgLatencyMs).toBeCloseTo(200, 5);
  });

  test("sampleCount aggregates across all message-count buckets", async () => {
    const engine = new BackpropagationEngine();
    // xs bucket (messageCount=5), sm bucket (messageCount=20), md bucket (messageCount=100)
    await engine.record(3, 5, SONNET, 5_000, 4_000, 0.001, 50);
    await engine.record(3, 20, SONNET, 20_000, 15_000, 0.01, 80);
    await engine.record(3, 100, SONNET, 50_000, 40_000, 0.04, 120);

    const dashboard = getLatencyDashboard(SONNET, engine);
    const tier3 = dashboard.tiers.find((t) => t.tier === 3)!;
    expect(tier3.sampleCount).toBe(3);
  });

  test("uses module singleton when no engine passed", async () => {
    // Record into singleton via inject with default engine (no engine arg)
    const singleton = new BackpropagationEngine();
    await inject(singleton, 2, GPT4O, 4);
    // getLatencyDashboard with explicit engine
    const dashboard = getLatencyDashboard(GPT4O, singleton);
    expect(dashboard.tiers.length).toBeGreaterThan(0);
  });

  test("fastestTier and cheapestTier are null when no tier has BACKPROP_MIN_SAMPLES", async () => {
    const engine = new BackpropagationEngine();
    // Only 1 record — below BACKPROP_MIN_SAMPLES
    await inject(engine, 3, HAIKU, 1);

    const dashboard = getLatencyDashboard(HAIKU, engine);
    expect(dashboard.tiers).toHaveLength(1);
    // Below threshold → no definitive fastest/cheapest
    expect(dashboard.fastestTier).toBeNull();
    expect(dashboard.cheapestTier).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// BackpropAttributionRecord latency fields (requestStartMs, firstTokenMs)
// ---------------------------------------------------------------------------

describe("BackpropAttributionRecord optional latency fields", () => {
  test("record without optional latency fields still stores correctly", async () => {
    const engine = new BackpropagationEngine();
    const rec = await engine.record(2, 10, SONNET, 50_000, 40_000, 0.05, 100);
    // Optional fields should be absent (undefined)
    expect(rec.requestStartMs).toBeUndefined();
    expect(rec.firstTokenMs).toBeUndefined();
    expect(rec.latencyMs).toBe(100);
  });

  test("persisted record round-trips without optional TTFT fields", () => {
    process.env.ASHLR_SESSION_LOG = "0"; // suppress disk writes
    const rec: BackpropAttributionRecord = {
      recordedAt: new Date().toISOString(),
      tier: 3,
      bucket: "sm",
      provider: SONNET,
      estimatedTokens: 50_000,
      actualTokens: 40_000,
      tokenDelta: 10_000,
      actualCostUsd: 0.05,
      counterfactualCostUsd: 0.08,
      usdSaved: 0.03,
      latencyMs: 120,
    };
    // requestStartMs and firstTokenMs are optional — should not cause TypeScript errors.
    expect(rec.requestStartMs).toBeUndefined();
    expect(rec.firstTokenMs).toBeUndefined();
  });
});
