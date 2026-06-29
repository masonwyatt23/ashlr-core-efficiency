/**
 * Tests for BudgetMultiObjectiveLearner — Pareto frontier, budget fitting,
 * allocation recommendation, and feasibility validation.
 */

import { describe, expect, test, beforeEach } from "bun:test";
import { mkdtemp, rm } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";

import {
  BudgetMultiObjectiveLearner,
  computeParetoFrontier,
  bucketMessageCount,
} from "../src/budget/multi-objective-learner.ts";
import type {
  ParetoPoint,
  SessionSummary,
  BudgetAllocation,
} from "../src/budget/multi-objective-learner.ts";
import { applyAdaptiveBudget } from "../src/compression/context.ts";
import {
  SYSTEM_PROMPT_BUDGET_RATIO,
  SYSTEM_PROMPT_BUDGET_CAP,
} from "../src/budget/index.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeSession(
  overrides: Partial<SessionSummary> = {},
): SessionSummary {
  return {
    provider: "anthropic",
    modelId: "claude-sonnet-4",
    messageCount: 20,
    totalCostUsd: 0.05,
    p99LatencyMs: 600,
    systemPromptTokensUsed: 10_000,
    cacheWriteBreakpointFraction: 0.3,
    reserveTokensForResponse: 8192,
    recordedAt: new Date().toISOString(),
    ...overrides,
  };
}

function makeSessions(
  n: number,
  overrides: Partial<SessionSummary> = {},
): SessionSummary[] {
  return Array.from({ length: n }, (_, i) =>
    makeSession({
      ...overrides,
      recordedAt: new Date(Date.now() - i * 60_000).toISOString(),
    }),
  );
}

// ---------------------------------------------------------------------------
// bucketMessageCount
// ---------------------------------------------------------------------------

describe("bucketMessageCount", () => {
  test("xs: 1–10 messages", () => {
    expect(bucketMessageCount(1)).toBe("xs");
    expect(bucketMessageCount(10)).toBe("xs");
  });

  test("sm: 11–50 messages", () => {
    expect(bucketMessageCount(11)).toBe("sm");
    expect(bucketMessageCount(50)).toBe("sm");
  });

  test("md: 51–200 messages", () => {
    expect(bucketMessageCount(51)).toBe("md");
    expect(bucketMessageCount(200)).toBe("md");
  });

  test("lg: 201+ messages", () => {
    expect(bucketMessageCount(201)).toBe("lg");
    expect(bucketMessageCount(10_000)).toBe("lg");
  });
});

// ---------------------------------------------------------------------------
// computeParetoFrontier
// ---------------------------------------------------------------------------

describe("computeParetoFrontier", () => {
  function makePoint(cost: number, latency: number): ParetoPoint {
    return {
      allocation: {
        systemPromptRatio: 0.05,
        systemPromptTokens: 10_000,
        cacheWriteBreakpointFraction: 0.3,
        reserveTokensForResponse: 8192,
        estimatedCostUsd: cost,
        estimatedP99LatencyMs: latency,
      },
      costUsd: cost,
      p99LatencyMs: latency,
      isPareto: false,
    };
  }

  test("single point is always Pareto", () => {
    const points = computeParetoFrontier([makePoint(1.0, 500)]);
    expect(points[0]!.isPareto).toBe(true);
  });

  test("dominated point is not Pareto", () => {
    const points = computeParetoFrontier([
      makePoint(2.0, 800), // dominated by (1.0, 500)
      makePoint(1.0, 500), // Pareto
    ]);
    const byLatency = points.sort((a, b) => a.p99LatencyMs - b.p99LatencyMs);
    expect(byLatency[0]!.isPareto).toBe(true);  // (1.0, 500)
    expect(byLatency[1]!.isPareto).toBe(false); // (2.0, 800) dominated
  });

  test("tradeoff points are both Pareto", () => {
    // (0.5 cost, 900ms) and (1.5 cost, 300ms) — neither dominates the other
    const points = computeParetoFrontier([
      makePoint(0.5, 900),
      makePoint(1.5, 300),
    ]);
    expect(points.every((p) => p.isPareto)).toBe(true);
  });

  test("all dominated by single best point", () => {
    const points = computeParetoFrontier([
      makePoint(0.1, 100), // best on both axes
      makePoint(0.5, 500),
      makePoint(1.0, 800),
    ]);
    const pareto = points.filter((p) => p.isPareto);
    expect(pareto).toHaveLength(1);
    expect(pareto[0]!.costUsd).toBe(0.1);
    expect(pareto[0]!.p99LatencyMs).toBe(100);
  });

  test("equal-cost different-latency: only lower latency is Pareto", () => {
    const points = computeParetoFrontier([
      makePoint(1.0, 200),
      makePoint(1.0, 800), // same cost, higher latency → dominated
    ]);
    const pareto = points.filter((p) => p.isPareto);
    expect(pareto).toHaveLength(1);
    expect(pareto[0]!.p99LatencyMs).toBe(200);
  });

  test("empty input returns empty", () => {
    expect(computeParetoFrontier([])).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// BudgetMultiObjectiveLearner — static fallback (no data)
// ---------------------------------------------------------------------------

describe("BudgetMultiObjectiveLearner — static fallback", () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), "ashlr-budget-test-"));
  });

  test("returns static allocation when no sessions ingested", async () => {
    const learner = new BudgetMultiObjectiveLearner(tmpDir);
    const alloc = await learner.getRobustBudgetAllocation("anthropic", "claude-sonnet-4");

    expect(alloc.provider).toBe("anthropic");
    expect(alloc.modelId).toBe("claude-sonnet-4");
    expect(alloc.systemPromptRatio).toBe(SYSTEM_PROMPT_BUDGET_RATIO);
    expect(alloc.systemPromptTokens).toBe(
      Math.min(
        Math.floor(200_000 * SYSTEM_PROMPT_BUDGET_RATIO),
        SYSTEM_PROMPT_BUDGET_CAP,
      ),
    );
    expect(alloc.reserveTokensForResponse).toBe(8192);
    // Static fallback sentinel
    expect(alloc.estimatedCostUsd).toBe(0);
  });

  test("static allocation is feasible", async () => {
    const learner = new BudgetMultiObjectiveLearner(tmpDir);
    const alloc = await learner.getRobustBudgetAllocation("anthropic", "claude-sonnet-4");
    const diag = learner.validateBudgetFeasibility(alloc, "anthropic");
    expect(diag.feasible).toBe(true);
    expect(diag.issues).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// BudgetMultiObjectiveLearner — learning from sessions
// ---------------------------------------------------------------------------

describe("BudgetMultiObjectiveLearner — learning", () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), "ashlr-budget-learn-"));
  });

  test("fitAllocation returns null with fewer than MIN_SESSIONS sessions", () => {
    const learner = new BudgetMultiObjectiveLearner(tmpDir);
    learner.ingest(makeSessions(4)); // below MIN_SESSIONS_FOR_LEARNING=5
    const rec = learner.fitAllocation("anthropic", "claude-sonnet-4", "sm");
    expect(rec).toBeNull();
  });

  test("fitAllocation succeeds with 5+ sessions", () => {
    const learner = new BudgetMultiObjectiveLearner(tmpDir);
    learner.ingest(makeSessions(8));
    const rec = learner.fitAllocation("anthropic", "claude-sonnet-4", "sm");
    expect(rec).not.toBeNull();
    expect(rec!.provider).toBe("anthropic");
    expect(rec!.modelId).toBe("claude-sonnet-4");
    expect(rec!.bucket).toBe("sm");
    expect(rec!.sessionCount).toBe(8);
    expect(rec!.paretoFront.length).toBeGreaterThan(0);
  });

  test("getRobustBudgetAllocation uses learned data", async () => {
    const learner = new BudgetMultiObjectiveLearner(tmpDir);
    learner.ingest(makeSessions(10));
    const alloc = await learner.getRobustBudgetAllocation(
      "anthropic",
      "claude-sonnet-4",
      { messageCount: 20 },
    );
    // With learned data, estimatedCostUsd > 0 (non-sentinel)
    expect(alloc.estimatedCostUsd).toBeGreaterThan(0);
    expect(alloc.systemPromptRatio).toBeGreaterThan(0);
    expect(alloc.systemPromptTokens).toBeGreaterThan(0);
    expect(alloc.computedAt).toBeDefined();
  });

  test("learned allocation has valid systemPromptTokens range", async () => {
    const learner = new BudgetMultiObjectiveLearner(tmpDir);
    learner.ingest(makeSessions(10));
    const alloc = await learner.getRobustBudgetAllocation("anthropic", "claude-sonnet-4");
    // Must be within [0, SYSTEM_PROMPT_BUDGET_CAP]
    expect(alloc.systemPromptTokens).toBeGreaterThan(0);
    expect(alloc.systemPromptTokens).toBeLessThanOrEqual(SYSTEM_PROMPT_BUDGET_CAP);
  });

  test("Pareto front contains only non-dominated points", () => {
    const learner = new BudgetMultiObjectiveLearner(tmpDir);
    learner.ingest(makeSessions(10));
    const rec = learner.fitAllocation("anthropic", "claude-sonnet-4", "sm");
    expect(rec).not.toBeNull();
    // All returned Pareto points should have isPareto === true
    for (const p of rec!.paretoFront) {
      expect(p.isPareto).toBe(true);
    }
  });

  test("provider-level query (empty modelId) works", async () => {
    const learner = new BudgetMultiObjectiveLearner(tmpDir);
    learner.ingest(makeSessions(10, { modelId: "" }));
    const alloc = await learner.getRobustBudgetAllocation("anthropic", "");
    expect(alloc.provider).toBe("anthropic");
    expect(alloc.systemPromptTokens).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// targetLatencyMs constraint
// ---------------------------------------------------------------------------

describe("BudgetMultiObjectiveLearner — targetLatencyMs constraint", () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), "ashlr-budget-latency-"));
  });

  test("respects targetLatencyMs when Pareto points satisfy it", async () => {
    const learner = new BudgetMultiObjectiveLearner(tmpDir);
    // Ingest sessions with low latency to ensure the learner builds a frontier
    // with fast-latency options.
    learner.ingest(
      makeSessions(10, { p99LatencyMs: 300, totalCostUsd: 0.02 }),
    );
    const alloc = await learner.getRobustBudgetAllocation(
      "anthropic",
      "claude-sonnet-4",
      { targetLatencyMs: 1000, messageCount: 20 },
    );
    // With an ample latency budget the learner should return a valid allocation.
    expect(alloc.systemPromptTokens).toBeGreaterThan(0);
    expect(alloc.estimatedP99LatencyMs).toBeGreaterThan(0);
  });

  test("falls back gracefully when targetLatencyMs is very tight", async () => {
    const learner = new BudgetMultiObjectiveLearner(tmpDir);
    learner.ingest(makeSessions(10, { p99LatencyMs: 2000 }));
    // Request latency lower than anything achievable — should return lowest-latency point
    const alloc = await learner.getRobustBudgetAllocation(
      "anthropic",
      "claude-sonnet-4",
      { targetLatencyMs: 1, messageCount: 20 },
    );
    expect(alloc).toBeDefined();
    expect(alloc.systemPromptTokens).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// validateBudgetFeasibility
// ---------------------------------------------------------------------------

describe("validateBudgetFeasibility", () => {
  let tmpDir: string;
  let learner: BudgetMultiObjectiveLearner;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), "ashlr-budget-val-"));
    learner = new BudgetMultiObjectiveLearner(tmpDir);
  });

  function makeAlloc(overrides: Partial<BudgetAllocation> = {}): BudgetAllocation {
    return {
      provider: "anthropic",
      modelId: "claude-sonnet-4",
      systemPromptRatio: 0.05,
      systemPromptTokens: 10_000,
      cacheWriteBreakpointFraction: 0.3,
      reserveTokensForResponse: 8192,
      estimatedCostUsd: 0.05,
      estimatedP99LatencyMs: 500,
      computedAt: new Date().toISOString(),
      ...overrides,
    };
  }

  test("valid allocation is feasible", () => {
    const diag = learner.validateBudgetFeasibility(makeAlloc(), "anthropic");
    expect(diag.feasible).toBe(true);
    expect(diag.issues).toHaveLength(0);
    expect(diag.remainingMessageBudget).toBeGreaterThan(0);
  });

  test("detects systemPromptTokens exceeding 80% of context", () => {
    const diag = learner.validateBudgetFeasibility(
      makeAlloc({ systemPromptTokens: 180_000 }),
      "anthropic",
    );
    expect(diag.feasible).toBe(false);
    expect(diag.issues.some((i) => i.includes("80%"))).toBe(true);
  });

  test("warns at 50–80% utilization but still feasible", () => {
    // 120K out of 200K = 60% utilization
    const diag = learner.validateBudgetFeasibility(
      makeAlloc({ systemPromptTokens: 120_000 }),
      "anthropic",
    );
    // Not infeasible on its own, but warns
    const hasWarn = diag.issues.some((i) => i.includes("50%"));
    expect(hasWarn).toBe(true);
  });

  test("detects systemPrompt + reserve exceeding context", () => {
    // anthropic context limit = 200K; 190K + 20K = 210K > 200K
    const diag = learner.validateBudgetFeasibility(
      makeAlloc({
        systemPromptTokens: 190_000,
        reserveTokensForResponse: 20_000,
      }),
      "anthropic",
    );
    expect(diag.feasible).toBe(false);
    expect(diag.remainingMessageBudget).toBeLessThan(0);
  });

  test("detects below-minimum reserve", () => {
    const diag = learner.validateBudgetFeasibility(
      makeAlloc({ reserveTokensForResponse: 128 }),
      "anthropic",
    );
    expect(diag.feasible).toBe(false);
    expect(diag.issues.some((i) => i.includes("minimum recommended"))).toBe(true);
  });

  test("detects invalid cacheWriteBreakpointFraction", () => {
    const diag = learner.validateBudgetFeasibility(
      makeAlloc({ cacheWriteBreakpointFraction: 1.5 }),
      "anthropic",
    );
    expect(diag.feasible).toBe(false);
    expect(diag.issues.some((i) => i.includes("[0, 1]"))).toBe(true);
  });

  test("provides fallback when infeasible", () => {
    const diag = learner.validateBudgetFeasibility(
      makeAlloc({ systemPromptTokens: 190_000, reserveTokensForResponse: 20_000 }),
      "anthropic",
    );
    expect(diag.feasible).toBe(false);
    expect(diag.fallback).toBeDefined();
    expect(diag.fallback!.systemPromptRatio).toBe(SYSTEM_PROMPT_BUDGET_RATIO);
  });

  test("utilization field is computed correctly", () => {
    // 10K / 200K = 0.05
    const diag = learner.validateBudgetFeasibility(
      makeAlloc({ systemPromptTokens: 10_000 }),
      "anthropic",
    );
    expect(diag.systemPromptUtilization).toBeCloseTo(0.05, 3);
  });
});

// ---------------------------------------------------------------------------
// applyAdaptiveBudget (context.ts integration)
// ---------------------------------------------------------------------------

describe("applyAdaptiveBudget", () => {
  function makeAlloc(overrides: Partial<BudgetAllocation> = {}): BudgetAllocation {
    return {
      provider: "anthropic",
      modelId: "claude-sonnet-4",
      systemPromptRatio: 0.05,
      systemPromptTokens: 10_000,
      cacheWriteBreakpointFraction: 0.3,
      reserveTokensForResponse: 4096,
      estimatedCostUsd: 0.05,
      estimatedP99LatencyMs: 500,
      computedAt: new Date().toISOString(),
      ...overrides,
    };
  }

  test("returns reserveTokens override when allocation has learned data", () => {
    const override = applyAdaptiveBudget(makeAlloc({ reserveTokensForResponse: 4096 }));
    expect(override).toEqual({ reserveTokens: 4096 });
  });

  test("returns empty object for static-fallback allocation (estimatedCostUsd === 0)", () => {
    const override = applyAdaptiveBudget(makeAlloc({ estimatedCostUsd: 0 }));
    expect(override).toEqual({});
  });

  test("learned reserve different from default is forwarded", () => {
    const override = applyAdaptiveBudget(makeAlloc({ reserveTokensForResponse: 2048, estimatedCostUsd: 0.1 }));
    expect(override.reserveTokens).toBe(2048);
  });
});

// ---------------------------------------------------------------------------
// triggerRecalibration + persistCalibrations
// ---------------------------------------------------------------------------

describe("BudgetMultiObjectiveLearner — recalibration + persistence", () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), "ashlr-budget-recal-"));
  });

  test("triggerRecalibration returns 0 with sparse data", async () => {
    const learner = new BudgetMultiObjectiveLearner(tmpDir);
    learner.ingest(makeSessions(3)); // below threshold
    const fitted = await learner.triggerRecalibration();
    expect(fitted).toBe(0);
  });

  test("triggerRecalibration fits multiple buckets", async () => {
    const learner = new BudgetMultiObjectiveLearner(tmpDir);
    // xs bucket (1–10)
    learner.ingest(makeSessions(5, { messageCount: 5 }));
    // sm bucket (11–50)
    learner.ingest(makeSessions(5, { messageCount: 25 }));
    const fitted = await learner.triggerRecalibration();
    expect(fitted).toBeGreaterThanOrEqual(2);
  });

  test("persistCalibrations writes to disk and loadCalibrations reads it back", async () => {
    const learner = new BudgetMultiObjectiveLearner(tmpDir);
    learner.ingest(makeSessions(10));
    learner.fitAllocation("anthropic", "claude-sonnet-4", "sm");
    await learner.persistCalibrations();

    // New learner instance reads from disk
    const learner2 = new BudgetMultiObjectiveLearner(tmpDir);
    await learner2.loadCalibrations();
    const alloc = await learner2.getRobustBudgetAllocation(
      "anthropic",
      "claude-sonnet-4",
      { messageCount: 20 },
    );
    // Should come from loaded calibration (estimatedCostUsd > 0)
    expect(alloc.estimatedCostUsd).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// visualizeParetoFront
// ---------------------------------------------------------------------------

describe("visualizeParetoFront", () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), "ashlr-budget-viz-"));
  });

  test("returns not-available message when no data", () => {
    const learner = new BudgetMultiObjectiveLearner(tmpDir);
    const out = learner.visualizeParetoFront("anthropic", "claude-sonnet-4");
    expect(out).toContain("No Pareto front available");
  });

  test("returns formatted table when data exists", () => {
    const learner = new BudgetMultiObjectiveLearner(tmpDir);
    learner.ingest(makeSessions(10));
    learner.fitAllocation("anthropic", "claude-sonnet-4", "sm");
    const out = learner.visualizeParetoFront("anthropic", "claude-sonnet-4", "sm");
    expect(out).toContain("Pareto Front");
    expect(out).toContain("ratio");
    expect(out).toContain("costUsd");
  });
});
