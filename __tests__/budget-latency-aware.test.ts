/**
 * Tests for LatencyAwareAllocator — latency model fitting, Pareto frontier
 * coverage, budget binding, provider fallback, cache effect, and extreme
 * parameter corners.
 */

import { describe, expect, test, beforeEach } from "bun:test";
import { mkdtemp } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";

import {
  LatencyAwareAllocator,
  CONTEXT_WINDOW_STEPS,
} from "../src/budget/latency-aware-allocator.ts";
import type {
  LatencyCostRecord,
  LatencyAwareAllocation,
  FittedLatencyModel,
} from "../src/budget/latency-aware-allocator.ts";
import { BudgetMultiObjectiveLearner } from "../src/budget/multi-objective-learner.ts";
import type { SessionSummary } from "../src/budget/multi-objective-learner.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeTmpDir(): Promise<string> {
  return mkdtemp(join(tmpdir(), "ashlr-latency-test-"));
}

function makeRecord(overrides: Partial<LatencyCostRecord> = {}): LatencyCostRecord {
  return {
    provider: "anthropic",
    modelId: "claude-sonnet-4",
    promptTokens: 32_000,
    cachedFraction: 0.5,
    responseTokenBudget: 2048,
    observedLatencyP50Ms: 450,
    observedLatencyP99Ms: 900,
    billedCostUsd: 0.08,
    recordedAt: new Date().toISOString(),
    ...overrides,
  };
}

function makeRecords(
  n: number,
  overrides: Partial<LatencyCostRecord> = {},
): LatencyCostRecord[] {
  return Array.from({ length: n }, (_, i) =>
    makeRecord({
      ...overrides,
      recordedAt: new Date(Date.now() - i * 60_000).toISOString(),
      // Vary prompt tokens slightly so regression has signal.
      promptTokens: (overrides.promptTokens ?? 32_000) + i * 1_000,
    }),
  );
}

function makeSession(overrides: Partial<SessionSummary> = {}): SessionSummary {
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

// ---------------------------------------------------------------------------
// 1. Latency model fitting — default coefficients (< MIN_FIT_RECORDS)
// ---------------------------------------------------------------------------

describe("LatencyAwareAllocator — latency model (default coefficients)", () => {
  let tmpDir: string;

  beforeEach(async () => { tmpDir = await makeTmpDir(); });

  test("returns a model even with zero records (uses built-in defaults)", () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const model = allocator.getModel("anthropic", "claude-sonnet-4");
    expect(model).toBeDefined();
    expect(model.provider).toBe("anthropic");
    expect(model.p50Coefficients).toHaveLength(4);
    expect(model.p99Coefficients).toHaveLength(4);
    // With no data, R² sentinel is -1.
    expect(model.p50R2).toBe(-1);
    expect(model.p99R2).toBe(-1);
  });

  test("default p99 coefficients produce latency > p50 coefficients", () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const model = allocator.getModel("anthropic", "claude-sonnet-4");
    // β0 for p99 should be larger than for p50.
    expect(model.p99Coefficients[0]).toBeGreaterThan(model.p50Coefficients[0]);
  });

  test("unknown provider gets sensible fallback coefficients", () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const model = allocator.getModel("my-unknown-provider", "my-model");
    expect(model.p50Coefficients[0]).toBeGreaterThan(0);
    expect(model.p99Coefficients[0]).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// 2. Latency model fitting — fitted from real records (>= MIN_FIT_RECORDS)
// ---------------------------------------------------------------------------

describe("LatencyAwareAllocator — latency model (fitted)", () => {
  let tmpDir: string;

  beforeEach(async () => { tmpDir = await makeTmpDir(); });

  test("fitModel uses ingested records when >= MIN_FIT_RECORDS present", () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    // Provide records with increasing latency as prompt grows.
    const records = makeRecords(8, { observedLatencyP50Ms: 300, observedLatencyP99Ms: 700 });
    allocator.ingestRecords(records);
    const model = allocator.getModel("anthropic", "claude-sonnet-4");
    expect(model.recordCount).toBe(8);
    // Fitted model should have R² values (may still be low with synthetic data).
    expect(model.p50R2).toBeGreaterThanOrEqual(-1);
    expect(model.p99R2).toBeGreaterThanOrEqual(-1);
  });

  test("ingestRecords invalidates cached model and refits", () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    // First fit with 5 records.
    allocator.ingestRecords(makeRecords(5));
    const model1 = allocator.getModel("anthropic", "claude-sonnet-4");
    expect(model1.recordCount).toBe(5);

    // Ingest more records — cache should be cleared.
    allocator.ingestRecords(makeRecords(3, { observedLatencyP50Ms: 600 }));
    const model2 = allocator.getModel("anthropic", "claude-sonnet-4");
    expect(model2.recordCount).toBe(8);
  });

  test("fitted model predicts monotonically increasing latency with prompt size", () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    // Create records where latency scales linearly with prompt size.
    const records: LatencyCostRecord[] = [8_000, 16_000, 32_000, 64_000, 96_000, 128_000].map(
      (tokens, i) =>
        makeRecord({
          promptTokens: tokens,
          observedLatencyP50Ms: 100 + tokens * 0.004,
          observedLatencyP99Ms: 200 + tokens * 0.007,
          recordedAt: new Date(Date.now() - i * 1000).toISOString(),
        }),
    );
    allocator.ingestRecords(records);
    const model = allocator.getModel("anthropic", "claude-sonnet-4");

    // Indirect check: the model β1 coefficient should be positive (more tokens = more latency).
    expect(model.p50Coefficients[1]).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// 3. Pareto frontier coverage
// ---------------------------------------------------------------------------

describe("LatencyAwareAllocator — Pareto frontier coverage", () => {
  let tmpDir: string;

  beforeEach(async () => { tmpDir = await makeTmpDir(); });

  test("optimalContextAllocation returns 1–3 allocations", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const result = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.10, 800,
    );
    expect(result.length).toBeGreaterThanOrEqual(1);
    expect(result.length).toBeLessThanOrEqual(3);
  });

  test("allocations are sorted by paretoRank 1, 2, 3", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const result = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.10, 1000,
    );
    for (let i = 0; i < result.length; i++) {
      expect(result[i]!.paretoRank).toBe(i + 1);
    }
  });

  test("all returned contextWindowTokens are from CONTEXT_WINDOW_STEPS", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const result = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.20, 1500,
    );
    const validSteps = new Set<number>(CONTEXT_WINDOW_STEPS);
    for (const alloc of result) {
      expect(validSteps.has(alloc.contextWindowTokens)).toBe(true);
    }
  });

  test("rank-1 allocation has the highest feasibilityScore", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const result = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.10, 800,
    );
    if (result.length > 1) {
      expect(result[0]!.feasibilityScore).toBeGreaterThanOrEqual(
        result[1]!.feasibilityScore,
      );
    }
    if (result.length > 2) {
      expect(result[1]!.feasibilityScore).toBeGreaterThanOrEqual(
        result[2]!.feasibilityScore,
      );
    }
  });

  test("allocations include all required fields", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const [best] = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.10, 800,
    );
    expect(best).toBeDefined();
    expect(typeof best!.contextWindowTokens).toBe("number");
    expect(typeof best!.estimatedLatencyP50Ms).toBe("number");
    expect(typeof best!.estimatedLatencyP99Ms).toBe("number");
    expect(typeof best!.estimatedCostUsd).toBe("number");
    expect(typeof best!.cacheWriteCostUsd).toBe("number");
    expect(typeof best!.cacheMissProbability).toBe("number");
    expect(typeof best!.paretoRank).toBe("number");
    expect(typeof best!.feasibilityScore).toBe("number");
  });

  test("p99 latency >= p50 latency for all allocations", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const result = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.10, 800,
    );
    for (const alloc of result) {
      expect(alloc.estimatedLatencyP99Ms).toBeGreaterThanOrEqual(
        alloc.estimatedLatencyP50Ms,
      );
    }
  });
});

// ---------------------------------------------------------------------------
// 4. Budget binding constraint
// ---------------------------------------------------------------------------

describe("LatencyAwareAllocator — budget binding", () => {
  let tmpDir: string;

  beforeEach(async () => { tmpDir = await makeTmpDir(); });

  test("rank-1 allocation respects tight budget (very low maxBudgetUsd)", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    // With a near-zero budget, the allocator should pick the smallest/cheapest window.
    const result = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.0001, 800,
    );
    expect(result.length).toBeGreaterThanOrEqual(1);
    // The cheapest window should be smallest (fewest tokens billed).
    const cheapest = result.reduce((a, b) =>
      a.estimatedCostUsd < b.estimatedCostUsd ? a : b,
    );
    expect(cheapest.contextWindowTokens).toBe(CONTEXT_WINDOW_STEPS[0]);
  });

  test("larger budget allows larger context windows in the result set", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const tightResult = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.001, 5000,
    );
    const generousResult = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 100.0, 5000,
    );
    // The generous budget should allow at least as large a window.
    const maxTight = Math.max(...tightResult.map((a) => a.contextWindowTokens));
    const maxGenerous = Math.max(...generousResult.map((a) => a.contextWindowTokens));
    expect(maxGenerous).toBeGreaterThanOrEqual(maxTight);
  });

  test("feasibilityScore is in [0, 1]", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const result = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.10, 800,
    );
    for (const alloc of result) {
      expect(alloc.feasibilityScore).toBeGreaterThanOrEqual(0);
      expect(alloc.feasibilityScore).toBeLessThanOrEqual(1);
    }
  });
});

// ---------------------------------------------------------------------------
// 5. Provider fallback
// ---------------------------------------------------------------------------

describe("LatencyAwareAllocator — provider fallback", () => {
  let tmpDir: string;

  beforeEach(async () => { tmpDir = await makeTmpDir(); });

  test("groq has lower base latency than anthropic (default model)", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const groqModel  = allocator.getModel("groq", "llama3");
    const anthropicModel = allocator.getModel("anthropic", "claude-sonnet-4");
    // Groq is an inference-optimized provider with very low TTFT.
    expect(groqModel.p50Coefficients[0]).toBeLessThan(anthropicModel.p50Coefficients[0]);
  });

  test("ollama has higher base latency than groq (default model)", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const ollamaModel = allocator.getModel("ollama", "llama3");
    const groqModel   = allocator.getModel("groq", "llama3");
    expect(ollamaModel.p50Coefficients[0]).toBeGreaterThan(groqModel.p50Coefficients[0]);
  });

  test("unknown provider falls back gracefully without throwing", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const result = await allocator.optimalContextAllocation(
      "my-unknown-provider-xyz", "some-model", 0.10, 1000,
    );
    expect(result.length).toBeGreaterThanOrEqual(1);
  });

  test("empty modelId resolves to provider-level model", () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const model = allocator.getModel("anthropic", "");
    expect(model).toBeDefined();
    expect(model.p50Coefficients[0]).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// 6. Cache effect on latency and cost
// ---------------------------------------------------------------------------

describe("LatencyAwareAllocator — cache effect", () => {
  let tmpDir: string;

  beforeEach(async () => { tmpDir = await makeTmpDir(); });

  test("fully cached prompt (cachedFraction=1.0) has lower latency than cold (0.0)", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const [warm] = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.5, 2000, 1.0,
    );
    const [cold] = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.5, 2000, 0.0,
    );
    expect(warm!.estimatedLatencyP99Ms).toBeLessThan(cold!.estimatedLatencyP99Ms);
  });

  test("fully cached prompt has lower cost (no input billing) than cold", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const [warm] = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.5, 2000, 1.0,
    );
    const [cold] = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.5, 2000, 0.0,
    );
    // Warm cache reads at 0.1× input rate → significantly cheaper.
    expect(warm!.estimatedCostUsd).toBeLessThan(cold!.estimatedCostUsd);
  });

  test("cacheMissProbability is higher for low cachedFraction", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const [warm] = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.5, 2000, 0.9,
    );
    const [cold] = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.5, 2000, 0.1,
    );
    expect(cold!.cacheMissProbability).toBeGreaterThan(warm!.cacheMissProbability);
  });

  test("cacheWriteCostUsd is zero for groq (no cache write support)", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const [result] = await allocator.optimalContextAllocation(
      "groq", "llama3", 0.5, 2000, 0.5,
    );
    expect(result!.cacheWriteCostUsd).toBe(0);
  });

  test("cacheWriteCostUsd is positive for anthropic (supports prompt caching)", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const [result] = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.5, 2000, 0.5,
    );
    expect(result!.cacheWriteCostUsd).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// 7. Extreme parameter corners
// ---------------------------------------------------------------------------

describe("LatencyAwareAllocator — extreme parameter corners", () => {
  let tmpDir: string;

  beforeEach(async () => { tmpDir = await makeTmpDir(); });

  test("maxBudgetUsd = 0 still returns a result (not NaN/crash)", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const result = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0, 800,
    );
    expect(result.length).toBeGreaterThanOrEqual(1);
    for (const a of result) {
      expect(isNaN(a.feasibilityScore)).toBe(false);
    }
  });

  test("targetLatencyMs = 1 still returns a result (very tight latency)", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const result = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 10.0, 1,
    );
    expect(result.length).toBeGreaterThanOrEqual(1);
  });

  test("targetLatencyMs = 1e9 (very loose) still returns a result", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const result = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 10.0, 1_000_000_000,
    );
    expect(result.length).toBeGreaterThanOrEqual(1);
  });

  test("model with tiny context limit still returns a result", async () => {
    // ollama default context limit is 32K — only CONTEXT_WINDOW_STEPS[0] (16K) and [1] (32K) fit.
    const allocator = new LatencyAwareAllocator(tmpDir);
    const result = await allocator.optimalContextAllocation(
      "ollama", "llama2", 1.0, 5000,
    );
    expect(result.length).toBeGreaterThanOrEqual(1);
    // All returned windows should fit within ollama's 32K limit.
    for (const a of result) {
      expect(a.contextWindowTokens).toBeLessThanOrEqual(32_768);
    }
  });

  test("responseTokenBudget = 0 is handled without NaN", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const result = await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.10, 800, 0.5, 0,
    );
    expect(result.length).toBeGreaterThanOrEqual(1);
    for (const a of result) {
      expect(isNaN(a.estimatedCostUsd)).toBe(false);
    }
  });
});

// ---------------------------------------------------------------------------
// 8. Persistence — audit log
// ---------------------------------------------------------------------------

describe("LatencyAwareAllocator — persistence", () => {
  let tmpDir: string;

  beforeEach(async () => { tmpDir = await makeTmpDir(); });

  test("optimalContextAllocation writes to latency-aware-allocations.jsonl", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.10, 800,
    );
    const records = await allocator.loadPersistedAllocations();
    expect(records.length).toBeGreaterThanOrEqual(1);
    expect(records[0]!.provider).toBe("anthropic");
    expect(records[0]!.modelId).toBe("claude-sonnet-4");
    expect(records[0]!.allocations.length).toBeGreaterThanOrEqual(1);
  });

  test("modelSource is 'default' when no records ingested", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.10, 800,
    );
    const records = await allocator.loadPersistedAllocations();
    expect(records[0]!.modelSource).toBe("default");
  });

  test("modelSource is 'fitted' when >= MIN_FIT_RECORDS ingested", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    allocator.ingestRecords(makeRecords(6));
    await allocator.optimalContextAllocation(
      "anthropic", "claude-sonnet-4", 0.10, 800,
    );
    const records = await allocator.loadPersistedAllocations();
    expect(records[0]!.modelSource).toBe("fitted");
  });

  test("multiple calls append multiple records", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    await allocator.optimalContextAllocation("anthropic", "claude-sonnet-4", 0.10, 800);
    await allocator.optimalContextAllocation("openai", "gpt-4o", 0.05, 600);
    const records = await allocator.loadPersistedAllocations();
    expect(records.length).toBeGreaterThanOrEqual(2);
  });
});

// ---------------------------------------------------------------------------
// 9. refineWithLatency — BudgetMultiObjectiveLearner integration
// ---------------------------------------------------------------------------

describe("LatencyAwareAllocator.refineWithLatency", () => {
  let tmpDir: string;

  beforeEach(async () => { tmpDir = await makeTmpDir(); });

  test("refineWithLatency returns a BudgetAllocation with updated latency", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const learner = new BudgetMultiObjectiveLearner(tmpDir);
    const sessions: SessionSummary[] = Array.from({ length: 10 }, (_, i) =>
      makeSession({ recordedAt: new Date(Date.now() - i * 60_000).toISOString() }),
    );
    learner.ingest(sessions);
    const baseAlloc = await learner.getRobustBudgetAllocation("anthropic", "claude-sonnet-4");

    const refined = await allocator.refineWithLatency(baseAlloc, 1000, 0.20);

    // Should return the core BudgetAllocation fields.
    expect(refined.provider).toBe("anthropic");
    expect(refined.modelId).toBe("claude-sonnet-4");
    // latencyAwareAllocations is the extended field.
    expect(refined.latencyAwareAllocations.length).toBeGreaterThanOrEqual(1);
  });

  test("refined estimatedP99LatencyMs is a positive number", async () => {
    const allocator = new LatencyAwareAllocator(tmpDir);
    const learner = new BudgetMultiObjectiveLearner(tmpDir);
    const baseAlloc = await learner.getRobustBudgetAllocation("anthropic", "claude-sonnet-4");

    const refined = await allocator.refineWithLatency(baseAlloc, 1000, 0.20);
    expect(refined.estimatedP99LatencyMs).toBeGreaterThan(0);
    expect(isNaN(refined.estimatedP99LatencyMs)).toBe(false);
  });

  test("getRobustBudgetAllocation with targetLatencyMs triggers latency refinement", async () => {
    const learner = new BudgetMultiObjectiveLearner(tmpDir);
    learner.ingest(
      Array.from({ length: 10 }, (_, i) =>
        makeSession({ recordedAt: new Date(Date.now() - i * 60_000).toISOString() }),
      ),
    );
    // With targetLatencyMs provided, latency-aware post-processing kicks in.
    const alloc = await learner.getRobustBudgetAllocation(
      "anthropic",
      "claude-sonnet-4",
      { targetLatencyMs: 800, messageCount: 20 },
    );
    // Should still return a valid BudgetAllocation.
    expect(alloc.provider).toBe("anthropic");
    expect(alloc.systemPromptTokens).toBeGreaterThan(0);
    expect(alloc.estimatedP99LatencyMs).toBeGreaterThan(0);
  });
});
