/**
 * Tests for CompressorLearner — cost-history based ROI tier selection.
 *
 * Covers:
 *   - ROI computation accuracy
 *   - Bucket assignments (xs/sm/md/lg)
 *   - Tier recommendation stability (no thrashing)
 *   - Provider-switch cost absorption
 *   - Graceful degradation when history is sparse
 *   - Integration with selectCompressionTierAdaptive via roiBreakdown param
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, rm } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";
import {
  CompressorLearner,
  selectCompressionTierAdaptive,
  type CostHistoryRecord,
  type TierROIBreakdown,
} from "../src/compression/adaptive.ts";
import type { Message } from "../src/types/index.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function makeTmpDir(): Promise<string> {
  const base = join(
    tmpdir(),
    `ashlr-cost-learning-test-${Date.now()}-${Math.random().toString(36).slice(2)}`,
  );
  await mkdir(base, { recursive: true });
  return base;
}

function textMsg(role: "user" | "assistant", text: string): Message {
  return { role, content: text };
}

/** Build a minimal CostHistoryRecord (omitting recordedAt). */
function makeRecord(
  overrides: Partial<CostHistoryRecord> & {
    tier: 1 | 2 | 3 | 4;
    messageCount: number;
    provider: string;
    actual_cost_usd: number;
  },
): Omit<CostHistoryRecord, "recordedAt"> {
  return {
    estimated_tokens: 1000,
    actual_tokens: 1050,
    success: true,
    ...overrides,
  };
}

/** Seed a learner with an array of records (no recordedAt needed). */
async function seedRecords(
  learner: CompressorLearner,
  records: Omit<CostHistoryRecord, "recordedAt">[],
): Promise<void> {
  for (const r of records) {
    await learner.record(r);
  }
}

// ---------------------------------------------------------------------------
// costHistoryPath
// ---------------------------------------------------------------------------

describe("CompressorLearner.costHistoryPath", () => {
  test("is inside genome/evolution directory", () => {
    const l = new CompressorLearner("/projects/myapp");
    const p = l.costHistoryPath();
    expect(p).toContain(".ashlrcode/genome/evolution");
    expect(p).toContain("compression-cost-history.jsonl");
  });

  test("is deterministic for same cwd", () => {
    const l = new CompressorLearner("/x");
    expect(l.costHistoryPath()).toBe(l.costHistoryPath());
  });

  test("differs for different cwd values", () => {
    const a = new CompressorLearner("/a").costHistoryPath();
    const b = new CompressorLearner("/b").costHistoryPath();
    expect(a).not.toBe(b);
  });
});

// ---------------------------------------------------------------------------
// record() — file management
// ---------------------------------------------------------------------------

describe("CompressorLearner.record()", () => {
  let tmp: string;
  let learner: CompressorLearner;

  beforeEach(async () => {
    tmp = await makeTmpDir();
    learner = new CompressorLearner(tmp);
  });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("creates the evolution dir and file on first record", async () => {
    const { existsSync } = await import("fs");
    await learner.record(makeRecord({ tier: 3, messageCount: 10, provider: "claude-3-5-sonnet", actual_cost_usd: 0.002 }));
    expect(existsSync(learner.costHistoryPath())).toBe(true);
  });

  test("accumulates multiple records", async () => {
    await learner.record(makeRecord({ tier: 3, messageCount: 10, provider: "claude-3-5-sonnet", actual_cost_usd: 0.002 }));
    await learner.record(makeRecord({ tier: 2, messageCount: 20, provider: "claude-3-5-sonnet", actual_cost_usd: 0.003 }));
    const records = await learner.readHistory();
    expect(records.length).toBe(2);
  });

  test("stores all required fields", async () => {
    await learner.record(makeRecord({
      tier: 2,
      messageCount: 15,
      provider: "claude-3-opus",
      estimated_tokens: 2000,
      actual_tokens: 2100,
      actual_cost_usd: 0.005,
      success: true,
    }));
    const records = await learner.readHistory();
    expect(records[0]!.tier).toBe(2);
    expect(records[0]!.messageCount).toBe(15);
    expect(records[0]!.provider).toBe("claude-3-opus");
    expect(records[0]!.actual_cost_usd).toBeCloseTo(0.005);
    expect(typeof records[0]!.recordedAt).toBe("string");
  });

  test("enforces 50-record sliding window", async () => {
    // Record 55 entries — only the latest 50 should be kept.
    for (let i = 0; i < 55; i++) {
      await learner.record(makeRecord({
        tier: 3,
        messageCount: i,
        provider: "claude-3-5-sonnet",
        actual_cost_usd: i * 0.001,
      }));
    }
    const records = await learner.readHistory();
    expect(records.length).toBe(50);
    // First kept record should have messageCount=5 (records 0-4 dropped)
    expect(records[0]!.messageCount).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// Bucket assignments
// ---------------------------------------------------------------------------

describe("Bucket assignments (xs/sm/md/lg)", () => {
  let tmp: string;
  let learner: CompressorLearner;

  beforeEach(async () => {
    tmp = await makeTmpDir();
    learner = new CompressorLearner(tmp);
  });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("messageCount 1 → xs bucket", async () => {
    const records: CostHistoryRecord[] = [
      { tier: 3, messageCount: 1, provider: "p1", estimated_tokens: 100, actual_tokens: 105, actual_cost_usd: 0.001, success: true, recordedAt: new Date().toISOString() },
      { tier: 3, messageCount: 5, provider: "p1", estimated_tokens: 100, actual_tokens: 105, actual_cost_usd: 0.001, success: true, recordedAt: new Date().toISOString() },
      { tier: 3, messageCount: 10, provider: "p1", estimated_tokens: 100, actual_tokens: 105, actual_cost_usd: 0.001, success: true, recordedAt: new Date().toISOString() },
    ];
    const breakdown = await learner.computeROI("p1", 1, records);
    expect(breakdown.bucket).toBe("xs");
  });

  test("messageCount 11 → sm bucket", async () => {
    const breakdown = await learner.computeROI("p1", 11, []);
    expect(breakdown.bucket).toBe("sm");
  });

  test("messageCount 50 → sm bucket", async () => {
    const breakdown = await learner.computeROI("p1", 50, []);
    expect(breakdown.bucket).toBe("sm");
  });

  test("messageCount 51 → md bucket", async () => {
    const breakdown = await learner.computeROI("p1", 51, []);
    expect(breakdown.bucket).toBe("md");
  });

  test("messageCount 200 → md bucket", async () => {
    const breakdown = await learner.computeROI("p1", 200, []);
    expect(breakdown.bucket).toBe("md");
  });

  test("messageCount 201 → lg bucket", async () => {
    const breakdown = await learner.computeROI("p1", 201, []);
    expect(breakdown.bucket).toBe("lg");
  });

  test("records for sm bucket do not pollute md bucket ROI", async () => {
    // Seed 5 records for sm (messageCount 20) with low cost on tier 2
    // and 0 records for md — md ROI should have no recommended tier.
    const records: CostHistoryRecord[] = [];
    for (let i = 0; i < 5; i++) {
      records.push({
        tier: 4, messageCount: 20, provider: "p1",
        estimated_tokens: 1000, actual_tokens: 1000, actual_cost_usd: 0.010,
        success: true, recordedAt: new Date().toISOString(),
      });
      records.push({
        tier: 2, messageCount: 20, provider: "p1",
        estimated_tokens: 1000, actual_tokens: 1000, actual_cost_usd: 0.002,
        success: true, recordedAt: new Date().toISOString(),
      });
    }
    const breakdownSm = await learner.computeROI("p1", 20, records);
    const breakdownMd = await learner.computeROI("p1", 100, records);
    // sm should find something; md has no records so no recommendation
    expect(breakdownSm.bucket).toBe("sm");
    expect(breakdownMd.recommendedTier).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// ROI computation accuracy
// ---------------------------------------------------------------------------

describe("ROI computation accuracy", () => {
  let tmp: string;
  let learner: CompressorLearner;

  beforeEach(async () => {
    tmp = await makeTmpDir();
    learner = new CompressorLearner(tmp);
  });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("tier 4 ROI is always 0 (it is the baseline)", async () => {
    const records: CostHistoryRecord[] = Array.from({ length: 5 }, (_, i) => ({
      tier: 4 as const,
      messageCount: 10, provider: "claude-3-5-sonnet",
      estimated_tokens: 1000, actual_tokens: 1000,
      actual_cost_usd: 0.010 + i * 0.001,
      success: true, recordedAt: new Date().toISOString(),
    }));
    const breakdown = await learner.computeROI("claude-3-5-sonnet", 10, records);
    expect(breakdown.byTier[4].roi).toBeCloseTo(0, 5);
  });

  test("positive ROI when tier 1 is significantly cheaper than baseline tier 4", async () => {
    // Tier 4 costs 0.020 USD median; tier 1 costs 0.004 USD median.
    // Savings = 0.016; switch_cost is small → high positive ROI.
    const records: CostHistoryRecord[] = [
      ...Array.from({ length: 5 }, () => ({
        tier: 4 as const, messageCount: 5, provider: "p1",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.020, success: true, recordedAt: new Date().toISOString(),
      })),
      ...Array.from({ length: 5 }, () => ({
        tier: 1 as const, messageCount: 5, provider: "p1",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.004, success: true, recordedAt: new Date().toISOString(),
      })),
    ];
    const breakdown = await learner.computeROI("p1", 5, records);
    expect(breakdown.byTier[1].roi).toBeGreaterThan(0);
  });

  test("negative ROI when tier 1 costs more than baseline tier 4", async () => {
    // Tier 1 (autoCompact with LLM overhead) costs more than tier 4 in cheap sessions.
    const records: CostHistoryRecord[] = [
      ...Array.from({ length: 5 }, () => ({
        tier: 4 as const, messageCount: 5, provider: "p1",
        estimated_tokens: 500, actual_tokens: 500,
        actual_cost_usd: 0.001, success: true, recordedAt: new Date().toISOString(),
      })),
      ...Array.from({ length: 5 }, () => ({
        tier: 1 as const, messageCount: 5, provider: "p1",
        estimated_tokens: 500, actual_tokens: 500,
        actual_cost_usd: 0.050, success: true, recordedAt: new Date().toISOString(),
      })),
    ];
    const breakdown = await learner.computeROI("p1", 5, records);
    expect(breakdown.byTier[1].roi).toBeLessThan(0);
  });

  test("ROI uses median not mean — outliers don't dominate", async () => {
    // 4 records at 0.002 and 1 extreme outlier at 2.000 → median = 0.002
    const records: CostHistoryRecord[] = [
      ...Array.from({ length: 5 }, () => ({
        tier: 4 as const, messageCount: 5, provider: "p1",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.010, success: true, recordedAt: new Date().toISOString(),
      })),
      { tier: 1 as const, messageCount: 5, provider: "p1", estimated_tokens: 1000, actual_tokens: 1000, actual_cost_usd: 0.002, success: true, recordedAt: new Date().toISOString() },
      { tier: 1 as const, messageCount: 5, provider: "p1", estimated_tokens: 1000, actual_tokens: 1000, actual_cost_usd: 0.002, success: true, recordedAt: new Date().toISOString() },
      { tier: 1 as const, messageCount: 5, provider: "p1", estimated_tokens: 1000, actual_tokens: 1000, actual_cost_usd: 0.002, success: true, recordedAt: new Date().toISOString() },
      { tier: 1 as const, messageCount: 5, provider: "p1", estimated_tokens: 1000, actual_tokens: 1000, actual_cost_usd: 0.002, success: true, recordedAt: new Date().toISOString() },
      { tier: 1 as const, messageCount: 5, provider: "p1", estimated_tokens: 1000, actual_tokens: 1000, actual_cost_usd: 2.000, success: true, recordedAt: new Date().toISOString() }, // outlier
    ];
    const breakdown = await learner.computeROI("p1", 5, records);
    // Median of [0.002, 0.002, 0.002, 0.002, 2.000] = 0.002 → positive ROI
    expect(breakdown.byTier[1].medianCostUsd).toBeCloseTo(0.002, 3);
    expect(breakdown.byTier[1].roi).toBeGreaterThan(0);
  });

  test("sampleCount reflects actual number of records in bucket", async () => {
    const records: CostHistoryRecord[] = Array.from({ length: 7 }, () => ({
      tier: 2 as const, messageCount: 30, provider: "p1",
      estimated_tokens: 1000, actual_tokens: 1000,
      actual_cost_usd: 0.003, success: true, recordedAt: new Date().toISOString(),
    }));
    const breakdown = await learner.computeROI("p1", 30, records);
    expect(breakdown.byTier[2].sampleCount).toBe(7);
  });
});

// ---------------------------------------------------------------------------
// Tier recommendation stability (no thrashing)
// ---------------------------------------------------------------------------

describe("Tier recommendation stability", () => {
  let tmp: string;
  let learner: CompressorLearner;

  beforeEach(async () => {
    tmp = await makeTmpDir();
    learner = new CompressorLearner(tmp);
  });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("recommended tier does not change with small cost fluctuations", async () => {
    // Seed tier 1 with consistently low cost (ROI clearly above threshold)
    const makeBaselineAndTier1Records = (tier1Cost: number) =>
      [
        ...Array.from({ length: 5 }, () => ({
          tier: 4 as const, messageCount: 10, provider: "p1",
          estimated_tokens: 1000, actual_tokens: 1000,
          actual_cost_usd: 0.020, success: true, recordedAt: new Date().toISOString(),
        })),
        ...Array.from({ length: 5 }, () => ({
          tier: 1 as const, messageCount: 10, provider: "p1",
          estimated_tokens: 1000, actual_tokens: 1000,
          actual_cost_usd: tier1Cost, success: true, recordedAt: new Date().toISOString(),
        })),
      ] as CostHistoryRecord[];

    // Small fluctuations around 0.004: ROI should remain positive and recommend tier 1.
    const tiers = await Promise.all(
      [0.003, 0.004, 0.0035, 0.0042, 0.0038].map((cost) =>
        learner.computeROI("p1", 10, makeBaselineAndTier1Records(cost)).then((b) => b.recommendedTier),
      ),
    );
    // All should be non-null and consistent (no thrashing)
    const unique = new Set(tiers.filter((t) => t !== null));
    expect(unique.size).toBeLessThanOrEqual(1);
  });

  test("tier recommendation is stable over repeated computeROI calls with same data", async () => {
    const records: CostHistoryRecord[] = [
      ...Array.from({ length: 5 }, () => ({
        tier: 4 as const, messageCount: 10, provider: "p1",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.015, success: true, recordedAt: new Date().toISOString(),
      })),
      ...Array.from({ length: 5 }, () => ({
        tier: 2 as const, messageCount: 10, provider: "p1",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.003, success: true, recordedAt: new Date().toISOString(),
      })),
    ];
    // Call computeROI 5 times with same data — result must be identical each time.
    const results = await Promise.all(
      Array.from({ length: 5 }, () => learner.computeROI("p1", 10, records)),
    );
    const tiers = results.map((r) => r.recommendedTier);
    expect(new Set(tiers).size).toBe(1);
  });

  test("no recommendation flip when new records are within normal variance", async () => {
    // Start with 5 tier-1 records cheaply; add 5 more with slight increase.
    // The median should still favour tier 1.
    const baseRecords: CostHistoryRecord[] = [
      ...Array.from({ length: 5 }, () => ({
        tier: 4 as const, messageCount: 10, provider: "p1",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.020, success: true, recordedAt: new Date().toISOString(),
      })),
      ...Array.from({ length: 5 }, () => ({
        tier: 1 as const, messageCount: 10, provider: "p1",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.004, success: true, recordedAt: new Date().toISOString(),
      })),
    ];
    const extendedRecords: CostHistoryRecord[] = [
      ...baseRecords,
      ...Array.from({ length: 5 }, () => ({
        tier: 1 as const, messageCount: 10, provider: "p1",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.0045, // slightly higher but still cheap
        success: true, recordedAt: new Date().toISOString(),
      })),
    ];

    const t1 = (await learner.computeROI("p1", 10, baseRecords)).recommendedTier;
    const t2 = (await learner.computeROI("p1", 10, extendedRecords)).recommendedTier;
    // Both should recommend tier 1 (no flip)
    expect(t1).toBe(t2);
  });
});

// ---------------------------------------------------------------------------
// Graceful degradation when history is sparse
// ---------------------------------------------------------------------------

describe("Graceful degradation on sparse history", () => {
  let tmp: string;
  let learner: CompressorLearner;

  beforeEach(async () => {
    tmp = await makeTmpDir();
    learner = new CompressorLearner(tmp);
  });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("returns null recommendedTier when no history exists", async () => {
    const breakdown = await learner.computeROI("p1", 10);
    expect(breakdown.recommendedTier).toBeNull();
  });

  test("returns null recommendedTier when < MIN_COST_SAMPLES per tier", async () => {
    // Only 2 records per tier (below threshold of 3)
    const records: CostHistoryRecord[] = [
      ...Array.from({ length: 2 }, () => ({
        tier: 4 as const, messageCount: 10, provider: "p1",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.020, success: true, recordedAt: new Date().toISOString(),
      })),
      ...Array.from({ length: 2 }, () => ({
        tier: 1 as const, messageCount: 10, provider: "p1",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.001, success: true, recordedAt: new Date().toISOString(),
      })),
    ];
    const breakdown = await learner.computeROI("p1", 10, records);
    expect(breakdown.recommendedTier).toBeNull();
    expect(breakdown.byTier[1].recommended).toBe(false);
  });

  test("recommended=false for tiers with insufficient samples even if ROI would be high", async () => {
    // 1 record for tier 1 showing huge savings, but sample count too low
    const records: CostHistoryRecord[] = [
      { tier: 4, messageCount: 10, provider: "p1", estimated_tokens: 1000, actual_tokens: 1000, actual_cost_usd: 0.100, success: true, recordedAt: new Date().toISOString() },
      { tier: 1, messageCount: 10, provider: "p1", estimated_tokens: 1000, actual_tokens: 1000, actual_cost_usd: 0.001, success: true, recordedAt: new Date().toISOString() },
    ];
    const breakdown = await learner.computeROI("p1", 10, records);
    expect(breakdown.byTier[1].recommended).toBe(false);
  });

  test("exactly MIN_COST_SAMPLES records crosses the threshold", async () => {
    // 3 records for both tier 4 (expensive) and tier 1 (cheap) → should trigger recommendation
    const records: CostHistoryRecord[] = [
      ...Array.from({ length: 3 }, () => ({
        tier: 4 as const, messageCount: 10, provider: "p1",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.020, success: true, recordedAt: new Date().toISOString(),
      })),
      ...Array.from({ length: 3 }, () => ({
        tier: 1 as const, messageCount: 10, provider: "p1",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.002, success: true, recordedAt: new Date().toISOString(),
      })),
    ];
    const breakdown = await learner.computeROI("p1", 10, records);
    expect(breakdown.byTier[1].sampleCount).toBe(3);
    // ROI should be positive and recommended should be true (large savings)
    expect(breakdown.byTier[1].roi).toBeGreaterThan(0.15);
    expect(breakdown.byTier[1].recommended).toBe(true);
    expect(breakdown.recommendedTier).not.toBeNull();
  });

  test("selectTier falls back to fallbackTier when history is empty", async () => {
    const tier = await learner.selectTier("p1", 10, 3 as const);
    expect(tier).toBe(3);
  });

  test("selectTier falls back when all records are for a different provider", async () => {
    // Records exist but for a different provider
    const records: CostHistoryRecord[] = Array.from({ length: 5 }, () => ({
      tier: 1 as const, messageCount: 10, provider: "other-provider",
      estimated_tokens: 1000, actual_tokens: 1000,
      actual_cost_usd: 0.001, success: true, recordedAt: new Date().toISOString(),
    }));
    const breakdown = await learner.computeROI("p1", 10, records);
    expect(breakdown.recommendedTier).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Provider-switch cost absorption
// ---------------------------------------------------------------------------

describe("Provider-switch cost absorption", () => {
  let tmp: string;
  let learner: CompressorLearner;

  beforeEach(async () => {
    tmp = await makeTmpDir();
    learner = new CompressorLearner(tmp);
  });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("records for provider A do not influence ROI for provider B", async () => {
    // Provider A: tier 1 very cheap → should recommend tier 1
    const recordsA: CostHistoryRecord[] = [
      ...Array.from({ length: 5 }, () => ({
        tier: 4 as const, messageCount: 10, provider: "provider-a",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.020, success: true, recordedAt: new Date().toISOString(),
      })),
      ...Array.from({ length: 5 }, () => ({
        tier: 1 as const, messageCount: 10, provider: "provider-a",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.002, success: true, recordedAt: new Date().toISOString(),
      })),
    ];
    // Provider B: no records
    const allRecords = recordsA; // Only provider-a records

    const breakdownA = await learner.computeROI("provider-a", 10, allRecords);
    const breakdownB = await learner.computeROI("provider-b", 10, allRecords);

    expect(breakdownA.recommendedTier).not.toBeNull();
    expect(breakdownB.recommendedTier).toBeNull(); // no provider-b records
  });

  test("switching from cheap to expensive provider starts fresh (no carryover)", async () => {
    // Provider A history says tier 1 is cheap; but we ask about provider B
    // Provider B should have no preconceived ROI.
    const records: CostHistoryRecord[] = Array.from({ length: 5 }, () => ({
      tier: 1 as const, messageCount: 10, provider: "cheap-provider",
      estimated_tokens: 1000, actual_tokens: 1000,
      actual_cost_usd: 0.001, success: true, recordedAt: new Date().toISOString(),
    }));

    const breakdown = await learner.computeROI("expensive-provider", 10, records);
    // No records for expensive-provider → all sampleCounts are 0, no recommendation
    for (const t of [1, 2, 3, 4] as const) {
      expect(breakdown.byTier[t].sampleCount).toBe(0);
    }
    expect(breakdown.recommendedTier).toBeNull();
  });

  test("learner handles multiple providers in same history file independently", async () => {
    // Mix of providers in the same record set
    const records: CostHistoryRecord[] = [
      ...Array.from({ length: 5 }, () => ({
        tier: 4 as const, messageCount: 10, provider: "anthropic",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.015, success: true, recordedAt: new Date().toISOString(),
      })),
      ...Array.from({ length: 5 }, () => ({
        tier: 2 as const, messageCount: 10, provider: "anthropic",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.002, success: true, recordedAt: new Date().toISOString(),
      })),
      ...Array.from({ length: 5 }, () => ({
        tier: 4 as const, messageCount: 10, provider: "openrouter",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.001, success: true, recordedAt: new Date().toISOString(),
      })),
      ...Array.from({ length: 5 }, () => ({
        tier: 2 as const, messageCount: 10, provider: "openrouter",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.005, success: true, recordedAt: new Date().toISOString(),
      })),
    ];

    const anthropicBreakdown = await learner.computeROI("anthropic", 10, records);
    const openrouterBreakdown = await learner.computeROI("openrouter", 10, records);

    // anthropic: tier 2 is cheaper than tier 4 → positive ROI → recommend tier 2
    expect(anthropicBreakdown.byTier[2].roi).toBeGreaterThan(0);
    // openrouter: tier 2 is MORE expensive than tier 4 → negative ROI → no recommendation for tier 2
    expect(openrouterBreakdown.byTier[2].roi).toBeLessThan(0);
    expect(openrouterBreakdown.byTier[2].recommended).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Integration with selectCompressionTierAdaptive via roiBreakdown param
// ---------------------------------------------------------------------------

describe("selectCompressionTierAdaptive — roiBreakdown integration", () => {
  function textMessages(n: number): Message[] {
    return Array.from({ length: n }, (_, i) => textMsg(i % 2 === 0 ? "user" : "assistant", `msg ${i}`));
  }

  function makeROIBreakdown(recommendedTier: 1 | 2 | 3 | 4 | null): TierROIBreakdown {
    const entry = (t: 1 | 2 | 3 | 4, recommended: boolean) => ({
      tier: t,
      sampleCount: recommended ? 10 : 0,
      medianCostUsd: 0.002,
      roi: recommended ? 0.5 : 0,
      recommended,
    });
    return {
      provider: "p1",
      bucket: "sm" as const,
      roiThreshold: 0.15,
      byTier: {
        1: entry(1, recommendedTier === 1),
        2: entry(2, recommendedTier === 2),
        3: entry(3, recommendedTier === 3),
        4: entry(4, recommendedTier === 4),
      },
      recommendedTier,
      computedAt: new Date().toISOString(),
    };
  }

  test("roiBreakdown with recommendedTier=1 overrides history and returns tier 1", () => {
    const msgs = textMessages(8);
    // History says tier 4 is fine (high success, no overshoot)
    const history = {
      byTier: {
        1: { tier: 1 as const, sampleCount: 5, successRate: 0.5, avgOvershootPct: 0 },
        2: { tier: 2 as const, sampleCount: 5, successRate: 0.9, avgOvershootPct: 0 },
        3: { tier: 3 as const, sampleCount: 5, successRate: 0.95, avgOvershootPct: 0 },
        4: { tier: 4 as const, sampleCount: 5, successRate: 0.99, avgOvershootPct: 0 },
      },
    };
    const roiBreakdown = makeROIBreakdown(1);
    const tier = selectCompressionTierAdaptive(
      msgs, 0, { maxContextTokens: 1_000_000, reserveTokens: 8192 },
      history, null, null, null, true, roiBreakdown,
    );
    expect(tier).toBe(1);
  });

  test("roiBreakdown with recommendedTier=2 overrides static/history tier", () => {
    const msgs = textMessages(8);
    const roiBreakdown = makeROIBreakdown(2);
    const tier = selectCompressionTierAdaptive(
      msgs, 0, { maxContextTokens: 1_000_000, reserveTokens: 8192 },
      null, null, null, null, true, roiBreakdown,
    );
    expect(tier).toBe(2);
  });

  test("null roiBreakdown does not change tier selection", () => {
    const msgs = textMessages(5);
    const withoutROI = selectCompressionTierAdaptive(
      msgs, 0, { maxContextTokens: 1_000_000, reserveTokens: 8192 },
      null, null, null, null, true,
    );
    const withNullROI = selectCompressionTierAdaptive(
      msgs, 0, { maxContextTokens: 1_000_000, reserveTokens: 8192 },
      null, null, null, null, true, null,
    );
    expect(withNullROI).toBe(withoutROI);
  });

  test("roiBreakdown with null recommendedTier does not change tier selection", () => {
    const msgs = textMessages(5);
    const withoutROI = selectCompressionTierAdaptive(
      msgs, 0, { maxContextTokens: 1_000_000, reserveTokens: 8192 },
      null, null, null, null, true,
    );
    const withNullRecommended = selectCompressionTierAdaptive(
      msgs, 0, { maxContextTokens: 1_000_000, reserveTokens: 8192 },
      null, null, null, null, true, makeROIBreakdown(null),
    );
    expect(withNullRecommended).toBe(withoutROI);
  });

  test("roiBreakdown overrides consolidation profile", () => {
    // Consolidation profile says tier 3; ROI says tier 1 → ROI wins (applied last)
    const msgs = textMessages(8);
    const consolidationProfile = {
      byProvider: {
        "p1": [{ tier: 3 as const, sampleCount: 10, avgCostSaved: 0.005 }],
      },
    };
    const history = {
      byTier: {
        1: { tier: 1 as const, sampleCount: 5, successRate: 0.9, avgOvershootPct: 0 },
        2: { tier: 2 as const, sampleCount: 5, successRate: 0.9, avgOvershootPct: 0 },
        3: { tier: 3 as const, sampleCount: 5, successRate: 0.9, avgOvershootPct: 0 },
        4: { tier: 4 as const, sampleCount: 5, successRate: 0.9, avgOvershootPct: 0 },
      },
    };
    const roiBreakdown = makeROIBreakdown(1);
    const tier = selectCompressionTierAdaptive(
      msgs, 0, { maxContextTokens: 1_000_000, reserveTokens: 8192 },
      history, null, "p1", consolidationProfile as any, true, roiBreakdown,
    );
    expect(tier).toBe(1);
  });

  test("roiBreakdown with recommendedTier=3 applied over static selection", () => {
    // Static would pick tier 4 for small messages + huge budget.
    // ROI says tier 3 → should return 3.
    const msgs = textMessages(6);
    const roiBreakdown = makeROIBreakdown(3);
    const tier = selectCompressionTierAdaptive(
      msgs, 0, { maxContextTokens: 1_000_000, reserveTokens: 8192 },
      null, null, null, null, true, roiBreakdown,
    );
    expect(tier).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// End-to-end: record → computeROI → selectTier
// ---------------------------------------------------------------------------

describe("End-to-end: record → computeROI → selectTier", () => {
  let tmp: string;
  let learner: CompressorLearner;

  beforeEach(async () => {
    tmp = await makeTmpDir();
    learner = new CompressorLearner(tmp);
  });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("after seeding expensive tier-4 and cheap tier-1, selectTier recommends tier 1", async () => {
    // Seed: tier 4 baseline is expensive; tier 1 cuts cost significantly
    for (let i = 0; i < 5; i++) {
      await learner.record(makeRecord({ tier: 4, messageCount: 15, provider: "p1", actual_cost_usd: 0.025 }));
      await learner.record(makeRecord({ tier: 1, messageCount: 15, provider: "p1", actual_cost_usd: 0.003 }));
    }
    const selected = await learner.selectTier("p1", 15, 4 as const);
    expect(selected).toBe(1);
  });

  test("readHistory returns all seeded records", async () => {
    await learner.record(makeRecord({ tier: 2, messageCount: 20, provider: "p1", actual_cost_usd: 0.004 }));
    await learner.record(makeRecord({ tier: 3, messageCount: 25, provider: "p1", actual_cost_usd: 0.002 }));
    const records = await learner.readHistory();
    expect(records.length).toBe(2);
    expect(records[0]!.tier).toBe(2);
    expect(records[1]!.tier).toBe(3);
  });

  test("custom roiThreshold changes recommendation boundary", async () => {
    // With threshold=0.01 (very low bar), tier 2 should be recommended
    // With threshold=100 (impossibly high), nothing is recommended
    const records: CostHistoryRecord[] = [
      ...Array.from({ length: 5 }, () => ({
        tier: 4 as const, messageCount: 10, provider: "p1",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.010, success: true, recordedAt: new Date().toISOString(),
      })),
      ...Array.from({ length: 5 }, () => ({
        tier: 2 as const, messageCount: 10, provider: "p1",
        estimated_tokens: 1000, actual_tokens: 1000,
        actual_cost_usd: 0.008, success: true, recordedAt: new Date().toISOString(),
      })),
    ];
    const lowThresholdLearner = new CompressorLearner(tmp, 0.01);
    const highThresholdLearner = new CompressorLearner(tmp, 100);

    const lowResult = await lowThresholdLearner.computeROI("p1", 10, records);
    const highResult = await highThresholdLearner.computeROI("p1", 10, records);

    // With a very low threshold, the small cost savings should be recommended
    expect(lowResult.byTier[2].recommended).toBe(true);
    // With a very high threshold, nothing clears the bar
    expect(highResult.byTier[2].recommended).toBe(false);
    expect(highResult.recommendedTier).toBeNull();
  });
});
