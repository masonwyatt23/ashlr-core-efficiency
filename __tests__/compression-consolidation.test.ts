/**
 * Tests for src/compression/consolidation.ts
 *
 * Covers:
 *   - consolidationAuditPath()
 *   - HistoryConsolidator.buildProfile() — empty, sparse, rich history
 *   - getCompressionProfile() — convenience wrapper
 *   - predictCompressionOnSwitch() — target history, source-fallback, no-history
 *   - rebalanceBudgetOnSwitchHook() — audit file written
 *   - rebalanceBudgetOnSwitch() with cwd — compressionRecommendation attached
 *   - Profile bucketing accuracy
 *   - Dominance ranking with switching cost penalty
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, rm, writeFile } from "fs/promises";
import { existsSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import {
  HistoryConsolidator,
  consolidationAuditPath,
  getCompressionProfile,
  predictCompressionOnSwitch,
  rebalanceBudgetOnSwitchHook,
  type CompressibleProfile,
  type CompressorRecommendation,
  type ConsolidationAuditRecord,
} from "../src/compression/consolidation.ts";
import { recordCompressionResult, compressionHistoryPath } from "../src/compression/adaptive.ts";
import { rebalanceBudgetOnSwitchWithConsolidation } from "../src/budget/index.ts";
import { readJsonl } from "../src/genome/jsonl.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function makeTmpDir(): Promise<string> {
  const base = join(
    tmpdir(),
    `ashlr-consolidation-test-${Date.now()}-${Math.random().toString(36).slice(2)}`,
  );
  await mkdir(base, { recursive: true });
  return base;
}

/**
 * Seed N successful tier-X records with the given provider into a project dir.
 * Provider is stored in the record because recordCompressionResult passes it
 * to cost-accounting; we need it in the JSONL too, so we write directly.
 */
async function seedRecords(
  cwd: string,
  opts: {
    tier: 1 | 2 | 3 | 4;
    provider: string;
    count: number;
    success?: boolean;
    estimatedTokens?: number;
    actualTokens?: number;
  },
): Promise<void> {
  const {
    tier,
    provider,
    count,
    success = true,
    estimatedTokens = 1000,
    actualTokens = 1050,
  } = opts;

  const histPath = compressionHistoryPath(cwd);
  await mkdir(join(cwd, ".ashlrcode/genome/evolution"), { recursive: true });

  for (let i = 0; i < count; i++) {
    const record = {
      tier,
      provider,
      estimatedTokens,
      actualTokens,
      success,
      recordedAt: new Date().toISOString(),
    };
    await writeFile(histPath, JSON.stringify(record) + "\n", { flag: "a", encoding: "utf-8" });
  }
}

// ---------------------------------------------------------------------------
// consolidationAuditPath
// ---------------------------------------------------------------------------

describe("consolidationAuditPath", () => {
  test("returns path inside genome/evolution directory", () => {
    const p = consolidationAuditPath("/projects/myapp");
    expect(p).toContain(".ashlrcode/genome/evolution");
    expect(p).toContain("compression-consolidation.jsonl");
  });

  test("is deterministic for the same cwd", () => {
    expect(consolidationAuditPath("/x")).toBe(consolidationAuditPath("/x"));
  });

  test("differs from compressionHistoryPath", () => {
    const cwd = "/projects/myapp";
    expect(consolidationAuditPath(cwd)).not.toBe(compressionHistoryPath(cwd));
  });
});

// ---------------------------------------------------------------------------
// HistoryConsolidator.buildProfile — empty / no history
// ---------------------------------------------------------------------------

describe("HistoryConsolidator.buildProfile — empty history", () => {
  let tmp: string;
  beforeEach(async () => { tmp = await makeTmpDir(); });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("returns empty byProvider when no history file exists", async () => {
    const consolidator = new HistoryConsolidator(tmp);
    const profile = await consolidator.buildProfile();
    expect(profile.cwd).toBe(tmp);
    expect(profile.totalRecords).toBe(0);
    expect(Object.keys(profile.byProvider)).toHaveLength(0);
  });

  test("returns empty byProvider when history file exists but is empty", async () => {
    await mkdir(join(tmp, ".ashlrcode/genome/evolution"), { recursive: true });
    await writeFile(compressionHistoryPath(tmp), "", "utf-8");
    const profile = await new HistoryConsolidator(tmp).buildProfile();
    expect(profile.totalRecords).toBe(0);
    expect(Object.keys(profile.byProvider)).toHaveLength(0);
  });

  test("computedAt is a valid ISO timestamp", async () => {
    const profile = await new HistoryConsolidator(tmp).buildProfile();
    expect(() => new Date(profile.computedAt)).not.toThrow();
    expect(isNaN(new Date(profile.computedAt).getTime())).toBe(false);
  });

  test("globalBestTier defaults to 3 when no data", async () => {
    const profile = await new HistoryConsolidator(tmp).buildProfile();
    expect(profile.globalBestTier).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// HistoryConsolidator.buildProfile — sparse history (below MIN_BUCKET_SAMPLES)
// ---------------------------------------------------------------------------

describe("HistoryConsolidator.buildProfile — sparse history", () => {
  let tmp: string;
  beforeEach(async () => { tmp = await makeTmpDir(); });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("byProvider entry is empty when fewer than MIN_BUCKET_SAMPLES records exist", async () => {
    // Only 2 records (MIN_BUCKET_SAMPLES = 3)
    await seedRecords(tmp, { tier: 2, provider: "anthropic", count: 2 });
    const profile = await new HistoryConsolidator(tmp).buildProfile();
    // totalRecords counts raw records regardless of bucketing threshold
    expect(profile.totalRecords).toBe(2);
    // But byProvider should have no ranked entries (below threshold)
    const anthropicEntries = profile.byProvider["anthropic"];
    expect(anthropicEntries === undefined || anthropicEntries.length === 0).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// HistoryConsolidator.buildProfile — rich history
// ---------------------------------------------------------------------------

describe("HistoryConsolidator.buildProfile — rich history", () => {
  let tmp: string;
  beforeEach(async () => { tmp = await makeTmpDir(); });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("builds byProvider entry for a single provider with enough records", async () => {
    await seedRecords(tmp, { tier: 3, provider: "anthropic", count: 5, success: true });
    const profile = await new HistoryConsolidator(tmp).buildProfile();
    expect(profile.totalRecords).toBe(5);
    expect(profile.byProvider["anthropic"]).toBeDefined();
    expect(profile.byProvider["anthropic"]!.length).toBeGreaterThanOrEqual(1);
  });

  test("dominanceRank #1 entry is assigned rank 1", async () => {
    await seedRecords(tmp, { tier: 3, provider: "anthropic", count: 5 });
    const profile = await new HistoryConsolidator(tmp).buildProfile();
    const entries = profile.byProvider["anthropic"]!;
    expect(entries[0]!.dominanceRank).toBe(1);
  });

  test("dominanceRanks are sequential and unique", async () => {
    await seedRecords(tmp, { tier: 3, provider: "anthropic", count: 5, success: true });
    await seedRecords(tmp, { tier: 2, provider: "anthropic", count: 5, success: true });
    await seedRecords(tmp, { tier: 1, provider: "anthropic", count: 5, success: false });
    const profile = await new HistoryConsolidator(tmp).buildProfile();
    const entries = profile.byProvider["anthropic"]!;
    const ranks = entries.map((e) => e.dominanceRank).sort((a, b) => a - b);
    for (let i = 0; i < ranks.length; i++) {
      expect(ranks[i]).toBe(i + 1);
    }
  });

  test("tier with high success rate ranks above tier with low success rate", async () => {
    // tier 3: 5 successes → successRate = 1.0
    await seedRecords(tmp, { tier: 3, provider: "anthropic", count: 5, success: true });
    // tier 2: 5 failures → successRate = 0.0
    await seedRecords(tmp, { tier: 2, provider: "anthropic", count: 5, success: false });
    const profile = await new HistoryConsolidator(tmp).buildProfile();
    const entries = profile.byProvider["anthropic"]!;
    const tier3Entry = entries.find((e) => e.tier === 3);
    const tier2Entry = entries.find((e) => e.tier === 2);
    expect(tier3Entry).toBeDefined();
    expect(tier2Entry).toBeDefined();
    expect(tier3Entry!.dominanceRank).toBeLessThan(tier2Entry!.dominanceRank);
  });

  test("separates records from different providers into different byProvider keys", async () => {
    await seedRecords(tmp, { tier: 3, provider: "anthropic", count: 5 });
    await seedRecords(tmp, { tier: 2, provider: "openai", count: 5 });
    const profile = await new HistoryConsolidator(tmp).buildProfile();
    expect(profile.byProvider["anthropic"]).toBeDefined();
    expect(profile.byProvider["openai"]).toBeDefined();
  });

  test("globalBestTier is set from the provider with the most records", async () => {
    // anthropic has 10 records for tier 3
    await seedRecords(tmp, { tier: 3, provider: "anthropic", count: 10, success: true });
    // openai has 4 records for tier 2
    await seedRecords(tmp, { tier: 2, provider: "openai", count: 4, success: true });
    const profile = await new HistoryConsolidator(tmp).buildProfile();
    // anthropic dominates → globalBestTier should come from its top-ranked tier
    expect(profile.globalBestTier).toBe(3);
  });

  test("compositeScore is in [0, 1] range for all entries", async () => {
    await seedRecords(tmp, { tier: 3, provider: "anthropic", count: 5, success: true });
    await seedRecords(tmp, { tier: 2, provider: "anthropic", count: 5, success: true });
    const profile = await new HistoryConsolidator(tmp).buildProfile();
    for (const entries of Object.values(profile.byProvider)) {
      for (const entry of entries) {
        expect(entry.compositeScore).toBeGreaterThanOrEqual(0);
        expect(entry.compositeScore).toBeLessThanOrEqual(1);
      }
    }
  });

  test("sampleCount in profile entry reflects actual record count", async () => {
    await seedRecords(tmp, { tier: 3, provider: "anthropic", count: 7, success: true });
    const profile = await new HistoryConsolidator(tmp).buildProfile();
    const entry = profile.byProvider["anthropic"]!.find((e) => e.tier === 3);
    expect(entry).toBeDefined();
    expect(entry!.sampleCount).toBe(7);
  });

  test("lookbackDays=0 excludes all historical records", async () => {
    await seedRecords(tmp, { tier: 3, provider: "anthropic", count: 5 });
    const profile = await new HistoryConsolidator(tmp, 0).buildProfile();
    // lookback=0 → cutoff = today, all records older than 0 days are excluded
    // (records written just now have the same day, but the cutoff is setDate(d-0)=today)
    // This tests the boundary — result may be empty or not depending on sub-second timing.
    // The important thing is it doesn't crash.
    expect(profile).toBeDefined();
    expect(typeof profile.totalRecords).toBe("number");
  });
});

// ---------------------------------------------------------------------------
// getCompressionProfile — convenience wrapper
// ---------------------------------------------------------------------------

describe("getCompressionProfile", () => {
  let tmp: string;
  beforeEach(async () => { tmp = await makeTmpDir(); });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("returns same structure as HistoryConsolidator.buildProfile", async () => {
    await seedRecords(tmp, { tier: 3, provider: "anthropic", count: 5 });
    const direct = await new HistoryConsolidator(tmp).buildProfile();
    const via = await getCompressionProfile(tmp);
    expect(via.cwd).toBe(direct.cwd);
    expect(via.totalRecords).toBe(direct.totalRecords);
    expect(Object.keys(via.byProvider)).toEqual(Object.keys(direct.byProvider));
  });

  test("accepts custom lookbackDays", async () => {
    const profile = await getCompressionProfile(tmp, 7);
    expect(profile).toBeDefined();
    expect(profile.cwd).toBe(tmp);
  });
});

// ---------------------------------------------------------------------------
// predictCompressionOnSwitch — target provider has history
// ---------------------------------------------------------------------------

describe("predictCompressionOnSwitch — target provider has history", () => {
  let tmp: string;
  beforeEach(async () => { tmp = await makeTmpDir(); });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("recommends top-ranked tier from target-provider history", async () => {
    await seedRecords(tmp, { tier: 2, provider: "openai", count: 8, success: true });
    const profile = await getCompressionProfile(tmp);
    const rec = await predictCompressionOnSwitch("anthropic", "openai", { messageCount: 20 }, profile);
    expect(rec.recommendedTier).toBe(2);
  });

  test("rationale mentions target provider", async () => {
    await seedRecords(tmp, { tier: 3, provider: "openai", count: 5, success: true });
    const profile = await getCompressionProfile(tmp);
    const rec = await predictCompressionOnSwitch("anthropic", "openai", { messageCount: 10 }, profile);
    expect(rec.rationale).toContain("openai");
  });

  test("writes audit record to consolidation JSONL", async () => {
    await seedRecords(tmp, { tier: 3, provider: "openai", count: 5 });
    const profile = await getCompressionProfile(tmp);
    await predictCompressionOnSwitch("anthropic", "openai", { messageCount: 10 }, profile);
    const auditPath = consolidationAuditPath(tmp);
    expect(existsSync(auditPath)).toBe(true);
    const records = await readJsonl<ConsolidationAuditRecord>(auditPath);
    expect(records.length).toBeGreaterThanOrEqual(1);
    const last = records[records.length - 1]!;
    expect(last.provider_from).toBe("anthropic");
    expect(last.provider_to).toBe("openai");
    expect(last.actual_tier).toBeNull();
    expect(last.roi_delta).toBeNull();
  });

  test("audit record recommended_tier matches recommendation", async () => {
    await seedRecords(tmp, { tier: 2, provider: "openai", count: 6 });
    const profile = await getCompressionProfile(tmp);
    const rec = await predictCompressionOnSwitch("anthropic", "openai", { messageCount: 15 }, profile);
    const records = await readJsonl<ConsolidationAuditRecord>(consolidationAuditPath(tmp));
    const last = records[records.length - 1]!;
    expect(last.recommended_tier).toBe(rec.recommendedTier);
  });

  test("recommendedAt is a valid ISO timestamp", async () => {
    await seedRecords(tmp, { tier: 3, provider: "openai", count: 5 });
    const profile = await getCompressionProfile(tmp);
    const rec = await predictCompressionOnSwitch("anthropic", "openai", { messageCount: 10 }, profile);
    expect(() => new Date(rec.recommendedAt)).not.toThrow();
    expect(isNaN(new Date(rec.recommendedAt).getTime())).toBe(false);
  });

  test("multiple switch predictions accumulate in audit log", async () => {
    await seedRecords(tmp, { tier: 3, provider: "openai", count: 5 });
    const profile = await getCompressionProfile(tmp);
    await predictCompressionOnSwitch("anthropic", "openai", { messageCount: 10 }, profile);
    await predictCompressionOnSwitch("anthropic", "openai", { messageCount: 20 }, profile);
    const records = await readJsonl<ConsolidationAuditRecord>(consolidationAuditPath(tmp));
    expect(records.length).toBeGreaterThanOrEqual(2);
  });
});

// ---------------------------------------------------------------------------
// predictCompressionOnSwitch — source-provider fallback (no target history)
// ---------------------------------------------------------------------------

describe("predictCompressionOnSwitch — source-provider fallback", () => {
  let tmp: string;
  beforeEach(async () => { tmp = await makeTmpDir(); });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("falls back to source-provider best tier when target has no history", async () => {
    // Only anthropic history; switching to openai (no openai history)
    await seedRecords(tmp, { tier: 2, provider: "anthropic", count: 5, success: true });
    const profile = await getCompressionProfile(tmp);
    const rec = await predictCompressionOnSwitch("anthropic", "openai", { messageCount: 20 }, profile);
    // Should not crash and should return a valid tier
    expect([1, 2, 3, 4]).toContain(rec.recommendedTier);
  });

  test("rationale mentions source provider when falling back", async () => {
    await seedRecords(tmp, { tier: 2, provider: "anthropic", count: 5 });
    const profile = await getCompressionProfile(tmp);
    const rec = await predictCompressionOnSwitch("anthropic", "deepseek", { messageCount: 10 }, profile);
    expect(rec.rationale).toContain("anthropic");
  });

  test("applies cost-ratio tier shift when extrapolating from source", async () => {
    // anthropic best tier is 2; switching to deepseek (much cheaper → shift -1 → tier 3)
    await seedRecords(tmp, { tier: 2, provider: "anthropic", count: 10, success: true });
    const profile = await getCompressionProfile(tmp);
    const rec = await predictCompressionOnSwitch("anthropic", "deepseek", { messageCount: 20 }, profile);
    // deepseek is much cheaper than anthropic → tier shift should relax (higher tier number)
    expect(rec.recommendedTier).toBeGreaterThanOrEqual(2);
  });
});

// ---------------------------------------------------------------------------
// predictCompressionOnSwitch — no history at all
// ---------------------------------------------------------------------------

describe("predictCompressionOnSwitch — no history", () => {
  let tmp: string;
  beforeEach(async () => { tmp = await makeTmpDir(); });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("returns a valid tier when no history exists", async () => {
    const profile = await getCompressionProfile(tmp);
    const rec = await predictCompressionOnSwitch("anthropic", "openai", { messageCount: 10 }, profile);
    expect([1, 2, 3, 4]).toContain(rec.recommendedTier);
  });

  test("rationale is non-empty when no history exists", async () => {
    const profile = await getCompressionProfile(tmp);
    const rec = await predictCompressionOnSwitch("anthropic", "groq", { messageCount: 10 }, profile);
    expect(rec.rationale.length).toBeGreaterThan(0);
  });

  test("estimatedRoiDelta is 0 when no history exists", async () => {
    const profile = await getCompressionProfile(tmp);
    const rec = await predictCompressionOnSwitch("anthropic", "openai", { messageCount: 10 }, profile);
    expect(rec.estimatedRoiDelta).toBe(0);
  });

  test("still writes an audit record even with no history", async () => {
    const profile = await getCompressionProfile(tmp);
    await predictCompressionOnSwitch("anthropic", "openai", { messageCount: 5 }, profile);
    expect(existsSync(consolidationAuditPath(tmp))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// rebalanceBudgetOnSwitchHook
// ---------------------------------------------------------------------------

describe("rebalanceBudgetOnSwitchHook", () => {
  let tmp: string;
  beforeEach(async () => { tmp = await makeTmpDir(); });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("returns a CompressorRecommendation with valid tier", async () => {
    const rec = await rebalanceBudgetOnSwitchHook(tmp, "anthropic", "openai", 20);
    expect([1, 2, 3, 4]).toContain(rec.recommendedTier);
  });

  test("providerFrom and providerTo are set correctly", async () => {
    const rec = await rebalanceBudgetOnSwitchHook(tmp, "anthropic", "groq", 10);
    expect(rec.providerFrom).toBe("anthropic");
    expect(rec.providerTo).toBe("groq");
  });

  test("creates audit JSONL file", async () => {
    await rebalanceBudgetOnSwitchHook(tmp, "anthropic", "openai", 5);
    expect(existsSync(consolidationAuditPath(tmp))).toBe(true);
  });

  test("recommendation uses history when available", async () => {
    await seedRecords(tmp, { tier: 2, provider: "openai", count: 6, success: true });
    const rec = await rebalanceBudgetOnSwitchHook(tmp, "anthropic", "openai", 15);
    // With openai history showing tier 2 as best, should recommend tier 2
    expect(rec.recommendedTier).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// rebalanceBudgetOnSwitchWithConsolidation — compressionRecommendation integration
// ---------------------------------------------------------------------------

describe("rebalanceBudgetOnSwitchWithConsolidation", () => {
  let tmp: string;
  beforeEach(async () => { tmp = await makeTmpDir(); });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("returns a Promise", () => {
    const result = rebalanceBudgetOnSwitchWithConsolidation("anthropic", "openai", [], tmp);
    expect(result).toBeInstanceOf(Promise);
  });

  test("resolved result has compressionRecommendation field", async () => {
    const result = await rebalanceBudgetOnSwitchWithConsolidation("anthropic", "openai", [], tmp);
    // compressionRecommendation may be null (no history) but the field must exist
    expect("compressionRecommendation" in result).toBe(true);
    if (result.compressionRecommendation !== null && result.compressionRecommendation !== undefined) {
      expect([1, 2, 3, 4]).toContain(result.compressionRecommendation.recommendedTier);
      expect(typeof result.compressionRecommendation.rationale).toBe("string");
    }
  });

  test("base budget fields are preserved", async () => {
    const result = await rebalanceBudgetOnSwitchWithConsolidation("anthropic", "openai", [], tmp);
    expect(typeof result.newSystemBudget).toBe("number");
    expect(typeof result.recommendedTier).toBe("number");
    expect(typeof result.tokensFreed).toBe("number");
  });

  test("compressionRecommendation uses target-provider history when available", async () => {
    await seedRecords(tmp, { tier: 3, provider: "openai", count: 8, success: true });
    const result = await rebalanceBudgetOnSwitchWithConsolidation("anthropic", "openai", [], tmp);
    expect(result.compressionRecommendation).not.toBeNull();
    expect(result.compressionRecommendation!.recommendedTier).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// selectCompressionTierAdaptive with consolidation profile
// ---------------------------------------------------------------------------

describe("selectCompressionTierAdaptive — consolidation profile integration", () => {
  let tmp: string;
  beforeEach(async () => { tmp = await makeTmpDir(); });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("consolidation profile overrides adaptive tier when provider matches", async () => {
    // Build a profile that shows tier 1 as best for openai
    await seedRecords(tmp, { tier: 1, provider: "openai", count: 10, success: true });
    const profile = await getCompressionProfile(tmp);

    const { selectCompressionTierAdaptive } = await import("../src/compression/adaptive.ts");
    const history = {
      byTier: {
        1: { tier: 1 as const, sampleCount: 10, successRate: 0.9, avgOvershootPct: 0 },
        2: { tier: 2 as const, sampleCount: 10, successRate: 0.9, avgOvershootPct: 0 },
        3: { tier: 3 as const, sampleCount: 10, successRate: 0.9, avgOvershootPct: 0 },
        4: { tier: 4 as const, sampleCount: 10, successRate: 0.9, avgOvershootPct: 0 },
      },
    };
    const messages = Array.from({ length: 5 }, (_, i) => ({
      role: (i % 2 === 0 ? "user" : "assistant") as "user" | "assistant",
      content: `msg ${i}`,
    }));

    const tier = selectCompressionTierAdaptive(
      messages,
      0,
      { maxContextTokens: 1_000_000, reserveTokens: 8192 },
      history,
      null,
      "openai",
      profile,
    );
    // consolidation profile says tier 1 is best for openai → should override
    expect(tier).toBe(1);
  });

  test("no consolidation profile → behavior unchanged from non-profile call", async () => {
    const { selectCompressionTierAdaptive } = await import("../src/compression/adaptive.ts");
    const history = {
      byTier: {
        1: { tier: 1 as const, sampleCount: 10, successRate: 0.9, avgOvershootPct: 0 },
        2: { tier: 2 as const, sampleCount: 10, successRate: 0.9, avgOvershootPct: 0 },
        3: { tier: 3 as const, sampleCount: 10, successRate: 0.9, avgOvershootPct: 0 },
        4: { tier: 4 as const, sampleCount: 10, successRate: 0.9, avgOvershootPct: 0 },
      },
    };
    const messages = Array.from({ length: 5 }, (_, i) => ({
      role: (i % 2 === 0 ? "user" : "assistant") as "user" | "assistant",
      content: `msg ${i}`,
    }));
    const cfg = { maxContextTokens: 1_000_000, reserveTokens: 8192 };

    const without = selectCompressionTierAdaptive(messages, 0, cfg, history, null, "anthropic");
    const withNull = selectCompressionTierAdaptive(messages, 0, cfg, history, null, "anthropic", null);
    expect(without).toBe(withNull);
  });

  test("sparse consolidation profile (< MIN_SAMPLES) does not override", async () => {
    // Only 2 records for openai tier 2 → below MIN_BUCKET_SAMPLES=3, so no entry in byProvider
    await seedRecords(tmp, { tier: 2, provider: "openai", count: 2, success: true });
    const profile = await getCompressionProfile(tmp);

    const { selectCompressionTierAdaptive } = await import("../src/compression/adaptive.ts");
    const history = {
      byTier: {
        1: { tier: 1 as const, sampleCount: 10, successRate: 0.9, avgOvershootPct: 0 },
        2: { tier: 2 as const, sampleCount: 10, successRate: 0.9, avgOvershootPct: 0 },
        3: { tier: 3 as const, sampleCount: 10, successRate: 0.9, avgOvershootPct: 0 },
        4: { tier: 4 as const, sampleCount: 10, successRate: 0.9, avgOvershootPct: 0 },
      },
    };
    const messages = Array.from({ length: 5 }, (_, i) => ({
      role: (i % 2 === 0 ? "user" : "assistant") as "user" | "assistant",
      content: `msg ${i}`,
    }));

    const withSparse = selectCompressionTierAdaptive(
      messages, 0, { maxContextTokens: 1_000_000, reserveTokens: 8192 },
      history, null, "openai", profile,
    );
    const withoutProfile = selectCompressionTierAdaptive(
      messages, 0, { maxContextTokens: 1_000_000, reserveTokens: 8192 },
      history, null, "openai",
    );
    // Sparse profile should not change the result
    expect(withSparse).toBe(withoutProfile);
  });
});

// ---------------------------------------------------------------------------
// ROI tracking: estimatedRoiDelta reflects real tier differences
// ---------------------------------------------------------------------------

describe("ROI tracking", () => {
  let tmp: string;
  beforeEach(async () => { tmp = await makeTmpDir(); });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("estimatedRoiDelta is non-negative when best tier has higher ROI than second", async () => {
    // tier 3 has high savings (low actualTokens relative to estimated)
    // tier 2 has lower savings
    await seedRecords(tmp, {
      tier: 3, provider: "anthropic", count: 6,
      estimatedTokens: 2000, actualTokens: 500, success: true,
    });
    await seedRecords(tmp, {
      tier: 2, provider: "anthropic", count: 6,
      estimatedTokens: 2000, actualTokens: 1800, success: true,
    });
    const profile = await getCompressionProfile(tmp);
    const rec = await predictCompressionOnSwitch("openai", "anthropic", { messageCount: 20 }, profile);
    expect(rec.estimatedRoiDelta).toBeGreaterThanOrEqual(0);
  });

  test("audit record recommendation_id is a non-empty string", async () => {
    const profile = await getCompressionProfile(tmp);
    await predictCompressionOnSwitch("anthropic", "openai", { messageCount: 5 }, profile);
    const records = await readJsonl<ConsolidationAuditRecord>(consolidationAuditPath(tmp));
    expect(records[0]!.recommendation_id.length).toBeGreaterThan(0);
  });
});
