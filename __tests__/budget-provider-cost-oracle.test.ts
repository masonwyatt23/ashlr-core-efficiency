/**
 * Tests for ProviderCostOracle — EMA accuracy, cache rotation,
 * rate stability bounds, and graceful degradation when cache is missing.
 */

import { describe, expect, test, beforeEach, afterEach } from "bun:test";
import { mkdtemp, rm, mkdir, writeFile } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";

import {
  ProviderCostOracle,
  EMA_ALPHA,
  MAX_CACHE_RECORDS,
  MAX_CACHE_AGE_MS,
} from "../src/budget/provider-cost-oracle.ts";
import type { ProviderRateRecord } from "../src/budget/provider-cost-oracle.ts";
import { PROVIDER_PRICING } from "../src/budget/index.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeRecord(
  overrides: Partial<ProviderRateRecord> = {},
): ProviderRateRecord {
  return {
    provider: "anthropic",
    model: "claude-sonnet-4",
    inputRate: 3.0,
    outputRate: 15.0,
    cacheRate: 0.3,
    timestamp: new Date().toISOString(),
    ...overrides,
  };
}

async function writeCacheFile(cwd: string, records: ProviderRateRecord[]): Promise<void> {
  const dir = join(cwd, ".ashlr");
  await mkdir(dir, { recursive: true });
  const lines = records.map((r) => JSON.stringify(r)).join("\n") + "\n";
  await writeFile(join(dir, "provider-rates-cache.jsonl"), lines, "utf-8");
}

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

let tmpDir: string;

beforeEach(async () => {
  tmpDir = await mkdtemp(join(tmpdir(), "oracle-test-"));
});

afterEach(async () => {
  await rm(tmpDir, { recursive: true, force: true });
});

// ---------------------------------------------------------------------------
// Graceful degradation — missing cache
// ---------------------------------------------------------------------------

describe("graceful degradation", () => {
  test("returns static fallback when cache file is absent", async () => {
    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();

    const rate = oracle.getCalibratedRate("anthropic");
    expect(rate.observationCount).toBe(0);
    expect(rate.inputPerMToken).toBeCloseTo(PROVIDER_PRICING["anthropic"]!.inputPerMToken, 6);
    expect(rate.outputPerMToken).toBeCloseTo(PROVIDER_PRICING["anthropic"]!.outputPerMToken, 6);
    expect(rate.cacheReadPerMToken).toBeCloseTo(PROVIDER_PRICING["anthropic"]!.cacheReadPerMToken, 6);
  });

  test("returns static fallback for unknown provider", async () => {
    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();

    const rate = oracle.getCalibratedRate("unknown-llm-provider");
    expect(rate.observationCount).toBe(0);
    // Mid-range fallback values (same as computeProviderCostRatio unknown)
    expect(rate.inputPerMToken).toBeGreaterThan(0);
    expect(rate.outputPerMToken).toBeGreaterThan(0);
  });

  test("initialize() is idempotent — calling twice does not double-apply EMA", async () => {
    const records = [makeRecord({ inputRate: 6.0, outputRate: 30.0, cacheRate: 0.6 })];
    await writeCacheFile(tmpDir, records);

    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();
    await oracle.initialize(); // second call is no-op

    const rate = oracle.getCalibratedRate("anthropic");
    // Exactly one EMA application: (1-α)*3 + α*6
    const expected = (1 - EMA_ALPHA) * 3.0 + EMA_ALPHA * 6.0;
    expect(rate.inputPerMToken).toBeCloseTo(expected, 6);
  });
});

// ---------------------------------------------------------------------------
// EMA accuracy
// ---------------------------------------------------------------------------

describe("EMA accuracy", () => {
  test("single observation moves rate by α", async () => {
    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();

    const staticInput = PROVIDER_PRICING["anthropic"]!.inputPerMToken; // 3.0
    const observed = 6.0;

    await oracle.recordObservation("anthropic", "claude-sonnet-4", observed, null, null);

    const rate = oracle.getCalibratedRate("anthropic");
    const expected = (1 - EMA_ALPHA) * staticInput + EMA_ALPHA * observed;
    expect(rate.inputPerMToken).toBeCloseTo(expected, 6);
    // Output and cache should be unchanged
    expect(rate.outputPerMToken).toBeCloseTo(PROVIDER_PRICING["anthropic"]!.outputPerMToken, 6);
  });

  test("repeated identical observations converge to that value", async () => {
    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();

    const target = 10.0;
    // After many identical observations the EMA converges to the target.
    for (let i = 0; i < 200; i++) {
      await oracle.recordObservation("anthropic", "claude-sonnet-4", target, null, null);
    }

    const rate = oracle.getCalibratedRate("anthropic");
    // Within 0.01 of target after 200 steps with α=0.1
    expect(Math.abs(rate.inputPerMToken - target)).toBeLessThan(0.01);
  });

  test("EMA blends all three rate fields independently", async () => {
    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();

    const newInput = 1.5;
    const newOutput = 7.5;
    const newCache = 0.15;
    await oracle.recordObservation("anthropic", "claude-sonnet-4", newInput, newOutput, newCache);

    const rate = oracle.getCalibratedRate("anthropic");
    const p = PROVIDER_PRICING["anthropic"]!;

    expect(rate.inputPerMToken).toBeCloseTo(
      (1 - EMA_ALPHA) * p.inputPerMToken + EMA_ALPHA * newInput, 6,
    );
    expect(rate.outputPerMToken).toBeCloseTo(
      (1 - EMA_ALPHA) * p.outputPerMToken + EMA_ALPHA * newOutput, 6,
    );
    expect(rate.cacheReadPerMToken).toBeCloseTo(
      (1 - EMA_ALPHA) * p.cacheReadPerMToken + EMA_ALPHA * newCache, 6,
    );
  });

  test("null rate fields do not corrupt EMA", async () => {
    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();

    // Record with only input rate provided
    await oracle.recordObservation("anthropic", "claude-sonnet-4", 4.0, null, null);

    const rate = oracle.getCalibratedRate("anthropic");
    // Output should remain at static fallback
    expect(rate.outputPerMToken).toBeCloseTo(PROVIDER_PRICING["anthropic"]!.outputPerMToken, 6);
    // Input should be EMA-blended
    expect(rate.inputPerMToken).not.toBeCloseTo(PROVIDER_PRICING["anthropic"]!.inputPerMToken, 3);
  });

  test("negative and non-finite rates are rejected", async () => {
    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();

    const before = oracle.getCalibratedRate("anthropic");

    await oracle.recordObservation("anthropic", "claude-sonnet-4", -5.0, Infinity, NaN);

    const after = oracle.getCalibratedRate("anthropic");
    // All three fields should be unchanged since all values were invalid
    expect(after.inputPerMToken).toBeCloseTo(before.inputPerMToken, 10);
    expect(after.outputPerMToken).toBeCloseTo(before.outputPerMToken, 10);
    expect(after.cacheReadPerMToken).toBeCloseTo(before.cacheReadPerMToken, 10);
  });

  test("EMA applied to cached records on startup", async () => {
    const records = [
      makeRecord({ inputRate: 6.0, outputRate: 30.0, cacheRate: 0.6, provider: "anthropic" }),
      makeRecord({ inputRate: 6.0, outputRate: 30.0, cacheRate: 0.6, provider: "anthropic" }),
    ];
    await writeCacheFile(tmpDir, records);

    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();

    const rate = oracle.getCalibratedRate("anthropic");
    const p = PROVIDER_PRICING["anthropic"]!;

    // Two EMA applications: ema0 = static, ema1 = (1-α)*ema0 + α*6, ema2 = (1-α)*ema1 + α*6
    const ema1 = (1 - EMA_ALPHA) * p.inputPerMToken + EMA_ALPHA * 6.0;
    const ema2 = (1 - EMA_ALPHA) * ema1 + EMA_ALPHA * 6.0;
    expect(rate.inputPerMToken).toBeCloseTo(ema2, 6);
  });

  test("new unknown provider is seeded and blended correctly", async () => {
    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();

    await oracle.recordObservation("newprovider", "newmodel-v1", 2.0, 8.0, 0.2);

    const rate = oracle.getCalibratedRate("newprovider");
    expect(rate.observationCount).toBe(1);
    // Rate should be blended from the mid-range fallback (3.0 input) with observed 2.0
    const expectedInput = (1 - EMA_ALPHA) * 3.0 + EMA_ALPHA * 2.0;
    expect(rate.inputPerMToken).toBeCloseTo(expectedInput, 6);
  });
});

// ---------------------------------------------------------------------------
// Cache rotation / housekeeping
// ---------------------------------------------------------------------------

describe("cache rotation and housekeeping", () => {
  test("housekeeping is triggered when record count >= MAX_CACHE_RECORDS", async () => {
    const records: ProviderRateRecord[] = Array.from({ length: MAX_CACHE_RECORDS }, (_, i) =>
      makeRecord({ timestamp: new Date(Date.now() - i * 1000).toISOString() }),
    );
    await writeCacheFile(tmpDir, records);

    const warnings: string[] = [];
    const orig = console.warn;
    console.warn = (...args: unknown[]) => { warnings.push(String(args[0])); };

    try {
      const oracle = new ProviderCostOracle(tmpDir);
      await oracle.initialize();
      expect(warnings.some((w) => w.includes("Housekeeping"))).toBe(true);
    } finally {
      console.warn = orig;
    }
  });

  test("housekeeping is triggered when cache is stale (>7 days)", async () => {
    const staleTimestamp = new Date(Date.now() - MAX_CACHE_AGE_MS - 1000).toISOString();
    const records = [makeRecord({ timestamp: staleTimestamp })];
    await writeCacheFile(tmpDir, records);

    const warnings: string[] = [];
    const orig = console.warn;
    console.warn = (...args: unknown[]) => { warnings.push(String(args[0])); };

    try {
      const oracle = new ProviderCostOracle(tmpDir);
      await oracle.initialize();
      expect(warnings.some((w) => w.includes("Housekeeping"))).toBe(true);
    } finally {
      console.warn = orig;
    }
  });

  test("no housekeeping warning for fresh, small cache", async () => {
    const records = Array.from({ length: 5 }, () => makeRecord());
    await writeCacheFile(tmpDir, records);

    const warnings: string[] = [];
    const orig = console.warn;
    console.warn = (...args: unknown[]) => { warnings.push(String(args[0])); };

    try {
      const oracle = new ProviderCostOracle(tmpDir);
      await oracle.initialize();
      expect(warnings.some((w) => w.includes("Housekeeping"))).toBe(false);
    } finally {
      console.warn = orig;
    }
  });

  test("housekeeping records are excluded from EMA and observation count", async () => {
    const records: ProviderRateRecord[] = [
      makeRecord({ inputRate: 6.0 }),
      { ...makeRecord(), housekeeping: true, provider: "_housekeeping", model: "_housekeeping" },
    ];
    await writeCacheFile(tmpDir, records);

    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();

    const rate = oracle.getCalibratedRate("anthropic");
    // Only one real record blended
    const expected = (1 - EMA_ALPHA) * 3.0 + EMA_ALPHA * 6.0;
    expect(rate.inputPerMToken).toBeCloseTo(expected, 6);
    // Observation count reflects only real records
    expect(rate.observationCount).toBe(0); // loaded from cache, not via recordObservation
  });

  test("recordObservation persists to cache file", async () => {
    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();

    await oracle.recordObservation("anthropic", "claude-haiku-4", 0.8, 4.0, 0.08);

    // Re-instantiate to force fresh load from disk
    const oracle2 = new ProviderCostOracle(tmpDir);
    await oracle2.initialize();

    const rate = oracle2.getCalibratedRate("anthropic");
    // Should have been blended from the written record
    const p = PROVIDER_PRICING["anthropic"]!;
    const expected = (1 - EMA_ALPHA) * p.inputPerMToken + EMA_ALPHA * 0.8;
    expect(rate.inputPerMToken).toBeCloseTo(expected, 6);
  });
});

// ---------------------------------------------------------------------------
// Rate stability bounds
// ---------------------------------------------------------------------------

describe("rate stability bounds", () => {
  test("calibrated rate never goes negative", async () => {
    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();

    // Feed 0-valued observations (e.g. an ollama-style free provider)
    for (let i = 0; i < 50; i++) {
      await oracle.recordObservation("anthropic", "local", 0.0, 0.0, 0.0);
    }

    const rate = oracle.getCalibratedRate("anthropic");
    expect(rate.inputPerMToken).toBeGreaterThanOrEqual(0);
    expect(rate.outputPerMToken).toBeGreaterThanOrEqual(0);
    expect(rate.cacheReadPerMToken).toBeGreaterThanOrEqual(0);
  });

  test("large observed rate spike does not permanently dominate EMA", async () => {
    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();

    // One spike
    await oracle.recordObservation("anthropic", "claude-sonnet-4", 999.0, null, null);

    // Followed by many on-target observations
    const target = 3.0;
    for (let i = 0; i < 100; i++) {
      await oracle.recordObservation("anthropic", "claude-sonnet-4", target, null, null);
    }

    const rate = oracle.getCalibratedRate("anthropic");
    // Should be very close to target (spike damped out)
    expect(Math.abs(rate.inputPerMToken - target)).toBeLessThan(0.5);
  });

  test("observationCount increments correctly", async () => {
    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();

    for (let i = 0; i < 5; i++) {
      await oracle.recordObservation("openai", "gpt-4o", 2.5, 10.0, 1.25);
    }

    const rate = oracle.getCalibratedRate("openai");
    expect(rate.observationCount).toBe(5);
  });

  test("getCalibratedRate is usable before initialize()", async () => {
    const oracle = new ProviderCostOracle(tmpDir);
    // Intentionally skip initialize()
    const rate = oracle.getCalibratedRate("anthropic");
    // Should return static fallback without throwing
    expect(rate.inputPerMToken).toBeGreaterThan(0);
  });

  test("all known providers have valid static fallback rates", () => {
    const oracle = new ProviderCostOracle(tmpDir);
    for (const provider of Object.keys(PROVIDER_PRICING)) {
      const rate = oracle.getCalibratedRate(provider);
      expect(rate.inputPerMToken).toBeGreaterThanOrEqual(0);
      expect(rate.outputPerMToken).toBeGreaterThanOrEqual(0);
      expect(rate.cacheReadPerMToken).toBeGreaterThanOrEqual(0);
    }
  });

  test("EMA_ALPHA constant is correct value", () => {
    expect(EMA_ALPHA).toBe(0.1);
  });

  test("MAX_CACHE_RECORDS constant is 500", () => {
    expect(MAX_CACHE_RECORDS).toBe(500);
  });

  test("MAX_CACHE_AGE_MS constant is 7 days", () => {
    expect(MAX_CACHE_AGE_MS).toBe(7 * 24 * 60 * 60 * 1000);
  });
});
