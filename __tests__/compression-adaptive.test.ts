/**
 * Tests for src/compression/adaptive.ts
 *
 * Covers:
 *   - compressionHistoryPath()
 *   - recordCompressionResult() — appends JSONL, creates dir
 *   - learnCompressionThresholds() — aggregates success rate + overshoot
 *   - selectCompressionTierAdaptive() — biases toward high-success tiers,
 *     falls back to static when history is empty/sparse
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { existsSync } from "fs";
import { mkdir, rm, writeFile } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";
import {
  compressionHistoryPath,
  learnCompressionThresholds,
  recordCompressionResult,
  selectCompressionTierAdaptive,
  type CompressionFeedback,
  type LearnedThresholds,
} from "../src/compression/adaptive.ts";
import { selectCompressionTier } from "../src/compression/context.ts";
import type { Message } from "../src/types/index.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Create a fresh temp directory for each test. */
async function makeTmpDir(): Promise<string> {
  const base = join(tmpdir(), `ashlr-adaptive-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  await mkdir(base, { recursive: true });
  return base;
}

function textMsg(role: "user" | "assistant", text: string): Message {
  return { role, content: text };
}

function largeTool(id: string, content: string): Message {
  return {
    role: "user",
    content: [{ type: "tool_result", tool_use_id: id, content }],
  };
}

/** Build a minimal history with the given feedback records written to cwd. */
async function seedHistory(cwd: string, records: Omit<CompressionFeedback, "recordedAt">[]): Promise<void> {
  for (const r of records) {
    await recordCompressionResult(cwd, r.tier, r.estimatedTokens, r.actualTokens, r.success);
  }
}

/**
 * Build a LearnedThresholds object programmatically (no file I/O)
 * for unit testing selectCompressionTierAdaptive directly.
 */
function makeThresholds(overrides: Partial<Record<1 | 2 | 3, { successRate: number; avgOvershootPct: number; sampleCount: number }>>): LearnedThresholds {
  const defaults = { successRate: 1.0, avgOvershootPct: 0, sampleCount: 5 };
  return {
    byTier: {
      1: { tier: 1, ...defaults, ...overrides[1] },
      2: { tier: 2, ...defaults, ...overrides[2] },
      3: { tier: 3, ...defaults, ...overrides[3] },
    },
  };
}

// ---------------------------------------------------------------------------
// compressionHistoryPath
// ---------------------------------------------------------------------------

describe("compressionHistoryPath", () => {
  test("returns path inside genome/evolution directory", () => {
    const p = compressionHistoryPath("/projects/myapp");
    expect(p).toContain(".ashlrcode/genome/evolution");
    expect(p).toContain("compression-history.jsonl");
  });

  test("is deterministic for the same cwd", () => {
    expect(compressionHistoryPath("/x")).toBe(compressionHistoryPath("/x"));
  });

  test("is different for different cwd values", () => {
    expect(compressionHistoryPath("/a")).not.toBe(compressionHistoryPath("/b"));
  });
});

// ---------------------------------------------------------------------------
// recordCompressionResult
// ---------------------------------------------------------------------------

describe("recordCompressionResult", () => {
  let tmp: string;
  beforeEach(async () => { tmp = await makeTmpDir(); });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("creates the evolution directory and history file if absent", async () => {
    await recordCompressionResult(tmp, 3, 1000, 1050, true);
    const p = compressionHistoryPath(tmp);
    expect(existsSync(p)).toBe(true);
  });

  test("appends valid JSONL — one line per call", async () => {
    await recordCompressionResult(tmp, 3, 1000, 1050, true);
    await recordCompressionResult(tmp, 2, 2000, 2200, false);
    const { readFile } = await import("fs/promises");
    const raw = await readFile(compressionHistoryPath(tmp), "utf-8");
    const lines = raw.trim().split("\n").filter(Boolean);
    expect(lines.length).toBe(2);
    const r0 = JSON.parse(lines[0]!);
    expect(r0.tier).toBe(3);
    expect(r0.estimatedTokens).toBe(1000);
    expect(r0.actualTokens).toBe(1050);
    expect(r0.success).toBe(true);
    expect(typeof r0.recordedAt).toBe("string");
  });

  test("records tier 1 entries correctly", async () => {
    await recordCompressionResult(tmp, 1, 500, 3000, false);
    const { readFile } = await import("fs/promises");
    const raw = await readFile(compressionHistoryPath(tmp), "utf-8");
    const r = JSON.parse(raw.trim());
    expect(r.tier).toBe(1);
    expect(r.success).toBe(false);
  });

  test("multiple appends accumulate — does not overwrite", async () => {
    for (let i = 0; i < 5; i++) {
      await recordCompressionResult(tmp, 2, 1000, 1100, true);
    }
    const { readFile } = await import("fs/promises");
    const raw = await readFile(compressionHistoryPath(tmp), "utf-8");
    const lines = raw.trim().split("\n").filter(Boolean);
    expect(lines.length).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// learnCompressionThresholds
// ---------------------------------------------------------------------------

describe("learnCompressionThresholds", () => {
  let tmp: string;
  beforeEach(async () => { tmp = await makeTmpDir(); });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("returns null when no history file exists", async () => {
    const result = await learnCompressionThresholds(tmp);
    expect(result).toBeNull();
  });

  test("returns null when history file is empty", async () => {
    const p = compressionHistoryPath(tmp);
    await mkdir(join(tmp, ".ashlrcode/genome/evolution"), { recursive: true });
    await writeFile(p, "", "utf-8");
    const result = await learnCompressionThresholds(tmp);
    expect(result).toBeNull();
  });

  test("computes success rate of 1.0 when all tier-3 outcomes succeed", async () => {
    await seedHistory(tmp, [
      { tier: 3, estimatedTokens: 1000, actualTokens: 1050, success: true },
      { tier: 3, estimatedTokens: 1200, actualTokens: 1230, success: true },
      { tier: 3, estimatedTokens: 900, actualTokens: 940, success: true },
    ]);
    const learned = await learnCompressionThresholds(tmp);
    expect(learned).not.toBeNull();
    expect(learned!.byTier[3].successRate).toBeCloseTo(1.0);
    expect(learned!.byTier[3].sampleCount).toBe(3);
  });

  test("computes success rate of 0.0 when all tier-2 outcomes fail", async () => {
    await seedHistory(tmp, [
      { tier: 2, estimatedTokens: 2000, actualTokens: 8000, success: false },
      { tier: 2, estimatedTokens: 2000, actualTokens: 9000, success: false },
      { tier: 2, estimatedTokens: 2000, actualTokens: 8500, success: false },
    ]);
    const learned = await learnCompressionThresholds(tmp);
    expect(learned!.byTier[2].successRate).toBeCloseTo(0.0);
  });

  test("computes positive avgOvershootPct when actual > estimated", async () => {
    // actual is always 20% more than estimated
    await seedHistory(tmp, [
      { tier: 3, estimatedTokens: 1000, actualTokens: 1200, success: true },
      { tier: 3, estimatedTokens: 2000, actualTokens: 2400, success: true },
      { tier: 3, estimatedTokens: 500, actualTokens: 600, success: true },
    ]);
    const learned = await learnCompressionThresholds(tmp);
    expect(learned!.byTier[3].avgOvershootPct).toBeCloseTo(20, 0);
  });

  test("computes negative avgOvershootPct when actual < estimated", async () => {
    // actual is always 10% less than estimated
    await seedHistory(tmp, [
      { tier: 3, estimatedTokens: 1000, actualTokens: 900, success: true },
      { tier: 3, estimatedTokens: 2000, actualTokens: 1800, success: true },
      { tier: 3, estimatedTokens: 500, actualTokens: 450, success: true },
    ]);
    const learned = await learnCompressionThresholds(tmp);
    expect(learned!.byTier[3].avgOvershootPct).toBeCloseTo(-10, 0);
  });

  test("empty-bucket tiers get default effectiveness (rate=1, overshoot=0)", async () => {
    // Only seed tier 3; tiers 1 and 2 should get defaults
    await seedHistory(tmp, [
      { tier: 3, estimatedTokens: 1000, actualTokens: 1000, success: true },
      { tier: 3, estimatedTokens: 1000, actualTokens: 1000, success: true },
      { tier: 3, estimatedTokens: 1000, actualTokens: 1000, success: true },
    ]);
    const learned = await learnCompressionThresholds(tmp);
    expect(learned!.byTier[1].sampleCount).toBe(0);
    expect(learned!.byTier[1].successRate).toBe(1.0);
    expect(learned!.byTier[2].sampleCount).toBe(0);
  });

  test("mixed tiers are bucketed independently", async () => {
    await seedHistory(tmp, [
      { tier: 1, estimatedTokens: 500, actualTokens: 600, success: false },
      { tier: 1, estimatedTokens: 500, actualTokens: 700, success: false },
      { tier: 2, estimatedTokens: 2000, actualTokens: 2100, success: true },
      { tier: 2, estimatedTokens: 2000, actualTokens: 1900, success: true },
      { tier: 3, estimatedTokens: 3000, actualTokens: 3000, success: true },
    ]);
    const learned = await learnCompressionThresholds(tmp);
    expect(learned!.byTier[1].sampleCount).toBe(2);
    expect(learned!.byTier[1].successRate).toBeCloseTo(0.0);
    expect(learned!.byTier[2].sampleCount).toBe(2);
    expect(learned!.byTier[2].successRate).toBeCloseTo(1.0);
    expect(learned!.byTier[3].sampleCount).toBe(1);
  });

  test("skips corrupt JSONL lines gracefully", async () => {
    const p = compressionHistoryPath(tmp);
    await mkdir(join(tmp, ".ashlrcode/genome/evolution"), { recursive: true });
    // Mix of valid and corrupt lines
    const valid = JSON.stringify({ tier: 3, estimatedTokens: 1000, actualTokens: 1000, success: true, recordedAt: new Date().toISOString() });
    await writeFile(p, `${valid}\n{corrupt json here\n${valid}\n${valid}\n`, "utf-8");
    const learned = await learnCompressionThresholds(tmp);
    expect(learned).not.toBeNull();
    expect(learned!.byTier[3].sampleCount).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// selectCompressionTierAdaptive — null/empty history fallback
// ---------------------------------------------------------------------------

describe("selectCompressionTierAdaptive — fallback to static", () => {
  const smallMsgs: Message[] = Array.from({ length: 5 }, (_, i) =>
    textMsg("user", `msg ${i}`),
  );

  test("returns same tier as static when history is null", () => {
    const staticTier = selectCompressionTier(smallMsgs, 0);
    const adaptiveTier = selectCompressionTierAdaptive(smallMsgs, 0, {}, null);
    expect(adaptiveTier).toBe(staticTier);
  });

  test("returns same tier as static when history is undefined", () => {
    const staticTier = selectCompressionTier(smallMsgs, 0);
    const adaptiveTier = selectCompressionTierAdaptive(smallMsgs, 0, {});
    expect(adaptiveTier).toBe(staticTier);
  });

  test("returns tier 1 with null history when budget is impossibly tight", () => {
    const msgs = Array.from({ length: 20 }, (_, i) =>
      largeTool(`t${i}`, "X".repeat(5000)),
    );
    const tier = selectCompressionTierAdaptive(msgs, 0, { maxContextTokens: 100, reserveTokens: 0 }, null);
    expect(tier).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// selectCompressionTierAdaptive — tier 3 preferred when it historically succeeds
// ---------------------------------------------------------------------------

describe("selectCompressionTierAdaptive — prefers tier 3 when historically successful", () => {
  test("chooses tier 3 when it has high success rate and fits budget", () => {
    // Small messages easily fit within tier 3 after collapse
    const msgs = Array.from({ length: 8 }, (_, i) =>
      textMsg(i % 2 === 0 ? "user" : "assistant", `turn ${i} text`),
    );
    const history = makeThresholds({
      3: { successRate: 0.95, avgOvershootPct: 5, sampleCount: 10 },
      2: { successRate: 0.80, avgOvershootPct: 10, sampleCount: 10 },
      1: { successRate: 0.99, avgOvershootPct: 0, sampleCount: 10 },
    });
    const tier = selectCompressionTierAdaptive(msgs, 0, { maxContextTokens: 1_000_000, reserveTokens: 8192 }, history);
    expect(tier).toBe(3);
  });

  test("tier 3 chosen more often (in a sweep) when it consistently succeeds", () => {
    // Simulate 10 calls with different message sizes — tier 3 should win for small ones
    const history = makeThresholds({
      3: { successRate: 0.95, avgOvershootPct: 2, sampleCount: 20 },
      2: { successRate: 0.70, avgOvershootPct: 15, sampleCount: 20 },
      1: { successRate: 0.99, avgOvershootPct: 0, sampleCount: 20 },
    });
    let tier3Count = 0;
    for (let size = 2; size <= 20; size += 2) {
      const msgs = Array.from({ length: size }, (_, i) =>
        textMsg(i % 2 === 0 ? "user" : "assistant", `turn ${i}`),
      );
      const t = selectCompressionTierAdaptive(msgs, 0, { maxContextTokens: 1_000_000, reserveTokens: 8192 }, history);
      if (t === 3) tier3Count++;
    }
    // With a huge budget, tier 3 should be chosen for most/all sizes
    expect(tier3Count).toBeGreaterThanOrEqual(7);
  });
});

// ---------------------------------------------------------------------------
// selectCompressionTierAdaptive — prefers tier 1 when tier 2/3 historically fail
// ---------------------------------------------------------------------------

describe("selectCompressionTierAdaptive — escalates when lower tiers fail", () => {
  test("escalates from tier 3 to tier 2 when tier 3 has poor success rate", () => {
    // Messages that the static selector would choose tier 3 for (small)
    const msgs = Array.from({ length: 8 }, (_, i) =>
      textMsg(i % 2 === 0 ? "user" : "assistant", `turn ${i}`),
    );
    // Tier 3 has failed consistently; tier 2 is better
    const history = makeThresholds({
      3: { successRate: 0.2, avgOvershootPct: 80, sampleCount: 10 },
      2: { successRate: 0.9, avgOvershootPct: 5, sampleCount: 10 },
      1: { successRate: 0.99, avgOvershootPct: 0, sampleCount: 10 },
    });
    const tier = selectCompressionTierAdaptive(msgs, 0, { maxContextTokens: 1_000_000, reserveTokens: 8192 }, history);
    // Should escalate: tier 3 poor → try tier 2
    expect(tier).toBeLessThanOrEqual(2);
  });

  test("escalates from tier 2 to tier 1 when tier 2 has poor success rate", () => {
    // Messages that put static selector at tier 2 (large tool results, snip fits but collapse doesn't)
    const msgs: Message[] = [];
    for (let i = 0; i < 10; i++) {
      msgs.push(largeTool(`t${i}`, "X".repeat(5000)));
    }

    // Tier 3 and tier 2 both have poor history — only tier 1 passes the successRate threshold.
    // With enough samples on tiers 3 and 2, the learned logic should walk past them and pick tier 1.
    const history = makeThresholds({
      3: { successRate: 0.3, avgOvershootPct: 60, sampleCount: 10 },
      2: { successRate: 0.25, avgOvershootPct: 70, sampleCount: 10 },
      1: { successRate: 0.95, avgOvershootPct: 0, sampleCount: 10 },
    });
    // Use a very large budget so static would pick tier 3; the history should override.
    const tier = selectCompressionTierAdaptive(msgs, 0, {
      maxContextTokens: 1_000_000,
      reserveTokens: 0,
    }, history);
    // Both tier 3 and tier 2 have poor history → should prefer tier 1
    expect(tier).toBe(1);
  });

  test("sparse history (< MIN_SAMPLES) does not override static selection", () => {
    const msgs = Array.from({ length: 8 }, (_, i) =>
      textMsg(i % 2 === 0 ? "user" : "assistant", `turn ${i}`),
    );
    // Only 1 sample per tier — below MIN_SAMPLES_FOR_LEARNING threshold
    const history = makeThresholds({
      3: { successRate: 0.0, avgOvershootPct: 100, sampleCount: 1 },
      2: { successRate: 0.0, avgOvershootPct: 100, sampleCount: 1 },
      1: { successRate: 0.0, avgOvershootPct: 100, sampleCount: 1 },
    });
    const staticTier = selectCompressionTier(msgs, 0, { maxContextTokens: 1_000_000, reserveTokens: 8192 });
    const adaptiveTier = selectCompressionTierAdaptive(msgs, 0, { maxContextTokens: 1_000_000, reserveTokens: 8192 }, history);
    // With sparse data, adaptive should match static
    expect(adaptiveTier).toBe(staticTier);
  });
});

// ---------------------------------------------------------------------------
// selectCompressionTierAdaptive — calibration via overshoot
// ---------------------------------------------------------------------------

describe("selectCompressionTierAdaptive — calibration via overshoot", () => {
  test("large positive overshoot on tier 3 causes escalation when budget is tight", async () => {
    // Msgs whose tier-3 collapsed estimate is ~small (say ~10 tokens).
    // With 200% overshoot the calibrated prediction is 30 tokens.
    // We set a budget of 20 tokens so: raw estimate fits, calibrated does not.
    // Tier 2 has only 5% overshoot so its calibrated estimate easily fits.
    const msgs = Array.from({ length: 8 }, (_, i) =>
      textMsg(i % 2 === 0 ? "user" : "assistant", `turn ${i}`),
    );
    const history = makeThresholds({
      3: { successRate: 0.9, avgOvershootPct: 200, sampleCount: 10 },
      2: { successRate: 0.9, avgOvershootPct: 5, sampleCount: 10 },
      1: { successRate: 0.99, avgOvershootPct: 0, sampleCount: 10 },
    });
    // Import inline — these are pure functions, no side effects
    const { estimateTokensFromMessages: est } = await import("../src/tokens/index.ts");
    const { contextCollapse: cc } = await import("../src/compression/context.ts");
    const t3Tokens = est(cc(msgs));
    // Budget: above raw tier-3 estimate but below calibrated (3×) tier-3 estimate
    const tightBudget = Math.round(t3Tokens * 1.5);

    const tier = selectCompressionTierAdaptive(msgs, 0, {
      maxContextTokens: tightBudget + 8192,
      reserveTokens: 8192,
    }, history);
    // With 200% overshoot calibration, tier 3 no longer fits → should pick tier 2 or 1
    expect(tier).toBeLessThanOrEqual(2);
  });

  test("slight overshoot does not change tier when budget is ample", () => {
    const msgs = Array.from({ length: 5 }, (_, i) => textMsg("user", `msg ${i}`));
    const history = makeThresholds({
      3: { successRate: 0.95, avgOvershootPct: 10, sampleCount: 10 },
      2: { successRate: 0.95, avgOvershootPct: 10, sampleCount: 10 },
      1: { successRate: 0.99, avgOvershootPct: 0, sampleCount: 10 },
    });
    const tier = selectCompressionTierAdaptive(msgs, 0, { maxContextTokens: 1_000_000, reserveTokens: 8192 }, history);
    expect(tier).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// End-to-end: record → learn → select
// ---------------------------------------------------------------------------

describe("end-to-end: record → learn → select", () => {
  let tmp: string;
  beforeEach(async () => { tmp = await makeTmpDir(); });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("tier 3 is chosen after recording many tier-3 successes", async () => {
    // Seed 5 tier-3 successes with accurate estimates
    for (let i = 0; i < 5; i++) {
      await recordCompressionResult(tmp, 3, 1000, 1010, true);
    }
    const learned = await learnCompressionThresholds(tmp);
    expect(learned).not.toBeNull();

    const msgs = Array.from({ length: 8 }, (_, i) =>
      textMsg(i % 2 === 0 ? "user" : "assistant", `turn ${i}`),
    );
    const tier = selectCompressionTierAdaptive(msgs, 0, { maxContextTokens: 1_000_000, reserveTokens: 8192 }, learned);
    expect(tier).toBe(3);
  });

  test("returns null learned thresholds when cwd has no genome", async () => {
    const freshTmp = await makeTmpDir();
    try {
      const result = await learnCompressionThresholds(freshTmp);
      expect(result).toBeNull();
    } finally {
      await rm(freshTmp, { recursive: true, force: true });
    }
  });

  test("roundtrip: recorded feedback influences learned successRate correctly", async () => {
    // 4 successes, 1 failure for tier 2
    await recordCompressionResult(tmp, 2, 2000, 2100, true);
    await recordCompressionResult(tmp, 2, 2000, 2050, true);
    await recordCompressionResult(tmp, 2, 2000, 2200, true);
    await recordCompressionResult(tmp, 2, 2000, 2300, true);
    await recordCompressionResult(tmp, 2, 2000, 9000, false);

    const learned = await learnCompressionThresholds(tmp);
    expect(learned!.byTier[2].successRate).toBeCloseTo(0.8, 1);
    expect(learned!.byTier[2].sampleCount).toBe(5);
  });
});
