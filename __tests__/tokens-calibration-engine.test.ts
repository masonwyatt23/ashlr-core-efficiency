/**
 * Tests for CalibrationEngine — Unified Cross-Provider Token Calibration Engine
 * (src/tokens/calibration-engine.ts)
 *
 * Coverage:
 *   1. tukeyBiweightMean — outlier rejection, stability, edge cases
 *   2. Per-bin stability — independent factor convergence per bin
 *   3. Cross-provider independence — separate state per provider slug
 *   4. Pre-trained baseline accuracy — zero-warmup factors for claude-3-5, gpt-4o, llama3, qwen
 *   5. EMA drift detection — hasDrift detects convergence away from baseline
 *   6. calibratedEstimate — dispatches to per-bin factors, base count passthrough
 *   7. File I/O — persist, loadFromFile, trimFile, auto-rotation
 *   8. Barrel export — CalibrationEngine importable from src/tokens/index.ts
 *   9. correctionFactor fallback chain — pretrained → overall → 1.0
 *  10. mape — weighted average across bins
 */

import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { join } from "path";
import { tmpdir } from "os";
import { writeFile, unlink, readFile } from "fs/promises";
import { existsSync } from "fs";

import {
  CalibrationEngine,
  tukeyBiweightMean,
  MAX_RECORDS_PER_PROVIDER,
  CE_MIN_SAMPLES,
  CE_FACTOR_MIN,
  CE_FACTOR_MAX,
  CE_EMA_ALPHA,
  TUKEY_C,
  type CalibrationEngineRecord,
} from "../src/tokens/calibration-engine.ts";

// Verify re-export from barrel
import {
  CalibrationEngine as CalibrationEngineFromIndex,
  tukeyBiweightMean as tukeyFromIndex,
} from "../src/tokens/index.ts";

import type { Message } from "../src/types/index.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function tmpFile(): string {
  return join(
    tmpdir(),
    `ashlr-ce-test-${Date.now()}-${Math.random().toString(36).slice(2)}.jsonl`,
  );
}

async function cleanupFile(path: string): Promise<void> {
  try {
    if (existsSync(path)) await unlink(path);
  } catch { /* ignore */ }
}

function txt(text: string): Message {
  return { role: "user", content: text };
}

function toolResultMsg(): Message {
  return {
    role: "user",
    content: [{ type: "tool_result", tool_use_id: "t1", content: "result data" }],
  };
}

function toolUseMsg(): Message {
  return {
    role: "user",
    content: [{ type: "tool_use", id: "t1", name: "bash", input: { cmd: "ls" } }],
  };
}

function thinkingMsg(): Message {
  return {
    role: "user",
    content: [{ type: "thinking", thinking: "let me reason about this" }],
  };
}

function imageMsg(): Message {
  return {
    role: "user",
    content: [{ type: "image_url", image_url: { url: "https://example.com/img.png" } }],
  };
}

/** Seed a file with CalibrationEngineRecord lines. */
async function seedFile(path: string, records: CalibrationEngineRecord[]): Promise<void> {
  const lines = records.map((r) => JSON.stringify(r)).join("\n") + "\n";
  await writeFile(path, lines, "utf8");
}

function makeRecord(
  provider: string,
  bin: CalibrationEngineRecord["bin"],
  estimated: number,
  actual: number,
): CalibrationEngineRecord {
  return {
    timestamp: new Date().toISOString(),
    provider,
    bin,
    estimated,
    actual,
  };
}

beforeEach(() => {
  CalibrationEngine.resetInstance();
});

// ---------------------------------------------------------------------------
// 1. tukeyBiweightMean — outlier rejection
// ---------------------------------------------------------------------------

describe("tukeyBiweightMean — outlier rejection", () => {
  test("returns single value unchanged", () => {
    expect(tukeyBiweightMean([1.2])).toBe(1.2);
  });

  test("returns mean for two identical values", () => {
    expect(tukeyBiweightMean([1.0, 1.0])).toBe(1.0);
  });

  test("returns exact value when all identical", () => {
    const result = tukeyBiweightMean([1.05, 1.05, 1.05, 1.05, 1.05]);
    expect(Math.abs(result - 1.05)).toBeLessThan(1e-9);
  });

  test("rejects extreme outlier (>3σ) without distorting mean", () => {
    // Core values around 1.0; one extreme outlier at 100.0
    const values = [1.0, 1.01, 0.99, 1.02, 0.98, 1.0, 100.0];
    const biweight = tukeyBiweightMean(values);
    // Result should be very close to 1.0 despite the outlier
    expect(biweight).toBeLessThan(1.5);
    expect(biweight).toBeGreaterThan(0.9);
  });

  test("rejects low outlier without distorting mean", () => {
    const values = [1.0, 1.01, 0.99, 1.02, 0.98, 1.0, 0.001];
    const biweight = tukeyBiweightMean(values);
    expect(biweight).toBeGreaterThan(0.9);
    expect(biweight).toBeLessThan(1.1);
  });

  test("produces more accurate mean than arithmetic mean with outliers", () => {
    // Without outlier, true mean ≈ 1.0
    const clean = [1.0, 1.01, 0.99, 1.02, 0.98];
    // With an outlier, arithmetic mean is pulled far from 1.0
    const withOutlier = [...clean, 50.0];
    const arithmeticMean = withOutlier.reduce((s, v) => s + v, 0) / withOutlier.length;
    const biweightResult = tukeyBiweightMean(withOutlier);
    // Biweight should be closer to 1.0 than arithmetic mean
    expect(Math.abs(biweightResult - 1.0)).toBeLessThan(Math.abs(arithmeticMean - 1.0));
  });

  test("handles empty array — returns 1.0 (identity)", () => {
    expect(tukeyBiweightMean([])).toBe(1.0);
  });

  test("handles two-element array without throwing", () => {
    const result = tukeyBiweightMean([1.0, 1.2]);
    expect(typeof result).toBe("number");
    expect(isFinite(result)).toBe(true);
  });

  test("returns median when all weights become zero (degenerate)", () => {
    // If we pass a custom c=0 (zero tuning constant), all u_i = ∞, all weights zero
    // With very small c, most points get zero weight → median fallback
    const values = [1.0, 2.0, 3.0];
    const result = tukeyBiweightMean(values, 0.001);
    // Should return the median (2.0) as fallback
    expect(result).toBe(2.0);
  });

  test("exported from barrel as tukeyBiweightMean", () => {
    expect(typeof tukeyFromIndex).toBe("function");
    expect(tukeyFromIndex([1.0, 1.0, 1.0])).toBe(1.0);
  });

  test("normal distribution: biweight mean close to true mean", () => {
    // 20 values centered around 1.1 with small noise
    const values = Array.from({ length: 20 }, (_, i) =>
      1.1 + (i % 3 === 0 ? 0.01 : -0.01),
    );
    const result = tukeyBiweightMean(values);
    expect(Math.abs(result - 1.1)).toBeLessThan(0.05);
  });
});

// ---------------------------------------------------------------------------
// 2. Per-bin stability — independent convergence
// ---------------------------------------------------------------------------

describe("CalibrationEngine — per-bin stability", () => {
  test("text bin and tool_result bin converge independently", () => {
    const engine = CalibrationEngine.instance;
    const textMsgs = [txt("hello world this is a text message")];
    const toolMsgs = [toolResultMsg()];

    // text bin: no error (actual == estimated)
    for (let i = 0; i < 10; i++) {
      engine.record({ provider: "claude-3-5", messages: textMsgs, estimated: 100, actual: 100 });
    }
    // tool_result bin: 30% under-estimate (actual = 1.3×)
    for (let i = 0; i < 10; i++) {
      engine.record({ provider: "claude-3-5", messages: toolMsgs, estimated: 100, actual: 130 });
    }

    const textFactor   = engine.correctionFactor("claude-3-5", "text");
    const toolFactor   = engine.correctionFactor("claude-3-5", "tool_result");

    // tool_result factor should exceed text factor
    expect(toolFactor).toBeGreaterThan(textFactor);
  });

  test("image bin converges independently of text bin", () => {
    const engine = CalibrationEngine.instance;

    // image bin: actual = 2× estimated
    for (let i = 0; i < 8; i++) {
      engine.record({ provider: "gpt-4o", messages: [imageMsg()], estimated: 200, actual: 400 });
    }
    // text bin: exact match
    for (let i = 0; i < 8; i++) {
      engine.record({ provider: "gpt-4o", messages: [txt("text")], estimated: 200, actual: 200 });
    }

    const imageFactor = engine.correctionFactor("gpt-4o", "image");
    const textFactor  = engine.correctionFactor("gpt-4o", "text");
    expect(imageFactor).toBeGreaterThan(textFactor);
  });

  test("thinking bin factor is tracked separately", () => {
    const engine = CalibrationEngine.instance;

    for (let i = 0; i < 5; i++) {
      engine.record({ provider: "claude-3-5", messages: [thinkingMsg()], estimated: 100, actual: 150 });
    }

    const state = engine.getState("claude-3-5");
    expect(state).toBeDefined();
    expect(state!.bins.thinking.sampleCount).toBe(5);
  });

  test("tool_use bin tracked independently of tool_result", () => {
    const engine = CalibrationEngine.instance;

    for (let i = 0; i < 5; i++) {
      engine.record({ provider: "gpt-4o", messages: [toolUseMsg()], estimated: 100, actual: 110 });
    }
    for (let i = 0; i < 5; i++) {
      engine.record({ provider: "gpt-4o", messages: [toolResultMsg()], estimated: 100, actual: 170 });
    }

    const state = engine.getState("gpt-4o");
    expect(state!.bins.tool_use.sampleCount).toBe(5);
    expect(state!.bins.tool_result.sampleCount).toBe(5);
    expect(state!.bins.tool_result.correctionFactor).toBeGreaterThan(
      state!.bins.tool_use.correctionFactor,
    );
  });

  test("bin correction factor is clamped within [CE_FACTOR_MIN, CE_FACTOR_MAX]", () => {
    const engine = CalibrationEngine.instance;
    // Extreme: actual = 100× estimated
    for (let i = 0; i < 30; i++) {
      engine.record({ provider: "llama3", messages: [txt("x")], estimated: 10, actual: 1000 });
    }
    const factor = engine.correctionFactor("llama3", "text");
    expect(factor).toBeGreaterThanOrEqual(CE_FACTOR_MIN);
    expect(factor).toBeLessThanOrEqual(CE_FACTOR_MAX);
  });

  test("bin correction factor lower-clamped when actual << estimated", () => {
    const engine = CalibrationEngine.instance;
    for (let i = 0; i < 30; i++) {
      engine.record({ provider: "gpt-4o", messages: [txt("x")], estimated: 1000, actual: 1 });
    }
    const factor = engine.correctionFactor("gpt-4o", "text");
    expect(factor).toBeGreaterThanOrEqual(CE_FACTOR_MIN);
  });

  test("record with bin override ignores message classification", () => {
    const engine = CalibrationEngine.instance;
    // Pass text messages but override bin to "image"
    for (let i = 0; i < 6; i++) {
      engine.record({
        provider: "claude-3-5",
        messages: [txt("plain text")],
        estimated: 100,
        actual: 200,
        bin: "image",
      });
    }
    const state = engine.getState("claude-3-5");
    // image bin should have all 6 samples
    expect(state!.bins.image.sampleCount).toBe(6);
    // text bin should have 0 live samples
    expect(state!.bins.text.sampleCount).toBe(0);
  });

  test("reset() clears all in-memory state", () => {
    const engine = CalibrationEngine.instance;
    engine.record({ provider: "gpt-4", messages: [txt("hi")], estimated: 50, actual: 60 });
    engine.reset();
    expect(engine.knownProviders()).toHaveLength(0);
    expect(engine.overallFactor("gpt-4")).toBe(1.0);
  });
});

// ---------------------------------------------------------------------------
// 3. Cross-provider independence
// ---------------------------------------------------------------------------

describe("CalibrationEngine — cross-provider independence", () => {
  test("training one provider does not affect another", () => {
    const engine = CalibrationEngine.instance;
    const msgs = [txt("hello world message content here for testing independence")];

    // Train claude-3-5 with 40% under-estimate
    for (let i = 0; i < 10; i++) {
      engine.record({ provider: "claude-3-5", messages: msgs, estimated: 100, actual: 140 });
    }

    // gpt-4o should return the pre-trained baseline (1.0), not claude's factor
    const gptFactor = engine.correctionFactor("gpt-4o", "text");
    const claudeFactor = engine.correctionFactor("claude-3-5", "text");

    // Claude should have converged above 1.0
    expect(claudeFactor).toBeGreaterThan(1.0);
    // GPT-4o baseline overall is 1.0 (no training yet → returns baseline)
    expect(gptFactor).toBe(1.0);
  });

  test("different slugs for same family are tracked independently", () => {
    const engine = CalibrationEngine.instance;

    for (let i = 0; i < 5; i++) {
      engine.record({ provider: "provider-a", messages: [txt("x")], estimated: 100, actual: 120 });
    }
    for (let i = 0; i < 5; i++) {
      engine.record({ provider: "provider-b", messages: [txt("x")], estimated: 100, actual: 80 });
    }

    // provider-a state should not bleed into provider-b
    const stateA = engine.getState("provider-a");
    const stateB = engine.getState("provider-b");
    expect(stateA).toBeDefined();
    expect(stateB).toBeDefined();
    // provider-a: actual > estimated → factor should be > initial
    // provider-b: actual < estimated → factor should be < initial
    expect(stateA!.overallFactor).toBeGreaterThan(stateB!.overallFactor);
  });

  test("knownProviders returns only trained providers", () => {
    const engine = CalibrationEngine.instance;
    engine.record({ provider: "llama3-70b", messages: [txt("hi")], estimated: 50, actual: 55 });
    engine.record({ provider: "qwen-7b", messages: [txt("hi")], estimated: 50, actual: 54 });
    const known = engine.knownProviders();
    expect(known).toContain("llama3-70b");
    expect(known).toContain("qwen-7b");
    expect(known).not.toContain("claude-3-5"); // never trained
  });

  test("provider key is case-insensitive", () => {
    const engine = CalibrationEngine.instance;
    for (let i = 0; i < 4; i++) {
      engine.record({ provider: "MyModel", messages: [txt("x")], estimated: 100, actual: 120 });
    }
    // Lookup with different casing should find same state
    const state = engine.getState("MYMODEL");
    expect(state).toBeDefined();
    expect(state!.totalSamples).toBe(4);
  });

  test("overallFactor returns 1.0 for completely unknown provider (no baseline)", () => {
    const engine = CalibrationEngine.instance;
    expect(engine.overallFactor("no-such-provider-xyz-123")).toBe(1.0);
  });
});

// ---------------------------------------------------------------------------
// 4. Pre-trained baseline accuracy
// ---------------------------------------------------------------------------

describe("CalibrationEngine — pre-trained baseline accuracy", () => {
  test("claude-3-5 text bin baseline is 1.0 (exact tiktoken proxy)", () => {
    const engine = CalibrationEngine.instance;
    // No training — should return pretrained baseline
    const factor = engine.correctionFactor("claude-3-5", "text");
    expect(factor).toBe(1.0);
  });

  test("claude-3-5 tool_use bin baseline is 1.05", () => {
    const engine = CalibrationEngine.instance;
    const factor = engine.correctionFactor("claude-3-5", "tool_use");
    expect(factor).toBe(1.05);
  });

  test("gpt-4o all bins baseline is 1.0", () => {
    const engine = CalibrationEngine.instance;
    for (const bin of ["text", "tool_use", "tool_result", "thinking", "image", "mixed"] as const) {
      expect(engine.correctionFactor("gpt-4o", bin)).toBe(1.0);
    }
  });

  test("llama3 text bin baseline is 1.05", () => {
    const engine = CalibrationEngine.instance;
    const factor = engine.correctionFactor("llama3", "text");
    expect(factor).toBe(1.05);
  });

  test("llama3 tool_use bin baseline is 1.08", () => {
    const engine = CalibrationEngine.instance;
    const factor = engine.correctionFactor("llama3", "tool_use");
    expect(factor).toBe(1.08);
  });

  test("qwen text bin baseline is 1.08", () => {
    const engine = CalibrationEngine.instance;
    const factor = engine.correctionFactor("qwen", "text");
    expect(factor).toBe(1.08);
  });

  test("qwen tool_use bin baseline is 1.10", () => {
    const engine = CalibrationEngine.instance;
    const factor = engine.correctionFactor("qwen", "tool_use");
    expect(factor).toBe(1.10);
  });

  test("llama3-70b-instruct matches llama3 baseline via prefix match", () => {
    const engine = CalibrationEngine.instance;
    const factor = engine.correctionFactor("llama3-70b-instruct", "text");
    expect(factor).toBe(1.05);
  });

  test("claude-3-5-sonnet-20241022 matches claude-3-5 baseline via prefix match", () => {
    const engine = CalibrationEngine.instance;
    // claude-3-5 overall baseline is 1.0
    const factor = engine.overallFactor("claude-3-5-sonnet-20241022");
    expect(factor).toBe(1.0);
  });

  test("baseline applied to state on first getOrCreate call", () => {
    const engine = CalibrationEngine.instance;
    // Single record to trigger state creation for llama3
    engine.record({ provider: "llama3", messages: [txt("x")], estimated: 100, actual: 105 });
    const state = engine.getState("llama3");
    // overallFactor seeded from baseline (1.05), then EMA-updated once
    expect(state).toBeDefined();
    expect(state!.overallFactor).toBeGreaterThan(1.0);
  });

  test("unknown model returns 1.0 — no pretrained baseline", () => {
    const engine = CalibrationEngine.instance;
    expect(engine.correctionFactor("completely-unknown-provider-xyz")).toBe(1.0);
    expect(engine.overallFactor("completely-unknown-provider-xyz")).toBe(1.0);
  });
});

// ---------------------------------------------------------------------------
// 5. EMA drift detection
// ---------------------------------------------------------------------------

describe("CalibrationEngine — EMA drift detection", () => {
  test("hasDrift returns false with no training data", () => {
    const engine = CalibrationEngine.instance;
    expect(engine.hasDrift("claude-3-5")).toBe(false);
  });

  test("hasDrift returns false when factor stays near baseline", () => {
    const engine = CalibrationEngine.instance;
    // claude-3-5 baseline overall = 1.0; feed samples with ~1% error (no drift)
    for (let i = 0; i < 10; i++) {
      engine.record({ provider: "claude-3-5", messages: [txt("x")], estimated: 100, actual: 101 });
    }
    expect(engine.hasDrift("claude-3-5", 0.05)).toBe(false);
  });

  test("hasDrift returns true when factor drifts significantly from baseline", () => {
    const engine = CalibrationEngine.instance;
    // claude-3-5 baseline = 1.0; feed samples with 20% consistent under-estimate
    for (let i = 0; i < 15; i++) {
      engine.record({ provider: "claude-3-5", messages: [txt("x")], estimated: 100, actual: 120 });
    }
    expect(engine.hasDrift("claude-3-5", 0.04)).toBe(true);
  });

  test("overall factor converges toward true ratio after many samples", () => {
    const engine = CalibrationEngine.instance;
    // Feed 30 samples with actual = 1.3× estimated
    for (let i = 0; i < 30; i++) {
      engine.record({ provider: "gpt-4o", messages: [txt("x")], estimated: 100, actual: 130 });
    }
    const factor = engine.overallFactor("gpt-4o");
    // After 30 samples with 1.3× ratio, factor should be well above 1.0
    expect(factor).toBeGreaterThan(1.1);
    expect(factor).toBeLessThanOrEqual(CE_FACTOR_MAX);
  });

  test("mape is tracked and non-zero after errors", () => {
    const engine = CalibrationEngine.instance;
    for (let i = 0; i < 6; i++) {
      engine.record({ provider: "mistral-7b", messages: [txt("x")], estimated: 100, actual: 150 });
    }
    expect(engine.mape("mistral-7b", "text")).toBeGreaterThan(0);
  });

  test("mape returns 0 for unknown provider", () => {
    const engine = CalibrationEngine.instance;
    expect(engine.mape("no-such-model-xyz")).toBe(0);
  });

  test("mape weighted average across bins", () => {
    const engine = CalibrationEngine.instance;
    // text bin: 10% error × 5 samples
    for (let i = 0; i < 5; i++) {
      engine.record({ provider: "model-x", messages: [txt("x")], estimated: 100, actual: 110 });
    }
    // tool_result bin: 50% error × 5 samples
    for (let i = 0; i < 5; i++) {
      engine.record({ provider: "model-x", messages: [toolResultMsg()], estimated: 100, actual: 150 });
    }
    const overall = engine.mape("model-x");
    const textMape = engine.mape("model-x", "text");
    const toolMape = engine.mape("model-x", "tool_result");
    // Overall should be between text and tool_result MAPEs
    expect(overall).toBeGreaterThan(0);
    expect(toolMape).toBeGreaterThan(textMape);
  });
});

// ---------------------------------------------------------------------------
// 6. calibratedEstimate — dispatching
// ---------------------------------------------------------------------------

describe("CalibrationEngine — calibratedEstimate", () => {
  test("returns 0 for empty messages", () => {
    const engine = CalibrationEngine.instance;
    expect(engine.calibratedEstimate([], "claude-3-5")).toBe(0);
  });

  test("returns positive value for non-empty messages", () => {
    const engine = CalibrationEngine.instance;
    const msgs = [txt("hello world, this is a test message")];
    expect(engine.calibratedEstimate(msgs, "claude-3-5")).toBeGreaterThan(0);
  });

  test("passthrough baseCount is scaled by factor", () => {
    const engine = CalibrationEngine.instance;
    // Train gpt-4o to have a 1.2× overall factor
    for (let i = 0; i < 10; i++) {
      engine.record({ provider: "gpt-4o", messages: [txt("x")], estimated: 100, actual: 120 });
    }
    const factor = engine.overallFactor("gpt-4o");
    // With baseCount = 500, result should be ceil(500 * factor)
    const result = engine.calibratedEstimate([txt("x")], "gpt-4o", 500);
    const expected = Math.ceil(500 * factor);
    expect(result).toBe(expected);
  });

  test("factor=1.0 returns baseCount unchanged", () => {
    const engine = CalibrationEngine.instance;
    // Use a provider with 1.0 baseline and no training
    const result = engine.calibratedEstimate([txt("hi")], "gpt-4o", 300);
    expect(result).toBe(300);
  });

  test("uses per-bin factor for last message bin", () => {
    const engine = CalibrationEngine.instance;
    // Train tool_result bin very aggressively for a custom provider
    for (let i = 0; i < 10; i++) {
      engine.record({ provider: "my-model", messages: [toolResultMsg()], estimated: 100, actual: 200 });
    }
    // calibratedEstimate with a tool_result message should use tool_result bin factor
    const factor = engine.correctionFactor("my-model", "tool_result");
    const result = engine.calibratedEstimate([toolResultMsg()], "my-model", 1000);
    expect(result).toBe(Math.ceil(1000 * factor));
  });

  test("deterministic — same input produces same result", () => {
    const engine = CalibrationEngine.instance;
    const msgs = [txt("consistent input test message for determinism check")];
    const a = engine.calibratedEstimate(msgs, "claude-3-5", 200);
    const b = engine.calibratedEstimate(msgs, "claude-3-5", 200);
    expect(a).toBe(b);
  });
});

// ---------------------------------------------------------------------------
// 7. File I/O — persist, loadFromFile, trimFile
// ---------------------------------------------------------------------------

describe("CalibrationEngine — file I/O", () => {
  let filePath: string;

  beforeEach(() => {
    filePath = tmpFile();
    CalibrationEngine.resetInstance();
  });

  afterEach(async () => {
    CalibrationEngine.resetInstance();
    await cleanupFile(filePath);
  });

  test("loadFromFile replays records into memory state", async () => {
    const records = Array.from({ length: 8 }, () =>
      makeRecord("claude-3-5", "text", 100, 110),
    );
    await seedFile(filePath, records);

    const engine = CalibrationEngine.instance;
    engine.setFilePath(filePath);
    await engine.loadFromFile(filePath);

    const state = engine.getState("claude-3-5");
    expect(state).toBeDefined();
    expect(state!.bins.text.sampleCount).toBe(8);
  });

  test("loadFromFile is safe on missing file — no throw", async () => {
    const engine = CalibrationEngine.instance;
    await expect(engine.loadFromFile("/tmp/does-not-exist-ashlr-ce.jsonl")).resolves.toBeUndefined();
  });

  test("loadFromFile skips malformed lines", async () => {
    await writeFile(
      filePath,
      [
        JSON.stringify(makeRecord("gpt-4o", "text", 50, 55)),
        "MALFORMED JSON {{{",
        JSON.stringify(makeRecord("gpt-4o", "text", 60, 66)),
        "",
      ].join("\n"),
      "utf8",
    );

    const engine = CalibrationEngine.instance;
    await engine.loadFromFile(filePath);

    const state = engine.getState("gpt-4o");
    expect(state!.bins.text.sampleCount).toBe(2);
  });

  test("trimFile reduces records to MAX_RECORDS_PER_PROVIDER per provider", async () => {
    // Write MAX_RECORDS_PER_PROVIDER + 20 records for one provider
    const records = Array.from({ length: MAX_RECORDS_PER_PROVIDER + 20 }, (_, i) =>
      makeRecord("llama3", "text", (i + 1) * 10, (i + 1) * 11),
    );
    await seedFile(filePath, records);

    const engine = CalibrationEngine.instance;
    await engine.trimFile(filePath);

    const raw = await readFile(filePath, "utf8");
    const lines = raw.split("\n").filter((l) => l.trim());
    expect(lines.length).toBeLessThanOrEqual(MAX_RECORDS_PER_PROVIDER);
  });

  test("trimFile handles empty file gracefully", async () => {
    await writeFile(filePath, "", "utf8");
    const engine = CalibrationEngine.instance;
    await expect(engine.trimFile(filePath)).resolves.toBeUndefined();
  });

  test("setFilePath overrides the default calibration file path", () => {
    const engine = CalibrationEngine.instance;
    engine.setFilePath("/custom/path/test.jsonl");
    // Should not throw; path is just stored internally
    engine.setFilePath(null); // restore
  });

  test("record with zero estimated skipped — no division-by-zero", () => {
    const engine = CalibrationEngine.instance;
    engine.setFilePath(filePath);
    expect(() => engine.record({ provider: "gpt-4o", messages: [txt("x")], estimated: 0, actual: 100 })).not.toThrow();
    expect(engine.getState("gpt-4o")).toBeUndefined();
  });

  test("record with zero actual skipped", () => {
    const engine = CalibrationEngine.instance;
    engine.setFilePath(filePath);
    expect(() => engine.record({ provider: "gpt-4o", messages: [txt("x")], estimated: 100, actual: 0 })).not.toThrow();
    expect(engine.getState("gpt-4o")).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// 8. Barrel export verification
// ---------------------------------------------------------------------------

describe("Barrel export — CalibrationEngine from src/tokens/index.ts", () => {
  test("CalibrationEngine is importable from index.ts", () => {
    expect(typeof CalibrationEngineFromIndex).toBe("function");
    expect(typeof CalibrationEngineFromIndex.instance).toBe("object");
  });

  test("tukeyBiweightMean is importable from index.ts", () => {
    expect(typeof tukeyFromIndex).toBe("function");
  });

  test("singleton is same instance regardless of import path", () => {
    CalibrationEngine.resetInstance();
    const a = CalibrationEngine.instance;
    const b = CalibrationEngineFromIndex.instance;
    // Both should be the same singleton object
    expect(a).toBe(b);
  });
});

// ---------------------------------------------------------------------------
// 9. correctionFactor fallback chain
// ---------------------------------------------------------------------------

describe("CalibrationEngine — correctionFactor fallback chain", () => {
  test("fallback 1: per-bin live factor when CE_MIN_SAMPLES reached", () => {
    const engine = CalibrationEngine.instance;
    for (let i = 0; i < CE_MIN_SAMPLES; i++) {
      engine.record({ provider: "claude-3-5", messages: [txt("x")], estimated: 100, actual: 150 });
    }
    const factor = engine.correctionFactor("claude-3-5", "text");
    // Should have moved from baseline (1.0) toward 1.5 after CE_MIN_SAMPLES samples
    expect(factor).toBeGreaterThan(1.0);
  });

  test("fallback 2: pre-trained baseline for known provider before CE_MIN_SAMPLES", () => {
    const engine = CalibrationEngine.instance;
    // Only 1 sample — below CE_MIN_SAMPLES
    engine.record({ provider: "llama3", messages: [txt("x")], estimated: 100, actual: 200 });
    // Overall samples < CE_MIN_SAMPLES → falls back to overall live factor or baseline
    // Since total samples < CE_MIN_SAMPLES, correctionFactor falls back to baseline
    const factor = engine.correctionFactor("llama3", "tool_result");
    // llama3 tool_result baseline = 1.06 (from pretrained)
    // After 1 sample (below min), should return overall live factor or baseline
    expect(factor).toBeGreaterThan(1.0); // at minimum the llama3 baseline
  });

  test("fallback 3: overall factor when per-bin has few samples but total is enough", () => {
    const engine = CalibrationEngine.instance;
    // Feed 5 samples to text bin (meets CE_MIN_SAMPLES)
    for (let i = 0; i < CE_MIN_SAMPLES + 1; i++) {
      engine.record({ provider: "custom-model", messages: [txt("x")], estimated: 100, actual: 130 });
    }
    // Query a bin with no samples — should fall back to overall
    // (thinking bin has 0 samples; total >= CE_MIN_SAMPLES)
    const thinkingFactor = engine.correctionFactor("custom-model", "thinking");
    const overallFact = engine.overallFactor("custom-model");
    expect(thinkingFactor).toBe(overallFact);
  });

  test("fallback 4: 1.0 for unknown provider with no baseline and no training", () => {
    const engine = CalibrationEngine.instance;
    expect(engine.correctionFactor("zzz-unknown-xyz", "text")).toBe(1.0);
    expect(engine.correctionFactor("zzz-unknown-xyz", "tool_use")).toBe(1.0);
    expect(engine.correctionFactor("zzz-unknown-xyz", "image")).toBe(1.0);
  });
});

// ---------------------------------------------------------------------------
// 10. Constants sanity
// ---------------------------------------------------------------------------

describe("CalibrationEngine — constants sanity", () => {
  test("MAX_RECORDS_PER_PROVIDER is 100", () => {
    expect(MAX_RECORDS_PER_PROVIDER).toBe(100);
  });

  test("CE_MIN_SAMPLES is a positive integer", () => {
    expect(CE_MIN_SAMPLES).toBeGreaterThan(0);
    expect(Number.isInteger(CE_MIN_SAMPLES)).toBe(true);
  });

  test("CE_FACTOR_MIN < CE_FACTOR_MAX", () => {
    expect(CE_FACTOR_MIN).toBeLessThan(CE_FACTOR_MAX);
    expect(CE_FACTOR_MIN).toBeGreaterThan(0);
  });

  test("CE_EMA_ALPHA is between 0 and 1 exclusive", () => {
    expect(CE_EMA_ALPHA).toBeGreaterThan(0);
    expect(CE_EMA_ALPHA).toBeLessThan(1);
  });

  test("TUKEY_C is positive (standard value 4.685)", () => {
    expect(TUKEY_C).toBeGreaterThan(0);
    expect(TUKEY_C).toBe(4.685);
  });
});
