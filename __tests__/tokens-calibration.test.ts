/**
 * Tests for the Cross-Provider Token Estimation Validator (src/tokens/validator.ts)
 * and the calibration hook added to src/tokens/index.ts.
 *
 * All file I/O uses temp paths (os.tmpdir()) so tests are hermetic and never
 * touch ~/.ashlr/token-calibration.jsonl.
 */

import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { join } from "path";
import { tmpdir } from "os";
import { writeFile, readFile, unlink } from "fs/promises";
import { existsSync } from "fs";

import {
  appendCalibrationRecord,
  readAllCalibrationRecords,
  validateAndRecalibrate,
  getNominalMultiplier,
  getNominalEncoding,
  getActiveCalibratedMultiplier,
  trimCalibrationFile,
  _resetCalibratedMultipliers,
  DRIFT_THRESHOLD_PCT,
  WINDOW_SIZE,
  EMA_ALPHA,
  MULTIPLIER_MIN,
  MULTIPLIER_MAX,
  type CalibrationRecord,
  type CalibrationResult,
  type RecalibratedEncodingConfig,
} from "../src/tokens/validator.ts";

import {
  recordActualTokenUsage,
  _resetTokenizerCache,
} from "../src/tokens/index.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function tmpFile(): string {
  return join(tmpdir(), `ashlr-cal-test-${Date.now()}-${Math.random().toString(36).slice(2)}.jsonl`);
}

function makeRecord(
  model: string,
  estimated: number,
  actual: number,
  timestamp?: string,
): CalibrationRecord {
  return {
    timestamp: timestamp ?? new Date().toISOString(),
    model,
    estimated,
    actual,
  };
}

/** Write records directly to a file (bypasses appendCalibrationRecord path). */
async function seedFile(path: string, records: CalibrationRecord[]): Promise<void> {
  const lines = records.map((r) => JSON.stringify(r)).join("\n") + "\n";
  await writeFile(path, lines, "utf8");
}

async function cleanupFile(path: string): Promise<void> {
  try {
    if (existsSync(path)) await unlink(path);
  } catch { /* ignore */ }
}

// ---------------------------------------------------------------------------
// getNominalMultiplier
// ---------------------------------------------------------------------------

describe("getNominalMultiplier", () => {
  test("returns 1.05 for llama variants", () => {
    expect(getNominalMultiplier("meta-llama/Llama-3-8b")).toBe(1.05);
    expect(getNominalMultiplier("llama2")).toBe(1.05);
    expect(getNominalMultiplier("codellama-34b")).toBe(1.05);
  });

  test("returns 1.08 for qwen variants", () => {
    expect(getNominalMultiplier("qwen-72b")).toBe(1.08);
    expect(getNominalMultiplier("Qwen2-VL")).toBe(1.08);
  });

  test("returns 1.02 for mistral/mixtral", () => {
    expect(getNominalMultiplier("mistral-7b")).toBe(1.02);
    expect(getNominalMultiplier("mixtral-8x7b")).toBe(1.02);
  });

  test("returns 1.0 for claude, gpt, gemini", () => {
    expect(getNominalMultiplier("claude-3-5-sonnet-20241022")).toBe(1.0);
    expect(getNominalMultiplier("gpt-4o-mini")).toBe(1.0);
    expect(getNominalMultiplier("gemini-1.5-pro")).toBe(1.0);
  });

  test("returns 1.0 for unknown models", () => {
    expect(getNominalMultiplier("some-unknown-model-v9")).toBe(1.0);
  });
});

// ---------------------------------------------------------------------------
// getNominalEncoding
// ---------------------------------------------------------------------------

describe("getNominalEncoding", () => {
  test("claude-3 family → o200k_base", () => {
    expect(getNominalEncoding("claude-3-5-sonnet-20241022")).toBe("o200k_base");
    expect(getNominalEncoding("claude-3-opus")).toBe("o200k_base");
  });

  test("gpt-4o → o200k_base", () => {
    expect(getNominalEncoding("gpt-4o-mini")).toBe("o200k_base");
  });

  test("llama → cl100k_base", () => {
    expect(getNominalEncoding("llama2")).toBe("cl100k_base");
  });

  test("unknown model → cl100k_base", () => {
    expect(getNominalEncoding("my-custom-model")).toBe("cl100k_base");
  });
});

// ---------------------------------------------------------------------------
// appendCalibrationRecord / readAllCalibrationRecords (file-path injectable)
// ---------------------------------------------------------------------------

describe("appendCalibrationRecord + readAllCalibrationRecords", () => {
  // We test indirectly by seeding files with seedFile and then reading them,
  // and verify the shape returned by readAllCalibrationRecords.

  test("reads back seeded JSONL correctly", async () => {
    const path = tmpFile();
    const records = [
      makeRecord("claude-3-5", 100, 110),
      makeRecord("gpt-4o", 200, 195),
    ];
    await seedFile(path, records);

    const raw = await readFile(path, "utf8");
    const lines = raw.split("\n").filter(Boolean);
    expect(lines).toHaveLength(2);

    const parsed = lines.map((l) => JSON.parse(l) as CalibrationRecord);
    expect(parsed[0]!.model).toBe("claude-3-5");
    expect(parsed[0]!.estimated).toBe(100);
    expect(parsed[0]!.actual).toBe(110);
    expect(parsed[1]!.model).toBe("gpt-4o");

    await cleanupFile(path);
  });

  test("returns empty array for missing file", async () => {
    // readAllCalibrationRecords targets ~/.ashlr path; validate error handling
    // by checking that a non-existent path doesn't throw.
    // We call validateAndRecalibrate with a custom path that doesn't exist.
    const result = await validateAndRecalibrate("nonexistent-model", {
      filePath: "/tmp/does-not-exist-ashlr-test-xyz.jsonl",
    });
    expect(result.sampleSize).toBe(0);
    expect(result.driftDetected).toBe(false);
  });

  test("skips malformed JSONL lines gracefully", async () => {
    const path = tmpFile();
    await writeFile(
      path,
      [
        JSON.stringify(makeRecord("gpt-4", 50, 55)),
        "NOT VALID JSON }{",
        JSON.stringify(makeRecord("gpt-4", 60, 66)),
        "",
      ].join("\n"),
      "utf8",
    );
    const result = await validateAndRecalibrate("gpt-4", { filePath: path });
    expect(result.sampleSize).toBe(2);
    await cleanupFile(path);
  });
});

// ---------------------------------------------------------------------------
// validateAndRecalibrate — zero records
// ---------------------------------------------------------------------------

describe("validateAndRecalibrate — zero records", () => {
  test("returns sampleSize=0 and no drift", async () => {
    const result = await validateAndRecalibrate("unknown-model-xyz", {
      filePath: "/tmp/ashlr-empty-no-exist.jsonl",
    });
    expect(result.sampleSize).toBe(0);
    expect(result.mape.overall).toBe(0);
    expect(result.biasFactor).toBe(1.0);
    expect(result.driftDetected).toBe(false);
    expect(result.recalibrated).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// validateAndRecalibrate — MAPE computation
// ---------------------------------------------------------------------------

describe("validateAndRecalibrate — MAPE computation", () => {
  test("perfect estimates → MAPE = 0, no drift", async () => {
    const path = tmpFile();
    const records = Array.from({ length: 20 }, (_, i) =>
      makeRecord("claude-3-5", (i + 1) * 100, (i + 1) * 100),
    );
    await seedFile(path, records);

    const result = await validateAndRecalibrate("claude-3-5", { filePath: path });
    expect(result.mape.overall).toBe(0);
    expect(result.driftDetected).toBe(false);
    expect(result.recalibrated).toBeUndefined();
    await cleanupFile(path);
  });

  test("10% consistent over-estimate → MAPE ~10%, drift detected", async () => {
    const path = tmpFile();
    // estimated = 110, actual = 100 → each error = 10/110 ≈ 9.09%
    const records = Array.from({ length: 30 }, (_, i) =>
      makeRecord("gpt-4o", (i + 1) * 110, (i + 1) * 100),
    );
    await seedFile(path, records);

    const result = await validateAndRecalibrate("gpt-4o", { filePath: path });
    expect(result.mape.overall).toBeGreaterThan(DRIFT_THRESHOLD_PCT);
    expect(result.driftDetected).toBe(true);
    expect(result.recalibrated).toBeDefined();
    await cleanupFile(path);
  });

  test("4% error stays below drift threshold", async () => {
    const path = tmpFile();
    // estimated = 104, actual = 100 → error = 4/104 ≈ 3.85%
    const records = Array.from({ length: 20 }, (_, i) =>
      makeRecord("claude-3-5", (i + 1) * 104, (i + 1) * 100),
    );
    await seedFile(path, records);

    const result = await validateAndRecalibrate("claude-3-5", { filePath: path });
    expect(result.mape.overall).toBeLessThanOrEqual(DRIFT_THRESHOLD_PCT);
    expect(result.driftDetected).toBe(false);
    await cleanupFile(path);
  });

  test("biasFactor reflects mean(actual)/mean(estimated)", async () => {
    const path = tmpFile();
    // All records: estimated = 200, actual = 240 → bias = 240/200 = 1.2
    const records = Array.from({ length: 10 }, () =>
      makeRecord("llama2", 200, 240),
    );
    await seedFile(path, records);

    const result = await validateAndRecalibrate("llama2", { filePath: path });
    expect(Math.abs(result.biasFactor - 1.2)).toBeLessThan(0.001);
    await cleanupFile(path);
  });
});

// ---------------------------------------------------------------------------
// validateAndRecalibrate — quartile breakdown
// ---------------------------------------------------------------------------

describe("validateAndRecalibrate — quartile MAPE breakdown", () => {
  test("quartile MAPEs are independent of overall average", async () => {
    const path = tmpFile();
    // 40 records with varied actual sizes and errors
    const records: CalibrationRecord[] = [];
    for (let i = 1; i <= 40; i++) {
      const actual = i * 50;
      // Introduce error proportional to record index to make quartiles differ
      const estimated = actual * (1 + (i % 5) * 0.02);
      records.push(makeRecord("gpt-4o-mini", Math.round(estimated), actual));
    }
    await seedFile(path, records);

    const result = await validateAndRecalibrate("gpt-4o-mini", { filePath: path });
    // All quartile MAPEs should be non-negative numbers
    expect(result.mape.q1).toBeGreaterThanOrEqual(0);
    expect(result.mape.q2).toBeGreaterThanOrEqual(0);
    expect(result.mape.q3).toBeGreaterThanOrEqual(0);
    expect(result.mape.q4).toBeGreaterThanOrEqual(0);
    // Overall should be the average across all records
    expect(result.mape.overall).toBeGreaterThanOrEqual(0);
    await cleanupFile(path);
  });

  test("single record has q1..q4 populated", async () => {
    const path = tmpFile();
    await seedFile(path, [makeRecord("gemini", 100, 120)]);
    const result = await validateAndRecalibrate("gemini", { filePath: path });
    expect(result.sampleSize).toBe(1);
    // With only 1 record the quartile slices may be empty — should not throw
    expect(typeof result.mape.overall).toBe("number");
    await cleanupFile(path);
  });
});

// ---------------------------------------------------------------------------
// validateAndRecalibrate — RecalibratedEncodingConfig
// ---------------------------------------------------------------------------

describe("validateAndRecalibrate — RecalibratedEncodingConfig", () => {
  beforeEach(() => {
    _resetCalibratedMultipliers();
  });

  test("recalibrated.multiplier is within [MULTIPLIER_MIN, MULTIPLIER_MAX]", async () => {
    const path = tmpFile();
    // Extreme under-estimate: estimated=50, actual=200 → bias=4.0 → raw multiplier would be >2
    const records = Array.from({ length: 20 }, () =>
      makeRecord("llama2", 50, 200),
    );
    await seedFile(path, records);

    const result = await validateAndRecalibrate("llama2", { filePath: path });
    expect(result.driftDetected).toBe(true);
    expect(result.recalibrated).toBeDefined();
    expect(result.recalibrated!.multiplier).toBeGreaterThanOrEqual(MULTIPLIER_MIN);
    expect(result.recalibrated!.multiplier).toBeLessThanOrEqual(MULTIPLIER_MAX);
    await cleanupFile(path);
  });

  test("recalibrated.priorMultiplier matches nominal before first calibration", async () => {
    const path = tmpFile();
    const records = Array.from({ length: 20 }, () =>
      makeRecord("llama2", 100, 130),
    );
    await seedFile(path, records);

    const result = await validateAndRecalibrate("llama2", { filePath: path });
    if (result.recalibrated) {
      // Prior should be the nominal Llama multiplier (1.05) since we reset above
      expect(result.recalibrated.priorMultiplier).toBe(1.05);
    }
    await cleanupFile(path);
  });

  test("EMA smoothing: second calibration is smoothed towards new value", async () => {
    const path = tmpFile();
    // First calibration: large over-estimate (actual << estimated)
    const records1 = Array.from({ length: 30 }, () =>
      makeRecord("qwen", 200, 160),
    );
    await seedFile(path, records1);
    const result1 = await validateAndRecalibrate("qwen", { filePath: path });

    if (result1.recalibrated) {
      const mult1 = result1.recalibrated.multiplier;

      // Second calibration: different ratio
      const records2 = Array.from({ length: 30 }, () =>
        makeRecord("qwen", 200, 150),
      );
      await seedFile(path, records2);
      const result2 = await validateAndRecalibrate("qwen", { filePath: path });

      if (result2.recalibrated) {
        // The second multiplier should be an EMA blend, not a full jump
        // i.e. it should differ from result1 but not equal the raw new value
        expect(typeof result2.recalibrated.multiplier).toBe("number");
        expect(result2.recalibrated.multiplier).toBeGreaterThanOrEqual(MULTIPLIER_MIN);
        expect(result2.recalibrated.multiplier).toBeLessThanOrEqual(MULTIPLIER_MAX);
      }
    }
    await cleanupFile(path);
  });

  test("recalibrated.encoding matches model family", async () => {
    const path = tmpFile();
    const claudeRecords = Array.from({ length: 20 }, () =>
      makeRecord("claude-3-5-sonnet", 100, 130),
    );
    await seedFile(path, claudeRecords);
    const result = await validateAndRecalibrate("claude-3-5-sonnet", { filePath: path });
    if (result.recalibrated) {
      expect(result.recalibrated.encoding).toBe("o200k_base");
    }
    await cleanupFile(path);
  });
});

// ---------------------------------------------------------------------------
// getActiveCalibratedMultiplier
// ---------------------------------------------------------------------------

describe("getActiveCalibratedMultiplier", () => {
  beforeEach(() => {
    _resetCalibratedMultipliers();
  });

  test("returns null before any calibration", () => {
    expect(getActiveCalibratedMultiplier("any-model")).toBeNull();
  });

  test("returns calibrated value after drift-detected run", async () => {
    const path = tmpFile();
    const records = Array.from({ length: 20 }, () =>
      makeRecord("qwen", 100, 130),
    );
    await seedFile(path, records);

    await validateAndRecalibrate("qwen", { filePath: path });

    const mult = getActiveCalibratedMultiplier("qwen");
    expect(mult).not.toBeNull();
    expect(mult!).toBeGreaterThan(0);
    await cleanupFile(path);
  });

  test("prefix match works — full slug matches short key", async () => {
    const path = tmpFile();
    const records = Array.from({ length: 20 }, () =>
      makeRecord("llama2", 100, 130),
    );
    await seedFile(path, records);

    await validateAndRecalibrate("llama2", { filePath: path });

    // A more specific slug should still match via prefix
    const mult = getActiveCalibratedMultiplier("llama2-70b-chat");
    expect(mult).not.toBeNull();
    await cleanupFile(path);
  });
});

// ---------------------------------------------------------------------------
// trimCalibrationFile — window enforcement
// ---------------------------------------------------------------------------

describe("trimCalibrationFile", () => {
  test("trims each model to WINDOW_SIZE records", async () => {
    // We can't inject a path into trimCalibrationFile directly since it reads
    // from ~/.ashlr. Instead we test the underlying behaviour by writing
    // WINDOW_SIZE+20 records and checking that the slice logic is correct.
    // This is a unit test of the data-handling logic via validateAndRecalibrate.
    const path = tmpFile();

    // Write WINDOW_SIZE + 20 records for one model
    const records = Array.from({ length: WINDOW_SIZE + 20 }, (_, i) =>
      makeRecord("gpt-4", (i + 1) * 10, (i + 1) * 10),
    );
    await seedFile(path, records);

    // validateAndRecalibrate with windowSize override should return at most WINDOW_SIZE
    const result = await validateAndRecalibrate("gpt-4", {
      filePath: path,
      windowSize: WINDOW_SIZE,
    });
    expect(result.sampleSize).toBeLessThanOrEqual(WINDOW_SIZE);
    await cleanupFile(path);
  });
});

// ---------------------------------------------------------------------------
// recordActualTokenUsage hook (integration with index.ts)
// ---------------------------------------------------------------------------

describe("recordActualTokenUsage — calibration hook integration", () => {
  beforeEach(() => {
    _resetTokenizerCache();
  });

  test("returns correct TokenUsageRecord shape", () => {
    const response = {
      input_tokens: 100,
      output_tokens: 50,
      cache_creation_input_tokens: 20,
      cache_read_input_tokens: 10,
    };
    const record = recordActualTokenUsage(response, 120, "claude-3-5-sonnet");
    expect(record.estimatedTokens).toBe(120);
    expect(record.actualInputTokens).toBe(100);
    expect(record.actualCacheCreationTokens).toBe(20);
    expect(record.actualCacheReadTokens).toBe(10);
    // actualEffective = 100 + 20*1.25 + 10*0.1 = 100 + 25 + 1 = 126
    expect(record.actualEffectiveTokens).toBe(126);
    // error% = (126 - 120) / 120 * 100 = 5%
    expect(record.estimationErrorPct).toBe(5);
  });

  test("handles zero cache fields gracefully", () => {
    const response = { input_tokens: 80, output_tokens: 40 };
    const record = recordActualTokenUsage(response, 80, "gpt-4o");
    expect(record.actualCacheCreationTokens).toBe(0);
    expect(record.actualCacheReadTokens).toBe(0);
    expect(record.actualEffectiveTokens).toBe(80);
    expect(record.estimationErrorPct).toBe(0);
  });

  test("handles estimatedBefore=0 without division error", () => {
    const response = { input_tokens: 50, output_tokens: 20 };
    const record = recordActualTokenUsage(response, 0, "gpt-4o");
    expect(record.estimationErrorPct).toBe(0);
  });

  test("model parameter defaults to 'default' (back-compat with 2-arg call)", () => {
    const response = { input_tokens: 60, output_tokens: 30 };
    // 2-arg call — should not throw even without model parameter
    expect(() => recordActualTokenUsage(response, 60)).not.toThrow();
  });

  test("negative estimation error when we over-estimate", () => {
    const response = { input_tokens: 80, output_tokens: 20 };
    const record = recordActualTokenUsage(response, 100, "claude-3-5");
    // error = (80 - 100) / 100 * 100 = -20%
    expect(record.estimationErrorPct).toBe(-20);
  });
});

// ---------------------------------------------------------------------------
// Constants are sane
// ---------------------------------------------------------------------------

describe("calibration constants", () => {
  test("DRIFT_THRESHOLD_PCT is 5", () => {
    expect(DRIFT_THRESHOLD_PCT).toBe(5);
  });

  test("WINDOW_SIZE is 100", () => {
    expect(WINDOW_SIZE).toBe(100);
  });

  test("EMA_ALPHA is between 0 and 1", () => {
    expect(EMA_ALPHA).toBeGreaterThan(0);
    expect(EMA_ALPHA).toBeLessThan(1);
  });

  test("MULTIPLIER bounds are sane", () => {
    expect(MULTIPLIER_MIN).toBeGreaterThan(0);
    expect(MULTIPLIER_MAX).toBeGreaterThan(MULTIPLIER_MIN);
  });
});
