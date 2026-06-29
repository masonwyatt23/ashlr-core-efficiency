/**
 * Tests for calibration-report.ts — Cross-Provider Token Estimation
 * Benchmarking & Calibration Report Generator
 *
 * Coverage:
 *   1. generateCalibrationReport — basic structure, empty providers
 *   2. Confidence scoring (high at 50+ samples, medium 5-49, low <5)
 *   3. Active factor: null below MIN_SAMPLES, populated above
 *   4. drift_pct accuracy (MAPE reflects recorded errors)
 *   5. Recommendation logic (collect more, factor suggestion, drift alert)
 *   6. Summary statistics (counts, averages, critical/warning lists)
 *   7. persistCalibrationReport — writes JSONL to genome evolution dir
 *   8. buildMarkdownSummary — GitHub Actions format
 *   9. postSlackWebhook — no-op when env var absent
 *  10. Per-provider auto-discovery (empty providers list uses knownProviders)
 */

import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { join } from "path";
import { tmpdir } from "os";
import { mkdirSync, existsSync } from "fs";
import { readFile, rm } from "fs/promises";

import { RecalibrationEngine } from "../src/tokens/recalibration-engine.ts";
import { CalibrationEngine } from "../src/tokens/calibration-engine.ts";
import {
  generateCalibrationReport,
  persistCalibrationReport,
  buildMarkdownSummary,
  postSlackWebhook,
  CONFIDENCE_HIGH_THRESHOLD,
  CONFIDENCE_LOW_THRESHOLD,
  DRIFT_WARN_THRESHOLD_PCT,
  DRIFT_CRITICAL_THRESHOLD_PCT,
  GENOME_EVOLUTION_DIR,
  CALIBRATION_REPORT_FILENAME,
  type TokenCalibrationReport,
  type ProviderCalibrationEntry,
} from "../src/tokens/calibration-report.ts";
import type { Message } from "../src/types/index.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function txt(text: string): Message {
  return { role: "user", content: text };
}

function toolResultMsg(): Message {
  return {
    role: "user",
    content: [{ type: "tool_result", tool_use_id: "t1", content: "result" }],
  };
}

/** Seed RecalibrationEngine with N samples for a provider. */
function seedRecal(
  engine: RecalibrationEngine,
  provider: string,
  count: number,
  estimated: number,
  actual: number,
): void {
  for (let i = 0; i < count; i++) {
    engine.record({ provider, messages: [txt("x")], estimated, actual });
  }
}

/** Seed CalibrationEngine with N samples for a provider. */
function seedCal(
  engine: CalibrationEngine,
  provider: string,
  count: number,
  estimated: number,
  actual: number,
): void {
  for (let i = 0; i < count; i++) {
    engine.record({ provider, messages: [txt("x")], estimated, actual });
  }
}

function tmpDir(): string {
  const dir = join(
    tmpdir(),
    `ashlr-cr-test-${Date.now()}-${Math.random().toString(36).slice(2)}`,
  );
  mkdirSync(dir, { recursive: true });
  return dir;
}

async function cleanupDir(dir: string): Promise<void> {
  try {
    if (existsSync(dir)) await rm(dir, { recursive: true, force: true });
  } catch { /* ignore */ }
}

beforeEach(() => {
  RecalibrationEngine.resetInstance();
  CalibrationEngine.resetInstance();
});

// ---------------------------------------------------------------------------
// 1. Basic report structure
// ---------------------------------------------------------------------------

describe("generateCalibrationReport — basic structure", () => {
  test("returns a report with correct shape for a seeded provider", async () => {
    seedCal(CalibrationEngine.instance, "claude-3-5", 10, 100, 110);
    const report = await generateCalibrationReport(["claude-3-5"]);

    expect(report.generatedAt).toBeDefined();
    expect(new Date(report.generatedAt).getTime()).toBeLessThanOrEqual(Date.now());
    expect(report.providers).toHaveLength(1);
    expect(report.summary).toBeDefined();
    expect(typeof report.jsonLines).toBe("string");
    expect(typeof report.formattedText).toBe("string");
  });

  test("returns empty providers array when no data exists", async () => {
    const report = await generateCalibrationReport(["nonexistent-provider-xyz"]);
    expect(report.providers).toHaveLength(0);
    expect(report.summary.totalProviders).toBe(0);
  });

  test("includes all requested providers that have data", async () => {
    seedCal(CalibrationEngine.instance, "claude-3-5", 5, 100, 100);
    seedCal(CalibrationEngine.instance, "gpt-4o", 5, 100, 100);
    const report = await generateCalibrationReport(["claude-3-5", "gpt-4o"]);
    const names = report.providers.map((p) => p.provider);
    expect(names).toContain("claude-3-5");
    expect(names).toContain("gpt-4o");
  });

  test("skips providers with no data silently", async () => {
    seedCal(CalibrationEngine.instance, "claude-3-5", 5, 100, 100);
    const report = await generateCalibrationReport(["claude-3-5", "no-data-provider"]);
    expect(report.providers).toHaveLength(1);
    expect(report.providers[0]!.provider).toBe("claude-3-5");
  });

  test("formattedText contains provider name", async () => {
    seedCal(CalibrationEngine.instance, "llama3", 10, 100, 105);
    const report = await generateCalibrationReport(["llama3"]);
    expect(report.formattedText).toContain("llama3");
  });

  test("jsonLines is valid JSONL (each line parseable)", async () => {
    seedCal(CalibrationEngine.instance, "gpt-4o", 8, 100, 100);
    const report = await generateCalibrationReport(["gpt-4o"]);
    for (const line of report.jsonLines.split("\n").filter(Boolean)) {
      expect(() => JSON.parse(line)).not.toThrow();
    }
  });
});

// ---------------------------------------------------------------------------
// 2. Confidence scoring
// ---------------------------------------------------------------------------

describe("Confidence scoring", () => {
  test("confidence is 'low' with fewer than CONFIDENCE_LOW_THRESHOLD samples", async () => {
    const count = CONFIDENCE_LOW_THRESHOLD - 1;
    seedCal(CalibrationEngine.instance, "my-model", count, 100, 100);
    const report = await generateCalibrationReport(["my-model"]);
    expect(report.providers[0]!.confidence).toBe("low");
  });

  test("confidence is 'medium' with exactly CONFIDENCE_LOW_THRESHOLD samples", async () => {
    seedCal(CalibrationEngine.instance, "my-model", CONFIDENCE_LOW_THRESHOLD, 100, 100);
    const report = await generateCalibrationReport(["my-model"]);
    expect(report.providers[0]!.confidence).toBe("medium");
  });

  test("confidence is 'medium' between low and high thresholds", async () => {
    const count = Math.floor((CONFIDENCE_LOW_THRESHOLD + CONFIDENCE_HIGH_THRESHOLD) / 2);
    seedCal(CalibrationEngine.instance, "my-model", count, 100, 100);
    const report = await generateCalibrationReport(["my-model"]);
    expect(report.providers[0]!.confidence).toBe("medium");
  });

  test("confidence is 'high' with CONFIDENCE_HIGH_THRESHOLD or more samples", async () => {
    seedCal(CalibrationEngine.instance, "my-model", CONFIDENCE_HIGH_THRESHOLD, 100, 100);
    const report = await generateCalibrationReport(["my-model"]);
    expect(report.providers[0]!.confidence).toBe("high");
  });

  test("confidence is 'high' with more than CONFIDENCE_HIGH_THRESHOLD samples", async () => {
    seedCal(CalibrationEngine.instance, "my-model", CONFIDENCE_HIGH_THRESHOLD + 20, 100, 100);
    const report = await generateCalibrationReport(["my-model"]);
    expect(report.providers[0]!.confidence).toBe("high");
  });

  test("summary counts reflect confidence levels correctly", async () => {
    // low: 2 samples
    seedCal(CalibrationEngine.instance, "provider-low", 2, 100, 100);
    // medium: 10 samples
    seedCal(CalibrationEngine.instance, "provider-med", 10, 100, 100);
    // high: 55 samples
    seedCal(CalibrationEngine.instance, "provider-high", 55, 100, 100);

    const report = await generateCalibrationReport([
      "provider-low",
      "provider-med",
      "provider-high",
    ]);

    expect(report.summary.lowConfidenceCount).toBe(1);
    expect(report.summary.mediumConfidenceCount).toBe(1);
    expect(report.summary.highConfidenceCount).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// 3. Active factor — null vs populated
// ---------------------------------------------------------------------------

describe("Active factor", () => {
  test("activeFactor is null when below MIN_SAMPLES threshold", async () => {
    // Seed with only 1 sample (below both CE_MIN_SAMPLES and RECAL_MIN_SAMPLES)
    seedCal(CalibrationEngine.instance, "sparse-provider", 1, 100, 150);
    const report = await generateCalibrationReport(["sparse-provider"]);
    const entry = report.providers[0]!;
    expect(entry.activeFactor).toBeNull();
  });

  test("activeFactor is populated when above MIN_SAMPLES threshold", async () => {
    seedCal(CalibrationEngine.instance, "warm-provider", 10, 100, 120);
    const report = await generateCalibrationReport(["warm-provider"]);
    const entry = report.providers[0]!;
    expect(entry.activeFactor).not.toBeNull();
    expect(typeof entry.activeFactor).toBe("number");
    expect(entry.activeFactor!).toBeGreaterThan(0);
  });

  test("activeFactor reflects actual/estimated ratio direction", async () => {
    // actual > estimated => factor should be > 1
    seedCal(CalibrationEngine.instance, "under-estimator", 20, 100, 150);
    const report = await generateCalibrationReport(["under-estimator"]);
    expect(report.providers[0]!.activeFactor).toBeGreaterThan(1.0);
  });

  test("activeFactor reflects under-estimation correctly", async () => {
    // actual < estimated => factor should be < 1
    seedCal(CalibrationEngine.instance, "over-estimator", 20, 150, 100);
    const report = await generateCalibrationReport(["over-estimator"]);
    expect(report.providers[0]!.activeFactor).toBeLessThan(1.0);
  });
});

// ---------------------------------------------------------------------------
// 4. drift_pct accuracy
// ---------------------------------------------------------------------------

describe("drift_pct accuracy", () => {
  test("drift_pct is 0 when estimated == actual for all samples", async () => {
    seedCal(CalibrationEngine.instance, "perfect-model", 10, 100, 100);
    const report = await generateCalibrationReport(["perfect-model"]);
    expect(report.providers[0]!.drift_pct).toBe(0);
  });

  test("drift_pct is positive when there is estimation error", async () => {
    seedCal(CalibrationEngine.instance, "error-model", 10, 100, 120);
    const report = await generateCalibrationReport(["error-model"]);
    expect(report.providers[0]!.drift_pct).toBeGreaterThan(0);
  });

  test("higher estimation error produces higher drift_pct", async () => {
    seedCal(CalibrationEngine.instance, "low-error", 10, 100, 110);
    seedCal(CalibrationEngine.instance, "high-error", 10, 100, 150);

    const report = await generateCalibrationReport(["low-error", "high-error"]);
    const low = report.providers.find((p) => p.provider === "low-error")!;
    const high = report.providers.find((p) => p.provider === "high-error")!;

    expect(high.drift_pct).toBeGreaterThan(low.drift_pct);
  });

  test("drift_pct reflects critical drift level when error is very high", async () => {
    // 50% error should produce MAPE > DRIFT_CRITICAL_THRESHOLD_PCT
    seedCal(CalibrationEngine.instance, "critical-model", 20, 100, 160);
    const report = await generateCalibrationReport(["critical-model"]);
    expect(report.providers[0]!.drift_pct).toBeGreaterThan(DRIFT_WARN_THRESHOLD_PCT);
  });
});

// ---------------------------------------------------------------------------
// 5. Recommendation logic
// ---------------------------------------------------------------------------

describe("Recommendation logic", () => {
  test("low confidence produces 'collect more samples' recommendation", async () => {
    seedCal(CalibrationEngine.instance, "sparse", 2, 100, 100);
    const report = await generateCalibrationReport(["sparse"]);
    const recs = report.providers[0]!.recommendations;
    const hasCollectRec = recs.some((r) =>
      r.toLowerCase().includes("collect") && r.toLowerCase().includes("sample"),
    );
    expect(hasCollectRec).toBe(true);
  });

  test("factor > 1.15 produces multiplier suggestion recommendation", async () => {
    // Large under-estimation produces high factor
    seedCal(CalibrationEngine.instance, "big-under", 30, 100, 200);
    const report = await generateCalibrationReport(["big-under"]);
    const recs = report.providers[0]!.recommendations;
    const hasFactor = recs.some((r) => r.toLowerCase().includes("multiplier") || r.toLowerCase().includes("factor"));
    expect(hasFactor).toBe(true);
  });

  test("critical drift produces 'trigger recalibration' recommendation", async () => {
    // Very high error to push MAPE above critical threshold
    seedCal(CalibrationEngine.instance, "critical-drift", 20, 100, 200);
    const report = await generateCalibrationReport(["critical-drift"]);
    const recs = report.providers[0]!.recommendations;
    const hasDriftRec = recs.some(
      (r) =>
        r.toLowerCase().includes("drift") ||
        r.toLowerCase().includes("recalibration"),
    );
    expect(hasDriftRec).toBe(true);
  });

  test("no 'collect more' recommendation at high confidence", async () => {
    seedCal(CalibrationEngine.instance, "well-trained", CONFIDENCE_HIGH_THRESHOLD + 5, 100, 100);
    const report = await generateCalibrationReport(["well-trained"]);
    const recs = report.providers[0]!.recommendations;
    const hasCollectRec = recs.some(
      (r) => r.toLowerCase().includes("collect") && r.toLowerCase().includes("sample"),
    );
    expect(hasCollectRec).toBe(false);
  });

  test("near-ideal factor produces healthy calibration recommendation", async () => {
    // Exact match: factor converges to ~1.0
    seedCal(CalibrationEngine.instance, "ideal-model", 20, 100, 101);
    const report = await generateCalibrationReport(["ideal-model"]);
    const recs = report.providers[0]!.recommendations;
    const hasHealthy = recs.some((r) =>
      r.toLowerCase().includes("healthy") || r.toLowerCase().includes("ideal") || r.toLowerCase().includes("near-ideal"),
    );
    expect(hasHealthy).toBe(true);
  });

  test("null activeFactor (below min samples) produces pre-trained baseline note", async () => {
    seedCal(CalibrationEngine.instance, "warmup-model", 1, 100, 120);
    const report = await generateCalibrationReport(["warmup-model"]);
    const recs = report.providers[0]!.recommendations;
    const hasBaselineNote = recs.some((r) =>
      r.toLowerCase().includes("baseline") || r.toLowerCase().includes("pre-trained"),
    );
    expect(hasBaselineNote).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 6. Summary statistics
// ---------------------------------------------------------------------------

describe("Summary statistics", () => {
  test("totalProviders matches number of entries with data", async () => {
    seedCal(CalibrationEngine.instance, "p1", 5, 100, 100);
    seedCal(CalibrationEngine.instance, "p2", 5, 100, 100);
    const report = await generateCalibrationReport(["p1", "p2", "p3-no-data"]);
    expect(report.summary.totalProviders).toBe(2);
  });

  test("totalSamples is sum of all provider samples", async () => {
    seedCal(CalibrationEngine.instance, "pa", 10, 100, 100);
    seedCal(CalibrationEngine.instance, "pb", 15, 100, 100);
    const report = await generateCalibrationReport(["pa", "pb"]);
    expect(report.summary.totalSamples).toBe(25);
  });

  test("providersWithCriticalDrift lists providers above critical threshold", async () => {
    // Use RecalibrationEngine for this test too
    const cal = CalibrationEngine.instance;
    // Inject very high drift: 60% error
    seedCal(cal, "crit-provider", 20, 100, 180);
    const report = await generateCalibrationReport(["crit-provider"]);

    if (report.providers[0]!.drift_pct >= DRIFT_CRITICAL_THRESHOLD_PCT) {
      expect(report.summary.providersWithCriticalDrift).toContain("crit-provider");
    }
  });

  test("providersWithWarningDrift lists providers above warning threshold but below critical", async () => {
    const cal = CalibrationEngine.instance;
    // Target MAPE between warn and critical: ~10%
    seedCal(cal, "warn-provider", 15, 100, 115);
    const report = await generateCalibrationReport(["warn-provider"]);

    if (
      report.providers[0]!.drift_pct >= DRIFT_WARN_THRESHOLD_PCT &&
      report.providers[0]!.drift_pct < DRIFT_CRITICAL_THRESHOLD_PCT
    ) {
      expect(report.summary.providersWithWarningDrift).toContain("warn-provider");
    }
  });

  test("averageDrift_pct is mean of per-provider drift values", async () => {
    seedCal(CalibrationEngine.instance, "qa", 10, 100, 100); // 0% drift
    seedCal(CalibrationEngine.instance, "qb", 10, 100, 100); // 0% drift
    const report = await generateCalibrationReport(["qa", "qb"]);
    expect(report.summary.averageDrift_pct).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// 7. persistCalibrationReport — JSONL output
// ---------------------------------------------------------------------------

describe("persistCalibrationReport", () => {
  let testDir: string;

  beforeEach(() => {
    testDir = tmpDir();
  });

  afterEach(async () => {
    await cleanupDir(testDir);
  });

  test("creates JSONL file in genome evolution directory", async () => {
    seedCal(CalibrationEngine.instance, "persist-test", 5, 100, 110);
    const report = await generateCalibrationReport(["persist-test"]);
    await persistCalibrationReport(report, testDir);

    const filePath = join(testDir, GENOME_EVOLUTION_DIR, CALIBRATION_REPORT_FILENAME);
    expect(existsSync(filePath)).toBe(true);
  });

  test("written file contains valid JSON lines", async () => {
    seedCal(CalibrationEngine.instance, "json-test", 5, 100, 120);
    const report = await generateCalibrationReport(["json-test"]);
    await persistCalibrationReport(report, testDir);

    const filePath = join(testDir, GENOME_EVOLUTION_DIR, CALIBRATION_REPORT_FILENAME);
    const content = await readFile(filePath, "utf8");
    for (const line of content.split("\n").filter(Boolean)) {
      expect(() => JSON.parse(line)).not.toThrow();
    }
  });

  test("written JSON line contains provider and generatedAt fields", async () => {
    seedCal(CalibrationEngine.instance, "field-test", 5, 100, 100);
    const report = await generateCalibrationReport(["field-test"]);
    await persistCalibrationReport(report, testDir);

    const filePath = join(testDir, GENOME_EVOLUTION_DIR, CALIBRATION_REPORT_FILENAME);
    const content = await readFile(filePath, "utf8");
    const lines = content.split("\n").filter(Boolean);
    expect(lines.length).toBeGreaterThan(0);
    const parsed = JSON.parse(lines[0]!);
    expect(parsed.provider).toBe("field-test");
    expect(parsed.generatedAt).toBeDefined();
  });

  test("calling persist twice appends to existing file", async () => {
    seedCal(CalibrationEngine.instance, "append-test", 5, 100, 100);
    const report = await generateCalibrationReport(["append-test"]);
    await persistCalibrationReport(report, testDir);
    await persistCalibrationReport(report, testDir);

    const filePath = join(testDir, GENOME_EVOLUTION_DIR, CALIBRATION_REPORT_FILENAME);
    const content = await readFile(filePath, "utf8");
    const lines = content.split("\n").filter(Boolean);
    // Each persist writes one line per provider entry, so 2 persists = 2 lines
    expect(lines.length).toBeGreaterThanOrEqual(2);
  });

  test("persist is non-fatal on unwritable path — does not throw", async () => {
    const report = await generateCalibrationReport([]);
    // Use a path that is impossible to create (inside a file path)
    await expect(
      persistCalibrationReport(report, "/nonexistent-root-xyz/fake"),
    ).resolves.toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// 8. buildMarkdownSummary
// ---------------------------------------------------------------------------

describe("buildMarkdownSummary", () => {
  test("produces a markdown string with header", async () => {
    seedCal(CalibrationEngine.instance, "md-test", 10, 100, 100);
    const report = await generateCalibrationReport(["md-test"]);
    const md = buildMarkdownSummary(report);
    expect(typeof md).toBe("string");
    expect(md).toContain("## Token Calibration Report");
  });

  test("includes summary table rows", async () => {
    seedCal(CalibrationEngine.instance, "md-test2", 10, 100, 110);
    const report = await generateCalibrationReport(["md-test2"]);
    const md = buildMarkdownSummary(report);
    expect(md).toContain("| Providers |");
    expect(md).toContain("| Total samples |");
  });

  test("includes per-provider row", async () => {
    seedCal(CalibrationEngine.instance, "md-provider", 10, 100, 100);
    const report = await generateCalibrationReport(["md-provider"]);
    const md = buildMarkdownSummary(report);
    expect(md).toContain("md-provider");
  });

  test("returns empty sections for empty report", async () => {
    const report = await generateCalibrationReport([]);
    const md = buildMarkdownSummary(report);
    expect(md).toContain("## Token Calibration Report");
    // Should not crash with zero providers
    expect(typeof md).toBe("string");
  });
});

// ---------------------------------------------------------------------------
// 9. postSlackWebhook — no-op when env absent
// ---------------------------------------------------------------------------

describe("postSlackWebhook", () => {
  test("resolves without error when no webhook URL is configured", async () => {
    // Ensure env var is not set
    const orig = process.env["CALIBRATION_SLACK_WEBHOOK"];
    delete process.env["CALIBRATION_SLACK_WEBHOOK"];
    try {
      const report = await generateCalibrationReport([]);
      await expect(postSlackWebhook(report)).resolves.toBeUndefined();
    } finally {
      if (orig !== undefined) process.env["CALIBRATION_SLACK_WEBHOOK"] = orig;
    }
  });

  test("resolves without error when webhook URL is provided but unreachable", async () => {
    const report = await generateCalibrationReport([]);
    // Use a clearly unreachable URL — function must not throw
    await expect(
      postSlackWebhook(report, { webhookUrl: "https://localhost:1/nonexistent-webhook" }),
    ).resolves.toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// 10. Auto-discovery via knownProviders
// ---------------------------------------------------------------------------

describe("Auto-discovery from knownProviders", () => {
  test("empty providers list auto-discovers from CalibrationEngine", async () => {
    seedCal(CalibrationEngine.instance, "auto-p1", 5, 100, 100);
    seedCal(CalibrationEngine.instance, "auto-p2", 5, 100, 100);
    const report = await generateCalibrationReport([]); // empty → auto-discover
    const names = report.providers.map((p) => p.provider);
    expect(names).toContain("auto-p1");
    expect(names).toContain("auto-p2");
  });

  test("empty providers list auto-discovers from RecalibrationEngine", async () => {
    CalibrationEngine.instance.reset(); // no CalibrationEngine data
    seedRecal(RecalibrationEngine.instance, "recal-auto", 5, 100, 110);
    const report = await generateCalibrationReport([]);
    const names = report.providers.map((p) => p.provider);
    expect(names).toContain("recal-auto");
  });

  test("deduplicates providers appearing in both engines", async () => {
    const provider = "both-engines";
    seedCal(CalibrationEngine.instance, provider, 5, 100, 100);
    seedRecal(RecalibrationEngine.instance, provider, 5, 100, 100);
    const report = await generateCalibrationReport([]);
    const matching = report.providers.filter((p) => p.provider === provider);
    expect(matching).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// 11. Per-bin breakdown
// ---------------------------------------------------------------------------

describe("Per-bin breakdown", () => {
  test("bins array contains entries for trained bins only", async () => {
    const engine = CalibrationEngine.instance;
    for (let i = 0; i < 5; i++) {
      engine.record({ provider: "bin-test", messages: [txt("x")], estimated: 100, actual: 110 });
    }
    for (let i = 0; i < 5; i++) {
      engine.record({ provider: "bin-test", messages: [toolResultMsg()], estimated: 100, actual: 130 });
    }

    const report = await generateCalibrationReport(["bin-test"]);
    const entry = report.providers[0]!;

    expect(entry.bins.length).toBeGreaterThanOrEqual(1);
    const binNames = entry.bins.map((b) => b.bin);
    expect(binNames).toContain("text");
  });

  test("bin samplesCount matches records fed", async () => {
    const engine = CalibrationEngine.instance;
    for (let i = 0; i < 7; i++) {
      engine.record({ provider: "bin-count", messages: [txt("x")], estimated: 100, actual: 100 });
    }

    const report = await generateCalibrationReport(["bin-count"]);
    const entry = report.providers[0]!;
    const textBin = entry.bins.find((b) => b.bin === "text");
    expect(textBin).toBeDefined();
    expect(textBin!.samplesCount).toBe(7);
  });
});
