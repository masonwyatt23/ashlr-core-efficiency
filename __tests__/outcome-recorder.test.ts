/**
 * Tests for SessionOutcomeRecorder & Fitness Signal Backpropagation.
 *
 * Covers:
 *   1. Outcome recording — test_outcome, latency_outcome, cost_outcome
 *   2. JSONL persistence to disk (env-controlled path)
 *   3. Signal aggregation — computeFitnessSignalForGeneration
 *   4. Multi-provider deconvolution (v1/claude vs v1/openai stay separate)
 *   5. SLA compliance detection (latency hit/miss, cost within/over budget)
 *   6. Fitness delta computation (before/after outcome recording)
 *   7. Stale generation marking via updateGenomeFitness
 *   8. Write suppression when ASHLR_SESSION_LOG=0
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  SessionOutcomeRecorder,
  readSessionOutcomes,
  computeFitnessSignalForGeneration,
  updateGenomeFitness,
  _resolveOutcomesPath,
  type SessionOutcomeRecord,
  type FitnessSignal,
} from "../src/session-log/outcome-recorder.ts";

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

function makeTmpDir(): string {
  const dir = join(tmpdir(), `ashlr-outcome-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(dir, { recursive: true });
  return dir;
}

/** Minimal genome manifest so updateGenomeFitness can load+save it. */
function createMinimalManifest(cwd: string, generation = 3): void {
  const dir = join(cwd, ".ashlrcode", "genome");
  mkdirSync(dir, { recursive: true });
  const manifest = {
    version: 1,
    schemaVersion: 3,
    project: "test-project",
    sections: [],
    generation: { number: generation, milestone: "test", startedAt: new Date().toISOString() },
    fitnessHistory: [
      { generation: 1, scores: { testsPassRate: 0.5 } },
      { generation: 2, scores: { testsPassRate: 0.7 } },
    ],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  writeFileSync(join(dir, "manifest.json"), JSON.stringify(manifest, null, 2), "utf-8");
}

/** Base outcome params shared by most tests. */
const BASE: Omit<SessionOutcomeRecord, "recordedAt" | "outcome_type" | "fitness_metric_name" | "fitness_metric_value" | "meta"> = {
  session_id: "sess-abc",
  provider: "claude-3-5-sonnet",
  model: "claude-3-5-sonnet-20241022",
  compression_tier: 2,
  genome_version: 3,
  cost_actual: 0.012,
  latency_actual_ms: 3_200,
};

let tmpCwd: string;

beforeEach(() => {
  tmpCwd = makeTmpDir();
  // Redirect outcomes file into our temp dir so tests don't pollute real projects.
  process.env.ASHLR_SESSION_OUTCOMES_PATH = join(tmpCwd, "session-outcomes.jsonl");
  delete process.env.ASHLR_SESSION_LOG;
});

afterEach(() => {
  delete process.env.ASHLR_SESSION_OUTCOMES_PATH;
  delete process.env.ASHLR_SESSION_LOG;
  try { rmSync(tmpCwd, { recursive: true, force: true }); } catch { /* ignore */ }
});

// ---------------------------------------------------------------------------
// 1. Outcome recording
// ---------------------------------------------------------------------------

describe("outcome recording — test_outcome", () => {
  test("recordTestOutcome returns a well-formed record for pass", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    const rec = recorder.recordTestOutcome({ ...BASE, passed: true });
    expect(rec.outcome_type).toBe("test_outcome");
    expect(rec.fitness_metric_name).toBe("test_pass_rate");
    expect(rec.fitness_metric_value).toBe(1.0);
    expect(rec.meta.passed).toBe(true);
    expect(rec.session_id).toBe("sess-abc");
    expect(new Date(rec.recordedAt).getTime()).not.toBeNaN();
  });

  test("recordTestOutcome returns fitness_metric_value=0 for fail", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    const rec = recorder.recordTestOutcome({ ...BASE, passed: false });
    expect(rec.fitness_metric_value).toBe(0.0);
    expect(rec.meta.passed).toBe(false);
  });

  test("recordLatencyOutcome captures actual_ms, budget_ms, hit_sla=true", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    const rec = recorder.recordLatencyOutcome({ ...BASE, budget_ms: 5_000 });
    expect(rec.outcome_type).toBe("latency_outcome");
    expect(rec.fitness_metric_name).toBe("latency_actual_ms");
    expect(rec.fitness_metric_value).toBe(3_200);
    expect(rec.meta.budget_ms).toBe(5_000);
    expect(rec.meta.hit_sla).toBe(true); // 3200 <= 5000
  });

  test("recordLatencyOutcome sets hit_sla=false when over budget", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    const rec = recorder.recordLatencyOutcome({ ...BASE, budget_ms: 2_000 });
    expect(rec.meta.hit_sla).toBe(false); // 3200 > 2000
  });

  test("recordCostOutcome captures actual_usd, budget_usd, within_budget, cache_savings", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    const rec = recorder.recordCostOutcome({
      ...BASE,
      budget_usd: 0.02,
      cache_savings_usd: 0.005,
    });
    expect(rec.outcome_type).toBe("cost_outcome");
    expect(rec.fitness_metric_name).toBe("cost_actual_usd");
    expect(rec.fitness_metric_value).toBe(0.012);
    expect(rec.meta.budget_usd).toBe(0.02);
    expect(rec.meta.within_budget).toBe(true); // 0.012 <= 0.02
    expect(rec.meta.cache_savings_usd).toBe(0.005);
  });

  test("recordCostOutcome sets within_budget=false when over budget", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    const rec = recorder.recordCostOutcome({ ...BASE, budget_usd: 0.005 });
    expect(rec.meta.within_budget).toBe(false); // 0.012 > 0.005
  });
});

// ---------------------------------------------------------------------------
// 2. JSONL persistence
// ---------------------------------------------------------------------------

describe("JSONL persistence", () => {
  test("recorded outcomes appear in the JSONL file", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    recorder.recordTestOutcome({ ...BASE, passed: true });
    recorder.recordLatencyOutcome({ ...BASE, budget_ms: 5_000 });
    recorder.recordCostOutcome({ ...BASE, budget_usd: 0.02 });

    const outcomesPath = process.env.ASHLR_SESSION_OUTCOMES_PATH!;
    const lines = readFileSync(outcomesPath, "utf-8").trim().split("\n");
    expect(lines.length).toBe(3);
    const parsed = lines.map((l) => JSON.parse(l) as SessionOutcomeRecord);
    expect(parsed[0]!.outcome_type).toBe("test_outcome");
    expect(parsed[1]!.outcome_type).toBe("latency_outcome");
    expect(parsed[2]!.outcome_type).toBe("cost_outcome");
  });

  test("readSessionOutcomes returns all persisted records", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    recorder.recordTestOutcome({ ...BASE, passed: true });
    recorder.recordTestOutcome({ ...BASE, passed: false });
    const records = readSessionOutcomes(tmpCwd);
    expect(records.length).toBe(2);
  });

  test("readSessionOutcomes returns [] when file does not exist", () => {
    // No outcomes written yet
    delete process.env.ASHLR_SESSION_OUTCOMES_PATH;
    const records = readSessionOutcomes(tmpCwd);
    expect(records.length).toBe(0);
  });

  test("readSessionOutcomes skips corrupt JSONL lines gracefully", () => {
    const outcomesPath = process.env.ASHLR_SESSION_OUTCOMES_PATH!;
    const dir = outcomesPath.substring(0, outcomesPath.lastIndexOf("/"));
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      outcomesPath,
      '{"outcome_type":"test_outcome","session_id":"s1","provider":"p","model":"m","compression_tier":1,"genome_version":1,"cost_actual":0,"latency_actual_ms":0,"fitness_metric_name":"x","fitness_metric_value":1,"meta":{"passed":true},"recordedAt":"2026-01-01T00:00:00.000Z"}\nCORRUPT LINE\n',
      "utf-8",
    );
    const records = readSessionOutcomes(tmpCwd);
    expect(records.length).toBe(1);
    expect(records[0]!.session_id).toBe("s1");
  });

  test("writes are suppressed when ASHLR_SESSION_LOG=0", () => {
    process.env.ASHLR_SESSION_LOG = "0";
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    recorder.recordTestOutcome({ ...BASE, passed: true });

    const outcomesPath = process.env.ASHLR_SESSION_OUTCOMES_PATH!;
    let content = "";
    try { content = readFileSync(outcomesPath, "utf-8"); } catch { /* file not created */ }
    expect(content).toBe("");
  });
});

// ---------------------------------------------------------------------------
// 3. Signal aggregation
// ---------------------------------------------------------------------------

describe("computeFitnessSignalForGeneration — aggregation", () => {
  test("returns zero signal when no outcomes are recorded", () => {
    const signal = computeFitnessSignalForGeneration(tmpCwd, 3, "claude-3-5-sonnet");
    expect(signal.sample_count).toBe(0);
    expect(signal.tests_passed_rate).toBe(0);
    expect(signal.latency_compliance_rate).toBe(0);
    expect(signal.cost_compliance_rate).toBe(0);
    expect(signal.genome_version).toBe(3);
    expect(signal.provider).toBe("claude-3-5-sonnet");
    expect(new Date(signal.computed_at).getTime()).not.toBeNaN();
  });

  test("aggregates test outcomes correctly", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    // 3 pass, 1 fail → 0.75
    for (let i = 0; i < 3; i++) recorder.recordTestOutcome({ ...BASE, passed: true });
    recorder.recordTestOutcome({ ...BASE, passed: false });

    const signal = computeFitnessSignalForGeneration(tmpCwd, 3, "claude-3-5-sonnet");
    expect(signal.tests_passed_rate).toBeCloseTo(0.75, 5);
    expect(signal.sample_count).toBe(4);
  });

  test("aggregates latency compliance correctly", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    // 2 within SLA (budget=5000 > 3200), 2 over (budget=2000 < 3200)
    recorder.recordLatencyOutcome({ ...BASE, budget_ms: 5_000 });
    recorder.recordLatencyOutcome({ ...BASE, budget_ms: 5_000 });
    recorder.recordLatencyOutcome({ ...BASE, budget_ms: 2_000 });
    recorder.recordLatencyOutcome({ ...BASE, budget_ms: 2_000 });

    const signal = computeFitnessSignalForGeneration(tmpCwd, 3, "claude-3-5-sonnet");
    expect(signal.latency_compliance_rate).toBeCloseTo(0.5, 5);
  });

  test("aggregates cost compliance correctly", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    // 3 within budget (budget=0.02 > 0.012), 1 over (budget=0.005 < 0.012)
    for (let i = 0; i < 3; i++) recorder.recordCostOutcome({ ...BASE, budget_usd: 0.02 });
    recorder.recordCostOutcome({ ...BASE, budget_usd: 0.005 });

    const signal = computeFitnessSignalForGeneration(tmpCwd, 3, "claude-3-5-sonnet");
    expect(signal.cost_compliance_rate).toBeCloseTo(0.75, 5);
  });

  test("mixed outcome types all contribute to sample_count", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    recorder.recordTestOutcome({ ...BASE, passed: true });
    recorder.recordLatencyOutcome({ ...BASE, budget_ms: 5_000 });
    recorder.recordCostOutcome({ ...BASE, budget_usd: 0.02 });

    const signal = computeFitnessSignalForGeneration(tmpCwd, 3, "claude-3-5-sonnet");
    expect(signal.sample_count).toBe(3);
    expect(signal.tests_passed_rate).toBe(1.0);
    expect(signal.latency_compliance_rate).toBe(1.0);
    expect(signal.cost_compliance_rate).toBe(1.0);
  });
});

// ---------------------------------------------------------------------------
// 4. Multi-provider deconvolution
// ---------------------------------------------------------------------------

describe("multi-provider deconvolution", () => {
  test("signals are isolated by provider", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    // Provider A: all pass
    for (let i = 0; i < 4; i++) {
      recorder.recordTestOutcome({ ...BASE, provider: "claude-3-5-sonnet", passed: true });
    }
    // Provider B: all fail
    for (let i = 0; i < 4; i++) {
      recorder.recordTestOutcome({ ...BASE, provider: "claude-3-opus", passed: false });
    }

    const signalA = computeFitnessSignalForGeneration(tmpCwd, 3, "claude-3-5-sonnet");
    const signalB = computeFitnessSignalForGeneration(tmpCwd, 3, "claude-3-opus");

    expect(signalA.tests_passed_rate).toBe(1.0);
    expect(signalB.tests_passed_rate).toBe(0.0);
    expect(signalA.sample_count).toBe(4);
    expect(signalB.sample_count).toBe(4);
  });

  test("signals are isolated by genome_version", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    // v1: all pass
    for (let i = 0; i < 3; i++) {
      recorder.recordTestOutcome({ ...BASE, genome_version: 1, passed: true });
    }
    // v2: all fail
    for (let i = 0; i < 3; i++) {
      recorder.recordTestOutcome({ ...BASE, genome_version: 2, passed: false });
    }

    const signalV1 = computeFitnessSignalForGeneration(tmpCwd, 1, "claude-3-5-sonnet");
    const signalV2 = computeFitnessSignalForGeneration(tmpCwd, 2, "claude-3-5-sonnet");

    expect(signalV1.tests_passed_rate).toBe(1.0);
    expect(signalV2.tests_passed_rate).toBe(0.0);
  });

  test("unknown provider returns zero signal when no outcomes for that provider", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    recorder.recordTestOutcome({ ...BASE, provider: "claude-3-5-sonnet", passed: true });

    const signal = computeFitnessSignalForGeneration(tmpCwd, 3, "openai-gpt-4o");
    expect(signal.sample_count).toBe(0);
    expect(signal.tests_passed_rate).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// 5. SLA compliance detection
// ---------------------------------------------------------------------------

describe("SLA compliance detection", () => {
  test("latency exactly at budget is compliant (<=)", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    const rec = recorder.recordLatencyOutcome({ ...BASE, latency_actual_ms: 5_000, budget_ms: 5_000 });
    expect(rec.meta.hit_sla).toBe(true);
  });

  test("latency 1ms over budget is non-compliant", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    const rec = recorder.recordLatencyOutcome({ ...BASE, latency_actual_ms: 5_001, budget_ms: 5_000 });
    expect(rec.meta.hit_sla).toBe(false);
  });

  test("cost exactly at budget is within_budget", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    const rec = recorder.recordCostOutcome({ ...BASE, cost_actual: 0.01, budget_usd: 0.01 });
    expect(rec.meta.within_budget).toBe(true);
  });

  test("cost 1 cent over budget is not within_budget", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    const rec = recorder.recordCostOutcome({ ...BASE, cost_actual: 0.011, budget_usd: 0.01 });
    expect(rec.meta.within_budget).toBe(false);
  });

  test("cache_savings_usd defaults to 0 when not provided", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    const rec = recorder.recordCostOutcome({ ...BASE, budget_usd: 0.02 });
    expect(rec.meta.cache_savings_usd).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// 6. Fitness delta computation
// ---------------------------------------------------------------------------

describe("fitness delta computation", () => {
  test("test pass rate increases from 0 to 1 after recording all-pass outcomes", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    const before = computeFitnessSignalForGeneration(tmpCwd, 3, "claude-3-5-sonnet");
    expect(before.tests_passed_rate).toBe(0);

    for (let i = 0; i < 5; i++) recorder.recordTestOutcome({ ...BASE, passed: true });

    const after = computeFitnessSignalForGeneration(tmpCwd, 3, "claude-3-5-sonnet");
    expect(after.tests_passed_rate).toBe(1.0);
    expect(after.tests_passed_rate - before.tests_passed_rate).toBeCloseTo(1.0, 5);
  });

  test("cost compliance drops when budget violations are recorded", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    // Start with all within budget
    for (let i = 0; i < 4; i++) recorder.recordCostOutcome({ ...BASE, budget_usd: 0.02 });
    const before = computeFitnessSignalForGeneration(tmpCwd, 3, "claude-3-5-sonnet");
    expect(before.cost_compliance_rate).toBe(1.0);

    // Add 4 over-budget sessions — compliance drops to 0.5
    for (let i = 0; i < 4; i++) recorder.recordCostOutcome({ ...BASE, budget_usd: 0.005 });
    const after = computeFitnessSignalForGeneration(tmpCwd, 3, "claude-3-5-sonnet");
    expect(after.cost_compliance_rate).toBeCloseTo(0.5, 5);
    expect(after.cost_compliance_rate).toBeLessThan(before.cost_compliance_rate);
  });

  test("sample_count grows with each recorded outcome", () => {
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    for (let i = 1; i <= 5; i++) {
      recorder.recordTestOutcome({ ...BASE, passed: true });
      const signal = computeFitnessSignalForGeneration(tmpCwd, 3, "claude-3-5-sonnet");
      expect(signal.sample_count).toBe(i);
    }
  });
});

// ---------------------------------------------------------------------------
// 7. Stale generation marking via updateGenomeFitness
// ---------------------------------------------------------------------------

describe("updateGenomeFitness — stale generation marking", () => {
  test("updates fitness history for the current generation", async () => {
    createMinimalManifest(tmpCwd, 3);
    const recorder = new SessionOutcomeRecorder(tmpCwd);
    recorder.recordTestOutcome({ ...BASE, passed: true });
    recorder.recordCostOutcome({ ...BASE, budget_usd: 0.02 });
    recorder.recordLatencyOutcome({ ...BASE, budget_ms: 5_000 });

    const signal = computeFitnessSignalForGeneration(tmpCwd, 3, "claude-3-5-sonnet");
    await updateGenomeFitness(tmpCwd, 3, signal);

    // Read manifest back
    const raw = readFileSync(join(tmpCwd, ".ashlrcode", "genome", "manifest.json"), "utf-8");
    const manifest = JSON.parse(raw);
    const entry = manifest.fitnessHistory.find((h: { generation: number }) => h.generation === 3);
    expect(entry).toBeDefined();
    expect(entry.scores.outcome_tests_passed_rate).toBeCloseTo(1.0, 5);
    expect(entry.scores.outcome_cost_compliance_rate).toBeCloseTo(1.0, 5);
    expect(entry.scores.outcome_latency_compliance_rate).toBeCloseTo(1.0, 5);
    expect(entry.scores.outcome_sample_count).toBe(3);
  });

  test("marks older generations as stale after update", async () => {
    createMinimalManifest(tmpCwd, 3);
    const signal: FitnessSignal = {
      genome_version: 3,
      provider: "claude-3-5-sonnet",
      tests_passed_rate: 0.9,
      latency_compliance_rate: 0.8,
      cost_compliance_rate: 0.85,
      sample_count: 10,
      computed_at: new Date().toISOString(),
    };
    await updateGenomeFitness(tmpCwd, 3, signal);

    const raw = readFileSync(join(tmpCwd, ".ashlrcode", "genome", "manifest.json"), "utf-8");
    const manifest = JSON.parse(raw);

    // Generations 1 and 2 should be marked stale (encoded as 1 since scores is Record<string,number>)
    const gen1 = manifest.fitnessHistory.find((h: { generation: number }) => h.generation === 1);
    const gen2 = manifest.fitnessHistory.find((h: { generation: number }) => h.generation === 2);
    expect(gen1?.scores?.stale).toBe(1);
    expect(gen2?.scores?.stale).toBe(1);
  });

  test("creates a new fitnessHistory entry when none exists for the generation", async () => {
    createMinimalManifest(tmpCwd, 5); // manifest has gens 1,2 but generation.number=5
    const signal: FitnessSignal = {
      genome_version: 5,
      provider: "claude-3-5-sonnet",
      tests_passed_rate: 0.95,
      latency_compliance_rate: 0.9,
      cost_compliance_rate: 0.88,
      sample_count: 7,
      computed_at: new Date().toISOString(),
    };
    await updateGenomeFitness(tmpCwd, 5, signal);

    const raw = readFileSync(join(tmpCwd, ".ashlrcode", "genome", "manifest.json"), "utf-8");
    const manifest = JSON.parse(raw);
    const entry = manifest.fitnessHistory.find((h: { generation: number }) => h.generation === 5);
    expect(entry).toBeDefined();
    expect(entry.scores.outcome_sample_count).toBe(7);
  });

  test("updateGenomeFitness is a no-op when no manifest exists", async () => {
    // No manifest in tmpCwd — should not throw
    await expect(updateGenomeFitness(tmpCwd, 1)).resolves.toBeUndefined();
  });
});
