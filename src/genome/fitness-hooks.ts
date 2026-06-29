/**
 * Empirical fitness measurement via hook-based instrumentation.
 *
 * Agents and consumers call FitnessInstrument methods at key moments
 * (test runs, code changes, strategy attempts). Events are collected
 * in-memory and persisted to fitness-log.jsonl. measureFitnessEmpirical
 * reads the log and computes actual (not heuristic) fitness metrics.
 * FitnessCalibrator compares heuristic vs empirical and logs divergence.
 */

import { join } from "path";
import { appendJsonl, readJsonl } from "./jsonl.ts";
import type { FitnessMetrics } from "./fitness.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FITNESS_LOG_FILENAME = "fitness-log.jsonl";

function fitnessLogPath(cwd: string): string {
  return join(cwd, ".ashlrcode", "genome", "evolution", FITNESS_LOG_FILENAME);
}

// ---------------------------------------------------------------------------
// Event types
// ---------------------------------------------------------------------------

export type FitnessEventKind = "test_run" | "change" | "strategy";

export interface TestRunEvent {
  kind: "test_run";
  timestamp: string;
  passed: number;
  failed: number;
  durationMs: number;
}

export interface ChangeEvent {
  kind: "change";
  timestamp: string;
  filesChanged: number;
  linesAdded: number;
  linesRemoved: number;
}

export interface StrategyEvent {
  kind: "strategy";
  timestamp: string;
  category: string;
  outcome: "positive" | "negative" | "neutral";
}

export type FitnessEvent = TestRunEvent | ChangeEvent | StrategyEvent;

// ---------------------------------------------------------------------------
// Interface
// ---------------------------------------------------------------------------

/**
 * Interface that agents/consumers implement (or receive) to record fitness events.
 * All methods return void (fire-and-forget persistence); callers should await to
 * ensure the log is flushed before reading.
 */
export interface FitnessInstrument {
  /** Record the result of a test run. */
  recordTestRun(passed: number, failed: number, durationMs: number): Promise<void>;

  /** Record a code change (commit, edit session, etc.). */
  recordChange(filesChanged: number, linesAdded: number, linesRemoved: number): Promise<void>;

  /** Record the outcome of a strategy attempt. */
  recordStrategy(category: string, outcome: "positive" | "negative" | "neutral"): Promise<void>;
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

/**
 * Collects fitness events in-memory and persists each one to fitness-log.jsonl
 * as it is recorded. The in-memory buffer can be queried without disk I/O.
 */
export class InstrumentedFitness implements FitnessInstrument {
  private readonly cwd: string;
  private readonly events: FitnessEvent[] = [];

  constructor(cwd: string) {
    this.cwd = cwd;
  }

  async recordTestRun(passed: number, failed: number, durationMs: number): Promise<void> {
    const event: TestRunEvent = {
      kind: "test_run",
      timestamp: new Date().toISOString(),
      passed,
      failed,
      durationMs,
    };
    this.events.push(event);
    await appendJsonl(fitnessLogPath(this.cwd), event);
  }

  async recordChange(filesChanged: number, linesAdded: number, linesRemoved: number): Promise<void> {
    const event: ChangeEvent = {
      kind: "change",
      timestamp: new Date().toISOString(),
      filesChanged,
      linesAdded,
      linesRemoved,
    };
    this.events.push(event);
    await appendJsonl(fitnessLogPath(this.cwd), event);
  }

  async recordStrategy(category: string, outcome: "positive" | "negative" | "neutral"): Promise<void> {
    const event: StrategyEvent = {
      kind: "strategy",
      timestamp: new Date().toISOString(),
      category,
      outcome,
    };
    this.events.push(event);
    await appendJsonl(fitnessLogPath(this.cwd), event);
  }

  /** Return a snapshot of in-memory events (does not read from disk). */
  getEvents(): readonly FitnessEvent[] {
    return this.events;
  }
}

// ---------------------------------------------------------------------------
// Empirical measurement
// ---------------------------------------------------------------------------

/**
 * Read fitness-log.jsonl and compute empirical (not heuristic) fitness metrics.
 *
 * - testPassRate:          fraction of tests passing across all recorded runs
 * - codeChurn:             normalised 0-1 score (lower churn = higher fitness)
 * - strategyROI:           fraction of strategies with positive outcome
 *
 * Returns null metrics (0.5 fallback) when the log is absent or empty.
 */
export async function measureFitnessEmpirical(cwd: string): Promise<FitnessMetrics> {
  const events = await readJsonl<FitnessEvent>(fitnessLogPath(cwd));

  // Separate by kind
  const testRuns = events.filter((e): e is TestRunEvent => e.kind === "test_run");
  const changes = events.filter((e): e is ChangeEvent => e.kind === "change");
  const strategies = events.filter((e): e is StrategyEvent => e.kind === "strategy");

  // ── test pass rate ───────────────────────────────────────────────────────
  let testsPassRate: number;
  if (testRuns.length === 0) {
    testsPassRate = 0.5;
  } else {
    const totalPassed = testRuns.reduce((s, e) => s + e.passed, 0);
    const totalFailed = testRuns.reduce((s, e) => s + e.failed, 0);
    const total = totalPassed + totalFailed;
    testsPassRate = total === 0 ? 1.0 : totalPassed / total;
  }

  // ── code quality via churn ───────────────────────────────────────────────
  // High churn → low quality. Normalise: avg lines-changed per event.
  // Threshold: 200 lines/change = 0.0, 0 lines/change = 1.0.
  let codeQuality: number;
  if (changes.length === 0) {
    codeQuality = 0.5;
  } else {
    const avgChurn =
      changes.reduce((s, e) => s + e.linesAdded + e.linesRemoved, 0) / changes.length;
    const CHURN_MAX = 200;
    codeQuality = Math.max(0, Math.min(1, 1 - avgChurn / CHURN_MAX));
  }

  // ── milestone progress (not directly observable from hooks — use testsPassRate as proxy)
  const milestoneProgress = testsPassRate;

  // ── cost efficiency via duration ─────────────────────────────────────────
  // Faster test runs (within 10s) = more efficient.
  let costEfficiency: number;
  if (testRuns.length === 0) {
    costEfficiency = 0.5;
  } else {
    const avgDuration = testRuns.reduce((s, e) => s + e.durationMs, 0) / testRuns.length;
    const DURATION_MAX_MS = 60_000;
    costEfficiency = Math.max(0, Math.min(1, 1 - avgDuration / DURATION_MAX_MS));
  }

  // ── strategy ROI ─────────────────────────────────────────────────────────
  let strategySuccessRate: number;
  if (strategies.length === 0) {
    strategySuccessRate = 0.5;
  } else {
    const positive = strategies.filter((e) => e.outcome === "positive").length;
    strategySuccessRate = positive / strategies.length;
  }

  return {
    testsPassRate,
    codeQuality,
    milestoneProgress,
    costEfficiency,
    strategySuccessRate,
  };
}

// ---------------------------------------------------------------------------
// Calibrator
// ---------------------------------------------------------------------------

export interface CalibrationReport {
  cwd: string;
  timestamp: string;
  heuristic: FitnessMetrics;
  empirical: FitnessMetrics;
  /** Per-metric absolute divergence */
  divergence: Record<keyof FitnessMetrics, number>;
  /** Mean absolute divergence across all metrics */
  meanDivergence: number;
}

/**
 * Compare heuristic vs empirical fitness on the same cwd.
 * Logs the report to calibration-log.jsonl for tuning.
 */
export class FitnessCalibrator {
  private readonly cwd: string;

  constructor(cwd: string) {
    this.cwd = cwd;
  }

  async compare(heuristic: FitnessMetrics, empirical: FitnessMetrics): Promise<CalibrationReport> {
    const divergence = {} as Record<keyof FitnessMetrics, number>;
    let totalDiv = 0;
    let count = 0;

    for (const key of Object.keys(heuristic) as (keyof FitnessMetrics)[]) {
      const d = Math.abs(heuristic[key] - empirical[key]);
      divergence[key] = d;
      totalDiv += d;
      count++;
    }

    const meanDivergence = count > 0 ? totalDiv / count : 0;

    const report: CalibrationReport = {
      cwd: this.cwd,
      timestamp: new Date().toISOString(),
      heuristic,
      empirical,
      divergence,
      meanDivergence,
    };

    // Persist to calibration log for offline tuning
    const logPath = join(this.cwd, ".ashlrcode", "genome", "evolution", "calibration-log.jsonl");
    await appendJsonl(logPath, report);

    return report;
  }
}
