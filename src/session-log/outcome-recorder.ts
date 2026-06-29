/**
 * Session Outcome Recorder & Fitness Signal Backpropagation.
 *
 * Captures post-session telemetry (test pass/fail, user satisfaction, cost
 * overage, latency SLA) and backpropagates fitness signals into genome
 * generations and compression tiers to close the learning loop.
 *
 * ### Storage
 *
 * Outcomes are persisted to `<cwd>/.ashlrcode/genome/evolution/session-outcomes.jsonl`
 * (configurable via `ASHLR_SESSION_OUTCOMES_PATH`). Each line is one
 * `SessionOutcomeRecord`. Reading is crash-safe: corrupt lines are skipped.
 *
 * ### Design
 *
 * | Outcome type    | Key fields                                   |
 * |-----------------|----------------------------------------------|
 * | test_outcome    | passed (bool)                                |
 * | latency_outcome | actual_ms vs. budget_ms                      |
 * | cost_outcome    | actual_usd vs. budget_usd, cache_savings_usd |
 *
 * Fitness signals are aggregated per (genomeVersion, provider) tuple so
 * A/B tests across genome versions are naturally deconvolved.
 */

import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { dirname } from "node:path";
import { loadManifest, saveManifest } from "../genome/manifest.ts";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** The three outcome categories tracked by this recorder. */
export type OutcomeType = "test_outcome" | "latency_outcome" | "cost_outcome";

/**
 * A single recorded session outcome.
 *
 * All outcome types share the context fields at the top; outcome-specific
 * payload lives in `meta`.
 */
export interface SessionOutcomeRecord {
  /** ISO-8601 UTC timestamp when the outcome was recorded. */
  recordedAt: string;
  /** Opaque session identifier — links back to session-log entries. */
  session_id: string;
  /** Provider slug (e.g. "claude-3-5-sonnet"). */
  provider: string;
  /** LLM model identifier. */
  model: string;
  /** Compression tier active for this session (1-4). */
  compression_tier: 1 | 2 | 3 | 4;
  /** Genome generation number used in this session. */
  genome_version: number;
  /** Actual USD cost incurred by this session. */
  cost_actual: number;
  /** Actual wall-clock latency of the session in milliseconds. */
  latency_actual_ms: number;
  /** Outcome category. */
  outcome_type: OutcomeType;
  /** Generic metric name (e.g. "test_pass_rate", "p95_latency_ms"). */
  fitness_metric_name: string;
  /** Metric value — semantics depend on `outcome_type`. */
  fitness_metric_value: number;
  /**
   * Outcome-specific payload.
   *
   * test_outcome:    { passed: boolean }
   * latency_outcome: { budget_ms: number; hit_sla: boolean }
   * cost_outcome:    { budget_usd: number; within_budget: boolean; cache_savings_usd: number }
   */
  meta: Record<string, unknown>;
}

/**
 * Base event fields supplied by the caller for every outcome type.
 * `outcome_type`, `fitness_metric_name`, and `fitness_metric_value` are
 * inferred or derived by the recorder's typed helpers.
 */
export interface OutcomeEventBase {
  session_id: string;
  provider: string;
  model: string;
  compression_tier: 1 | 2 | 3 | 4;
  genome_version: number;
  cost_actual: number;
  latency_actual_ms: number;
}

/** Extra fields required for a test outcome. */
export interface TestOutcomeParams extends OutcomeEventBase {
  passed: boolean;
}

/** Extra fields required for a latency outcome. */
export interface LatencyOutcomeParams extends OutcomeEventBase {
  budget_ms: number;
}

/** Extra fields required for a cost outcome. */
export interface CostOutcomeParams extends OutcomeEventBase {
  budget_usd: number;
  cache_savings_usd?: number;
}

/**
 * Aggregated fitness signal for a (genomeVersion, provider) pair.
 * All rates are in [0, 1].
 */
export interface FitnessSignal {
  genome_version: number;
  provider: string;
  /** Fraction of sessions where tests passed. */
  tests_passed_rate: number;
  /** Fraction of sessions where latency was within the SLA budget. */
  latency_compliance_rate: number;
  /** Fraction of sessions where cost was within budget. */
  cost_compliance_rate: number;
  /** Number of sessions contributing to this signal. */
  sample_count: number;
  /** ISO-8601 timestamp when this signal was computed. */
  computed_at: string;
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

const OUTCOMES_FILENAME = "session-outcomes.jsonl";

/**
 * Default outcomes file path relative to the given cwd.
 * Respects `ASHLR_SESSION_OUTCOMES_PATH` env override (absolute path).
 */
function resolveOutcomesPath(cwd: string): string {
  const override = process.env.ASHLR_SESSION_OUTCOMES_PATH;
  if (override && override.length > 0) return override;
  return join(cwd, ".ashlrcode", "genome", "evolution", OUTCOMES_FILENAME);
}

function ensureDir(filePath: string): void {
  const dir = dirname(filePath);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

function writesDisabled(): boolean {
  return process.env.ASHLR_SESSION_LOG === "0";
}

// ---------------------------------------------------------------------------
// SessionOutcomeRecorder class
// ---------------------------------------------------------------------------

/**
 * Captures post-session telemetry and writes it to the outcomes JSONL file.
 *
 * Usage:
 * ```ts
 * const recorder = new SessionOutcomeRecorder("/path/to/project");
 * recorder.recordTestOutcome({ session_id: "abc", provider: "claude-3-5-sonnet",
 *   model: "claude-3-5-sonnet-20241022", compression_tier: 2, genome_version: 3,
 *   cost_actual: 0.012, latency_actual_ms: 4200, passed: true });
 * ```
 */
export class SessionOutcomeRecorder {
  private readonly cwd: string;

  constructor(cwd: string) {
    this.cwd = cwd;
  }

  // ---------- Typed recording methods -------------------------------------

  /**
   * Record a test outcome (pass or fail).
   *
   * Stores `fitness_metric_value` as 1.0 (pass) or 0.0 (fail).
   */
  recordTestOutcome(params: TestOutcomeParams): SessionOutcomeRecord {
    const record: SessionOutcomeRecord = {
      recordedAt: new Date().toISOString(),
      session_id: params.session_id,
      provider: params.provider,
      model: params.model,
      compression_tier: params.compression_tier,
      genome_version: params.genome_version,
      cost_actual: params.cost_actual,
      latency_actual_ms: params.latency_actual_ms,
      outcome_type: "test_outcome",
      fitness_metric_name: "test_pass_rate",
      fitness_metric_value: params.passed ? 1.0 : 0.0,
      meta: { passed: params.passed },
    };
    this._persist(record);
    return record;
  }

  /**
   * Record a latency outcome (actual ms vs. SLA budget).
   *
   * Stores `fitness_metric_value` as the raw latency in ms; `meta.hit_sla`
   * indicates whether the SLA was met.
   */
  recordLatencyOutcome(params: LatencyOutcomeParams): SessionOutcomeRecord {
    const hit_sla = params.latency_actual_ms <= params.budget_ms;
    const record: SessionOutcomeRecord = {
      recordedAt: new Date().toISOString(),
      session_id: params.session_id,
      provider: params.provider,
      model: params.model,
      compression_tier: params.compression_tier,
      genome_version: params.genome_version,
      cost_actual: params.cost_actual,
      latency_actual_ms: params.latency_actual_ms,
      outcome_type: "latency_outcome",
      fitness_metric_name: "latency_actual_ms",
      fitness_metric_value: params.latency_actual_ms,
      meta: { budget_ms: params.budget_ms, hit_sla },
    };
    this._persist(record);
    return record;
  }

  /**
   * Record a cost outcome (actual USD vs. budget + cache savings realized).
   *
   * Stores `fitness_metric_value` as the actual cost in USD; `meta.within_budget`
   * indicates whether the cost stayed within the budget.
   */
  recordCostOutcome(params: CostOutcomeParams): SessionOutcomeRecord {
    const cache_savings_usd = params.cache_savings_usd ?? 0;
    const within_budget = params.cost_actual <= params.budget_usd;
    const record: SessionOutcomeRecord = {
      recordedAt: new Date().toISOString(),
      session_id: params.session_id,
      provider: params.provider,
      model: params.model,
      compression_tier: params.compression_tier,
      genome_version: params.genome_version,
      cost_actual: params.cost_actual,
      latency_actual_ms: params.latency_actual_ms,
      outcome_type: "cost_outcome",
      fitness_metric_name: "cost_actual_usd",
      fitness_metric_value: params.cost_actual,
      meta: { budget_usd: params.budget_usd, within_budget, cache_savings_usd },
    };
    this._persist(record);
    return record;
  }

  // ---------- Aggregation -------------------------------------------------

  /**
   * Read all outcomes from the JSONL file for this project.
   * Crash-safe: corrupt lines are silently skipped.
   */
  readOutcomes(): SessionOutcomeRecord[] {
    return readSessionOutcomes(this.cwd);
  }

  /**
   * Compute a `FitnessSignal` for a specific (genomeVersion, provider) pair
   * by aggregating all matching outcomes recorded so far.
   *
   * Returns a zero-valued signal (sample_count=0) when no data is available.
   */
  computeFitnessSignalForGeneration(
    genomeVersion: number,
    provider: string,
  ): FitnessSignal {
    return computeFitnessSignalForGeneration(this.cwd, genomeVersion, provider);
  }

  // ---------- Private helpers ---------------------------------------------

  private _persist(record: SessionOutcomeRecord): void {
    if (writesDisabled()) return;
    try {
      const path = resolveOutcomesPath(this.cwd);
      ensureDir(path);
      appendFileSync(path, JSON.stringify(record) + "\n", { flag: "a" });
    } catch {
      // Best-effort — never throw from a logging path.
    }
  }
}

// ---------------------------------------------------------------------------
// Free-function API (for callers that don't instantiate the class)
// ---------------------------------------------------------------------------

/**
 * Read all session outcome records from a project's outcomes file.
 * Crash-safe: corrupt lines are silently skipped.
 */
export function readSessionOutcomes(cwd: string): SessionOutcomeRecord[] {
  const path = resolveOutcomesPath(cwd);
  if (!existsSync(path)) return [];
  let raw: string;
  try {
    raw = readFileSync(path, "utf-8");
  } catch {
    return [];
  }
  const results: SessionOutcomeRecord[] = [];
  for (const line of raw.split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      results.push(JSON.parse(line) as SessionOutcomeRecord);
    } catch {
      // Skip corrupt lines
    }
  }
  return results;
}

/**
 * Compute a `FitnessSignal` for a specific (genomeVersion, provider) pair.
 *
 * Aggregates all outcomes recorded to the project's outcomes file and computes
 * the three compliance rates:
 *   - `tests_passed_rate`:        fraction of test_outcome sessions that passed
 *   - `latency_compliance_rate`:  fraction of latency_outcome sessions within SLA
 *   - `cost_compliance_rate`:     fraction of cost_outcome sessions within budget
 *
 * Returns a zero-valued signal when no matching outcomes exist.
 */
export function computeFitnessSignalForGeneration(
  cwd: string,
  genomeVersion: number,
  provider: string,
): FitnessSignal {
  const all = readSessionOutcomes(cwd);
  const matching = all.filter(
    (r) => r.genome_version === genomeVersion && r.provider === provider,
  );

  if (matching.length === 0) {
    return {
      genome_version: genomeVersion,
      provider,
      tests_passed_rate: 0,
      latency_compliance_rate: 0,
      cost_compliance_rate: 0,
      sample_count: 0,
      computed_at: new Date().toISOString(),
    };
  }

  // test_outcome: passed = meta.passed === true
  const testOutcomes = matching.filter((r) => r.outcome_type === "test_outcome");
  const testsPassed = testOutcomes.filter((r) => r.meta.passed === true).length;
  const tests_passed_rate =
    testOutcomes.length > 0 ? testsPassed / testOutcomes.length : 0;

  // latency_outcome: compliance = meta.hit_sla === true
  const latencyOutcomes = matching.filter((r) => r.outcome_type === "latency_outcome");
  const latencyPassed = latencyOutcomes.filter((r) => r.meta.hit_sla === true).length;
  const latency_compliance_rate =
    latencyOutcomes.length > 0 ? latencyPassed / latencyOutcomes.length : 0;

  // cost_outcome: compliance = meta.within_budget === true
  const costOutcomes = matching.filter((r) => r.outcome_type === "cost_outcome");
  const costPassed = costOutcomes.filter((r) => r.meta.within_budget === true).length;
  const cost_compliance_rate =
    costOutcomes.length > 0 ? costPassed / costOutcomes.length : 0;

  return {
    genome_version: genomeVersion,
    provider,
    tests_passed_rate,
    latency_compliance_rate,
    cost_compliance_rate,
    sample_count: matching.length,
    computed_at: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Genome manifest integration
// ---------------------------------------------------------------------------

/**
 * Update the genome manifest's fitness history with real outcome telemetry.
 *
 * Computes a `FitnessSignal` for `(genomeVersion, provider)` from persisted
 * outcomes and merges it into the manifest's `fitnessHistory` array entry for
 * that generation. If no manifest entry exists for the generation, one is
 * created.
 *
 * Marks earlier generations as stale by setting a `stale: true` flag on all
 * `fitnessHistory` entries whose generation number is less than `genomeVersion`
 * and whose outcome_signal was last updated before this call.
 *
 * @param cwd           Project root (must contain `.ashlrcode/genome/manifest.json`).
 * @param genomeVersion Generation number to update.
 * @param fitnessSignal Pre-computed signal; if omitted, computed from disk.
 */
export async function updateGenomeFitness(
  cwd: string,
  genomeVersion: number,
  fitnessSignal?: FitnessSignal,
): Promise<void> {
  const manifest = await loadManifest(cwd);
  if (!manifest) return; // No genome — nothing to update.

  const signal =
    fitnessSignal ??
    computeFitnessSignalForGeneration(cwd, genomeVersion, "unknown");

  const now = new Date().toISOString();

  // Find or create the fitnessHistory entry for this generation.
  const existing = manifest.fitnessHistory.find(
    (h) => h.generation === genomeVersion,
  );
  if (existing) {
    // Merge outcome telemetry into the existing scores record.
    existing.scores["outcome_tests_passed_rate"] = signal.tests_passed_rate;
    existing.scores["outcome_latency_compliance_rate"] = signal.latency_compliance_rate;
    existing.scores["outcome_cost_compliance_rate"] = signal.cost_compliance_rate;
    existing.scores["outcome_sample_count"] = signal.sample_count;
    existing.scores["outcome_updated_at_ms"] = Date.now();
  } else {
    manifest.fitnessHistory.push({
      generation: genomeVersion,
      scores: {
        outcome_tests_passed_rate: signal.tests_passed_rate,
        outcome_latency_compliance_rate: signal.latency_compliance_rate,
        outcome_cost_compliance_rate: signal.cost_compliance_rate,
        outcome_sample_count: signal.sample_count,
        outcome_updated_at_ms: Date.now(),
      },
    });
  }

  // Mark all older generation entries as stale relative to the current update.
  for (const h of manifest.fitnessHistory) {
    if (h.generation < genomeVersion) {
      // Only mark stale if they haven't been updated after our current signal.
      const updatedMs = typeof h.scores["outcome_updated_at_ms"] === "number"
        ? (h.scores["outcome_updated_at_ms"] as number)
        : 0;
      if (updatedMs < Date.now()) {
        h.scores["stale"] = 1; // 1 = stale, 0 = fresh (bool encoded as number for Record<string,number>)
        h.scores["stale_since_ms"] = Date.now();
      }
    }
  }

  manifest.updatedAt = now;
  await saveManifest(cwd, manifest);
}

// ---------------------------------------------------------------------------
// Test hooks
// ---------------------------------------------------------------------------

/** Exposed for tests: read the raw JSONL path for a given cwd. */
export function _resolveOutcomesPath(cwd: string): string {
  return resolveOutcomesPath(cwd);
}
