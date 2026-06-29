/**
 * Adaptive Budget Rebalancer with Multi-Objective Cost/Latency Tradeoff Learner.
 *
 * Optimizes system-prompt budget allocation across compression tiers, cache
 * breakpoints, and reserved response tokens, accounting for both USD cost and
 * end-user latency.
 *
 * ### Problem statement
 *
 * The static `systemPromptBudget()` assigns 5% of context (capped at 50K)
 * regardless of provider, model, message count, or observed latency. This
 * one-size-fits-all rule leaves savings on the table:
 *
 * - A heavily cached model (e.g. claude-haiku) can tolerate a larger system
 *   prompt because cache reads are 10× cheaper than input tokens.
 * - A high-latency provider benefits from more aggressive compression
 *   (smaller prompts) even if that costs slightly more in USD.
 * - Conversely, a cheap provider (groq, deepseek) should relax compression
 *   because the USD cost of extra tokens is negligible.
 *
 * ### Algorithm
 *
 * For each historical session the learner computes a set of candidate
 * BudgetAllocation objects that vary:
 *   - `systemPromptRatio` (0.02–0.15)
 *   - `cacheWriteBreakpointFraction` (0.0–1.0)
 *   - `reserveTokensForResponse` (1024–16384)
 *
 * Each candidate is evaluated against two objectives:
 *   - `total_cost_usd`  (minimize)
 *   - `p99_latency_ms`  (minimize)
 *
 * Non-dominated candidates form the Pareto frontier. The learner fits a
 * weighted recommendation by applying the caller's `targetLatencyMs` /
 * `maxCostDelta` constraints to select the best point on the frontier.
 *
 * ### Persistence
 *
 * Calibration records are written to `evolution/budget-rebalance.jsonl` so
 * fleet operators can audit decisions. Quarterly recalibration is supported
 * via `triggerRecalibration()`.
 *
 * ### Public API
 *
 * ```ts
 * const learner = new BudgetMultiObjectiveLearner(cwd);
 * await learner.ingest(costRecords, cacheRecords, compressionRecords);
 *
 * const allocation = await learner.getRobustBudgetAllocation(
 *   "anthropic", "claude-sonnet-4", { targetLatencyMs: 800 }
 * );
 * const diag = learner.validateBudgetFeasibility(allocation, "anthropic");
 * ```
 */

import { join } from "path";
import { appendJsonl, readJsonl } from "../genome/jsonl.ts";
import {
  getModelContextLimit,
  getProviderContextLimit,
  SYSTEM_PROMPT_BUDGET_RATIO,
  SYSTEM_PROMPT_BUDGET_CAP,
  PROVIDER_CONTEXT_LIMITS,
} from "./index.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GENOME_DIR = ".ashlrcode/genome";
const BUDGET_REBALANCE_FILE = "budget-rebalance.jsonl";

/** Candidate ratio values to sweep during Pareto enumeration. */
const RATIO_CANDIDATES = [0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15];

/** Candidate cache-write breakpoint fractions. */
const CACHE_BREAKPOINT_CANDIDATES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0];

/** Candidate reserve-token values. */
const RESERVE_CANDIDATES = [1024, 2048, 4096, 8192, 16384];

/** Minimum sessions before the learner trusts its own output. */
const MIN_SESSIONS_FOR_LEARNING = 5;

/** How many Pareto-front points to store per (provider, model, bucket). */
const MAX_PARETO_POINTS = 20;

// ---------------------------------------------------------------------------
// Message-count bucketing
// ---------------------------------------------------------------------------

type MessageCountBucket = "xs" | "sm" | "md" | "lg";

function bucketMessageCount(count: number): MessageCountBucket {
  if (count <= 10) return "xs";
  if (count <= 50) return "sm";
  if (count <= 200) return "md";
  return "lg";
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/**
 * A fully-specified budget allocation for one (provider, model) configuration.
 *
 * All three parameters are optimized jointly by the learner; none is held fixed
 * while the others vary.
 */
export interface BudgetAllocation {
  /** Fraction of the provider context limit to reserve for the system prompt. */
  systemPromptRatio: number;
  /**
   * Absolute system-prompt token budget derived from `systemPromptRatio`.
   * Capped at SYSTEM_PROMPT_BUDGET_CAP.
   */
  systemPromptTokens: number;
  /**
   * Fraction of the remaining context at which to write a cache checkpoint.
   * 0.0 = never write; 1.0 = always write at the first opportunity.
   * Passed to the cache-optimizer layer.
   */
  cacheWriteBreakpointFraction: number;
  /**
   * Tokens to hold in reserve for the model's response.
   * Ensures the model never runs out of output budget.
   */
  reserveTokensForResponse: number;
  /**
   * Estimated total USD cost per session under this allocation.
   * Derived from the historical cost model.
   */
  estimatedCostUsd: number;
  /**
   * Estimated p99 latency in milliseconds under this allocation.
   * Derived from the historical latency model.
   */
  estimatedP99LatencyMs: number;
  /** Provider this allocation was computed for. */
  provider: string;
  /** Model ID this allocation was computed for. */
  modelId: string;
  /** ISO timestamp when this allocation was last computed. */
  computedAt: string;
}

/**
 * A single point on the Pareto frontier of (cost, latency) tradeoffs.
 */
export interface ParetoPoint {
  /** Budget allocation parameters at this point. */
  allocation: Omit<BudgetAllocation, "provider" | "modelId" | "computedAt">;
  /** Estimated USD cost at this allocation. */
  costUsd: number;
  /** Estimated p99 latency (ms) at this allocation. */
  p99LatencyMs: number;
  /**
   * Whether this point is on the strict Pareto frontier (non-dominated).
   * A point is dominated if another point is strictly better on ALL objectives.
   */
  isPareto: boolean;
}

/**
 * Feasibility diagnostics for a proposed budget allocation.
 */
export interface BudgetDiagnostics {
  /** Whether the allocation is feasible for the given provider. */
  feasible: boolean;
  /**
   * List of issues found. Empty when `feasible` is true.
   * Each string is a human-readable description of a violated constraint.
   */
  issues: string[];
  /**
   * Utilization fraction: systemPromptTokens / providerContextLimit.
   * Values > 0.50 are warned; values > 0.80 are flagged as infeasible.
   */
  systemPromptUtilization: number;
  /**
   * Remaining tokens for messages + response after subtracting systemPrompt
   * and reserveTokensForResponse from the provider context limit.
   * Negative means the allocation is infeasible.
   */
  remainingMessageBudget: number;
  /**
   * Suggested fallback allocation when `feasible` is false.
   * Uses the static defaults as a safe floor.
   */
  fallback?: Pick<
    BudgetAllocation,
    "systemPromptRatio" | "systemPromptTokens" | "reserveTokensForResponse"
  >;
}

/**
 * Historical session summary ingested by the learner.
 *
 * Callers construct these from cost-accounting records and compression
 * feedback by aggregating per-session totals.
 */
export interface SessionSummary {
  /** Provider used in this session. */
  provider: string;
  /** Model ID used in this session (may be empty string for provider-level data). */
  modelId: string;
  /** Number of messages in the session's context window at peak. */
  messageCount: number;
  /** Total USD cost for the session (input + output + cache). */
  totalCostUsd: number;
  /** Observed p99 latency for API calls in this session (ms). */
  p99LatencyMs: number;
  /** System prompt token budget that was actually used. */
  systemPromptTokensUsed: number;
  /** Cache write breakpoint fraction that was configured for the session. */
  cacheWriteBreakpointFraction: number;
  /** Reserve tokens that were configured for the session. */
  reserveTokensForResponse: number;
  /** ISO timestamp of the session. */
  recordedAt: string;
}

/**
 * A persisted calibration record appended to budget-rebalance.jsonl.
 */
export interface BudgetCalibrationRecord {
  /** ISO timestamp of this calibration run. */
  calibratedAt: string;
  /** Provider this record covers. */
  provider: string;
  /** Model ID (empty for provider-level). */
  modelId: string;
  /** Message count bucket. */
  bucket: MessageCountBucket;
  /** Number of sessions used to compute this calibration. */
  sessionCount: number;
  /** Pareto-front points derived from sessions. */
  paretoFront: ParetoPoint[];
  /** The recommended allocation selected from the Pareto front. */
  recommended: Omit<BudgetAllocation, "provider" | "modelId" | "computedAt">;
}

/**
 * Options for `getRobustBudgetAllocation`.
 */
export interface BudgetAllocationOptions {
  /**
   * Target p99 latency in milliseconds.
   * When provided, the learner selects the cheapest Pareto point whose
   * estimated latency is ≤ targetLatencyMs.
   */
  targetLatencyMs?: number;
  /**
   * Maximum allowable cost increase relative to the static baseline (USD).
   * When provided, the learner rejects allocations that exceed
   * staticBaselineCost + maxCostDelta.
   */
  maxCostDelta?: number;
  /**
   * Message count to use for bucket selection.
   * Defaults to "sm" (11–50 messages) when not provided.
   */
  messageCount?: number;
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

function budgetRebalancePath(cwd: string): string {
  return join(cwd, GENOME_DIR, "evolution", BUDGET_REBALANCE_FILE);
}

// ---------------------------------------------------------------------------
// Cost model helpers
// ---------------------------------------------------------------------------

/**
 * Estimate the USD cost of one session under a given budget allocation.
 *
 * Model:
 *   - System prompt tokens are billed on every turn (input rate).
 *   - Cache-write tokens are billed at cacheWriteRate when breakpoint fraction > 0.
 *   - Reserve tokens are "output" tokens (billed at output rate).
 *   - A larger systemPromptRatio → more tokens on every call.
 *   - A larger cacheWriteBreakpoint → more cache writes (higher upfront cost,
 *     lower re-read cost across subsequent turns).
 *
 * This is intentionally a *relative* model: absolute USD values are not the
 * goal; what matters is the shape of the cost surface so the Pareto filter
 * can rank candidates consistently.
 */
function estimateCostUsd(
  allocation: Pick<
    BudgetAllocation,
    | "systemPromptTokens"
    | "cacheWriteBreakpointFraction"
    | "reserveTokensForResponse"
  >,
  provider: string,
  sessionCostBaseline: number,
  contextLimit: number,
): number {
  // Pricing lookup (USD per million tokens). Use conservative defaults.
  const pricing: Record<string, { input: number; output: number; cacheWrite: number; cacheRead: number }> = {
    anthropic:  { input: 3.0,   output: 15.0,  cacheWrite: 3.75,  cacheRead: 0.30  },
    openai:     { input: 2.5,   output: 10.0,  cacheWrite: 0.0,   cacheRead: 1.25  },
    xai:        { input: 5.0,   output: 15.0,  cacheWrite: 0.0,   cacheRead: 0.25  },
    groq:       { input: 0.27,  output: 0.27,  cacheWrite: 0.0,   cacheRead: 0.027 },
    deepseek:   { input: 0.14,  output: 0.28,  cacheWrite: 0.0,   cacheRead: 0.014 },
    ollama:     { input: 0.0,   output: 0.0,   cacheWrite: 0.0,   cacheRead: 0.0   },
  };

  const lower = provider.toLowerCase();
  let p = pricing["anthropic"]!;
  for (const [key, val] of Object.entries(pricing)) {
    if (lower.includes(key)) { p = val; break; }
  }

  const perMToken = 1_000_000;
  const sysCost = (allocation.systemPromptTokens / perMToken) * p.input;
  const cacheCost =
    (allocation.cacheWriteBreakpointFraction * contextLimit * 0.1) / perMToken * p.cacheWrite;
  const reserveCost = (allocation.reserveTokensForResponse / perMToken) * p.output;

  // Scale against the observed session baseline so the learner anchors to real data.
  const parameterCost = sysCost + cacheCost + reserveCost;
  return Math.max(0, sessionCostBaseline * 0.8 + parameterCost);
}

/**
 * Estimate p99 latency (ms) under a given budget allocation.
 *
 * Latency model:
 *   - Larger system prompts → more tokens to process → higher TTFT.
 *   - Larger cache-write breakpoint fraction → periodic cache-write overhead
 *     spikes but subsequent reads are faster (net effect: reduces average
 *     latency if cache hit rate is high).
 *   - Larger reserve → nothing (output budget doesn't affect TTFT).
 *
 * The model uses a simple linear regression approximation anchored to
 * the observed baseline latency.
 */
function estimateP99LatencyMs(
  allocation: Pick<
    BudgetAllocation,
    "systemPromptTokens" | "cacheWriteBreakpointFraction"
  >,
  baselineLatencyMs: number,
  contextLimit: number,
): number {
  // System prompt size increases TTFT roughly linearly.
  // Empirical: ~0.005ms per token for hosted models (varies widely).
  const sysLatencyDelta = allocation.systemPromptTokens * 0.005;

  // Cache writes add a one-time overhead (~50ms at breakpoint fraction 1.0).
  // Subsequent reads reduce latency by ~30% of baseline per 10% fill.
  const cacheNetDelta =
    allocation.cacheWriteBreakpointFraction * 50 -
    allocation.cacheWriteBreakpointFraction * contextLimit * 0.00001;

  const adjusted = baselineLatencyMs + sysLatencyDelta + cacheNetDelta;
  return Math.max(100, adjusted); // floor at 100ms
}

// ---------------------------------------------------------------------------
// Pareto frontier computation
// ---------------------------------------------------------------------------

/**
 * Determine whether point A dominates point B.
 * A dominates B iff A is at least as good on all objectives and strictly better
 * on at least one. (Cost and latency are both minimized.)
 */
function dominates(a: ParetoPoint, b: ParetoPoint): boolean {
  return (
    a.costUsd <= b.costUsd &&
    a.p99LatencyMs <= b.p99LatencyMs &&
    (a.costUsd < b.costUsd || a.p99LatencyMs < b.p99LatencyMs)
  );
}

/**
 * Compute the Pareto frontier from a list of candidate points.
 *
 * Marks each point's `isPareto` flag. Returns the full list with flags set.
 */
export function computeParetoFrontier(points: ParetoPoint[]): ParetoPoint[] {
  const result = points.map((p) => ({ ...p, isPareto: true }));

  for (let i = 0; i < result.length; i++) {
    for (let j = 0; j < result.length; j++) {
      if (i === j) continue;
      if (dominates(result[j]!, result[i]!)) {
        result[i]!.isPareto = false;
        break;
      }
    }
  }

  return result;
}

/**
 * Enumerate candidate budget allocations for a (provider, contextLimit, baseline) tuple.
 *
 * Sweeps the parameter grid and evaluates each point against the cost and
 * latency models. Returns all candidates with Pareto flags computed.
 */
function enumerateCandidates(
  provider: string,
  contextLimit: number,
  sessionCostBaseline: number,
  baselineLatencyMs: number,
): ParetoPoint[] {
  const candidates: ParetoPoint[] = [];

  for (const ratio of RATIO_CANDIDATES) {
    const systemPromptTokens = Math.min(
      Math.floor(contextLimit * ratio),
      SYSTEM_PROMPT_BUDGET_CAP,
    );

    for (const cacheBreakpoint of CACHE_BREAKPOINT_CANDIDATES) {
      for (const reserve of RESERVE_CANDIDATES) {
        const alloc = {
          systemPromptRatio: ratio,
          systemPromptTokens,
          cacheWriteBreakpointFraction: cacheBreakpoint,
          reserveTokensForResponse: reserve,
        };

        const costUsd = estimateCostUsd(
          alloc,
          provider,
          sessionCostBaseline,
          contextLimit,
        );
        const p99LatencyMs = estimateP99LatencyMs(alloc, baselineLatencyMs, contextLimit);

        candidates.push({
          allocation: {
            ...alloc,
            estimatedCostUsd: costUsd,
            estimatedP99LatencyMs: p99LatencyMs,
          },
          costUsd,
          p99LatencyMs,
          isPareto: false, // will be computed next
        });
      }
    }
  }

  return computeParetoFrontier(candidates);
}

/**
 * Select the best point from the Pareto front given constraints.
 *
 * Selection rules (applied in priority order):
 *   1. Exclude points that violate `targetLatencyMs` (if provided).
 *   2. Exclude points that violate `maxCostDelta` above the static baseline (if provided).
 *   3. Among remaining Pareto points, pick the one with the lowest cost.
 *   4. If no Pareto point satisfies the constraints, fall back to the static baseline.
 */
function selectFromFrontier(
  points: ParetoPoint[],
  provider: string,
  contextLimit: number,
  opts: BudgetAllocationOptions,
): ParetoPoint | null {
  const paretoPoints = points.filter((p) => p.isPareto);
  if (paretoPoints.length === 0) return null;

  // Compute static baseline cost for delta comparison.
  const staticTokens = Math.min(
    Math.floor(contextLimit * SYSTEM_PROMPT_BUDGET_RATIO),
    SYSTEM_PROMPT_BUDGET_CAP,
  );
  const staticBaseline = estimateCostUsd(
    {
      systemPromptTokens: staticTokens,
      cacheWriteBreakpointFraction: 0.3,
      reserveTokensForResponse: 8192,
    },
    provider,
    0,
    contextLimit,
  );

  let candidates = paretoPoints;

  if (opts.targetLatencyMs !== undefined) {
    const latencyFiltered = candidates.filter(
      (p) => p.p99LatencyMs <= opts.targetLatencyMs!,
    );
    if (latencyFiltered.length > 0) candidates = latencyFiltered;
    // If all violate the constraint, relax and return the lowest-latency point.
    else {
      const minLatency = Math.min(...candidates.map((p) => p.p99LatencyMs));
      candidates = candidates.filter((p) => p.p99LatencyMs === minLatency);
    }
  }

  if (opts.maxCostDelta !== undefined) {
    const costCap = staticBaseline + opts.maxCostDelta;
    const costFiltered = candidates.filter((p) => p.costUsd <= costCap);
    if (costFiltered.length > 0) candidates = costFiltered;
  }

  // Pick the lowest-cost point among remaining candidates.
  return candidates.reduce((best, p) =>
    p.costUsd < best.costUsd ? p : best,
  );
}

// ---------------------------------------------------------------------------
// Main learner class
// ---------------------------------------------------------------------------

/**
 * Adaptive Budget Rebalancer — learns optimal budget allocation parameters
 * from historical session data across providers, models, and message counts.
 *
 * ### Usage
 *
 * ```ts
 * const learner = new BudgetMultiObjectiveLearner("/path/to/project");
 * await learner.ingest(sessions);
 *
 * const allocation = await learner.getRobustBudgetAllocation("anthropic", "claude-sonnet-4");
 * const diag = learner.validateBudgetFeasibility(allocation, "anthropic");
 * ```
 *
 * ### Thread safety
 *
 * Not thread-safe. The learner holds in-memory session state. For concurrent
 * fleet operations construct one learner per process.
 */
export class BudgetMultiObjectiveLearner {
  private readonly cwd: string;
  private sessions: SessionSummary[] = [];
  private calibrations: Map<string, BudgetCalibrationRecord> = new Map();

  constructor(cwd: string) {
    this.cwd = cwd;
  }

  // -------------------------------------------------------------------------
  // Ingestion
  // -------------------------------------------------------------------------

  /**
   * Ingest historical session summaries for learning.
   *
   * Accepts records from cost-accounting, cache-optimizer, and compression-
   * adaptive layers. Callers should aggregate per-session totals before
   * calling ingest; the learner works at session granularity (not per-call).
   *
   * @param sessions  Session summaries to add to the learner's history.
   */
  ingest(sessions: SessionSummary[]): void {
    this.sessions.push(...sessions);
  }

  /**
   * Load historical calibration records from the persistence file.
   *
   * Called automatically by `getRobustBudgetAllocation` when needed.
   * Exposed publicly for explicit preloading.
   */
  async loadCalibrations(): Promise<void> {
    const records = await readJsonl<BudgetCalibrationRecord>(
      budgetRebalancePath(this.cwd),
    );
    for (const rec of records) {
      const key = this._cacheKey(rec.provider, rec.modelId, rec.bucket);
      // Newer records overwrite older ones (latest wins).
      this.calibrations.set(key, rec);
    }
  }

  // -------------------------------------------------------------------------
  // Core learning: fit budget parameters from sessions
  // -------------------------------------------------------------------------

  /**
   * Compute the Pareto frontier and fit budget parameters for a given
   * (provider, model, bucket) combination.
   *
   * Uses sessions that match the provider + model (with substring matching),
   * bucketed by message count. Falls back to provider-level sessions when
   * model-specific data is sparse.
   *
   * @returns `null` when there are not enough sessions to trust the result.
   */
  fitAllocation(
    provider: string,
    modelId: string,
    bucket: MessageCountBucket,
  ): BudgetCalibrationRecord | null {
    const matchingSessions = this._selectSessions(provider, modelId, bucket);
    if (matchingSessions.length < MIN_SESSIONS_FOR_LEARNING) return null;

    const contextLimit = modelId
      ? getModelContextLimit(modelId, provider)
      : getProviderContextLimit(provider);

    // Derive baseline statistics from the matched sessions.
    const avgCost =
      matchingSessions.reduce((s, r) => s + r.totalCostUsd, 0) / matchingSessions.length;
    const sortedLatencies = matchingSessions
      .map((r) => r.p99LatencyMs)
      .sort((a, b) => a - b);
    const p99Latency =
      sortedLatencies[Math.floor(sortedLatencies.length * 0.99)] ??
      sortedLatencies[sortedLatencies.length - 1] ??
      500;

    const allPoints = enumerateCandidates(provider, contextLimit, avgCost, p99Latency);

    // Keep top MAX_PARETO_POINTS Pareto-front candidates (sorted by cost ASC).
    const paretoPoints = allPoints
      .filter((p) => p.isPareto)
      .sort((a, b) => a.costUsd - b.costUsd)
      .slice(0, MAX_PARETO_POINTS);

    // Default recommended: best point with no constraints.
    const bestPoint = paretoPoints[0] ?? allPoints.sort((a, b) => a.costUsd - b.costUsd)[0]!;

    const record: BudgetCalibrationRecord = {
      calibratedAt: new Date().toISOString(),
      provider,
      modelId,
      bucket,
      sessionCount: matchingSessions.length,
      paretoFront: paretoPoints,
      recommended: bestPoint.allocation,
    };

    const key = this._cacheKey(provider, modelId, bucket);
    this.calibrations.set(key, record);
    return record;
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Return a robust budget allocation for the given (provider, model).
   *
   * Procedure:
   *   1. Load persisted calibrations from budget-rebalance.jsonl.
   *   2. If in-memory sessions allow fitting, compute a fresh allocation.
   *   3. Select the best Pareto-front point given the caller's constraints.
   *   4. Fall back to a static-derived allocation when data is insufficient.
   *
   * @param provider       Provider name (case-insensitive substring match).
   * @param modelId        Model ID (may be empty for provider-level query).
   * @param opts           Optional latency and cost constraints.
   * @returns              `BudgetAllocation` — guaranteed non-null.
   */
  async getRobustBudgetAllocation(
    provider: string,
    modelId: string,
    opts: BudgetAllocationOptions = {},
  ): Promise<BudgetAllocation> {
    // Always load persisted calibrations so the learner benefits from prior runs.
    await this.loadCalibrations();

    const bucket = bucketMessageCount(opts.messageCount ?? 20);

    // Try to fit a fresh allocation from in-memory sessions.
    const fitted = this.fitAllocation(provider, modelId, bucket);

    // Retrieve cached calibration (may have been loaded from disk or just fitted).
    const key = this._cacheKey(provider, modelId, bucket);
    const calibration = fitted ?? this.calibrations.get(key) ?? null;

    const contextLimit = modelId
      ? getModelContextLimit(modelId, provider)
      : getProviderContextLimit(provider);

    if (calibration && calibration.paretoFront.length > 0) {
      const selected = selectFromFrontier(
        calibration.paretoFront,
        provider,
        contextLimit,
        opts,
      );

      if (selected) {
        return {
          ...selected.allocation,
          provider,
          modelId,
          computedAt: new Date().toISOString(),
        };
      }
    }

    // Static fallback.
    return this._staticFallback(provider, modelId, contextLimit);
  }

  /**
   * Validate that a proposed budget allocation is feasible for the given provider.
   *
   * Checks:
   *   - systemPromptTokens + reserveTokensForResponse < contextLimit
   *   - systemPromptUtilization ≤ 0.80 (warn at 0.50)
   *   - reserveTokensForResponse ≥ 512
   *   - cacheWriteBreakpointFraction ∈ [0, 1]
   *
   * @param allocation  Allocation to validate.
   * @param provider    Provider to validate against.
   * @returns           `BudgetDiagnostics` with feasibility flag and issue list.
   */
  validateBudgetFeasibility(
    allocation: BudgetAllocation,
    provider: string,
  ): BudgetDiagnostics {
    const contextLimit = allocation.modelId
      ? getModelContextLimit(allocation.modelId, provider)
      : getProviderContextLimit(provider);

    const issues: string[] = [];
    const sysUtil = allocation.systemPromptTokens / contextLimit;

    if (sysUtil > 0.80) {
      issues.push(
        `systemPromptTokens (${allocation.systemPromptTokens}) exceeds 80% of context limit (${contextLimit}).`,
      );
    } else if (sysUtil > 0.50) {
      issues.push(
        `systemPromptTokens (${allocation.systemPromptTokens}) exceeds 50% of context limit — consider reducing ratio.`,
      );
    }

    const remaining =
      contextLimit - allocation.systemPromptTokens - allocation.reserveTokensForResponse;

    if (remaining < 0) {
      issues.push(
        `systemPromptTokens + reserveTokensForResponse exceeds context limit ` +
          `(${allocation.systemPromptTokens} + ${allocation.reserveTokensForResponse} > ${contextLimit}).`,
      );
    } else if (remaining < 4096) {
      issues.push(
        `Only ${remaining} tokens remain for messages after reserving system prompt and response budget.`,
      );
    }

    if (allocation.reserveTokensForResponse < 512) {
      issues.push(
        `reserveTokensForResponse (${allocation.reserveTokensForResponse}) is below the minimum recommended value of 512.`,
      );
    }

    if (
      allocation.cacheWriteBreakpointFraction < 0 ||
      allocation.cacheWriteBreakpointFraction > 1
    ) {
      issues.push(
        `cacheWriteBreakpointFraction (${allocation.cacheWriteBreakpointFraction}) must be in [0, 1].`,
      );
    }

    const fallback = issues.length > 0
      ? {
          systemPromptRatio: SYSTEM_PROMPT_BUDGET_RATIO,
          systemPromptTokens: Math.min(
            Math.floor(contextLimit * SYSTEM_PROMPT_BUDGET_RATIO),
            SYSTEM_PROMPT_BUDGET_CAP,
          ),
          reserveTokensForResponse: 8192,
        }
      : undefined;

    return {
      feasible: issues.length === 0,
      issues,
      systemPromptUtilization: sysUtil,
      remainingMessageBudget: remaining,
      fallback,
    };
  }

  // -------------------------------------------------------------------------
  // Recalibration
  // -------------------------------------------------------------------------

  /**
   * Persist the current calibration state to budget-rebalance.jsonl.
   *
   * Should be called after `ingest()` + `fitAllocation()` to make the
   * calibration durable across restarts. In fleet deployments, call this
   * quarterly (every ~90 days) or when significant workload changes occur.
   *
   * @param providers   Optional allow-list of providers to persist.
   *                    When omitted, all fitted calibrations are persisted.
   */
  async persistCalibrations(providers?: string[]): Promise<void> {
    for (const [, rec] of this.calibrations) {
      if (providers && !providers.includes(rec.provider)) continue;
      await appendJsonl(budgetRebalancePath(this.cwd), rec);
    }
  }

  /**
   * Re-fit all calibrations from the current in-memory session pool and
   * persist the results.
   *
   * Call this quarterly or when `triggerRecalibration()` is invoked explicitly.
   * Returns the number of (provider, model, bucket) combinations that were
   * successfully fitted.
   */
  async triggerRecalibration(): Promise<number> {
    let fitted = 0;
    const providerModels = new Set<string>();

    for (const s of this.sessions) {
      providerModels.add(`${s.provider}:::${s.modelId}`);
    }

    for (const combo of providerModels) {
      const [prov, model] = combo.split(":::") as [string, string];
      for (const bucket of ["xs", "sm", "md", "lg"] as MessageCountBucket[]) {
        const rec = this.fitAllocation(prov, model, bucket);
        if (rec) fitted++;
      }
    }

    await this.persistCalibrations();
    return fitted;
  }

  /**
   * Emit a Pareto-front visualization as a text table.
   *
   * Lists up to `limit` Pareto points for the given (provider, model, bucket),
   * sorted by ascending cost. Each row shows ratio, cache breakpoint, reserve
   * tokens, estimated cost, and estimated latency.
   *
   * @param provider     Provider to query.
   * @param modelId      Model ID to query.
   * @param bucket       Message count bucket.
   * @param limit        Maximum rows to emit (default: 10).
   * @returns            Multi-line string suitable for logging / dashboard display.
   */
  visualizeParetoFront(
    provider: string,
    modelId: string,
    bucket: MessageCountBucket = "sm",
    limit = 10,
  ): string {
    const key = this._cacheKey(provider, modelId, bucket);
    const calibration = this.calibrations.get(key);

    if (!calibration || calibration.paretoFront.length === 0) {
      return `No Pareto front available for (${provider}, ${modelId}, ${bucket}).`;
    }

    const rows = calibration.paretoFront.slice(0, limit);
    const lines: string[] = [
      `Pareto Front — ${provider} / ${modelId || "any"} / bucket:${bucket} (${rows.length} points)`,
      `${"ratio".padEnd(6)} ${"cacheBreak".padEnd(11)} ${"reserve".padEnd(9)} ${"costUsd".padEnd(10)} ${"p99ms".padEnd(8)} pareto`,
      "-".repeat(60),
    ];

    for (const p of rows) {
      const a = p.allocation;
      lines.push(
        `${String(a.systemPromptRatio.toFixed(2)).padEnd(6)} ` +
          `${String(a.cacheWriteBreakpointFraction.toFixed(1)).padEnd(11)} ` +
          `${String(a.reserveTokensForResponse).padEnd(9)} ` +
          `${String(p.costUsd.toFixed(6)).padEnd(10)} ` +
          `${String(p.p99LatencyMs.toFixed(0)).padEnd(8)} ` +
          `${p.isPareto ? "✓" : " "}`,
      );
    }

    return lines.join("\n");
  }

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  private _cacheKey(
    provider: string,
    modelId: string,
    bucket: MessageCountBucket,
  ): string {
    return `${provider.toLowerCase()}:::${modelId.toLowerCase()}:::${bucket}`;
  }

  private _selectSessions(
    provider: string,
    modelId: string,
    bucket: MessageCountBucket,
  ): SessionSummary[] {
    const lowerProvider = provider.toLowerCase();
    const lowerModel = modelId.toLowerCase();

    return this.sessions.filter((s) => {
      const providerMatch = s.provider.toLowerCase().includes(lowerProvider) ||
        lowerProvider.includes(s.provider.toLowerCase());
      const modelMatch =
        !modelId ||
        !s.modelId ||
        s.modelId.toLowerCase().includes(lowerModel) ||
        lowerModel.includes(s.modelId.toLowerCase());
      const bucketMatch = bucketMessageCount(s.messageCount) === bucket;
      return providerMatch && modelMatch && bucketMatch;
    });
  }

  private _staticFallback(
    provider: string,
    modelId: string,
    contextLimit: number,
  ): BudgetAllocation {
    const systemPromptTokens = Math.min(
      Math.floor(contextLimit * SYSTEM_PROMPT_BUDGET_RATIO),
      SYSTEM_PROMPT_BUDGET_CAP,
    );
    return {
      systemPromptRatio: SYSTEM_PROMPT_BUDGET_RATIO,
      systemPromptTokens,
      cacheWriteBreakpointFraction: 0.3,
      reserveTokensForResponse: 8192,
      estimatedCostUsd: 0,
      estimatedP99LatencyMs: 500,
      provider,
      modelId,
      computedAt: new Date().toISOString(),
    };
  }
}

// ---------------------------------------------------------------------------
// Module-level convenience exports
// ---------------------------------------------------------------------------

export { bucketMessageCount };
export type { MessageCountBucket };
