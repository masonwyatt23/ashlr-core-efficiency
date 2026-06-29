/**
 * Adaptive Tier Selection via Sliding-Window Cost Regret Minimization.
 *
 * Replaces ad-hoc tier tuning with a principled bandit-style optimization loop:
 *
 * 1. **Cost tracking** — for each call, record `cost(selected_tier)` vs the
 *    counterfactual `cost(best_tier_in_hindsight)` in a rolling 50-call window.
 *
 * 2. **Regret aggregation** — accumulate regret per tier as an Exponential
 *    Moving Average (EMA) so recent outcomes are weighted more heavily than
 *    stale history.  Regret = cost(selected) − cost(best_in_hindsight).
 *
 * 3. **UCB exploration** — balance exploitation (pick the tier with the lowest
 *    historical cost) and exploration (occasionally try less-used tiers to keep
 *    estimates fresh).  Upper-Confidence-Bound:
 *      score(t) = avgCost(t) − c × sqrt(ln(N) / n_t)
 *    where N = total pulls, n_t = pulls for tier t, c = exploration constant.
 *
 * 4. **TierCostAnalysis** — emit a structured report with per-tier regret,
 *    95% confidence intervals, and human-readable recommendations.
 *
 * 5. **Auto-shift** — when a tier's EMA regret exceeds REGRET_THRESHOLD (10%
 *    of ideal cost) the learner recommends shifting away from that tier.
 *
 * ### Integration with cost-accounting
 *
 * Call `recordTierOutcome()` after every compression operation with the
 * provider-specific USD costs.  The cost-accounting module pipes data in via
 * `pipeCompressionCostToLearner()`, which is wired into `recordCompressionCost`.
 */

import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { CompressionTier } from "./context.ts";
import { defaultProviderRate } from "../tokens/index.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Rolling window size for cost/regret tracking. */
export const WINDOW_SIZE = 50;

/**
 * EMA smoothing factor α ∈ (0, 1].
 * Higher α reacts faster to recent outcomes; lower α smooths noise.
 * 0.15 ≈ a 12-call half-life.
 */
export const EMA_ALPHA = 0.15;

/**
 * UCB exploration constant c.
 * Larger → more exploration; smaller → more exploitation.
 * 0.5 is a conservative default (minimize wasted LLM tier-1 calls).
 */
export const UCB_C = 0.5;

/**
 * Regret threshold as a fraction of ideal cost.
 * When EMA regret for the selected tier exceeds this fraction of the cheapest
 * known tier cost, the learner recommends a tier shift.
 * 0.10 = 10% overspend above ideal.
 */
export const REGRET_THRESHOLD = 0.10;

/** Minimum number of observations per tier before UCB kicks in. */
const MIN_UCB_SAMPLES = 3;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * A single observed outcome recorded after one compression operation.
 */
export interface TierOutcome {
  /** Which tier was actually used. */
  selectedTier: CompressionTier;
  /** USD cost of the selected tier for this call. */
  selectedCost: number;
  /** USD cost of the cheapest tier in hindsight for this call. */
  bestCost: number;
  /**
   * Regret for this call: selectedCost − bestCost.
   * Always ≥ 0 (selected cost is never less than the best).
   */
  regret: number;
  /** Provider string for this call. */
  provider: string;
  /** ISO timestamp. */
  recordedAt: string;
}

/**
 * Per-tier aggregated statistics computed from the rolling window.
 */
export interface TierStats {
  tier: CompressionTier;
  /** Number of times this tier was selected (within the window). */
  pullCount: number;
  /** Average USD cost per call for this tier (from window). */
  avgCost: number;
  /** EMA of regret for calls where this tier was selected. */
  emaRegret: number;
  /**
   * 95% confidence interval half-width for `avgCost`.
   * CI = ±1.96 × stddev / sqrt(n).  Zero when n < 2.
   */
  ciHalfWidth: number;
  /**
   * UCB score (lower is better — this is a cost minimization problem).
   * score(t) = avgCost(t) − UCB_C × sqrt(ln(N) / n_t)
   * Unexplored tiers get a very low score to force exploration.
   */
  ucbScore: number;
  /**
   * True when this tier's EMA regret exceeds REGRET_THRESHOLD × idealCost.
   */
  exceedsRegretThreshold: boolean;
}

/**
 * Full cost-analysis report emitted by `computeTierCostAnalysis`.
 */
export interface TierCostAnalysis {
  /** Per-tier stats. */
  byTier: Record<CompressionTier, TierStats>;
  /**
   * The tier recommended by UCB for the *next* call.
   * This is the tier with the lowest ucbScore (minimum expected cost).
   */
  recommendedTier: CompressionTier;
  /**
   * True when at least one tier's EMA regret exceeds the threshold,
   * suggesting a tier shift would reduce costs.
   */
  shiftRecommended: boolean;
  /**
   * Human-readable recommendation string.
   * E.g. "Shift from tier 2 to tier 3 — EMA regret 12.3% above ideal."
   */
  recommendation: string;
  /** Number of outcomes in the current window. */
  windowSize: number;
  /** ISO timestamp of when this analysis was computed. */
  analyzedAt: string;
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

/** Rolling window of outcomes. Capped at WINDOW_SIZE. */
const _window: TierOutcome[] = [];

/** Per-tier EMA regret (updated incrementally). */
const _emaRegret: Record<CompressionTier, number> = { 1: 0, 2: 0, 3: 0, 4: 0 };

/** Per-tier all-time pull count (for UCB denominator). Not windowed. */
const _totalPulls: Record<CompressionTier, number> = { 1: 0, 2: 0, 3: 0, 4: 0 };

/** Total outcomes recorded (for UCB ln(N)). Not windowed. */
let _totalOutcomes = 0;

// ---------------------------------------------------------------------------
// Test hooks
// ---------------------------------------------------------------------------

/** Reset all learner state. For tests only. */
export function _resetLearner(): void {
  _window.length = 0;
  _emaRegret[1] = 0;
  _emaRegret[2] = 0;
  _emaRegret[3] = 0;
  _emaRegret[4] = 0;
  _totalPulls[1] = 0;
  _totalPulls[2] = 0;
  _totalPulls[3] = 0;
  _totalPulls[4] = 0;
  _totalOutcomes = 0;
}

/** Expose window for tests (read-only snapshot). */
export function _getWindow(): readonly TierOutcome[] {
  return [..._window];
}

/** Expose EMA regret snapshot for tests. */
export function _getEmaRegret(): Readonly<Record<CompressionTier, number>> {
  return { ..._emaRegret } as Readonly<Record<CompressionTier, number>>;
}

// ---------------------------------------------------------------------------
// Cost helpers
// ---------------------------------------------------------------------------

/**
 * Compute the USD cost of applying a given tier for a given token count.
 *
 * Cost model (mirrors cost-accounting.ts):
 *   Tier 1 (autoCompact)   : tokensRemoved × 0.1 × inputRate  (cache-read savings)
 *                            + summary LLM overhead
 *   Tier 2 (snipCompact)   : tokensRemoved × inputRate
 *   Tier 3 (contextCollapse): tokensRemoved × inputRate
 *
 * For regret minimization purposes we want the *expense* of using a tier,
 * not the savings. We define cost as the "effective spend" — how much we pay
 * per token after applying this tier.  For tier 1 this includes the LLM
 * overhead; for tiers 2/3 it is simply the per-token input rate.
 *
 * @param tier          Compression tier.
 * @param tokensRemoved Number of tokens the tier would remove.
 * @param provider      Provider/model string for pricing.
 * @returns             USD cost estimate for this tier application.
 */
export function computeTierCost(
  tier: CompressionTier,
  tokensRemoved: number,
  provider: string,
): number {
  const rate = defaultProviderRate(provider);
  switch (tier) {
    case 1: {
      // autoCompact: pay for the LLM summarization call.
      // Overhead: read tokensRemoved at cache-read rate + ~200 output tokens.
      const summaryInputCost = tokensRemoved * rate.inputRate * 0.1;
      const summaryOutputCost = 200 * rate.outputRate;
      return summaryInputCost + summaryOutputCost;
    }
    case 2:
    case 3: {
      // snipCompact / contextCollapse: no LLM call; cost is just the
      // per-token input rate for the tokens we're removing from future sends.
      // We treat this as zero direct cost but a non-zero regret signal when
      // the tier is heavier than necessary (e.g. using tier 1 when tier 3 suffices).
      // Return a small proxy cost proportional to aggressiveness so regret math works.
      return tokensRemoved * rate.inputRate * (tier === 2 ? 0.01 : 0.005);
    }
    case 4: {
      // treeCompact: no LLM call; prunes entire subtrees.
      // Proxy cost is lower than tier 3 (more selective, less aggressive per-token),
      // reflecting that it removes whole branches rather than per-message truncation.
      return tokensRemoved * rate.inputRate * 0.003;
    }
  }
}

/**
 * Identify the cheapest tier in hindsight for a given set of tier costs.
 *
 * @param costs Per-tier cost map.
 * @returns     The tier with the minimum cost.
 */
export function bestTierInHindsight(
  costs: Record<CompressionTier, number>,
): CompressionTier {
  let best: CompressionTier = 4;
  for (const t of [1, 2, 3, 4] as CompressionTier[]) {
    if (costs[t] < costs[best]) best = t;
  }
  return best;
}

// ---------------------------------------------------------------------------
// Core recording
// ---------------------------------------------------------------------------

/**
 * Record one tier outcome into the sliding window and update EMA regret.
 *
 * Called after every compression operation. Pipe provider-specific cost data
 * from `cost-accounting.ts` via `pipeCompressionCostToLearner`.
 *
 * @param selectedTier  Tier that was actually applied.
 * @param tokensRemoved Tokens removed by the operation.
 * @param provider      Provider string for pricing lookup.
 */
export function recordTierOutcome(
  selectedTier: CompressionTier,
  tokensRemoved: number,
  provider: string,
): TierOutcome {
  // Compute cost for every tier at this token count.
  const allCosts: Record<CompressionTier, number> = {
    1: computeTierCost(1, tokensRemoved, provider),
    2: computeTierCost(2, tokensRemoved, provider),
    3: computeTierCost(3, tokensRemoved, provider),
    4: computeTierCost(4, tokensRemoved, provider),
  };

  const selectedCost = allCosts[selectedTier];
  const best = bestTierInHindsight(allCosts);
  const bestCost = allCosts[best];
  const regret = Math.max(0, selectedCost - bestCost);

  const outcome: TierOutcome = {
    selectedTier,
    selectedCost,
    bestCost,
    regret,
    provider,
    recordedAt: new Date().toISOString(),
  };

  // Append to window, trimming to WINDOW_SIZE.
  _window.push(outcome);
  if (_window.length > WINDOW_SIZE) {
    _window.shift();
  }

  // Update EMA regret for the selected tier.
  _emaRegret[selectedTier] =
    EMA_ALPHA * regret + (1 - EMA_ALPHA) * _emaRegret[selectedTier];

  // Update pull counts.
  _totalPulls[selectedTier]++;
  _totalOutcomes++;

  return outcome;
}

/**
 * Pipe data from `cost-accounting.recordCompressionCost` into the regret learner.
 *
 * This is the integration point: after recording a compression cost event,
 * call this function so the learner's window stays up to date without
 * duplicating cost computation logic.
 *
 * @param tier          Tier that was applied.
 * @param tokensRemoved Tokens removed.
 * @param provider      Provider string.
 */
export function pipeCompressionCostToLearner(
  tier: CompressionTier,
  tokensRemoved: number,
  provider: string,
): void {
  recordTierOutcome(tier, tokensRemoved, provider);
}

// ---------------------------------------------------------------------------
// UCB tier selection
// ---------------------------------------------------------------------------

/**
 * Select the recommended compression tier using Upper-Confidence-Bound (UCB).
 *
 * In a cost-minimization context we invert the standard UCB formula:
 *   score(t) = avgCost(t) − UCB_C × sqrt(ln(N + 1) / n_t)
 *
 * The −exploration term means unexplored tiers look artificially cheap, so
 * the algorithm will try them before settling on an exploit decision.
 *
 * Returns the tier with the lowest UCB score (minimum expected cost).
 * Falls back to tier 4 (treeCompact — lightest real compression) when no
 * data is available.  Tier 5 (validateAndMeasure) is intentionally excluded
 * from the UCB recommendation set: it performs zero compression and is
 * managed by the adaptive selector's fill-ratio fast-path, not by the bandit.
 *
 * @param windowOutcomes  Outcomes to base the decision on (defaults to internal window).
 * @returns               Recommended tier ∈ {1, 2, 3, 4} — always valid.
 */
export function selectTierUCB(
  windowOutcomes?: readonly TierOutcome[],
): CompressionTier {
  const outcomes = windowOutcomes ?? _window;

  // Empty window: default to tier 4 (lightest real compression tier).
  if (outcomes.length === 0) return 4;

  // Aggregate per-tier stats from outcomes across the four compression tiers.
  const counts: Record<CompressionTier, number> = { 1: 0, 2: 0, 3: 0, 4: 0 };
  const totalCosts: Record<CompressionTier, number> = { 1: 0, 2: 0, 3: 0, 4: 0 };

  for (const o of outcomes) {
    counts[o.selectedTier]++;
    totalCosts[o.selectedTier] += o.selectedCost;
  }

  const N = outcomes.length;

  // Phase 1: if any actionable tier (1–4) has fewer than MIN_UCB_SAMPLES,
  // explore it first.  Prefer lighter tiers (higher number) for exploration
  // order so we don't open with the most expensive tier.
  for (const t of [4, 3, 2, 1] as CompressionTier[]) {
    if (counts[t] < MIN_UCB_SAMPLES) {
      return t;
    }
  }

  // Phase 2: all tiers sufficiently explored — pick minimum UCB score.
  // Default to tier 4 so we always return a valid value even if the loop
  // produces no improvement over Infinity.
  let bestTier: CompressionTier = 4;
  let bestScore = Infinity;

  for (const t of [1, 2, 3, 4] as CompressionTier[]) {
    const n = counts[t];
    const avgCost = totalCosts[t] / n;
    const exploration = UCB_C * Math.sqrt(Math.log(N + 1) / n);
    const score = avgCost - exploration;

    if (score < bestScore) {
      bestScore = score;
      bestTier = t;
    }
  }

  return bestTier;
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

/**
 * Compute a full `TierCostAnalysis` report from the current window.
 *
 * This is the primary output surface: callers can emit this as a structured
 * log entry, surface it in a dashboard, or feed `recommendedTier` back into
 * `selectCompressionTierAdaptive`.
 *
 * @param windowOutcomes  Optional override for the outcomes window (for testing).
 * @returns               A `TierCostAnalysis` report.
 */
export function computeTierCostAnalysis(
  windowOutcomes?: readonly TierOutcome[],
): TierCostAnalysis {
  const outcomes = windowOutcomes ?? _window;
  const N = outcomes.length;

  // Bucket by selected tier.
  const buckets: Record<CompressionTier, TierOutcome[]> = { 1: [], 2: [], 3: [], 4: [] };
  for (const o of outcomes) {
    buckets[o.selectedTier].push(o);
  }

  // Compute the minimum ideal cost across the window (for regret threshold).
  const overallBestCosts = outcomes.map((o) => o.bestCost);
  const meanIdealCost =
    overallBestCosts.length > 0
      ? overallBestCosts.reduce((a, b) => a + b, 0) / overallBestCosts.length
      : 0;

  const byTier = {} as Record<CompressionTier, TierStats>;

  // Build stats for tiers 1–4 (active compression tiers tracked by the bandit).
  for (const t of [1, 2, 3, 4] as CompressionTier[]) {
    const bucket = buckets[t];
    const pullCount = bucket.length;

    // Average cost.
    const avgCost =
      pullCount > 0
        ? bucket.reduce((s, o) => s + o.selectedCost, 0) / pullCount
        : 0;

    // 95% CI half-width.
    let ciHalfWidth = 0;
    if (pullCount >= 2) {
      const mean = avgCost;
      const variance =
        bucket.reduce((s, o) => s + (o.selectedCost - mean) ** 2, 0) /
        (pullCount - 1);
      const stddev = Math.sqrt(variance);
      ciHalfWidth = (1.96 * stddev) / Math.sqrt(pullCount);
    }

    // UCB score for this tier.
    let ucbScore: number;
    if (pullCount < MIN_UCB_SAMPLES) {
      ucbScore = pullCount === 0 ? -Infinity : -1e6; // force exploration
    } else {
      const exploration = UCB_C * Math.sqrt(Math.log(N + 1) / pullCount);
      ucbScore = avgCost - exploration;
    }

    // EMA regret (from live state, not just window).
    const emaRegret = _emaRegret[t];
    // Only flag a tier as exceeding the threshold when it has been observed at
    // least MIN_UCB_SAMPLES times — unexplored tiers should never trigger a shift.
    const exceedsRegretThreshold =
      pullCount >= MIN_UCB_SAMPLES &&
      meanIdealCost > 0 &&
      emaRegret > REGRET_THRESHOLD * meanIdealCost;

    byTier[t] = {
      tier: t,
      pullCount,
      avgCost,
      emaRegret,
      ciHalfWidth,
      ucbScore,
      exceedsRegretThreshold,
    };
  }

  // Recommended tier — delegate entirely to selectTierUCB for consistency.
  // selectTierUCB always returns a value in {1,2,3,4}; never tier 5.
  const recommendedTier: CompressionTier = selectTierUCB(outcomes);

  // Build recommendation string.
  const shiftRecommended = ([1, 2, 3, 4] as CompressionTier[]).some(
    (t) => byTier[t].exceedsRegretThreshold,
  );

  let recommendation: string;
  if (N === 0) {
    recommendation = "Insufficient data — defaulting to tier 4 (lightest compression).";
  } else if (shiftRecommended) {
    const highRegretTiers = ([1, 2, 3, 4] as CompressionTier[]).filter(
      (t) => byTier[t].exceedsRegretThreshold,
    );
    const pct = (byTier[highRegretTiers[0]!].emaRegret / (meanIdealCost || 1)) * 100;
    recommendation = `Shift recommended — tier ${highRegretTiers[0]} EMA regret is ${pct.toFixed(1)}% above ideal. Recommended: tier ${recommendedTier}.`;
  } else {
    // byTier[recommendedTier] is always defined: selectTierUCB returns 1–4,
    // and byTier now covers all five tiers.
    const recStats = byTier[recommendedTier];
    recommendation = `No shift needed. Recommended tier: ${recommendedTier} (UCB score: ${recStats.ucbScore.toFixed(6)}).`;
  }

  return {
    byTier,
    recommendedTier,
    shiftRecommended,
    recommendation,
    windowSize: N,
    analyzedAt: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Convenience: get recommended tier from current learner state
// ---------------------------------------------------------------------------

/**
 * Convenience wrapper: return the UCB-recommended tier given current window state.
 * Always returns a valid tier ∈ {1,2,3,4}.  Tier 4 (treeCompact) is the safe
 * fallback when the window is empty or all tiers are under-explored.
 */
export function getRecommendedTier(): CompressionTier {
  return selectTierUCB(_window);
}

// ---------------------------------------------------------------------------
// Daily calibration report
// ---------------------------------------------------------------------------

/**
 * Shape of the compression bandit calibration report written to
 * `~/.ashlr/compression-bandit-report.json`.
 */
export interface CompressionBanditReport {
  /** ISO timestamp when the report was generated. */
  generatedAt: string;
  /** Number of outcomes in the current learner window. */
  windowSize: number;
  /** Recommended tier for the next compression call (always 1–4). */
  recommendedTier: CompressionTier;
  /** Per-tier summary statistics. */
  tierSummary: Array<{
    tier: CompressionTier;
    pullCount: number;
    avgCost: number;
    emaRegret: number;
    ucbScore: number;
    exceedsRegretThreshold: boolean;
    /** Estimated LLM tokens saved per call vs no compression (proxy metric). */
    estimatedTokensSavedPerCall: number;
  }>;
  /** True when at least one tier's EMA regret exceeds the threshold. */
  shiftRecommended: boolean;
  /** Human-readable recommendation string from the cost analysis. */
  recommendation: string;
  /** Total regret accumulated across the current window (lower is better). */
  totalWindowRegret: number;
  /** Average latency cost proxy: sum of all costs in window / windowSize. */
  avgCostPerCall: number;
}

/**
 * Emit a daily calibration report to `~/.ashlr/compression-bandit-report.json`.
 *
 * The report captures:
 * - LLM tokens saved vs latency cost (per tier)
 * - UCB recommendations and regret thresholds
 * - Window statistics for visibility into the live bandit state
 *
 * Write is best-effort: I/O errors are swallowed so this never breaks the
 * compression pipeline.  Safe to call at any time; the file is overwritten
 * (not appended) so the report always reflects the latest learner state.
 *
 * @param outputPath  Override the output path (default: `~/.ashlr/compression-bandit-report.json`).
 *                    Useful in tests.
 * @returns           The report object that was written (or null on error).
 */
export function emitDailyCalibrationReport(
  outputPath?: string,
): CompressionBanditReport | null {
  try {
    const analysis = computeTierCostAnalysis();
    const windowOutcomes = _getWindow();

    // Compute total window regret and average cost per call.
    let totalWindowRegret = 0;
    let totalWindowCost = 0;
    for (const o of windowOutcomes) {
      totalWindowRegret += o.regret;
      totalWindowCost += o.selectedCost;
    }
    const avgCostPerCall =
      windowOutcomes.length > 0 ? totalWindowCost / windowOutcomes.length : 0;

    // Build per-tier summary.  For each tier, estimate tokens saved per call
    // as a proxy: cost / inputRate ≈ tokens removed (using a neutral rate).
    const NEUTRAL_INPUT_RATE = 3e-6; // ~$3/M tokens — anthropic mid-tier
    const tierSummary = ([1, 2, 3, 4] as CompressionTier[]).map((t) => {
      const stats = analysis.byTier[t];
      const estimatedTokensSavedPerCall =
        stats.avgCost > 0 ? Math.round(stats.avgCost / NEUTRAL_INPUT_RATE) : 0;
      return {
        tier: t,
        pullCount: stats.pullCount,
        avgCost: stats.avgCost,
        emaRegret: stats.emaRegret,
        ucbScore: stats.ucbScore,
        exceedsRegretThreshold: stats.exceedsRegretThreshold,
        estimatedTokensSavedPerCall,
      };
    });

    const report: CompressionBanditReport = {
      generatedAt: new Date().toISOString(),
      windowSize: analysis.windowSize,
      recommendedTier: analysis.recommendedTier,
      tierSummary,
      shiftRecommended: analysis.shiftRecommended,
      recommendation: analysis.recommendation,
      totalWindowRegret,
      avgCostPerCall,
    };

    const filePath = outputPath ?? join(homedir(), ".ashlr", "compression-bandit-report.json");
    const dir = filePath.substring(0, filePath.lastIndexOf("/"));
    if (dir && !existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }
    writeFileSync(filePath, JSON.stringify(report, null, 2) + "\n", { flag: "w" });

    return report;
  } catch {
    // Best-effort — never throw from a reporting path.
    return null;
  }
}
