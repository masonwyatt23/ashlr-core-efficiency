/**
 * Cost Accounting & ROI Tracking for compression operations.
 *
 * Tracks real USD cost deltas for each compression tier so callers can
 * quantify the financial value of compression and drive data-driven tier
 * selection per workload.
 *
 * ### Tier cost model
 *
 * | Tier | Name          | LLM call? | Saved tokens billed at            |
 * |------|---------------|-----------|-----------------------------------|
 * |  1   | autoCompact   | Yes       | 0.1× input rate (cache-read rate) |
 * |  2   | snipCompact   | No        | 1.0× input rate                   |
 * |  3   | contextCollapse | No      | 1.0× input rate (free — no LLM)   |
 *
 * Tier 1 costs an extra LLM call; the removed tokens would otherwise have
 * been re-sent at full price every turn, but after autoCompact they are gone
 * and the summary is cheaper to cache-read. We model the ongoing savings as:
 *   savedCost = removedTokens × 0.1 × inputRate   (future reads at cache rate)
 *   minus the one-time LLM call cost.
 *
 * Tier 2/3 have no LLM overhead; every removed token is a direct saving:
 *   savedCost = removedTokens × inputRate
 *
 * ### Compression amnesia
 *
 * When tier 1 (autoCompact) produces a token estimation error beyond ±10%,
 * the entry is flagged for recalibration. Callers can query
 * `getAmnesiaFlags()` to find affected records and trigger re-tuning.
 */

import type { CompressionTier } from "../compression/context.ts";
import { PROVIDER_RATES, type ProviderName, defaultProviderRate } from "../tokens/index.ts";
import { pipeCompressionCostToLearner } from "../compression/regret-learner.ts";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/**
 * A single recorded compression cost event.
 */
export interface CompressionCostRecord {
  /** Wall-clock timestamp (ISO-8601). */
  recordedAt: string;
  /** Which compression tier was applied. */
  tier: CompressionTier;
  /** How many tokens were removed by the compression. */
  tokensRemoved: number;
  /** Time the compression operation took (milliseconds). */
  timeMs: number;
  /** Whether the compression call succeeded. */
  success: boolean;
  /** Provider whose pricing was used. */
  provider: ProviderName | string;
  /**
   * Net USD saved by this compression operation.
   *
   * Tier 1: removedTokens × 0.1 × inputRate − llmCallCost
   * Tier 2/3: removedTokens × inputRate
   */
  costSaved: number;
  /**
   * USD cost of the LLM call required for this tier, if any.
   * Zero for tiers 2 & 3.
   */
  llmCallCost: number;
  /**
   * True when tier === 1 AND the token estimation error exceeded ±10%.
   * Indicates the summary diverged significantly from the expectation.
   */
  amnesiaFlag: boolean;
  /**
   * Signed percentage estimation error for tier 1 calls:
   *   ((actualTokens − estimatedTokens) / estimatedTokens) × 100
   * Zero for tiers 2 & 3 (no LLM estimation involved).
   */
  estimationErrorPct: number;
}

/**
 * Per-tier breakdown inside a `SessionROI`.
 */
export interface TierROIBreakdown {
  /** Number of compression calls for this tier in the session. */
  callCount: number;
  /** Total tokens removed across all calls. */
  tokensRemoved: number;
  /** Total USD saved across all calls. */
  costSaved: number;
}

/**
 * Aggregated ROI summary for a compression session.
 *
 * Returned by `summarizeSessionROI()`.
 */
export interface SessionROI {
  /** Total tokens removed across all tiers and all calls. */
  totalTokensSaved: number;
  /** Total USD saved across all tiers and all calls. */
  totalCostSaved: number;
  /** Total wall-clock time spent on compression operations (ms). */
  compressionTimeMs: number;
  /** Per-tier breakdown. */
  tierBreakdown: Record<CompressionTier, TierROIBreakdown>;
  /** Number of tier-1 calls flagged for recalibration due to amnesia. */
  amnesiaCount: number;
  /** ISO-8601 timestamp when this summary was generated. */
  generatedAt: string;
}

// ---------------------------------------------------------------------------
// In-process record store
// ---------------------------------------------------------------------------

/** All compression cost records for the current process lifetime. */
const _records: CompressionCostRecord[] = [];

/** Expose records for tests (read-only snapshot). */
export function _getRecords(): readonly CompressionCostRecord[] {
  return _records;
}

/** Reset store — test hook only. */
export function _resetRecords(): void {
  _records.length = 0;
}

// ---------------------------------------------------------------------------
// Core recording function
// ---------------------------------------------------------------------------

/**
 * Record a compression operation and compute its USD cost delta.
 *
 * @param tier          Compression tier applied (1 = autoCompact, 2 = snip, 3 = collapse).
 * @param tokensRemoved Number of tokens eliminated by the compression.
 * @param timeMs        Wall-clock time the operation took in milliseconds.
 * @param success       Whether the operation succeeded.
 * @param provider      Provider name used for pricing lookup (e.g. `"claude-3-5-sonnet"`).
 * @param estimationErrorPct  For tier 1 only: signed % error between estimated and actual
 *                            tokens after the LLM summarisation call. Pass 0 for tiers 2/3.
 * @returns             The recorded `CompressionCostRecord`.
 */
export function recordCompressionCost(
  tier: 1 | 2 | 3,
  tokensRemoved: number,
  timeMs: number,
  success: boolean,
  provider: string,
  estimationErrorPct = 0,
): CompressionCostRecord {
  const rate = defaultProviderRate(provider);

  let costSaved: number;
  let llmCallCost: number;
  let amnesiaFlag = false;

  switch (tier) {
    case 1: {
      // autoCompact: ongoing token savings at cache-read rate (0.1×) minus LLM call cost.
      // LLM call cost is approximated as a minimum summary call:
      //   ~200 output tokens at output rate + tokensRemoved × 0.1 × inputRate (reading
      //   the context to summarise it).
      const summaryInputCost = tokensRemoved * rate.inputRate * 0.1;
      const summaryOutputCost = 200 * rate.outputRate;
      llmCallCost = summaryInputCost + summaryOutputCost;
      const grossSavings = tokensRemoved * 0.1 * rate.inputRate;
      costSaved = grossSavings - llmCallCost;

      // Amnesia: flag when estimation error exceeds ±10%
      if (Math.abs(estimationErrorPct) > 10) {
        amnesiaFlag = true;
      }
      break;
    }
    case 2: {
      // snipCompact: no LLM call; every removed token is a direct saving at full input rate.
      llmCallCost = 0;
      costSaved = tokensRemoved * rate.inputRate;
      break;
    }
    case 3: {
      // contextCollapse: purely local; every removed token saves its full input rate cost.
      llmCallCost = 0;
      costSaved = tokensRemoved * rate.inputRate;
      break;
    }
  }

  const record: CompressionCostRecord = {
    recordedAt: new Date().toISOString(),
    tier,
    tokensRemoved,
    timeMs,
    success,
    provider,
    costSaved,
    llmCallCost,
    amnesiaFlag,
    estimationErrorPct: tier === 1 ? estimationErrorPct : 0,
  };

  _records.push(record);

  // Feed into the regret learner so UCB tier selection stays up to date.
  pipeCompressionCostToLearner(tier, tokensRemoved, provider);

  return record;
}

// ---------------------------------------------------------------------------
// ROI summary
// ---------------------------------------------------------------------------

/**
 * Summarize all recorded compression operations into a `SessionROI` object.
 *
 * Includes both a JSON-serializable form (returned) and a CSV export helper
 * (`toCSV`) attached to the result. The CSV is suitable for feeding into
 * external optimization loops.
 */
export function summarizeSessionROI(): SessionROI & { toCSV(): string } {
  const tierBreakdown: Record<CompressionTier, TierROIBreakdown> = {
    1: { callCount: 0, tokensRemoved: 0, costSaved: 0 },
    2: { callCount: 0, tokensRemoved: 0, costSaved: 0 },
    3: { callCount: 0, tokensRemoved: 0, costSaved: 0 },
  };

  let totalTokensSaved = 0;
  let totalCostSaved = 0;
  let compressionTimeMs = 0;
  let amnesiaCount = 0;

  for (const r of _records) {
    const b = tierBreakdown[r.tier];
    b.callCount++;
    b.tokensRemoved += r.tokensRemoved;
    b.costSaved += r.costSaved;

    totalTokensSaved += r.tokensRemoved;
    totalCostSaved += r.costSaved;
    compressionTimeMs += r.timeMs;
    if (r.amnesiaFlag) amnesiaCount++;
  }

  const roi: SessionROI = {
    totalTokensSaved,
    totalCostSaved,
    compressionTimeMs,
    tierBreakdown,
    amnesiaCount,
    generatedAt: new Date().toISOString(),
  };

  return {
    ...roi,
    toCSV(): string {
      return buildCSV(_records);
    },
  };
}

// ---------------------------------------------------------------------------
// Amnesia queries
// ---------------------------------------------------------------------------

/**
 * Return all tier-1 records that were flagged for recalibration.
 *
 * An amnesia flag is set when the autoCompact LLM call produced a token
 * count that deviated by more than ±10% from the pre-call estimate. A high
 * rate of amnesia flags indicates the tier-1 cost model needs re-tuning.
 */
export function getAmnesiaFlags(): readonly CompressionCostRecord[] {
  return _records.filter((r) => r.amnesiaFlag);
}

/**
 * True if any tier-1 record has an amnesia flag.
 * Convenience predicate for quick checks.
 */
export function hasAmnesiaFlags(): boolean {
  return _records.some((r) => r.amnesiaFlag);
}

// ---------------------------------------------------------------------------
// CSV export helper
// ---------------------------------------------------------------------------

function buildCSV(records: readonly CompressionCostRecord[]): string {
  const header =
    "recordedAt,tier,tokensRemoved,timeMs,success,provider,costSaved,llmCallCost,amnesiaFlag,estimationErrorPct";
  const rows = records.map((r) =>
    [
      r.recordedAt,
      r.tier,
      r.tokensRemoved,
      r.timeMs,
      r.success,
      r.provider,
      r.costSaved.toFixed(8),
      r.llmCallCost.toFixed(8),
      r.amnesiaFlag,
      r.estimationErrorPct.toFixed(2),
    ].join(","),
  );
  return [header, ...rows].join("\n");
}
