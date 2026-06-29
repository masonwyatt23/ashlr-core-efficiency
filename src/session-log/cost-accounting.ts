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
 *
 * ### Regret Backpropagation
 *
 * After an LLM call completes (post-message), the actual token cost is known.
 * `computeCompressorRegret()` attributes the counterfactual cost delta to the
 * specific compression tier that triggered compaction:
 *   regret = cost_if_tier_not_fired − actual_cost_after_tier_fired
 *
 * `recordCompressorOutcome()` persists `CompressorRoleRecord` entries to
 * `~/.ashlr/compressor-outcomes.jsonl` alongside the session log, enabling
 * cross-session fleet-level "tier 2 saved $X this week" queries and automatic
 * threshold tuning via the regret learner.
 */

import { appendFileSync, existsSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

import type { CompressionTier } from "../compression/context.ts";
import { PROVIDER_RATES, type ProviderName, defaultProviderRate } from "../tokens/index.ts";
import { pipeCompressionCostToLearner } from "../compression/regret-learner.ts";
import { CompressorStreamFeedback, observeCompressionOutcome } from "../compression/streaming-feedback.ts";

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
// Module-level CompressorStreamFeedback singleton
// ---------------------------------------------------------------------------

/**
 * Lazily-initialised `CompressorStreamFeedback` instance wired into
 * `recordCompressionCost` so every compression event automatically feeds
 * the in-flight EMA optimizer without callers needing to opt in.
 *
 * Tests can replace this via `_setFeedbackInstance()`.
 */
let _feedbackInstance: CompressorStreamFeedback | null = null;

function _getFeedbackInstance(): CompressorStreamFeedback {
  if (!_feedbackInstance) {
    _feedbackInstance = new CompressorStreamFeedback({ cwd: process.cwd() });
  }
  return _feedbackInstance;
}

/**
 * Override the feedback instance — for tests only.
 */
export function _setFeedbackInstance(instance: CompressorStreamFeedback | null): void {
  _feedbackInstance = instance;
}

/**
 * Return the current feedback instance (read-only access for tests).
 */
export function _getFeedbackInstanceForTest(): CompressorStreamFeedback | null {
  return _feedbackInstance;
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
  tier: 1 | 2 | 3 | 4,
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
    case 4: {
      // treeCompact: purely local subtree pruning; every removed token saves at full input rate.
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

  // Feed into the streaming feedback EMA optimizer so the in-flight
  // switchboard stays current. latencyMs is not available here (this is a
  // synchronous cost-record path), so we pass 0; the ROI computation inside
  // observeCompressionOutcome uses max(1, latencyMs) as the denominator.
  // toolCallSuccess defaults to true (cost recording implies a successful call).
  observeCompressionOutcome(_getFeedbackInstance(), {
    tier,
    provider,
    tokensRemoved,
    latencyMs: timeMs,
    toolCallSuccess: success,
    clarificationRequested: false,
    messageCount: 0, // unknown at this call site — bucket defaults to "xs"
    estimatedByBlock: {},
  });

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
    4: { callCount: 0, tokensRemoved: 0, costSaved: 0 },
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

// ---------------------------------------------------------------------------
// Regret Backpropagation Engine
// ---------------------------------------------------------------------------

/**
 * Role a compressor tier played in a specific query/session.
 *
 * Emitted by each tier function when it fires, then enriched with actual LLM
 * cost after the post-message callback so `regret` can be computed precisely.
 */
export interface CompressorRoleRecord {
  /** Agent or session identifier that triggered the compression. */
  agent: string;
  /** Which compression tier fired. */
  role: "tier1_autoCompact" | "tier2_snipCompact" | "tier3_contextCollapse" | "tier4_treeCompact";
  /** ISO-8601 wall-clock timestamp of when this tier triggered. */
  triggeredAt: string;
  /** Tokens removed (or estimated to be saved) by this tier invocation. */
  tokensSaved: number;
  /**
   * Net USD impact of this tier firing.
   * Positive = cost savings; negative = cost overhead (e.g. LLM summary call).
   */
  costImpactUsd: number;
  /** Round-trip latency of the LLM call that followed this compression (ms). */
  queryLatencyMs: number;
}

/**
 * Causal chain from tool-result size → tier selection → LLM call → cost delta.
 *
 * Records the "before" and "after" state so the backpropagation engine can
 * attribute responsibility to the specific tier that fired.
 */
export type BackpropagationPath = {
  /** Tokens in context before any tier fired. */
  tokensBefore: number;
  /** Tokens in context after the tier fired (i.e. tokensBefore − tokensSaved). */
  tokensAfter: number;
  /** USD cost of the LLM call *with* the tier applied (actual observed cost). */
  actualLlmCostUsd: number;
  /**
   * USD cost the LLM call *would have* cost without the tier
   * (counterfactual — computed from tokensAfter + tokensSaved at input rate).
   */
  counterfactualLlmCostUsd: number;
  /** Net cost delta: counterfactual − actual. Positive = the tier saved money. */
  costDeltaUsd: number;
  /** The tier whose firing is attributed with this cost delta. */
  attributedTier: CompressionTier;
  /** ISO-8601 timestamp of when the LLM call completed (post-message). */
  attributedAt: string;
};

/**
 * Output of `computeCompressorRegret` — quantifies "if this tier had NOT
 * fired, what would the cost have been?"
 *
 * A positive `regretUsd` means the tier saved money (negative regret in the
 * classical RL sense — this is actually *reward*). We keep the sign convention
 * consistent with the rest of the regret-learner (lower is better).
 */
export interface RegretSignal {
  /** The tier being evaluated. */
  tier: CompressionTier;
  /** Tokens removed by this tier in the evaluated window. */
  tokensRemovedInWindow: number;
  /** Actual USD cost incurred in the window. */
  actualCostUsd: number;
  /** Counterfactual USD cost if the tier had NOT fired in the window. */
  counterfactualCostUsd: number;
  /**
   * USD saved by the tier: counterfactualCostUsd − actualCostUsd.
   * Positive = tier saved money. Negative = tier cost more than it saved.
   */
  regretUsd: number;
  /**
   * Per-query average USD saving (regretUsd / queryCount).
   * Useful for normalizing across sessions of different lengths.
   */
  avgSavingPerQuery: number;
  /** Number of queries considered in this regret computation. */
  queryCount: number;
  /** ISO-8601 timestamp when this signal was computed. */
  computedAt: string;
}

// ---------------------------------------------------------------------------
// Compressor-outcome persistence
// ---------------------------------------------------------------------------

/**
 * Resolved path for compressor outcome records.
 * Respects `ASHLR_COMPRESSOR_OUTCOMES_PATH`, else `~/.ashlr/compressor-outcomes.jsonl`.
 * Evaluated lazily so tests can override via env before first call.
 */
function resolveOutcomesPath(): string {
  const override = process.env.ASHLR_COMPRESSOR_OUTCOMES_PATH;
  if (override && override.length > 0) return override;
  return join(homedir(), ".ashlr", "compressor-outcomes.jsonl");
}

function ensureOutcomesDir(path: string): void {
  const dir = dirname(path);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

/**
 * Append a `CompressorRoleRecord` to `~/.ashlr/compressor-outcomes.jsonl`.
 *
 * Write is best-effort: any I/O error is swallowed so a broken filesystem
 * never crashes the compression pipeline. Uses POSIX O_APPEND for
 * concurrent-writer safety (lines ≤ PIPE_BUF = 4KB on macOS/Linux).
 *
 * Set `ASHLR_COMPRESSOR_OUTCOMES_PATH` to override the file location (useful
 * in tests). Set `ASHLR_SESSION_LOG=0` to disable all persistence writes.
 *
 * @param entry  The compressor outcome to persist.
 */
export function recordCompressorOutcome(entry: CompressorRoleRecord): void {
  if (process.env.ASHLR_SESSION_LOG === "0") return;
  try {
    const path = resolveOutcomesPath();
    ensureOutcomesDir(path);
    const line = JSON.stringify(entry) + "\n";
    appendFileSync(path, line, { flag: "a" });
  } catch {
    // Best-effort — never throw from a logging path.
  }
}

// ---------------------------------------------------------------------------
// In-process compressor-outcome store (parallel to _records for session ROI)
// ---------------------------------------------------------------------------

const _outcomeRecords: CompressorRoleRecord[] = [];

/** Expose outcome records for tests (read-only snapshot). */
export function _getOutcomeRecords(): readonly CompressorRoleRecord[] {
  return _outcomeRecords;
}

/** Reset outcome store — test hook only. */
export function _resetOutcomeRecords(): void {
  _outcomeRecords.length = 0;
}

/**
 * Record a compressor outcome both in-process and on disk.
 *
 * This is the primary call site used by tier hooks in `context.ts`.
 *
 * @param entry  The compressor outcome to record.
 */
export function trackCompressorOutcome(entry: CompressorRoleRecord): void {
  _outcomeRecords.push(entry);
  recordCompressorOutcome(entry);
}

// ---------------------------------------------------------------------------
// Regret computation
// ---------------------------------------------------------------------------

/**
 * Map a `CompressionTier` number to its `CompressorRoleRecord.role` string.
 */
export function tierToRole(
  tier: CompressionTier,
): CompressorRoleRecord["role"] {
  switch (tier) {
    case 1: return "tier1_autoCompact";
    case 2: return "tier2_snipCompact";
    case 3: return "tier3_contextCollapse";
    case 4: return "tier4_treeCompact";
  }
}

/**
 * Map a `CompressorRoleRecord.role` string back to its `CompressionTier` number.
 */
export function roleToTier(role: CompressorRoleRecord["role"]): CompressionTier {
  switch (role) {
    case "tier1_autoCompact": return 1;
    case "tier2_snipCompact": return 2;
    case "tier3_contextCollapse": return 3;
    case "tier4_treeCompact": return 4;
  }
}

/**
 * Build a `BackpropagationPath` after an LLM call completes.
 *
 * Called post-message once the actual token usage is known. The path captures
 * the causal chain from tier firing → token reduction → LLM cost delta.
 *
 * @param tokensBefore        Context tokens before the tier fired.
 * @param tokensSaved         Tokens removed by the tier.
 * @param actualLlmCostUsd    Observed USD cost of the LLM call after compression.
 * @param tier                Tier attributed with this reduction.
 * @param provider            Provider string for counterfactual pricing.
 * @returns                   A fully-populated `BackpropagationPath`.
 */
export function buildBackpropagationPath(
  tokensBefore: number,
  tokensSaved: number,
  actualLlmCostUsd: number,
  tier: CompressionTier,
  provider: string,
): BackpropagationPath {
  const rate = defaultProviderRate(provider);
  // Counterfactual: what would the LLM call have cost without this tier?
  // The tier removed `tokensSaved` tokens that would have been sent at
  // the full input rate.
  const counterfactualLlmCostUsd = actualLlmCostUsd + tokensSaved * rate.inputRate;
  const costDeltaUsd = counterfactualLlmCostUsd - actualLlmCostUsd;

  return {
    tokensBefore,
    tokensAfter: tokensBefore - tokensSaved,
    actualLlmCostUsd,
    counterfactualLlmCostUsd,
    costDeltaUsd,
    attributedTier: tier,
    attributedAt: new Date().toISOString(),
  };
}

/**
 * Quantify "if this tier had NOT fired, what would the cost have been?"
 *
 * Scans the `historyWindow` most-recent `CompressorRoleRecord` entries for
 * the given `tier`, then computes the aggregate counterfactual vs actual cost
 * from the `SessionROI` data.
 *
 * ### Algorithm
 *
 * For each matching outcome in the window:
 *   counterfactual += tokensSaved × inputRate  (tokens that would have been sent)
 *   actual         += |costImpactUsd|           (what we actually spent/saved)
 *
 * regretUsd = counterfactual − actual
 *   Positive → tier saved money (good).
 *   Negative → tier cost more than it saved (bad — recalibrate threshold).
 *
 * @param session        Current session ROI summary (for provider + cost data).
 * @param tier           Compression tier to evaluate.
 * @param historyWindow  Number of most-recent outcome records to consider.
 * @returns              A `RegretSignal` quantifying the tier's cost impact.
 */
export function computeCompressorRegret(
  session: SessionROI,
  tier: CompressionTier,
  historyWindow: number,
): RegretSignal {
  const role = tierToRole(tier);

  // Pull the most-recent `historyWindow` outcomes for this tier.
  const windowRecords = _outcomeRecords
    .filter((r) => r.role === role)
    .slice(-historyWindow);

  const queryCount = windowRecords.length;

  if (queryCount === 0) {
    return {
      tier,
      tokensRemovedInWindow: 0,
      actualCostUsd: 0,
      counterfactualCostUsd: 0,
      regretUsd: 0,
      avgSavingPerQuery: 0,
      queryCount: 0,
      computedAt: new Date().toISOString(),
    };
  }

  // Use the per-tier breakdown from SessionROI as the source of actual costs.
  const breakdown = session.tierBreakdown[tier];
  // Scale to the window fraction: window / total calls for this tier.
  const totalCallsForTier = breakdown.callCount;
  const windowFraction = totalCallsForTier > 0 ? Math.min(1, queryCount / totalCallsForTier) : 0;

  // Actual cost impact (sum of |costImpactUsd| for window outcomes).
  // costImpactUsd on a CompressorRoleRecord is the cost attributed at record time.
  const actualCostUsd =
    windowRecords.reduce((sum, r) => sum + Math.abs(r.costImpactUsd), 0);

  // Counterfactual: tokens that would have been sent at full input rate.
  // We reconstruct this from tokensSaved × default provider input rate.
  // Use the provider from the first record in the window (best proxy).
  const provider = windowRecords[0]!.agent; // agent field doubles as provider slug when set
  const tokensRemovedInWindow = windowRecords.reduce((sum, r) => sum + r.tokensSaved, 0);

  // If no provider info, use scaled SessionROI breakdown cost as counterfactual.
  // This makes the function robust when agent field doesn't carry a provider slug.
  const counterfactualCostUsd =
    breakdown.costSaved > 0
      ? breakdown.costSaved * windowFraction + actualCostUsd
      : tokensRemovedInWindow * defaultProviderRate("claude-3-5-sonnet").inputRate;

  const regretUsd = counterfactualCostUsd - actualCostUsd;
  const avgSavingPerQuery = queryCount > 0 ? regretUsd / queryCount : 0;

  return {
    tier,
    tokensRemovedInWindow,
    actualCostUsd,
    counterfactualCostUsd,
    regretUsd,
    avgSavingPerQuery,
    queryCount,
    computedAt: new Date().toISOString(),
  };
}
