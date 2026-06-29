/**
 * Adaptive compression tier selection with token feedback.
 *
 * Extends the static 3-tier ladder with a learning layer: as compression
 * operations complete the caller can record the real token counts from the
 * LLM response alongside the original estimate. `learnCompressionThresholds`
 * aggregates that history to compute per-tier effectiveness metrics, and
 * `selectCompressionTierAdaptive` uses those metrics to bias tier selection
 * toward tiers that historically succeed within budget.
 *
 * Falls back transparently to `selectCompressionTier` when history is empty
 * or unavailable, so there is no regression for brand-new installs.
 *
 * ### Calibration window
 *
 * `calibrateCompressionThresholds` adds a sliding-window calibrator that
 * operates over the latest 20 history records per tier. It computes the mean
 * absolute percentage error (MAPE) of token estimates for each tier and flags
 * tiers whose error has drifted beyond a configurable tolerance (default 15%).
 * When drift is detected it recommends a concrete weight adjustment factor so
 * `selectCompressionTierAdaptive` can apply tighter calibration and avoid
 * over- or under-selecting expensive tiers.
 */

import { join } from "path";
import { appendJsonl, readJsonl } from "../genome/jsonl.ts";
import { contextCollapse, selectCompressionTier, snipCompact, type CompressionTier, type ContextConfig, DEFAULT_CONFIG } from "./context.ts";
import { estimateTokensFromMessages } from "../tokens/index.ts";
import type { Message } from "../types/index.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GENOME_DIR = ".ashlrcode/genome";
const HISTORY_FILE = "compression-history.jsonl";

// Minimum number of recorded outcomes per tier before we trust the learned
// thresholds. Below this we fall back to static selection.
const MIN_SAMPLES_FOR_LEARNING = 3;

// Size of the sliding window used by the calibration subsystem.
const CALIBRATION_WINDOW_SIZE = 20;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Captures the actual vs estimated token usage from a single LLM response
 * after a compression tier was applied.
 */
export interface CompressionFeedback {
  /** Which tier was applied before the LLM call. */
  tier: CompressionTier;
  /** Token count estimated *before* the LLM call (chars/4 heuristic). */
  estimatedTokens: number;
  /** Actual input tokens reported by the LLM response. */
  actualTokens: number;
  /** Whether the call succeeded without hitting a context-limit error. */
  success: boolean;
  /** ISO timestamp recorded at append time. */
  recordedAt: string;
}

/**
 * Per-tier effectiveness derived from historical feedback records.
 */
export interface TierEffectiveness {
  tier: CompressionTier;
  /** Number of observations. */
  sampleCount: number;
  /** Fraction of calls where the tier succeeded (0–1). */
  successRate: number;
  /**
   * Average overshoot percentage: ((actual - estimated) / estimated) * 100.
   * Positive means the estimate under-counted actual tokens (LLM saw more).
   * Negative means the estimate over-counted (LLM saw fewer).
   */
  avgOvershootPct: number;
}

/**
 * Learned thresholds derived from `learnCompressionThresholds`.
 * Passed to `selectCompressionTierAdaptive` to influence tier selection.
 */
export interface LearnedThresholds {
  /** Indexed by tier number 1–3. */
  byTier: Record<CompressionTier, TierEffectiveness>;
}

/**
 * A single tier's calibration result from the sliding-window calibrator.
 */
export interface TierCalibration {
  tier: CompressionTier;
  /**
   * Number of records in the calibration window (up to CALIBRATION_WINDOW_SIZE).
   */
  windowSize: number;
  /**
   * Mean Absolute Percentage Error of token estimates across the window.
   * MAPE = mean(|actual - estimated| / estimated * 100).
   * A value of 0 means perfect estimates; 15 means estimates are off by 15% on average.
   */
  mape: number;
  /**
   * Whether this tier's estimation error has exceeded the tolerance threshold.
   * When true, the recommended weight adjustment should be applied.
   */
  driftDetected: boolean;
  /**
   * Recommended multiplicative weight adjustment for this tier's token estimate.
   * Derived from the signed mean overshoot so that calibrated_estimate = raw_estimate × weightAdjustment.
   * E.g. 1.20 means "inflate estimates by 20% before budget checks."
   * E.g. 0.90 means "deflate estimates by 10%."
   * Always ≥ 0.5 to prevent degenerate collapse.
   */
  weightAdjustment: number;
}

/**
 * Result returned by `calibrateCompressionThresholds`.
 *
 * Contains per-tier calibration data derived from the latest
 * CALIBRATION_WINDOW_SIZE history records. When `anyDriftDetected` is true
 * at least one tier should have its weight updated before the next call to
 * `selectCompressionTierAdaptive`.
 */
export interface CalibrationRecommendation {
  /** Per-tier calibration data. */
  byTier: Record<CompressionTier, TierCalibration>;
  /**
   * True if any tier exceeded the drift tolerance and weight adjustments are
   * recommended. Callers can gate on this flag before applying adjustments.
   */
  anyDriftDetected: boolean;
  /** The tolerance value (0–1) that was used, e.g. 0.15 for 15%. */
  tolerance: number;
  /** ISO timestamp of when the calibration was computed. */
  calibratedAt: string;
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

/**
 * Absolute path to the compression history JSONL file.
 * Located in the genome/evolution directory so it is co-located with other
 * genome audit trails and benefits from the same lifecycle management.
 */
export function compressionHistoryPath(cwd: string): string {
  return join(cwd, GENOME_DIR, "evolution", HISTORY_FILE);
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

/**
 * Append one compression outcome to the history file.
 *
 * @param cwd            Project working directory (genome root).
 * @param tier           Compression tier that was applied.
 * @param estimatedTokens Estimated token count before the LLM call.
 * @param actualTokens   Actual tokens reported by the LLM (from TokenUsage).
 * @param success        Whether the LLM call completed without a context error.
 */
export async function recordCompressionResult(
  cwd: string,
  tier: CompressionTier,
  estimatedTokens: number,
  actualTokens: number,
  success: boolean,
): Promise<void> {
  const record: CompressionFeedback = {
    tier,
    estimatedTokens,
    actualTokens,
    success,
    recordedAt: new Date().toISOString(),
  };
  await appendJsonl(compressionHistoryPath(cwd), record);
}

// ---------------------------------------------------------------------------
// Learning
// ---------------------------------------------------------------------------

/**
 * Read the compression history and compute per-tier effectiveness metrics.
 *
 * Returns `null` if no history file exists or it contains no valid records
 * (caller should fall back to static tier selection).
 */
export async function learnCompressionThresholds(
  cwd: string,
): Promise<LearnedThresholds | null> {
  const records = await readJsonl<CompressionFeedback>(compressionHistoryPath(cwd));
  if (records.length === 0) return null;

  // Bucket records by tier
  const buckets: Record<CompressionTier, CompressionFeedback[]> = {
    1: [],
    2: [],
    3: [],
  };
  for (const r of records) {
    if (r.tier === 1 || r.tier === 2 || r.tier === 3) {
      buckets[r.tier].push(r);
    }
  }

  const byTier = {} as Record<CompressionTier, TierEffectiveness>;
  for (const t of [1, 2, 3] as CompressionTier[]) {
    const samples = buckets[t];
    const sampleCount = samples.length;
    if (sampleCount === 0) {
      byTier[t] = { tier: t, sampleCount: 0, successRate: 1.0, avgOvershootPct: 0 };
      continue;
    }
    const successRate = samples.filter((s) => s.success).length / sampleCount;
    const overshootPcts = samples.map((s) =>
      s.estimatedTokens > 0
        ? ((s.actualTokens - s.estimatedTokens) / s.estimatedTokens) * 100
        : 0,
    );
    const avgOvershootPct = overshootPcts.reduce((a, b) => a + b, 0) / overshootPcts.length;
    byTier[t] = { tier: t, sampleCount, successRate, avgOvershootPct };
  }

  return { byTier };
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

/**
 * Compute a sliding-window calibration recommendation for compression tier
 * thresholds based on the latest CALIBRATION_WINDOW_SIZE (20) history records.
 *
 * For each tier the calibrator computes the Mean Absolute Percentage Error
 * (MAPE) of token estimates. When MAPE exceeds `tolerance` (default 0.15 =
 * 15%) drift is flagged and a concrete `weightAdjustment` factor is derived
 * from the signed mean overshoot. This factor can be fed directly into
 * `selectCompressionTierAdaptive` so tier selection automatically tightens
 * its budget predictions after an LLM version change, cache toggle, or any
 * other event that shifts the token distribution.
 *
 * @param cwd       Project working directory (genome root).
 * @param tolerance Drift threshold as a fraction (default 0.15 = 15%).
 *                  Tiers whose MAPE exceeds this will have driftDetected=true.
 * @returns         CalibrationRecommendation summarising per-tier drift status
 *                  and weight adjustments. Returns a zeroed recommendation when
 *                  no history is available (driftDetected=false everywhere).
 */
export async function calibrateCompressionThresholds(
  cwd: string,
  tolerance = 0.15,
): Promise<CalibrationRecommendation> {
  const allRecords = await readJsonl<CompressionFeedback>(compressionHistoryPath(cwd));

  // Bucket ALL records by tier first, then slice the latest window from each bucket.
  const buckets: Record<CompressionTier, CompressionFeedback[]> = { 1: [], 2: [], 3: [] };
  for (const r of allRecords) {
    if (r.tier === 1 || r.tier === 2 || r.tier === 3) {
      buckets[r.tier].push(r);
    }
  }

  const byTier = {} as Record<CompressionTier, TierCalibration>;
  let anyDriftDetected = false;

  for (const t of [1, 2, 3] as CompressionTier[]) {
    const all = buckets[t];
    // Sliding window: take only the most recent CALIBRATION_WINDOW_SIZE records.
    const window = all.slice(-CALIBRATION_WINDOW_SIZE);
    const windowSize = window.length;

    if (windowSize === 0) {
      byTier[t] = {
        tier: t,
        windowSize: 0,
        mape: 0,
        driftDetected: false,
        weightAdjustment: 1.0,
      };
      continue;
    }

    // Compute MAPE and signed mean overshoot over the window.
    let sumAbsError = 0;
    let sumSignedError = 0;
    let usableCount = 0;
    for (const r of window) {
      if (r.estimatedTokens <= 0) continue;
      const relError = (r.actualTokens - r.estimatedTokens) / r.estimatedTokens;
      sumAbsError += Math.abs(relError);
      sumSignedError += relError;
      usableCount++;
    }

    if (usableCount === 0) {
      byTier[t] = {
        tier: t,
        windowSize,
        mape: 0,
        driftDetected: false,
        weightAdjustment: 1.0,
      };
      continue;
    }

    const mape = (sumAbsError / usableCount) * 100; // percentage
    const meanSignedOvershootFraction = sumSignedError / usableCount;
    const driftDetected = mape / 100 > tolerance;

    // Weight adjustment: 1 + mean signed overshoot (clamped to [0.5, 3.0]).
    // If the model consistently under-counts (positive overshoot), inflate estimates.
    // If the model consistently over-counts (negative overshoot), deflate estimates.
    const rawAdjustment = 1 + meanSignedOvershootFraction;
    const weightAdjustment = Math.min(3.0, Math.max(0.5, rawAdjustment));

    if (driftDetected) anyDriftDetected = true;

    byTier[t] = { tier: t, windowSize, mape, driftDetected, weightAdjustment };
  }

  return {
    byTier,
    anyDriftDetected,
    tolerance,
    calibratedAt: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Adaptive tier selection
// ---------------------------------------------------------------------------

/**
 * Select a compression tier using learned history when available.
 *
 * Strategy:
 *   1. Compute the base static tier via `selectCompressionTier`.
 *   2. If we have enough history for the chosen tier, apply calibration:
 *      - Inflate the estimated token count by `avgOvershootPct` to get a
 *        calibrated estimate.
 *      - When a `CalibrationRecommendation` is supplied and drift has been
 *        detected for a tier, the window-derived `weightAdjustment` overrides
 *        the history-derived overshoot factor for that tier. This allows the
 *        calibration window (latest 20 records) to react faster to sudden
 *        shifts (e.g. after an LLM version change or cache toggle).
 *      - If a lower-cost tier (higher number) now fits within budget AND has
 *        a sufficient success rate, prefer it.
 *      - If the chosen tier has a low historical success rate, escalate to a
 *        more aggressive tier (lower number).
 *   3. Fall back to the static result if history is sparse.
 *
 * @param messages      Current message array.
 * @param systemTokens  Estimated tokens for the system prompt.
 * @param config        Context config (uses defaults if omitted).
 * @param history       Learned thresholds from `learnCompressionThresholds`.
 *                      Pass `null` or `undefined` to use static selection.
 * @param calibration   Optional calibration recommendation from
 *                      `calibrateCompressionThresholds`. When provided and
 *                      drift is detected for a tier, the window weight
 *                      adjustment supersedes the history overshoot factor for
 *                      that tier, producing tighter budget predictions.
 */
export function selectCompressionTierAdaptive(
  messages: Message[],
  systemTokens: number,
  config: Partial<ContextConfig> = {},
  history?: LearnedThresholds | null,
  calibration?: CalibrationRecommendation | null,
): CompressionTier {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  // Always compute static baseline first
  const staticTier = selectCompressionTier(messages, systemTokens, cfg);

  // No history → pure static
  if (!history) return staticTier;

  // Capture non-nullable reference so nested functions can reference it safely.
  const thresholds: LearnedThresholds = history;

  // Check if any tier has enough data to trust
  const hasLearnedData = ([3, 2, 1] as CompressionTier[]).some(
    (t) => thresholds.byTier[t].sampleCount >= MIN_SAMPLES_FOR_LEARNING,
  );
  if (!hasLearnedData) return staticTier;

  const limit = cfg.maxContextTokens - cfg.reserveTokens;

  /**
   * Determine the effective calibration factor for a tier.
   *
   * Priority order:
   *   1. If a CalibrationRecommendation exists and flags drift for this tier,
   *      use its window-derived weightAdjustment (reacts to recent shifts).
   *   2. Otherwise fall back to the history-derived avgOvershootPct factor
   *      (slower but stable long-term signal).
   */
  function effectiveCalibrationFactor(tier: CompressionTier): number {
    if (calibration && calibration.byTier[tier].driftDetected) {
      // Window-derived adjustment already accounts for direction and magnitude.
      return Math.max(calibration.byTier[tier].weightAdjustment, 0.5);
    }
    const eff = thresholds.byTier[tier];
    return Math.max(1 + eff.avgOvershootPct / 100, 0.5);
  }

  /**
   * Score each tier: returns true if history predicts this tier will fit AND
   * has an acceptable success rate. We apply the learned overshoot calibration
   * to the tier's post-compression token estimate.
   */
  function tierLikelySucceeds(tier: CompressionTier): boolean {
    const eff = thresholds.byTier[tier];
    if (eff.sampleCount < MIN_SAMPLES_FOR_LEARNING) {
      // No data → assume it's fine (don't penalise unexplored tiers)
      return true;
    }
    // Reject tiers with a poor track record
    if (eff.successRate < 0.5) return false;
    // Apply calibration: if the LLM historically sees more tokens than we
    // estimate, inflate our prediction accordingly before testing the budget.
    const calibrationFactor = effectiveCalibrationFactor(tier);
    const tierTokens = estimatedTierTokens(messages, tier);
    const calibratedTierTokens = Math.round(tierTokens * calibrationFactor);
    return calibratedTierTokens + systemTokens <= limit;
  }

  // Walk from cheapest (3) to most aggressive (1), picking the first tier
  // that is predicted to succeed according to history.
  for (const tier of [3, 2, 1] as CompressionTier[]) {
    if (tierLikelySucceeds(tier)) {
      return tier;
    }
  }

  // All learned tiers are predicted to fail — fall back to the most aggressive.
  return 1;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Estimate the token count that would result from applying the given tier.
 * Does NOT invoke autoCompact (tier 1) because that requires an LLM call —
 * instead it returns the current message estimate (conservative: tier 1 is
 * always the final fallback, never skipped if budget is truly exceeded).
 */
function estimatedTierTokens(messages: Message[], tier: CompressionTier): number {
  switch (tier) {
    case 3:
      return estimateTokensFromMessages(contextCollapse(messages));
    case 2:
      return estimateTokensFromMessages(snipCompact(messages));
    case 1:
      // autoCompact requires an LLM call; we can't simulate its output here.
      // Return a conservative fraction of the current size (summary is typically
      // 10–20% of the original), biased toward "it will fit" to allow escalation.
      return Math.round(estimateTokensFromMessages(messages) * 0.15);
  }
}
