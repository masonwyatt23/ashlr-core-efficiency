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
 *
 * ### Cost-history learning (CompressorLearner)
 *
 * `CompressorLearner` extends the token-feedback layer with a cost-delta
 * tracking layer. Each call records not just token counts but actual USD costs
 * from the API response and a "baseline cost" representing what the call would
 * have cost without any compression.
 *
 * Records are bucketed by provider × message-count range (xs/sm/md/lg) and a
 * per-bucket, per-tier ROI is computed:
 *   ROI = (baseline_cost − actual_cost) / switch_cost
 *
 * When ROI for a tier exceeds the threshold (default: 0.15 USD/call) the
 * learner emits a `TierROIBreakdown` recommending that tier. This allows
 * `selectCompressionTierAdaptive` to favour historically-cheap tiers over
 * current buffer fill ratio, delivering an expected 8–15% USD reduction on
 * steady-state sessions.
 */

import { join } from "path";
import { appendJsonl, readJsonl } from "../genome/jsonl.ts";
import { contextCollapse, selectCompressionTier, snipCompact, treeCompact, type CompressionTier, type ContextConfig, DEFAULT_CONFIG } from "./context.ts";
import { estimateTokensFromMessages } from "../tokens/index.ts";
import { recordCompressionCost } from "../session-log/cost-accounting.ts";
import { computeProviderCostRatio } from "../budget/index.ts";
import type { Message } from "../types/index.ts";
import type { CompressibleProfile } from "./consolidation.ts";
import { SemanticPreFilterer } from "./semantic-prefilter.ts";
import { bucketMessageCount, type MessageCountBucket } from "../budget/index.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GENOME_DIR = ".ashlrcode/genome";
const HISTORY_FILE = "compression-history.jsonl";
const COST_HISTORY_FILE = "compression-cost-history.jsonl";

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
 * Also feeds results into the cost-accounting layer so USD savings are
 * tracked alongside the learning history. Pass `provider` and `timeMs`
 * for full cost accounting; they default to sensible values if omitted.
 *
 * @param cwd             Project working directory (genome root).
 * @param tier            Compression tier that was applied.
 * @param estimatedTokens Estimated token count before the LLM call.
 * @param actualTokens    Actual tokens reported by the LLM (from TokenUsage).
 * @param success         Whether the LLM call completed without a context error.
 * @param provider        Provider/model string for pricing lookup (default: "claude-3-5-sonnet").
 * @param timeMs          Wall-clock time the compression took in ms (default: 0).
 */
export async function recordCompressionResult(
  cwd: string,
  tier: CompressionTier,
  estimatedTokens: number,
  actualTokens: number,
  success: boolean,
  provider = "claude-3-5-sonnet",
  timeMs = 0,
): Promise<void> {
  const record: CompressionFeedback = {
    tier,
    estimatedTokens,
    actualTokens,
    success,
    recordedAt: new Date().toISOString(),
  };
  await appendJsonl(compressionHistoryPath(cwd), record);

  // Feed into cost accounting. For tier 1, compute the signed estimation error.
  const tokensRemoved = Math.max(0, estimatedTokens - actualTokens);
  const estimationErrorPct =
    tier === 1 && estimatedTokens > 0
      ? ((actualTokens - estimatedTokens) / estimatedTokens) * 100
      : 0;

  recordCompressionCost(tier, tokensRemoved, timeMs, success, provider, estimationErrorPct);
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
    4: [],
  };
  for (const r of records) {
    if (r.tier === 1 || r.tier === 2 || r.tier === 3 || r.tier === 4) {
      buckets[r.tier].push(r);
    }
  }

  const byTier = {} as Record<CompressionTier, TierEffectiveness>;
  for (const t of [1, 2, 3, 4] as CompressionTier[]) {
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
  const buckets: Record<CompressionTier, CompressionFeedback[]> = { 1: [], 2: [], 3: [], 4: [] };
  for (const r of allRecords) {
    if (r.tier === 1 || r.tier === 2 || r.tier === 3 || r.tier === 4) {
      buckets[r.tier].push(r);
    }
  }

  const byTier = {} as Record<CompressionTier, TierCalibration>;
  let anyDriftDetected = false;

  for (const t of [1, 2, 3, 4] as CompressionTier[]) {
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
 * @param provider      Optional current provider name. When supplied the
 *                      function consults `computeProviderCostRatio` relative to
 *                      a neutral baseline ("anthropic") and applies the
 *                      recommended tier shift so cheaper providers naturally
 *                      use lighter compression and more expensive providers use
 *                      more aggressive compression. Has no effect when
 *                      `history` is `null` (static path remains unchanged).
 * @param consolidationProfile  Optional CompressibleProfile from
 *                      `getCompressionProfile`. When supplied and the profile
 *                      contains a ranked entry for `provider`, the
 *                      dominance-rank #1 tier from the profile overrides the
 *                      UCB/history-derived tier — but only when the profile
 *                      has sufficient confidence (≥ MIN_SAMPLES_FOR_LEARNING
 *                      observations). This allows cross-session compression
 *                      history to bootstrap tier selection on provider switch,
 *                      eliminating the per-project re-learn cost.
 * @param skipPreFilter Optional flag to bypass the semantic pre-filter (e.g.
 *                      in tests or when the caller has already applied its own
 *                      keyword ranking). Defaults to false.
 * @param roiBreakdown  Optional pre-computed `TierROIBreakdown` from
 *                      `CompressorLearner.computeROI`. When provided and
 *                      `recommendedTier` is non-null, the ROI-recommended tier
 *                      overrides the history/calibration-derived tier. This is
 *                      the integration point for cost-history learning: callers
 *                      that instantiate a `CompressorLearner` and call
 *                      `computeROI` before each LLM call can feed the result
 *                      here so that USD-based historical signal takes precedence
 *                      over token-count-based heuristics.
 */
export function selectCompressionTierAdaptive(
  messages: Message[],
  systemTokens: number,
  config: Partial<ContextConfig> = {},
  history?: LearnedThresholds | null,
  calibration?: CalibrationRecommendation | null,
  provider?: string | null,
  consolidationProfile?: CompressibleProfile | null,
  skipPreFilter = false,
  roiBreakdown?: TierROIBreakdown | null,
): CompressionTier {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  // Always compute static baseline first
  const staticTier = selectCompressionTier(messages, systemTokens, cfg);

  // No history → pure static. Apply ROI override first if available, then return.
  if (!history) {
    if (roiBreakdown?.recommendedTier != null) return roiBreakdown.recommendedTier;
    return staticTier;
  }

  // ---------------------------------------------------------------------------
  // Semantic pre-filter: TF-IDF + BM25 keyword ranking before Ollama calls.
  //
  // When history IS available (the learned path), we run the pre-filter to
  // check whether keyword signal alone is decisive.  Two fast-exit paths:
  //   • "pick_top" / "pick_cheaper" → return the keyword-chosen tier
  //     immediately, skipping the full history + calibration walk below.
  //   • "invoke_ollama" → fall through to the full history-based selection.
  //
  // The pre-filter is skipped when:
  //   - skipPreFilter is explicitly true (testing / caller override)
  //   - messages array is empty (no signal to extract)
  // ---------------------------------------------------------------------------
  if (!skipPreFilter && messages.length > 0) {
    const limit = cfg.maxContextTokens - cfg.reserveTokens;
    const prefilter = new SemanticPreFilterer();
    const preResult = prefilter.rank(messages, limit);

    if (preResult.recommendation !== "invoke_ollama") {
      const keywordTier = prefilter.pickTier(preResult);
      if (keywordTier !== null) {
        // Validate the keyword-chosen tier still fits within budget (basic
        // sanity check — the pre-filter is not budget-aware by design).
        const tierTokens = estimatedTierTokens(messages, keywordTier as CompressionTier);
        if (tierTokens + systemTokens <= limit) {
          return keywordTier as CompressionTier;
        }
        // Budget check failed for keyword tier: fall through to full selection
      }
    }
  }

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

  // Walk from cheapest (4) to most aggressive (1), picking the first tier
  // that is predicted to succeed according to history.
  let selectedTier: CompressionTier = 1; // default: most aggressive fallback
  for (const tier of [4, 3, 2, 1] as CompressionTier[]) {
    if (tierLikelySucceeds(tier)) {
      selectedTier = tier;
      break;
    }
  }

  // Apply provider-pricing tier shift when a provider hint is supplied.
  // We compare against "anthropic" as the neutral baseline (mid-range pricing).
  // A cheaper provider (e.g. deepseek, groq) relaxes compression by one tier;
  // a more expensive provider (e.g. o1, xai) tightens it by one tier.
  if (provider) {
    const { recommendedTierShift } = computeProviderCostRatio("anthropic", provider);
    if (recommendedTierShift !== 0) {
      // Tier numbers are inverted: lower number = more aggressive.
      // A shift of -1 (cheaper provider) means less aggressive → higher tier number.
      // A shift of +1 (pricier provider) means more aggressive → lower tier number.
      const shifted = (selectedTier - recommendedTierShift) as CompressionTier;
      selectedTier = Math.min(3, Math.max(1, shifted)) as CompressionTier;
    }
  }

  // Apply cross-session consolidation profile when available.
  // When the profile has a high-confidence dominance-rank #1 tier for the
  // current provider, prefer it over the per-session learned tier.
  // This bootstraps compression selection on provider switch without re-learning.
  if (consolidationProfile && provider) {
    const providerEntries = consolidationProfile.byProvider[provider];
    if (providerEntries && providerEntries.length > 0) {
      const top = providerEntries[0];
      if (top !== undefined && top.sampleCount >= MIN_SAMPLES_FOR_LEARNING) {
        // Consolidation profile has enough data: prefer its top-ranked tier.
        // Clamp to valid range to be defensive.
        selectedTier = Math.min(4, Math.max(1, top.tier)) as CompressionTier;
      }
    }
  }

  // Apply cost-history ROI recommendation when available.
  // The ROI breakdown (from CompressorLearner.computeROI) takes precedence
  // over token-count heuristics when historical USD signal is available.
  // Only override when the breakdown has a concrete recommendation, ensuring
  // graceful degradation when history is sparse.
  if (roiBreakdown?.recommendedTier != null) {
    selectedTier = roiBreakdown.recommendedTier;
  }

  return selectedTier;
}

// ---------------------------------------------------------------------------
// CompressorLearner — cost-history based ROI tier selection
// ---------------------------------------------------------------------------

/**
 * A single cost-history record captured after an API call completes.
 *
 * Contains both token and USD cost data so ROI can be computed per bucket.
 */
export interface CostHistoryRecord {
  /** Which tier was applied. */
  tier: CompressionTier;
  /** Number of messages in the context at time of call. */
  messageCount: number;
  /** Provider/model string (e.g. "claude-3-5-sonnet"). */
  provider: string;
  /** Token count estimated before the call (chars/4 heuristic). */
  estimated_tokens: number;
  /** Actual input tokens reported by the LLM response. */
  actual_tokens: number;
  /** Actual USD cost of the LLM call as reported by the API. */
  actual_cost_usd: number;
  /** Whether the call succeeded without a context-limit error. */
  success: boolean;
  /** ISO timestamp. */
  recordedAt: string;
}

/**
 * Per-tier ROI computed for a specific provider × message-count bucket.
 *
 * ROI = (baseline_cost − actual_cost) / switch_cost
 *
 * where:
 *  - baseline_cost is the median actual_cost_usd for the lightest tier (tier 4)
 *    in the same bucket (representing "no compression overhead").
 *  - actual_cost is the median actual_cost_usd for this tier in the bucket.
 *  - switch_cost is the estimated cost of switching from tier 4 to this tier
 *    (approximated as the difference in per-token input rates).
 */
export interface TierROIEntry {
  tier: CompressionTier;
  /** Number of records in this bucket for this tier. */
  sampleCount: number;
  /** Median actual_cost_usd across bucket records for this tier. */
  medianCostUsd: number;
  /** Computed ROI value. Negative means this tier costs more than baseline. */
  roi: number;
  /** True when roi >= roiThreshold and sampleCount >= MIN_COST_SAMPLES. */
  recommended: boolean;
}

/**
 * Full ROI breakdown emitted by `CompressorLearner.computeROI`.
 *
 * Contains per-bucket, per-tier ROI data and the overall recommended tier.
 */
export interface TierROIBreakdown {
  /** Provider this breakdown was computed for. */
  provider: string;
  /** Message-count bucket: xs/sm/md/lg. */
  bucket: MessageCountBucket;
  /** ROI threshold used (default: 0.15). */
  roiThreshold: number;
  /** Per-tier ROI entries for this provider+bucket. */
  byTier: Record<CompressionTier, TierROIEntry>;
  /**
   * The tier recommended by historical ROI.
   * The lightest tier (highest number) whose ROI exceeds roiThreshold.
   * Falls back to null when history is insufficient.
   */
  recommendedTier: CompressionTier | null;
  /** ISO timestamp. */
  computedAt: string;
}

/** Minimum number of cost records per tier+bucket before ROI is trusted. */
const MIN_COST_SAMPLES = 3;

/** Default ROI threshold: tier 1 ROI > 0.15 USD/call is recommended. */
const DEFAULT_ROI_THRESHOLD = 0.15;

/** Sliding window size for cost history (records kept per file). */
const COST_HISTORY_WINDOW = 50;

/**
 * Approximate switch cost between two tiers for the same provider.
 *
 * Modelled as the cost of one extra LLM input scan of an average-sized
 * context (2000 tokens) at the provider's input rate, multiplied by
 * the tier aggressiveness delta. This is a conservative estimate that
 * keeps the ROI denominator non-zero.
 *
 * @param fromTier  Source tier (lighter).
 * @param toTier    Target tier (more aggressive).
 * @param inputRate Provider's per-token input rate in USD.
 */
function estimateSwitchCost(
  fromTier: CompressionTier,
  toTier: CompressionTier,
  inputRate: number,
): number {
  // Aggressiveness delta: tier 1 is most aggressive, tier 4 least.
  const delta = Math.abs(fromTier - toTier);
  // 2000-token average context × inputRate × delta × overhead factor 0.05
  const base = 2000 * inputRate * 0.05 * Math.max(delta, 1);
  return Math.max(base, 1e-9); // never zero — prevents division by zero
}

/**
 * Compute the median of an array of numbers.
 * Returns 0 for empty arrays.
 */
function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1]! + sorted[mid]!) / 2
    : sorted[mid]!;
}

/**
 * Learning-based compression tier selector that tracks actual USD cost deltas.
 *
 * Maintains a sliding window of cost-history records in
 * `compression-cost-history.jsonl` (co-located with `compression-history.jsonl`
 * in the genome evolution directory). Records are bucketed by
 * `provider × messageCount range` and a per-tier ROI is computed:
 *
 *   ROI = (baseline_cost − actual_cost) / switch_cost
 *
 * When ROI for a tier exceeds `roiThreshold` (default 0.15) the learner
 * recommends that tier. `selectCompressionTierAdaptive` consults the learner
 * when a `CompressorLearner` instance is passed in, using historical ROI
 * instead of (or in addition to) static heuristics.
 *
 * ### Usage
 *
 * ```ts
 * const learner = new CompressorLearner(cwd);
 * // After each API call:
 * await learner.record({ tier, messageCount, provider, estimated_tokens,
 *                        actual_tokens, actual_cost_usd, success });
 * // Before next call:
 * const breakdown = await learner.computeROI(provider, messageCount);
 * if (breakdown.recommendedTier !== null) {
 *   // use breakdown.recommendedTier
 * }
 * ```
 */
export class CompressorLearner {
  private readonly cwd: string;
  private readonly roiThreshold: number;

  constructor(cwd: string, roiThreshold = DEFAULT_ROI_THRESHOLD) {
    this.cwd = cwd;
    this.roiThreshold = roiThreshold;
  }

  /** Absolute path to the cost-history JSONL file. */
  costHistoryPath(): string {
    return join(this.cwd, GENOME_DIR, "evolution", COST_HISTORY_FILE);
  }

  /**
   * Append one cost-history record and enforce the sliding window.
   *
   * Reads the current file, appends the new record, then trims to the
   * latest COST_HISTORY_WINDOW records to keep the file bounded.
   */
  async record(entry: Omit<CostHistoryRecord, "recordedAt">): Promise<void> {
    const record: CostHistoryRecord = {
      ...entry,
      recordedAt: new Date().toISOString(),
    };

    // Read existing records, append new, trim to window, rewrite.
    // This keeps the file bounded at COST_HISTORY_WINDOW entries.
    const existing = await readJsonl<CostHistoryRecord>(this.costHistoryPath());
    existing.push(record);
    const windowed = existing.slice(-COST_HISTORY_WINDOW);

    // Rewrite file with windowed records (append-only per-line format).
    const { mkdir, writeFile } = await import("fs/promises");
    const { dirname } = await import("path");
    await mkdir(dirname(this.costHistoryPath()), { recursive: true });
    const content = windowed.map((r) => JSON.stringify(r)).join("\n") + "\n";
    await writeFile(this.costHistoryPath(), content, "utf-8");
  }

  /**
   * Read all cost-history records from the JSONL file.
   * Returns an empty array when the file does not exist.
   */
  async readHistory(): Promise<CostHistoryRecord[]> {
    return readJsonl<CostHistoryRecord>(this.costHistoryPath());
  }

  /**
   * Compute per-tier ROI for a given provider and message count.
   *
   * Filters records by provider and message-count bucket, then for each tier
   * computes:
   *   ROI = (baseline_median_cost − tier_median_cost) / switch_cost
   *
   * where baseline is the median cost for tier 4 (lightest) in the same bucket,
   * and switch_cost accounts for the overhead of switching tiers.
   *
   * Returns a `TierROIBreakdown` with `recommendedTier = null` when history is
   * insufficient (< MIN_COST_SAMPLES per tier).
   *
   * @param provider     Provider/model string to filter on.
   * @param messageCount Current message count to determine bucket.
   * @param records      Optional pre-loaded records (avoids re-reading in tests).
   */
  async computeROI(
    provider: string,
    messageCount: number,
    records?: CostHistoryRecord[],
  ): Promise<TierROIBreakdown> {
    const allRecords = records ?? await this.readHistory();
    const bucket = bucketMessageCount(messageCount);

    // Filter to matching provider + bucket.
    const relevant = allRecords.filter(
      (r) => r.provider === provider && bucketMessageCount(r.messageCount) === bucket,
    );

    // Bucket records by tier.
    const byTierRecords: Record<CompressionTier, CostHistoryRecord[]> = {
      1: [],
      2: [],
      3: [],
      4: [],
    };
    for (const r of relevant) {
      if (r.tier === 1 || r.tier === 2 || r.tier === 3 || r.tier === 4) {
        byTierRecords[r.tier].push(r);
      }
    }

    // Baseline: median cost of tier 4 (lightest — reference point).
    const tier4Costs = byTierRecords[4].map((r) => r.actual_cost_usd);
    const baselineCost = tier4Costs.length > 0 ? median(tier4Costs) : 0;

    // Get provider input rate for switch_cost estimation.
    const { defaultProviderRate } = await import("../tokens/index.ts");
    const rates = defaultProviderRate(provider);

    const byTier = {} as Record<CompressionTier, TierROIEntry>;
    let recommendedTier: CompressionTier | null = null;

    // Walk from most aggressive (1) to lightest (4), computing ROI.
    // We recommend the lightest tier whose ROI still clears the threshold.
    for (const t of [4, 3, 2, 1] as CompressionTier[]) {
      const tierRecords = byTierRecords[t];
      const sampleCount = tierRecords.length;
      const tierCosts = tierRecords.map((r) => r.actual_cost_usd);
      const medianCostUsd = sampleCount > 0 ? median(tierCosts) : 0;

      let roi = 0;
      if (sampleCount >= MIN_COST_SAMPLES && baselineCost > 0) {
        // Tier 4 is the baseline; ROI for tier 4 itself is defined as 0.
        if (t === 4) {
          roi = 0;
        } else {
          const switchCost = estimateSwitchCost(4, t, rates.inputRate);
          roi = (baselineCost - medianCostUsd) / switchCost;
        }
      } else if (sampleCount >= MIN_COST_SAMPLES && t === 4) {
        // Tier 4 with enough data but no comparison baseline: ROI = 0.
        roi = 0;
      }

      const recommended =
        sampleCount >= MIN_COST_SAMPLES && roi >= this.roiThreshold && t !== 4;

      byTier[t] = { tier: t, sampleCount, medianCostUsd, roi, recommended };

      // Track the lightest tier (highest number, walking 4→3→2→1) that is recommended.
      // Since we walk 4→1, we take the last recommended tier seen (most aggressive).
      if (recommended) {
        recommendedTier = t;
      }
    }

    // Prefer the lightest recommended tier: re-walk to find highest tier number with recommended=true.
    let lightestRecommended: CompressionTier | null = null;
    for (const t of [4, 3, 2, 1] as CompressionTier[]) {
      if (byTier[t].recommended) {
        lightestRecommended = t;
        break; // first (lightest) wins
      }
    }
    // Fall back to any recommended tier if lightest not found differently
    recommendedTier = lightestRecommended ?? (recommendedTier !== null && byTier[recommendedTier].recommended ? recommendedTier : null);

    return {
      provider,
      bucket,
      roiThreshold: this.roiThreshold,
      byTier,
      recommendedTier,
      computedAt: new Date().toISOString(),
    };
  }

  /**
   * Select a compression tier using historical ROI.
   *
   * Consults `computeROI` for the current provider and message count.
   * When a recommended tier exists and it has sufficient samples, returns it.
   * Otherwise falls back to the provided static tier.
   *
   * @param provider       Provider/model string.
   * @param messageCount   Current message count.
   * @param fallbackTier   Tier to use when history is insufficient.
   * @param records        Optional pre-loaded records (avoids re-reading).
   */
  async selectTier(
    provider: string,
    messageCount: number,
    fallbackTier: CompressionTier,
    records?: CostHistoryRecord[],
  ): Promise<CompressionTier> {
    const breakdown = await this.computeROI(provider, messageCount, records);
    return breakdown.recommendedTier ?? fallbackTier;
  }
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
    case 4:
      // treeCompact: estimate using a large budget (simulate best-effort pruning)
      return estimateTokensFromMessages(
        treeCompact(messages, undefined, {
          tokenBudget: 0, // force maximum pruning for estimation purposes
          keepRecentMessages: DEFAULT_CONFIG.recentMessageCount,
        }).messages,
      );
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
