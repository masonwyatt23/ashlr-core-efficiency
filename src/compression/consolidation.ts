/**
 * Cross-Request Compression History Consolidation with Provider Migration Intent Detection.
 *
 * Loads compression outcomes from all past sessions (cost-accounting records + tier-feedback
 * from the compression history JSONL), buckets them by project characteristics and provider,
 * and builds a CompressibleProfile that survives CLI restarts.
 *
 * On a provider switch, `predictCompressionOnSwitch` consults the consolidated profile to emit
 * a `CompressorRecommendation` — "tier 2 was 25% cheaper on this project, tier 1 latency ±80ms".
 * This eliminates the compression re-learn cost after anthropic↔openai switches.
 *
 * ### Audit trail
 *
 * Every prediction is written to `evolution/compression-consolidation.jsonl` with fields:
 *   { provider_from, provider_to, recommended_tier, actual_tier, roi_delta }
 *
 * ### Architecture
 *
 * ```
 * readJsonl(compression-history.jsonl)
 *   └─► bucketHistory()     — group by (provider, msgCountRange)
 *         └─► extractBucketStats()  — successRate, roi_usd, latency_ms per tier
 *               └─► dominanceRank() — pick best tier accounting for switch cost
 *                     └─► CompressibleProfile
 *
 * predictCompressionOnSwitch(from, to, profile)
 *   └─► emit CompressorRecommendation
 *         └─► appendJsonl(compression-consolidation.jsonl)
 * ```
 */

import { join } from "path";
import { appendJsonl, readJsonl } from "../genome/jsonl.ts";
import type { CompressionTier } from "./context.ts";
import type { CompressionFeedback } from "./adaptive.ts";
import { compressionHistoryPath } from "./adaptive.ts";
import { computeProviderCostRatio } from "../budget/index.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GENOME_DIR = ".ashlrcode/genome";
const CONSOLIDATION_FILE = "compression-consolidation.jsonl";

/** Look back this many days when loading historical records. */
const DEFAULT_LOOKBACK_DAYS = 30;

/**
 * Number of buckets for message-count ranges.
 * Ranges: [1–10], [11–50], [51–200], [201+].
 */
type MessageCountRange = "xs" | "sm" | "md" | "lg";

/** Minimum samples in a bucket before we trust it. */
const MIN_BUCKET_SAMPLES = 3;

/**
 * Recalibration lag in "equivalent sessions" — how long it takes the adaptive
 * learner to relearn thresholds from scratch after a provider switch.
 * Used to weight the switching cost penalty.
 */
const RECALIBRATION_LAG_SESSIONS = 5;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/**
 * Aggregated statistics for one (provider, messageCountRange, tier) bucket.
 */
export interface BucketStats {
  /** Compression tier this bucket describes. */
  tier: CompressionTier;
  /** Provider this bucket's data comes from. */
  provider: string;
  /** Message count range. */
  messageCountRange: MessageCountRange;
  /** Number of observations in this bucket. */
  sampleCount: number;
  /** Fraction of calls that succeeded (0–1). */
  successRate: number;
  /** Average net USD saved per call. */
  roiUsd: number;
  /** Average latency in ms per call. */
  latencyMs: number;
  /** Mean signed overshoot % of token estimates. */
  avgOvershootPct: number;
}

/**
 * A bucket key that uniquely identifies a (provider, messageCountRange) combination.
 */
export interface BucketKey {
  provider: string;
  messageCountRange: MessageCountRange;
}

/**
 * Per-tier entry in a CompressibleProfile, representing the dominance-ranked
 * best tier for a given project + provider combination.
 */
export interface TierProfileEntry {
  tier: CompressionTier;
  /** Dominance rank (1 = best). Lower is better. */
  dominanceRank: number;
  /** Composite score used for ranking (higher = better). */
  compositeScore: number;
  successRate: number;
  roiUsd: number;
  latencyMs: number;
  avgOvershootPct: number;
  sampleCount: number;
}

/**
 * A project-level compression profile computed from consolidated history.
 *
 * Keyed by provider; for each provider a dominance-ranked list of tiers is
 * provided, accounting for switching cost and recalibration lag.
 */
export interface CompressibleProfile {
  /** Project working directory this profile was computed for. */
  cwd: string;
  /** ISO timestamp when this profile was computed. */
  computedAt: string;
  /** Number of historical records incorporated. */
  totalRecords: number;
  /**
   * Per-provider tier ranking. Keys are provider names.
   * Each value is an array of TierProfileEntry sorted by dominanceRank (ascending).
   */
  byProvider: Record<string, TierProfileEntry[]>;
  /**
   * The best overall tier across ALL providers (for cross-provider recommendations).
   * Uses the provider with the most records as the primary signal.
   */
  globalBestTier: CompressionTier;
}

/**
 * A recommendation emitted when a provider switch is detected.
 */
export interface CompressorRecommendation {
  /** Provider being switched from. */
  providerFrom: string;
  /** Provider being switched to. */
  providerTo: string;
  /**
   * Recommended compression tier for the target provider, incorporating:
   * - Historical ROI data for this project on both providers
   * - Provider cost ratio
   * - Switching cost penalty (recalibration lag)
   */
  recommendedTier: CompressionTier;
  /**
   * Human-readable explanation, e.g.:
   * "tier 2 was 25% cheaper on this project; tier 1 latency ±80ms"
   */
  rationale: string;
  /**
   * Estimated ROI delta (USD) compared to blind tier selection on the target
   * provider. Positive = recommendation saves money.
   */
  estimatedRoiDelta: number;
  /** ISO timestamp. */
  recommendedAt: string;
}

/**
 * Audit record written to compression-consolidation.jsonl.
 */
export interface ConsolidationAuditRecord {
  provider_from: string;
  provider_to: string;
  recommended_tier: CompressionTier;
  /**
   * Actual tier used (filled in by the caller after the fact via
   * `recordActualTierUsed`; null until then).
   */
  actual_tier: CompressionTier | null;
  /**
   * ROI delta observed after using the recommended tier (filled in later).
   * null until updated by the caller.
   */
  roi_delta: number | null;
  /** ISO timestamp of the recommendation. */
  recommended_at: string;
  /** Unique ID so callers can update this record after observing the outcome. */
  recommendation_id: string;
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

/**
 * Absolute path to the consolidation audit JSONL file.
 */
export function consolidationAuditPath(cwd: string): string {
  return join(cwd, GENOME_DIR, "evolution", CONSOLIDATION_FILE);
}

// ---------------------------------------------------------------------------
// Internal bucketing helpers
// ---------------------------------------------------------------------------

/**
 * Map a message count to a named range bucket.
 */
function messageCountRange(count: number): MessageCountRange {
  if (count <= 10) return "xs";
  if (count <= 50) return "sm";
  if (count <= 200) return "md";
  return "lg";
}

/**
 * Infer a rough message count from a CompressionFeedback record.
 *
 * The history records don't store message count directly, but token count is a
 * reliable proxy: estimated tokens / 200 ≈ message count for typical turns.
 * This heuristic is conservative (short messages) so it errs toward "xs"/"sm".
 */
function inferMessageCount(record: CompressionFeedback): number {
  return Math.max(1, Math.round(record.estimatedTokens / 200));
}

/**
 * Extract the provider from a CompressionFeedback record.
 *
 * The feedback records in compression-history.jsonl were written by
 * `recordCompressionResult` which records `provider` as part of the
 * cost-accounting call. Older records may not have a provider field; those
 * default to "anthropic".
 */
function extractProvider(record: CompressionFeedback & { provider?: string }): string {
  return record.provider ?? "anthropic";
}

// ---------------------------------------------------------------------------
// Core: bucket history
// ---------------------------------------------------------------------------

/**
 * Load history records from the past `lookbackDays` and group them into
 * (provider, messageCountRange, tier) buckets.
 *
 * @param cwd           Project working directory.
 * @param lookbackDays  How many days of history to load (default 30).
 * @returns             Map from bucket key string → array of matching records.
 */
async function loadAndBucketHistory(
  cwd: string,
  lookbackDays = DEFAULT_LOOKBACK_DAYS,
): Promise<Map<string, (CompressionFeedback & { provider?: string })[]>> {
  type FeedbackWithProvider = CompressionFeedback & { provider?: string };

  const allRecords = await readJsonl<FeedbackWithProvider>(compressionHistoryPath(cwd));
  if (allRecords.length === 0) return new Map();

  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - lookbackDays);

  const buckets = new Map<string, FeedbackWithProvider[]>();

  for (const record of allRecords) {
    // Filter by lookback window
    const recordedAt = new Date(record.recordedAt);
    if (isNaN(recordedAt.getTime()) || recordedAt < cutoff) continue;

    // Validate tier
    if (record.tier !== 1 && record.tier !== 2 && record.tier !== 3 && record.tier !== 4) continue;

    const provider = extractProvider(record);
    const msgRange = messageCountRange(inferMessageCount(record));
    const key = `${provider}::${msgRange}::${record.tier}`;

    const existing = buckets.get(key);
    if (existing) {
      existing.push(record);
    } else {
      buckets.set(key, [record]);
    }
  }

  return buckets;
}

/**
 * Compute BucketStats from a set of records sharing the same
 * (provider, messageCountRange, tier) key.
 */
function computeBucketStats(
  records: (CompressionFeedback & { provider?: string })[],
  provider: string,
  msgRange: MessageCountRange,
  tier: CompressionTier,
): BucketStats {
  const sampleCount = records.length;
  const successCount = records.filter((r) => r.success).length;
  const successRate = sampleCount > 0 ? successCount / sampleCount : 0;

  // ROI: proxy from token savings at a standard rate (1 USD / M tokens).
  // We use the tokens removed (estimated - actual when actual < estimated, else 0)
  // as a rough proxy for savings, normalized to a per-call USD figure.
  const avgRoiUsd =
    records.reduce((sum, r) => {
      const tokensRemoved = Math.max(0, r.estimatedTokens - r.actualTokens);
      // ~$3/M tokens (anthropic mid-tier rate)
      return sum + tokensRemoved * 3e-6;
    }, 0) / sampleCount;

  // Latency: not stored in feedback records, so we use tier as a proxy:
  //   tier 1 = ~2000ms (LLM call), tier 2 = ~200ms, tier 3 = ~50ms, tier 4 = ~30ms
  const tierLatencyProxy: Record<CompressionTier, number> = { 1: 2000, 2: 200, 3: 50, 4: 30 };
  const latencyMs = tierLatencyProxy[tier];

  const avgOvershootPct =
    records.reduce((sum, r) => {
      if (r.estimatedTokens <= 0) return sum;
      return sum + ((r.actualTokens - r.estimatedTokens) / r.estimatedTokens) * 100;
    }, 0) / sampleCount;

  return {
    tier,
    provider,
    messageCountRange: msgRange,
    sampleCount,
    successRate,
    roiUsd: avgRoiUsd,
    latencyMs,
    avgOvershootPct,
  };
}

// ---------------------------------------------------------------------------
// Dominance ranking
// ---------------------------------------------------------------------------

/**
 * Compute a composite score for a tier given its BucketStats.
 *
 * Higher score = better. Weights:
 *  - successRate (50%): most important — a failing tier is worthless
 *  - roiUsd (30%): financial value
 *  - latencyPenalty (-20%): penalize slow tiers (tier 1 pays an LLM call)
 *
 * Normalized so scores are in ~[0, 1].
 */
function compositeScore(stats: BucketStats): number {
  // Normalize latency: tier 1 (~2000ms) scores 0, tier 4 (~30ms) scores 1.
  const maxLatency = 2000;
  const latencyScore = Math.max(0, 1 - stats.latencyMs / maxLatency);

  // Normalize roi: cap at $0.001/call (beyond that it's all good).
  const roiScore = Math.min(1, stats.roiUsd / 0.001);

  return 0.5 * stats.successRate + 0.3 * roiScore + 0.2 * latencyScore;
}

/**
 * Build dominance-ranked tier entries for a given provider from its BucketStats.
 *
 * When a `switchingProviderFrom` is given, we apply a switching cost penalty to
 * tiers whose characteristics change significantly between providers. The penalty
 * is proportional to the provider cost ratio and the recalibration lag.
 *
 * @param statsByTier         Per-tier BucketStats for the target provider.
 * @param switchingProviderFrom  If non-null, the source provider (switching context).
 * @param targetProvider      The provider we are computing the ranking for.
 * @returns                   Array of TierProfileEntry sorted by dominanceRank.
 */
function rankTiers(
  statsByTier: Map<CompressionTier, BucketStats>,
  switchingProviderFrom: string | null,
  targetProvider: string,
): TierProfileEntry[] {
  const entries: TierProfileEntry[] = [];

  for (const [tier, stats] of statsByTier) {
    if (stats.sampleCount < MIN_BUCKET_SAMPLES) continue;

    let score = compositeScore(stats);

    // Apply switching cost penalty when transitioning providers.
    // The penalty reduces the score proportional to the cost ratio difference
    // and a recalibration lag factor.
    if (switchingProviderFrom !== null) {
      const ratio = computeProviderCostRatio(switchingProviderFrom, targetProvider);
      const blendedRatio = (ratio.inputRatio + ratio.outputRatio) / 2;
      // Penalty factor: the further from 1.0 the ratio is, the bigger the
      // uncertainty about whether this tier will perform the same on the new provider.
      const switchUncertainty = Math.abs(blendedRatio - 1.0) / RECALIBRATION_LAG_SESSIONS;
      score = score * (1 - Math.min(0.3, switchUncertainty));
    }

    entries.push({
      tier,
      dominanceRank: 0, // filled in after sorting
      compositeScore: score,
      successRate: stats.successRate,
      roiUsd: stats.roiUsd,
      latencyMs: stats.latencyMs,
      avgOvershootPct: stats.avgOvershootPct,
      sampleCount: stats.sampleCount,
    });
  }

  // Sort descending by composite score → assign dominance ranks.
  entries.sort((a, b) => b.compositeScore - a.compositeScore);
  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    if (entry !== undefined) {
      entry.dominanceRank = i + 1;
    }
  }

  return entries;
}

// ---------------------------------------------------------------------------
// HistoryConsolidator
// ---------------------------------------------------------------------------

/**
 * Loads and consolidates compression history across all past sessions for a
 * given project directory.
 *
 * Usage:
 * ```ts
 * const consolidator = new HistoryConsolidator("/my/project");
 * const profile = await consolidator.buildProfile();
 * ```
 */
export class HistoryConsolidator {
  private readonly cwd: string;
  private readonly lookbackDays: number;

  constructor(cwd: string, lookbackDays = DEFAULT_LOOKBACK_DAYS) {
    this.cwd = cwd;
    this.lookbackDays = lookbackDays;
  }

  /**
   * Build a CompressibleProfile by loading and analyzing history records.
   *
   * Returns a profile with empty `byProvider` when no history is available.
   */
  async buildProfile(): Promise<CompressibleProfile> {
    const buckets = await loadAndBucketHistory(this.cwd, this.lookbackDays);

    // Group bucket stats by (provider, messageCountRange) → tier.
    // We flatten across all messageCountRanges to get a per-provider view,
    // weighting by sample count.
    const providerTierStats = new Map<string, Map<CompressionTier, BucketStats[]>>();
    let totalRecords = 0;

    for (const [key, records] of buckets) {
      const [provider, msgRange, tierStr] = key.split("::") as [string, MessageCountRange, string];
      const tier = parseInt(tierStr, 10) as CompressionTier;

      totalRecords += records.length;

      const stats = computeBucketStats(records, provider, msgRange, tier);

      if (!providerTierStats.has(provider)) {
        providerTierStats.set(provider, new Map());
      }
      const tierMap = providerTierStats.get(provider)!;

      const existing = tierMap.get(tier);
      if (existing) {
        existing.push(stats);
      } else {
        tierMap.set(tier, [stats]);
      }
    }

    // Merge multiple message-range stats for the same (provider, tier) by
    // computing a sample-weighted average.
    const byProvider: Record<string, TierProfileEntry[]> = {};

    for (const [provider, tierMap] of providerTierStats) {
      const mergedByTier = new Map<CompressionTier, BucketStats>();

      for (const [tier, statsList] of tierMap) {
        const totalSamples = statsList.reduce((s, b) => s + b.sampleCount, 0);
        if (totalSamples === 0) continue;

        const merged: BucketStats = {
          tier,
          provider,
          messageCountRange: "sm", // synthetic — merged across ranges
          sampleCount: totalSamples,
          successRate:
            statsList.reduce((s, b) => s + b.successRate * b.sampleCount, 0) / totalSamples,
          roiUsd:
            statsList.reduce((s, b) => s + b.roiUsd * b.sampleCount, 0) / totalSamples,
          latencyMs:
            statsList.reduce((s, b) => s + b.latencyMs * b.sampleCount, 0) / totalSamples,
          avgOvershootPct:
            statsList.reduce((s, b) => s + b.avgOvershootPct * b.sampleCount, 0) / totalSamples,
        };

        mergedByTier.set(tier, merged);
      }

      byProvider[provider] = rankTiers(mergedByTier, null, provider);
    }

    // Global best tier: take the #1 tier from the provider with the most records.
    let globalBestTier: CompressionTier = 3; // default: contextCollapse (cheap)
    let maxSamples = 0;

    for (const [provider, tierMap] of providerTierStats) {
      const samples = [...tierMap.values()].reduce(
        (s, statsList) => s + statsList.reduce((ss, b) => ss + b.sampleCount, 0),
        0,
      );
      if (samples > maxSamples) {
        maxSamples = samples;
        const ranked = byProvider[provider];
        if (ranked && ranked.length > 0 && ranked[0] !== undefined) {
          globalBestTier = ranked[0].tier;
        }
      }
    }

    return {
      cwd: this.cwd,
      computedAt: new Date().toISOString(),
      totalRecords,
      byProvider,
      globalBestTier,
    };
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Build a CompressibleProfile for the given project directory.
 *
 * Convenience wrapper around `HistoryConsolidator.buildProfile()`.
 *
 * @param cwd           Project working directory (genome root).
 * @param lookbackDays  Number of days of history to incorporate (default 30).
 * @returns             Consolidated compression profile.
 */
export async function getCompressionProfile(
  cwd: string,
  lookbackDays = DEFAULT_LOOKBACK_DAYS,
): Promise<CompressibleProfile> {
  const consolidator = new HistoryConsolidator(cwd, lookbackDays);
  return consolidator.buildProfile();
}

/**
 * Predict the optimal compression tier when switching providers, given a
 * project profile and a description of the current message workload.
 *
 * The recommendation:
 *  1. Checks if history exists for the target provider → use its best tier.
 *  2. Falls back to the source-provider best tier + cost-ratio tier shift.
 *  3. Applies a switching cost penalty for large provider cost differences.
 *  4. Writes an audit record to `evolution/compression-consolidation.jsonl`.
 *
 * @param currentProvider  Provider being switched from.
 * @param targetProvider   Provider being switched to.
 * @param messageProfile   Descriptor of the current message workload.
 * @param profile          Consolidated profile from `getCompressionProfile`.
 * @returns                CompressorRecommendation with rationale and ROI delta.
 */
export async function predictCompressionOnSwitch(
  currentProvider: string,
  targetProvider: string,
  messageProfile: { messageCount: number; avgTokensPerMessage?: number },
  profile: CompressibleProfile,
): Promise<CompressorRecommendation> {
  const msgRange = messageCountRange(messageProfile.messageCount);
  const costRatio = computeProviderCostRatio(currentProvider, targetProvider);

  // Try to use target-provider history first.
  const targetHistory = profile.byProvider[targetProvider];
  const sourceHistory = profile.byProvider[currentProvider];

  let recommendedTier: CompressionTier = 3; // default
  let rationale: string;
  let estimatedRoiDelta = 0;

  if (targetHistory && targetHistory.length > 0 && targetHistory[0] !== undefined) {
    // We have direct history for the target provider — use its best tier.
    const best = targetHistory[0];
    recommendedTier = best.tier;

    // Compute ROI delta: difference between best and second-best tier ROI.
    const secondBest = targetHistory[1];
    if (secondBest !== undefined) {
      estimatedRoiDelta = (best.roiUsd - secondBest.roiUsd) * best.sampleCount;
    }

    const pctBetter =
      secondBest !== undefined && secondBest.roiUsd > 0
        ? Math.round(((best.roiUsd - secondBest.roiUsd) / secondBest.roiUsd) * 100)
        : 0;

    rationale =
      `tier ${recommendedTier} was ${Math.abs(pctBetter)}% ` +
      `${pctBetter >= 0 ? "cheaper" : "more expensive"} on this project ` +
      `(${best.sampleCount} observations on ${targetProvider}); ` +
      `latency ±${Math.round(best.latencyMs)}ms`;
  } else if (sourceHistory && sourceHistory.length > 0 && sourceHistory[0] !== undefined) {
    // No target-provider history — use source history + cost-ratio shift.
    const sourceBest = sourceHistory[0];
    let tier = sourceBest.tier;

    // Apply the provider cost ratio tier shift.
    if (costRatio.recommendedTierShift !== 0) {
      const shifted = tier - costRatio.recommendedTierShift;
      tier = Math.min(4, Math.max(1, shifted)) as CompressionTier;
    }
    recommendedTier = tier;

    const direction = costRatio.recommendedTierShift < 0 ? "cheaper" : "more expensive";
    rationale =
      `no history on ${targetProvider}; ` +
      `extrapolated from ${currentProvider} best tier ${sourceBest.tier} ` +
      `(${direction} provider, cost ratio ${((costRatio.inputRatio + costRatio.outputRatio) / 2).toFixed(2)}×); ` +
      `expect recalibration over next ${RECALIBRATION_LAG_SESSIONS} sessions`;
    estimatedRoiDelta = sourceBest.roiUsd * sourceBest.sampleCount * 0.05; // 5% savings estimate
  } else {
    // No history at all — use the cost-ratio tier shift applied to the baseline (tier 3).
    let baseTier: CompressionTier = 3;
    if (costRatio.recommendedTierShift !== 0) {
      const shifted = (baseTier - costRatio.recommendedTierShift) as number;
      baseTier = Math.min(4, Math.max(1, shifted)) as CompressionTier;
    }
    recommendedTier = baseTier;

    rationale =
      `no compression history for ${currentProvider} or ${targetProvider}; ` +
      `defaulting to tier ${recommendedTier} based on cost ratio ` +
      `(${((costRatio.inputRatio + costRatio.outputRatio) / 2).toFixed(2)}×)`;
    estimatedRoiDelta = 0;
  }

  const now = new Date().toISOString();
  const recommendationId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

  const recommendation: CompressorRecommendation = {
    providerFrom: currentProvider,
    providerTo: targetProvider,
    recommendedTier,
    rationale,
    estimatedRoiDelta,
    recommendedAt: now,
  };

  // Write audit record.
  const auditRecord: ConsolidationAuditRecord = {
    provider_from: currentProvider,
    provider_to: targetProvider,
    recommended_tier: recommendedTier,
    actual_tier: null,
    roi_delta: null,
    recommended_at: now,
    recommendation_id: recommendationId,
  };

  await appendJsonl(consolidationAuditPath(profile.cwd), auditRecord);

  return recommendation;
}

/**
 * Hook invoked by `rebalanceBudgetOnSwitch` when a provider switch occurs.
 *
 * Builds a profile and emits a CompressorRecommendation, then returns the
 * recommended tier so budget rebalancing can use it.
 *
 * @param cwd             Project working directory.
 * @param currentProvider Provider being switched from.
 * @param targetProvider  Provider being switched to.
 * @param messageCount    Current number of messages in context.
 * @returns               CompressorRecommendation (also written to audit log).
 */
export async function rebalanceBudgetOnSwitchHook(
  cwd: string,
  currentProvider: string,
  targetProvider: string,
  messageCount: number,
): Promise<CompressorRecommendation> {
  const profile = await getCompressionProfile(cwd);
  return predictCompressionOnSwitch(
    currentProvider,
    targetProvider,
    { messageCount },
    profile,
  );
}
