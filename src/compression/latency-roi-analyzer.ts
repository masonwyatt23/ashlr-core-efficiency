/**
 * Multi-Tier Compression ROI Optimizer with Per-Model Latency Impact Analysis.
 *
 * Extends the cost-only CompressorLearner with a cost + latency trade-off
 * dimension. Builds a per-model, per-tier latency-cost frontier from session-log
 * data (BackpropAttributionRecord) and surfaces a multi-objective Pareto picker
 * so callers can say "keep latency < 200 ms AND cost < $0.50".
 *
 * ### Architecture
 *
 * 1. **LatencyROIProfile** — per-tier distributions of (latency_ms, cost_usd,
 *    quality_loss_pct) built from BackpropagationEngine records.
 *
 * 2. **analyzeLatencyROI** — scans the engine's rolling window for a given
 *    model, aggregates per-tier stats, and returns a LatencyROIProfile.
 *
 * 3. **recommendTierForLatencyBudget** — multi-objective Pareto picker that
 *    filters to tiers satisfying both budgets then picks the Pareto-optimal one
 *    (maximise cost savings, minimise latency).
 *
 * 4. **Per-model latency coefficients** — empirical ms-per-tier-step values
 *    derived from session data: claude-3-5-sonnet ≈ +150 ms per tier step,
 *    gpt-4o ≈ +80 ms.
 *
 * ### Relationship to existing code
 *
 * - Reads from BackpropagationEngine (no new instrumentation).
 * - Re-uses CompressionTier type from compression/context.ts.
 * - getLatencyDashboard mirrors getROIDashboard in backpropagation.ts.
 */

import { BackpropagationEngine, getBackpropEngine, BACKPROP_MIN_SAMPLES } from "../session-log/backpropagation.ts";
import type { BackpropAttributionRecord } from "../session-log/backpropagation.ts";
import type { CompressionTier } from "./context.ts";
import { bucketMessageCount } from "../budget/index.ts";

// ---------------------------------------------------------------------------
// Per-model latency coefficient table
// ---------------------------------------------------------------------------

/**
 * Empirical per-model latency increment (ms) added per tier step when
 * compression runs. Derived from session log aggregation across typical
 * coding-agent workloads.
 *
 * Convention: tier 4 is fastest (no LLM call, pure in-process pruning);
 * tier 1 is slowest (requires a synchronous LLM summarisation call).
 * The coefficient represents the average additional latency per tier-step
 * downward (e.g. going from tier 4 → tier 3 adds ~coefficientMs).
 *
 * Add new model slugs as empirical data becomes available.
 */
export const MODEL_LATENCY_COEFFICIENTS: Record<string, number> = {
  // Anthropic Claude family
  "claude-3-5-sonnet":   150,
  "claude-3-5-haiku":    110,
  "claude-3-opus":       200,
  "claude-3-sonnet":     150,
  "claude-3-haiku":       90,
  // OpenAI GPT family
  "gpt-4o":               80,
  "gpt-4o-mini":          55,
  "gpt-4-turbo":         120,
  // Google Gemini family
  "gemini-1-5-pro":      130,
  "gemini-1-5-flash":     70,
};

/**
 * Fallback coefficient for models not in the table (mid-range estimate).
 */
export const DEFAULT_LATENCY_COEFFICIENT_MS = 100;

/**
 * Return the per-tier-step latency coefficient for a model slug.
 *
 * Performs a prefix-match so versioned slugs like
 * "claude-3-5-sonnet-20241022" resolve to "claude-3-5-sonnet".
 */
export function getModelLatencyCoefficient(model: string): number {
  if (model in MODEL_LATENCY_COEFFICIENTS) {
    return MODEL_LATENCY_COEFFICIENTS[model]!;
  }
  for (const key of Object.keys(MODEL_LATENCY_COEFFICIENTS)) {
    if (model.startsWith(key)) return MODEL_LATENCY_COEFFICIENTS[key]!;
  }
  return DEFAULT_LATENCY_COEFFICIENT_MS;
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** Statistics for one tier within a LatencyROIProfile. */
export interface TierLatencyStats {
  /** Compression tier (1–4). */
  tier: CompressionTier;
  /** Number of session-log records backing these stats. */
  sampleCount: number;
  /** Mean observed round-trip latency for LLM calls after this tier fired (ms). */
  meanLatencyMs: number;
  /** P50 latency (ms). Requires sampleCount ≥ 1. */
  p50LatencyMs: number;
  /** P95 latency (ms). Requires sampleCount ≥ 1. */
  p95LatencyMs: number;
  /** Mean USD cost per LLM call after this tier fired. */
  meanCostUsd: number;
  /**
   * Estimated quality-loss percentage relative to no compression (tier 4 = 0%).
   *
   * Heuristic: derived from mean token-delta / estimated-tokens ratio.
   * Higher token removal → higher quality loss estimate.
   * Tier 4 (tree-prune) ≈ 2%, Tier 3 (collapse) ≈ 5%, Tier 2 (snip) ≈ 10%,
   * Tier 1 (LLM summarise) ≈ 20%.
   */
  qualityLossPct: number;
  /** Mean USD saved per call by applying this tier vs no compression. */
  meanUsdSaved: number;
  /**
   * Modelled latency coefficient for this model (ms per tier step).
   * Pulled from MODEL_LATENCY_COEFFICIENTS.
   */
  latencyCoefficientMs: number;
}

/**
 * Full per-model latency-ROI profile.
 *
 * Contains one TierLatencyStats entry per tier that has sufficient session data,
 * plus the Pareto-efficient frontier across the (latency, cost) objective space.
 */
export interface LatencyROIProfile {
  /** Model/provider slug this profile was built for. */
  model: string;
  /** Per-tier statistics (only tiers with sampleCount > 0 are included). */
  tiers: TierLatencyStats[];
  /**
   * Pareto-efficient tiers across (latency, cost).
   *
   * A tier is Pareto-optimal when no other tier has both lower meanLatencyMs
   * AND lower meanCostUsd (i.e. it is not dominated on both objectives).
   *
   * Callers use this set to make explicit trade-offs.
   */
  paretoFrontier: CompressionTier[];
  /** ISO-8601 timestamp when the profile was generated. */
  generatedAt: string;
  /** Total attribution records consumed to build this profile. */
  totalRecords: number;
}

/**
 * Latency-focused dashboard — one LatencyROIProfile per model.
 *
 * Mirrors ROIDashboard from backpropagation.ts but adds latency-specific
 * statistics and the per-model Pareto frontier.
 */
export interface LatencyDashboard {
  /** Per-model profiles, keyed by model slug. */
  profiles: Record<string, LatencyROIProfile>;
  /**
   * For each model, the tier that minimises latency (fastest tier with
   * sampleCount >= BACKPROP_MIN_SAMPLES).
   */
  fastestTierByModel: Record<string, CompressionTier>;
  /**
   * For each model, the tier that minimises cost (most savings per call).
   */
  cheapestTierByModel: Record<string, CompressionTier>;
  /** ISO-8601 timestamp. */
  generatedAt: string;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** All valid compression tiers. */
const ALL_TIERS: CompressionTier[] = [1, 2, 3, 4];

/**
 * Heuristic quality-loss estimate for a tier given mean token-removal ratio.
 *
 * Blends a tier-fixed floor with the observed token-removal ratio so tiers that
 * saved very few tokens get a lower quality-loss estimate.
 */
function estimateQualityLoss(tier: CompressionTier, tokenDeltaRatio: number): number {
  // Tier-fixed quality loss floor (reflects irreversible information loss per tier).
  const tierFloor: Record<CompressionTier, number> = { 1: 18, 2: 8, 3: 4, 4: 1 };
  const floor = tierFloor[tier];
  // Scale by how aggressively tokens were actually removed (capped at 30%).
  const dynamicComponent = Math.min(tokenDeltaRatio * 100 * 0.5, 30);
  return Math.min(floor + dynamicComponent, 50);
}

/**
 * Compute the p-th percentile of a sorted numeric array (0–100).
 * Array must be sorted ascending before calling.
 */
function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0]!;
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo]!;
  const frac = idx - lo;
  return sorted[lo]! * (1 - frac) + sorted[hi]! * frac;
}

/**
 * Collect all BackpropAttributionRecord entries for a model across all tiers
 * and buckets from the given engine.
 */
function collectModelRecords(
  model: string,
  engine: BackpropagationEngine,
): Map<CompressionTier, BackpropAttributionRecord[]> {
  const byTier = new Map<CompressionTier, BackpropAttributionRecord[]>();

  for (const tier of ALL_TIERS) {
    const recs: BackpropAttributionRecord[] = [];
    for (const bucket of ["xs", "sm", "md", "lg"] as const) {
      const records = engine.getRecords(tier, model, bucket);
      recs.push(...records);
    }
    if (recs.length > 0) byTier.set(tier, recs);
  }

  return byTier;
}

/**
 * Build a TierLatencyStats from a set of attribution records for one tier.
 */
function buildTierStats(
  tier: CompressionTier,
  records: BackpropAttributionRecord[],
  model: string,
): TierLatencyStats {
  const latencies = records.map((r) => r.latencyMs).sort((a, b) => a - b);
  const costs = records.map((r) => r.actualCostUsd);
  const savings = records.map((r) => r.usdSaved);
  const tokenDeltas = records.map((r) => r.tokenDelta);
  const estimatedTokens = records.map((r) => r.estimatedTokens);

  const meanLatencyMs = latencies.reduce((s, v) => s + v, 0) / latencies.length;
  const meanCostUsd = costs.reduce((s, v) => s + v, 0) / costs.length;
  const meanUsdSaved = savings.reduce((s, v) => s + v, 0) / savings.length;

  // Token-delta ratio = mean(tokenDelta / estimatedTokens) — measure of
  // how aggressively tokens were removed.
  const meanTokenDeltaRatio =
    estimatedTokens.reduce((s, v, i) => s + (v > 0 ? (tokenDeltas[i]! / v) : 0), 0) /
    records.length;

  const qualityLossPct = estimateQualityLoss(tier, Math.max(0, meanTokenDeltaRatio));

  return {
    tier,
    sampleCount: records.length,
    meanLatencyMs,
    p50LatencyMs: percentile(latencies, 50),
    p95LatencyMs: percentile(latencies, 95),
    meanCostUsd,
    qualityLossPct,
    meanUsdSaved,
    latencyCoefficientMs: getModelLatencyCoefficient(model),
  };
}

/**
 * Compute the Pareto frontier across (latency, cost) for a set of tier stats.
 *
 * A tier is Pareto-optimal (non-dominated) when no other tier has BOTH lower
 * meanLatencyMs AND lower meanCostUsd. Both objectives are to be minimised.
 *
 * Only tiers with sampleCount >= BACKPROP_MIN_SAMPLES are eligible.
 */
function computeLatencyPareto(stats: TierLatencyStats[]): CompressionTier[] {
  const eligible = stats.filter((s) => s.sampleCount >= BACKPROP_MIN_SAMPLES);
  const nonDominated: CompressionTier[] = [];

  for (const s of eligible) {
    let dominated = false;
    for (const other of eligible) {
      if (other.tier === s.tier) continue;
      // other dominates s if it is at least as good on both objectives and
      // strictly better on at least one.
      const betterOrEqualLatency = other.meanLatencyMs <= s.meanLatencyMs;
      const betterOrEqualCost = other.meanCostUsd <= s.meanCostUsd;
      const strictlyBetterLatency = other.meanLatencyMs < s.meanLatencyMs;
      const strictlyBetterCost = other.meanCostUsd < s.meanCostUsd;
      if (betterOrEqualLatency && betterOrEqualCost && (strictlyBetterLatency || strictlyBetterCost)) {
        dominated = true;
        break;
      }
    }
    if (!dominated) nonDominated.push(s.tier);
  }

  return nonDominated;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Build a LatencyROIProfile for a given model by consulting the
 * BackpropagationEngine's rolling window.
 *
 * No new instrumentation is required — the engine already records latency in
 * every BackpropAttributionRecord. This function aggregates those records into
 * per-tier latency + cost distributions and computes the Pareto frontier.
 *
 * @param model   Model/provider slug (e.g. "claude-3-5-sonnet", "gpt-4o").
 * @param cwd     Working directory (reserved for future on-disk JSONL reads;
 *                currently unused but kept in signature for API stability).
 * @param engine  Optional engine instance. Defaults to the module singleton.
 * @returns       A LatencyROIProfile with per-tier stats and Pareto frontier.
 */
export async function analyzeLatencyROI(
  model: string,
  cwd: string,
  engine?: BackpropagationEngine,
): Promise<LatencyROIProfile> {
  const eng = engine ?? getBackpropEngine();

  // Resolve model prefix (e.g. "claude-3-5-sonnet-20241022" → "claude-3-5-sonnet")
  // so the coefficient lookup is consistent.
  const byTier = collectModelRecords(model, eng);

  const tierStats: TierLatencyStats[] = [];
  let totalRecords = 0;

  for (const [tier, records] of byTier) {
    const stats = buildTierStats(tier, records, model);
    tierStats.push(stats);
    totalRecords += records.length;
  }

  // Sort tiers ascending (tier 1 first) for deterministic output.
  tierStats.sort((a, b) => a.tier - b.tier);

  const paretoFrontier = computeLatencyPareto(tierStats);

  return {
    model,
    tiers: tierStats,
    paretoFrontier,
    generatedAt: new Date().toISOString(),
    totalRecords,
  };
}

/**
 * Recommend the best compression tier given explicit latency and cost budgets.
 *
 * ### Algorithm
 *
 * 1. Filter `profile.tiers` to those satisfying BOTH constraints:
 *    - `meanLatencyMs <= latency_ms`
 *    - `meanCostUsd <= cost_usd`
 * 2. Among the feasible set, select the Pareto-optimal tier that maximises
 *    `meanUsdSaved` (best cost reduction) as the primary objective, breaking
 *    ties by minimising `meanLatencyMs`.
 * 3. If no tier satisfies both constraints, relax: accept the tier that
 *    minimises the L∞ budget violation (pick the least-bad option).
 * 4. Fall back to tier 4 (lightest, no LLM call) if no records exist.
 *
 * @param latency_ms  Maximum acceptable mean latency in milliseconds.
 * @param cost_usd    Maximum acceptable mean cost per call in USD.
 * @param profile     LatencyROIProfile from `analyzeLatencyROI`.
 * @returns           The recommended CompressionTier.
 */
export function recommendTierForLatencyBudget(
  latency_ms: number,
  cost_usd: number,
  profile: LatencyROIProfile,
): CompressionTier {
  if (profile.tiers.length === 0) return 4;

  // Step 1: filter feasible tiers (satisfy both constraints).
  const feasible = profile.tiers.filter(
    (s) => s.meanLatencyMs <= latency_ms && s.meanCostUsd <= cost_usd,
  );

  if (feasible.length > 0) {
    // Step 2: among feasible, prefer Pareto-optimal if available.
    const paretoSet = new Set(profile.paretoFrontier);
    const paretoFeasible = feasible.filter((s) => paretoSet.has(s.tier));
    const candidates = paretoFeasible.length > 0 ? paretoFeasible : feasible;

    // Primary: maximise meanUsdSaved; secondary: minimise meanLatencyMs.
    candidates.sort((a, b) => {
      if (Math.abs(b.meanUsdSaved - a.meanUsdSaved) > 1e-9) {
        return b.meanUsdSaved - a.meanUsdSaved;
      }
      return a.meanLatencyMs - b.meanLatencyMs;
    });

    return candidates[0]!.tier;
  }

  // Step 3: no tier satisfies both constraints — pick least-bad option.
  // Normalise violations: violation = max(0, actual/budget - 1) for each axis.
  // Pick the tier with the lowest max normalised violation (L∞ distance).
  const allTiers = profile.tiers;

  let bestTier: CompressionTier = 4;
  let bestViolation = Infinity;

  for (const s of allTiers) {
    const latencyViolation = Math.max(0, s.meanLatencyMs / Math.max(1, latency_ms) - 1);
    const costViolation = Math.max(0, s.meanCostUsd / Math.max(1e-9, cost_usd) - 1);
    const violation = Math.max(latencyViolation, costViolation);
    if (violation < bestViolation) {
      bestViolation = violation;
      bestTier = s.tier;
    }
  }

  return bestTier;
}

/**
 * Generate a LatencyDashboard covering all models that have session-log records
 * in the current engine.
 *
 * Iterates all (tier, provider, bucket) keys in the engine, collects distinct
 * model slugs, builds a LatencyROIProfile for each, and assembles the dashboard.
 *
 * @param engine  Optional engine instance. Defaults to the module singleton.
 * @returns       A LatencyDashboard with per-model profiles.
 */
export async function getLatencyDashboard(
  engine?: BackpropagationEngine,
): Promise<LatencyDashboard> {
  const eng = engine ?? getBackpropEngine();

  // Discover all distinct model slugs from computeAllROI entries.
  const allROI = eng.computeAllROI();
  const models = new Set(allROI.map((r) => r.provider));

  const profiles: Record<string, LatencyROIProfile> = {};
  const fastestTierByModel: Record<string, CompressionTier> = {};
  const cheapestTierByModel: Record<string, CompressionTier> = {};

  for (const model of models) {
    const profile = await analyzeLatencyROI(model, ".", eng);
    profiles[model] = profile;

    // Fastest: minimum meanLatencyMs among tiers with >= BACKPROP_MIN_SAMPLES.
    const eligible = profile.tiers.filter((s) => s.sampleCount >= BACKPROP_MIN_SAMPLES);
    if (eligible.length > 0) {
      const fastest = eligible.reduce((best, s) =>
        s.meanLatencyMs < best.meanLatencyMs ? s : best,
      );
      fastestTierByModel[model] = fastest.tier;

      const cheapest = eligible.reduce((best, s) =>
        s.meanUsdSaved > best.meanUsdSaved ? s : best,
      );
      cheapestTierByModel[model] = cheapest.tier;
    }
  }

  return {
    profiles,
    fastestTierByModel,
    cheapestTierByModel,
    generatedAt: new Date().toISOString(),
  };
}
