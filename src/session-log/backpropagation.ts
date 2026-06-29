/**
 * Streaming Session Cost Attribution with Provider-Aware Backpropagation.
 *
 * Extends session-log cost accounting to attribute LLM token/USD costs back
 * through the compression tier that enabled the savings.
 *
 * ### Architecture
 *
 * 1. **BackpropagationEngine** — tracks `(compression_tier, message_count_bin,
 *    provider)` tuples and records the actual vs estimated token delta post-call.
 *    Operates over a rolling 30-turn window per (tier, bucket, provider) key.
 *
 * 2. **Per-provider, per-tier ROI metrics** — USD saved per tier application,
 *    aggregated via rolling 30-turn window. Surfaces as `ProviderTierROI` rows
 *    that the adaptive selector can consume.
 *
 * 3. **Adaptive bias integration** — `biasAdaptiveTierSelection` adjusts the
 *    recommended tier from `selectCompressionTierAdaptive` toward tiers with
 *    historically high ROI for the current (provider, bucket) combination.
 *
 * 4. **Pareto frontier** — `computeParetoFrontier` identifies non-dominated
 *    tier/provider combos across the (ROI, latency) objective space so callers
 *    can make multi-objective trade-offs.
 *
 * ### Export surface
 *
 * The `/backpropagation` export path (re-exported from session-log/index.ts)
 * yields ROI dashboards per tier per provider via `getROIDashboard()`.
 *
 * ### Concurrency safety
 *
 * All writes to shared state use a lightweight in-process mutex so concurrent
 * async callers on the same event loop don't produce torn reads. Filesystem
 * persistence uses POSIX O_APPEND (atomic for lines ≤ PIPE_BUF = 4KB).
 */

import { appendFileSync, existsSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

import type { CompressionTier } from "../compression/context.ts";
import { PROVIDER_RATES, type ProviderName, defaultProviderRate } from "../tokens/index.ts";
import { bucketMessageCount, type MessageCountBucket } from "../budget/index.ts";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/**
 * A single attribution record linking one LLM call back to the compression
 * tier that shaped its token budget.
 *
 * Stored in the rolling window and aggregated into ROI metrics.
 */
export interface BackpropAttributionRecord {
  /** Wall-clock timestamp (ISO-8601). */
  recordedAt: string;
  /** Which compression tier was applied before the call. */
  tier: CompressionTier;
  /** Message-count bucket at call time (xs/sm/md/lg). */
  bucket: MessageCountBucket;
  /** Provider/model string (e.g. "claude-3-5-sonnet"). */
  provider: string;
  /**
   * Estimated tokens before the tier fired.
   * Used as the counterfactual "what we would have sent".
   */
  estimatedTokens: number;
  /**
   * Actual tokens observed in the LLM response after tier applied.
   * Delta from estimated = tokens saved by compression.
   */
  actualTokens: number;
  /**
   * Token delta: estimatedTokens − actualTokens.
   * Positive = tier removed tokens (saved money).
   * Negative = tier added overhead (rare — e.g. tier 1 summary).
   */
  tokenDelta: number;
  /**
   * USD cost actually incurred for this call.
   */
  actualCostUsd: number;
  /**
   * Counterfactual USD cost — what the call would have cost without compression.
   * = actualCostUsd + tokenDelta × provider.inputRate
   */
  counterfactualCostUsd: number;
  /**
   * Net USD saved by the tier for this call.
   * = counterfactualCostUsd − actualCostUsd
   * Positive = tier saved money.
   */
  usdSaved: number;
  /**
   * Round-trip latency of the LLM call (ms).
   * Used in the Pareto frontier (ROI vs latency trade-off).
   */
  latencyMs: number;
  /**
   * Wall-clock ms at which the LLM request was dispatched (Date.now() value).
   * Optional — populated when the caller has request-start instrumentation.
   */
  requestStartMs?: number;
  /**
   * Wall-clock ms at which the first streamed token was received.
   * Optional — populated when the caller streams responses and can observe
   * time-to-first-token. Enables TTFT analysis separate from total latency.
   */
  firstTokenMs?: number;
}

/**
 * Aggregated ROI metrics for a (tier, provider) pair over the rolling window.
 */
export interface ProviderTierROI {
  /** Compression tier. */
  tier: CompressionTier;
  /** Provider/model string. */
  provider: string;
  /** Message-count bucket this ROI applies to. */
  bucket: MessageCountBucket;
  /** Number of attribution records in the rolling window. */
  sampleCount: number;
  /** Total tokens saved across all calls in the window. */
  totalTokensSaved: number;
  /** Total USD saved across all calls in the window. */
  totalUsdSaved: number;
  /** Average USD saved per tier application (usdSaved / sampleCount). */
  avgUsdSavedPerCall: number;
  /** Average latency of LLM calls after this tier was applied (ms). */
  avgLatencyMs: number;
  /**
   * ROI efficiency: totalUsdSaved / (max(1, sampleCount) × estimated tier cost).
   * A unitless ratio — higher is better. Used for bias and Pareto ranking.
   */
  roiEfficiency: number;
  /** ISO-8601 timestamp of when this ROI was last updated. */
  updatedAt: string;
}

/**
 * A non-dominated point on the Pareto frontier of (ROI, latency).
 *
 * A tier/provider combo is Pareto-optimal if no other combo has both higher
 * ROI *and* lower average latency. Callers can use the frontier to make
 * explicit trade-offs (e.g. "accept 5% lower ROI for 20% lower latency").
 */
export interface ParetoPoint {
  tier: CompressionTier;
  provider: string;
  bucket: MessageCountBucket;
  /** Total USD saved per call (ROI objective — maximize). */
  avgUsdSavedPerCall: number;
  /** Average LLM latency after tier applied (ms, latency objective — minimize). */
  avgLatencyMs: number;
  /** roiEfficiency score used for inter-point comparison. */
  roiEfficiency: number;
  /** Sample count backing this point. */
  sampleCount: number;
  /**
   * True when no other ParetoPoint dominates this one.
   * A point P dominates Q when P.avgUsdSavedPerCall >= Q.avgUsdSavedPerCall
   * AND P.avgLatencyMs <= Q.avgLatencyMs (with at least one strict inequality).
   */
  isNonDominated: boolean;
}

/**
 * Full ROI dashboard returned by `getROIDashboard`.
 *
 * Indexed by provider → tier → bucket for O(1) lookup.
 */
export interface ROIDashboard {
  /**
   * All ROI entries, keyed by `"${provider}:${tier}:${bucket}"`.
   * Iterate with `Object.values(entries)` for aggregations.
   */
  entries: Record<string, ProviderTierROI>;
  /** Pareto-optimal tier/provider combinations across (ROI, latency). */
  paretoFrontier: ParetoPoint[];
  /**
   * For each provider, the tier with the highest `avgUsdSavedPerCall`
   * across all buckets (best overall ROI).
   */
  bestTierByProvider: Record<string, CompressionTier>;
  /**
   * For each tier, the provider with the highest `avgUsdSavedPerCall`
   * across all buckets (best overall ROI).
   */
  bestProviderByTier: Record<CompressionTier, string>;
  /** ISO-8601 timestamp when this dashboard was generated. */
  generatedAt: string;
  /** Total number of attribution records in the engine's rolling window. */
  totalRecords: number;
}

/**
 * Result of biasing adaptive tier selection with historical ROI.
 */
export interface BiasedTierSelection {
  /** The original tier from `selectCompressionTierAdaptive`. */
  baseTier: CompressionTier;
  /** The tier after applying ROI-based bias (may equal baseTier). */
  biasedTier: CompressionTier;
  /**
   * Whether the bias changed the tier recommendation.
   * False when history is insufficient or ROI signal is too weak.
   */
  biasApplied: boolean;
  /**
   * ROI entry that drove the bias decision, if any.
   * Null when no bias was applied (e.g. insufficient history).
   */
  drivingROI: ProviderTierROI | null;
  /** ISO-8601 timestamp. */
  decidedAt: string;
}

// ---------------------------------------------------------------------------
// Internal key helpers
// ---------------------------------------------------------------------------

function roiKey(tier: CompressionTier, provider: string, bucket: MessageCountBucket): string {
  return `${provider}:${tier}:${bucket}`;
}

// ---------------------------------------------------------------------------
// Rolling-window configuration
// ---------------------------------------------------------------------------

/** Rolling window depth (number of attribution records per key). */
export const BACKPROP_WINDOW_SIZE = 30;

/**
 * Minimum ROI efficiency ratio before the backpropagation engine will bias
 * tier selection toward a historically high-ROI tier.
 * 0.05 = a tier must save at least 5% more than its estimated cost.
 */
export const BACKPROP_MIN_ROI_EFFICIENCY = 0.05;

/**
 * Minimum sample count before an ROI entry is trusted for bias decisions.
 */
export const BACKPROP_MIN_SAMPLES = 3;

// ---------------------------------------------------------------------------
// Backpropagation Engine
// ---------------------------------------------------------------------------

/**
 * Core backpropagation engine.
 *
 * Tracks `(tier, bucket, provider)` tuples. After each LLM call, records the
 * actual vs estimated token delta so USD savings can be attributed back to the
 * specific compression tier that fired.
 *
 * Maintains per-key rolling windows of up to `BACKPROP_WINDOW_SIZE` records
 * and derives per-provider, per-tier ROI metrics on demand.
 *
 * ### Thread safety
 *
 * All mutations go through `_applyRecord` which is synchronous and operates
 * on the module-level `_store`. The async `record()` wrapper serializes
 * concurrent callers via a simple promise chain so event-loop interleaving
 * cannot produce torn state.
 */
export class BackpropagationEngine {
  /** Rolling window store: key → attribution records (capped at BACKPROP_WINDOW_SIZE). */
  private readonly _store: Map<string, BackpropAttributionRecord[]> = new Map();

  /** Serialization lock for concurrent async callers. */
  private _writeChain: Promise<void> = Promise.resolve();

  /**
   * Record one LLM call's attribution data.
   *
   * Safe to call concurrently — writes are serialized internally via a
   * promise chain so no record is lost or partially applied.
   *
   * @param tier             Compression tier applied before this call.
   * @param messageCount     Number of messages at call time (used for bucket).
   * @param provider         Provider/model string.
   * @param estimatedTokens  Tokens estimated before the call.
   * @param actualTokens     Tokens actually sent (post-compression, per API).
   * @param actualCostUsd    Actual USD cost reported by the API for this call.
   * @param latencyMs        Round-trip latency of the LLM call in ms.
   */
  async record(
    tier: CompressionTier,
    messageCount: number,
    provider: string,
    estimatedTokens: number,
    actualTokens: number,
    actualCostUsd: number,
    latencyMs: number,
  ): Promise<BackpropAttributionRecord> {
    let result!: BackpropAttributionRecord;
    // Serialize: chain onto the existing write promise so concurrent callers
    // never interleave their mutations.
    this._writeChain = this._writeChain.then(() => {
      result = this._applyRecord(
        tier,
        messageCount,
        provider,
        estimatedTokens,
        actualTokens,
        actualCostUsd,
        latencyMs,
      );
    });
    await this._writeChain;
    return result;
  }

  /**
   * Synchronous core: build + store the attribution record.
   * Must only be called from within the serialized `_writeChain`.
   */
  private _applyRecord(
    tier: CompressionTier,
    messageCount: number,
    provider: string,
    estimatedTokens: number,
    actualTokens: number,
    actualCostUsd: number,
    latencyMs: number,
  ): BackpropAttributionRecord {
    const bucket = bucketMessageCount(messageCount);
    const rate = defaultProviderRate(provider);

    const tokenDelta = estimatedTokens - actualTokens;
    const counterfactualCostUsd = actualCostUsd + tokenDelta * rate.inputRate;
    const usdSaved = counterfactualCostUsd - actualCostUsd;

    const rec: BackpropAttributionRecord = {
      recordedAt: new Date().toISOString(),
      tier,
      bucket,
      provider,
      estimatedTokens,
      actualTokens,
      tokenDelta,
      actualCostUsd,
      counterfactualCostUsd,
      usdSaved,
      latencyMs,
    };

    const key = roiKey(tier, provider, bucket);
    const existing = this._store.get(key) ?? [];
    existing.push(rec);
    // Enforce rolling window: keep only the most recent BACKPROP_WINDOW_SIZE records.
    if (existing.length > BACKPROP_WINDOW_SIZE) {
      existing.splice(0, existing.length - BACKPROP_WINDOW_SIZE);
    }
    this._store.set(key, existing);

    return rec;
  }

  /**
   * Compute aggregated ROI metrics for a specific (tier, provider, bucket) key.
   *
   * Returns `null` when no records exist for that key.
   */
  computeROI(
    tier: CompressionTier,
    provider: string,
    bucket: MessageCountBucket,
  ): ProviderTierROI | null {
    const key = roiKey(tier, provider, bucket);
    const records = this._store.get(key);
    if (!records || records.length === 0) return null;

    const sampleCount = records.length;
    const totalTokensSaved = records.reduce((s, r) => s + Math.max(0, r.tokenDelta), 0);
    const totalUsdSaved = records.reduce((s, r) => s + r.usdSaved, 0);
    const avgUsdSavedPerCall = totalUsdSaved / sampleCount;
    const avgLatencyMs =
      records.reduce((s, r) => s + r.latencyMs, 0) / sampleCount;

    // ROI efficiency: ratio of average USD saved to estimated tier application cost.
    // Tier application cost estimated as 2000 tokens × inputRate × tier overhead factor.
    const rate = defaultProviderRate(provider);
    const tierOverhead = tier === 1 ? 0.1 : tier === 2 ? 0.01 : tier === 3 ? 0.005 : 0.003;
    const estimatedTierCost = 2000 * rate.inputRate * tierOverhead;
    const roiEfficiency =
      estimatedTierCost > 0 ? avgUsdSavedPerCall / estimatedTierCost : 0;

    return {
      tier,
      provider,
      bucket,
      sampleCount,
      totalTokensSaved,
      totalUsdSaved,
      avgUsdSavedPerCall,
      avgLatencyMs,
      roiEfficiency,
      updatedAt: new Date().toISOString(),
    };
  }

  /**
   * Compute ROI metrics for all (tier, provider, bucket) combinations that
   * have at least one record in the rolling window.
   *
   * Returns an array of `ProviderTierROI` entries, suitable for dashboard
   * rendering or Pareto frontier computation.
   */
  computeAllROI(): ProviderTierROI[] {
    const results: ProviderTierROI[] = [];
    for (const [key] of this._store) {
      const parts = key.split(":");
      // key format: "${provider}:${tier}:${bucket}"
      // provider can contain colons (e.g. "anthropic:claude-3-5-sonnet") — not
      // in practice with current provider names, but we parse safely.
      // Format is always <provider>:<tier number>:<bucket string>
      // Bucket is always one of: xs, sm, md, lg (no colons)
      // Tier is always a single digit
      // So split from the right: last element is bucket, second-to-last is tier
      if (parts.length < 3) continue;
      const bucket = parts[parts.length - 1] as MessageCountBucket;
      const tierStr = parts[parts.length - 2];
      const provider = parts.slice(0, parts.length - 2).join(":");
      const tier = parseInt(tierStr ?? "0", 10) as CompressionTier;
      if (tier < 1 || tier > 5) continue;

      const roi = this.computeROI(tier, provider, bucket);
      if (roi) results.push(roi);
    }
    return results;
  }

  /**
   * Return the raw attribution records for a (tier, provider, bucket) key.
   * Returns an empty array when no records exist for that key.
   */
  getRecords(
    tier: CompressionTier,
    provider: string,
    bucket: MessageCountBucket,
  ): readonly BackpropAttributionRecord[] {
    return this._store.get(roiKey(tier, provider, bucket)) ?? [];
  }

  /**
   * Return the total number of attribution records across all keys.
   */
  totalRecordCount(): number {
    let total = 0;
    for (const [, records] of this._store) total += records.length;
    return total;
  }

  /**
   * Reset all state. For tests only.
   */
  reset(): void {
    this._store.clear();
    this._writeChain = Promise.resolve();
  }
}

// ---------------------------------------------------------------------------
// Module-level singleton engine (for in-process session use)
// ---------------------------------------------------------------------------

/** The shared in-process engine instance. */
const _engine = new BackpropagationEngine();

/** Expose the engine instance for advanced callers and tests. */
export function getBackpropEngine(): BackpropagationEngine {
  return _engine;
}

/** Reset the module-level engine. For tests only. */
export function _resetBackpropEngine(): void {
  _engine.reset();
}

// ---------------------------------------------------------------------------
// Pareto frontier computation
// ---------------------------------------------------------------------------

/**
 * Compute the Pareto frontier over the (ROI, latency) objective space.
 *
 * A tier/provider combo is Pareto-optimal (non-dominated) when no other combo
 * has both higher `avgUsdSavedPerCall` AND lower `avgLatencyMs`.
 *
 * Only combos with `sampleCount >= BACKPROP_MIN_SAMPLES` are considered so
 * sparse buckets don't pollute the frontier.
 *
 * @param roiEntries  Per-tier/provider ROI entries (from `computeAllROI`).
 * @returns           Array of `ParetoPoint` objects, each flagged with
 *                    `isNonDominated`. Points are sorted descending by
 *                    `avgUsdSavedPerCall` for easy ranking.
 */
export function computeParetoFrontier(roiEntries: ProviderTierROI[]): ParetoPoint[] {
  // Filter to entries with sufficient samples.
  const candidates = roiEntries.filter((e) => e.sampleCount >= BACKPROP_MIN_SAMPLES);

  // Convert to ParetoPoint shapes.
  const points: ParetoPoint[] = candidates.map((e) => ({
    tier: e.tier,
    provider: e.provider,
    bucket: e.bucket,
    avgUsdSavedPerCall: e.avgUsdSavedPerCall,
    avgLatencyMs: e.avgLatencyMs,
    roiEfficiency: e.roiEfficiency,
    sampleCount: e.sampleCount,
    isNonDominated: false, // computed below
  }));

  // Mark non-dominated points.
  // P dominates Q when:
  //   P.avgUsdSavedPerCall >= Q.avgUsdSavedPerCall AND
  //   P.avgLatencyMs <= Q.avgLatencyMs
  //   (with at least one strict inequality)
  for (let i = 0; i < points.length; i++) {
    const p = points[i]!;
    let dominated = false;
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      const q = points[j]!;
      const qDominatesP =
        q.avgUsdSavedPerCall >= p.avgUsdSavedPerCall &&
        q.avgLatencyMs <= p.avgLatencyMs &&
        (q.avgUsdSavedPerCall > p.avgUsdSavedPerCall ||
          q.avgLatencyMs < p.avgLatencyMs);
      if (qDominatesP) {
        dominated = true;
        break;
      }
    }
    p.isNonDominated = !dominated;
  }

  // Sort descending by avgUsdSavedPerCall for ranking.
  points.sort((a, b) => b.avgUsdSavedPerCall - a.avgUsdSavedPerCall);
  return points;
}

// ---------------------------------------------------------------------------
// ROI Dashboard
// ---------------------------------------------------------------------------

/**
 * Generate the full ROI dashboard from the current engine state.
 *
 * Aggregates all rolling-window records into per-provider, per-tier ROI
 * metrics and computes the Pareto frontier across (ROI, latency).
 *
 * @param engine  Optional engine instance. Defaults to the module singleton.
 * @returns       A `ROIDashboard` ready for rendering or export.
 */
export function getROIDashboard(engine?: BackpropagationEngine): ROIDashboard {
  const eng = engine ?? _engine;
  const allROI = eng.computeAllROI();

  // Build entries record.
  const entries: Record<string, ProviderTierROI> = {};
  for (const roi of allROI) {
    entries[roiKey(roi.tier, roi.provider, roi.bucket)] = roi;
  }

  // Pareto frontier.
  const paretoFrontier = computeParetoFrontier(allROI);

  // Best tier per provider (highest avgUsdSavedPerCall across all buckets).
  const bestTierByProvider: Record<string, CompressionTier> = {};
  for (const roi of allROI) {
    const existing = bestTierByProvider[roi.provider];
    if (existing === undefined) {
      bestTierByProvider[roi.provider] = roi.tier;
    } else {
      const existingROI = entries[roiKey(existing, roi.provider, roi.bucket)];
      if (
        existingROI === undefined ||
        roi.avgUsdSavedPerCall > existingROI.avgUsdSavedPerCall
      ) {
        bestTierByProvider[roi.provider] = roi.tier;
      }
    }
  }

  // Best provider per tier (highest avgUsdSavedPerCall across all buckets).
  const bestProviderByTier: Record<CompressionTier, string> = {} as Record<
    CompressionTier,
    string
  >;
  for (const roi of allROI) {
    const existing = bestProviderByTier[roi.tier];
    if (existing === undefined) {
      bestProviderByTier[roi.tier] = roi.provider;
    } else {
      const existingROI = entries[roiKey(roi.tier, existing, roi.bucket)];
      if (
        existingROI === undefined ||
        roi.avgUsdSavedPerCall > existingROI.avgUsdSavedPerCall
      ) {
        bestProviderByTier[roi.tier] = roi.provider;
      }
    }
  }

  return {
    entries,
    paretoFrontier,
    bestTierByProvider,
    bestProviderByTier,
    generatedAt: new Date().toISOString(),
    totalRecords: eng.totalRecordCount(),
  };
}

// ---------------------------------------------------------------------------
// Adaptive bias integration
// ---------------------------------------------------------------------------

/**
 * Apply ROI-based backpropagation bias to an adaptively-selected tier.
 *
 * After `selectCompressionTierAdaptive` produces a `baseTier`, this function
 * checks whether a historically higher-ROI tier is available for the current
 * (provider, messageCount) combination. If so, and the ROI signal is strong
 * enough (>= `BACKPROP_MIN_ROI_EFFICIENCY` and >= `BACKPROP_MIN_SAMPLES`
 * records), the recommendation is shifted to the highest-ROI tier.
 *
 * The bias is conservative: it only shifts AWAY from tier 1 (most aggressive /
 * most expensive LLM call) toward cheaper tiers when history clearly supports
 * it. It will not shift toward tier 1 — that requires explicit thresholds.
 *
 * @param baseTier      The tier selected by `selectCompressionTierAdaptive`.
 * @param provider      Current provider/model string.
 * @param messageCount  Current message count (determines bucket).
 * @param engine        Optional engine instance. Defaults to module singleton.
 * @returns             A `BiasedTierSelection` indicating the final tier and why.
 */
export function biasAdaptiveTierSelection(
  baseTier: CompressionTier,
  provider: string,
  messageCount: number,
  engine?: BackpropagationEngine,
): BiasedTierSelection {
  const eng = engine ?? _engine;
  const bucket = bucketMessageCount(messageCount);

  // Collect all ROI entries for this provider+bucket across all tiers.
  const candidates: ProviderTierROI[] = [];
  for (const tier of [2, 3, 4] as CompressionTier[]) {
    const roi = eng.computeROI(tier, provider, bucket);
    if (
      roi !== null &&
      roi.sampleCount >= BACKPROP_MIN_SAMPLES &&
      roi.roiEfficiency >= BACKPROP_MIN_ROI_EFFICIENCY
    ) {
      candidates.push(roi);
    }
  }

  if (candidates.length === 0) {
    return {
      baseTier,
      biasedTier: baseTier,
      biasApplied: false,
      drivingROI: null,
      decidedAt: new Date().toISOString(),
    };
  }

  // Pick the tier with the highest ROI efficiency among candidates.
  // If it is lighter than baseTier (higher number = less aggressive), prefer it.
  candidates.sort((a, b) => b.roiEfficiency - a.roiEfficiency);
  const best = candidates[0]!;

  // Only bias toward lighter tiers (higher number = less aggressive/cheaper).
  // Never bias toward tier 1 via this path — tier 1 has its own LLM cost model.
  if (best.tier >= baseTier) {
    return {
      baseTier,
      biasedTier: baseTier,
      biasApplied: false,
      drivingROI: best,
      decidedAt: new Date().toISOString(),
    };
  }

  return {
    baseTier,
    biasedTier: best.tier,
    biasApplied: true,
    drivingROI: best,
    decidedAt: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Filesystem persistence (optional, for cross-session durability)
// ---------------------------------------------------------------------------

/** Resolved path for backpropagation records on disk. */
function resolveBackpropPath(): string {
  const override = process.env.ASHLR_BACKPROP_PATH;
  if (override && override.length > 0) return override;
  return join(homedir(), ".ashlr", "backprop-records.jsonl");
}

function ensureBackpropDir(path: string): void {
  const dir = dirname(path);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

/**
 * Persist one `BackpropAttributionRecord` to disk (best-effort, JSONL).
 *
 * Set `ASHLR_BACKPROP_PATH` to override the location (useful in tests).
 * Set `ASHLR_SESSION_LOG=0` to disable persistence entirely.
 */
export function persistBackpropRecord(rec: BackpropAttributionRecord): void {
  if (process.env.ASHLR_SESSION_LOG === "0") return;
  try {
    const path = resolveBackpropPath();
    ensureBackpropDir(path);
    appendFileSync(path, JSON.stringify(rec) + "\n", { flag: "a" });
  } catch {
    // Best-effort — never throw from a logging path.
  }
}

// ---------------------------------------------------------------------------
// Streaming attribution helper (record + persist in one call)
// ---------------------------------------------------------------------------

/**
 * Record one attribution in the engine AND persist it to disk.
 *
 * This is the primary call site for streaming session attribution. Call it
 * after each LLM response when the actual token count is known.
 *
 * @param tier             Compression tier that fired before the call.
 * @param messageCount     Message count at call time.
 * @param provider         Provider/model string.
 * @param estimatedTokens  Tokens estimated before compression.
 * @param actualTokens     Tokens actually used (from API response).
 * @param actualCostUsd    Actual USD cost (from API response).
 * @param latencyMs        Round-trip LLM call latency in ms.
 * @param engine           Optional engine. Defaults to module singleton.
 * @returns                The attributed record.
 */
export async function attributeSessionCost(
  tier: CompressionTier,
  messageCount: number,
  provider: string,
  estimatedTokens: number,
  actualTokens: number,
  actualCostUsd: number,
  latencyMs: number,
  engine?: BackpropagationEngine,
): Promise<BackpropAttributionRecord> {
  const eng = engine ?? _engine;
  const rec = await eng.record(
    tier,
    messageCount,
    provider,
    estimatedTokens,
    actualTokens,
    actualCostUsd,
    latencyMs,
  );
  persistBackpropRecord(rec);
  return rec;
}

// ---------------------------------------------------------------------------
// Latency dashboard
// ---------------------------------------------------------------------------

/**
 * Per-tier latency summary for one model, returned by `getLatencyDashboard`.
 */
export interface TierLatencySummary {
  /** Compression tier (1–4). */
  tier: CompressionTier;
  /** Number of records in the rolling window for this tier + model. */
  sampleCount: number;
  /** Mean LLM round-trip latency after this tier was applied (ms). */
  avgLatencyMs: number;
  /** Minimum observed latency (ms). */
  minLatencyMs: number;
  /** Maximum observed latency (ms). */
  maxLatencyMs: number;
  /** Mean time-to-first-token when requestStartMs + firstTokenMs are available (ms). */
  avgTtftMs: number | null;
  /** Mean USD cost per call. */
  avgCostUsd: number;
  /** Mean USD saved per call vs no compression. */
  avgUsdSaved: number;
}

/**
 * Latency dashboard for a single model — mirrors `ROIDashboard` but adds
 * latency-specific statistics per tier.
 */
export interface ModelLatencyDashboard {
  /** Model/provider slug. */
  model: string;
  /** Per-tier latency summaries (tiers with ≥ 1 record). */
  tiers: TierLatencySummary[];
  /** Tier with the lowest avgLatencyMs (among tiers with ≥ BACKPROP_MIN_SAMPLES records). */
  fastestTier: CompressionTier | null;
  /** Tier with the highest avgUsdSaved (among tiers with ≥ BACKPROP_MIN_SAMPLES records). */
  cheapestTier: CompressionTier | null;
  /** ISO-8601 timestamp. */
  generatedAt: string;
}

/**
 * Return a latency-focused dashboard for a specific model.
 *
 * Aggregates all BackpropAttributionRecord entries across tiers and buckets
 * for the given model slug into per-tier latency + cost summaries. Mirrors
 * `getROIDashboard` but surfaces latency as the primary metric instead of ROI.
 *
 * @param model   Model/provider slug (e.g. "claude-3-5-sonnet", "gpt-4o").
 * @param engine  Optional engine instance. Defaults to module singleton.
 * @returns       A `ModelLatencyDashboard` ready for rendering or export.
 */
export function getLatencyDashboard(
  model: string,
  engine?: BackpropagationEngine,
): ModelLatencyDashboard {
  const eng = engine ?? _engine;
  const buckets = ["xs", "sm", "md", "lg"] as const;
  const tiers: TierLatencySummary[] = [];

  for (const tier of [1, 2, 3, 4] as CompressionTier[]) {
    // Collect records across all buckets for this (tier, model) pair.
    const allRecords: BackpropAttributionRecord[] = [];
    for (const bucket of buckets) {
      const recs = eng.getRecords(tier, model, bucket);
      allRecords.push(...recs);
    }
    if (allRecords.length === 0) continue;

    const latencies = allRecords.map((r) => r.latencyMs);
    const avgLatencyMs = latencies.reduce((s, v) => s + v, 0) / latencies.length;
    const minLatencyMs = Math.min(...latencies);
    const maxLatencyMs = Math.max(...latencies);

    // TTFT: only computable for records that carry both requestStartMs and firstTokenMs.
    const ttftRecords = allRecords.filter(
      (r) => r.requestStartMs !== undefined && r.firstTokenMs !== undefined,
    );
    const avgTtftMs =
      ttftRecords.length > 0
        ? ttftRecords.reduce((s, r) => s + (r.firstTokenMs! - r.requestStartMs!), 0) /
          ttftRecords.length
        : null;

    const avgCostUsd =
      allRecords.reduce((s, r) => s + r.actualCostUsd, 0) / allRecords.length;
    const avgUsdSaved =
      allRecords.reduce((s, r) => s + r.usdSaved, 0) / allRecords.length;

    tiers.push({
      tier,
      sampleCount: allRecords.length,
      avgLatencyMs,
      minLatencyMs,
      maxLatencyMs,
      avgTtftMs,
      avgCostUsd,
      avgUsdSaved,
    });
  }

  // Determine fastest + cheapest tiers (require BACKPROP_MIN_SAMPLES).
  const eligible = tiers.filter((t) => t.sampleCount >= BACKPROP_MIN_SAMPLES);

  const fastestTier =
    eligible.length > 0
      ? eligible.reduce((best, t) => (t.avgLatencyMs < best.avgLatencyMs ? t : best)).tier
      : null;

  const cheapestTier =
    eligible.length > 0
      ? eligible.reduce((best, t) => (t.avgUsdSaved > best.avgUsdSaved ? t : best)).tier
      : null;

  return {
    model,
    tiers,
    fastestTier,
    cheapestTier,
    generatedAt: new Date().toISOString(),
  };
}
