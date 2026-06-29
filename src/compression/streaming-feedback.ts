/**
 * Streaming Compression Feedback Channel
 *
 * Instruments LLM API streams in real-time to adjust compression tier
 * mid-session based on actual token burn observed from cache miss ratios and
 * latency markers.
 *
 * ### Design overview
 *
 * `StreamingCompressionMonitor` wraps `LLMSummarizer.stream()` without
 * altering the semantics of the underlying generator. It:
 *
 *   1. Tracks a per-block-type (text, tool_result, thinking, tool_use) token
 *      budget, comparing the estimate produced before the call to the real
 *      token count reported in the `usage` stream event.
 *
 *   2. When the actual count is ≥15% higher than the estimate (per-provider
 *      codec registry calibration tolerance), it records a soft signal to
 *      `evolution/streaming-compression.jsonl` and sets `needsReTierRecommendation()`.
 *
 *   3. Backprops live tier feedback into the existing adaptive regret learner
 *      so calibration happens within the current session, not only across sessions.
 *
 * ### Thread safety
 *
 * Instances are not shared across concurrent streams. Each stream invocation
 * should use its own `StreamingCompressionMonitor` instance.
 *
 * ### CompressorStreamFeedback (in-flight optimizer)
 *
 * `CompressorStreamFeedback` upgrades the monitor from a post-hoc learner to
 * an in-flight optimizer. It hooks into the SDK streaming event emitter and:
 *
 *   1. Records per-stream: tier applied, tokens removed, actual API latency,
 *      response quality signal (tool-call success, no clarification requests).
 *
 *   2. Maintains a lightweight EMA (exponential moving average) of tier ROI
 *      per (message-count-bucket, content-type) pair to predict the best tier
 *      for the next message.
 *
 *   3. Live switchboard: if a tier is underperforming (autoCompact latency >
 *      AUTO_COMPACT_LATENCY_THRESHOLD_MS with low ROI), emits a TierRecommendation
 *      to the next compression call.
 *
 *   4. Backpropagation hook: user-correction signals (e.g. "you missed that")
 *      drive a negative ROI penalty for the tier combination that caused it.
 *
 *   5. Cross-session persistence via `genome/evolution/compressor-feedback.jsonl`
 *      with week-over-week drift detection.
 */

import { join } from "path";
import { appendJsonl, readJsonl } from "../genome/jsonl.ts";
import type { LLMSummarizer, ProviderRequest, StreamEvent } from "../types/index.ts";
import type { CompressionTier } from "./context.ts";
import { recordTierOutcome, computeTierCost } from "./regret-learner.ts";
import { CodecRegistry } from "../tokens/index.ts";
import type { MessageBin } from "../tokens/provider-codec.ts";
import { classifyMessage } from "../tokens/provider-codec.ts";

// Convenience alias so call sites below read clearly.
const registry = CodecRegistry.instance;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GENOME_DIR = ".ashlrcode/genome";
const STREAMING_FEEDBACK_FILE = "streaming-compression.jsonl";

/**
 * Overage threshold (fraction).
 * When actual tokens exceed estimated by this fraction, a re-tier signal fires.
 * Mirrors the calibration tolerance in adaptive.ts.
 */
const OVERAGE_THRESHOLD = 0.15;

/**
 * Exponential backoff base factor for repeated overages within a session.
 * After each consecutive overage event the effective threshold tightens so
 * the monitor is more sensitive to continued drift.
 */
const BACKOFF_TIGHTEN_FACTOR = 0.8;

/**
 * Minimum effective threshold after backoff (prevents degenerate sensitivity).
 */
const MIN_THRESHOLD = 0.05;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Per-block-type token budget snapshot tracked during a stream.
 */
export interface BlockTypeBudget {
  /** Content block type being tracked. */
  blockType: MessageBin;
  /** Estimated tokens for this block type (pre-stream). */
  estimated: number;
  /** Accumulated actual tokens observed from usage events. */
  actual: number;
  /** Overage fraction: (actual - estimated) / estimated. */
  overageFraction: number;
}

/**
 * A soft signal appended to `evolution/streaming-compression.jsonl` when
 * significant budget overage is detected during streaming.
 */
export interface StreamingCompressionSignal {
  /** ISO timestamp of detection. */
  timestamp: string;
  /** Current compression tier at the time of detection. */
  tier: CompressionTier;
  /** Provider/model slug. */
  provider: string;
  /** Overage percentage (e.g. 23.5 means 23.5% over estimate). */
  overage_pct: number;
  /** Recommended next tier (lower number = more aggressive compression). */
  recommended_next_tier: CompressionTier;
  /** Block type that triggered the overage signal. */
  trigger_block_type: MessageBin;
  /** Total estimated tokens across all block types. */
  total_estimated: number;
  /** Total actual tokens from the usage event. */
  total_actual: number;
}

/**
 * Session-level summary produced after the stream ends.
 */
export interface StreamingSessionSummary {
  /** True if a re-tier recommendation was triggered during this stream. */
  needsReTier: boolean;
  /** The recommended tier if needsReTier is true; null otherwise. */
  recommendedTier: CompressionTier | null;
  /** The maximum overage fraction observed across all block types. */
  maxOverageFraction: number;
  /** Per-block-type budget snapshots. */
  blockBudgets: BlockTypeBudget[];
  /** Number of overage events recorded this session. */
  overageEventCount: number;
}

// ---------------------------------------------------------------------------
// StreamingCompressionMonitor
// ---------------------------------------------------------------------------

/**
 * Wraps `LLMSummarizer.stream()` to instrument token usage in real time.
 *
 * Usage:
 * ```ts
 * const monitor = new StreamingCompressionMonitor({
 *   summarizer,
 *   tier: 2,
 *   provider: "claude-sonnet",
 *   estimatedTokensByBlock: { text: 800, tool_result: 1200 },
 *   cwd: process.cwd(),
 * });
 *
 * for await (const event of monitor.stream(request)) {
 *   // handle event normally
 * }
 *
 * if (monitor.needsReTierRecommendation()) {
 *   const next = monitor.getRecommendedTier();
 *   // re-tier next call
 * }
 * ```
 */
export class StreamingCompressionMonitor {
  private readonly _summarizer: LLMSummarizer;
  private readonly _tier: CompressionTier;
  private readonly _provider: string;
  private readonly _cwd: string;
  private readonly _estimatedByBlock: Partial<Record<MessageBin, number>>;

  // Live state updated as stream events arrive
  private _totalEstimated = 0;
  private _totalActual = 0;
  private _blockBudgets: Map<MessageBin, BlockTypeBudget> = new Map();
  private _overageEventCount = 0;
  private _currentThreshold = OVERAGE_THRESHOLD;
  private _needsReTier = false;
  private _recommendedTier: CompressionTier | null = null;
  private _streamEnded = false;

  constructor(opts: {
    summarizer: LLMSummarizer;
    tier: CompressionTier;
    provider: string;
    estimatedTokensByBlock: Partial<Record<MessageBin, number>>;
    cwd: string;
  }) {
    this._summarizer = opts.summarizer;
    this._tier = opts.tier;
    this._provider = opts.provider;
    this._estimatedByBlock = opts.estimatedTokensByBlock;
    this._cwd = opts.cwd;

    // Initialize per-block budgets from the pre-stream estimates
    for (const [bt, est] of Object.entries(opts.estimatedTokensByBlock) as [MessageBin, number][]) {
      if (est > 0) {
        this._blockBudgets.set(bt, {
          blockType: bt,
          estimated: est,
          actual: 0,
          overageFraction: 0,
        });
        this._totalEstimated += est;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Wrap the underlying summarizer stream. Passes every event through
   * unchanged while tracking usage events in the background.
   *
   * The caller iterates this exactly as they would the raw stream — semantics
   * are preserved end-to-end.
   */
  async *stream(request: ProviderRequest): AsyncGenerator<StreamEvent> {
    for await (const event of this._summarizer.stream(request)) {
      // Track token usage events
      if (event.type === "usage" && event.usage) {
        await this._handleUsageEvent(event.usage.inputTokens, event.usage.outputTokens);
      }

      if (event.type === "message_end") {
        this._streamEnded = true;
        // Final usage is typically in the message_end event too; update if present
        if (event.usage) {
          await this._handleUsageEvent(event.usage.inputTokens, event.usage.outputTokens);
        }
      }

      yield event;
    }
  }

  /**
   * Wrap the underlying summarizer stream with a pre-classified message list
   * for per-block-type tracking.
   *
   * This is the preferred entry point when the caller has the original messages
   * available so per-block-type budget tracking can be bootstrapped from actual
   * message classification rather than caller-supplied estimates.
   */
  async *streamWithMessages(
    request: ProviderRequest,
  ): AsyncGenerator<StreamEvent> {
    // Re-initialise block budgets from the actual messages in the request
    this._blockBudgets.clear();
    this._totalEstimated = 0;

    const codec = registry.getCodec(this._provider);
    for (const msg of request.messages) {
      const bin = classifyMessage(msg);
      const tokenEst = codec.encodeMessage(msg);
      const existing = this._blockBudgets.get(bin);
      if (existing) {
        existing.estimated += tokenEst;
        this._totalEstimated += tokenEst;
      } else {
        this._blockBudgets.set(bin, {
          blockType: bin,
          estimated: tokenEst,
          actual: 0,
          overageFraction: 0,
        });
        this._totalEstimated += tokenEst;
      }
    }

    // Merge any caller-supplied overrides
    for (const [bt, est] of Object.entries(this._estimatedByBlock) as [MessageBin, number][]) {
      if (est > 0 && !this._blockBudgets.has(bt)) {
        this._blockBudgets.set(bt, {
          blockType: bt,
          estimated: est,
          actual: 0,
          overageFraction: 0,
        });
        this._totalEstimated += est;
      }
    }

    yield* this.stream(request);
  }

  /**
   * Returns true if a significant budget overage was observed during streaming
   * and the caller should consider re-tiering for the next LLM call.
   */
  needsReTierRecommendation(): boolean {
    return this._needsReTier;
  }

  /**
   * Returns the recommended next tier when `needsReTierRecommendation()` is true.
   * Returns null if no re-tier recommendation has been generated.
   */
  getRecommendedTier(): CompressionTier | null {
    return this._recommendedTier;
  }

  /**
   * Returns the current session summary including per-block budgets and
   * overage statistics.
   */
  getSessionSummary(): StreamingSessionSummary {
    return {
      needsReTier: this._needsReTier,
      recommendedTier: this._recommendedTier,
      maxOverageFraction: this._computeMaxOverage(),
      blockBudgets: Array.from(this._blockBudgets.values()),
      overageEventCount: this._overageEventCount,
    };
  }

  /**
   * Returns per-block-type budget snapshots for inspection.
   */
  getBlockBudgets(): BlockTypeBudget[] {
    return Array.from(this._blockBudgets.values());
  }

  // ---------------------------------------------------------------------------
  // Internal
  // ---------------------------------------------------------------------------

  private async _handleUsageEvent(inputTokens: number, _outputTokens: number): Promise<void> {
    if (inputTokens <= 0) return;

    this._totalActual = inputTokens;

    // Distribute the observed actual tokens proportionally across tracked block types
    // when we have per-block estimates. This is a best-effort attribution — the API
    // reports aggregate tokens, not per-block breakdown.
    if (this._totalEstimated > 0 && this._blockBudgets.size > 0) {
      for (const [, budget] of this._blockBudgets) {
        if (budget.estimated > 0) {
          const share = budget.estimated / this._totalEstimated;
          budget.actual = Math.round(inputTokens * share);
          budget.overageFraction =
            budget.estimated > 0
              ? (budget.actual - budget.estimated) / budget.estimated
              : 0;
        }
      }
    }

    // Check for overage against the current (possibly backoff-tightened) threshold
    if (this._totalEstimated > 0) {
      const totalOverage = (inputTokens - this._totalEstimated) / this._totalEstimated;
      if (totalOverage >= this._currentThreshold) {
        await this._handleOverage(totalOverage, inputTokens);
      }
    }
  }

  private async _handleOverage(overageFraction: number, actualTokens: number): Promise<void> {
    this._overageEventCount++;

    // Apply exponential backoff: tighten threshold for subsequent checks
    this._currentThreshold = Math.max(
      MIN_THRESHOLD,
      this._currentThreshold * BACKOFF_TIGHTEN_FACTOR,
    );

    // Derive recommended next tier: escalate by 1 (lower tier number = more aggressive)
    const recommended = Math.max(1, this._tier - 1) as CompressionTier;
    this._needsReTier = true;
    this._recommendedTier = recommended;

    // Find the block type with the highest overage for the signal record
    let maxOverageBin: MessageBin = "text";
    let maxOverageVal = -Infinity;
    for (const [bin, budget] of this._blockBudgets) {
      if (budget.overageFraction > maxOverageVal) {
        maxOverageVal = budget.overageFraction;
        maxOverageBin = bin;
      }
    }

    // Record the soft signal to the evolution JSONL
    const signal: StreamingCompressionSignal = {
      timestamp: new Date().toISOString(),
      tier: this._tier,
      provider: this._provider,
      overage_pct: overageFraction * 100,
      recommended_next_tier: recommended,
      trigger_block_type: maxOverageBin,
      total_estimated: this._totalEstimated,
      total_actual: actualTokens,
    };

    const signalPath = join(this._cwd, GENOME_DIR, "evolution", STREAMING_FEEDBACK_FILE);
    // Fire-and-forget: we don't want I/O to stall the stream
    appendJsonl(signalPath, signal).catch(() => {
      // Silently ignore I/O failures — monitoring must never break the stream
    });

    // Backprop into regret learner for within-session calibration.
    // recordTierOutcome(tier, tokensRemoved, provider) internally computes costs
    // and regret across all tiers — we pass the overage as "tokens removed" so
    // the learner sees the excess spend attributable to the mis-estimate.
    const tokensOverage = Math.max(0, actualTokens - this._totalEstimated);
    recordTierOutcome(this._tier, tokensOverage, this._provider);
  }

  private _computeMaxOverage(): number {
    let max = 0;
    for (const [, budget] of this._blockBudgets) {
      if (budget.overageFraction > max) max = budget.overageFraction;
    }
    return max;
  }
}

// ---------------------------------------------------------------------------
// Path helper (exported for tests)
// ---------------------------------------------------------------------------

/**
 * Absolute path to the streaming compression feedback JSONL file.
 */
export function streamingFeedbackPath(cwd: string): string {
  return join(cwd, GENOME_DIR, "evolution", STREAMING_FEEDBACK_FILE);
}

// ---------------------------------------------------------------------------
// Factory helper
// ---------------------------------------------------------------------------

/**
 * Create a `StreamingCompressionMonitor` from a live message array.
 *
 * Classifies messages using the provider codec to derive per-block-type
 * token estimates automatically — no manual breakdown required.
 *
 * @param summarizer  The underlying LLM summarizer to wrap.
 * @param request     The provider request about to be sent.
 * @param tier        Current compression tier.
 * @param provider    Provider/model slug.
 * @param cwd         Project working directory (for JSONL paths).
 */
export function createMonitor(
  summarizer: LLMSummarizer,
  request: ProviderRequest,
  tier: CompressionTier,
  provider: string,
  cwd: string,
): StreamingCompressionMonitor {
  const codec = registry.getCodec(provider);
  const estimatedByBlock: Partial<Record<MessageBin, number>> = {};

  for (const msg of request.messages) {
    const bin = classifyMessage(msg);
    const tokens = codec.encodeMessage(msg);
    estimatedByBlock[bin] = (estimatedByBlock[bin] ?? 0) + tokens;
  }

  return new StreamingCompressionMonitor({
    summarizer,
    tier,
    provider,
    estimatedTokensByBlock: estimatedByBlock,
    cwd,
  });
}

// ===========================================================================
// CompressorStreamFeedback — in-flight optimizer
// ===========================================================================

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Latency threshold (ms) above which autoCompact (tier 1) is flagged as underperforming. */
export const AUTO_COMPACT_LATENCY_THRESHOLD_MS = 5000;

/** EMA smoothing factor for per-bucket ROI estimates. 0.2 ≈ 4-call half-life. */
export const FEEDBACK_EMA_ALPHA = 0.2;

/** Minimum observations per (bucket, contentType) cell before the EMA recommendation is trusted. */
export const FEEDBACK_MIN_SAMPLES = 3;

/** Negative ROI penalty applied when a user correction (backprop signal) is received. */
export const CORRECTION_PENALTY_FACTOR = 2.0;

/** Cross-session feedback JSONL filename. */
const COMPRESSOR_FEEDBACK_FILE = "compressor-feedback.jsonl";

/** Number of ISO-week buckets to compare for drift detection. */
const DRIFT_COMPARISON_WEEKS = 2;

/** Minimum change in best-tier fraction to trigger a drift alert. */
const DRIFT_ALERT_THRESHOLD = 0.25;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Content-type label used as a dimension in the EMA model.
 * Derived from the dominant MessageBin in a request.
 */
export type ContentType = "text" | "tool_result" | "thinking" | "tool_use" | "mixed";

/**
 * Message-count bucket label — xs (<10), sm (10–29), md (30–59), lg (≥60).
 * Mirrors MessageCountBucket from budget/index.ts but kept local to avoid
 * an extra import and satisfy strict isolation.
 */
export type FeedbackBucket = "xs" | "sm" | "md" | "lg";

/**
 * A single feedback observation recorded after a streaming API call.
 */
export interface CompressionOutcome {
  /** ISO timestamp of the observation. */
  timestamp: string;
  /** Compression tier that was applied. */
  tier: CompressionTier;
  /** Provider/model slug. */
  provider: string;
  /** Tokens removed by compression (estimate − actual before tier was applied). */
  tokensRemoved: number;
  /** Actual API round-trip latency in milliseconds. */
  latencyMs: number;
  /** Whether the LLM response contained a tool call (quality proxy). */
  toolCallSuccess: boolean;
  /** True if the next user turn contains a clarification request. */
  clarificationRequested: boolean;
  /** Message-count bucket at call time. */
  bucket: FeedbackBucket;
  /** Content type dominant in this request. */
  contentType: ContentType;
  /**
   * Computed ROI for this observation.
   * roi = tokensRemoved / max(1, latencyMs) — tokens saved per ms of latency cost.
   * Higher is better.
   */
  roi: number;
  /** True when this record was produced by a user-correction backprop event. */
  isCorrection?: boolean;
}

/**
 * A recommendation emitted by the live switchboard for the next compression call.
 */
export interface TierRecommendation {
  /** Recommended tier for the next call. */
  tier: CompressionTier;
  /**
   * Reason the switchboard produced this recommendation.
   * One of:
   *   "ema_roi"             — EMA model predicted this tier has best ROI
   *   "latency_penalty"     — current tier latency exceeded threshold
   *   "correction_backprop" — user correction penalised current tier
   *   "insufficient_data"   — not enough observations yet, returning default
   */
  reason: "ema_roi" | "latency_penalty" | "correction_backprop" | "insufficient_data";
  /** EMA ROI score of the recommended tier for this (bucket, contentType) cell. */
  emaRoi: number;
  /** ISO timestamp of when this recommendation was generated. */
  timestamp: string;
}

/**
 * Per-(bucket, contentType) cell state maintained by the EMA model.
 */
interface EmaCell {
  /** EMA of ROI values observed for each tier in this cell. */
  emaRoi: Record<CompressionTier, number>;
  /** Observation count per tier in this cell. */
  count: Record<CompressionTier, number>;
  /** True if the most recent observation was a correction event. */
  lastWasCorrection: boolean;
}

/**
 * Cross-session record persisted to `genome/evolution/compressor-feedback.jsonl`.
 * Superset of `CompressionOutcome` with an ISO-week tag for drift detection.
 */
export interface FeedbackPersistRecord extends CompressionOutcome {
  /** ISO week string "YYYY-Www" (e.g. "2026-W26") for drift bucketing. */
  isoWeek: string;
}

/**
 * Result of a drift-detection scan across the JSONL feedback log.
 */
export interface DriftDetectionResult {
  /** True when the best tier changed significantly week-over-week. */
  driftDetected: boolean;
  /** Best tier in the earlier week window. */
  previousBestTier: CompressionTier | null;
  /** Best tier in the most recent week window. */
  currentBestTier: CompressionTier | null;
  /** The (bucket, contentType) key where drift was detected first. */
  driftKey: string | null;
  /** ISO timestamp of the scan. */
  scannedAt: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Return an ISO week string "YYYY-Www" for a given Date. */
function isoWeek(date: Date): string {
  // ISO 8601: week starts Monday; week 1 contains first Thursday.
  const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  const dayNum = d.getUTCDay() || 7; // make Sunday = 7
  d.setUTCDate(d.getUTCDate() + 4 - dayNum);
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
  const weekNo = Math.ceil((((d.getTime() - yearStart.getTime()) / 86400000) + 1) / 7);
  return `${d.getUTCFullYear()}-W${String(weekNo).padStart(2, "0")}`;
}

/**
 * Classify the dominant content type from a partial Record of MessageBin estimates.
 */
function classifyContentType(estimates: Partial<Record<MessageBin, number>>): ContentType {
  const entries = Object.entries(estimates) as [MessageBin, number][];
  if (entries.length === 0) return "text";
  if (entries.length === 1) {
    const bin = entries[0]![0];
    if (bin === "text" || bin === "tool_result" || bin === "thinking" || bin === "tool_use") {
      return bin as ContentType;
    }
    return "text";
  }
  // Multiple bins — check if one dominates (>60% of total)
  const total = entries.reduce((s, [, v]) => s + v, 0);
  for (const [bin, v] of entries) {
    if (total > 0 && v / total > 0.6) {
      if (bin === "text" || bin === "tool_result" || bin === "thinking" || bin === "tool_use") {
        return bin as ContentType;
      }
    }
  }
  return "mixed";
}

/**
 * Map a message count to a FeedbackBucket.
 */
function toFeedbackBucket(count: number): FeedbackBucket {
  if (count < 10) return "xs";
  if (count < 30) return "sm";
  if (count < 60) return "md";
  return "lg";
}

/** Build the EMA cell key string. */
function cellKey(bucket: FeedbackBucket, contentType: ContentType): string {
  return `${bucket}:${contentType}`;
}

/** Initialise a fresh EmaCell with zero values for all tiers. */
function freshCell(): EmaCell {
  return {
    emaRoi: { 1: 0, 2: 0, 3: 0, 4: 0 },
    count: { 1: 0, 2: 0, 3: 0, 4: 0 },
    lastWasCorrection: false,
  };
}

// ---------------------------------------------------------------------------
// CompressorStreamFeedback
// ---------------------------------------------------------------------------

/**
 * Real-time feedback loop that observes compression tier effectiveness during
 * API streaming and adaptively reweights tier selection for the next message.
 *
 * ### Usage
 *
 * ```ts
 * const feedback = new CompressorStreamFeedback({ cwd: process.cwd() });
 *
 * // After each streaming call:
 * const outcome = await observeCompressionOutcome(feedback, {
 *   tier, provider, tokensRemoved, latencyMs,
 *   toolCallSuccess, clarificationRequested,
 *   messageCount, estimatedByBlock,
 * });
 *
 * // Before next compression call:
 * const rec = await recommendTierForNextMessage(feedback, {
 *   messageCount, estimatedByBlock, provider,
 * });
 * if (rec.reason !== "insufficient_data") {
 *   // use rec.tier
 * }
 *
 * // When user corrects a mistake caused by excessive compression:
 * await feedback.recordCorrection({ tier, provider, tokensRemoved, messageCount, estimatedByBlock });
 * ```
 */
export class CompressorStreamFeedback {
  private readonly _cwd: string;
  /** EMA model: (bucket:contentType) → per-tier EMA state. */
  private readonly _cells: Map<string, EmaCell> = new Map();
  /** Last TierRecommendation emitted by the switchboard (for inspection/tests). */
  private _lastRecommendation: TierRecommendation | null = null;
  /** Whether a correction backprop event was recorded since last recommendation. */
  private _correctionPending = false;
  /** The tier that most recently triggered a correction. */
  private _correctedTier: CompressionTier | null = null;

  constructor(opts: { cwd: string }) {
    this._cwd = opts.cwd;
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Record one compression outcome observation and update the EMA model.
   *
   * This is called internally by `observeCompressionOutcome()` but is also
   * public so callers that build outcomes manually can feed them in directly.
   */
  record(outcome: CompressionOutcome): void {
    const key = cellKey(outcome.bucket, outcome.contentType);
    if (!this._cells.has(key)) {
      this._cells.set(key, freshCell());
    }
    const cell = this._cells.get(key)!;

    const alpha = FEEDBACK_EMA_ALPHA;
    const tier = outcome.tier;

    // Apply correction penalty if this is a user correction event.
    const effectiveRoi = outcome.isCorrection
      ? -Math.abs(outcome.roi) * CORRECTION_PENALTY_FACTOR
      : outcome.roi;

    // EMA update: ema_new = α × roi + (1 − α) × ema_old
    cell.emaRoi[tier] = alpha * effectiveRoi + (1 - alpha) * cell.emaRoi[tier];
    cell.count[tier]++;
    cell.lastWasCorrection = outcome.isCorrection === true;

    // Persist to cross-session JSONL (fire-and-forget — never blocks stream)
    const record: FeedbackPersistRecord = {
      ...outcome,
      isoWeek: isoWeek(new Date()),
    };
    const path = this.feedbackPath();
    appendJsonl(path, record).catch(() => {
      // Best-effort — never break the compression pipeline
    });
  }

  /**
   * Record a user-correction backprop signal.
   *
   * When the user explicitly corrects a mistake caused by excessive compression
   * (e.g. "you missed that context"), this records a negative signal to penalise
   * the tier combination that caused it.
   *
   * @param opts.tier              Tier that caused the mistake.
   * @param opts.provider          Provider string.
   * @param opts.tokensRemoved     Tokens removed by that tier (for ROI context).
   * @param opts.messageCount      Message count at time of call.
   * @param opts.estimatedByBlock  Token estimates by block type.
   */
  recordCorrection(opts: {
    tier: CompressionTier;
    provider: string;
    tokensRemoved: number;
    messageCount: number;
    estimatedByBlock: Partial<Record<MessageBin, number>>;
  }): void {
    const bucket = toFeedbackBucket(opts.messageCount);
    const contentType = classifyContentType(opts.estimatedByBlock);
    const roi = opts.tokensRemoved / 1; // latency=1 to avoid division by zero when not available

    const outcome: CompressionOutcome = {
      timestamp: new Date().toISOString(),
      tier: opts.tier,
      provider: opts.provider,
      tokensRemoved: opts.tokensRemoved,
      latencyMs: 0,
      toolCallSuccess: false,
      clarificationRequested: true,
      bucket,
      contentType,
      roi,
      isCorrection: true,
    };
    this._correctionPending = true;
    this._correctedTier = opts.tier;
    this.record(outcome);
  }

  /**
   * Recommend the best compression tier for the next message based on the
   * current EMA model state.
   *
   * Implements the live switchboard logic:
   *   1. If a correction backprop is pending, immediately recommend stepping
   *      up from the penalised tier (more aggressive, lower tier number).
   *   2. If the last latency exceeded AUTO_COMPACT_LATENCY_THRESHOLD_MS and
   *      the tier was autoCompact (tier 1), recommend tier 2 instead.
   *   3. Otherwise, pick the tier with the highest EMA ROI for the current
   *      (bucket, contentType) cell — provided it has enough observations.
   *   4. Fall back to "insufficient_data" when the cell is too sparse.
   */
  recommend(opts: {
    messageCount: number;
    estimatedByBlock: Partial<Record<MessageBin, number>>;
    lastLatencyMs?: number;
    lastTier?: CompressionTier;
  }): TierRecommendation {
    const bucket = toFeedbackBucket(opts.messageCount);
    const contentType = classifyContentType(opts.estimatedByBlock);

    // Priority 1: correction backprop — penalise the offending tier
    if (this._correctionPending && this._correctedTier !== null) {
      this._correctionPending = false;
      const penalised = this._correctedTier;
      // Step to more aggressive tier (lower number), but clamp to [1, 4]
      const recommended = Math.max(1, penalised - 1) as CompressionTier;
      const rec: TierRecommendation = {
        tier: recommended,
        reason: "correction_backprop",
        emaRoi: this._getEmaRoi(bucket, contentType, recommended),
        timestamp: new Date().toISOString(),
      };
      this._lastRecommendation = rec;
      return rec;
    }

    // Priority 2: latency penalty for tier 1 (autoCompact)
    if (
      opts.lastTier === 1 &&
      opts.lastLatencyMs !== undefined &&
      opts.lastLatencyMs > AUTO_COMPACT_LATENCY_THRESHOLD_MS
    ) {
      const roi2 = this._getEmaRoi(bucket, contentType, 2);
      const rec: TierRecommendation = {
        tier: 2,
        reason: "latency_penalty",
        emaRoi: roi2,
        timestamp: new Date().toISOString(),
      };
      this._lastRecommendation = rec;
      return rec;
    }

    // Priority 3: EMA ROI model
    const key = cellKey(bucket, contentType);
    const cell = this._cells.get(key);

    if (!cell) {
      const rec: TierRecommendation = {
        tier: 3,
        reason: "insufficient_data",
        emaRoi: 0,
        timestamp: new Date().toISOString(),
      };
      this._lastRecommendation = rec;
      return rec;
    }

    // Find the tier with the highest EMA ROI among those with enough data
    let bestTier: CompressionTier = 3;
    let bestRoi = -Infinity;
    let hasTrustedData = false;

    for (const t of [4, 3, 2, 1] as CompressionTier[]) {
      if (cell.count[t] >= FEEDBACK_MIN_SAMPLES) {
        hasTrustedData = true;
        if (cell.emaRoi[t] > bestRoi) {
          bestRoi = cell.emaRoi[t];
          bestTier = t;
        }
      }
    }

    if (!hasTrustedData) {
      const rec: TierRecommendation = {
        tier: 3,
        reason: "insufficient_data",
        emaRoi: 0,
        timestamp: new Date().toISOString(),
      };
      this._lastRecommendation = rec;
      return rec;
    }

    const rec: TierRecommendation = {
      tier: bestTier,
      reason: "ema_roi",
      emaRoi: bestRoi,
      timestamp: new Date().toISOString(),
    };
    this._lastRecommendation = rec;
    return rec;
  }

  /**
   * Return the most recent TierRecommendation emitted, or null if none yet.
   */
  getLastRecommendation(): TierRecommendation | null {
    return this._lastRecommendation;
  }

  /**
   * Return the current EMA ROI for a specific tier in the given (bucket, contentType) cell.
   * Returns 0 when the cell or tier has no observations.
   */
  getEmaRoi(
    bucket: FeedbackBucket,
    contentType: ContentType,
    tier: CompressionTier,
  ): number {
    return this._getEmaRoi(bucket, contentType, tier);
  }

  /**
   * Return the observation count for a tier in a (bucket, contentType) cell.
   */
  getObservationCount(
    bucket: FeedbackBucket,
    contentType: ContentType,
    tier: CompressionTier,
  ): number {
    const key = cellKey(bucket, contentType);
    const cell = this._cells.get(key);
    return cell?.count[tier] ?? 0;
  }

  /**
   * Scan the cross-session JSONL for week-over-week drift in best-tier selection.
   *
   * Compares per-(bucket, contentType) best-tier in the previous ISO week to the
   * current ISO week. Alerts when the best tier changes by ≥ DRIFT_ALERT_THRESHOLD
   * of observations.
   *
   * Returns a DriftDetectionResult. When no historical data exists the result
   * has driftDetected=false.
   */
  async detectDrift(): Promise<DriftDetectionResult> {
    const path = this.feedbackPath();
    const records = await readJsonl<FeedbackPersistRecord>(path);

    if (records.length === 0) {
      return {
        driftDetected: false,
        previousBestTier: null,
        currentBestTier: null,
        driftKey: null,
        scannedAt: new Date().toISOString(),
      };
    }

    const now = new Date();
    const currentWeek = isoWeek(now);
    // Previous week: subtract 7 days
    const prevDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const previousWeek = isoWeek(prevDate);

    // Bucket records by isoWeek
    const byWeek = new Map<string, FeedbackPersistRecord[]>();
    for (const r of records) {
      const w = r.isoWeek ?? "unknown";
      if (!byWeek.has(w)) byWeek.set(w, []);
      byWeek.get(w)!.push(r);
    }

    const currentWeekRecords = byWeek.get(currentWeek) ?? [];
    // For previous week comparison, look back DRIFT_COMPARISON_WEEKS weeks
    const prevWeekRecords: FeedbackPersistRecord[] = [];
    for (let i = 1; i <= DRIFT_COMPARISON_WEEKS; i++) {
      const d = new Date(now.getTime() - i * 7 * 24 * 60 * 60 * 1000);
      const w = isoWeek(d);
      const recs = byWeek.get(w) ?? [];
      prevWeekRecords.push(...recs);
    }

    if (currentWeekRecords.length === 0 || prevWeekRecords.length === 0) {
      return {
        driftDetected: false,
        previousBestTier: null,
        currentBestTier: null,
        driftKey: null,
        scannedAt: new Date().toISOString(),
      };
    }

    // Per (bucket, contentType) key, compute the dominant tier fraction for each window
    const computeBestTierFractions = (
      recs: FeedbackPersistRecord[],
    ): Map<string, Record<CompressionTier, number>> => {
      const keyTierCounts = new Map<string, Record<CompressionTier, number>>();
      for (const r of recs) {
        const k = cellKey(r.bucket, r.contentType);
        if (!keyTierCounts.has(k)) {
          keyTierCounts.set(k, { 1: 0, 2: 0, 3: 0, 4: 0 });
        }
        const counts = keyTierCounts.get(k)!;
        if (r.tier === 1 || r.tier === 2 || r.tier === 3 || r.tier === 4) {
          counts[r.tier]++;
        }
      }
      // Convert to fractions
      const result = new Map<string, Record<CompressionTier, number>>();
      for (const [k, counts] of keyTierCounts) {
        const total = Object.values(counts).reduce((a, b) => a + b, 0);
        result.set(k, {
          1: total > 0 ? counts[1] / total : 0,
          2: total > 0 ? counts[2] / total : 0,
          3: total > 0 ? counts[3] / total : 0,
          4: total > 0 ? counts[4] / total : 0,
        });
      }
      return result;
    };

    const currentFractions = computeBestTierFractions(currentWeekRecords);
    const prevFractions = computeBestTierFractions(prevWeekRecords);

    // Check for best-tier change in any shared key
    for (const [key, currentFrac] of currentFractions) {
      const prevFrac = prevFractions.get(key);
      if (!prevFrac) continue;

      // Best tier = highest fraction
      const bestTierOf = (fracs: Record<CompressionTier, number>): CompressionTier => {
        let best: CompressionTier = 4;
        let bestF = -1;
        for (const t of [1, 2, 3, 4] as CompressionTier[]) {
          if (fracs[t] > bestF) {
            bestF = fracs[t];
            best = t;
          }
        }
        return best;
      };

      const currentBest = bestTierOf(currentFrac);
      const prevBest = bestTierOf(prevFrac);

      if (currentBest !== prevBest) {
        // Check if the fraction shift is significant
        const fractionChange = Math.abs(
          currentFrac[currentBest] - prevFrac[currentBest],
        );
        if (fractionChange >= DRIFT_ALERT_THRESHOLD) {
          return {
            driftDetected: true,
            previousBestTier: prevBest,
            currentBestTier: currentBest,
            driftKey: key,
            scannedAt: new Date().toISOString(),
          };
        }
      }
    }

    return {
      driftDetected: false,
      previousBestTier: null,
      currentBestTier: null,
      driftKey: null,
      scannedAt: new Date().toISOString(),
    };
  }

  /**
   * Absolute path to the cross-session compressor-feedback JSONL file.
   */
  feedbackPath(): string {
    return join(this._cwd, GENOME_DIR, "evolution", COMPRESSOR_FEEDBACK_FILE);
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  private _getEmaRoi(
    bucket: FeedbackBucket,
    contentType: ContentType,
    tier: CompressionTier,
  ): number {
    const key = cellKey(bucket, contentType);
    return this._cells.get(key)?.emaRoi[tier] ?? 0;
  }
}

// ---------------------------------------------------------------------------
// Module-level singleton and convenience functions
// ---------------------------------------------------------------------------

/**
 * Module-level singleton `CompressorStreamFeedback` instance.
 * Initialised lazily using `process.cwd()` as the working directory.
 * Tests should use `createFeedback()` to build isolated instances.
 */
let _feedbackSingleton: CompressorStreamFeedback | null = null;

function _getSingleton(): CompressorStreamFeedback {
  if (!_feedbackSingleton) {
    _feedbackSingleton = new CompressorStreamFeedback({ cwd: process.cwd() });
  }
  return _feedbackSingleton;
}

/**
 * Reset the module-level singleton. For tests only.
 */
export function _resetFeedbackSingleton(): void {
  _feedbackSingleton = null;
}

/**
 * Create an isolated `CompressorStreamFeedback` instance for the given cwd.
 *
 * Preferred over the singleton for tests and multi-project scenarios.
 */
export function createFeedback(cwd: string): CompressorStreamFeedback {
  return new CompressorStreamFeedback({ cwd });
}

/**
 * Record a compression outcome observation.
 *
 * Computes ROI from `tokensRemoved / max(1, latencyMs)` and feeds the
 * resulting `CompressionOutcome` into the EMA model.
 *
 * @param feedback          The `CompressorStreamFeedback` instance to update.
 * @param opts.tier         Compression tier that was applied.
 * @param opts.provider     Provider/model slug.
 * @param opts.tokensRemoved Tokens removed by compression (estimated − actual).
 * @param opts.latencyMs    API round-trip latency in milliseconds.
 * @param opts.toolCallSuccess Whether the response contained a successful tool call.
 * @param opts.clarificationRequested Whether the next user turn is a correction/clarification.
 * @param opts.messageCount Number of messages in the context at call time.
 * @param opts.estimatedByBlock Per-block-type token estimates (used to derive content type).
 * @returns                 The `CompressionOutcome` that was recorded.
 */
export function observeCompressionOutcome(
  feedback: CompressorStreamFeedback,
  opts: {
    tier: CompressionTier;
    provider: string;
    tokensRemoved: number;
    latencyMs: number;
    toolCallSuccess: boolean;
    clarificationRequested: boolean;
    messageCount: number;
    estimatedByBlock: Partial<Record<MessageBin, number>>;
  },
): CompressionOutcome {
  const bucket = toFeedbackBucket(opts.messageCount);
  const contentType = classifyContentType(opts.estimatedByBlock);
  const roi = opts.tokensRemoved / Math.max(1, opts.latencyMs);

  const outcome: CompressionOutcome = {
    timestamp: new Date().toISOString(),
    tier: opts.tier,
    provider: opts.provider,
    tokensRemoved: opts.tokensRemoved,
    latencyMs: opts.latencyMs,
    toolCallSuccess: opts.toolCallSuccess,
    clarificationRequested: opts.clarificationRequested,
    bucket,
    contentType,
    roi,
  };

  feedback.record(outcome);
  return outcome;
}

/**
 * Recommend the best compression tier for the next message based on the EMA model.
 *
 * This is the primary entry point for the live switchboard. Consults the
 * `CompressorStreamFeedback` instance's EMA model and live-switchboard logic,
 * then returns a `TierRecommendation` that callers can pass to
 * `selectCompressionTierAdaptive` via the `liveRecommendation` parameter.
 *
 * Falls back to the module-level singleton when no explicit `feedback` is supplied.
 *
 * @param feedback          The feedback instance (or null to use the singleton).
 * @param opts.messageCount Current message count (used to derive bucket).
 * @param opts.estimatedByBlock Per-block-type token estimates.
 * @param opts.lastLatencyMs Latency of the most recent API call (optional).
 * @param opts.lastTier     Tier used in the most recent call (optional).
 * @returns                 A `TierRecommendation`.
 */
export function recommendTierForNextMessage(
  feedback: CompressorStreamFeedback | null,
  opts: {
    messageCount: number;
    estimatedByBlock: Partial<Record<MessageBin, number>>;
    lastLatencyMs?: number;
    lastTier?: CompressionTier;
  },
): TierRecommendation {
  const fb = feedback ?? _getSingleton();
  return fb.recommend(opts);
}
