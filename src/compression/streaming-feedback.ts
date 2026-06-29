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
 */

import { join } from "path";
import { appendJsonl } from "../genome/jsonl.ts";
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
