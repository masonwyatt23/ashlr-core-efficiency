/**
 * cache-optimizer.ts — Multi-tier prompt caching cost analyzer +
 * auto-breakpoint optimizer.
 *
 * The `CacheOptimizer` class:
 *   1. Analyzes the cost-benefit of different cache breakpoint strategies.
 *   2. Tracks actual cache hit/miss patterns from MessageResponse metadata.
 *   3. Learns optimal breakpoint placement via a JSONL audit trail.
 *   4. Recommends breakpoint shifts when the provider changes
 *      (via `rebalanceBudgetOnSwitch` integration).
 *   5. Emits a `CacheOptimizationReport` showing ROI per strategy.
 *
 * All learning state lives in the JSONL file — the class itself is stateless
 * and can be reconstructed cheaply between requests.
 */

import {
  appendCacheAudit,
  loadCacheAudit,
  summarizeAuditWindow,
  computeEffectiveCostMicroUsd,
  computeBaselineCostMicroUsd,
  type CacheStrategy,
  type CacheAuditEntry,
  type CacheAuditSummary,
} from "./cache-audit.ts";
import {
  rebalanceBudgetOnSwitch,
  EXTENDED_CONTEXT_MODELS,
  PROVIDER_PRICING,
  type BudgetRebalanceResult,
} from "../budget/index.ts";
import type { MessageResponse } from "../types/index.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Options for constructing a CacheOptimizer.
 */
export interface CacheOptimizerOptions {
  /**
   * Working directory used to locate the audit JSONL.
   * Defaults to `process.cwd()`.
   */
  cwd?: string;
  /**
   * Provider name (e.g. "anthropic"). Used for pricing lookups.
   * Defaults to "anthropic".
   */
  provider?: string;
  /**
   * Model ID (e.g. "claude-sonnet-4"). Used for pricing lookups.
   * Defaults to "claude-sonnet-4".
   */
  modelId?: string;
  /**
   * How many recent audit entries to consider when scoring strategies.
   * Defaults to 100.
   */
  windowSize?: number;
  /**
   * Minimum number of entries per strategy before adaptive selection activates.
   * Below this threshold the optimizer returns heuristic defaults.
   * Defaults to 5.
   */
  minSamplesForLearning?: number;
  /**
   * Optional session tag to attach to every audit record.
   */
  sessionId?: string;
}

/**
 * ROI breakdown for a single cache strategy.
 */
export interface StrategyROI {
  strategy: CacheStrategy;
  /** Number of audit records for this strategy in the analysis window. */
  sampleCount: number;
  /** Average saving per request vs. the no-cache baseline, in micro-USD. */
  avgSavingMicroUsd: number;
  /** Total saving across all records in the window, in micro-USD. */
  totalSavingMicroUsd: number;
  /** Fraction of requests where at least one cache-read token was returned. */
  hitRate: number;
  /**
   * Whether this strategy is recommended based on the current audit data.
   * Exactly one strategy will be marked recommended (the one with highest ROI
   * when enough data exists).
   */
  recommended: boolean;
}

/**
 * Recommendation returned when a provider switch is detected.
 */
export interface ProviderSwitchRecommendation {
  fromProvider: string;
  toProvider: string;
  /** Budget rebalance result from `rebalanceBudgetOnSwitch`. */
  budgetRebalance: BudgetRebalanceResult;
  /**
   * Suggested cache strategy for the new provider.
   *
   * - Switching to a provider without native caching support (e.g. groq, deepseek,
   *   ollama) → `"none"`.
   * - Switching to Anthropic or OpenAI (with caching) → best strategy from
   *   current audit data, or `"system-only"` as a conservative default.
   */
  suggestedStrategy: CacheStrategy;
  /** Human-readable rationale for the suggestion. */
  rationale: string;
}

/**
 * Full optimization report emitted by `CacheOptimizer.generateReport`.
 */
export interface CacheOptimizationReport {
  /** ISO-8601 timestamp when the report was generated. */
  generatedAt: string;
  /** Provider used for pricing calculations. */
  provider: string;
  /** Model used for pricing calculations. */
  modelId: string;
  /** Total number of audit entries in the analysis window. */
  totalAuditEntries: number;
  /** Per-strategy ROI breakdown (all known strategies). */
  strategyROI: StrategyROI[];
  /**
   * The single recommended strategy given current hit/miss patterns.
   * `null` when there is insufficient data.
   */
  recommendedStrategy: CacheStrategy | null;
  /**
   * Overall cache effectiveness: total saved micro-USD across all strategies
   * vs. the all-uncached baseline.
   */
  totalSavingMicroUsd: number;
  /**
   * Percentage saving vs. uncached baseline (0–100).
   * Negative means caching is costing more than it saves.
   */
  savingPct: number;
  /**
   * Breakpoint placement advice. Lists the strategies ordered from best to
   * worst ROI so the caller can decide how to configure `cacheBreakpoints()`.
   */
  breakpointAdvice: string[];
}

// ---------------------------------------------------------------------------
// Pricing resolver
// ---------------------------------------------------------------------------

/** Pricing resolved for a specific provider + model combination. */
interface ResolvedPricing {
  inputPricePerMToken: number;
  outputPricePerMToken: number;
}

/**
 * Resolve input/output pricing for a provider+model combination.
 *
 * Priority: model-specific EXTENDED_CONTEXT_MODELS → PROVIDER_PRICING family
 * rate → conservative fallback (anthropic family rate).
 */
export function resolveModelPricing(
  provider: string,
  modelId: string,
): ResolvedPricing {
  // 1. Exact model-level pricing from EXTENDED_CONTEXT_MODELS
  const lower = modelId.toLowerCase();
  for (const [key, model] of Object.entries(EXTENDED_CONTEXT_MODELS)) {
    if (lower === key || lower.includes(key) || key.includes(lower)) {
      return {
        inputPricePerMToken:  model.pricing.inputPricePerMToken,
        outputPricePerMToken: model.pricing.outputPricePerMToken,
      };
    }
  }

  // 2. Provider-family pricing
  const lowerProvider = provider.toLowerCase();
  for (const [key, pricing] of Object.entries(PROVIDER_PRICING)) {
    if (lowerProvider.includes(key)) {
      return {
        inputPricePerMToken:  pricing.inputPerMToken,
        outputPricePerMToken: pricing.outputPerMToken,
      };
    }
  }

  // 3. Conservative fallback (anthropic claude-sonnet-4 rate)
  return { inputPricePerMToken: 3.0, outputPricePerMToken: 15.0 };
}

// ---------------------------------------------------------------------------
// Providers that support prompt caching
// ---------------------------------------------------------------------------

const CACHING_PROVIDERS = new Set(["anthropic", "openai"]);

function providerSupportsCaching(provider: string): boolean {
  const lower = provider.toLowerCase();
  for (const p of CACHING_PROVIDERS) {
    if (lower.includes(p)) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Strategy heuristics
// ---------------------------------------------------------------------------

const ALL_STRATEGIES: CacheStrategy[] = [
  "system-only",
  "system+context",
  "multi-turn",
  "tools-only",
  "full",
  "none",
];

/**
 * Heuristic default strategy for a provider when there is insufficient data.
 *
 * - No caching support → "none".
 * - Anthropic/OpenAI with no data → "system-only" (safest / cheapest write).
 */
function heuristicDefaultStrategy(provider: string): CacheStrategy {
  if (!providerSupportsCaching(provider)) return "none";
  return "system-only";
}

// ---------------------------------------------------------------------------
// CacheOptimizer
// ---------------------------------------------------------------------------

/**
 * Multi-tier prompt caching cost analyzer and auto-breakpoint optimizer.
 *
 * ### Typical usage
 * ```ts
 * const optimizer = new CacheOptimizer({ cwd, provider: "anthropic", modelId: "claude-sonnet-4" });
 *
 * // After each API call:
 * await optimizer.recordResponse(response, "system+context");
 *
 * // Before each new request — get the currently-recommended strategy:
 * const strategy = await optimizer.recommendStrategy();
 *
 * // Generate a full report:
 * const report = await optimizer.generateReport();
 * ```
 */
export class CacheOptimizer {
  private readonly cwd: string;
  private readonly provider: string;
  private readonly modelId: string;
  private readonly windowSize: number;
  private readonly minSamples: number;
  private readonly sessionId: string | undefined;

  constructor(options: CacheOptimizerOptions = {}) {
    this.cwd       = options.cwd       ?? process.cwd();
    this.provider  = options.provider  ?? "anthropic";
    this.modelId   = options.modelId   ?? "claude-sonnet-4";
    this.windowSize = options.windowSize ?? 100;
    this.minSamples = options.minSamplesForLearning ?? 5;
    this.sessionId = options.sessionId;
  }

  // -------------------------------------------------------------------------
  // Recording
  // -------------------------------------------------------------------------

  /**
   * Record the cache metadata from an Anthropic MessageResponse and append it
   * to the JSONL audit trail.
   *
   * Call this immediately after each API response to build the learning corpus.
   *
   * @param response  Raw MessageResponse (or the usage sub-object from the SDK).
   * @param strategy  The breakpoint strategy that was active for this request.
   * @returns         The generated audit entry ID.
   */
  async recordResponse(
    response: MessageResponse,
    strategy: CacheStrategy,
  ): Promise<string> {
    const pricing = resolveModelPricing(this.provider, this.modelId);

    const inputTokens         = response.input_tokens;
    const cacheCreationTokens = response.cache_creation_input_tokens ?? 0;
    const cacheReadTokens     = response.cache_read_input_tokens     ?? 0;
    const outputTokens        = response.output_tokens;

    const effectiveCostMicroUsd = computeEffectiveCostMicroUsd(
      inputTokens,
      cacheCreationTokens,
      cacheReadTokens,
      outputTokens,
      pricing.inputPricePerMToken,
      pricing.outputPricePerMToken,
    );

    const baselineCostMicroUsd = computeBaselineCostMicroUsd(
      inputTokens,
      cacheCreationTokens,
      cacheReadTokens,
      outputTokens,
      pricing.inputPricePerMToken,
      pricing.outputPricePerMToken,
    );

    return appendCacheAudit(this.cwd, {
      provider:  this.provider,
      modelId:   this.modelId,
      strategy,
      inputTokens,
      cacheCreationTokens,
      cacheReadTokens,
      outputTokens,
      effectiveCostMicroUsd,
      baselineCostMicroUsd,
      ...(this.sessionId !== undefined ? { sessionId: this.sessionId } : {}),
    });
  }

  // -------------------------------------------------------------------------
  // Strategy recommendation
  // -------------------------------------------------------------------------

  /**
   * Recommend the best cache breakpoint strategy based on accumulated audit data.
   *
   * Falls back to a heuristic default when there is insufficient data.
   */
  async recommendStrategy(): Promise<CacheStrategy> {
    if (!providerSupportsCaching(this.provider)) return "none";

    const entries = await loadCacheAudit(this.cwd);
    const filtered = this._filterEntries(entries);

    return this._selectBestStrategy(filtered);
  }

  // -------------------------------------------------------------------------
  // Provider-switch integration
  // -------------------------------------------------------------------------

  /**
   * Generate a recommendation when switching from one provider to another.
   *
   * Integrates with `rebalanceBudgetOnSwitch` from the budget module to
   * compute both the context-budget rebalance and the optimal cache strategy
   * for the new provider.
   *
   * @param toProvider      The provider being switched to.
   * @param currentMessages Current messages in the context (for budget sizing).
   */
  async recommendOnProviderSwitch(
    toProvider: string,
    currentMessages: { role: string; content: string | unknown[] }[],
  ): Promise<ProviderSwitchRecommendation> {
    const budgetRebalance = rebalanceBudgetOnSwitch(
      this.provider,
      toProvider,
      currentMessages,
    );

    let suggestedStrategy: CacheStrategy;
    let rationale: string;

    if (!providerSupportsCaching(toProvider)) {
      suggestedStrategy = "none";
      rationale = `${toProvider} does not support prompt caching — disabling cache breakpoints saves unnecessary overhead.`;
    } else {
      // Load audit data and see if we have enough samples for the new provider
      const entries = await loadCacheAudit(this.cwd);
      const forNewProvider = entries.filter(
        (e) => e.provider.toLowerCase().includes(toProvider.toLowerCase()),
      );

      if (forNewProvider.length >= this.minSamples) {
        suggestedStrategy = this._selectBestStrategy(forNewProvider.slice(-this.windowSize));
        rationale = `Based on ${forNewProvider.length} prior audit records for ${toProvider}, strategy "${suggestedStrategy}" yields the best ROI.`;
      } else {
        // No prior data for this provider — use heuristic
        suggestedStrategy = heuristicDefaultStrategy(toProvider);
        rationale = `Insufficient audit data for ${toProvider} (${forNewProvider.length} records); defaulting to conservative "${suggestedStrategy}" strategy.`;
      }
    }

    return {
      fromProvider: this.provider,
      toProvider,
      budgetRebalance,
      suggestedStrategy,
      rationale,
    };
  }

  // -------------------------------------------------------------------------
  // Report generation
  // -------------------------------------------------------------------------

  /**
   * Generate a full `CacheOptimizationReport` summarizing ROI per strategy,
   * overall savings, and breakpoint placement advice.
   */
  async generateReport(): Promise<CacheOptimizationReport> {
    const entries = await loadCacheAudit(this.cwd);
    const filtered = this._filterEntries(entries);
    const window   = filtered.slice(-this.windowSize);

    const strategyROI: StrategyROI[] = [];
    let bestStrategy: CacheStrategy | null = null;
    let bestAvgSaving = -Infinity;

    for (const strategy of ALL_STRATEGIES) {
      const summary: CacheAuditSummary | null = summarizeAuditWindow(window, strategy, this.windowSize);

      if (summary === null) {
        // No data for this strategy yet — include a zero-row so the report is complete
        strategyROI.push({
          strategy,
          sampleCount: 0,
          avgSavingMicroUsd: 0,
          totalSavingMicroUsd: 0,
          hitRate: 0,
          recommended: false,
        });
        continue;
      }

      const totalSaving = summary.totalBaselineCostMicroUsd - summary.totalEffectiveCostMicroUsd;
      const avgSaving   = summary.avgSavingMicroUsd;

      if (
        summary.sampleCount >= this.minSamples &&
        avgSaving > bestAvgSaving
      ) {
        bestAvgSaving = avgSaving;
        bestStrategy  = strategy;
      }

      strategyROI.push({
        strategy,
        sampleCount:        summary.sampleCount,
        avgSavingMicroUsd:  avgSaving,
        totalSavingMicroUsd: totalSaving,
        hitRate:            summary.hitRate,
        recommended:        false, // patched below
      });
    }

    // Mark the recommended strategy
    if (bestStrategy !== null) {
      for (const row of strategyROI) {
        row.recommended = row.strategy === bestStrategy;
      }
    }

    // Overall totals
    let totalEffective = 0;
    let totalBaseline  = 0;
    for (const e of window) {
      totalEffective += e.effectiveCostMicroUsd;
      totalBaseline  += e.baselineCostMicroUsd;
    }
    const totalSavingMicroUsd = totalBaseline - totalEffective;
    const savingPct = totalBaseline > 0
      ? Math.round((totalSavingMicroUsd / totalBaseline) * 10000) / 100
      : 0;

    // Breakpoint advice: sort strategies by ROI (highest to lowest, exclude "none")
    const sorted = [...strategyROI]
      .filter((r) => r.strategy !== "none" && r.sampleCount > 0)
      .sort((a, b) => b.avgSavingMicroUsd - a.avgSavingMicroUsd);

    const breakpointAdvice = sorted.map((r) => {
      const hitPct = Math.round(r.hitRate * 100);
      const usdPerReq = (r.avgSavingMicroUsd / 1_000_000).toFixed(6);
      return `${r.strategy}: ${hitPct}% hit-rate, avg saving $${usdPerReq}/req (${r.sampleCount} samples)`;
    });

    if (breakpointAdvice.length === 0) {
      breakpointAdvice.push(
        `No cache data collected yet. Start with "${heuristicDefaultStrategy(this.provider)}" strategy.`,
      );
    }

    return {
      generatedAt:        new Date().toISOString(),
      provider:           this.provider,
      modelId:            this.modelId,
      totalAuditEntries:  window.length,
      strategyROI,
      recommendedStrategy: bestStrategy,
      totalSavingMicroUsd,
      savingPct,
      breakpointAdvice,
    };
  }

  // -------------------------------------------------------------------------
  // Cost delta analysis (static utility)
  // -------------------------------------------------------------------------

  /**
   * Compute the cost delta between two strategies for a hypothetical request.
   *
   * Useful for offline analysis: "if I switch from system-only to full, how much
   * more/less would request X cost?"
   *
   * @param response         MessageResponse from the API.
   * @param fromStrategy     Currently active strategy.
   * @param toStrategy       Candidate alternative strategy.
   * @param provider         Provider name for pricing lookup.
   * @param modelId          Model ID for pricing lookup.
   * @returns                Cost delta in micro-USD (negative = toStrategy is cheaper).
   */
  static computeStrategyDelta(
    response: MessageResponse,
    fromStrategy: CacheStrategy,
    toStrategy: CacheStrategy,
    provider = "anthropic",
    modelId  = "claude-sonnet-4",
  ): number {
    const pricing = resolveModelPricing(provider, modelId);

    const inputTokens         = response.input_tokens;
    const cacheCreationTokens = response.cache_creation_input_tokens ?? 0;
    const cacheReadTokens     = response.cache_read_input_tokens     ?? 0;
    const outputTokens        = response.output_tokens;

    // Both strategies share the same raw token counts; the delta comes from
    // whether those tokens are attributed to cache-write, cache-read, or uncached.
    //
    // For this analysis we use the from-strategy as the actual observed cost and
    // the baseline (no-cache) cost as the reference for the to-strategy.
    // In practice a full simulation would require re-running the request;
    // this approximation is useful for direction/magnitude guidance.

    const fromCost = computeEffectiveCostMicroUsd(
      inputTokens,
      cacheCreationTokens,
      cacheReadTokens,
      outputTokens,
      pricing.inputPricePerMToken,
      pricing.outputPricePerMToken,
    );

    // "none" strategy → baseline cost (no caching).
    // Any other strategy: use observed cost from response (best available proxy).
    const toCost = toStrategy === "none"
      ? computeBaselineCostMicroUsd(
          inputTokens,
          cacheCreationTokens,
          cacheReadTokens,
          outputTokens,
          pricing.inputPricePerMToken,
          pricing.outputPricePerMToken,
        )
      : fromStrategy === "none"
        ? computeEffectiveCostMicroUsd(
            inputTokens,
            cacheCreationTokens,
            cacheReadTokens,
            outputTokens,
            pricing.inputPricePerMToken,
            pricing.outputPricePerMToken,
          )
        : fromCost; // same observed cost; caller would need real data to diff

    return toCost - fromCost;
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  /**
   * Filter audit entries to those matching this optimizer's provider + model.
   * When both are the defaults, all entries are included for maximum learning.
   */
  private _filterEntries(entries: CacheAuditEntry[]): CacheAuditEntry[] {
    return entries.filter((e) => {
      const providerMatch =
        e.provider.toLowerCase().includes(this.provider.toLowerCase()) ||
        this.provider.toLowerCase().includes(e.provider.toLowerCase());
      const modelMatch =
        e.modelId.toLowerCase().includes(this.modelId.toLowerCase()) ||
        this.modelId.toLowerCase().includes(e.modelId.toLowerCase());
      return providerMatch && modelMatch;
    });
  }

  /**
   * Select the best strategy from a slice of audit entries.
   *
   * Scoring: average saving per request (positive = profitable).
   * Requires at least `minSamples` for a strategy to be eligible.
   * Falls back to heuristic default when no strategy meets the threshold.
   */
  private _selectBestStrategy(entries: CacheAuditEntry[]): CacheStrategy {
    let bestStrategy: CacheStrategy = heuristicDefaultStrategy(this.provider);
    let bestScore = -Infinity;

    for (const strategy of ALL_STRATEGIES) {
      if (strategy === "none") continue; // "none" is only recommended explicitly
      const summary = summarizeAuditWindow(entries, strategy, this.windowSize);
      if (!summary || summary.sampleCount < this.minSamples) continue;
      if (summary.avgSavingMicroUsd > bestScore) {
        bestScore    = summary.avgSavingMicroUsd;
        bestStrategy = strategy;
      }
    }

    return bestStrategy;
  }
}
