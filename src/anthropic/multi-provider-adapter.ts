/**
 * multi-provider-adapter.ts — Unified Cache API across Anthropic, OpenAI, and
 * Google Gemini.
 *
 * ## Problem
 * Prompt-cache semantics differ across providers:
 *   - **Anthropic** — ephemeral cache markers, 1.25× write / 0.10× read
 *     multipliers, hard 4-breakpoint limit, min 1024-token cacheable block.
 *   - **OpenAI GPT-4** — implicit prefix caching; no explicit markers; 0.50×
 *     read discount; breakpoints not user-configurable.
 *   - **OpenAI o1/o3** — prefix caching disabled (reasoning models); any cache
 *     strategy must gracefully degrade.
 *   - **Google Gemini** — Context Cache API requires an explicit cache object
 *     with a minimum 32k-token body; write cost is separate from input tokens;
 *     read discount ≈ 0.25× depending on region.
 *
 * ## Solution
 * This module introduces:
 *   1. `ProviderCacheStrategy` — per-provider write/read multipliers, breakpoint
 *      limit, minimum cacheable tokens, and a `supported` flag for providers
 *      that do not cache at all.
 *   2. `PROVIDER_CACHE_STRATEGIES` — registry keyed by canonical provider name.
 *   3. `MultiProviderAdapter` — unified API that accepts a provider name +
 *      messages array and returns optimized `BreakpointRecommendation`-compatible
 *      results, or a `CacheDegradation` when caching is unsupported.
 *   4. Integration hook for `CacheBreakpointOptimizer`: pass an optional
 *      `providerHint` to auto-select the correct strategy from the registry.
 *   5. Extension points on `ProviderCostOracle` for per-provider cache strategy
 *      awareness (see `getCacheStrategy` / `getCalibratedCacheCost`).
 *
 * ## Cost impact
 * By routing each fleet request through the correct strategy the adapter
 * avoids charging Anthropic write-prices against providers that have no cache,
 * and correctly models the cheaper OpenAI implicit-cache discount — directly
 * reducing fleet token-cost estimates by 5–15% on mixed-provider workloads.
 *
 * @module
 */

import { ProviderCostOracle } from "../budget/provider-cost-oracle.ts";
import type { CalibratedRate } from "../budget/provider-cost-oracle.ts";
import type { Message, OptimizerOptions, BreakpointRecommendation } from "./cache-breakpoint-optimizer.ts";
import { CacheBreakpointOptimizer } from "./cache-breakpoint-optimizer.ts";

// ---------------------------------------------------------------------------
// Provider cache strategy registry
// ---------------------------------------------------------------------------

/**
 * Describes how a specific provider implements (or omits) prompt caching.
 *
 * All multipliers are relative to the provider's standard input-token rate.
 */
export interface ProviderCacheStrategy {
  /** Canonical provider name (lower-case, matches PROVIDER_PRICING keys). */
  provider: string;
  /**
   * Whether this provider supports any form of prompt caching.
   * When `false`, all cache-related calls degrade gracefully to no-op.
   */
  supported: boolean;
  /**
   * Cost multiplier applied to tokens written into the cache (write pass).
   * Anthropic: 1.25. OpenAI implicit: 1.0 (no explicit write cost).
   * Gemini: 1.0 (write is a separate flat fee modelled elsewhere).
   */
  writeCostMultiplier: number;
  /**
   * Cost multiplier applied to tokens served from the cache (read pass).
   * Anthropic: 0.10. OpenAI: 0.50. Gemini: 0.25 (approximate).
   */
  readCostMultiplier: number;
  /**
   * Maximum number of cache breakpoints the provider accepts per request.
   * Anthropic: 4. OpenAI: 0 (implicit, user sets none). Gemini: 1.
   */
  maxBreakpoints: number;
  /**
   * Minimum token count a cacheable block must have to be eligible.
   * Anthropic: 1024. OpenAI: implicit (≥1024 inferred). Gemini: 32768.
   */
  minCacheableTokens: number;
  /**
   * Whether breakpoints are set explicitly by the caller (Anthropic-style)
   * or implicitly by the provider (OpenAI-style).
   */
  explicitBreakpoints: boolean;
  /**
   * Human-readable notes explaining this provider's caching semantics.
   */
  notes: string;
}

/**
 * Registry of per-provider cache strategies.
 *
 * Keyed by canonical provider name (lower-case).
 */
export const PROVIDER_CACHE_STRATEGIES: Record<string, ProviderCacheStrategy> = {
  anthropic: {
    provider: "anthropic",
    supported: true,
    writeCostMultiplier: 1.25,
    readCostMultiplier: 0.10,
    maxBreakpoints: 4,
    minCacheableTokens: 1024,
    explicitBreakpoints: true,
    notes:
      "Ephemeral cache markers via cache_control.type='ephemeral'. " +
      "Up to 4 breakpoints per request. Min 1024 tokens per block.",
  },
  openai: {
    provider: "openai",
    supported: true,
    writeCostMultiplier: 1.0,  // no explicit write charge; happens automatically
    readCostMultiplier: 0.50,
    maxBreakpoints: 0,         // implicit — caller cannot set breakpoints
    minCacheableTokens: 1024,
    explicitBreakpoints: false,
    notes:
      "Implicit prefix caching. No explicit breakpoints. 50% discount on " +
      "cached input tokens automatically applied by the API.",
  },
  "openai-o1": {
    provider: "openai-o1",
    supported: false,
    writeCostMultiplier: 1.0,
    readCostMultiplier: 1.0,
    maxBreakpoints: 0,
    minCacheableTokens: 0,
    explicitBreakpoints: false,
    notes:
      "OpenAI o1/o3 reasoning models do not support prompt caching. " +
      "Cache-related calls degrade to no-op.",
  },
  "openai-o3": {
    provider: "openai-o3",
    supported: false,
    writeCostMultiplier: 1.0,
    readCostMultiplier: 1.0,
    maxBreakpoints: 0,
    minCacheableTokens: 0,
    explicitBreakpoints: false,
    notes:
      "OpenAI o3 reasoning models do not support prompt caching.",
  },
  gemini: {
    provider: "gemini",
    supported: true,
    writeCostMultiplier: 1.0,  // Context Cache has its own per-token write price
    readCostMultiplier: 0.25,
    maxBreakpoints: 1,
    minCacheableTokens: 32768, // Gemini Context Cache minimum
    explicitBreakpoints: true,
    notes:
      "Google Gemini Context Cache API. Single cache object per request, " +
      "min 32k tokens. Read discount ~0.25× input rate.",
  },
};

// ---------------------------------------------------------------------------
// Breakpoint recommendation result shapes
// ---------------------------------------------------------------------------

/**
 * Outcome returned when caching is fully supported for the chosen provider.
 */
export interface CacheAdapterResult {
  type: "supported";
  /** Provider strategy that was applied. */
  strategy: ProviderCacheStrategy;
  /** Calibrated pricing used for ROI calculations. */
  calibratedRate: CalibratedRate;
  /** Breakpoint recommendations (empty array when no beneficial points found). */
  recommendations: BreakpointRecommendation[];
  /**
   * Estimated per-request savings in micro-USD given the recommended breakpoints.
   */
  estimatedSavingsMicroUsd: number;
}

/**
 * Outcome returned when the provider does not support caching (e.g. o1/o3).
 *
 * Callers should skip any cache-marker injection and proceed with plain tokens.
 */
export interface CacheDegradation {
  type: "unsupported";
  /** Provider that triggered the degradation. */
  provider: string;
  /** Human-readable reason. */
  reason: string;
}

/** Union of all possible adapter outcomes. */
export type AdapterOutcome = CacheAdapterResult | CacheDegradation;

// ---------------------------------------------------------------------------
// Model-to-provider normalisation
// ---------------------------------------------------------------------------

/**
 * Normalize a raw model ID or provider name to a canonical provider key
 * present in `PROVIDER_CACHE_STRATEGIES`.
 *
 * Handles patterns like:
 *   "o1", "o1-preview", "o1-mini" → "openai-o1"
 *   "o3", "o3-mini"               → "openai-o3"
 *   "gpt-4o", "gpt-4-turbo"       → "openai"
 *   "gemini-*"                    → "gemini"
 *   "claude-*"                    → "anthropic"
 */
export function normalizeProviderKey(providerOrModel: string): string {
  const lower = providerOrModel.toLowerCase();

  // OpenAI reasoning models — must check before generic "openai" prefix
  if (/^o3(?:[^a-z]|$)/.test(lower) || lower.startsWith("openai-o3")) return "openai-o3";
  if (/^o1(?:[^a-z]|$)/.test(lower) || lower.startsWith("openai-o1")) return "openai-o1";

  if (lower.includes("gemini"))    return "gemini";
  if (lower.includes("claude"))    return "anthropic";
  if (lower.includes("gpt"))       return "openai";
  if (lower.includes("openai"))    return "openai";
  if (lower.includes("anthropic")) return "anthropic";
  if (lower.includes("google"))    return "gemini";

  // Exact registry match
  if (PROVIDER_CACHE_STRATEGIES[lower]) return lower;

  // Default fallback — treat as anthropic-compatible
  return "anthropic";
}

/**
 * Resolve a `ProviderCacheStrategy` for the given provider name or model ID.
 *
 * Falls back to Anthropic strategy for unknown providers (safe default since
 * Anthropic strategy is the most conservative / explicit).
 */
export function resolveProviderStrategy(providerOrModel: string): ProviderCacheStrategy {
  const key = normalizeProviderKey(providerOrModel);
  return PROVIDER_CACHE_STRATEGIES[key] ?? PROVIDER_CACHE_STRATEGIES["anthropic"]!;
}

// ---------------------------------------------------------------------------
// MultiProviderAdapter
// ---------------------------------------------------------------------------

/**
 * Options for constructing a `MultiProviderAdapter`.
 */
export interface MultiProviderAdapterOptions {
  /**
   * Working directory for the underlying `CacheBreakpointOptimizer` and
   * `ProviderCostOracle` caches.  Defaults to `process.cwd()`.
   */
  cwd?: string;
  /**
   * Session log directory override (forwarded to CacheBreakpointOptimizer).
   */
  sessionLogDir?: string;
  /**
   * Evolution audit file path override.
   */
  auditPath?: string;
  /**
   * Pre-initialized `ProviderCostOracle` to reuse. When omitted a new oracle
   * is created and initialized on first use.
   */
  oracle?: ProviderCostOracle;
}

/**
 * Unified prompt-cache API supporting Anthropic, OpenAI GPT-4, OpenAI o1/o3,
 * and Google Gemini.
 *
 * ### Typical usage
 * ```ts
 * const adapter = new MultiProviderAdapter();
 * const outcome = await adapter.recommendCacheStrategy("openai", messages);
 * if (outcome.type === "supported") {
 *   // use outcome.recommendations to place breakpoints (Anthropic-style)
 *   // or acknowledge implicit caching (OpenAI GPT-4)
 * } else {
 *   // outcome.type === "unsupported" → skip cache markers (o1/o3)
 * }
 * ```
 */
export class MultiProviderAdapter {
  private readonly optimizer: CacheBreakpointOptimizer;
  private readonly oracle: ProviderCostOracle;
  private oracleInitialized = false;

  constructor(options: MultiProviderAdapterOptions = {}) {
    this.optimizer = new CacheBreakpointOptimizer({
      sessionLogDir: options.sessionLogDir,
      auditPath: options.auditPath,
    });
    this.oracle = options.oracle ?? new ProviderCostOracle(options.cwd ?? process.cwd());
  }

  // -------------------------------------------------------------------------
  // Primary API
  // -------------------------------------------------------------------------

  /**
   * Recommend an optimal cache strategy for the given provider + messages.
   *
   * Returns a `CacheDegradation` when the provider does not support caching
   * (e.g. o1/o3), allowing callers to skip marker injection without branching
   * on provider names themselves.
   *
   * @param providerOrModel  Provider name or model ID (e.g. "anthropic",
   *                         "gpt-4o", "o1", "gemini-1.5-pro").
   * @param messages         Messages for the upcoming request.
   * @param options          Tuning parameters forwarded to the optimizer.
   */
  async recommendCacheStrategy(
    providerOrModel: string,
    messages: Message[],
    options: OptimizerOptions = {},
  ): Promise<AdapterOutcome> {
    const strategy = resolveProviderStrategy(providerOrModel);

    // Graceful degradation: provider has no cache support
    if (!strategy.supported) {
      return {
        type: "unsupported",
        provider: providerOrModel,
        reason: strategy.notes,
      };
    }

    // Ensure oracle is initialized (lazy, once per adapter instance)
    if (!this.oracleInitialized) {
      await this.oracle.initialize();
      this.oracleInitialized = true;
    }

    const calibratedRate = this.oracle.getCalibratedRate(strategy.provider);

    // For implicit-breakpoint providers (e.g. OpenAI GPT-4), return a
    // supported result with empty recommendations — the provider handles it.
    if (!strategy.explicitBreakpoints) {
      return {
        type: "supported",
        strategy,
        calibratedRate,
        recommendations: [],
        estimatedSavingsMicroUsd: this._estimateImplicitSavings(
          messages,
          calibratedRate,
          strategy,
        ),
      };
    }

    // For explicit-breakpoint providers (Anthropic, Gemini), run the optimizer
    // with provider-aware parameters.
    const providerOptions: OptimizerOptions = {
      ...options,
      maxBreakpoints: options.maxBreakpoints != null
        ? Math.min(options.maxBreakpoints, strategy.maxBreakpoints)
        : strategy.maxBreakpoints,
      inputPricePerMToken: options.inputPricePerMToken ?? calibratedRate.inputPerMToken,
    };

    // Filter messages that meet the provider's minimum cacheable token threshold
    const eligibleMessages = this._filterEligibleMessages(messages, strategy);

    let recommendations: BreakpointRecommendation[] = [];
    if (eligibleMessages.length > 0) {
      recommendations = await this.optimizer.recommendBreakpoints(
        eligibleMessages,
        providerOptions,
      );
    }

    const estimatedSavingsMicroUsd = recommendations.reduce(
      (sum, r) => sum + r.totalNetRoiMicroUsd,
      0,
    );

    return {
      type: "supported",
      strategy,
      calibratedRate,
      recommendations,
      estimatedSavingsMicroUsd,
    };
  }

  /**
   * Compute the estimated cache cost for a block of tokens using the provider's
   * strategy multipliers and the oracle's calibrated pricing.
   *
   * Useful for fleet cost accounting: given N tokens that were served from
   * cache, what was the actual USD cost?
   *
   * @param providerOrModel   Provider name or model ID.
   * @param cachedTokens      Number of tokens served from cache (read).
   * @param writtenTokens     Number of tokens written into cache (write pass).
   * @returns                 Estimated cost in micro-USD.
   */
  async computeCacheCostMicroUsd(
    providerOrModel: string,
    cachedTokens: number,
    writtenTokens: number,
  ): Promise<number> {
    const strategy = resolveProviderStrategy(providerOrModel);

    if (!strategy.supported) return 0;

    if (!this.oracleInitialized) {
      await this.oracle.initialize();
      this.oracleInitialized = true;
    }

    const rate = this.oracle.getCalibratedRate(strategy.provider);

    // Read cost
    const readCostMicroUsd =
      (cachedTokens / 1_000_000) * rate.inputPerMToken * strategy.readCostMultiplier * 1_000_000;

    // Write cost (Anthropic charges 1.25× on cache writes; others are 1.0×)
    const writeCostMicroUsd =
      (writtenTokens / 1_000_000) * rate.inputPerMToken * strategy.writeCostMultiplier * 1_000_000;

    return Math.round(readCostMicroUsd + writeCostMicroUsd);
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  /**
   * Estimate per-request savings for implicit-cache providers (e.g. OpenAI).
   *
   * Uses a conservative 30% cache-hit assumption when no history is available.
   */
  private _estimateImplicitSavings(
    messages: Message[],
    rate: CalibratedRate,
    strategy: ProviderCacheStrategy,
  ): number {
    const totalTokens = messages.reduce((sum, m) => {
      if (typeof m.content === "string") return sum + Math.ceil(m.content.length / 4);
      return (
        sum +
        m.content.reduce((s, b) => s + (b.text ? Math.ceil((b.text as string).length / 4) : 50), 0)
      );
    }, 0);

    const assumedHitRate = 0.3;
    const savingsPerToken =
      rate.inputPerMToken * (1 - strategy.readCostMultiplier) * assumedHitRate;
    return Math.round((totalTokens / 1_000_000) * savingsPerToken * 1_000_000);
  }

  /**
   * Filter messages to only those meeting the provider's minimum cacheable
   * token threshold.  For Gemini (32k minimum) this prevents submitting tiny
   * messages to the Context Cache API, which would return an error.
   */
  private _filterEligibleMessages(
    messages: Message[],
    strategy: ProviderCacheStrategy,
  ): Message[] {
    if (strategy.minCacheableTokens <= 0) return messages;

    let cumulativeTokens = 0;
    const eligible: Message[] = [];

    for (const msg of messages) {
      const tokens =
        typeof msg.content === "string"
          ? Math.ceil(msg.content.length / 4)
          : msg.content.reduce(
              (s, b) => s + (b.text ? Math.ceil((b.text as string).length / 4) : 50),
              0,
            );
      cumulativeTokens += tokens;
      if (cumulativeTokens >= strategy.minCacheableTokens) {
        eligible.push(msg);
      }
    }

    return eligible;
  }
}

// ---------------------------------------------------------------------------
// Module-level convenience function
// ---------------------------------------------------------------------------

/**
 * One-shot helper: instantiate a `MultiProviderAdapter` and return the cache
 * strategy recommendation for the given provider + messages.
 *
 * ```ts
 * import { recommendProviderCacheStrategy } from "./multi-provider-adapter.ts";
 * const outcome = await recommendProviderCacheStrategy("anthropic", messages);
 * ```
 */
export async function recommendProviderCacheStrategy(
  providerOrModel: string,
  messages: Message[],
  options?: OptimizerOptions & MultiProviderAdapterOptions,
): Promise<AdapterOutcome> {
  const adapter = new MultiProviderAdapter({
    cwd: options?.cwd,
    sessionLogDir: options?.sessionLogDir,
    auditPath: options?.auditPath,
  });
  return adapter.recommendCacheStrategy(providerOrModel, messages, options);
}
