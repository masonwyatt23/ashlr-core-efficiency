/**
 * Provider-aware prompt budgeting — extracted from ashlrcode.
 *
 * Behavior preserved exactly: substring match + 100K default fallback,
 * matching ashlrcode/src/agent/context.ts behavior so tests pass against
 * either consumer.
 *
 * ### Consolidation integration
 *
 * `rebalanceBudgetOnSwitch` accepts an optional `cwd` parameter. When
 * provided it calls `rebalanceBudgetOnSwitchHook` from
 * `../compression/consolidation.ts` to emit a `CompressorRecommendation`
 * based on cross-session history. The recommendation is attached to the
 * returned `BudgetRebalanceResult` as `compressionRecommendation`.
 *
 * ### Multi-objective learner
 *
 * `BudgetMultiObjectiveLearner` and related types are re-exported from
 * `./multi-objective-learner.ts` for convenience. The learner ingests
 * historical session data and emits Pareto-optimal budget allocations
 * balancing USD cost against end-user latency.
 */

// Re-export the provider cost ratio oracle.
export type {
  CalibratedRate,
  ProviderRateRecord,
} from "./provider-cost-oracle.ts";
export {
  EMA_ALPHA,
  MAX_CACHE_AGE_MS,
  MAX_CACHE_RECORDS,
  ProviderCostOracle,
} from "./provider-cost-oracle.ts";

// Re-export the adaptive multi-objective learner and its public API.
export type {
  BudgetAllocation,
  BudgetAllocationOptions,
  BudgetCalibrationRecord,
  BudgetDiagnostics,
  MessageCountBucket,
  ParetoPoint,
  SessionSummary,
} from "./multi-objective-learner.ts";
export {
  BudgetMultiObjectiveLearner,
  bucketMessageCount,
  computeParetoFrontier,
} from "./multi-objective-learner.ts";

// Re-export the latency-aware context window allocator.
export type {
  FittedLatencyModel,
  LatencyAwareAllocation,
  LatencyAwareAllocationRecord,
  LatencyCostRecord,
} from "./latency-aware-allocator.ts";
export {
  CONTEXT_WINDOW_STEPS,
  LatencyAwareAllocator,
} from "./latency-aware-allocator.ts";

/** System prompt gets this fraction of the provider's context limit. */
export const SYSTEM_PROMPT_BUDGET_RATIO = 0.05;
/** Hard cap so 2M-context providers (xAI) don't give a runaway budget. */
export const SYSTEM_PROMPT_BUDGET_CAP = 50_000;
/** Fallback when the provider name isn't recognized. Matches ashlrcode. */
export const DEFAULT_CONTEXT_LIMIT = 100_000;

/** Known provider context limits (tokens). */
export const PROVIDER_CONTEXT_LIMITS: Record<string, number> = {
  xai: 2_000_000,
  anthropic: 200_000,
  openai: 128_000,
  ollama: 32_000, // Conservative default; most local models are 4K-128K
  groq: 128_000,
  deepseek: 128_000,
};

// ---------- Extended-context model registry ----------

/**
 * Input/output pricing tier for a model (USD per million tokens).
 */
export interface ModelPricingTier {
  /** USD per million input tokens billed at standard rate. */
  inputPricePerMToken: number;
  /** USD per million output tokens. */
  outputPricePerMToken: number;
  /**
   * USD per million tokens written into the prompt cache (1.25× input rate).
   * Omitted for models that do not support prompt caching.
   */
  cacheWritePricePerMToken?: number;
  /**
   * USD per million tokens read from the prompt cache (0.1× input rate).
   * Omitted for models that do not support prompt caching.
   */
  cacheReadPricePerMToken?: number;
}

/**
 * Descriptor for a model that supports an extended (non-standard) context
 * window or occupies its own pricing tier distinct from its provider family.
 */
export interface ExtendedContextModel {
  /** Canonical model identifier (e.g. "claude-opus-4", "o1-preview"). */
  modelId: string;
  /** Provider family this model belongs to. */
  provider: string;
  /** Maximum context window in tokens. */
  contextLimit: number;
  /** Pricing for this specific model. */
  pricing: ModelPricingTier;
}

/**
 * Registry of extended-context / custom-priced models.
 *
 * Keyed by canonical model ID.  The lookup helpers below resolve by exact
 * match first, then fall back to provider-level limits from
 * `PROVIDER_CONTEXT_LIMITS`.
 *
 * Pricing data sourced from public Anthropic and OpenAI pricing pages.
 * Prices are in USD per million tokens.
 */
export const EXTENDED_CONTEXT_MODELS: Record<string, ExtendedContextModel> = {
  // ── Anthropic Claude ────────────────────────────────────────────────────
  "claude-opus-4": {
    modelId: "claude-opus-4",
    provider: "anthropic",
    contextLimit: 200_000,
    pricing: {
      inputPricePerMToken: 15.0,
      outputPricePerMToken: 75.0,
      cacheWritePricePerMToken: 18.75, // 15.0 × 1.25
      cacheReadPricePerMToken: 1.5,    // 15.0 × 0.10
    },
  },
  "claude-sonnet-4": {
    modelId: "claude-sonnet-4",
    provider: "anthropic",
    contextLimit: 200_000,
    pricing: {
      inputPricePerMToken: 3.0,
      outputPricePerMToken: 15.0,
      cacheWritePricePerMToken: 3.75,  // 3.0 × 1.25
      cacheReadPricePerMToken: 0.3,    // 3.0 × 0.10
    },
  },
  "claude-haiku-4": {
    modelId: "claude-haiku-4",
    provider: "anthropic",
    contextLimit: 200_000,
    pricing: {
      inputPricePerMToken: 0.8,
      outputPricePerMToken: 4.0,
      cacheWritePricePerMToken: 1.0,   // 0.8 × 1.25
      cacheReadPricePerMToken: 0.08,   // 0.8 × 0.10
    },
  },
  // ── OpenAI reasoning models ─────────────────────────────────────────────
  "o1-preview": {
    modelId: "o1-preview",
    provider: "openai",
    contextLimit: 128_000,
    pricing: {
      inputPricePerMToken: 15.0,
      outputPricePerMToken: 60.0,
      // o1-preview supports input caching (at OpenAI's 0.5× discount, not
      // Anthropic's; store the equivalent read price here for accounting).
      cacheReadPricePerMToken: 7.5,    // 15.0 × 0.50
    },
  },
  "o1": {
    modelId: "o1",
    provider: "openai",
    contextLimit: 200_000,
    pricing: {
      inputPricePerMToken: 15.0,
      outputPricePerMToken: 60.0,
      cacheReadPricePerMToken: 7.5,
    },
  },
  "o3": {
    modelId: "o3",
    provider: "openai",
    contextLimit: 200_000,
    pricing: {
      inputPricePerMToken: 10.0,
      outputPricePerMToken: 40.0,
      cacheReadPricePerMToken: 2.5,
    },
  },
};

/**
 * Look up extended-context model metadata by model ID.
 *
 * Performs an exact match first; if that fails it tries a substring match
 * across all registered model IDs (handles version suffixes like
 * "claude-opus-4-20250514").
 *
 * Returns `undefined` when the model is not in the extended registry.
 */
export function getExtendedContextModel(
  modelId: string,
): ExtendedContextModel | undefined {
  const lower = modelId.toLowerCase();
  // Exact match
  if (EXTENDED_CONTEXT_MODELS[lower]) return EXTENDED_CONTEXT_MODELS[lower];
  // Substring match (handles dated variants)
  for (const [key, model] of Object.entries(EXTENDED_CONTEXT_MODELS)) {
    if (lower.includes(key) || key.includes(lower)) return model;
  }
  return undefined;
}

/**
 * Return the context limit for a model, checking the extended registry first
 * and falling back to `getProviderContextLimit` for the provider family.
 */
export function getModelContextLimit(
  modelId: string,
  providerHint?: string,
): number {
  const extended = getExtendedContextModel(modelId);
  if (extended) return extended.contextLimit;
  if (providerHint) return getProviderContextLimit(providerHint);
  // Last resort: try to infer provider from model ID substring
  return getProviderContextLimit(modelId);
}

/**
 * Look up a provider's context token limit.
 *
 * Uses case-insensitive substring match so that compound provider names like
 * "xai-grok-4" or "anthropic-claude-sonnet" still resolve. Falls back to
 * {@link DEFAULT_CONTEXT_LIMIT} for unknown providers.
 */
export function getProviderContextLimit(providerName: string): number {
  const lower = providerName.toLowerCase();
  for (const [key, limit] of Object.entries(PROVIDER_CONTEXT_LIMITS)) {
    if (lower.includes(key)) return limit;
  }
  return DEFAULT_CONTEXT_LIMIT;
}

/**
 * Compute system-prompt token budget for a given provider.
 *
 * Default: 5% of the provider context limit, capped at 50K.
 * Replaces the inline math at `ashlrcode/src/cli.ts:274-281` and the
 * hardcoded `50_000` at `ashlrcode/src/agent/bootstrap.ts:201`.
 */
export function systemPromptBudget(
  providerName: string,
  ratio: number = SYSTEM_PROMPT_BUDGET_RATIO,
  cap: number = SYSTEM_PROMPT_BUDGET_CAP,
): number {
  const limit = getProviderContextLimit(providerName);
  return Math.min(Math.floor(limit * ratio), cap);
}

// ---------------------------------------------------------------------------
// Provider pricing registry
// ---------------------------------------------------------------------------

/**
 * Canonical per-token pricing for each provider (USD per million tokens).
 * Input/output prices reflect the most common non-reasoning model in the family.
 * Cache read prices use Anthropic's 0.1× rule where applicable; OpenAI uses 0.5×.
 * Providers without published pricing use conservative estimates based on
 * comparable model families.
 */
export const PROVIDER_PRICING: Record<
  string,
  { inputPerMToken: number; outputPerMToken: number; cacheReadPerMToken: number }
> = {
  anthropic:  { inputPerMToken: 3.0,   outputPerMToken: 15.0,  cacheReadPerMToken: 0.30  },
  openai:     { inputPerMToken: 2.5,   outputPerMToken: 10.0,  cacheReadPerMToken: 1.25  },
  xai:        { inputPerMToken: 5.0,   outputPerMToken: 15.0,  cacheReadPerMToken: 0.25  },
  groq:       { inputPerMToken: 0.27,  outputPerMToken: 0.27,  cacheReadPerMToken: 0.027 },
  deepseek:   { inputPerMToken: 0.14,  outputPerMToken: 0.28,  cacheReadPerMToken: 0.014 },
  ollama:     { inputPerMToken: 0.0,   outputPerMToken: 0.0,   cacheReadPerMToken: 0.0   },
};

// ---------------------------------------------------------------------------
// Provider cost-ratio computation
// ---------------------------------------------------------------------------

/**
 * Result of comparing per-token costs between two providers.
 *
 * Ratios < 1.0 mean the target provider is cheaper than the source on that
 * dimension. For example, `inputRatio: 0.83` when switching anthropic → openai
 * means OpenAI charges 83% of what Anthropic charges for input tokens.
 */
export interface ProviderCostRatio {
  /** inputPerMToken(to) / inputPerMToken(from). Clamped to [0, 10]. */
  inputRatio: number;
  /** outputPerMToken(to) / outputPerMToken(from). Clamped to [0, 10]. */
  outputRatio: number;
  /** cacheReadPerMToken(to) / cacheReadPerMToken(from). Clamped to [0, 10]. */
  cacheReadRatio: number;
  /**
   * Recommended tier shift as a signed integer.
   *
   *  0 — no change recommended.
   * +1 — move one tier more aggressive (e.g. tier 2 → tier 1: LLM summarisation).
   * -1 — move one tier less aggressive (e.g. tier 2 → tier 3: cheaper snipping
   *       is now sufficient because the target provider is significantly cheaper).
   *
   * The recommendation is based on the blended cost ratio:
   *   blended = (inputRatio + outputRatio) / 2
   *   blended < 0.5  → cheaper provider: relax by one tier (-1)
   *   blended > 2.0  → more expensive:   tighten by one tier (+1)
   *   otherwise      → no change (0)
   */
  recommendedTierShift: -1 | 0 | 1;
}

/**
 * Compute the relative per-token cost ratios when switching from one provider
 * to another.
 *
 * Falls back to the DEFAULT_CONTEXT_LIMIT provider's pricing (treated as
 * unknown = mid-range) when either provider is not in `PROVIDER_PRICING`.
 *
 * @param from  Source provider name (case-insensitive substring match).
 * @param to    Target provider name (case-insensitive substring match).
 * @returns     `ProviderCostRatio` with ratio values and tier recommendation.
 */
export function computeProviderCostRatio(from: string, to: string): ProviderCostRatio {
  const unknownPricing = { inputPerMToken: 3.0, outputPerMToken: 15.0, cacheReadPerMToken: 0.3 };

  function resolvePricing(name: string) {
    const lower = name.toLowerCase();
    for (const [key, pricing] of Object.entries(PROVIDER_PRICING)) {
      if (lower.includes(key)) return pricing;
    }
    return unknownPricing;
  }

  const fromP = resolvePricing(from);
  const toP   = resolvePricing(to);

  // Guard against divide-by-zero for zero-cost providers (e.g. ollama → anthropic).
  function safeRatio(numerator: number, denominator: number): number {
    if (denominator === 0) return numerator === 0 ? 1.0 : 10.0;
    return Math.min(10.0, Math.max(0, numerator / denominator));
  }

  const inputRatio     = safeRatio(toP.inputPerMToken,     fromP.inputPerMToken);
  const outputRatio    = safeRatio(toP.outputPerMToken,    fromP.outputPerMToken);
  const cacheReadRatio = safeRatio(toP.cacheReadPerMToken, fromP.cacheReadPerMToken);

  const blended = (inputRatio + outputRatio) / 2;
  let recommendedTierShift: -1 | 0 | 1 = 0;
  if (blended < 0.5)  recommendedTierShift = -1; // much cheaper → relax compression
  if (blended > 2.0)  recommendedTierShift = +1; // much pricier  → tighten compression

  return { inputRatio, outputRatio, cacheReadRatio, recommendedTierShift };
}

// ---------------------------------------------------------------------------
// Budget rebalancing on provider switch
// ---------------------------------------------------------------------------

/**
 * Result returned by `rebalanceBudgetOnSwitch`.
 */
export interface BudgetRebalanceResult {
  /**
   * New system-prompt token budget computed for the target provider.
   * Capped at `SYSTEM_PROMPT_BUDGET_CAP` (50K).
   */
  newSystemBudget: number;
  /**
   * Recommended compression tier for the target provider (1–3).
   *
   * Derived by applying `recommendedTierShift` from `computeProviderCostRatio`
   * to a baseline tier inferred from the current message token count.
   */
  recommendedTier: 1 | 2 | 3;
  /**
   * Delta between the old system-prompt budget and the new one.
   * Positive means the new budget is larger (target has more context headroom).
   * Negative means it shrank.
   */
  tokensFreed: number;
  /**
   * Compression recommendation from the cross-session consolidation profile,
   * when `cwd` was provided to `rebalanceBudgetOnSwitch`.
   *
   * Contains the historically best tier for this project + provider pair,
   * along with a rationale string and estimated ROI delta.
   *
   * `null` when no `cwd` was supplied or no history exists.
   */
  compressionRecommendation?: {
    recommendedTier: number;
    rationale: string;
    estimatedRoiDelta: number;
  } | null;
}

/**
 * Rebalance compression tier thresholds and system-prompt budget when
 * switching the active LLM provider mid-session.
 *
 * Steps:
 *  1. Recompute system-prompt budget using the new provider's context limit.
 *  2. Estimate message token usage from `currentMessages`.
 *  3. Derive a baseline tier from how full the new provider's context window is.
 *  4. Apply the cost-ratio tier shift: cheaper provider → relax one tier,
 *     more expensive → tighten one tier.
 *
 * @param oldProvider      The provider being switched away from.
 * @param newProvider      The provider being switched to.
 * @param currentMessages  Messages currently in the context window.
 * @returns                New budget, recommended tier, and token delta.
 */
export function rebalanceBudgetOnSwitch(
  oldProvider: string,
  newProvider: string,
  currentMessages: { role: string; content: string | unknown[] }[],
): BudgetRebalanceResult {
  const oldBudget = systemPromptBudget(oldProvider);
  const newBudget = systemPromptBudget(newProvider);

  // Estimate message tokens via the chars/4 heuristic used elsewhere in the lib.
  let messageTokens = 0;
  for (const m of currentMessages) {
    if (typeof m.content === "string") {
      messageTokens += Math.ceil(m.content.length / 4);
    } else if (Array.isArray(m.content)) {
      for (const block of m.content) {
        if (block && typeof block === "object") {
          const b = block as Record<string, unknown>;
          if (typeof b["text"] === "string")    messageTokens += Math.ceil(b["text"].length / 4);
          if (typeof b["thinking"] === "string") messageTokens += Math.ceil(b["thinking"].length / 4);
          if (typeof b["content"] === "string") messageTokens += Math.ceil(b["content"].length / 4);
        }
      }
    }
  }

  const newContextLimit = getProviderContextLimit(newProvider);
  const reserveTokens   = 8192;
  const available       = newContextLimit - reserveTokens - newBudget;
  const fillRatio       = available > 0 ? messageTokens / available : 1.0;

  // Baseline tier from fill ratio:
  //   < 50% full → tier 3 (cheapest, light snipping)
  //   50–80% full → tier 2 (moderate snip-compact)
  //   > 80% full → tier 1 (full LLM summarisation)
  let baselineTier: 1 | 2 | 3 = 3;
  if (fillRatio >= 0.8) baselineTier = 1;
  else if (fillRatio >= 0.5) baselineTier = 2;

  // Apply cost-ratio tier shift (clamp to valid range 1–3).
  const { recommendedTierShift } = computeProviderCostRatio(oldProvider, newProvider);
  const shiftedTier = Math.min(3, Math.max(1, baselineTier - recommendedTierShift)) as 1 | 2 | 3;

  return {
    newSystemBudget: newBudget,
    recommendedTier: shiftedTier,
    tokensFreed: newBudget - oldBudget,
    compressionRecommendation: null,
  };
}

/**
 * Async variant of `rebalanceBudgetOnSwitch` that additionally consults the
 * cross-session compression history consolidation hook.
 *
 * Use this overload when a project `cwd` is available. It runs the same
 * synchronous budget computation and then asynchronously emits a
 * `CompressorRecommendation` based on historical tier performance for this
 * project + provider combination. The recommendation is attached to the
 * returned `BudgetRebalanceResult` as `compressionRecommendation`.
 *
 * Consolidation is best-effort: if the history file is absent or an I/O error
 * occurs, the function resolves with `compressionRecommendation: null` rather
 * than rejecting.
 *
 * @param oldProvider      The provider being switched away from.
 * @param newProvider      The provider being switched to.
 * @param currentMessages  Messages currently in the context window.
 * @param cwd              Project working directory (genome root).
 * @returns                Promise of BudgetRebalanceResult with compression recommendation.
 */
export async function rebalanceBudgetOnSwitchWithConsolidation(
  oldProvider: string,
  newProvider: string,
  currentMessages: { role: string; content: string | unknown[] }[],
  cwd: string,
): Promise<BudgetRebalanceResult> {
  const base = rebalanceBudgetOnSwitch(oldProvider, newProvider, currentMessages);

  // Lazy import to avoid a circular-dependency at the module level.
  // consolidation.ts imports from budget/index.ts (computeProviderCostRatio),
  // so we must not import it at the top level here.
  try {
    const { rebalanceBudgetOnSwitchHook } = await import("../compression/consolidation.ts");
    const rec = await rebalanceBudgetOnSwitchHook(
      cwd,
      oldProvider,
      newProvider,
      currentMessages.length,
    );

    // Additionally consult the multi-objective learner for an adaptive allocation.
    // This is best-effort: if no history exists the learner returns the static fallback
    // which is identical to the pre-existing behavior.
    const { BudgetMultiObjectiveLearner } = await import("./multi-objective-learner.ts");
    const learner = new BudgetMultiObjectiveLearner(cwd);
    const adaptiveAlloc = await learner.getRobustBudgetAllocation(
      newProvider,
      "",
      { messageCount: currentMessages.length },
    );

    return {
      ...base,
      // Prefer the adaptive system budget when it was learned from real data
      // (i.e. not the static fallback value). The static fallback has
      // estimatedCostUsd === 0, so we use that as the sentinel.
      newSystemBudget:
        adaptiveAlloc.estimatedCostUsd > 0
          ? adaptiveAlloc.systemPromptTokens
          : base.newSystemBudget,
      compressionRecommendation: {
        recommendedTier: rec.recommendedTier,
        rationale: rec.rationale,
        estimatedRoiDelta: rec.estimatedRoiDelta,
      },
    };
  } catch {
    // Consolidation is best-effort — never fail a provider switch.
    return base;
  }
}
