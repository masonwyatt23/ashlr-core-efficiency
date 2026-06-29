/**
 * Provider-aware prompt budgeting — extracted from ashlrcode.
 *
 * Behavior preserved exactly: substring match + 100K default fallback,
 * matching ashlrcode/src/agent/context.ts behavior so tests pass against
 * either consumer.
 */

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
