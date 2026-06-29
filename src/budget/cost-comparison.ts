/**
 * Provider cost comparison utilities.
 *
 * Exposes `compareProviderCosts` which computes the expected USD cost of a
 * workload across all known providers so callers can make data-driven switching
 * decisions. Also surfaces the cache break-even query count: how many repeated
 * queries must be served from cache before caching saves money vs re-sending
 * the full prompt.
 *
 * All prices in USD; token counts in raw tokens (not millions).
 */

import { PROVIDER_PRICING, getProviderContextLimit } from "./index.ts";
import type { Message } from "../types/index.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Cost breakdown for a single provider given a workload.
 */
export interface ProviderCostBreakdown {
  /** Provider name. */
  provider: string;
  /** Total USD cost for input tokens across `queryCount` queries. */
  inputCost: number;
  /** Total USD cost for output tokens across `queryCount` queries. */
  outputCost: number;
  /** Total combined USD cost (inputCost + outputCost). */
  totalCost: number;
  /**
   * Number of queries that must be served from the prompt cache before
   * caching breaks even vs sending the full prompt every time.
   *
   * Computed as:
   *   cacheWriteCost / (inputCostPerQuery - cacheReadCostPerQuery)
   *
   * `null` when:
   *  - The provider does not support prompt caching (no cacheReadPerMToken).
   *  - The cache read cost equals or exceeds the full input cost (caching
   *    would never save money, e.g. providers with very cheap input pricing).
   *  - `inputTokens` is 0.
   */
  cacheBreakeven_queries: number | null;
}

/**
 * Options for `compareProviderCosts`.
 */
export interface CompareProviderCostsOptions {
  /**
   * Explicit list of provider names to include in the comparison.
   * Defaults to all providers in `PROVIDER_PRICING` when omitted.
   */
  providers?: string[];
  /**
   * Cache write multiplier relative to the input price (default: 1.25,
   * matching Anthropic's published 1.25× write rate).
   * Passed through to break-even calculation; set to 0 to skip caching math.
   */
  cacheWriteMultiplier?: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Estimate tokens for a message array using the chars/4 heuristic.
 * Mirrors the approach used throughout this library to avoid pulling in
 * the full token estimator when only a rough count is needed.
 */
function estimateTokens(messages: Message[]): number {
  let total = 0;
  for (const m of messages) {
    if (typeof m.content === "string") {
      total += Math.ceil(m.content.length / 4);
    } else if (Array.isArray(m.content)) {
      for (const block of m.content) {
        if (block && typeof block === "object") {
          const b = block as Record<string, unknown>;
          if (typeof b["text"] === "string")     total += Math.ceil(b["text"].length / 4);
          if (typeof b["thinking"] === "string") total += Math.ceil(b["thinking"].length / 4);
          if (typeof b["content"] === "string")  total += Math.ceil(b["content"].length / 4);
        }
      }
    }
  }
  return total;
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

/**
 * Compare the USD cost of a representative workload across LLM providers.
 *
 * The workload is described by:
 *  - `messages`      — current context window (used to estimate input tokens).
 *  - `queryCount`    — number of times this context will be sent (i.e. turns).
 *  - `outputTokens`  — expected output tokens per query.
 *
 * Returns a map from provider name to `ProviderCostBreakdown` sorted by
 * `totalCost` ascending (cheapest provider first).
 *
 * @example
 * ```ts
 * const results = compareProviderCosts(messages, 100, 512);
 * for (const [provider, breakdown] of Object.entries(results)) {
 *   console.log(provider, breakdown.totalCost.toFixed(4));
 * }
 * ```
 *
 * @param messages      Current message array to estimate input tokens from.
 * @param queryCount    Number of queries to model (default: 1).
 * @param outputTokens  Output tokens per query (default: 512).
 * @param options       Optional provider filter and cache-write multiplier.
 * @returns             Record keyed by provider name, ordered cheapest → costliest.
 */
export function compareProviderCosts(
  messages: Message[],
  queryCount = 1,
  outputTokens = 512,
  options: CompareProviderCostsOptions = {},
): Record<string, ProviderCostBreakdown> {
  const {
    providers = Object.keys(PROVIDER_PRICING),
    cacheWriteMultiplier = 1.25,
  } = options;

  const inputTokens = estimateTokens(messages);

  const results: ProviderCostBreakdown[] = [];

  for (const provider of providers) {
    const pricing = PROVIDER_PRICING[provider];
    if (!pricing) continue;

    const contextLimit = getProviderContextLimit(provider);

    // Clamp input tokens to provider context limit to avoid nonsensical costs
    // for providers with very small windows (e.g. ollama at 32K).
    const effectiveInputTokens = Math.min(inputTokens, contextLimit);

    // USD costs: price is per-million tokens
    const perQueryInputCost  = (effectiveInputTokens / 1_000_000) * pricing.inputPerMToken;
    const perQueryOutputCost = (outputTokens         / 1_000_000) * pricing.outputPerMToken;

    const inputCost  = perQueryInputCost  * queryCount;
    const outputCost = perQueryOutputCost * queryCount;
    const totalCost  = inputCost + outputCost;

    // Cache break-even calculation.
    // Write cost = cacheWriteMultiplier × inputCost for one query (first write).
    // Read cost per subsequent query = cacheReadPerMToken × tokens / 1M.
    // Break-even N satisfies:
    //   cacheWriteCost + N × cacheReadCostPerQuery = N × perQueryInputCost
    //   N = cacheWriteCost / (perQueryInputCost - cacheReadCostPerQuery)
    let cacheBreakeven_queries: number | null = null;

    const hasCaching =
      pricing.cacheReadPerMToken > 0 &&
      cacheWriteMultiplier > 0 &&
      effectiveInputTokens > 0;

    if (hasCaching) {
      const cacheWriteCost     = (effectiveInputTokens / 1_000_000) * pricing.inputPerMToken * cacheWriteMultiplier;
      const cacheReadPerQuery  = (effectiveInputTokens / 1_000_000) * pricing.cacheReadPerMToken;
      const savingsPerQuery    = perQueryInputCost - cacheReadPerQuery;

      if (savingsPerQuery > 0) {
        // Round up — you need at least this many reads to recoup the write cost.
        cacheBreakeven_queries = Math.ceil(cacheWriteCost / savingsPerQuery);
      }
      // If savingsPerQuery <= 0, cache reads cost as much as full input → null.
    }

    results.push({
      provider,
      inputCost,
      outputCost,
      totalCost,
      cacheBreakeven_queries,
    });
  }

  // Sort cheapest first.
  results.sort((a, b) => a.totalCost - b.totalCost);

  const record: Record<string, ProviderCostBreakdown> = {};
  for (const r of results) {
    record[r.provider] = r;
  }
  return record;
}
