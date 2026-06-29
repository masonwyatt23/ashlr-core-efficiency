/**
 * Incremental Context Window Predictor with Latency-Cost Pareto Frontier Tracking.
 *
 * Extends the budget module with a latency-aware context allocation optimizer
 * that predicts per-provider end-to-end latency from (prompt_size,
 * cached_fraction, response_token_budget, model_tier) and finds the optimal
 * context window on a Pareto curve of (minimize latency, minimize USD cost).
 *
 * ### Latency model
 *
 * A lightweight polynomial regression is fit per (provider, model) pair from
 * session logs stored at `evolution/latency-cost-records.jsonl`. The model
 * predicts p50 and p99 latency as:
 *
 *   latency_ms = β0
 *              + β1 * prompt_tokens          (linear TTFT term)
 *              + β2 * prompt_tokens²          (quadratic prefill saturation)
 *              + β3 * (1 - cached_fraction)   (cache miss penalty)
 *
 * When fewer than MIN_FIT_RECORDS records exist the allocator falls back to a
 * hand-tuned default coefficient table per provider.
 *
 * ### Context window sweep
 *
 * `optimalContextAllocation()` sweeps context window sizes [16K, 32K, 64K,
 * 128K] (tokens), evaluating latency + cost at each size and returning the
 * top-3 Pareto-optimal allocations ranked by feasibility score.
 *
 * ### Integration with BudgetMultiObjectiveLearner
 *
 * After `BudgetMultiObjectiveLearner.getRobustBudgetAllocation()` returns a
 * recommendation, call `LatencyAwareAllocator.refineWithLatency()` to
 * post-process the allocation: the method maps the recommended
 * `systemPromptTokens` to the nearest context window bucket and replaces the
 * latency estimates with values from the fitted polynomial model.
 *
 * ### Persistence
 *
 * Allocation recommendations are appended to
 * `evolution/latency-aware-allocations.jsonl` for audit.
 */

import { join } from "path";
import { appendJsonl, readJsonl } from "../genome/jsonl.ts";
import { getProviderContextLimit, EXTENDED_CONTEXT_MODELS } from "./index.ts";
import type { BudgetAllocation } from "./multi-objective-learner.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GENOME_DIR = ".ashlrcode/genome";
const LATENCY_COST_RECORDS_FILE = "evolution/latency-cost-records.jsonl";
const LATENCY_AWARE_ALLOCATIONS_FILE = "evolution/latency-aware-allocations.jsonl";

/** Minimum records needed to fit a model vs. using built-in defaults. */
const MIN_FIT_RECORDS = 5;

/** Context window sizes (tokens) to sweep during Pareto enumeration. */
export const CONTEXT_WINDOW_STEPS = [16_384, 32_768, 65_536, 131_072] as const;

/**
 * Model tier labels mapped to relative latency multipliers.
 * Larger/smarter models are slower per token (prefill + decode).
 */
const MODEL_TIER_LATENCY_MULTIPLIER: Record<string, number> = {
  nano:  0.6,
  micro: 0.75,
  small: 0.85,
  base:  1.0,
  large: 1.3,
  xl:    1.6,
  ultra: 2.0,
};

/**
 * Default polynomial regression coefficients per provider when insufficient
 * session data is available.
 *
 * Format: [β0, β1, β2, β3]
 *   β0: base latency (ms)
 *   β1: ms per input token (linear)
 *   β2: ms per input token² (quadratic saturation, scaled by 1e-9)
 *   β3: cache-miss penalty multiplier (applied to β0 * (1 - cached_fraction))
 */
const DEFAULT_COEFFICIENTS: Record<string, [number, number, number, number]> = {
  anthropic: [180,  0.0035,  1.2, 0.45],
  openai:    [120,  0.0028,  0.9, 0.35],
  xai:       [160,  0.0040,  1.4, 0.50],
  groq:      [40,   0.0008,  0.3, 0.20],
  deepseek:  [90,   0.0022,  0.8, 0.30],
  ollama:    [250,  0.0060,  2.0, 0.10],
};

/** Fallback when provider is unknown. */
const UNKNOWN_COEFFICIENTS: [number, number, number, number] = [150, 0.003, 1.0, 0.40];

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/**
 * A single raw record from `evolution/latency-cost-records.jsonl`.
 *
 * Fleet operators append these after each API call with observed latency and
 * cost. The allocator ingests these to fit the polynomial latency model.
 */
export interface LatencyCostRecord {
  /** Provider name (e.g. "anthropic", "openai"). */
  provider: string;
  /** Model ID (e.g. "claude-sonnet-4"). */
  modelId: string;
  /** Number of tokens in the prompt sent to the API. */
  promptTokens: number;
  /** Fraction of prompt tokens served from cache (0–1). */
  cachedFraction: number;
  /** Number of output tokens requested (response budget). */
  responseTokenBudget: number;
  /** Observed wall-clock latency from request send to first token (ms). */
  observedLatencyP50Ms: number;
  /** Observed p99 latency (ms) over a rolling window. */
  observedLatencyP99Ms: number;
  /** Billed USD cost for this call. */
  billedCostUsd: number;
  /** ISO timestamp. */
  recordedAt: string;
}

/**
 * Fitted polynomial latency model coefficients for a (provider, model) pair.
 */
export interface FittedLatencyModel {
  /** Provider this model covers. */
  provider: string;
  /** Model ID (empty = provider-level). */
  modelId: string;
  /** Number of records used to fit the model. */
  recordCount: number;
  /**
   * Polynomial coefficients [β0, β1, β2, β3] for p50 latency.
   * See module docstring for interpretation.
   */
  p50Coefficients: [number, number, number, number];
  /**
   * Polynomial coefficients [β0, β1, β2, β3] for p99 latency.
   */
  p99Coefficients: [number, number, number, number];
  /** R² of the p50 fit (0–1, higher = better). -1 if data insufficient. */
  p50R2: number;
  /** R² of the p99 fit. -1 if data insufficient. */
  p99R2: number;
  /** ISO timestamp of the fit. */
  fittedAt: string;
}

/**
 * A single context window allocation on the Pareto frontier.
 */
export interface LatencyAwareAllocation {
  /** Context window size in tokens evaluated at this point. */
  contextWindowTokens: number;
  /** Predicted p50 end-to-end latency in milliseconds. */
  estimatedLatencyP50Ms: number;
  /** Predicted p99 end-to-end latency in milliseconds. */
  estimatedLatencyP99Ms: number;
  /** Estimated total USD cost for this allocation. */
  estimatedCostUsd: number;
  /**
   * Estimated cost of writing to the prompt cache at this context size.
   * Zero for providers/models that do not support prompt caching.
   */
  cacheWriteCostUsd: number;
  /**
   * Probability of a cache miss (1 - expected cache hit rate) at this context
   * window size, accounting for the cached_fraction of the prompt.
   */
  cacheMissProbability: number;
  /**
   * Pareto rank: 1 = best trade-off, 2 = second best, 3 = third best.
   * Only the top-3 Pareto-optimal points are returned.
   */
  paretoRank: number;
  /**
   * Feasibility score in [0, 1].
   *
   * Combines:
   *   - latency score: 1 - clamp(p99 / targetLatencyMs, 0, 1)
   *   - cost score: 1 - clamp(costUsd / maxBudgetUsd, 0, 1)
   *
   * Higher = more feasible. The top-3 allocations are sorted by this score.
   */
  feasibilityScore: number;
}

/**
 * Persisted audit record appended to `latency-aware-allocations.jsonl`.
 */
export interface LatencyAwareAllocationRecord {
  /** ISO timestamp. */
  recordedAt: string;
  /** Provider queried. */
  provider: string;
  /** Model ID queried. */
  modelId: string;
  /** Maximum USD budget supplied. */
  maxBudgetUsd: number;
  /** Target latency (ms) supplied. */
  targetLatencyMs: number;
  /** Top-3 Pareto-optimal allocations returned. */
  allocations: LatencyAwareAllocation[];
  /** Whether the latency model was fitted from data or used defaults. */
  modelSource: "fitted" | "default";
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

function latencyCostRecordsPath(cwd: string): string {
  return join(cwd, GENOME_DIR, LATENCY_COST_RECORDS_FILE);
}

function latencyAwareAllocationsPath(cwd: string): string {
  return join(cwd, GENOME_DIR, LATENCY_AWARE_ALLOCATIONS_FILE);
}

// ---------------------------------------------------------------------------
// Polynomial latency prediction
// ---------------------------------------------------------------------------

/**
 * Evaluate the polynomial latency model for a given input.
 *
 * latency = β0 + β1*tokens + β2*tokens²*1e-9 + β3*baseLat*(1-cachedFraction)
 *
 * The quadratic term uses a 1e-9 scale factor so β2 stays in a reasonable
 * range (avoids numerical overflow for large token counts).
 */
function predictLatencyMs(
  coefficients: [number, number, number, number],
  promptTokens: number,
  cachedFraction: number,
  modelTierMultiplier: number,
): number {
  const [β0, β1, β2, β3] = coefficients;
  const base = β0 + β1 * promptTokens + β2 * promptTokens * promptTokens * 1e-9;
  const cachePenalty = β3 * β0 * (1 - cachedFraction);
  return Math.max(20, (base + cachePenalty) * modelTierMultiplier);
}

// ---------------------------------------------------------------------------
// Ordinary Least Squares (1D weighted regression)
// ---------------------------------------------------------------------------

/**
 * Fit a simple linear regression y = a + b*x and return [a, b].
 * Used internally to fit the latency model from records.
 */
function fitLinear(xs: number[], ys: number[]): [number, number] {
  const n = xs.length;
  if (n < 2) return [ys[0] ?? 0, 0];

  let sumX = 0, sumY = 0, sumXX = 0, sumXY = 0;
  for (let i = 0; i < n; i++) {
    sumX  += xs[i]!;
    sumY  += ys[i]!;
    sumXX += xs[i]! * xs[i]!;
    sumXY += xs[i]! * ys[i]!;
  }
  const denom = n * sumXX - sumX * sumX;
  if (Math.abs(denom) < 1e-12) return [sumY / n, 0];
  const b = (n * sumXY - sumX * sumY) / denom;
  const a = (sumY - b * sumX) / n;
  return [a, b];
}

/**
 * Compute R² (coefficient of determination) for a set of predictions vs actuals.
 */
function computeR2(actuals: number[], predicted: number[]): number {
  const n = actuals.length;
  if (n < 2) return -1;
  const mean = actuals.reduce((s, v) => s + v, 0) / n;
  let ssTot = 0, ssRes = 0;
  for (let i = 0; i < n; i++) {
    ssTot += (actuals[i]! - mean) ** 2;
    ssRes += (actuals[i]! - predicted[i]!) ** 2;
  }
  if (ssTot < 1e-12) return 1;
  return 1 - ssRes / ssTot;
}

/**
 * Fit a FittedLatencyModel from a set of LatencyCostRecords.
 *
 * Uses a 3-feature polynomial basis:
 *   x1 = promptTokens
 *   x2 = promptTokens² * 1e-9
 *   x3 = (1 - cachedFraction)
 *
 * We simplify to independent regressions over each feature and combine
 * coefficients analytically (feasible for 3 features with small N).
 *
 * For production use with large N, consider a proper multi-variate OLS;
 * this implementation is intentionally lightweight.
 */
function fitLatencyModel(
  provider: string,
  modelId: string,
  records: LatencyCostRecord[],
): FittedLatencyModel {
  const n = records.length;
  const now = new Date().toISOString();

  if (n < MIN_FIT_RECORDS) {
    // Fall back to built-in defaults.
    const defaultKey = Object.keys(DEFAULT_COEFFICIENTS).find(
      (k) => provider.toLowerCase().includes(k),
    );
    const def = defaultKey
      ? DEFAULT_COEFFICIENTS[defaultKey]!
      : UNKNOWN_COEFFICIENTS;

    return {
      provider,
      modelId,
      recordCount: n,
      p50Coefficients: [...def] as [number, number, number, number],
      p99Coefficients: [def[0] * 1.6, def[1] * 1.5, def[2] * 1.5, def[3] * 1.3] as [number, number, number, number],
      p50R2: -1,
      p99R2: -1,
      fittedAt: now,
    };
  }

  // Build feature arrays.
  const tokensArr = records.map((r) => r.promptTokens);
  const tokens2Arr = records.map((r) => r.promptTokens * r.promptTokens * 1e-9);
  const cacheMissArr = records.map((r) => 1 - r.cachedFraction);
  const p50Arr = records.map((r) => r.observedLatencyP50Ms);
  const p99Arr = records.map((r) => r.observedLatencyP99Ms);

  // Fit p50: regress onto tokens, then residuals onto tokens², then onto cacheMiss.
  const [β0_50, β1_50] = fitLinear(tokensArr, p50Arr);
  const resid1_50 = p50Arr.map((y, i) => y - (β0_50 + β1_50 * tokensArr[i]!));
  const [, β2_50] = fitLinear(tokens2Arr, resid1_50);
  const resid2_50 = resid1_50.map((y, i) => y - β2_50 * tokens2Arr[i]!);
  const [, β3_50_raw] = fitLinear(cacheMissArr, resid2_50);
  // β3 is stored as relative to β0 to keep the prediction formula consistent.
  const β3_50 = Math.abs(β0_50) > 1e-6 ? β3_50_raw / β0_50 : 0;

  const p50Coefs: [number, number, number, number] = [β0_50, β1_50, β2_50, β3_50];

  // Predicted p50 for R² calculation.
  const p50Predicted = records.map((r) =>
    predictLatencyMs(p50Coefs, r.promptTokens, r.cachedFraction, 1.0),
  );

  // Fit p99 similarly.
  const [β0_99, β1_99] = fitLinear(tokensArr, p99Arr);
  const resid1_99 = p99Arr.map((y, i) => y - (β0_99 + β1_99 * tokensArr[i]!));
  const [, β2_99] = fitLinear(tokens2Arr, resid1_99);
  const resid2_99 = resid1_99.map((y, i) => y - β2_99 * tokens2Arr[i]!);
  const [, β3_99_raw] = fitLinear(cacheMissArr, resid2_99);
  const β3_99 = Math.abs(β0_99) > 1e-6 ? β3_99_raw / β0_99 : 0;

  const p99Coefs: [number, number, number, number] = [β0_99, β1_99, β2_99, β3_99];

  const p99Predicted = records.map((r) =>
    predictLatencyMs(p99Coefs, r.promptTokens, r.cachedFraction, 1.0),
  );

  return {
    provider,
    modelId,
    recordCount: n,
    p50Coefficients: p50Coefs,
    p99Coefficients: p99Coefs,
    p50R2: computeR2(p50Arr, p50Predicted),
    p99R2: computeR2(p99Arr, p99Predicted),
    fittedAt: now,
  };
}

// ---------------------------------------------------------------------------
// Cost estimation
// ---------------------------------------------------------------------------

const PROVIDER_PRICING_LA: Record<
  string,
  { inputPerMToken: number; outputPerMToken: number; cacheWritePerMToken: number; cacheReadPerMToken: number }
> = {
  anthropic: { inputPerMToken: 3.0,  outputPerMToken: 15.0,  cacheWritePerMToken: 3.75,  cacheReadPerMToken: 0.30  },
  openai:    { inputPerMToken: 2.5,  outputPerMToken: 10.0,  cacheWritePerMToken: 0.0,   cacheReadPerMToken: 1.25  },
  xai:       { inputPerMToken: 5.0,  outputPerMToken: 15.0,  cacheWritePerMToken: 0.0,   cacheReadPerMToken: 0.25  },
  groq:      { inputPerMToken: 0.27, outputPerMToken: 0.27,  cacheWritePerMToken: 0.0,   cacheReadPerMToken: 0.027 },
  deepseek:  { inputPerMToken: 0.14, outputPerMToken: 0.28,  cacheWritePerMToken: 0.0,   cacheReadPerMToken: 0.014 },
  ollama:    { inputPerMToken: 0.0,  outputPerMToken: 0.0,   cacheWritePerMToken: 0.0,   cacheReadPerMToken: 0.0   },
};

function resolvePricing(provider: string) {
  const lower = provider.toLowerCase();
  for (const [key, p] of Object.entries(PROVIDER_PRICING_LA)) {
    if (lower.includes(key)) return p;
  }
  return PROVIDER_PRICING_LA["anthropic"]!;
}

/**
 * Estimate cost of a single call at a given context window size.
 *
 * @param contextWindowTokens  Context window size being evaluated.
 * @param cachedFraction       Fraction of prompt tokens served from cache.
 * @param responseTokenBudget  Number of output tokens allocated.
 * @param provider             Provider name.
 * @returns                    { totalCostUsd, cacheWriteCostUsd, cacheMissProbability }
 */
function estimateCostForWindow(
  contextWindowTokens: number,
  cachedFraction: number,
  responseTokenBudget: number,
  provider: string,
): { totalCostUsd: number; cacheWriteCostUsd: number; cacheMissProbability: number } {
  const p = resolvePricing(provider);
  const perM = 1_000_000;

  // Tokens actually charged at input rate = prompt that isn't cached.
  const inputTokens = contextWindowTokens * (1 - cachedFraction);
  const cacheReadTokens = contextWindowTokens * cachedFraction;

  const inputCost      = (inputTokens / perM) * p.inputPerMToken;
  const cacheReadCost  = (cacheReadTokens / perM) * p.cacheReadPerMToken;
  const outputCost     = (responseTokenBudget / perM) * p.outputPerMToken;
  const cacheWriteCost = (contextWindowTokens / perM) * p.cacheWritePerMToken;

  // Cache miss probability: lower cached_fraction → higher chance of a full miss.
  // Modeled as a sigmoid: P(miss) ≈ 1 / (1 + exp(10 * (cachedFraction - 0.5)))
  const cacheMissProbability = 1 / (1 + Math.exp(10 * (cachedFraction - 0.5)));

  const totalCostUsd = inputCost + cacheReadCost + outputCost;
  return { totalCostUsd, cacheWriteCostUsd: cacheWriteCost, cacheMissProbability };
}

// ---------------------------------------------------------------------------
// Pareto dominance check (2-objective: minimize latency, minimize cost)
// ---------------------------------------------------------------------------

function dominatesLA(
  a: LatencyAwareAllocation,
  b: LatencyAwareAllocation,
): boolean {
  return (
    a.estimatedLatencyP99Ms <= b.estimatedLatencyP99Ms &&
    a.estimatedCostUsd <= b.estimatedCostUsd &&
    (a.estimatedLatencyP99Ms < b.estimatedLatencyP99Ms ||
      a.estimatedCostUsd < b.estimatedCostUsd)
  );
}

// ---------------------------------------------------------------------------
// Model tier inference
// ---------------------------------------------------------------------------

/**
 * Infer a model tier label from a model ID string.
 * Returns "base" for unrecognized model IDs.
 */
function inferModelTier(modelId: string): string {
  const lower = modelId.toLowerCase();
  if (lower.includes("nano") || lower.includes("flash")) return "nano";
  if (lower.includes("micro") || lower.includes("mini")) return "micro";
  if (lower.includes("haiku") || lower.includes("small")) return "small";
  if (lower.includes("opus") || lower.includes("ultra") || lower.includes("o3")) return "ultra";
  if (lower.includes("sonnet") || lower.includes("large")) return "large";
  if (lower.includes("o1") || lower.includes("reasoning")) return "xl";
  return "base";
}

// ---------------------------------------------------------------------------
// LatencyAwareAllocator
// ---------------------------------------------------------------------------

/**
 * Context window allocator that optimizes on a Pareto curve of
 * (minimize end-to-end latency, minimize USD cost).
 *
 * ### Usage
 *
 * ```ts
 * const allocator = new LatencyAwareAllocator("/path/to/project");
 * await allocator.loadRecords();
 *
 * const top3 = await allocator.optimalContextAllocation(
 *   "anthropic", "claude-sonnet-4",
 *   0.10,   // maxBudgetUsd
 *   800,    // targetLatencyMs
 * );
 * // top3[0] is the best feasible allocation
 * ```
 */
export class LatencyAwareAllocator {
  private readonly cwd: string;
  private records: LatencyCostRecord[] = [];
  private fittedModels: Map<string, FittedLatencyModel> = new Map();

  constructor(cwd: string) {
    this.cwd = cwd;
  }

  // -------------------------------------------------------------------------
  // Data loading
  // -------------------------------------------------------------------------

  /**
   * Load session records from `evolution/latency-cost-records.jsonl`.
   * Safe to call multiple times (records are deduplicated by recordedAt+provider).
   */
  async loadRecords(): Promise<void> {
    const path = latencyCostRecordsPath(this.cwd);
    const loaded = await readJsonl<LatencyCostRecord>(path);
    // Merge, keeping newer duplicates (same provider+model+recordedAt).
    const seen = new Set<string>(
      this.records.map((r) => `${r.provider}:${r.modelId}:${r.recordedAt}`),
    );
    for (const rec of loaded) {
      const key = `${rec.provider}:${rec.modelId}:${rec.recordedAt}`;
      if (!seen.has(key)) {
        this.records.push(rec);
        seen.add(key);
      }
    }
  }

  /**
   * Directly ingest LatencyCostRecords without loading from disk.
   * Useful in tests or when records come from an in-memory pipeline.
   */
  ingestRecords(records: LatencyCostRecord[]): void {
    this.records.push(...records);
    // Invalidate fitted models so they are re-fit on next call.
    this.fittedModels.clear();
  }

  // -------------------------------------------------------------------------
  // Model fitting
  // -------------------------------------------------------------------------

  /**
   * Fit (or retrieve cached) latency model for a (provider, model) pair.
   */
  fitModel(provider: string, modelId: string): FittedLatencyModel {
    const key = `${provider.toLowerCase()}:::${modelId.toLowerCase()}`;
    const cached = this.fittedModels.get(key);
    if (cached) return cached;

    const matching = this.records.filter((r) => {
      const pm = r.provider.toLowerCase().includes(provider.toLowerCase()) ||
        provider.toLowerCase().includes(r.provider.toLowerCase());
      const mm = !modelId ||
        !r.modelId ||
        r.modelId.toLowerCase().includes(modelId.toLowerCase()) ||
        modelId.toLowerCase().includes(r.modelId.toLowerCase());
      return pm && mm;
    });

    const model = fitLatencyModel(provider, modelId, matching);
    this.fittedModels.set(key, model);
    return model;
  }

  // -------------------------------------------------------------------------
  // Core public API
  // -------------------------------------------------------------------------

  /**
   * Compute the top-3 Pareto-optimal context window allocations for a given
   * (provider, model, budget, latency target) tuple.
   *
   * Steps:
   *   1. Fit the latency model from available records (or use defaults).
   *   2. For each context window step in CONTEXT_WINDOW_STEPS, predict
   *      p50/p99 latency and estimate cost.
   *   3. Compute the Pareto frontier over (latency p99, cost).
   *   4. Score feasibility: prefer allocations that satisfy both constraints.
   *   5. Return top-3 Pareto-optimal allocations sorted by feasibility score.
   *   6. Persist the result to `evolution/latency-aware-allocations.jsonl`.
   *
   * @param provider         Provider name.
   * @param modelId          Model ID (may be empty for provider-level).
   * @param maxBudgetUsd     Maximum USD cost per call (hard budget constraint).
   * @param targetLatencyMs  Target p99 latency in milliseconds.
   * @param cachedFraction   Expected fraction of prompt tokens in cache (0–1).
   *                         Defaults to 0.5 (typical mid-session cache hit rate).
   * @param responseTokenBudget  Output tokens allocated. Defaults to 2048.
   * @returns                Top-3 LatencyAwareAllocation objects, ranked 1–3.
   */
  async optimalContextAllocation(
    provider: string,
    modelId: string,
    maxBudgetUsd: number,
    targetLatencyMs: number,
    cachedFraction = 0.5,
    responseTokenBudget = 2048,
  ): Promise<LatencyAwareAllocation[]> {
    const model = this.fitModel(provider, modelId);
    const tierMultiplier = MODEL_TIER_LATENCY_MULTIPLIER[inferModelTier(modelId)] ?? 1.0;

    const providerContextLimit = getProviderContextLimit(provider);

    // Check for model-specific context limit overrides.
    const modelContextLimit = (() => {
      const lower = modelId.toLowerCase();
      for (const [key, ext] of Object.entries(EXTENDED_CONTEXT_MODELS)) {
        if (lower.includes(key) || key.includes(lower)) {
          return ext.contextLimit;
        }
      }
      return providerContextLimit;
    })();

    // Evaluate each context window step.
    const candidates: LatencyAwareAllocation[] = [];

    for (const windowSize of CONTEXT_WINDOW_STEPS) {
      // Skip windows that exceed the model's context limit.
      if (windowSize > modelContextLimit) continue;

      // Use up to windowSize tokens as the effective prompt size.
      const effectivePromptTokens = Math.min(windowSize, modelContextLimit);

      const p50 = predictLatencyMs(
        model.p50Coefficients,
        effectivePromptTokens,
        cachedFraction,
        tierMultiplier,
      );
      const p99 = predictLatencyMs(
        model.p99Coefficients,
        effectivePromptTokens,
        cachedFraction,
        tierMultiplier,
      );

      const { totalCostUsd, cacheWriteCostUsd, cacheMissProbability } =
        estimateCostForWindow(windowSize, cachedFraction, responseTokenBudget, provider);

      // Feasibility score: higher = better (both constraints satisfied).
      const latencyScore = Math.max(0, 1 - p99 / Math.max(targetLatencyMs, 1));
      const costScore = Math.max(0, 1 - totalCostUsd / Math.max(maxBudgetUsd, 1e-9));
      const feasibilityScore = (latencyScore + costScore) / 2;

      candidates.push({
        contextWindowTokens: windowSize,
        estimatedLatencyP50Ms: Math.round(p50),
        estimatedLatencyP99Ms: Math.round(p99),
        estimatedCostUsd: totalCostUsd,
        cacheWriteCostUsd,
        cacheMissProbability,
        paretoRank: 0, // filled in below
        feasibilityScore,
      });
    }

    if (candidates.length === 0) {
      // Edge case: all windows exceed model context limit. Use smallest available.
      const fallbackWindow = Math.min(...CONTEXT_WINDOW_STEPS);
      const p50 = predictLatencyMs(model.p50Coefficients, fallbackWindow, cachedFraction, tierMultiplier);
      const p99 = predictLatencyMs(model.p99Coefficients, fallbackWindow, cachedFraction, tierMultiplier);
      const { totalCostUsd, cacheWriteCostUsd, cacheMissProbability } =
        estimateCostForWindow(fallbackWindow, cachedFraction, responseTokenBudget, provider);
      candidates.push({
        contextWindowTokens: fallbackWindow,
        estimatedLatencyP50Ms: Math.round(p50),
        estimatedLatencyP99Ms: Math.round(p99),
        estimatedCostUsd: totalCostUsd,
        cacheWriteCostUsd,
        cacheMissProbability,
        paretoRank: 1,
        feasibilityScore: 0.5,
      });
    }

    // Mark Pareto-dominated points.
    const nonDominated = candidates.filter((a) =>
      !candidates.some((b) => b !== a && dominatesLA(b, a)),
    );

    // Sort Pareto-optimal points by feasibility score descending.
    const ranked = nonDominated
      .sort((a, b) => b.feasibilityScore - a.feasibilityScore)
      .slice(0, 3)
      .map((alloc, idx) => ({ ...alloc, paretoRank: idx + 1 }));

    // If we have fewer than 3 Pareto points, fill from dominated candidates by score.
    if (ranked.length < 3) {
      const dominated = candidates
        .filter((c) => !nonDominated.includes(c))
        .sort((a, b) => b.feasibilityScore - a.feasibilityScore);
      for (const d of dominated) {
        if (ranked.length >= 3) break;
        ranked.push({ ...d, paretoRank: ranked.length + 1 });
      }
    }

    // Persist to audit log.
    const record: LatencyAwareAllocationRecord = {
      recordedAt: new Date().toISOString(),
      provider,
      modelId,
      maxBudgetUsd,
      targetLatencyMs,
      allocations: ranked,
      modelSource: model.recordCount >= MIN_FIT_RECORDS ? "fitted" : "default",
    };
    await appendJsonl(latencyAwareAllocationsPath(this.cwd), record);

    return ranked;
  }

  /**
   * Post-process a `BudgetMultiObjectiveLearner` allocation by replacing its
   * latency estimates with values from the fitted polynomial model.
   *
   * Maps `allocation.systemPromptTokens` to the nearest context window step and
   * recomputes p50/p99 from the fitted model. The resulting allocation has more
   * accurate latency predictions than the linear approximation used by the learner.
   *
   * @param allocation       Allocation from `BudgetMultiObjectiveLearner`.
   * @param targetLatencyMs  Desired p99 latency (used to compute feasibility).
   * @param maxBudgetUsd     Budget constraint (used to compute feasibility).
   * @returns                A refined allocation with updated latency estimates
   *                         and a `latencyAwareAllocations` array containing the
   *                         top-3 Pareto-optimal context window options.
   */
  async refineWithLatency(
    allocation: BudgetAllocation,
    targetLatencyMs: number,
    maxBudgetUsd: number,
  ): Promise<BudgetAllocation & { latencyAwareAllocations: LatencyAwareAllocation[] }> {
    const top3 = await this.optimalContextAllocation(
      allocation.provider,
      allocation.modelId,
      maxBudgetUsd,
      targetLatencyMs,
    );

    // Pick the best allocation (rank 1) and update latency estimates.
    const best = top3[0];
    if (!best) {
      return { ...allocation, latencyAwareAllocations: [] };
    }

    return {
      ...allocation,
      estimatedP99LatencyMs: best.estimatedLatencyP99Ms,
      latencyAwareAllocations: top3,
    };
  }

  // -------------------------------------------------------------------------
  // Inspection helpers
  // -------------------------------------------------------------------------

  /**
   * Return the fitted model for (provider, modelId), or the default model
   * when no records have been loaded.
   */
  getModel(provider: string, modelId: string): FittedLatencyModel {
    return this.fitModel(provider, modelId);
  }

  /**
   * Read previously persisted allocation recommendations from disk.
   */
  async loadPersistedAllocations(): Promise<LatencyAwareAllocationRecord[]> {
    return readJsonl<LatencyAwareAllocationRecord>(
      latencyAwareAllocationsPath(this.cwd),
    );
  }
}
