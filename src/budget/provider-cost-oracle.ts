/**
 * Provider Cost Ratio Oracle with Live Market-Rate Updates.
 *
 * Extends the static pricing table in `./index.ts` with a lightweight
 * local-learning mechanism. Instead of making network calls, the oracle
 * observes per-token costs that arrive in LLM API response metadata and
 * progressively calibrates the pricing table via an exponential moving
 * average (EMA, α=0.1).
 *
 * ### Persistence
 *
 * Observed rates are appended to `.ashlr/provider-rates-cache.jsonl`
 * (relative to the supplied `cwd`, defaulting to `process.cwd()`).
 * On startup, the cache is read and blended with static fallbacks.
 *
 * ### Housekeeping
 *
 * If the cache is >7 days old or contains >500 records, the oracle emits
 * a housekeeping record and logs a console warning.  No records are
 * deleted — trimming is left to the operator so audit history is preserved.
 *
 * ### Expected impact
 *
 * 2–5 % USD savings by auto-detecting providers whose effective per-token
 * cost has drifted below the static table, allowing the fleet router to
 * prefer those providers without manual pricing-table maintenance.
 *
 * @module
 */

import { join } from "path";
import { appendJsonl, readJsonl } from "../genome/jsonl.ts";
import { PROVIDER_PRICING } from "./index.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Directory (relative to cwd) where the cache file lives. */
const CACHE_DIR = ".ashlr";
/** JSONL cache filename. */
const CACHE_FILE = "provider-rates-cache.jsonl";

/** EMA smoothing factor — 0.1 means new observations move the rate by 10 %. */
export const EMA_ALPHA = 0.1;

/** Trigger housekeeping after this many records in the cache. */
export const MAX_CACHE_RECORDS = 500;

/** Trigger housekeeping when the most-recent entry is older than this (ms). */
export const MAX_CACHE_AGE_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * A single observed-rate record written into the JSONL cache.
 *
 * All rates are in USD per million tokens (matching `PROVIDER_PRICING` units).
 */
export interface ProviderRateRecord {
  /** Provider family (e.g. "anthropic", "openai"). */
  provider: string;
  /** Specific model ID that produced these rates. */
  model: string;
  /**
   * Observed input rate (USD / M tokens).
   * `null` when not available in the response metadata.
   */
  inputRate: number | null;
  /**
   * Observed output rate (USD / M tokens).
   * `null` when not available.
   */
  outputRate: number | null;
  /**
   * Observed cache-read rate (USD / M tokens).
   * `null` when not available or unsupported by the model.
   */
  cacheRate: number | null;
  /** ISO-8601 timestamp of when this observation was recorded. */
  timestamp: string;
  /** `true` for synthetic housekeeping entries (not real observations). */
  housekeeping?: boolean;
}

/**
 * Calibrated (EMA-blended) per-provider rate returned by `getCalibratedRate`.
 *
 * All rates are in USD per million tokens.
 */
export interface CalibratedRate {
  /** Provider name this rate applies to. */
  provider: string;
  /** EMA-blended input rate. */
  inputPerMToken: number;
  /** EMA-blended output rate. */
  outputPerMToken: number;
  /** EMA-blended cache-read rate. */
  cacheReadPerMToken: number;
  /**
   * Number of real (non-housekeeping) observations used to calibrate these
   * rates.  0 means the oracle is running on static fallback values.
   */
  observationCount: number;
}

// ---------------------------------------------------------------------------
// ProviderCostOracle
// ---------------------------------------------------------------------------

/**
 * Maintains calibrated per-provider token-pricing estimates derived from a
 * blend of:
 *  1. Static fallback values from `PROVIDER_PRICING`.
 *  2. Historically observed rates stored in `.ashlr/provider-rates-cache.jsonl`.
 *
 * Instantiate once per process (or once per project cwd).  Call
 * `initialize()` before reading rates; this is async and reads the JSONL
 * cache.  Every time a response is received, call `recordObservation()` to
 * feed real data into the EMA.
 */
export class ProviderCostOracle {
  private readonly cachePath: string;

  /**
   * EMA state keyed by provider name.
   * Values mirror the shape of `PROVIDER_PRICING` entries.
   */
  private ema: Record<
    string,
    { inputPerMToken: number; outputPerMToken: number; cacheReadPerMToken: number }
  > = {};

  /** Observation counts per provider (housekeeping records excluded). */
  private observationCount: Record<string, number> = {};

  /** Whether `initialize()` has been called at least once. */
  private initialized = false;

  constructor(cwd: string = process.cwd()) {
    this.cachePath = join(cwd, CACHE_DIR, CACHE_FILE);
  }

  // -------------------------------------------------------------------------
  // Initialization
  // -------------------------------------------------------------------------

  /**
   * Load the JSONL cache, seed the EMA from static fallbacks, then blend in
   * cached observations.
   *
   * Safe to call multiple times — subsequent calls are no-ops.
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;
    this.initialized = true;

    // Seed EMA from static pricing table.
    for (const [provider, pricing] of Object.entries(PROVIDER_PRICING)) {
      this.ema[provider] = { ...pricing };
      this.observationCount[provider] = 0;
    }

    // Load and replay cached records.
    const records = await readJsonl<ProviderRateRecord>(this.cachePath);

    // Check housekeeping triggers before blending.
    this._checkHousekeeping(records);

    for (const rec of records) {
      if (rec.housekeeping) continue;
      this._applyObservationToEma(rec.provider, rec.inputRate, rec.outputRate, rec.cacheRate);
    }
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Return the EMA-calibrated rate for a provider.
   *
   * Falls back to the static `PROVIDER_PRICING` entry when the provider is
   * unknown or `initialize()` was never called.
   *
   * @param provider  Case-insensitive provider name (substring match).
   */
  getCalibratedRate(provider: string): CalibratedRate {
    const key = this._resolveProviderKey(provider);
    const fallback = this._staticFallback(key ?? provider);

    if (!key || !this.ema[key]) {
      return {
        provider: provider,
        ...fallback,
        observationCount: 0,
      };
    }

    return {
      provider: key,
      ...this.ema[key],
      observationCount: this.observationCount[key] ?? 0,
    };
  }

  /**
   * Record a single observation of per-token rates from an LLM API response.
   *
   * Immediately updates the EMA and appends the record to the JSONL cache.
   * `null` values are ignored — useful when only some rate fields are present
   * in the response metadata.
   *
   * @param provider   Provider name (e.g. "anthropic").
   * @param model      Specific model ID (e.g. "claude-sonnet-4").
   * @param inputRate  Observed input rate (USD / M tokens), or null.
   * @param outputRate Observed output rate (USD / M tokens), or null.
   * @param cacheRate  Observed cache-read rate (USD / M tokens), or null.
   */
  async recordObservation(
    provider: string,
    model: string,
    inputRate: number | null,
    outputRate: number | null,
    cacheRate: number | null,
  ): Promise<void> {
    // Ensure EMA is seeded even if initialize() was skipped.
    if (!this.initialized) await this.initialize();

    const key = this._resolveProviderKey(provider) ?? provider.toLowerCase();

    // Seed EMA for new / unknown providers using their reported rates.
    if (!this.ema[key]) {
      this.ema[key] = this._staticFallback(key);
      this.observationCount[key] = 0;
    }

    this._applyObservationToEma(key, inputRate, outputRate, cacheRate);
    this.observationCount[key] = (this.observationCount[key] ?? 0) + 1;

    const record: ProviderRateRecord = {
      provider: key,
      model,
      inputRate,
      outputRate,
      cacheRate,
      timestamp: new Date().toISOString(),
    };

    await appendJsonl(this.cachePath, record);
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  /**
   * Apply a single observation to the in-memory EMA state.
   *
   * Only non-null, finite, non-negative values are blended; null fields are
   * silently ignored so partial metadata doesn't corrupt the EMA.
   */
  private _applyObservationToEma(
    providerKey: string,
    inputRate: number | null,
    outputRate: number | null,
    cacheRate: number | null,
  ): void {
    if (!this.ema[providerKey]) {
      this.ema[providerKey] = this._staticFallback(providerKey);
      this.observationCount[providerKey] = 0;
    }

    const state = this.ema[providerKey]!;

    if (inputRate !== null && isFinite(inputRate) && inputRate >= 0) {
      state.inputPerMToken =
        (1 - EMA_ALPHA) * state.inputPerMToken + EMA_ALPHA * inputRate;
    }
    if (outputRate !== null && isFinite(outputRate) && outputRate >= 0) {
      state.outputPerMToken =
        (1 - EMA_ALPHA) * state.outputPerMToken + EMA_ALPHA * outputRate;
    }
    if (cacheRate !== null && isFinite(cacheRate) && cacheRate >= 0) {
      state.cacheReadPerMToken =
        (1 - EMA_ALPHA) * state.cacheReadPerMToken + EMA_ALPHA * cacheRate;
    }
  }

  /**
   * Resolve a provider name to an exact key in `this.ema` via case-insensitive
   * substring matching (same strategy used by `computeProviderCostRatio`).
   *
   * Returns `null` when no match is found.
   */
  private _resolveProviderKey(provider: string): string | null {
    const lower = provider.toLowerCase();
    // Exact match first.
    if (this.ema[lower] !== undefined) return lower;
    // Substring match.
    for (const key of Object.keys(this.ema)) {
      if (lower.includes(key)) return key;
    }
    return null;
  }

  /**
   * Return the static fallback pricing for a provider key.
   *
   * Uses `PROVIDER_PRICING` when the key is known; falls back to a
   * mid-range unknown estimate matching the one in `computeProviderCostRatio`.
   */
  private _staticFallback(providerKey: string): {
    inputPerMToken: number;
    outputPerMToken: number;
    cacheReadPerMToken: number;
  } {
    const lower = providerKey.toLowerCase();
    for (const [key, pricing] of Object.entries(PROVIDER_PRICING)) {
      if (lower.includes(key) || key.includes(lower)) return { ...pricing };
    }
    // Unknown provider — use Anthropic Sonnet equivalent as mid-range fallback.
    return { inputPerMToken: 3.0, outputPerMToken: 15.0, cacheReadPerMToken: 0.30 };
  }

  // -------------------------------------------------------------------------
  // Cache strategy extensions
  // -------------------------------------------------------------------------

  /**
   * Return the `ProviderCacheStrategy` for the given provider name or model ID.
   *
   * Delegates to `resolveProviderStrategy` in `multi-provider-adapter.ts` via
   * a dynamic import to avoid a circular module dependency.
   *
   * @param providerOrModel  Provider name (e.g. "anthropic") or model ID
   *                         (e.g. "o1", "gemini-1.5-pro").
   */
  async getCacheStrategy(
    providerOrModel: string,
  ): Promise<import("../anthropic/multi-provider-adapter.ts").ProviderCacheStrategy> {
    const { resolveProviderStrategy } = await import("../anthropic/multi-provider-adapter.ts");
    return resolveProviderStrategy(providerOrModel);
  }

  /**
   * Compute the estimated cache cost in micro-USD for a given number of
   * cached (read) and written tokens, using the oracle's EMA-calibrated
   * pricing combined with the provider's cache cost multipliers.
   *
   * Returns 0 for providers that do not support caching (e.g. o1/o3).
   *
   * @param providerOrModel   Provider name or model ID.
   * @param cachedTokens      Tokens served from the cache (read pass).
   * @param writtenTokens     Tokens written into the cache (write pass).
   */
  async getCalibratedCacheCost(
    providerOrModel: string,
    cachedTokens: number,
    writtenTokens: number,
  ): Promise<number> {
    const strategy = await this.getCacheStrategy(providerOrModel);
    if (!strategy.supported) return 0;

    const rate = this.getCalibratedRate(strategy.provider);

    const readCostMicroUsd =
      (cachedTokens / 1_000_000) *
      rate.inputPerMToken *
      strategy.readCostMultiplier *
      1_000_000;

    const writeCostMicroUsd =
      (writtenTokens / 1_000_000) *
      rate.inputPerMToken *
      strategy.writeCostMultiplier *
      1_000_000;

    return Math.round(readCostMicroUsd + writeCostMicroUsd);
  }

  // -------------------------------------------------------------------------
  // Housekeeping (private)
  // -------------------------------------------------------------------------

  /**
   * Check whether the cache needs housekeeping (age or size thresholds).
   *
   * When triggered, appends a housekeeping record to the JSONL file and
   * warns via `console.warn`.  This is fire-and-forget — the promise is not
   * awaited from `initialize()` so startup is not delayed.
   */
  private _checkHousekeeping(records: ProviderRateRecord[]): void {
    const real = records.filter((r) => !r.housekeeping);

    let needsHousekeeping = false;
    let reason = "";

    if (real.length >= MAX_CACHE_RECORDS) {
      needsHousekeeping = true;
      reason = `cache has ${real.length} records (limit ${MAX_CACHE_RECORDS})`;
    }

    if (!needsHousekeeping && real.length > 0) {
      const latest = real[real.length - 1];
      if (latest?.timestamp) {
        const age = Date.now() - new Date(latest.timestamp).getTime();
        if (age > MAX_CACHE_AGE_MS) {
          needsHousekeeping = true;
          const days = Math.round(age / (24 * 60 * 60 * 1000));
          reason = `cache is ${days} days old (limit 7 days)`;
        }
      }
    }

    if (needsHousekeeping) {
      console.warn(
        `[ProviderCostOracle] Housekeeping recommended: ${reason}. ` +
          `Consider archiving ${this.cachePath} to keep it compact.`,
      );
      const housekeepingRecord: ProviderRateRecord = {
        provider: "_housekeeping",
        model: "_housekeeping",
        inputRate: null,
        outputRate: null,
        cacheRate: null,
        timestamp: new Date().toISOString(),
        housekeeping: true,
      };
      // Fire-and-forget — startup must not block on this write.
      appendJsonl(this.cachePath, housekeepingRecord).catch(() => {
        // Best-effort: ignore I/O errors when emitting housekeeping records.
      });
    }
  }
}
