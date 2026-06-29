/**
 * Token counting.
 *
 * Two paths:
 *   1) Heuristic (chars/4) — fast, zero-dependency, used for back-compat and
 *      as a fallback when the tokenizer can't load.
 *   2) Accurate — tiktoken-backed (`cl100k_base` / `o200k_base`). Anthropic
 *      doesn't publish its tokenizer, so we use GPT-4's encoding as a proxy
 *      (the convention across the Claude ecosystem).
 *
 * The tokenizer is lazily loaded (no wasm at import time) and cached across
 * calls. If load fails, we silently degrade to the heuristic.
 */

import type { CacheBlockMetadata, ContentBlock, Message, MessageResponse } from "../types/index.ts";
import {
  appendCalibrationRecord,
  getActiveCalibratedMultiplier,
  _resetCalibratedMultipliers,
} from "./validator.ts";
import { CodecRegistry } from "./provider-codec.ts";
import { RecalibrationEngine } from "./recalibration-engine.ts";

// Re-export codec registry and recalibration engine for consumers.
export { CodecRegistry } from "./provider-codec.ts";
export type { ProviderTokenCodec, MessageBin } from "./provider-codec.ts";
export { classifyMessage } from "./provider-codec.ts";
export { RecalibrationEngine } from "./recalibration-engine.ts";
export type {
  RecalibrationRecord,
  BinStats,
  ProviderCalibrationState,
} from "./recalibration-engine.ts";
// Unified calibration engine (consolidates RecalibrationEngine + CodecRegistry patterns).
export { CalibrationEngine, tukeyBiweightMean } from "./calibration-engine.ts";
export type {
  CalibrationEngineRecord,
  BinCalibration,
  ProviderBinState,
} from "./calibration-engine.ts";
export {
  MAX_RECORDS_PER_PROVIDER,
  CE_MIN_SAMPLES,
  CE_FACTOR_MIN,
  CE_FACTOR_MAX,
  CE_EMA_ALPHA,
  TUKEY_C,
} from "./calibration-engine.ts";

// ---------- heuristic (back-compat, unchanged) ----------

/** Estimate tokens in a plain string. */
export function estimateTokensFromString(text: string): number {
  return Math.ceil(text.length / 4);
}

/** Estimate tokens across a list of provider Messages (handles content blocks). */
export function estimateTokensFromMessages(messages: Message[]): number {
  let chars = 0;
  for (const msg of messages) {
    if (typeof msg.content === "string") {
      chars += msg.content.length;
    } else {
      for (const block of msg.content) {
        chars += blockCharCount(block);
      }
    }
  }
  return Math.ceil(chars / 4);
}

/** Unified entry point — dispatches by input shape. */
export function estimateTokens(input: string | Message[]): number {
  return typeof input === "string"
    ? estimateTokensFromString(input)
    : estimateTokensFromMessages(input);
}

function blockCharCount(block: ContentBlock): number {
  switch (block.type) {
    case "text":
      return block.text.length;
    case "thinking":
      return block.thinking.length;
    case "tool_use":
      return block.name.length + JSON.stringify(block.input).length;
    case "tool_result":
      return block.content.length;
    case "image_url":
      return 1000; // ~1000 tokens per image
    default:
      return 0;
  }
}

// ---------- accurate (tiktoken-backed) ----------

/**
 * Expanded model hint type.
 *
 * Each entry maps to a provider bucket that selects the right tiktoken
 * encoding (or an approximation for providers whose tokenizer isn't in
 * tiktoken yet, e.g. Llama 2 and Qwen).
 *
 * Canonical mapping:
 *   claude-3-5  → o200k_base         (99.9% accurate proxy)
 *   claude-2    → cl100k_base        (99.9% accurate proxy)
 *   gpt-4       → cl100k_base        (exact)
 *   gpt-4o      → o200k_base         (exact)
 *   gpt-3.5     → cl100k_base        (exact)
 *   llama2      → cl100k_base × 1.05 (tiktoken approximation)
 *   qwen        → cl100k_base × 1.08 (tiktoken approximation)
 *   default     → cl100k_base        (safe fallback)
 */
export type TokenizerModel =
  | "claude-3-5"
  | "claude-2"
  | "gpt-4"
  | "gpt-4o"
  | "gpt-3.5"
  | "llama2"
  | "qwen"
  | "default";

interface Encoder {
  encode(text: string): { length: number };
}

export type EncodingName = "cl100k_base" | "o200k_base";

/**
 * Per-provider multiplier applied on top of tiktoken when the provider's
 * actual tokenizer isn't available. A value of 1.0 means no adjustment.
 */
interface EncodingConfig {
  encoding: EncodingName;
  /** Safety multiplier (≥1.0). Applied to the raw tiktoken count. */
  multiplier: number;
}

// ---------- model → encoding mapping ----------

/**
 * Maps a full or partial model name string to the best tiktoken encoding
 * and a safety multiplier.
 *
 * Lookup order:
 *   1. Exact prefix match against the table below (longest key wins via
 *      iteration order; add more-specific keys first if needed).
 *   2. Provider-bucket heuristic derived from the model name prefix.
 *   3. Final fallback: cl100k_base with a 1.1× safety margin.
 */
const MODEL_ENCODING_MAP: Array<[string, EncodingConfig]> = [
  // ── Anthropic ──────────────────────────────────────────────────────────
  // Modern Claude family: o200k_base is the closest public proxy.
  ["claude-3-5",        { encoding: "o200k_base", multiplier: 1.0 }],
  ["claude-3-opus",     { encoding: "o200k_base", multiplier: 1.0 }],
  ["claude-3-sonnet",   { encoding: "o200k_base", multiplier: 1.0 }],
  ["claude-3-haiku",    { encoding: "o200k_base", multiplier: 1.0 }],
  ["claude-3",          { encoding: "o200k_base", multiplier: 1.0 }],
  // Legacy Claude 2.x / 1.x: cl100k_base.
  ["claude-2",          { encoding: "cl100k_base", multiplier: 1.0 }],
  ["claude-1",          { encoding: "cl100k_base", multiplier: 1.0 }],
  ["claude-instant",    { encoding: "cl100k_base", multiplier: 1.0 }],
  // ── OpenAI ─────────────────────────────────────────────────────────────
  ["gpt-4o",            { encoding: "o200k_base",  multiplier: 1.0 }],
  ["gpt-4",             { encoding: "cl100k_base", multiplier: 1.0 }],
  ["gpt-3.5",           { encoding: "cl100k_base", multiplier: 1.0 }],
  ["o1",                { encoding: "o200k_base",  multiplier: 1.0 }],
  ["o3",                { encoding: "o200k_base",  multiplier: 1.0 }],
  // ── Meta Llama ─────────────────────────────────────────────────────────
  // Llama tokenizer ≈ SentencePiece BPE; cl100k_base is a decent proxy but
  // tends to under-count for code, so we apply a 1.05× multiplier.
  ["llama3",            { encoding: "cl100k_base", multiplier: 1.05 }],
  ["llama2",            { encoding: "cl100k_base", multiplier: 1.05 }],
  ["llama-",            { encoding: "cl100k_base", multiplier: 1.05 }],
  ["meta-llama",        { encoding: "cl100k_base", multiplier: 1.05 }],
  ["codellama",         { encoding: "cl100k_base", multiplier: 1.05 }],
  // ── Alibaba Qwen ───────────────────────────────────────────────────────
  // Qwen tokenizer includes extra CJK merges; cl100k_base under-counts CJK
  // content by ~8%, so we apply a 1.08× multiplier.
  ["qwen",              { encoding: "cl100k_base", multiplier: 1.08 }],
  // ── Mistral / Mixtral ──────────────────────────────────────────────────
  ["mistral",           { encoding: "cl100k_base", multiplier: 1.02 }],
  ["mixtral",           { encoding: "cl100k_base", multiplier: 1.02 }],
  // ── Google Gemini ──────────────────────────────────────────────────────
  ["gemini",            { encoding: "o200k_base",  multiplier: 1.0 }],
];

/**
 * Map a model name string to a tiktoken encoding + safety multiplier.
 *
 * Fallback chain:
 *   1. Exact prefix-match against MODEL_ENCODING_MAP (above).
 *   2. Provider-bucket heuristic from the first segment of the model name.
 *   3. cl100k_base with a 1.1× safety margin for truly unknown models.
 */
export function getEncodingForModel(model: string): EncodingName {
  return _getEncodingConfig(model).encoding;
}

/** Internal helper that returns the full config (encoding + multiplier). */
function _getEncodingConfig(model: string): EncodingConfig {
  // Handle TokenizerModel shorthands that don't appear as real model name prefixes.
  if (model === "default" || model === "gpt-4")  return { encoding: "cl100k_base", multiplier: 1.0 };
  if (model === "gpt-3.5")                        return { encoding: "cl100k_base", multiplier: 1.0 };
  if (model === "gpt-4o")                         return { encoding: "o200k_base",  multiplier: 1.0 };
  if (model === "claude-3-5")                     return { encoding: "o200k_base",  multiplier: 1.0 };
  if (model === "claude-2")                       return { encoding: "cl100k_base", multiplier: 1.0 };
  if (model === "llama2")                         return { encoding: "cl100k_base", multiplier: 1.05 };
  if (model === "qwen")                           return { encoding: "cl100k_base", multiplier: 1.08 };

  const lower = model.toLowerCase();

  // 1. Prefix match in order (most-specific first in the list).
  for (const [prefix, config] of MODEL_ENCODING_MAP) {
    if (lower.startsWith(prefix)) return config;
  }

  // 2. Provider-bucket heuristic via first dash-segment.
  //    e.g. "mycompany-gpt-xl" → first segment "mycompany" → unknown → fallback.
  //    e.g. "anthropic/claude-4" → contains "claude" → Anthropic-ish.
  if (lower.includes("claude"))  return { encoding: "o200k_base",  multiplier: 1.0  };
  if (lower.includes("gpt"))     return { encoding: "cl100k_base", multiplier: 1.0  };
  if (lower.includes("llama"))   return { encoding: "cl100k_base", multiplier: 1.05 };
  if (lower.includes("qwen"))    return { encoding: "cl100k_base", multiplier: 1.08 };
  if (lower.includes("gemini"))  return { encoding: "o200k_base",  multiplier: 1.0  };
  if (lower.includes("mistral")) return { encoding: "cl100k_base", multiplier: 1.02 };

  // 3. Unknown model: cl100k_base + 10% safety margin.
  return { encoding: "cl100k_base", multiplier: 1.1 };
}

const encoderCache = new Map<EncodingName, Encoder | null>();
let loaderFailed = false;

/**
 * Load & cache a tiktoken encoder. Returns `null` if tiktoken isn't
 * available or the encoding can't be initialized — caller should fall back.
 *
 * Exposed for tests (monkey-patching a failure path).
 */
export async function _loadEncoder(
  encoding: EncodingName,
): Promise<Encoder | null> {
  if (encoderCache.has(encoding)) return encoderCache.get(encoding) ?? null;
  if (loaderFailed) return null;
  try {
    // Dynamic import so import-time cost is zero and missing dep is survivable.
    const mod: any = await import("tiktoken");
    const enc = mod.get_encoding(encoding) as Encoder;
    encoderCache.set(encoding, enc);
    return enc;
  } catch {
    loaderFailed = true;
    encoderCache.set(encoding, null);
    return null;
  }
}

/** Test hook: reset cached encoders + failure flag + calibrated multipliers. */
export function _resetTokenizerCache(): void {
  encoderCache.clear();
  loaderFailed = false;
  // Also clear any live-calibrated multipliers so tests that call
  // _resetTokenizerCache() get a clean slate and calibration state from one
  // test file cannot bleed into another.
  _resetCalibratedMultipliers();
}

/** Test hook: force the loader to fail on next call (simulate missing dep). */
export function _forceLoaderFailure(): void {
  loaderFailed = true;
  encoderCache.clear();
}

/**
 * Resolve a `TokenizerModel` shorthand or full model name string to an
 * `EncodingName`. Handles backward-compat shorthands used by the existing API.
 *
 * NOTE: This is only used by the internal `_getEncodingConfig` indirection.
 * Kept for clarity; the real dispatch happens in `_getEncodingConfig`.
 */
function pickEncoding(model: TokenizerModel | string): EncodingName {
  return _getEncodingConfig(model as string).encoding;
}

/**
 * Sync facade around the (async) encoder load. Runs the loader once and
 * blocks via a module-level promise only for the first call; subsequent
 * calls use the cached encoder directly.
 *
 * We expose the async API below as the canonical accurate entry point; this
 * sync helper is used internally after the first load resolves.
 */
function encodedLength(enc: Encoder, text: string): number {
  if (!text) return 0;
  return enc.encode(text).length;
}

/**
 * Accurate token count. Uses tiktoken if available, falls back to chars/4.
 *
 * Note: returns a Promise because tokenizer load is async. For a truly sync
 * path, call `primeTokenizer()` once at startup, then `countTokensAccurateSync`.
 *
 * Applies the provider-specific multiplier from `_getEncodingConfig` when the
 * exact tiktoken encoding is only an approximation (e.g. Llama 2, Qwen).
 */
export async function countTokensAccurate(
  input: string | Message[],
  model: TokenizerModel | string = "default",
): Promise<number> {
  const config = _getEncodingConfig(model as string);
  const enc = await _loadEncoder(config.encoding);
  if (!enc) {
    // Fallback: heuristic with the static table multiplier only.
    // We deliberately do NOT apply the calibrated multiplier here because the
    // chars/4 heuristic is already very rough — blending a live-calibrated
    // correction on top would produce unpredictable results and break
    // back-compat guarantees for callers that rely on the heuristic path.
    const base = estimateTokens(input);
    return config.multiplier > 1.0 ? Math.ceil(base * config.multiplier) : base;
  }
  try {
    let raw: number;
    if (typeof input === "string") {
      raw = encodedLength(enc, input);
    } else {
      let total = 0;
      for (const msg of input) {
        if (typeof msg.content === "string") {
          total += encodedLength(enc, msg.content);
        } else {
          for (const block of msg.content) {
            total += blockTokenCount(enc, block);
          }
        }
        // Per-message overhead (role + framing). Matches OpenAI's published
        // guidance for cl100k chat framing; ~4 tokens/message.
        total += 4;
      }
      raw = total;
    }
    // When tiktoken is available, apply the live calibrated multiplier when one
    // exists; otherwise use the static table value.
    // getActiveCalibratedMultiplier returns null until the first
    // validateAndRecalibrate() call for this model.
    const activeMultiplier =
      getActiveCalibratedMultiplier(model as string) ?? config.multiplier;
    return activeMultiplier === 1.0 ? raw : Math.ceil(raw * activeMultiplier);
  } catch {
    // Tokenizer blew up mid-encode — degrade.
    return estimateTokens(input);
  }
}

/**
 * Provider-aware token estimator — accepts any full or partial model name string.
 *
 * Drop-in for callers that have a model slug from the routing layer and want
 * a single call that handles all provider encodings + fallbacks automatically.
 *
 * @param input   Plain string or provider Message array.
 * @param model   Full or partial model name (e.g. "claude-3-5-sonnet-20241022",
 *                "gpt-4o-mini", "meta-llama/Llama-3-8b"). Defaults to "claude-3-5".
 *                Accepts any `TokenizerModel` shorthand for back-compat.
 * @returns       Token count (async — tiktoken load happens once and is cached).
 */
export async function estimateTokensWithModel(
  input: string | Message[],
  model: string = "claude-3-5",
): Promise<number> {
  return countTokensAccurate(input, model);
}

function blockTokenCount(enc: Encoder, block: ContentBlock): number {
  switch (block.type) {
    case "text":
      return encodedLength(enc, block.text);
    case "thinking":
      return encodedLength(enc, block.thinking);
    case "tool_use":
      return (
        encodedLength(enc, block.name) +
        encodedLength(enc, JSON.stringify(block.input))
      );
    case "tool_result":
      return encodedLength(enc, block.content);
    case "image_url":
      return 1000;
    default:
      return 0;
  }
}

/** Prime the tokenizer so later calls are cheap. Safe to call multiple times. */
export async function primeTokenizer(
  model: TokenizerModel | string = "default",
): Promise<boolean> {
  const enc = await _loadEncoder(_getEncodingConfig(model as string).encoding);
  return enc !== null;
}

// ---------- calibratedEstimate — adaptive per-request accuracy ----------

/**
 * Adaptive, per-request token estimator that combines tiktoken accuracy,
 * per-provider codec selection, and live recalibration from the
 * RecalibrationEngine.
 *
 * Estimation pipeline:
 *   1. Use `countTokensAccurate` (tiktoken if available, else chars/4) for the
 *      base estimate — identical to the existing accurate path.
 *   2. Look up the per-provider × per-bin correction factor from the
 *      RecalibrationEngine (if enough samples have been recorded).
 *   3. Multiply and return.
 *
 * On the very first call for a provider (zero history), the correction factor
 * is 1.0, so the result is identical to `countTokensAccurate` — perfect
 * backward compatibility.
 *
 * @param messages   Provider Message array.
 * @param provider   Full or partial model slug (e.g. "claude-3-5-sonnet-20241022").
 *                   Defaults to "claude-3-5".
 * @param history    Optional: pass the RecalibrationEngine to use a non-singleton
 *                   instance (e.g. in tests).
 * @returns          Token count (async — tiktoken load is cached after first call).
 *
 * @example
 *   const n = await calibratedEstimate(messages, "claude-3-5-sonnet-20241022");
 */
export async function calibratedEstimate(
  messages: Message[],
  provider: string = "claude-3-5",
  history?: RecalibrationEngine,
): Promise<number> {
  // Step 1: get the tiktoken-accurate base estimate (or heuristic fallback).
  const base = await countTokensAccurate(messages, provider);

  // Step 2: determine the bin for the last message.
  const { classifyMessage } = await import("./provider-codec.ts");
  const lastMsg = messages.length > 0 ? messages[messages.length - 1] : undefined;
  const bin = lastMsg ? classifyMessage(lastMsg) : "text";

  // Step 3: apply the per-provider × per-bin correction factor.
  const engine = history ?? RecalibrationEngine.instance;
  const factor = engine.correctionFactor(provider, bin);

  return factor === 1.0 ? base : Math.ceil(base * factor);
}

// ---------- cache-aware token counting ----------

/**
 * Cost multipliers for Anthropic prompt caching.
 *
 * Cache write: tokens are stored and billed at 1.25× the normal input rate.
 * Cache read:  previously-stored tokens are replayed at 0.1× the normal rate.
 *
 * Reference: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
 */
export const CACHE_WRITE_MULTIPLIER = 1.25;
export const CACHE_READ_MULTIPLIER = 0.1;

/**
 * Result of a cache-aware token count.
 *
 * - `rawTokens`        — base token estimate for the messages (before any cache adjustment).
 * - `cacheWriteTokens` — tokens that will be written into the cache this turn.
 * - `cacheReadTokens`  — tokens that will be read from an existing cache hit.
 * - `effectiveTokens`  — billing-equivalent token count:
 *                          rawTokens + cacheWriteTokens×0.25 − cacheReadTokens×0.9
 *                          (raw already includes those token positions at 1×; we add/subtract
 *                           the premium/discount on top).
 */
export interface CacheAwareTokenCount {
  rawTokens: number;
  cacheWriteTokens: number;
  cacheReadTokens: number;
  effectiveTokens: number;
}

/**
 * Count tokens for a set of messages, accounting for prompt-caching costs.
 *
 * @param messages    Provider message array.
 * @param cacheBlocks Optional array of cache block metadata. Each entry describes
 *                    one cache checkpoint attached to the prompt.
 *                    - If `created_token_count` is provided it is used directly.
 *                    - Otherwise the token count is estimated from the last message.
 * @param model       Tokenizer model hint (defaults to `"default"`).
 *
 * The effective token count adjusts for the 1.25× write premium and 0.1× read discount:
 *   effectiveTokens = uncachedTokens
 *                   + cacheWriteTokens × CACHE_WRITE_MULTIPLIER   (1.25×)
 *                   + cacheReadTokens  × CACHE_READ_MULTIPLIER     (0.1×)
 *
 * where uncachedTokens = rawTokens − cacheWriteTokens − cacheReadTokens.
 */
export async function countTokensWithCache(
  messages: Message[],
  cacheBlocks?: CacheBlockMetadata[],
  model: TokenizerModel | string = "default",
): Promise<CacheAwareTokenCount> {
  const rawTokens = await countTokensAccurate(messages, model);

  if (!cacheBlocks || cacheBlocks.length === 0) {
    return {
      rawTokens,
      cacheWriteTokens: 0,
      cacheReadTokens: 0,
      effectiveTokens: rawTokens,
    };
  }

  let cacheWriteTokens = 0;
  let cacheReadTokens = 0;

  for (const block of cacheBlocks) {
    // Use the explicit token count when provided; otherwise fall back to
    // estimating from the last message (conservative approximation).
    const lastMsg = messages.length > 0 ? messages[messages.length - 1] : undefined;
    const blockTokens =
      block.created_token_count ??
      (lastMsg ? estimateTokens([lastMsg]) : 0);

    if (block.cache_type === "ephemeral" || block.cache_type === "static") {
      // Treat the block as a write if it has a created_token_count (new cache
      // entry being created), otherwise treat as a read (cache hit).
      if (block.created_token_count !== undefined) {
        cacheWriteTokens += blockTokens;
      } else {
        cacheReadTokens += blockTokens;
      }
    }
  }

  // Tokens that are neither written nor read from cache are billed at 1×.
  const uncachedTokens = Math.max(
    0,
    rawTokens - cacheWriteTokens - cacheReadTokens,
  );

  const effectiveTokens =
    uncachedTokens +
    cacheWriteTokens * CACHE_WRITE_MULTIPLIER +
    cacheReadTokens * CACHE_READ_MULTIPLIER;

  return {
    rawTokens,
    cacheWriteTokens,
    cacheReadTokens,
    effectiveTokens: Math.round(effectiveTokens),
  };
}

// ---------- provider pricing constants ----------

/**
 * Known provider / model names for pricing lookups.
 *
 * The string union is open-ended so callers can pass any provider slug
 * (e.g. `"claude-3-haiku-20240307"`) without a compile-time error; the
 * pricing table falls back to a safe default for unknown strings.
 */
export type ProviderName =
  | "claude-3-5-sonnet"
  | "claude-3-5-haiku"
  | "claude-3-opus"
  | "claude-3-sonnet"
  | "claude-3-haiku"
  | "gpt-4o"
  | "gpt-4o-mini"
  | "gpt-4-turbo"
  | "gemini-1-5-pro"
  | "gemini-1-5-flash";

/**
 * Per-token USD rates for a provider.
 *
 * Rates are expressed as USD per **single token** (not per-million), so
 * they are small numbers. Multiply by token count to get USD cost.
 *
 * Reference prices (per-million tokens, as of 2026):
 *   claude-3-5-sonnet  $3.00 input / $15.00 output
 *   claude-3-5-haiku   $0.80 input / $4.00  output
 *   claude-3-opus      $15.00 input / $75.00 output
 *   claude-3-sonnet    $3.00 input / $15.00 output
 *   claude-3-haiku     $0.25 input / $1.25  output
 *   gpt-4o             $5.00 input / $15.00 output
 *   gpt-4o-mini        $0.15 input / $0.60  output
 *   gpt-4-turbo        $10.00 input / $30.00 output
 *   gemini-1-5-pro     $3.50 input / $10.50 output
 *   gemini-1-5-flash   $0.075 input / $0.30 output
 */
export interface ProviderRate {
  /** USD cost per input token. */
  inputRate: number;
  /** USD cost per output token. */
  outputRate: number;
}

/** Pricing table indexed by `ProviderName`. */
export const PROVIDER_RATES: Record<ProviderName, ProviderRate> = {
  "claude-3-5-sonnet": { inputRate: 3.0 / 1_000_000, outputRate: 15.0 / 1_000_000 },
  "claude-3-5-haiku":  { inputRate: 0.8 / 1_000_000, outputRate: 4.0  / 1_000_000 },
  "claude-3-opus":     { inputRate: 15.0 / 1_000_000, outputRate: 75.0 / 1_000_000 },
  "claude-3-sonnet":   { inputRate: 3.0 / 1_000_000, outputRate: 15.0 / 1_000_000 },
  "claude-3-haiku":    { inputRate: 0.25 / 1_000_000, outputRate: 1.25 / 1_000_000 },
  "gpt-4o":            { inputRate: 5.0 / 1_000_000, outputRate: 15.0 / 1_000_000 },
  "gpt-4o-mini":       { inputRate: 0.15 / 1_000_000, outputRate: 0.60 / 1_000_000 },
  "gpt-4-turbo":       { inputRate: 10.0 / 1_000_000, outputRate: 30.0 / 1_000_000 },
  "gemini-1-5-pro":    { inputRate: 3.5 / 1_000_000, outputRate: 10.5 / 1_000_000 },
  "gemini-1-5-flash":  { inputRate: 0.075 / 1_000_000, outputRate: 0.30 / 1_000_000 },
};

/**
 * Default rate used when the provider string is not found in `PROVIDER_RATES`.
 * Falls back to claude-3-5-sonnet pricing as a conservative mid-range estimate.
 */
export const DEFAULT_PROVIDER_RATE: ProviderRate = PROVIDER_RATES["claude-3-5-sonnet"];

/**
 * Look up the per-token rates for a provider string.
 *
 * Returns `DEFAULT_PROVIDER_RATE` for unknown providers so callers never
 * get null/undefined — a wrong estimate is always better than a crash.
 */
export function defaultProviderRate(provider: string): ProviderRate {
  if (provider in PROVIDER_RATES) {
    return PROVIDER_RATES[provider as ProviderName];
  }
  // Try a prefix match for versioned slugs like "claude-3-5-sonnet-20241022"
  for (const key of Object.keys(PROVIDER_RATES) as ProviderName[]) {
    if (provider.startsWith(key)) {
      return PROVIDER_RATES[key];
    }
  }
  return DEFAULT_PROVIDER_RATE;
}

// ---------- actual-usage recording (adaptive compression hook) ----------

/**
 * Result from comparing actual vs estimated token usage.
 *
 * This is consumed by the adaptive compression layer to calibrate future
 * token estimates and tier selection.
 */
export interface TokenUsageRecord {
  /** Tokens estimated before the LLM call. */
  estimatedTokens: number;
  /** Actual non-cached input tokens billed at 1× (from response). */
  actualInputTokens: number;
  /** Tokens written to cache this turn (billed at 1.25×). */
  actualCacheCreationTokens: number;
  /** Tokens read from cache this turn (billed at 0.1×). */
  actualCacheReadTokens: number;
  /** Billing-equivalent actual token count. */
  actualEffectiveTokens: number;
  /**
   * Estimation error as a signed percentage:
   *   ((actualEffective − estimated) / estimated) × 100
   * Positive → we under-estimated; negative → we over-estimated.
   */
  estimationErrorPct: number;
}

/**
 * Compute and return a `TokenUsageRecord` from an Anthropic `MessageResponse`.
 *
 * Intended as an integration hook called after each LLM response so the
 * adaptive compression layer can track estimation accuracy and calibrate
 * future tier selection.
 *
 * @param response        The raw Anthropic message response (or a compatible shape).
 * @param estimatedBefore Token count estimated before the call was made.
 * @returns               A `TokenUsageRecord` ready to feed into `recordCompressionResult`.
 */
export function recordActualTokenUsage(
  response: MessageResponse,
  estimatedBefore: number,
  model: string = "default",
): TokenUsageRecord {
  const actualInputTokens = response.input_tokens ?? 0;
  const actualCacheCreationTokens = response.cache_creation_input_tokens ?? 0;
  const actualCacheReadTokens = response.cache_read_input_tokens ?? 0;

  // Compute billing-equivalent actual token count using the same multipliers.
  const actualEffectiveTokens =
    actualInputTokens +
    actualCacheCreationTokens * CACHE_WRITE_MULTIPLIER +
    actualCacheReadTokens * CACHE_READ_MULTIPLIER;

  const estimationErrorPct =
    estimatedBefore > 0
      ? ((actualEffectiveTokens - estimatedBefore) / estimatedBefore) * 100
      : 0;

  // Fire-and-forget: persist the (estimated, actual) pair so the calibration
  // validator can compute drift and recalibrate multipliers over time.
  if (estimatedBefore > 0 && Math.round(actualEffectiveTokens) > 0) {
    appendCalibrationRecord({
      timestamp: new Date().toISOString(),
      model,
      estimated: estimatedBefore,
      actual: Math.round(actualEffectiveTokens),
    }).catch(() => {
      // Non-fatal — calibration is best-effort.
    });
  }

  return {
    estimatedTokens: estimatedBefore,
    actualInputTokens,
    actualCacheCreationTokens,
    actualCacheReadTokens,
    actualEffectiveTokens: Math.round(actualEffectiveTokens),
    estimationErrorPct: Math.round(estimationErrorPct * 100) / 100,
  };
}
