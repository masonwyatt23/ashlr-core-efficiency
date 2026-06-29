/**
 * cache-audit.ts — JSONL audit trail for prompt-cache events.
 *
 * Every time a MessageResponse arrives the caller records a CacheAuditEntry
 * via `appendCacheAudit`. The optimizer later replays these to learn which
 * breakpoint strategies are actually saving money.
 *
 * File layout mirrors genome/retrieval-adapter: one append-only JSONL per
 * working directory, stored under <cwd>/.ashlr/cache-audit.jsonl.
 */

import { appendJsonl, readJsonl } from "../genome/jsonl.ts";
import { join } from "path";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Sub-directory (relative to cwd) where cache audit files live. */
export const CACHE_AUDIT_DIR = ".ashlr";

/** Filename for the cache audit trail. */
export const CACHE_AUDIT_FILE = "cache-audit.jsonl";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * The cache-breakpoint strategy that was active when the request was made.
 *
 * - `system-only`      — only the system prompt is breakpointed.
 * - `system+context`   — system prompt + static user-context messages.
 * - `multi-turn`       — windowed multi-turn history with rolling breakpoints.
 * - `tools-only`       — only tool definitions are breakpointed.
 * - `full`             — tools + system + context (all four Anthropic slots).
 * - `none`             — no caching applied.
 */
export type CacheStrategy =
  | "system-only"
  | "system+context"
  | "multi-turn"
  | "tools-only"
  | "full"
  | "none";

/**
 * A single record appended to the JSONL audit file after each API response.
 */
export interface CacheAuditEntry {
  /** Unique ID for this audit record. */
  id: string;
  /** ISO-8601 timestamp. */
  timestamp: string;
  /** Provider being used (e.g. "anthropic", "openai"). */
  provider: string;
  /** Model ID (e.g. "claude-sonnet-4"). */
  modelId: string;
  /** The breakpoint strategy active for this request. */
  strategy: CacheStrategy;
  /** Raw input tokens billed at 1× rate (from MessageResponse). */
  inputTokens: number;
  /** Tokens written into cache (billed at 1.25×). Zero when no write occurred. */
  cacheCreationTokens: number;
  /** Tokens read from cache (billed at 0.1×). Zero on a miss. */
  cacheReadTokens: number;
  /** Output tokens generated. */
  outputTokens: number;
  /**
   * Effective cost in micro-USD (1e-6 USD) for this request, accounting for
   * cache pricing differentials. Stored as integer to avoid float precision drift.
   */
  effectiveCostMicroUsd: number;
  /**
   * Hypothetical cost without any caching (all tokens at standard input rate).
   * Used to compute savings ROI.
   */
  baselineCostMicroUsd: number;
  /**
   * Optional session tag so a single audit trail can be segmented by session.
   */
  sessionId?: string;
}

/**
 * Compact stats summary built from a slice of audit entries.
 * Returned by `summarizeAuditWindow`.
 */
export interface CacheAuditSummary {
  strategy: CacheStrategy;
  sampleCount: number;
  totalInputTokens: number;
  totalCacheCreationTokens: number;
  totalCacheReadTokens: number;
  totalOutputTokens: number;
  totalEffectiveCostMicroUsd: number;
  totalBaselineCostMicroUsd: number;
  /** Fraction of requests where at least one token was read from cache. */
  hitRate: number;
  /**
   * Average saving per request vs. the no-cache baseline.
   * Positive means caching is saving money; negative means it is costing more.
   */
  avgSavingMicroUsd: number;
}

// ---------------------------------------------------------------------------
// Path helper
// ---------------------------------------------------------------------------

export function cacheAuditPath(cwd: string): string {
  return join(cwd, CACHE_AUDIT_DIR, CACHE_AUDIT_FILE);
}

// ---------------------------------------------------------------------------
// Pricing helpers
// ---------------------------------------------------------------------------

/**
 * Provider-level cache pricing multipliers.
 * Matches the constants used in tokens/index.ts (CACHE_WRITE_MULTIPLIER = 1.25,
 * CACHE_READ_MULTIPLIER = 0.1) and the EXTENDED_CONTEXT_MODELS registry.
 */
const CACHE_WRITE_MULT = 1.25;
const CACHE_READ_MULT  = 0.1;

/**
 * Compute effective cost in micro-USD given token counts and per-million-token
 * input price. Output and cache tiers use the standard Anthropic price ratios.
 *
 * @param inputTokens          Normal (non-cached) input tokens.
 * @param cacheCreationTokens  Tokens written into cache (1.25× input rate).
 * @param cacheReadTokens      Tokens read from cache (0.1× input rate).
 * @param outputTokens         Output tokens.
 * @param inputPricePerMToken  USD per million standard input tokens.
 * @param outputPricePerMToken USD per million output tokens.
 * @returns                    Cost in micro-USD (integer-rounded).
 */
export function computeEffectiveCostMicroUsd(
  inputTokens: number,
  cacheCreationTokens: number,
  cacheReadTokens: number,
  outputTokens: number,
  inputPricePerMToken: number,
  outputPricePerMToken: number,
): number {
  const inputCost    = (inputTokens / 1_000_000) * inputPricePerMToken;
  const writeCost    = (cacheCreationTokens / 1_000_000) * inputPricePerMToken * CACHE_WRITE_MULT;
  const readCost     = (cacheReadTokens / 1_000_000) * inputPricePerMToken * CACHE_READ_MULT;
  const outputCost   = (outputTokens / 1_000_000) * outputPricePerMToken;
  return Math.round((inputCost + writeCost + readCost + outputCost) * 1_000_000);
}

/**
 * Compute the hypothetical cost as if no caching were used (all tokens at
 * standard 1× input rate + normal output rate).
 */
export function computeBaselineCostMicroUsd(
  inputTokens: number,
  cacheCreationTokens: number,
  cacheReadTokens: number,
  outputTokens: number,
  inputPricePerMToken: number,
  outputPricePerMToken: number,
): number {
  const totalInput = inputTokens + cacheCreationTokens + cacheReadTokens;
  const inputCost  = (totalInput / 1_000_000) * inputPricePerMToken;
  const outputCost = (outputTokens / 1_000_000) * outputPricePerMToken;
  return Math.round((inputCost + outputCost) * 1_000_000);
}

// ---------------------------------------------------------------------------
// Append + load
// ---------------------------------------------------------------------------

/**
 * Append one cache audit entry to the JSONL trail.
 *
 * Typically called immediately after receiving a MessageResponse from the API.
 * Parent directories are created automatically (via appendJsonl).
 */
export async function appendCacheAudit(
  cwd: string,
  entry: Omit<CacheAuditEntry, "id" | "timestamp">,
): Promise<string> {
  const id = `ca-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
  const record: CacheAuditEntry = {
    id,
    timestamp: new Date().toISOString(),
    ...entry,
  };
  await appendJsonl(cacheAuditPath(cwd), record);
  return id;
}

/**
 * Load all cache audit entries from the JSONL trail.
 * Returns an empty array if the file does not exist or is corrupt.
 */
export async function loadCacheAudit(cwd: string): Promise<CacheAuditEntry[]> {
  return readJsonl<CacheAuditEntry>(cacheAuditPath(cwd));
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

/**
 * Summarize a window of audit entries for a given strategy.
 *
 * Pass `windowSize` to limit to the most recent N records.
 * Returns `null` when there are no entries for the requested strategy.
 */
export function summarizeAuditWindow(
  entries: CacheAuditEntry[],
  strategy: CacheStrategy,
  windowSize = 100,
): CacheAuditSummary | null {
  const relevant = entries
    .filter((e) => e.strategy === strategy)
    .slice(-windowSize);

  if (relevant.length === 0) return null;

  let totalInputTokens          = 0;
  let totalCacheCreationTokens  = 0;
  let totalCacheReadTokens      = 0;
  let totalOutputTokens         = 0;
  let totalEffectiveCostMicroUsd = 0;
  let totalBaselineCostMicroUsd  = 0;
  let hitCount                  = 0;

  for (const e of relevant) {
    totalInputTokens          += e.inputTokens;
    totalCacheCreationTokens  += e.cacheCreationTokens;
    totalCacheReadTokens      += e.cacheReadTokens;
    totalOutputTokens         += e.outputTokens;
    totalEffectiveCostMicroUsd += e.effectiveCostMicroUsd;
    totalBaselineCostMicroUsd  += e.baselineCostMicroUsd;
    if (e.cacheReadTokens > 0) hitCount++;
  }

  const hitRate         = relevant.length > 0 ? hitCount / relevant.length : 0;
  const totalSaving     = totalBaselineCostMicroUsd - totalEffectiveCostMicroUsd;
  const avgSavingMicroUsd = relevant.length > 0 ? totalSaving / relevant.length : 0;

  return {
    strategy,
    sampleCount: relevant.length,
    totalInputTokens,
    totalCacheCreationTokens,
    totalCacheReadTokens,
    totalOutputTokens,
    totalEffectiveCostMicroUsd,
    totalBaselineCostMicroUsd,
    hitRate,
    avgSavingMicroUsd,
  };
}
