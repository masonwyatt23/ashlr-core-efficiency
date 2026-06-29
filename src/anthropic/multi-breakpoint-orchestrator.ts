/**
 * MultiBreakpointOrchestrator — turn-tiering cache-breakpoint placement for
 * multi-turn Anthropic Messages requests.
 *
 * `cache-breakpoint-optimizer.ts` answers "given a flat message array and
 * session history, which message indices should carry a cache_control marker?".
 * This module operates one level up: it classifies each transcript turn by
 * *stability tier* (frozen / stable / volatile) and orchestrates placement of
 * up to MAX_BREAKPOINTS (4) markers across those tiers, honoring the Anthropic
 * prompt-caching invariants:
 *
 *   - Prompt caching is a prefix match: stable content MUST precede volatile
 *     content, or the volatile bytes invalidate every breakpoint after them.
 *   - Max 4 cache_control breakpoints per request.
 *   - A breakpoint walks back at most 20 content blocks to find a prior cache
 *     entry; in long turns an intermediate breakpoint is needed every ~15
 *     blocks or the next request silently misses.
 *   - A prefix below the model's minimum cacheable size (≈1024–4096 tokens)
 *     silently won't cache — don't spend a breakpoint on it.
 *
 * The orchestrator does not call the Anthropic SDK; it produces a strategy
 * (breakpoint turn indices + rationale + ROI) that a caller applies to its own
 * request. It reuses MAX_BREAKPOINTS and recommendBreakpoints from
 * cache-breakpoint-optimizer.ts rather than re-deriving them.
 */

import { existsSync } from "fs";
import { appendFile, mkdir, readFile } from "fs/promises";
import { dirname, join } from "path";
import { homedir } from "os";
import { MAX_BREAKPOINTS, type Message } from "./cache-breakpoint-optimizer.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Cache-read price multiplier relative to base input (~0.1×). */
const CACHE_READ_MULT = 0.1;
/** Cache-write price multiplier relative to base input (~1.25× for 5-min TTL). */
const CACHE_WRITE_MULT = 1.25;
/**
 * A breakpoint walks back at most 20 content blocks to find a prior cache
 * entry. We place an intermediate breakpoint before crossing this many turns
 * within a single tier so long tiers don't outrun the lookback window.
 */
const LOOKBACK_WINDOW_TURNS = 15;
/**
 * Default minimum cacheable prefix (tokens). Below this the API silently
 * refuses to cache, so a breakpoint there is wasted. Caller can override per
 * model (Opus = 4096, Sonnet = 2048, older Sonnet = 1024).
 */
const DEFAULT_MIN_CACHEABLE_TOKENS = 1024;
/** Default Anthropic input price per million tokens (claude-sonnet rate). */
const DEFAULT_INPUT_PRICE_PER_MTOKEN = 3.0;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Stability tier of a transcript turn. Drives breakpoint placement:
 *   - "frozen"   — never changes across requests (system prompt, tool defs,
 *                  pinned few-shot examples). The most valuable cache prefix.
 *   - "stable"   — changes per session but not per turn (retrieved docs,
 *                  established conversation history).
 *   - "volatile" — changes every request (latest user question, timestamps,
 *                  per-request IDs). Must come last; never gets a breakpoint.
 */
export type TurnTier = "frozen" | "stable" | "volatile";

/**
 * A single turn in a multi-turn transcript, tagged with its stability tier.
 * Extends the optimizer's `Message` so the two modules interoperate.
 */
export interface TranscriptTurn extends Message {
  /**
   * Stability tier. When omitted, the orchestrator infers it: the contiguous
   * leading run of system turns is "frozen", the trailing turn is "volatile",
   * and everything between is "stable".
   */
  tier?: TurnTier;
}

/** Options controlling orchestration. */
export interface OrchestratorOptions {
  /** Max breakpoints to place (1–4). Clamped to MAX_BREAKPOINTS. Default 4. */
  maxBreakpoints?: number;
  /** Anthropic input price per million tokens (USD). Default $3.00/M. */
  inputPricePerMToken?: number;
  /**
   * Minimum cacheable prefix in tokens for the target model. A tier boundary
   * whose cumulative prefix is below this does not receive a breakpoint.
   * Default 1024.
   */
  minCacheableTokens?: number;
  /**
   * Expected number of times the cached prefix will be read before it is
   * rewritten (for ROI amortization). Default 3.
   */
  expectedReadsPerWrite?: number;
  /** Override the audit file path (useful in tests). */
  auditPath?: string;
}

/** A single placed breakpoint within a strategy. */
export interface MultiBreakpointStrategy {
  /** Zero-based turn index that carries the cache_control marker. */
  turnIndex: number;
  /** The tier of the turn at this boundary. */
  tier: TurnTier;
  /** Cumulative prefix tokens up to and including this turn. */
  prefixTokens: number;
  /** Net ROI of this breakpoint in micro-USD (read savings − amortized write). */
  netRoiMicroUsd: number;
  /** Why this position was chosen (tier boundary vs lookback split). */
  reason: "tier_boundary" | "lookback_split";
}

/**
 * Full multi-breakpoint cache strategy for a request.
 *
 * (Named CacheStrategy_MultiBreakpoint to avoid colliding with the
 * `CacheStrategy` exported from cache-audit.ts.)
 */
export interface CacheStrategy_MultiBreakpoint {
  /** ISO-8601 timestamp when the strategy was computed. */
  generatedAt: string;
  /** Ordered (ascending) breakpoint placements. Length ≤ maxBreakpoints. */
  breakpoints: MultiBreakpointStrategy[];
  /** Total turns analyzed. */
  turnCount: number;
  /** Total prefix tokens of the whole transcript. */
  totalTokens: number;
  /** Sum of net ROI across all placed breakpoints, micro-USD. */
  totalNetRoiMicroUsd: number;
  /** Human-readable summary. */
  summary: string;
}

/** An audit record persisted per orchestration, for offline analysis. */
export interface OrchestratorAuditRecord {
  generatedAt: string;
  turnCount: number;
  breakpointCount: number;
  totalNetRoiMicroUsd: number;
  tierHistogram: Record<TurnTier, number>;
}

// ---------------------------------------------------------------------------
// Token estimation (local; matches the optimizer's 4-chars/token heuristic)
// ---------------------------------------------------------------------------

function estimateTurnTokens(turn: TranscriptTurn): number {
  if (turn.tokenCount !== undefined && turn.tokenCount > 0) return turn.tokenCount;
  if (typeof turn.content === "string") return Math.ceil(turn.content.length / 4);
  let total = 0;
  for (const block of turn.content) {
    if (block.text && typeof block.text === "string") {
      total += Math.ceil(block.text.length / 4);
    } else {
      total += 50; // nominal non-text block
    }
  }
  return Math.max(total, 1);
}

// ---------------------------------------------------------------------------
// Tier inference
// ---------------------------------------------------------------------------

/**
 * Infer the stability tier of every turn when not explicitly tagged:
 *   - leading contiguous run of `system` turns → "frozen"
 *   - the final turn → "volatile"
 *   - everything else → "stable"
 *
 * Explicit `tier` values on a turn always win.
 */
function inferTiers(turns: TranscriptTurn[]): TurnTier[] {
  const n = turns.length;
  const tiers: TurnTier[] = new Array(n);
  let leadingSystem = true;
  for (let i = 0; i < n; i++) {
    const explicit = turns[i]!.tier;
    if (explicit) {
      tiers[i] = explicit;
      if (turns[i]!.role !== "system") leadingSystem = false;
      continue;
    }
    if (leadingSystem && turns[i]!.role === "system") {
      tiers[i] = "frozen";
    } else {
      leadingSystem = false;
      tiers[i] = i === n - 1 ? "volatile" : "stable";
    }
  }
  return tiers;
}

/**
 * Tier ordering rank. A prefix is only safely cacheable while tiers are
 * non-decreasing in volatility — once we hit a volatile turn, no later
 * breakpoint can be trusted.
 */
const TIER_RANK: Record<TurnTier, number> = { frozen: 0, stable: 1, volatile: 2 };

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/**
 * Places cache breakpoints across a tiered multi-turn transcript.
 *
 * Usage:
 *   const orch = new MultiBreakpointOrchestrator({ minCacheableTokens: 4096 });
 *   const strategy = orch.orchestrate(turns);
 *   await orch.recordAudit(strategy, turns);   // optional telemetry
 */
export class MultiBreakpointOrchestrator {
  private readonly maxBreakpoints: number;
  private readonly inputPricePerMToken: number;
  private readonly minCacheableTokens: number;
  private readonly expectedReadsPerWrite: number;
  private readonly auditPath: string;

  constructor(options: OrchestratorOptions = {}) {
    this.maxBreakpoints = Math.max(1, Math.min(options.maxBreakpoints ?? MAX_BREAKPOINTS, MAX_BREAKPOINTS));
    this.inputPricePerMToken = options.inputPricePerMToken ?? DEFAULT_INPUT_PRICE_PER_MTOKEN;
    this.minCacheableTokens = options.minCacheableTokens ?? DEFAULT_MIN_CACHEABLE_TOKENS;
    this.expectedReadsPerWrite = Math.max(1, options.expectedReadsPerWrite ?? 3);
    this.auditPath = options.auditPath ?? defaultAuditPath();
  }

  /**
   * Compute the breakpoint strategy for `turns`.
   *
   * Algorithm:
   *   1. Infer/read tiers; reject placement once a volatile turn is reached
   *      (the cacheable region is the frozen+stable prefix).
   *   2. Candidate positions are the last turn of each tier *run* (tier
   *      boundaries) plus lookback splits inside any run longer than
   *      LOOKBACK_WINDOW_TURNS.
   *   3. Keep only candidates whose cumulative prefix ≥ minCacheableTokens.
   *   4. Score each by net ROI; keep the top `maxBreakpoints`, in ascending
   *      turn order. Boundaries closest to the volatile suffix are most
   *      valuable (largest cached prefix), so ties favor later positions.
   */
  orchestrate(turns: TranscriptTurn[]): CacheStrategy_MultiBreakpoint {
    const generatedAt = new Date().toISOString();
    const tiers = inferTiers(turns);

    // Cumulative prefix tokens up to and including each turn.
    const cumTokens: number[] = [];
    let running = 0;
    for (let i = 0; i < turns.length; i++) {
      running += estimateTurnTokens(turns[i]!);
      cumTokens[i] = running;
    }
    const totalTokens = running;

    // The cacheable region ends at the last turn before the first volatile
    // turn (volatile content can't anchor a reusable prefix).
    let cacheableEnd = turns.length - 1;
    for (let i = 0; i < tiers.length; i++) {
      if (tiers[i] === "volatile") {
        cacheableEnd = i - 1;
        break;
      }
    }

    // Collect candidate breakpoint positions.
    const candidates: Array<{ index: number; reason: MultiBreakpointStrategy["reason"] }> = [];
    let runStart = 0;
    for (let i = 0; i <= cacheableEnd; i++) {
      const isTierBoundary =
        i === cacheableEnd || (i + 1 <= cacheableEnd && TIER_RANK[tiers[i + 1]!] !== TIER_RANK[tiers[i]!]);

      // Lookback split: within a long run, force an intermediate breakpoint.
      if (i - runStart >= LOOKBACK_WINDOW_TURNS && !isTierBoundary) {
        candidates.push({ index: i, reason: "lookback_split" });
        runStart = i;
      }

      if (isTierBoundary) {
        candidates.push({ index: i, reason: "tier_boundary" });
        runStart = i + 1;
      }
    }

    // Score candidates by net ROI; drop those below the cacheable minimum.
    const scored: MultiBreakpointStrategy[] = [];
    for (const cand of candidates) {
      const prefixTokens = cumTokens[cand.index] ?? 0;
      if (prefixTokens < this.minCacheableTokens) continue;
      const netRoiMicroUsd = this.netRoiMicroUsd(prefixTokens);
      if (netRoiMicroUsd <= 0) continue;
      scored.push({
        turnIndex: cand.index,
        tier: tiers[cand.index]!,
        prefixTokens,
        netRoiMicroUsd,
        reason: cand.reason,
      });
    }

    // Keep the top-N by ROI (descending), then return in ascending turn order.
    scored.sort((a, b) => b.netRoiMicroUsd - a.netRoiMicroUsd || b.turnIndex - a.turnIndex);
    const kept = scored.slice(0, this.maxBreakpoints);
    kept.sort((a, b) => a.turnIndex - b.turnIndex);

    const totalNetRoiMicroUsd = kept.reduce((s, b) => s + b.netRoiMicroUsd, 0);

    return {
      generatedAt,
      breakpoints: kept,
      turnCount: turns.length,
      totalTokens,
      totalNetRoiMicroUsd,
      summary:
        kept.length === 0
          ? `No cacheable prefix ≥ ${this.minCacheableTokens} tokens; skip breakpoints.`
          : `Placed ${kept.length} breakpoint(s) at turns [${kept.map((b) => b.turnIndex).join(", ")}] ` +
            `for ~${(totalNetRoiMicroUsd / 1_000).toFixed(1)} milli-USD/request net savings.`,
    };
  }

  /**
   * Net ROI (micro-USD) of caching a prefix of `prefixTokens`:
   *   read savings  = prefixTokens × (1 − CACHE_READ_MULT)   [per read]
   *   write overhead = prefixTokens × (CACHE_WRITE_MULT − 1) [one-time]
   *   net (amortized over expectedReadsPerWrite reads), priced at input rate.
   */
  private netRoiMicroUsd(prefixTokens: number): number {
    const readSavingsTokens = prefixTokens * (1 - CACHE_READ_MULT);
    const writeCostTokens =
      (prefixTokens * (CACHE_WRITE_MULT - 1)) / this.expectedReadsPerWrite;
    const netTokens = readSavingsTokens - writeCostTokens;
    // micro-USD = tokens × (price per token) × 1e6
    //           = tokens × (pricePerMToken / 1e6) × 1e6 = tokens × pricePerMToken
    return netTokens * this.inputPricePerMToken;
  }

  /** Append an audit record for offline analysis. Never throws on I/O. */
  async recordAudit(
    strategy: CacheStrategy_MultiBreakpoint,
    turns: TranscriptTurn[],
  ): Promise<void> {
    const tiers = inferTiers(turns);
    const tierHistogram: Record<TurnTier, number> = { frozen: 0, stable: 0, volatile: 0 };
    for (const t of tiers) tierHistogram[t]++;

    const record: OrchestratorAuditRecord = {
      generatedAt: strategy.generatedAt,
      turnCount: strategy.turnCount,
      breakpointCount: strategy.breakpoints.length,
      totalNetRoiMicroUsd: strategy.totalNetRoiMicroUsd,
      tierHistogram,
    };
    try {
      await mkdir(dirname(this.auditPath), { recursive: true });
      await appendFile(this.auditPath, JSON.stringify(record) + "\n", "utf-8");
    } catch {
      // best-effort telemetry
    }
  }

  /** Load all persisted audit records (skips malformed lines). */
  async loadAudit(): Promise<OrchestratorAuditRecord[]> {
    if (!existsSync(this.auditPath)) return [];
    let raw: string;
    try {
      raw = await readFile(this.auditPath, "utf-8");
    } catch {
      return [];
    }
    const out: OrchestratorAuditRecord[] = [];
    for (const line of raw.split("\n")) {
      const t = line.trim();
      if (!t) continue;
      try {
        out.push(JSON.parse(t) as OrchestratorAuditRecord);
      } catch {
        // skip
      }
    }
    return out;
  }
}

// ---------------------------------------------------------------------------
// Convenience wrapper
// ---------------------------------------------------------------------------

/**
 * One-shot helper: orchestrate a strategy for `turns` without managing an
 * instance. Does not persist an audit record.
 */
export function recommendCacheStrategy(
  turns: TranscriptTurn[],
  options: OrchestratorOptions = {},
): CacheStrategy_MultiBreakpoint {
  return new MultiBreakpointOrchestrator(options).orchestrate(turns);
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

function defaultAuditPath(): string {
  return join(homedir(), ".ashlr", "multi-breakpoint-orchestrator.jsonl");
}
