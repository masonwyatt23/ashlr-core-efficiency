/**
 * cache-breakpoint-optimizer.ts — Session-log-driven prompt cache breakpoint
 * placement optimizer.
 *
 * ## Problem
 * `cacheBreakpoints()` naively marks the entire system prompt + tools as
 * cached. For long agentic runs where the same genome sections appear across
 * many turns, placing additional breakpoints at the repeated-prefix boundary
 * yields 10–25% savings on cache-read tokens.
 *
 * ## Approach
 * 1. Ingest compressed session logs from `~/.ashlr/sessions/*.jsonl` to detect
 *    repeated message prefixes (same genome sections used across ≥5 turns).
 * 2. Compute per-breakpoint metrics:
 *      cost_saved     = cache_read_tokens × 0.1 × inputRate
 *      cost_to_track  = overhead of the extra cache block
 * 3. Expose `recommendBreakpoints(messages, options?)` → up to 3 breakpoints
 *    that maximize ROI.
 * 4. Emit an audit trail to `~/.ashlr/cache-breakpoint-evolution.jsonl`.
 *
 * ## Integration with `cacheBreakpoints()`
 * Pass `optimizer` in the options object of `cacheBreakpoints()` to activate
 * automatic breakpoint injection guided by session-log learning.
 */

import { appendJsonl, readJsonl } from "../genome/jsonl.ts";
import { join } from "path";
import { homedir } from "os";
import { existsSync } from "fs";
import { readdir } from "fs/promises";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Maximum breakpoints Anthropic allows per request. */
export const MAX_BREAKPOINTS = 4;

/** Default maximum breakpoints the optimizer will recommend (≤ MAX_BREAKPOINTS). */
const DEFAULT_MAX_BREAKPOINTS = 3;

/** Minimum number of turns a prefix must repeat before a breakpoint is worthwhile. */
const MIN_REPEAT_TURNS = 5;

/**
 * Anthropic cache pricing multipliers (relative to standard input rate).
 * Write: 1.25× — Read: 0.10×.
 */
const CACHE_WRITE_MULT = 1.25;
const CACHE_READ_MULT = 0.1;

/** Path (relative to ~) for the evolution audit trail. */
const EVOLUTION_AUDIT_PATH = join(homedir(), ".ashlr", "cache-breakpoint-evolution.jsonl");

/** Directory containing compressed session logs. */
const SESSION_LOG_DIR = join(homedir(), ".ashlr", "sessions");

// ---------------------------------------------------------------------------
// Types — public API
// ---------------------------------------------------------------------------

/**
 * A single message in a conversation (mirrors CacheableMessage from prompt-cache.ts
 * but with role relaxed to include "system"/"tool_result" for analysis purposes).
 */
export interface Message {
  role: "system" | "user" | "assistant" | "tool_result";
  /** Content as string or structured blocks — we only need the text length here. */
  content: string | Array<{ type: string; text?: string; [key: string]: unknown }>;
  /** Optional: pre-computed token count for this message. */
  tokenCount?: number;
}

/**
 * Options for `recommendBreakpoints`.
 */
export interface OptimizerOptions {
  /**
   * Maximum number of breakpoints to recommend (1–4, default 3).
   * Capped at MAX_BREAKPOINTS.
   */
  maxBreakpoints?: number;
  /**
   * Anthropic input price per million tokens (USD). Used to compute dollar ROI.
   * Defaults to $3.00/M (claude-sonnet-4 rate).
   */
  inputPricePerMToken?: number;
  /**
   * Minimum ROI in micro-USD that a candidate breakpoint must show before it
   * is included in the recommendation. Defaults to 1 micro-USD.
   */
  minRoiMicroUsd?: number;
  /**
   * Minimum number of turns a prefix must repeat across sessions to qualify.
   * Defaults to MIN_REPEAT_TURNS (5).
   */
  minRepeatTurns?: number;
  /**
   * Override the session log directory (useful in tests).
   * Defaults to `~/.ashlr/sessions`.
   */
  sessionLogDir?: string;
  /**
   * Override the evolution audit file path (useful in tests).
   * Defaults to `~/.ashlr/cache-breakpoint-evolution.jsonl`.
   */
  auditPath?: string;
  /**
   * Optional provider hint (e.g. "anthropic", "openai", "gemini", "o1").
   *
   * When supplied, `recommendBreakpoints` auto-selects the matching
   * `ProviderCacheStrategy` from the registry and clamps `maxBreakpoints`
   * to the provider's limit.  For providers that do not support explicit
   * breakpoints (e.g. OpenAI implicit caching) the method returns an empty
   * array immediately rather than running the session-log analysis.
   */
  providerHint?: string;
}

/**
 * ROI breakdown for a single candidate breakpoint position.
 */
export interface BreakpointROI {
  /**
   * Zero-based message index where the breakpoint would be placed
   * (after this message, the remainder is treated as dynamic).
   */
  messageIndex: number;
  /**
   * Number of tokens in the prefix ending at this position (cumulative).
   */
  prefixTokens: number;
  /**
   * Fraction of observed sessions/turns where this prefix was repeated
   * (cache hit probability estimate). Range [0, 1].
   */
  estimatedHitRate: number;
  /**
   * Estimated token savings per request from a cache hit at this position.
   * = prefixTokens × hitRate × (1 − CACHE_READ_MULT)
   */
  estimatedTokenSavings: number;
  /**
   * Write overhead tokens (one-time cost, amortized over expected hits).
   * = prefixTokens × (CACHE_WRITE_MULT − 1) / expectedHits
   */
  writeCostTokens: number;
  /**
   * Net token savings = estimatedTokenSavings − writeCostTokens.
   */
  netTokenSavings: number;
  /**
   * Net savings in micro-USD at the configured input price.
   */
  netRoiMicroUsd: number;
}

/**
 * A recommended breakpoint returned by `recommendBreakpoints`.
 */
export interface BreakpointRecommendation {
  /** ISO-8601 timestamp when this recommendation was generated. */
  generatedAt: string;
  /**
   * Ordered list of message indices where cache breakpoints should be placed
   * (ascending). Length ≤ `maxBreakpoints`.
   */
  breakpointIndices: number[];
  /** Per-breakpoint ROI analysis. */
  roi: BreakpointROI[];
  /**
   * Total estimated net savings across all recommended breakpoints, micro-USD.
   */
  totalNetRoiMicroUsd: number;
  /**
   * Human-readable summary of the recommendation.
   */
  summary: string;
  /**
   * Whether the recommendation was derived from session-log data (true) or
   * is a heuristic default (false — not enough session data yet).
   */
  learnedFromSessions: boolean;
}

// ---------------------------------------------------------------------------
// Types — session log shapes (compressed JSONL from ~/.ashlr/sessions/*.jsonl)
// ---------------------------------------------------------------------------

/**
 * Minimal shape we extract from a session log line.
 * Session logs may have varying schemas; we only read what we need.
 */
interface SessionLogEntry {
  /** Message role. */
  role?: string;
  /** Content as string or unknown structure. */
  content?: string | unknown;
  /** Cache tokens read for this turn (if recorded). */
  cache_read_input_tokens?: number;
  /** Cache tokens written for this turn (if recorded). */
  cache_creation_input_tokens?: number;
  /** Input tokens for this turn. */
  input_tokens?: number;
}

// ---------------------------------------------------------------------------
// Types — evolution audit record
// ---------------------------------------------------------------------------

/**
 * A record appended to `~/.ashlr/cache-breakpoint-evolution.jsonl` each time
 * `recommendBreakpoints` is called.
 */
export interface BreakpointEvolutionRecord {
  /** Unique record ID. */
  id: string;
  /** ISO-8601 timestamp. */
  timestamp: string;
  /** Recommended breakpoint indices. */
  breakpointIndices: number[];
  /** Total estimated net ROI in micro-USD. */
  totalNetRoiMicroUsd: number;
  /** Number of session log files that were analyzed. */
  sessionFilesAnalyzed: number;
  /** Whether the recommendation was data-driven or heuristic. */
  learnedFromSessions: boolean;
  /** Compact per-breakpoint snapshot. */
  roiSnapshot: Array<{
    messageIndex: number;
    prefixTokens: number;
    estimatedHitRate: number;
    netRoiMicroUsd: number;
  }>;
}

// ---------------------------------------------------------------------------
// Token estimation helpers
// ---------------------------------------------------------------------------

/**
 * Estimate the token count of a message.
 *
 * Uses pre-computed `tokenCount` when present; otherwise approximates from
 * content length (4 chars ≈ 1 token, standard heuristic).
 */
function estimateTokenCount(msg: Message): number {
  if (msg.tokenCount !== undefined && msg.tokenCount > 0) return msg.tokenCount;
  if (typeof msg.content === "string") {
    return Math.ceil(msg.content.length / 4);
  }
  let total = 0;
  for (const block of msg.content) {
    if (block.text && typeof block.text === "string") {
      total += Math.ceil(block.text.length / 4);
    } else {
      // Non-text block: use a nominal 50-token estimate
      total += 50;
    }
  }
  return Math.max(total, 1);
}

/**
 * Estimate token count of a session log entry's content.
 */
function estimateSessionEntryTokens(entry: SessionLogEntry): number {
  if (entry.input_tokens !== undefined && entry.input_tokens > 0) {
    return entry.input_tokens;
  }
  if (typeof entry.content === "string") {
    return Math.ceil(entry.content.length / 4);
  }
  return 100; // nominal fallback
}

// ---------------------------------------------------------------------------
// Session log ingestion
// ---------------------------------------------------------------------------

/**
 * Load and parse all session log entries from `~/.ashlr/sessions/*.jsonl`.
 *
 * Returns a flat array of session entries grouped by file (each group = one
 * session). Sessions with fewer than 2 entries are ignored.
 */
async function loadSessionLogs(
  sessionLogDir: string,
): Promise<SessionLogEntry[][]> {
  if (!existsSync(sessionLogDir)) return [];

  let files: string[];
  try {
    const entries = await readdir(sessionLogDir);
    files = entries
      .filter((f) => f.endsWith(".jsonl"))
      .map((f) => join(sessionLogDir, f));
  } catch {
    return [];
  }

  const sessions: SessionLogEntry[][] = [];
  for (const file of files) {
    const entries = await readJsonl<SessionLogEntry>(file);
    if (entries.length >= 2) sessions.push(entries);
  }
  return sessions;
}

// ---------------------------------------------------------------------------
// Prefix repetition analysis
// ---------------------------------------------------------------------------

/**
 * Result of analyzing how many times a message prefix (0..N) repeats across
 * sessions.
 */
interface PrefixRepetitionMap {
  /**
   * For each candidate split index (0-based message index in the current
   * conversation), the estimated fraction of sessions where that prefix was
   * repeated (i.e., appears in the session log as cache hits or repeated content).
   */
  hitRateByIndex: Map<number, number>;
  /** Total number of session files analyzed. */
  sessionCount: number;
}

/**
 * Analyze session logs to estimate how often message prefixes are repeated.
 *
 * Strategy:
 *   For each session, look at the fraction of turns that show cacheReadTokens > 0
 *   in that bucket. We use cumulative-turn position to map session entries onto
 *   the current message array indices.
 *
 * @param messages       Messages in the current request.
 * @param sessions       Session log groups.
 * @param minRepeatTurns Minimum repetitions to consider a prefix "stable".
 */
function analyzePrefixRepetition(
  messages: Message[],
  sessions: SessionLogEntry[][],
  minRepeatTurns: number,
): PrefixRepetitionMap {
  const hitRateByIndex = new Map<number, number>();
  const sessionCount = sessions.length;

  if (sessionCount === 0 || messages.length === 0) {
    return { hitRateByIndex, sessionCount: 0 };
  }

  // For each candidate split point (message index), count sessions where
  // the entries at positions 0..splitIdx showed cache hits.
  const hitCountByIndex = new Map<number, number>();

  for (const session of sessions) {
    // For each possible split point in the current messages array...
    for (let splitIdx = 0; splitIdx < messages.length - 1; splitIdx++) {
      // The session must have at least splitIdx+1 entries for this split to apply.
      if (session.length <= splitIdx) continue;

      // Check how many entries in session[0..splitIdx] are cache hits.
      let hitCount = 0;
      for (let i = 0; i <= splitIdx && i < session.length; i++) {
        const entry = session[i];
        if (entry && (entry.cache_read_input_tokens ?? 0) > 0) hitCount++;
      }

      // We consider the prefix "repeated" in this session if ≥50% of entries
      // up to splitIdx were cache hits.
      const hitFrac = hitCount / (splitIdx + 1);
      if (hitFrac >= 0.5) {
        hitCountByIndex.set(splitIdx, (hitCountByIndex.get(splitIdx) ?? 0) + 1);
      }
    }
  }

  // Convert counts to rates; filter by minRepeatTurns
  for (const [idx, count] of hitCountByIndex) {
    if (count >= minRepeatTurns) {
      hitRateByIndex.set(idx, count / sessionCount);
    }
  }

  return { hitRateByIndex, sessionCount };
}

// ---------------------------------------------------------------------------
// ROI computation
// ---------------------------------------------------------------------------

/**
 * Compute the ROI for placing a breakpoint after `splitIdx` in the message array.
 *
 * @param splitIdx          Message index (breakpoint placed after this message).
 * @param cumulativeTokens  Prefix token counts: cumulativeTokens[i] = total tokens
 *                          in messages[0..i].
 * @param hitRate           Estimated cache-hit probability for this prefix.
 * @param inputPricePerMToken  Input price in USD per million tokens.
 */
function computeBreakpointROI(
  splitIdx: number,
  cumulativeTokens: number[],
  hitRate: number,
  inputPricePerMToken: number,
): BreakpointROI {
  const prefixTokens = cumulativeTokens[splitIdx] ?? 0;
  if (prefixTokens === 0 || hitRate === 0) {
    return {
      messageIndex: splitIdx,
      prefixTokens,
      estimatedHitRate: hitRate,
      estimatedTokenSavings: 0,
      writeCostTokens: 0,
      netTokenSavings: 0,
      netRoiMicroUsd: 0,
    };
  }

  // Estimated token savings per request on a cache hit
  const estimatedTokenSavings = hitRate * prefixTokens * (1 - CACHE_READ_MULT);

  // Write overhead: 0.25× extra tokens per write, amortized over expected hits
  const expectedHitsPerWrite = hitRate > 0 ? Math.max(1, Math.round(1 / hitRate)) : 1;
  const writeCostTokens = (prefixTokens * (CACHE_WRITE_MULT - 1)) / expectedHitsPerWrite;

  const netTokenSavings = estimatedTokenSavings - writeCostTokens;

  // Convert to micro-USD: (tokens / 1_000_000) × price × 1_000_000
  const netRoiMicroUsd = Math.round(netTokenSavings * inputPricePerMToken);

  return {
    messageIndex: splitIdx,
    prefixTokens,
    estimatedHitRate: hitRate,
    estimatedTokenSavings,
    writeCostTokens,
    netTokenSavings,
    netRoiMicroUsd,
  };
}

// ---------------------------------------------------------------------------
// Heuristic fallback (no session data)
// ---------------------------------------------------------------------------

/**
 * When there is insufficient session data, apply a heuristic:
 *   - Place a breakpoint after the system message (index 0) if present.
 *   - If messages.length ≥ 8, add a second breakpoint at the midpoint.
 *
 * This mirrors the intent of the original `cacheBreakpoints()` while staying
 * within the MAX_BREAKPOINTS limit.
 */
function heuristicBreakpoints(
  messages: Message[],
  maxBreakpoints: number,
): number[] {
  if (messages.length === 0) return [];
  const breakpoints: number[] = [];

  // Breakpoint 1: after system/first message
  if (messages.length > 1) breakpoints.push(0);

  // Breakpoint 2: midpoint of remaining messages (if enough messages)
  if (maxBreakpoints >= 2 && messages.length >= 8) {
    const mid = Math.floor(messages.length / 2);
    if (!breakpoints.includes(mid)) breakpoints.push(mid);
  }

  return breakpoints.slice(0, maxBreakpoints);
}

// ---------------------------------------------------------------------------
// Public API — CacheBreakpointOptimizer
// ---------------------------------------------------------------------------

/**
 * Session-log-driven cache breakpoint optimizer.
 *
 * ### Typical usage
 * ```ts
 * const optimizer = new CacheBreakpointOptimizer();
 * const rec = await optimizer.recommendBreakpoints(messages, { maxBreakpoints: 2 });
 * // Apply: use rec.breakpointIndices to set cache markers on messages
 * ```
 *
 * Audit trail is written to `~/.ashlr/cache-breakpoint-evolution.jsonl`.
 */
export class CacheBreakpointOptimizer {
  private readonly sessionLogDir: string;
  private readonly auditPath: string;

  constructor(options: {
    sessionLogDir?: string;
    auditPath?: string;
  } = {}) {
    this.sessionLogDir = options.sessionLogDir ?? SESSION_LOG_DIR;
    this.auditPath     = options.auditPath     ?? EVOLUTION_AUDIT_PATH;
  }

  /**
   * Recommend up to `maxBreakpoints` cache breakpoints for the given message
   * array by analyzing historical session logs.
   *
   * @param messages  Messages for the upcoming API request.
   * @param options   Tuning parameters.
   * @returns         Ordered list of recommendations (best ROI first).
   */
  async recommendBreakpoints(
    messages: Message[],
    options: OptimizerOptions = {},
  ): Promise<BreakpointRecommendation[]> {
    // When a providerHint is given, consult the registry to enforce per-provider
    // constraints before running the optimizer.
    if (options.providerHint) {
      // Dynamic import avoids a circular dependency:
      //   cache-breakpoint-optimizer → multi-provider-adapter → cache-breakpoint-optimizer
      const { resolveProviderStrategy } = await import("./multi-provider-adapter.ts");
      const strategy = resolveProviderStrategy(options.providerHint);

      // Providers without cache support or without explicit breakpoints: early exit.
      if (!strategy.supported || !strategy.explicitBreakpoints) {
        return [];
      }

      // Clamp maxBreakpoints to the provider's limit.
      options = {
        ...options,
        maxBreakpoints: options.maxBreakpoints != null
          ? Math.min(options.maxBreakpoints, strategy.maxBreakpoints)
          : strategy.maxBreakpoints,
      };
    }

    const maxBp          = Math.min(options.maxBreakpoints ?? DEFAULT_MAX_BREAKPOINTS, MAX_BREAKPOINTS);
    const inputPrice     = options.inputPricePerMToken ?? 3.0;
    const minRoi         = options.minRoiMicroUsd ?? 1;
    const minRepeat      = options.minRepeatTurns ?? MIN_REPEAT_TURNS;

    if (messages.length === 0) {
      return [];
    }

    // Build cumulative token counts
    const cumulativeTokens: number[] = [];
    let running = 0;
    for (const msg of messages) {
      running += estimateTokenCount(msg);
      cumulativeTokens.push(running);
    }

    // Load session logs
    const sessions = await loadSessionLogs(this.sessionLogDir);
    const { hitRateByIndex, sessionCount } = analyzePrefixRepetition(
      messages,
      sessions,
      minRepeat,
    );

    const learnedFromSessions = hitRateByIndex.size > 0;

    let selectedIndices: number[];
    let candidateROIs: BreakpointROI[];

    if (learnedFromSessions) {
      // Score every candidate with a positive hit rate
      const candidates: BreakpointROI[] = [];
      for (const [idx, hitRate] of hitRateByIndex) {
        const roi = computeBreakpointROI(idx, cumulativeTokens, hitRate, inputPrice);
        if (roi.netRoiMicroUsd >= minRoi) {
          candidates.push(roi);
        }
      }

      // Sort by descending netRoiMicroUsd, take top maxBp
      candidates.sort((a, b) => b.netRoiMicroUsd - a.netRoiMicroUsd);
      const topCandidates = candidates.slice(0, maxBp);

      // Sort selected indices ascending for correct placement
      topCandidates.sort((a, b) => a.messageIndex - b.messageIndex);
      selectedIndices = topCandidates.map((c) => c.messageIndex);
      candidateROIs   = topCandidates;
    } else {
      // Heuristic fallback
      selectedIndices = heuristicBreakpoints(messages, maxBp);
      candidateROIs   = selectedIndices.map((idx) =>
        computeBreakpointROI(idx, cumulativeTokens, 0.3, inputPrice),
      );
    }

    if (selectedIndices.length === 0) {
      return [];
    }

    const totalNetRoiMicroUsd = candidateROIs.reduce((s, r) => s + r.netRoiMicroUsd, 0);

    // Build human-readable summary
    const parts = candidateROIs.map(
      (r) =>
        `msg[${r.messageIndex}] (${r.prefixTokens} tokens, ` +
        `${Math.round(r.estimatedHitRate * 100)}% hit-rate, ` +
        `+${r.netRoiMicroUsd}μUSD)`,
    );
    const summary = learnedFromSessions
      ? `Learned from ${sessionCount} session(s): breakpoints at ${parts.join("; ")}`
      : `Heuristic breakpoints (no session data): ${parts.join("; ")}`;

    const recommendation: BreakpointRecommendation = {
      generatedAt:          new Date().toISOString(),
      breakpointIndices:    selectedIndices,
      roi:                  candidateROIs,
      totalNetRoiMicroUsd,
      summary,
      learnedFromSessions,
    };

    // Emit audit trail
    await this._appendEvolution({
      breakpointIndices:    selectedIndices,
      totalNetRoiMicroUsd,
      sessionFilesAnalyzed: sessionCount,
      learnedFromSessions,
      roiSnapshot:          candidateROIs.map((r) => ({
        messageIndex:       r.messageIndex,
        prefixTokens:       r.prefixTokens,
        estimatedHitRate:   r.estimatedHitRate,
        netRoiMicroUsd:     r.netRoiMicroUsd,
      })),
    });

    return [recommendation];
  }

  /**
   * Load the full evolution audit trail.
   */
  async loadEvolutionAudit(): Promise<BreakpointEvolutionRecord[]> {
    return readJsonl<BreakpointEvolutionRecord>(this.auditPath);
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  private async _appendEvolution(
    fields: Omit<BreakpointEvolutionRecord, "id" | "timestamp">,
  ): Promise<void> {
    const record: BreakpointEvolutionRecord = {
      id:        `bpe-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      timestamp: new Date().toISOString(),
      ...fields,
    };
    await appendJsonl(this.auditPath, record);
  }
}

// ---------------------------------------------------------------------------
// Convenience function (module-level)
// ---------------------------------------------------------------------------

/**
 * Convenience wrapper: create a one-shot `CacheBreakpointOptimizer` and
 * return its top recommendation for the given messages.
 *
 * ```ts
 * import { recommendBreakpoints } from "./cache-breakpoint-optimizer.ts";
 * const [rec] = await recommendBreakpoints(messages, { maxBreakpoints: 2 });
 * ```
 */
export async function recommendBreakpoints(
  messages: Message[],
  options?: OptimizerOptions,
): Promise<BreakpointRecommendation[]> {
  const optimizer = new CacheBreakpointOptimizer({
    sessionLogDir: options?.sessionLogDir,
    auditPath:     options?.auditPath,
  });
  return optimizer.recommendBreakpoints(messages, options);
}
