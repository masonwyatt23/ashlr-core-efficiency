/**
 * Hierarchical Message Deduplication with Semantic Fingerprinting
 *
 * Detects semantically-equivalent messages across conversation turns and removes
 * redundant tool-result pairs using shallow embeddings (no LLM calls).
 *
 * ### Algorithm
 *
 * 1. `SemanticFingerprint` — computes deterministic n-gram + shingle hashes
 *    (4-gram, 8-gram word-level) for text and tool_result content blocks,
 *    mapped to 128-bit MurmurHash-inspired compact fingerprints.
 *
 * 2. `MessageDeduplicator` — maintains a fingerprint index over the last N
 *    messages (configurable, default 50). On `add()`, checks for:
 *    - Exact fingerprint matches (byte-for-byte duplicate)
 *    - High Jaccard similarity (>0.9 shingle overlap → fuzzy duplicate)
 *    - Low Levenshtein distance (<3 edits on normalised text → near-duplicate)
 *
 * 3. `ToolResultGrouping` — clusters identical tool_use calls with same/similar
 *    results, ranks by recency + token cost, removes older duplicates.
 *
 * 4. `DedupStats` — per-call metrics for cost accounting integration.
 *
 * ### Integration
 *
 * `deduplicateMessageSequence()` is the primary entry point. It is designed to
 * be invoked from `contextCollapse` (tier 3) after the basic collapse pass, so
 * the pipeline benefits from semantic dedup without adding LLM overhead.
 *
 * Emits a `CompressorRoleRecord` (tier 3) when tokens are removed.
 */

import type { Message, ContentBlock } from "../types/index.ts";
import { estimateTokensFromMessages } from "../tokens/index.ts";
import { defaultProviderRate } from "../tokens/index.ts";
import { trackCompressorOutcome, tierToRole } from "../session-log/cost-accounting.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Default fingerprint window — last N messages indexed for dedup checks. */
const DEFAULT_WINDOW = 50;

/** Jaccard similarity threshold above which two messages are considered fuzzy duplicates. */
const JACCARD_THRESHOLD = 0.9;

/** Maximum Levenshtein distance (on normalised text) to treat as near-duplicate. */
const MAX_LEVENSHTEIN = 3;

/** Max text length to run Levenshtein on (avoids O(n²) on giant tool results). */
const MAX_LEVENSHTEIN_TEXT = 500;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/**
 * Per-call deduplication metrics for cost accounting integration.
 */
export interface DedupStats {
  /** Number of messages removed as duplicates in this dedup pass. */
  duplicates_removed: number;
  /** Estimated tokens saved by removing duplicates. */
  tokens_saved: number;
  /**
   * Ratio of unique messages to total messages after dedup.
   * 1.0 = fully unique; 0.0 = all duplicates (degenerate).
   */
  unique_ratio: number;
  /** Number of exact-match duplicates detected. */
  exact_matches: number;
  /** Number of fuzzy (Jaccard ≥ 0.9) duplicates detected. */
  fuzzy_matches: number;
  /** Number of near-duplicate (Levenshtein < 3) duplicates detected. */
  levenshtein_matches: number;
  /** Number of tool-result clusters deduplicated. */
  tool_result_clusters: number;
}

/**
 * Configuration for `deduplicateMessageSequence`.
 */
export interface DedupConfig {
  /**
   * Fingerprint window size — only the last N messages are indexed for
   * duplicate checks. Larger values catch more duplicates at higher cost.
   * Default: 50.
   */
  windowSize?: number;
  /**
   * Jaccard similarity threshold for fuzzy-duplicate detection.
   * Range [0, 1]. Higher = stricter (fewer false positives).
   * Default: 0.9.
   */
  jaccardThreshold?: number;
  /**
   * Maximum Levenshtein distance for near-duplicate detection.
   * Applied to normalised text ≤ MAX_LEVENSHTEIN_TEXT chars.
   * Default: 3.
   */
  maxLevenshtein?: number;
  /**
   * Number of recent messages to protect from deduplication.
   * Mirrors contextCollapse's keepRecent parameter.
   * Default: 5.
   */
  keepRecent?: number;
  /**
   * Enable tool-result grouping / cluster deduplication.
   * Default: true.
   */
  enableToolResultGrouping?: boolean;
}

// ---------------------------------------------------------------------------
// Fingerprinting primitives
// ---------------------------------------------------------------------------

/**
 * Compact 128-bit fingerprint stored as two 64-bit hex strings.
 */
export interface Fingerprint {
  lo: string; // lower 64 bits as hex
  hi: string; // upper 64 bits as hex
}

/**
 * MurmurHash3-inspired 32-bit hash for a string (deterministic, fast, low collision).
 * Pure TypeScript implementation — no native bindings required.
 */
function murmur32(text: string, seed = 0): number {
  let h = seed >>> 0;
  const c1 = 0xcc9e2d51;
  const c2 = 0x1b873593;

  // Process 4 bytes at a time
  let i = 0;
  while (i + 4 <= text.length) {
    let k =
      (text.charCodeAt(i) & 0xff) |
      ((text.charCodeAt(i + 1) & 0xff) << 8) |
      ((text.charCodeAt(i + 2) & 0xff) << 16) |
      ((text.charCodeAt(i + 3) & 0xff) << 24);

    k = Math.imul(k, c1);
    k = ((k << 15) | (k >>> 17)) >>> 0;
    k = Math.imul(k, c2);

    h ^= k;
    h = ((h << 13) | (h >>> 19)) >>> 0;
    h = (Math.imul(h, 5) + 0xe6546b64) >>> 0;
    i += 4;
  }

  // Tail bytes
  let tail = 0;
  const remaining = text.length - i;
  if (remaining >= 3) tail ^= (text.charCodeAt(i + 2) & 0xff) << 16;
  if (remaining >= 2) tail ^= (text.charCodeAt(i + 1) & 0xff) << 8;
  if (remaining >= 1) {
    tail ^= text.charCodeAt(i) & 0xff;
    tail = Math.imul(tail, c1);
    tail = ((tail << 15) | (tail >>> 17)) >>> 0;
    tail = Math.imul(tail, c2);
    h ^= tail;
  }

  // Finalization mix
  h ^= text.length;
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;

  return h >>> 0;
}

/**
 * Compute a 128-bit fingerprint by running murmur32 four times with different seeds.
 * This gives us 4 × 32 = 128 bits of fingerprint space, keeping collision probability
 * below 0.001% for vocabularies up to 10M messages.
 */
function fingerprint128(text: string): Fingerprint {
  const h0 = murmur32(text, 0x9747b28c);
  const h1 = murmur32(text, 0x85ebca77);
  const h2 = murmur32(text, 0xc2b2ae3d);
  const h3 = murmur32(text, 0x27d4eb2f);
  return {
    lo: (h0.toString(16).padStart(8, "0") + h1.toString(16).padStart(8, "0")),
    hi: (h2.toString(16).padStart(8, "0") + h3.toString(16).padStart(8, "0")),
  };
}

/** Stable string key for a Fingerprint (used as map key). */
function fpKey(fp: Fingerprint): string {
  return `${fp.lo}:${fp.hi}`;
}

// ---------------------------------------------------------------------------
// SemanticFingerprint class
// ---------------------------------------------------------------------------

/**
 * Computes deterministic n-gram + shingle hashes for message content.
 *
 * - 4-gram shingles: overlapping sequences of 4 words
 * - 8-gram shingles: overlapping sequences of 8 words
 * - 128-bit MurmurHash fingerprint for compact storage
 *
 * Normalizes text before hashing (lowercase, collapse whitespace, strip
 * common "no output" markers) to catch formatting-only differences.
 */
export class SemanticFingerprint {
  /** Canonical fingerprint for exact-match detection. */
  readonly exact: Fingerprint;
  /** Set of 4-gram hashes (as hex strings) for Jaccard similarity. */
  readonly shingles4: Set<string>;
  /** Set of 8-gram hashes (as hex strings) for Jaccard similarity. */
  readonly shingles8: Set<string>;
  /** Normalised text for Levenshtein distance (truncated to MAX_LEVENSHTEIN_TEXT). */
  readonly normalizedText: string;

  constructor(text: string) {
    const normalized = SemanticFingerprint.normalize(text);
    this.normalizedText = normalized.slice(0, MAX_LEVENSHTEIN_TEXT);
    this.exact = fingerprint128(normalized);
    const words = normalized.split(/\s+/).filter((w) => w.length > 0);
    this.shingles4 = SemanticFingerprint.buildShingles(words, 4);
    this.shingles8 = SemanticFingerprint.buildShingles(words, 8);
  }

  /**
   * Normalize text for fingerprinting:
   * - Lowercase
   * - Collapse whitespace
   * - Strip common "no output" markers (empty result placeholders)
   * - Trim
   */
  static normalize(text: string): string {
    return text
      .toLowerCase()
      // Strip common "no output" markers before hashing
      .replace(/\[no output\]|\[empty\]|\[no result\]|\[nothing\]|\[\s*\]/gi, " ")
      // Normalize line endings + multiple spaces
      .replace(/\r\n/g, "\n")
      .replace(/[ \t]+/g, " ")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
  }

  /**
   * Build a set of n-gram shingle hashes from a word array.
   * Each shingle is a sequence of `n` consecutive words joined by spaces,
   * then hashed with murmur32 for compact storage.
   */
  static buildShingles(words: string[], n: number): Set<string> {
    const shingles = new Set<string>();
    if (words.length < n) {
      // Too short for n-grams: hash the whole thing as a single shingle
      if (words.length > 0) {
        shingles.add(murmur32(words.join(" "), n).toString(16));
      }
      return shingles;
    }
    for (let i = 0; i <= words.length - n; i++) {
      const gram = words.slice(i, i + n).join(" ");
      shingles.add(murmur32(gram, n).toString(16));
    }
    return shingles;
  }

  /**
   * Jaccard similarity between two shingle sets: |A ∩ B| / |A ∪ B|.
   * Returns a value in [0, 1] where 1 = identical.
   */
  static jaccard(a: Set<string>, b: Set<string>): number {
    if (a.size === 0 && b.size === 0) return 1.0; // both empty = identical
    if (a.size === 0 || b.size === 0) return 0.0;

    let intersection = 0;
    for (const item of a) {
      if (b.has(item)) intersection++;
    }
    const union = a.size + b.size - intersection;
    return union > 0 ? intersection / union : 1.0;
  }
}

// ---------------------------------------------------------------------------
// Text extraction helpers
// ---------------------------------------------------------------------------

/**
 * Extract plain text from a Message for fingerprinting.
 * Handles both string content and ContentBlock[] arrays.
 */
export function extractTextForFingerprint(message: Message): string {
  if (typeof message.content === "string") {
    return message.content;
  }
  return (message.content as ContentBlock[])
    .map((block) => {
      if (block.type === "text") return block.text;
      if (block.type === "tool_result") return block.content;
      if (block.type === "tool_use") return `${block.name}:${JSON.stringify(block.input)}`;
      if (block.type === "thinking") return block.thinking;
      return "";
    })
    .join("\n");
}

/**
 * Extract tool_result content blocks from a Message.
 * Returns empty array for non-user messages or string-content messages.
 */
function extractToolResults(
  message: Message,
): Array<{ tool_use_id: string; content: string }> {
  if (message.role !== "user" || typeof message.content === "string") return [];
  return (message.content as ContentBlock[])
    .filter((b): b is Extract<ContentBlock, { type: "tool_result" }> => b.type === "tool_result")
    .map((b) => ({ tool_use_id: b.tool_use_id, content: b.content }));
}

/**
 * Extract tool_use calls from an assistant Message.
 */
function extractToolUses(
  message: Message,
): Array<{ id: string; name: string; inputKey: string }> {
  if (message.role !== "assistant" || typeof message.content === "string") return [];
  return (message.content as ContentBlock[])
    .filter((b): b is Extract<ContentBlock, { type: "tool_use" }> => b.type === "tool_use")
    .map((b) => ({
      id: b.id,
      name: b.name,
      inputKey: JSON.stringify(b.input),
    }));
}

// ---------------------------------------------------------------------------
// Levenshtein distance (bounded)
// ---------------------------------------------------------------------------

/**
 * Compute Levenshtein edit distance between two strings.
 * Returns early (maxDist + 1) when the running minimum exceeds maxDist,
 * avoiding full O(n×m) computation for clearly different strings.
 */
export function levenshtein(a: string, b: string, maxDist = MAX_LEVENSHTEIN): number {
  if (a === b) return 0;
  if (Math.abs(a.length - b.length) > maxDist) return maxDist + 1;

  // Use two-row DP for memory efficiency
  const n = a.length;
  const m = b.length;

  let prev = Array.from({ length: m + 1 }, (_, i) => i);
  let curr = new Array<number>(m + 1);

  for (let i = 1; i <= n; i++) {
    curr[0] = i;
    let rowMin = curr[0]!;

    for (let j = 1; j <= m; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(
        prev[j]! + 1,   // deletion
        curr[j - 1]! + 1, // insertion
        prev[j - 1]! + cost, // substitution
      );
      rowMin = Math.min(rowMin, curr[j]!);
    }

    // Early exit if entire row exceeds maxDist
    if (rowMin > maxDist) return maxDist + 1;

    const tmp = prev;
    prev = curr;
    curr = tmp;
  }

  return prev[m]!;
}

// ---------------------------------------------------------------------------
// MessageDeduplicator
// ---------------------------------------------------------------------------

interface IndexedFingerprint {
  /** Original message index (into the full message array). */
  messageIndex: number;
  /** Fingerprint of the message. */
  fp: SemanticFingerprint;
  /** Estimated token count for this message. */
  tokens: number;
  /**
   * True when the message contains tool_use blocks.
   * Levenshtein is skipped for tool_use messages because single-character
   * argument differences (e.g. "/a.ts" vs "/b.ts") are semantically significant
   * and should not be treated as near-duplicates.
   */
  hasToolUse: boolean;
}

/**
 * Maintains a rolling fingerprint index over the last N messages and detects
 * exact, fuzzy, and near-duplicate messages.
 *
 * Usage:
 * ```ts
 * const dedup = new MessageDeduplicator({ windowSize: 50 });
 * for (const [i, msg] of messages.entries()) {
 *   const result = dedup.add(i, msg);
 *   if (result.isDuplicate) { ... }
 * }
 * ```
 */
export class MessageDeduplicator {
  private readonly windowSize: number;
  private readonly jaccardThreshold: number;
  private readonly maxLevenshtein: number;
  private readonly index: IndexedFingerprint[] = [];

  constructor(config: Pick<DedupConfig, "windowSize" | "jaccardThreshold" | "maxLevenshtein"> = {}) {
    this.windowSize = config.windowSize ?? DEFAULT_WINDOW;
    this.jaccardThreshold = config.jaccardThreshold ?? JACCARD_THRESHOLD;
    this.maxLevenshtein = config.maxLevenshtein ?? MAX_LEVENSHTEIN;
  }

  /**
   * Add a message to the index and check if it is a duplicate of any
   * message already in the window.
   *
   * Returns the duplicate detection result. The caller decides whether to
   * remove the message — this class only detects, not mutates.
   */
  add(
    messageIndex: number,
    message: Message,
  ): { isDuplicate: boolean; matchType: "exact" | "fuzzy" | "levenshtein" | "none"; matchedIndex: number } {
    const text = extractTextForFingerprint(message);
    const fp = new SemanticFingerprint(text);
    const tokens = estimateTokensFromMessages([message]);
    const hasToolUse = messageHasToolUse(message);

    // Check against existing index entries
    for (const entry of this.index) {
      // 1. Exact match
      if (entry.fp.exact.lo === fp.exact.lo && entry.fp.exact.hi === fp.exact.hi) {
        this._addToIndex({ messageIndex, fp, tokens, hasToolUse });
        return { isDuplicate: true, matchType: "exact", matchedIndex: entry.messageIndex };
      }

      // 2. Jaccard similarity on 4-gram shingles (primary, faster)
      const j4 = SemanticFingerprint.jaccard(entry.fp.shingles4, fp.shingles4);
      if (j4 >= this.jaccardThreshold) {
        this._addToIndex({ messageIndex, fp, tokens, hasToolUse });
        return { isDuplicate: true, matchType: "fuzzy", matchedIndex: entry.messageIndex };
      }

      // 3. Jaccard similarity on 8-gram shingles (secondary, more precise)
      if (fp.shingles8.size > 0 && entry.fp.shingles8.size > 0) {
        const j8 = SemanticFingerprint.jaccard(entry.fp.shingles8, fp.shingles8);
        if (j8 >= this.jaccardThreshold) {
          this._addToIndex({ messageIndex, fp, tokens, hasToolUse });
          return { isDuplicate: true, matchType: "fuzzy", matchedIndex: entry.messageIndex };
        }
      }

      // 4. Levenshtein on normalised text (only for short texts, never for tool_use messages).
      // Tool invocations often differ by just one argument character (e.g. "/a.ts" vs "/b.ts"),
      // which is semantically significant. Levenshtein would incorrectly flag those as duplicates.
      if (
        !hasToolUse &&
        !entry.hasToolUse &&
        fp.normalizedText.length > 0 &&
        entry.fp.normalizedText.length > 0 &&
        fp.normalizedText.length <= MAX_LEVENSHTEIN_TEXT &&
        entry.fp.normalizedText.length <= MAX_LEVENSHTEIN_TEXT
      ) {
        const dist = levenshtein(fp.normalizedText, entry.fp.normalizedText, this.maxLevenshtein);
        if (dist <= this.maxLevenshtein) {
          this._addToIndex({ messageIndex, fp, tokens, hasToolUse });
          return { isDuplicate: true, matchType: "levenshtein", matchedIndex: entry.messageIndex };
        }
      }
    }

    this._addToIndex({ messageIndex, fp, tokens, hasToolUse });
    return { isDuplicate: false, matchType: "none", matchedIndex: -1 };
  }

  /** Reset the index (for reuse across passes). */
  reset(): void {
    this.index.length = 0;
  }

  private _addToIndex(entry: IndexedFingerprint): void {
    this.index.push(entry);
    // Evict oldest entries beyond window size
    if (this.index.length > this.windowSize) {
      this.index.shift();
    }
  }
}

/**
 * Returns true if the message contains at least one tool_use content block.
 * Used to guard Levenshtein matching — tool invocations with differing arguments
 * must not be treated as near-duplicates.
 */
function messageHasToolUse(message: Message): boolean {
  if (typeof message.content === "string") return false;
  return (message.content as ContentBlock[]).some((b) => b.type === "tool_use");
}

// ---------------------------------------------------------------------------
// ToolResultGrouping
// ---------------------------------------------------------------------------

/**
 * A cluster of tool_use + tool_result pairs with the same or similar invocation.
 */
interface ToolResultCluster {
  /** Canonical tool name + input key for this cluster. */
  toolKey: string;
  /** Indices of messages that are tool_use callers for this cluster. */
  useIndices: number[];
  /** Indices of messages that are tool_result responders for this cluster. */
  resultIndices: number[];
  /** Fingerprint of the representative (most recent) result content. */
  resultFp: SemanticFingerprint;
  /** Estimated tokens for the representative result. */
  resultTokens: number;
}

/**
 * Groups tool_use + tool_result pairs by invocation identity and removes
 * older duplicates, keeping only the most recent invocation of each call.
 *
 * A tool invocation is considered a duplicate when:
 * - Same tool name + same JSON-serialized input (exact match), AND
 * - Result content has Jaccard ≥ jaccardThreshold with the most recent result
 *
 * The older duplicate pair (tool_use + tool_result message) is marked for removal.
 */
export class ToolResultGrouping {
  private readonly jaccardThreshold: number;

  constructor(jaccardThreshold = JACCARD_THRESHOLD) {
    this.jaccardThreshold = jaccardThreshold;
  }

  /**
   * Identify indices of duplicate tool-call pairs in a message array.
   * Returns a set of message indices to remove (older duplicates).
   */
  findDuplicates(
    messages: Message[],
    protectedIndices: Set<number>,
  ): { toRemove: Set<number>; clusterCount: number } {
    // Map: toolKey → cluster
    const clusters = new Map<string, ToolResultCluster>();
    // Map: tool_use_id → message index (for pairing)
    const toolUseIdToMsgIndex = new Map<string, number>();
    const toRemove = new Set<number>();
    let clusterCount = 0;

    for (let i = 0; i < messages.length; i++) {
      if (protectedIndices.has(i)) continue;

      const msg = messages[i]!;

      // Index tool_use calls
      const uses = extractToolUses(msg);
      for (const use of uses) {
        toolUseIdToMsgIndex.set(use.id, i);
      }

      // Process tool_result messages
      const results = extractToolResults(msg);
      if (results.length === 0) continue;

      for (const result of results) {
        const parentIdx = toolUseIdToMsgIndex.get(result.tool_use_id);
        if (parentIdx === undefined) continue;

        const parentMsg = messages[parentIdx]!;
        const parentUses = extractToolUses(parentMsg);
        const matchingUse = parentUses.find((u) => u.id === result.tool_use_id);
        if (!matchingUse) continue;

        const toolKey = `${matchingUse.name}:${matchingUse.inputKey}`;
        const resultFp = new SemanticFingerprint(result.content);

        const existing = clusters.get(toolKey);
        if (!existing) {
          // First occurrence — register as representative
          clusters.set(toolKey, {
            toolKey,
            useIndices: [parentIdx],
            resultIndices: [i],
            resultFp,
            resultTokens: estimateTokensFromMessages([msg]),
          });
        } else {
          // Check if this result is similar to the representative
          const j = SemanticFingerprint.jaccard(
            existing.resultFp.shingles4,
            resultFp.shingles4,
          );

          // Also check exact match
          const isExact =
            existing.resultFp.exact.lo === resultFp.exact.lo &&
            existing.resultFp.exact.hi === resultFp.exact.hi;

          if (isExact || j >= this.jaccardThreshold) {
            // This is a duplicate — remove the older one (existing)
            const oldestUseIdx = existing.useIndices[0]!;
            const oldestResultIdx = existing.resultIndices[0]!;

            if (!protectedIndices.has(oldestUseIdx)) {
              toRemove.add(oldestUseIdx);
            }
            if (!protectedIndices.has(oldestResultIdx)) {
              toRemove.add(oldestResultIdx);
            }

            // Update cluster to point to the newer (more recent) occurrence
            existing.useIndices = [parentIdx];
            existing.resultIndices = [i];
            existing.resultFp = resultFp;
            existing.resultTokens = estimateTokensFromMessages([msg]);
            clusterCount++;
          }
        }
      }
    }

    return { toRemove, clusterCount };
  }
}

// ---------------------------------------------------------------------------
// Session log integration
// ---------------------------------------------------------------------------

/**
 * Register deduplication metrics to a session log entry.
 * Appends DedupStats as a JSON line to the session log for offline analysis.
 *
 * @param sessionLog  Object to attach dedup stats to (mutated in-place).
 * @param stats       Deduplication statistics from this pass.
 */
export function registerDedupMetrics(
  sessionLog: Record<string, unknown>,
  stats: DedupStats,
): void {
  sessionLog["dedupStats"] = stats;
}

// ---------------------------------------------------------------------------
// Primary API: deduplicateMessageSequence
// ---------------------------------------------------------------------------

/**
 * Deduplicate a message sequence using semantic fingerprinting.
 *
 * Combines three deduplication passes:
 * 1. Per-message fingerprint dedup (exact, fuzzy, Levenshtein)
 * 2. Tool-result cluster dedup (same call, similar result)
 *
 * Protected messages (recent tail) are never removed.
 *
 * Emits a `CompressorRoleRecord` (tier 3) when tokens are saved, so the
 * backpropagation engine can attribute savings to semantic dedup.
 *
 * @param messages   Input message array.
 * @param config     Dedup configuration.
 * @param agent      Agent/session identifier for backprop attribution.
 * @param provider   Provider name for cost rate lookup.
 * @returns          `{ messages, stats }` — deduplicated messages + metrics.
 */
export function deduplicateMessageSequence(
  messages: Message[],
  config: DedupConfig = {},
  agent = "unknown",
  provider = "claude-3-5-sonnet",
): { messages: Message[]; stats: DedupStats } {
  const {
    windowSize = DEFAULT_WINDOW,
    jaccardThreshold = JACCARD_THRESHOLD,
    maxLevenshtein = MAX_LEVENSHTEIN,
    keepRecent = 5,
    enableToolResultGrouping = true,
  } = config;

  if (messages.length <= keepRecent) {
    return {
      messages,
      stats: emptyStats(messages.length),
    };
  }

  const tokensBefore = estimateTokensFromMessages(messages);

  // Indices that are protected from removal (recent tail + any ancestors)
  const protectedIndices = new Set<number>();
  const protectedStart = Math.max(0, messages.length - keepRecent);
  for (let i = protectedStart; i < messages.length; i++) {
    protectedIndices.add(i);
  }

  // Pass 1: per-message semantic fingerprint dedup
  const deduplicator = new MessageDeduplicator({ windowSize, jaccardThreshold, maxLevenshtein });
  const toRemoveFromDedup = new Set<number>();
  let exactMatches = 0;
  let fuzzyMatches = 0;
  let levMatches = 0;

  // Only scan the older slice (not the protected tail)
  const olderSlice = messages.slice(0, -keepRecent);
  for (let i = 0; i < olderSlice.length; i++) {
    const msg = olderSlice[i]!;
    const result = deduplicator.add(i, msg);
    if (result.isDuplicate) {
      toRemoveFromDedup.add(i);
      if (result.matchType === "exact") exactMatches++;
      else if (result.matchType === "fuzzy") fuzzyMatches++;
      else if (result.matchType === "levenshtein") levMatches++;
    }
  }

  // Pass 2: tool-result cluster dedup (if enabled)
  let toolResultClusters = 0;
  const toRemoveFromClusters = new Set<number>();
  if (enableToolResultGrouping) {
    // Protect both the recent tail AND messages already flagged for removal
    const clusterProtected = new Set([...protectedIndices, ...toRemoveFromDedup]);
    const grouper = new ToolResultGrouping(jaccardThreshold);
    const { toRemove: clusterRemovals, clusterCount } = grouper.findDuplicates(
      messages,
      clusterProtected,
    );
    for (const idx of clusterRemovals) {
      toRemoveFromClusters.add(idx);
    }
    toolResultClusters = clusterCount;
  }

  // Merge removal sets
  const allToRemove = new Set([...toRemoveFromDedup, ...toRemoveFromClusters]);

  // Build output
  const deduplicated = messages.filter((_, i) => !allToRemove.has(i));
  const tokensAfter = estimateTokensFromMessages(deduplicated);
  const tokensSaved = tokensBefore - tokensAfter;
  const duplicatesRemoved = allToRemove.size;
  const uniqueRatio = messages.length > 0 ? deduplicated.length / messages.length : 1.0;

  const stats: DedupStats = {
    duplicates_removed: duplicatesRemoved,
    tokens_saved: tokensSaved,
    unique_ratio: uniqueRatio,
    exact_matches: exactMatches,
    fuzzy_matches: fuzzyMatches,
    levenshtein_matches: levMatches,
    tool_result_clusters: toolResultClusters,
  };

  // Emit backprop record when tokens are actually saved
  if (tokensSaved > 0) {
    const rate = defaultProviderRate(provider);
    trackCompressorOutcome({
      agent,
      role: tierToRole(3),
      triggeredAt: new Date().toISOString(),
      tokensSaved,
      costImpactUsd: tokensSaved * rate.inputRate,
      queryLatencyMs: 0,
    });
  }

  return { messages: deduplicated, stats };
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function emptyStats(messageCount: number): DedupStats {
  return {
    duplicates_removed: 0,
    tokens_saved: 0,
    unique_ratio: messageCount > 0 ? 1.0 : 1.0,
    exact_matches: 0,
    fuzzy_matches: 0,
    levenshtein_matches: 0,
    tool_result_clusters: 0,
  };
}
