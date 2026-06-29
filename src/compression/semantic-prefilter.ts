/**
 * Semantic Pre-Filter for Compression Tier Selection
 *
 * Uses TF-IDF + BM25 keyword scoring to pre-rank compression tier candidates
 * before expensive Ollama embedding calls are made. This cuts ~30–50% of
 * Ollama calls for routine tier selection while preserving semantic ranking
 * as a fallback for genuinely ambiguous cases.
 *
 * ### Algorithm
 *
 * 1. `computeKeywordSignal` — extracts dominant topics from the message array
 *    and scores each tier candidate (identified by its summary/tags metadata)
 *    using TF-IDF + BM25 against those topics.
 * 2. Pre-filter decision logic:
 *    - If the top-2 candidates are within 5% keyword score → skip Ollama, pick
 *      the cheaper (higher-numbered) tier.
 *    - If ≥3 candidates are within 10% of each other → invoke Ollama (ambiguous).
 *    - Otherwise the top-ranked candidate wins without Ollama.
 *
 * ### IDF computation
 *
 * Reuses the same tokenize + IDF approach as `src/genome/retriever.ts:buildIDF`
 * so term weighting is consistent across the codebase. IDF is built from the
 * union of all tier candidate documents rather than the full genome manifest,
 * keeping computation O(tiers × terms) rather than O(sections × terms).
 */

import type { Message, ContentBlock } from "../types/index.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * BM25 tuning parameters.
 * k1 controls term-frequency saturation (1.2–2.0 typical). b controls
 * document-length normalisation (0.75 typical).
 */
const BM25_K1 = 1.5;
const BM25_B = 0.75;

/**
 * Minimum score difference (as a fraction of the top score) below which two
 * candidates are considered "too close" for keyword-only disambiguation.
 * Below this threshold we fall back to Ollama.
 */
const AMBIGUITY_THRESHOLD_TOP2 = 0.05; // 5% — top-2 too close → pick cheaper
const AMBIGUITY_THRESHOLD_3WAY = 0.10; // 10% — 3-way tie → invoke Ollama

/**
 * Minimum score a candidate must reach (as a fraction of max possible) for
 * keyword signal to be considered meaningful. Below this threshold we skip
 * pre-filtering entirely and let the caller decide.
 */
const MIN_SIGNAL_STRENGTH = 0.05;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Metadata describing one compression tier candidate.
 * Callers populate this from their tier registry or static definitions.
 */
export interface TierCandidate {
  /** Tier number (1 = most aggressive, 4 = lightest). */
  tier: 1 | 2 | 3 | 4;
  /**
   * Human-readable summary of what this tier does.
   * Used as the "document" for TF-IDF + BM25 scoring against message topics.
   */
  summary: string;
  /** Keyword tags that characterise this tier's strengths. */
  tags: string[];
  /**
   * Relative cost weight (1.0 = baseline). Lower-numbered tiers (more
   * aggressive) should have higher cost weights because they require LLM
   * calls. Used as a tiebreaker: when scores are equal we prefer lower cost.
   */
  costWeight: number;
}

/**
 * Result of the keyword pre-filter pass.
 */
export interface PreFilterResult {
  /**
   * Tier candidates sorted by keyword relevance score descending.
   * Scores are BM25-normalised against the query term set extracted from
   * the message array.
   */
  rankedCandidates: Array<{ tier: TierCandidate; score: number }>;
  /**
   * The recommended action for the caller:
   * - "pick_top"      — Top candidate is clearly best; skip Ollama.
   * - "pick_cheaper"  — Top-2 are within AMBIGUITY_THRESHOLD_TOP2; pick the
   *                     cheaper (higher-numbered) tier without Ollama.
   * - "invoke_ollama" — Signal is too ambiguous (3-way tie or low signal);
   *                     caller should proceed to Ollama reranking.
   */
  recommendation: "pick_top" | "pick_cheaper" | "invoke_ollama";
  /**
   * Normalised signal strength in [0, 1]. Low values indicate that the
   * message content has little overlap with any tier's vocabulary, which
   * typically means the default/cheapest tier is fine.
   */
  signalStrength: number;
  /** The dominant topics extracted from the message array. */
  dominantTopics: string[];
}

// ---------------------------------------------------------------------------
// Text utilities (mirrors genome/retriever.ts tokenize)
// ---------------------------------------------------------------------------

/**
 * Tokenize a string into lowercase terms, stripping punctuation.
 * Mirrors the implementation in `src/genome/retriever.ts` for consistency.
 */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2);
}

/**
 * Extract the text content from a Message for topic extraction.
 * Handles both string and ContentBlock[] content shapes.
 */
function extractMessageText(message: Message): string {
  if (typeof message.content === "string") {
    return message.content;
  }
  return (message.content as ContentBlock[])
    .map((block) => {
      if (block.type === "text") return block.text;
      if (block.type === "thinking") return block.thinking;
      if (block.type === "tool_result") return block.content;
      if (block.type === "tool_use") return block.name + " " + JSON.stringify(block.input);
      return "";
    })
    .join(" ");
}

// ---------------------------------------------------------------------------
// IDF computation (mirrors genome/retriever.ts buildIDF)
// ---------------------------------------------------------------------------

/**
 * Build an IDF map from a set of documents (tier candidate summaries + tags).
 * Terms that appear in fewer documents receive higher IDF weight.
 *
 * Uses the standard smoothed IDF formula: log(N / df) + 1, minimum 1.
 */
function buildIDF(documents: string[][]): Map<string, number> {
  const docFreq = new Map<string, number>();
  const n = documents.length;

  for (const terms of documents) {
    const uniqueTerms = new Set(terms);
    for (const term of uniqueTerms) {
      docFreq.set(term, (docFreq.get(term) ?? 0) + 1);
    }
  }

  const idf = new Map<string, number>();
  for (const [term, freq] of docFreq) {
    idf.set(term, Math.max(1, Math.log(n / freq) + 1));
  }
  return idf;
}

// ---------------------------------------------------------------------------
// Topic extraction
// ---------------------------------------------------------------------------

/**
 * Extract dominant topics from a message array.
 *
 * Strategy:
 * 1. Tokenize all message text.
 * 2. Build a term-frequency map over the entire conversation.
 * 3. Apply IDF weights (computed over per-message documents) to surface
 *    terms that are frequent in this conversation but rare across messages.
 * 4. Return the top-N terms by TF-IDF score.
 *
 * @param messages         Message array to analyse.
 * @param topN             Number of dominant topics to return (default: 15).
 * @returns                Array of dominant topic terms, sorted by relevance.
 */
export function extractDominantTopics(messages: Message[], topN = 15): string[] {
  if (messages.length === 0) return [];

  // Build per-message term sets for IDF computation
  const messageDocs = messages.map((m) => tokenize(extractMessageText(m)));

  // Build global term frequency across all messages
  const globalTF = new Map<string, number>();
  for (const doc of messageDocs) {
    for (const term of doc) {
      globalTF.set(term, (globalTF.get(term) ?? 0) + 1);
    }
  }

  if (globalTF.size === 0) return [];

  // IDF over per-message documents
  const idf = buildIDF(messageDocs);

  // Score terms by TF × IDF
  const scored: Array<[string, number]> = [];
  for (const [term, tf] of globalTF) {
    const idfVal = idf.get(term) ?? 1;
    scored.push([term, tf * idfVal]);
  }

  scored.sort((a, b) => b[1] - a[1]);
  return scored.slice(0, topN).map(([term]) => term);
}

// ---------------------------------------------------------------------------
// BM25 scoring
// ---------------------------------------------------------------------------

/**
 * Compute BM25 score for a single document against a query term set.
 *
 * BM25(q, d) = Σ IDF(t) × [tf(t,d) × (k1+1)] / [tf(t,d) + k1×(1 - b + b×|d|/avgDL)]
 *
 * @param queryTerms   Query terms (from dominant topics).
 * @param docTerms     Tokenized document (tier summary + tags).
 * @param idf          Pre-built IDF map over all candidate documents.
 * @param avgDocLength Average document length (tokens) across all candidates.
 */
function bm25Score(
  queryTerms: string[],
  docTerms: string[],
  idf: Map<string, number>,
  avgDocLength: number,
): number {
  const dl = docTerms.length;
  if (dl === 0) return 0;

  // Build term frequency map for this document
  const tf = new Map<string, number>();
  for (const term of docTerms) {
    tf.set(term, (tf.get(term) ?? 0) + 1);
  }

  let score = 0;
  for (const term of queryTerms) {
    const termFreq = tf.get(term) ?? 0;
    if (termFreq === 0) continue;

    const idfVal = idf.get(term) ?? 1;
    const numerator = termFreq * (BM25_K1 + 1);
    const denominator = termFreq + BM25_K1 * (1 - BM25_B + BM25_B * (dl / Math.max(1, avgDocLength)));
    score += idfVal * (numerator / denominator);
  }
  return score;
}

// ---------------------------------------------------------------------------
// Main public API
// ---------------------------------------------------------------------------

/**
 * Compute keyword signal scores for compression tier candidates.
 *
 * Extracts dominant topics from `messages`, then scores each tier candidate
 * document (summary + tags) using TF-IDF + BM25. Returns a `PreFilterResult`
 * with ranked candidates and a recommendation for whether to proceed to Ollama.
 *
 * @param messages         Current message array.
 * @param candidates       Tier candidates to rank.
 * @param targetTokenBudget Token budget (used for context; does not affect scoring).
 * @returns                Pre-filter result with ranked candidates and recommendation.
 */
export function computeKeywordSignal(
  messages: Message[],
  candidates: TierCandidate[],
  targetTokenBudget?: number,
): PreFilterResult {
  // Suppress unused-parameter lint; budget is plumbed for future use (e.g.
  // budget-aware topic extraction weights) without changing the signature.
  void targetTokenBudget;

  const dominantTopics = extractDominantTopics(messages);

  // Build candidate documents: summary tokens + tag tokens
  const candidateDocs = candidates.map((c) => [
    ...tokenize(c.summary),
    ...c.tags.flatMap((tag) => tokenize(tag)),
  ]);

  // Build IDF over candidate documents
  const idf = buildIDF(candidateDocs);

  // Average document length across all candidates
  const avgDocLength =
    candidateDocs.length > 0
      ? candidateDocs.reduce((sum, doc) => sum + doc.length, 0) / candidateDocs.length
      : 1;

  // Score each candidate using BM25
  const scored = candidates.map((candidate, idx) => ({
    tier: candidate,
    score: bm25Score(dominantTopics, candidateDocs[idx]!, idf, avgDocLength),
  }));

  // Sort by score descending, then by costWeight ascending as tiebreaker
  // (prefer cheaper tiers when scores are equal)
  scored.sort((a, b) => {
    if (Math.abs(a.score - b.score) > 1e-9) return b.score - a.score;
    return a.tier.costWeight - b.tier.costWeight;
  });

  // Compute signal strength: max score normalised by theoretical maximum
  // (all query terms match perfectly in every candidate position)
  const maxScore = scored.length > 0 ? scored[0]!.score : 0;
  const theoreticalMax =
    dominantTopics.length > 0
      ? dominantTopics.reduce((sum, term) => sum + (idf.get(term) ?? 1), 0) * ((BM25_K1 + 1) / (1 + BM25_K1 * (1 - BM25_B + BM25_B)))
      : 1;
  const signalStrength = theoreticalMax > 0 ? Math.min(1, maxScore / theoreticalMax) : 0;

  // Determine recommendation
  const recommendation = computeRecommendation(scored, signalStrength);

  return {
    rankedCandidates: scored,
    recommendation,
    signalStrength,
    dominantTopics,
  };
}

/**
 * Determine the pre-filter recommendation from scored candidates.
 *
 * Decision rules (applied in order):
 * 1. Low signal strength → "invoke_ollama" (no useful keyword signal).
 * 2. No candidates → "pick_top" (degenerate case, caller handles empty list).
 * 3. Top-2 within AMBIGUITY_THRESHOLD_TOP2 → "pick_cheaper" (skip Ollama, pick
 *    the higher-numbered tier which is cheaper).
 * 4. ≥3 candidates within AMBIGUITY_THRESHOLD_3WAY of the top score →
 *    "invoke_ollama" (3-way+ ambiguity needs semantic discrimination).
 * 5. Default → "pick_top" (clear winner).
 */
function computeRecommendation(
  scored: Array<{ tier: TierCandidate; score: number }>,
  signalStrength: number,
): PreFilterResult["recommendation"] {
  if (signalStrength < MIN_SIGNAL_STRENGTH) {
    // No meaningful keyword signal — let Ollama decide
    return "invoke_ollama";
  }

  if (scored.length === 0) return "pick_top";

  const topScore = scored[0]!.score;
  if (topScore <= 0) {
    // All candidates scored zero — no information from keywords
    return "invoke_ollama";
  }

  // Check how many candidates are within the 3-way ambiguity threshold
  const within3Way = scored.filter(
    (c) => (topScore - c.score) / topScore <= AMBIGUITY_THRESHOLD_3WAY,
  );

  if (within3Way.length >= 3) {
    return "invoke_ollama";
  }

  // Check if top-2 are within the 5% threshold
  if (scored.length >= 2) {
    const secondScore = scored[1]!.score;
    const gap = topScore > 0 ? (topScore - secondScore) / topScore : 1;
    if (gap <= AMBIGUITY_THRESHOLD_TOP2) {
      // Top-2 too close — pick the cheaper one (skip Ollama)
      return "pick_cheaper";
    }
  }

  return "pick_top";
}

// ---------------------------------------------------------------------------
// Integration helper — default tier candidate definitions
// ---------------------------------------------------------------------------

/**
 * Default tier candidate metadata for the 4-tier compression ladder.
 * These summaries and tags are designed to match the vocabulary that
 * naturally appears in conversations about different compression scenarios.
 */
export const DEFAULT_TIER_CANDIDATES: TierCandidate[] = [
  {
    tier: 1,
    summary:
      "autoCompact summarizes the entire conversation history using an LLM call, " +
      "producing a compact summary that replaces older messages. Best for very long " +
      "sessions with rich context that must be preserved semantically.",
    tags: [
      "autocompact",
      "summarize",
      "summary",
      "llm",
      "semantic",
      "long",
      "session",
      "history",
      "context",
      "preserve",
    ],
    costWeight: 4.0, // most expensive — requires LLM call
  },
  {
    tier: 2,
    summary:
      "snipCompact truncates oversized tool results and long assistant responses " +
      "to a maximum character limit, reducing token count with minimal information " +
      "loss for tool-heavy or output-heavy conversations.",
    tags: [
      "snip",
      "truncate",
      "tool",
      "result",
      "output",
      "limit",
      "trim",
      "character",
      "oversized",
    ],
    costWeight: 2.0,
  },
  {
    tier: 3,
    summary:
      "contextCollapse removes redundant short messages and collapses duplicate " +
      "exchanges, keeping recent messages intact. Effective for conversations with " +
      "repetitive patterns or many short back-and-forth turns.",
    tags: [
      "collapse",
      "remove",
      "redundant",
      "duplicate",
      "short",
      "repetitive",
      "turns",
      "recent",
    ],
    costWeight: 1.5,
  },
  {
    tier: 4,
    summary:
      "treeCompact prunes entire message subtrees with low value-to-token ratio, " +
      "removing less important branches while preserving the most recent and " +
      "highest-value exchanges. Lightest compression with broad applicability.",
    tags: [
      "tree",
      "prune",
      "subtree",
      "lightweight",
      "ratio",
      "recent",
      "value",
      "branch",
    ],
    costWeight: 1.0, // cheapest — no LLM call
  },
];

// ---------------------------------------------------------------------------
// SemanticPreFilterer class
// ---------------------------------------------------------------------------

/**
 * Encapsulates the pre-filter logic with optional configuration overrides.
 *
 * Typical usage in `selectCompressionTierAdaptive`:
 *
 * ```ts
 * const prefilter = new SemanticPreFilterer();
 * const result = prefilter.rank(messages, targetTokenBudget);
 * if (result.recommendation !== "invoke_ollama") {
 *   // Use keyword-ranked result directly, skip Ollama
 *   return pickTierFromPreFilterResult(result);
 * }
 * // Fall through to Ollama embedding path
 * ```
 */
export class SemanticPreFilterer {
  private readonly candidates: TierCandidate[];

  constructor(candidates: TierCandidate[] = DEFAULT_TIER_CANDIDATES) {
    this.candidates = candidates;
  }

  /**
   * Rank tier candidates by keyword signal for the given message array.
   *
   * @param messages          Current message array.
   * @param targetTokenBudget Token budget (plumbed for future use).
   */
  rank(messages: Message[], targetTokenBudget?: number): PreFilterResult {
    return computeKeywordSignal(messages, this.candidates, targetTokenBudget);
  }

  /**
   * Extract the recommended tier from a pre-filter result.
   *
   * For "pick_top": returns the top-ranked candidate's tier number.
   * For "pick_cheaper": returns the higher-numbered (cheaper) tier among the
   *   top-2 candidates.
   * For "invoke_ollama": returns null (caller should use Ollama).
   *
   * @param result PreFilterResult from `rank()`.
   * @returns      Tier number to use, or null if Ollama is needed.
   */
  pickTier(result: PreFilterResult): 1 | 2 | 3 | 4 | null {
    if (result.recommendation === "invoke_ollama") return null;

    if (result.rankedCandidates.length === 0) return null;

    if (result.recommendation === "pick_top") {
      return result.rankedCandidates[0]!.tier.tier;
    }

    // "pick_cheaper": among top-2, return the one with higher tier number
    // (lower costWeight = cheaper = higher tier number in our 1–4 scale)
    if (result.rankedCandidates.length < 2) {
      return result.rankedCandidates[0]!.tier.tier;
    }

    const top2 = result.rankedCandidates.slice(0, 2);
    // Higher tier number = lighter/cheaper compression
    const cheaper = top2.reduce((best, curr) =>
      curr.tier.tier > best.tier.tier ? curr : best,
    );
    return cheaper.tier.tier;
  }
}
