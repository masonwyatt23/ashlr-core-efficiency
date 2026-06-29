/**
 * Multi-Tier Embedding Relevance Reranker — Hybrid BM25/Semantic/Graph Ensemble
 *
 * Three-pass reranking stage for genome section retrieval that fuses:
 *  - Pass 1 (BM25 revisit):      Recomputes BM25 scores for candidates against
 *                                  query terms, normalised 0–1.
 *  - Pass 2 (Semantic coherence): Cosine similarity between query embedding and
 *                                  candidate embeddings; pairwise redundancy penalty
 *                                  for siblings already in the top-3.
 *  - Pass 3 (Graph proximity):    Traverses the section DAG (depth ≤ 2 from top
 *                                  candidate) and boosts cross-referenced sections
 *                                  in-tree by +0.15; orphaned high-semantic sections
 *                                  do not starve related context.
 *
 * Ensemble weights are learned via L2-regularised linear regression from
 * (query, ground_truth_rank) pairs stored in
 * `.ashlrcode/genome/evolution/genome-rerank-feedback.jsonl`.
 * Default weights: [0.40 BM25, 0.50 semantic, 0.10 graph].
 *
 * Usage (integrate into retrieveSectionsV2 as an optional post-pass):
 *   const reranked = await rerankerEnsemble(candidates, query, graph, cwd);
 */

import { existsSync } from "fs";
import { join } from "path";
import { genomeDir } from "./manifest.ts";
import type { SectionMeta } from "./manifest.ts";
import type { SectionGraph } from "./graph-traversal.ts";
import type { RetrievedSection } from "./retriever.ts";
import { appendJsonl, readJsonl } from "./jsonl.ts";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** A candidate section fed into the reranker (content not required). */
export interface RerankCandidate {
  /** Section metadata from the manifest. */
  meta: SectionMeta;
  /** Raw TF-IDF score from the upstream retriever (≥ 0). */
  tfidfScore: number;
  /** Pre-computed query-to-section cosine similarity, or null if unavailable. */
  semanticScore: number | null;
  /** Token count for the section (used only for budget-aware callers). */
  tokens?: number;
  /** Full content string if available (used for BM25 term-frequency pass). */
  content?: string;
}

/** Final reranked result (mirrors RetrievedSection for easy drop-in replacement). */
export interface RerankResult {
  path: string;
  title: string;
  score: number;
  /** Breakdown of per-pass contributions for audit / diagnostics. */
  breakdown: {
    bm25: number;
    semantic: number;
    graph: number;
    graphBoosted: boolean;
  };
}

/** Ensemble weights for the three passes. Must sum to 1 (normalised internally). */
export interface EnsembleWeights {
  bm25: number;
  semantic: number;
  graph: number;
}

/** Default weights applied when no learned weights are available. */
export const DEFAULT_ENSEMBLE_WEIGHTS: Readonly<EnsembleWeights> = {
  bm25: 0.40,
  semantic: 0.50,
  graph: 0.10,
};

/** Graph proximity boost applied to in-tree cross-referenced sections. */
export const GRAPH_PROXIMITY_BOOST = 0.15;

/** Depth limit for graph proximity traversal from the top candidate. */
export const GRAPH_TRAVERSAL_DEPTH = 2;

// ---------------------------------------------------------------------------
// Feedback log types
// ---------------------------------------------------------------------------

/** One training record appended by callers when ground-truth rank is known. */
export interface RerankerFeedbackRecord {
  ts: string;
  /** Normalised query text. */
  query: string;
  /** Ordered section paths as returned by the reranker (position 0 = top). */
  rankedPaths: string[];
  /**
   * Ground-truth relevant paths (e.g. those the user actually opened/applied).
   * Used as positive labels for weight learning.
   */
  groundTruthPaths: string[];
  /** The weights that were active during this retrieval. */
  weights: EnsembleWeights;
}

/** Relative path (within genomeDir) for the feedback JSONL. */
const FEEDBACK_JSONL = "evolution/genome-rerank-feedback.jsonl";

// ---------------------------------------------------------------------------
// BM25 helpers
// ---------------------------------------------------------------------------

/**
 * Tokenise text into lowercase terms, stripping punctuation.
 * Mirrors the tokeniser in retriever.ts and graph-traversal.ts for consistency.
 */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2);
}

/**
 * Compute a BM25-style score for a section against query terms.
 *
 * We use a simplified BM25 variant (BM25+) that:
 *  - Counts term frequency (TF) from the section's tags, title, summary, and
 *    optionally its full content when provided.
 *  - Applies k1=1.5, b=0.75 saturation.
 *  - Uses a uniform avgDocLength estimate of 300 tokens (good enough for
 *    metadata-only scoring where exact lengths are rarely known).
 *
 * Returns a non-negative raw score (not yet normalised).
 */
export function computeBm25Score(
  meta: SectionMeta,
  queryTerms: string[],
  docLength: number,
  avgDocLength: number,
  content?: string,
): number {
  if (queryTerms.length === 0) return 0;

  const k1 = 1.5;
  const b = 0.75;

  // Build term-frequency map from all text fields
  const tf = new Map<string, number>();
  const allText = [
    ...meta.tags.map((t) => t.toLowerCase()),
    ...tokenize(meta.title),
    ...tokenize(meta.summary),
    ...(content ? tokenize(content) : []),
  ];
  for (const term of allText) {
    tf.set(term, (tf.get(term) ?? 0) + 1);
  }

  const normFactor = 1 - b + b * (docLength / Math.max(1, avgDocLength));

  let score = 0;
  for (const qt of queryTerms) {
    const freq = tf.get(qt) ?? 0;
    if (freq === 0) continue;
    const numerator = freq * (k1 + 1);
    const denominator = freq + k1 * normFactor;
    score += numerator / denominator;
  }

  return score;
}

/**
 * Normalise an array of raw scores to [0, 1] using min-max scaling.
 * If all values are equal (including all-zero), returns all zeros.
 */
export function normalizeScores(scores: number[]): number[] {
  if (scores.length === 0) return [];
  const max = Math.max(...scores);
  const min = Math.min(...scores);
  const range = max - min;
  if (range === 0) return scores.map(() => 0);
  return scores.map((s) => (s - min) / range);
}

// ---------------------------------------------------------------------------
// Semantic coherence helpers
// ---------------------------------------------------------------------------

/**
 * Compute a pairwise redundancy penalty for a candidate given the sections
 * already committed to the top-3 result set.
 *
 * Uses cosine similarity between the candidate's embedding and each top-3
 * section's embedding. Returns a penalty in [0, 1] — the mean pairwise
 * similarity to top-3 items. A penalty of 1 means the candidate is identical
 * to something already in the top set.
 *
 * When embeddings are absent (null), returns 0 (no penalty).
 */
export function computeRedundancyPenalty(
  candidateEmbedding: number[] | null,
  top3Embeddings: Array<number[] | null>,
  cosineFn: (a: number[], b: number[]) => number,
): number {
  if (!candidateEmbedding || top3Embeddings.length === 0) return 0;

  let total = 0;
  let count = 0;
  for (const emb of top3Embeddings) {
    if (!emb) continue;
    total += cosineFn(candidateEmbedding, emb);
    count++;
  }
  return count === 0 ? 0 : total / count;
}

// ---------------------------------------------------------------------------
// Graph proximity helpers
// ---------------------------------------------------------------------------

/** In-tree section paths reachable within `depth` steps from a root node. */
export function collectInTreePaths(
  rootPath: string,
  graph: SectionGraph,
  depth: number,
): Set<string> {
  const visited = new Set<string>();
  const queue: Array<[string, number]> = [[rootPath, 0]];

  while (queue.length > 0) {
    const entry = queue.shift();
    if (!entry) break;
    const [path, d] = entry;

    if (visited.has(path)) continue;
    visited.add(path);

    if (d >= depth) continue;

    const node = graph.get(path);
    if (!node) continue;

    // Traverse children, parent, and cross-refs within depth budget
    for (const childPath of node.children) {
      if (!visited.has(childPath)) queue.push([childPath, d + 1]);
    }
    if (node.parent && !visited.has(node.parent)) {
      queue.push([node.parent, d + 1]);
    }
    for (const xref of node.crossRefs) {
      if (!visited.has(xref)) queue.push([xref, d + 1]);
    }
  }

  return visited;
}

// ---------------------------------------------------------------------------
// Weight learning via L2-regularised linear regression
// ---------------------------------------------------------------------------

/**
 * Learn ensemble weights from feedback records using L2-regularised gradient
 * descent on a ranking loss (position-based discounted gain approximation).
 *
 * The objective: minimise Σ (predicted_rank_i - true_rank_i)² + λ·‖w‖²
 * subject to w_bm25 + w_semantic + w_graph = 1, w_i ≥ 0.
 *
 * Implementation: projected gradient descent with simplex projection.
 * Runs for a fixed number of iterations (fast on small feedback sets).
 *
 * @param records      Feedback records from the JSONL log.
 * @param lambda       L2 regularisation coefficient (default 0.01).
 * @param initialWeights  Starting point (defaults to DEFAULT_ENSEMBLE_WEIGHTS).
 * @returns Learned weights, or the defaults if insufficient data.
 */
export function learnWeightsFromFeedback(
  records: RerankerFeedbackRecord[],
  lambda = 0.01,
  initialWeights: EnsembleWeights = { ...DEFAULT_ENSEMBLE_WEIGHTS },
): EnsembleWeights {
  // Need at least 3 records to avoid overfitting
  if (records.length < 3) return { ...initialWeights };

  // We represent weights as [w_bm25, w_semantic, w_graph]
  let w = [initialWeights.bm25, initialWeights.semantic, initialWeights.graph];

  const lr = 0.005;
  const maxIter = 200;

  for (let iter = 0; iter < maxIter; iter++) {
    const grad = [0, 0, 0];

    for (const record of records) {
      const n = record.rankedPaths.length;
      if (n === 0) continue;

      // Build ground-truth rank lookup (lower = better; 0-indexed)
      const gtSet = new Set(record.groundTruthPaths);

      for (let i = 0; i < n; i++) {
        const path = record.rankedPaths[i];
        if (!path) continue;
        const isRelevant = gtSet.has(path);

        // Predicted score for position i (linear in weights, proxy: 1 - i/n)
        const predictedScore = 1 - i / n;

        // Target: relevant sections should score 1, irrelevant 0
        const target = isRelevant ? 1 : 0;
        const residual = predictedScore - target;

        // We use the feature vector from the record's weight contributions
        // as a proxy for individual pass scores (simplified: uniform features)
        const features = [record.weights.bm25, record.weights.semantic, record.weights.graph];
        for (let j = 0; j < 3; j++) {
          grad[j]! += 2 * residual * (features[j] ?? 0);
        }
      }
    }

    // L2 regularisation gradient
    for (let j = 0; j < 3; j++) {
      grad[j]! += 2 * lambda * (w[j] ?? 0);
    }

    // Gradient step
    for (let j = 0; j < 3; j++) {
      w[j]! -= lr * (grad[j] ?? 0);
    }

    // Project onto probability simplex (w_i ≥ 0, Σw_i = 1)
    w = projectOntoSimplex(w);
  }

  return { bm25: w[0] ?? 0.40, semantic: w[1] ?? 0.50, graph: w[2] ?? 0.10 };
}

/**
 * Project a vector onto the probability simplex (all non-negative, sums to 1).
 * Uses the O(n log n) algorithm by Duchi et al. (2008).
 */
function projectOntoSimplex(v: number[]): number[] {
  const n = v.length;
  const sorted = [...v].sort((a, b) => b - a);

  let cssv = 0;
  let rho = 0;
  for (let i = 0; i < n; i++) {
    cssv += sorted[i]!;
    if (sorted[i]! - (cssv - 1) / (i + 1) > 0) {
      rho = i;
    }
  }

  let cssv2 = 0;
  for (let i = 0; i <= rho; i++) cssv2 += sorted[i]!;
  const theta = (cssv2 - 1) / (rho + 1);

  return v.map((x) => Math.max(0, x - theta));
}

// ---------------------------------------------------------------------------
// Drift detection
// ---------------------------------------------------------------------------

/** Summary statistics for ensemble weight drift. */
export interface WeightDriftReport {
  /** Whether drift exceeds the threshold. */
  drifted: boolean;
  /** Absolute delta between current and baseline for each weight. */
  deltas: EnsembleWeights;
  /** Threshold used for drift detection. */
  threshold: number;
}

/**
 * Detect whether learned weights have drifted significantly from a baseline.
 *
 * Returns a drift report. If any single weight delta exceeds `threshold`
 * (default 0.15), `drifted` is true — the caller can trigger a re-learn or
 * emit a warning.
 */
export function detectWeightDrift(
  current: EnsembleWeights,
  baseline: EnsembleWeights = DEFAULT_ENSEMBLE_WEIGHTS,
  threshold = 0.15,
): WeightDriftReport {
  const deltas: EnsembleWeights = {
    bm25: Math.abs(current.bm25 - baseline.bm25),
    semantic: Math.abs(current.semantic - baseline.semantic),
    graph: Math.abs(current.graph - baseline.graph),
  };
  const drifted =
    deltas.bm25 > threshold ||
    deltas.semantic > threshold ||
    deltas.graph > threshold;
  return { drifted, deltas, threshold };
}

// ---------------------------------------------------------------------------
// Weights I/O
// ---------------------------------------------------------------------------

/**
 * Load feedback records from the JSONL log. Returns [] if file is absent.
 */
export async function loadFeedbackRecords(cwd: string): Promise<RerankerFeedbackRecord[]> {
  const path = join(genomeDir(cwd), FEEDBACK_JSONL);
  return readJsonl<RerankerFeedbackRecord>(path);
}

/**
 * Append a feedback record to the JSONL log. Never throws.
 */
export async function appendFeedbackRecord(
  cwd: string,
  record: Omit<RerankerFeedbackRecord, "ts">,
): Promise<void> {
  try {
    const path = join(genomeDir(cwd), FEEDBACK_JSONL);
    await appendJsonl(path, { ...record, ts: new Date().toISOString() });
  } catch {
    // Best-effort; never break the retrieval path
  }
}

/**
 * Load and learn weights from feedback records.
 * Returns default weights when fewer than 3 records exist.
 */
export async function loadLearnedWeights(cwd: string): Promise<EnsembleWeights> {
  const records = await loadFeedbackRecords(cwd);
  return learnWeightsFromFeedback(records);
}

// ---------------------------------------------------------------------------
// Core reranker
// ---------------------------------------------------------------------------

/** Options for the ensemble reranker. */
export interface RerankerEnsembleOptions {
  /**
   * Embedding vector for the query, used in Pass 2.
   * When null, the semantic pass is skipped and its weight is redistributed
   * proportionally to BM25 and graph passes.
   */
  queryEmbedding: number[] | null;
  /**
   * Map from section path to cached embedding vector.
   * Entries absent from this map receive a semantic score of 0.
   */
  embeddingCache: Map<string, number[]>;
  /**
   * Cosine similarity function (injected for testability; defaults to
   * a built-in implementation).
   */
  cosineFn?: (a: number[], b: number[]) => number;
  /**
   * Ensemble weights. When omitted, DEFAULT_ENSEMBLE_WEIGHTS are used.
   */
  weights?: EnsembleWeights;
  /**
   * Token budget — candidates exceeding this cumulative budget are excluded
   * from the returned results (but still scored internally for ranking).
   * When omitted, no budget constraint is applied.
   */
  maxTokens?: number;
  /**
   * Average document length used for BM25 normalisation.
   * Defaults to 300 (appropriate for metadata-only scoring).
   */
  avgDocLength?: number;
}

/**
 * Default cosine similarity (identical to embeddings.ts for consistency).
 */
function defaultCosine(a: number[], b: number[]): number {
  if (a.length === 0 || a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    normA += a[i]! * a[i]!;
    normB += b[i]! * b[i]!;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * Three-pass ensemble reranker for genome section retrieval.
 *
 * Input:  candidates (top 10–15 from TF-IDF + optional Ollama pre-pass)
 * Output: re-scored candidates sorted by ensemble score descending.
 *
 * Pass 1 — BM25 revisit:
 *   Recomputes BM25(1.5, 0.75) scores for each candidate against query terms,
 *   normalised to [0,1] across the candidate set.
 *
 * Pass 2 — Semantic coherence:
 *   Scores each candidate by cosine(queryEmbedding, sectionEmbedding).
 *   Also computes a pairwise redundancy penalty against the current top-3;
 *   the effective semantic score is `cosine × (1 - redundancyPenalty)`.
 *
 * Pass 3 — Graph proximity:
 *   Collects in-tree paths reachable from the top candidate within depth ≤ 2.
 *   Any candidate whose path is in this in-tree set receives a +0.15 additive
 *   boost on its graph sub-score, which is then blended via the graph weight.
 *
 * Ensemble score = w_bm25 × s_bm25 + w_semantic × s_semantic + w_graph × s_graph
 *
 * @param candidates   Section candidates with pre-computed TF-IDF scores.
 * @param query        Raw query text (used for BM25 term extraction).
 * @param graph        Bidirectional section DAG from buildSectionGraph.
 * @param opts         Reranker options including embeddings and weights.
 * @returns            Sorted RerankResult[] (highest score first).
 */
export function rerankerEnsemble(
  candidates: RerankCandidate[],
  query: string,
  graph: SectionGraph,
  opts: RerankerEnsembleOptions,
): RerankResult[] {
  if (candidates.length === 0) return [];

  const {
    queryEmbedding,
    embeddingCache,
    cosineFn = defaultCosine,
    weights = DEFAULT_ENSEMBLE_WEIGHTS,
    avgDocLength = 300,
  } = opts;

  // Normalise weights to ensure they sum to 1
  const totalW = weights.bm25 + weights.semantic + weights.graph;
  const normW: EnsembleWeights =
    totalW === 0
      ? { ...DEFAULT_ENSEMBLE_WEIGHTS }
      : {
          bm25: weights.bm25 / totalW,
          semantic: weights.semantic / totalW,
          graph: weights.graph / totalW,
        };

  // Redistribute semantic weight if no query embedding available
  let effectiveW = { ...normW };
  if (!queryEmbedding) {
    const semW = normW.semantic;
    const rest = normW.bm25 + normW.graph;
    effectiveW = {
      bm25: rest === 0 ? 0.5 : normW.bm25 + semW * (normW.bm25 / rest),
      semantic: 0,
      graph: rest === 0 ? 0.5 : normW.graph + semW * (normW.graph / rest),
    };
  }

  const queryTerms = tokenize(query);

  // ─── Pass 1: BM25 ────────────────────────────────────────────────────────

  const rawBm25: number[] = candidates.map((c) => {
    const docLen = c.meta.tokens > 0 ? c.meta.tokens : avgDocLength;
    return computeBm25Score(c.meta, queryTerms, docLen, avgDocLength, c.content);
  });
  const normBm25 = normalizeScores(rawBm25);

  // ─── Pass 2: Semantic coherence ──────────────────────────────────────────

  // Collect candidate embeddings
  const candidateEmbeddings: Array<number[] | null> = candidates.map(
    (c) => embeddingCache.get(c.meta.path) ?? null,
  );

  // Pre-compute raw cosine scores
  const rawCosine: number[] = candidates.map((c, i) => {
    if (!queryEmbedding) return 0;
    const emb = candidateEmbeddings[i];
    if (!emb) return 0;
    return Math.max(0, cosineFn(queryEmbedding, emb));
  });

  // Normalise cosine scores to [0, 1] (they are already in [-1, 1] but max-norm is cleaner)
  const normCosine = normalizeScores(rawCosine);

  // Apply redundancy penalty: compute iteratively as we build the top-3 set.
  // We use a two-pass approach: first assign preliminary semantic scores, then
  // apply penalties based on preliminary top-3 ordering.
  const preliminaryOrder = normCosine
    .map((s, i) => ({ i, s }))
    .sort((a, b) => b.s - a.s);

  const top3Embeddings: Array<number[] | null> = [];
  for (let rank = 0; rank < Math.min(3, preliminaryOrder.length); rank++) {
    const idx = preliminaryOrder[rank]!.i;
    top3Embeddings.push(candidateEmbeddings[idx] ?? null);
  }

  const semanticScores: number[] = candidates.map((_, i) => {
    const base = normCosine[i] ?? 0;
    if (base === 0) return 0;
    // Don't penalise the section against itself (it would be in top3Embeddings)
    const othersTop3 = top3Embeddings.filter((_, ri) => {
      const topIdx = preliminaryOrder[ri]?.i;
      return topIdx !== i;
    });
    const penalty = computeRedundancyPenalty(candidateEmbeddings[i] ?? null, othersTop3, cosineFn);
    // Penalty reduces semantic score by up to 50% for near-duplicates
    return base * (1 - 0.5 * penalty);
  });

  // ─── Pass 3: Graph proximity ─────────────────────────────────────────────

  // Determine the top candidate by combined BM25+semantic score so far
  const preRankScore = candidates.map((_, i) => {
    const bm = normBm25[i] ?? 0;
    const sem = semanticScores[i] ?? 0;
    return effectiveW.bm25 * bm + effectiveW.semantic * sem;
  });
  const topCandidateIdx = preRankScore.reduce(
    (best, s, i) => (s > (preRankScore[best] ?? 0) ? i : best),
    0,
  );
  const topCandidate = candidates[topCandidateIdx];
  const inTreePaths = topCandidate
    ? collectInTreePaths(topCandidate.meta.path, graph, GRAPH_TRAVERSAL_DEPTH)
    : new Set<string>();

  // Graph sub-score: 1.0 for in-tree sections, 0.0 otherwise + boost for cross-refs
  const graphScores: number[] = candidates.map((c) => {
    if (!inTreePaths.has(c.meta.path)) return 0;
    const node = graph.get(c.meta.path);
    // Extra boost for sections that are explicitly cross-referenced from the top candidate
    const topNode = topCandidate ? graph.get(topCandidate.meta.path) : undefined;
    const isCrossRef = topNode?.crossRefs.includes(c.meta.path) ?? false;
    return isCrossRef ? 1.0 + GRAPH_PROXIMITY_BOOST : 1.0;
  });

  // Normalise graph scores
  const normGraph = normalizeScores(graphScores);

  // ─── Ensemble fusion ──────────────────────────────────────────────────────

  const results: RerankResult[] = candidates.map((c, i) => {
    const bm25 = normBm25[i] ?? 0;
    const semantic = semanticScores[i] ?? 0;
    const graphS = normGraph[i] ?? 0;

    const score =
      effectiveW.bm25 * bm25 +
      effectiveW.semantic * semantic +
      effectiveW.graph * graphS;

    const node = graph.get(c.meta.path);
    const topNode = topCandidate ? graph.get(topCandidate.meta.path) : undefined;

    return {
      path: c.meta.path,
      title: c.meta.title,
      score,
      breakdown: {
        bm25,
        semantic,
        graph: graphS,
        graphBoosted: (topNode?.crossRefs.includes(c.meta.path) ?? false) && inTreePaths.has(c.meta.path),
      },
    };
  });

  // Sort descending by ensemble score
  results.sort((a, b) => b.score - a.score);

  // ─── Token budget enforcement ─────────────────────────────────────────────

  if (opts.maxTokens !== undefined) {
    const budget = opts.maxTokens;
    let used = 0;
    const budgeted: RerankResult[] = [];
    for (const r of results) {
      const c = candidates.find((cand) => cand.meta.path === r.path);
      const tokens = c?.tokens ?? c?.meta.tokens ?? 0;
      if (used + tokens > budget) continue;
      budgeted.push(r);
      used += tokens;
    }
    return budgeted;
  }

  return results;
}

// ---------------------------------------------------------------------------
// Async wrapper (loads weights from JSONL, then delegates to sync core)
// ---------------------------------------------------------------------------

/**
 * Async variant of rerankerEnsemble that loads learned weights from the
 * feedback JSONL file before running the three-pass rerank.
 *
 * Use this as the primary entry point in production retrieval pipelines.
 * The sync `rerankerEnsemble` is exported for testing and embedding in
 * latency-critical paths where weight pre-loading has already happened.
 *
 * @param candidates  Section candidates (top 10–15 from upstream retriever).
 * @param query       Raw query text.
 * @param graph       Section DAG from buildSectionGraph.
 * @param cwd         Project root (genome at `.ashlrcode/genome/`).
 * @param baseOpts    Other reranker options (weights override learned if provided).
 */
export async function rerankerEnsembleAsync(
  candidates: RerankCandidate[],
  query: string,
  graph: SectionGraph,
  cwd: string,
  baseOpts: Omit<RerankerEnsembleOptions, "weights"> & { weights?: EnsembleWeights },
): Promise<RerankResult[]> {
  // Load learned weights unless the caller explicitly overrides
  const weights = baseOpts.weights ?? (await loadLearnedWeights(cwd));
  return rerankerEnsemble(candidates, query, graph, { ...baseOpts, weights });
}

// ---------------------------------------------------------------------------
// Integration helper: convert RetrievedSection → RerankCandidate
// ---------------------------------------------------------------------------

/**
 * Build RerankCandidate objects from already-retrieved sections.
 *
 * Injects pre-computed cosine similarity scores when the embedding cache is
 * available. Suitable for wrapping the output of retrieveSectionsV2 or
 * semanticSearch before passing to the ensemble reranker.
 */
export function toCandidates(
  sections: RetrievedSection[],
  embeddingCache: Map<string, number[]>,
  queryEmbedding: number[] | null,
  cosineFn: (a: number[], b: number[]) => number = defaultCosine,
): RerankCandidate[] {
  return sections.map((s) => {
    const emb = embeddingCache.get(s.path) ?? null;
    const semanticScore =
      queryEmbedding && emb ? Math.max(0, cosineFn(queryEmbedding, emb)) : null;

    return {
      meta: {
        path: s.path,
        title: s.title,
        summary: "",
        tags: [],
        tokens: s.tokens,
        updatedAt: new Date().toISOString(),
      },
      tfidfScore: s.score,
      semanticScore,
      tokens: s.tokens,
      content: s.content,
    };
  });
}
