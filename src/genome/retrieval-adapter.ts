/**
 * Genome retrieval adapter — A/B testing and adaptive strategy selection.
 *
 * Wraps both `retrieveSections` (keyword/TF-IDF) and `semanticSearch` (Ollama
 * embeddings), records outcome metrics per retrieval, and learns which strategy
 * to prefer based on accumulated results.
 *
 * Outcomes are persisted to JSONL for append-only durability (same pattern as
 * strategies.ts mutations). Strategy selection uses a scoring model over recent
 * outcome windows with latency-pressure fallback.
 */

import { join } from "path";
import { appendJsonl, readJsonl } from "./jsonl.ts";
import { genomeDir } from "./manifest.ts";
import type { RetrievedSection } from "./retriever.ts";
import type { EmbeddingModelName } from "./embedding-router.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type RetrievalStrategy = "keyword" | "semantic";

/** Metrics captured for a single retrieval attempt. */
export interface StrategyMetrics {
  /** Wall-clock time for the retrieval in milliseconds. */
  latencyMs: number;
  /** Number of sections returned. */
  sectionsReturned: number;
  /** Sum of relevance scores across returned sections (0 if none). */
  totalRelevanceScore: number;
  /** Whether the result fit within the requested token budget. */
  budgetFit: boolean;
  /** Caller-supplied quality signal: 1.0 = perfect, 0.0 = useless. */
  matchQuality: number;
}

/** A single logged retrieval outcome record. */
export interface RetrievalOutcomeRecord {
  id: string;
  query: string;
  queryType: QueryType;
  strategy: RetrievalStrategy;
  /**
   * The embedding model used when strategy === "semantic".
   * Undefined for keyword retrievals or legacy records without model tracking.
   */
  modelName?: EmbeddingModelName;
  metrics: StrategyMetrics;
  timestamp: string;
}

/** Coarse classification of a query for adaptive scoring. */
export type QueryType = "rare-term" | "semantic" | "general";

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

function outcomesPath(cwd: string): string {
  return join(genomeDir(cwd), "evolution", "retrieval-outcomes.jsonl");
}

// ---------------------------------------------------------------------------
// Query classification
// ---------------------------------------------------------------------------

/**
 * Classify a query as rare-term, semantic, or general.
 *
 * Heuristics:
 *  - rare-term: contains camelCase identifiers, snake_case tokens, file
 *    extensions, or operator-like symbols — specific technical lookups that
 *    keyword matching excels at.
 *  - semantic: natural-language phrases (multiple common words, no code tokens)
 *    that benefit from embedding-based similarity search.
 *  - general: short queries or mixed signals.
 */
export function classifyQuery(query: string): QueryType {
  const trimmed = query.trim();
  if (trimmed.length === 0) return "general";

  // Code-like signals: camelCase, snake_case, file extensions, dots, parens, brackets
  const codeSignals = [
    /[a-z][A-Z]/, // camelCase
    /[a-z]_[a-z]/, // snake_case
    /\.[a-z]{1,5}(\s|$)/, // file extension
    /[()[\]{}<>]/, // brackets
    /[=!<>]{1,2}/, // operators
    /\b0x[0-9a-fA-F]+\b/, // hex literals
    /#[a-zA-Z]/, // tag-like (#something)
  ];

  let codeScore = 0;
  for (const re of codeSignals) {
    if (re.test(trimmed)) codeScore++;
  }

  // A single strong code signal on a short input (≤4 words) is enough.
  const wordCount = trimmed.split(/\s+/).length;
  if (codeScore >= 2 || (codeScore >= 1 && wordCount <= 4)) return "rare-term";

  // Semantic signals: multiple common English stop-words or natural phrasing
  const words = trimmed.toLowerCase().split(/\s+/);
  const commonWords = new Set([
    "how", "what", "why", "when", "where", "which", "who",
    "does", "should", "will", "can", "would", "could",
    "the", "and", "for", "with", "from", "that", "this",
    "find", "show", "list", "explain", "describe",
  ]);
  const naturalCount = words.filter((w) => commonWords.has(w)).length;

  if (naturalCount >= 2 && words.length >= 4) return "semantic";

  return "general";
}

// ---------------------------------------------------------------------------
// Outcome logging
// ---------------------------------------------------------------------------

/**
 * Record the outcome of a retrieval attempt to JSONL.
 *
 * This is the primary learning signal: each call appends one line so the
 * adaptive selector can improve over time without any in-memory state.
 *
 * Pass `modelName` when `strategy === "semantic"` to enable per-model scoring.
 */
export async function recordRetrievalOutcome(
  cwd: string,
  query: string,
  strategy: RetrievalStrategy,
  metrics: StrategyMetrics,
  modelName?: EmbeddingModelName,
): Promise<string> {
  const id = `ro-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
  const record: RetrievalOutcomeRecord = {
    id,
    query,
    queryType: classifyQuery(query),
    strategy,
    ...(modelName !== undefined ? { modelName } : {}),
    metrics,
    timestamp: new Date().toISOString(),
  };
  await appendJsonl(outcomesPath(cwd), record);
  return id;
}

/**
 * Load all recorded outcome records.
 */
export async function loadRetrievalOutcomes(cwd: string): Promise<RetrievalOutcomeRecord[]> {
  return readJsonl<RetrievalOutcomeRecord>(outcomesPath(cwd));
}

// ---------------------------------------------------------------------------
// Scoring model
// ---------------------------------------------------------------------------

/**
 * Weighted score for a strategy from a slice of outcome records.
 *
 * When `modelName` is provided (semantic strategy), only records matching that
 * model are considered — enabling per-(queryType, modelName) scoring.
 * When `modelName` is omitted, all records for the strategy are scored
 * together (backward-compatible behaviour).
 */
function scoreOutcomes(
  records: RetrievalOutcomeRecord[],
  strategy: RetrievalStrategy,
  modelName?: EmbeddingModelName,
): number {
  let relevant = records.filter((r) => r.strategy === strategy);

  // Narrow to model-specific records when a model is specified and the strategy
  // is semantic. Records without modelName are included for backward compat.
  if (strategy === "semantic" && modelName !== undefined) {
    const modelSpecific = relevant.filter((r) => r.modelName === modelName);
    // Only filter down if we have model-specific data; otherwise fall back to
    // all semantic records so existing data still contributes.
    if (modelSpecific.length > 0) {
      relevant = modelSpecific;
    }
  }

  if (relevant.length === 0) return 0;

  let total = 0;
  for (const r of relevant) {
    const m = r.metrics;
    // Quality is the primary signal (0–1), supplemented by budget-fit and
    // a small latency penalty (>2s penalises).
    const latencyPenalty = m.latencyMs > 2000 ? 0.1 : 0;
    const budgetBonus = m.budgetFit ? 0.05 : 0;
    const score = Math.max(0, m.matchQuality + budgetBonus - latencyPenalty);
    total += score;
  }

  return total / relevant.length;
}

// ---------------------------------------------------------------------------
// AdaptiveRetriever
// ---------------------------------------------------------------------------

/** Options for constructing an AdaptiveRetriever. */
export interface AdaptiveRetrieverOptions {
  /**
   * Maximum number of recent outcome records to consider when selecting a
   * strategy. Defaults to 50.
   */
  windowSize?: number;
  /**
   * Latency budget in milliseconds. When the preferred strategy's average
   * latency in the window exceeds this, fall back to keyword search.
   * Defaults to 3000 ms.
   */
  latencyBudgetMs?: number;
  /**
   * Minimum number of outcomes needed before adaptive selection activates.
   * Below this threshold the adapter uses heuristic defaults.
   * Defaults to 4.
   */
  minOutcomesForLearning?: number;
  /**
   * When true, the AdaptiveRetriever uses the EmbeddingRouter to select the
   * best embedding model per query complexity. When false (legacy), semantic
   * search uses the default nomic-embed-text model.
   * Defaults to true.
   */
  useModelRouter?: boolean;
}

/**
 * AdaptiveRetriever wraps both retrieval strategies and learns which to prefer
 * based on accumulated outcome metrics stored in JSONL.
 *
 * Model-awareness: when `useModelRouter` is true (default), semantic retrievals
 * are routed through EmbeddingRouter, which selects the best embedding model per
 * query complexity and tracks per-(queryType, modelName) outcome metrics.
 */
export class AdaptiveRetriever {
  private readonly cwd: string;
  private readonly windowSize: number;
  private readonly latencyBudgetMs: number;
  private readonly minOutcomesForLearning: number;
  private readonly useModelRouter: boolean;

  constructor(cwd: string, options: AdaptiveRetrieverOptions = {}) {
    this.cwd = cwd;
    this.windowSize = options.windowSize ?? 50;
    this.latencyBudgetMs = options.latencyBudgetMs ?? 3000;
    this.minOutcomesForLearning = options.minOutcomesForLearning ?? 4;
    this.useModelRouter = options.useModelRouter ?? true;
  }

  /**
   * Select the best retrieval strategy for a query by scoring prior outcomes.
   *
   * Decision logic (in order):
   *  1. Load recent outcomes from JSONL.
   *  2. If fewer than `minOutcomesForLearning` records exist, fall back to
   *     heuristic: rare-term → keyword, semantic → semantic, general → keyword.
   *  3. Filter outcomes to the same query type, take the most recent `windowSize`.
   *  4. Score keyword vs semantic by average (matchQuality + budget_bonus - latency_penalty).
   *     When model-routing is enabled, semantic scores are computed per model;
   *     the best-model semantic score is used for the strategy comparison.
   *  5. If the winning strategy's average latency > `latencyBudgetMs`, prefer keyword.
   *  6. Return the winner.
   */
  async selectBestStrategy(query: string): Promise<RetrievalStrategy> {
    const queryType = classifyQuery(query);

    const all = await loadRetrievalOutcomes(this.cwd);

    // Heuristic fallback when there is not enough data to learn from.
    if (all.length < this.minOutcomesForLearning) {
      return this._heuristicDefault(queryType);
    }

    // Narrow to same query-type window for relevance; fall back to all records
    // if the type-specific window is too thin.
    let window = all
      .filter((r) => r.queryType === queryType)
      .slice(-this.windowSize);

    if (window.length < this.minOutcomesForLearning) {
      window = all.slice(-this.windowSize);
    }

    const keywordScore = scoreOutcomes(window, "keyword");

    // For semantic, compute the best per-model score when model routing is on.
    let semanticScore: number;
    if (this.useModelRouter) {
      semanticScore = this._bestSemanticModelScore(window);
    } else {
      semanticScore = scoreOutcomes(window, "semantic");
    }

    // Determine the preferred winner.
    let preferred: RetrievalStrategy = semanticScore > keywordScore ? "semantic" : "keyword";

    // Latency-pressure fallback: if the winner is consistently slow, use keyword.
    if (preferred === "semantic") {
      const avgLatency = this._avgLatency(window, "semantic");
      if (avgLatency > this.latencyBudgetMs) {
        preferred = "keyword";
      }
    }

    return preferred;
  }

  /**
   * Execute retrieval with automatic strategy selection.
   *
   * Runs the selected strategy, records timing, and returns both the results
   * and the strategy used. Callers should later call `recordRetrievalOutcome`
   * with their quality assessment.
   *
   * When model routing is enabled and the semantic strategy is chosen, the
   * embedding model is selected automatically by EmbeddingRouter and the
   * returned `modelName` indicates which model was used.
   */
  async retrieve(
    query: string,
    maxTokens: number,
  ): Promise<{
    results: RetrievedSection[];
    strategy: RetrievalStrategy;
    latencyMs: number;
    modelName?: EmbeddingModelName;
  }> {
    const strategy = await this.selectBestStrategy(query);
    const start = Date.now();

    let results: RetrievedSection[] = [];

    if (strategy === "semantic") {
      const semanticResult = await this._runSemantic(query, maxTokens);
      results = semanticResult.results;
      // If semantic returned nothing (Ollama unavailable), fall back to keyword.
      if (results.length === 0) {
        results = await this._runKeyword(query, maxTokens);
        const latencyMs = Date.now() - start;
        return { results, strategy: "keyword", latencyMs };
      }
      const latencyMs = Date.now() - start;
      return {
        results,
        strategy,
        latencyMs,
        modelName: semanticResult.modelName,
      };
    } else {
      results = await this._runKeyword(query, maxTokens);
    }

    const latencyMs = Date.now() - start;
    return { results, strategy, latencyMs };
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  private _heuristicDefault(queryType: QueryType): RetrievalStrategy {
    // Rare-term queries (code identifiers, exact names) → keyword wins.
    // Semantic natural-language queries → semantic may help.
    // General → keyword is safe default.
    return queryType === "semantic" ? "semantic" : "keyword";
  }

  private _avgLatency(records: RetrievalOutcomeRecord[], strategy: RetrievalStrategy): number {
    const relevant = records.filter((r) => r.strategy === strategy);
    if (relevant.length === 0) return 0;
    return relevant.reduce((sum, r) => sum + r.metrics.latencyMs, 0) / relevant.length;
  }

  /**
   * Compute the best semantic score across all known embedding models.
   * Returns the highest per-model score, enabling fair comparison with keyword.
   */
  private _bestSemanticModelScore(records: RetrievalOutcomeRecord[]): number {
    const { EMBEDDING_MODELS } = require("./embedding-router.ts");
    const modelNames = Object.keys(EMBEDDING_MODELS) as EmbeddingModelName[];
    let best = scoreOutcomes(records, "semantic"); // fallback: unfiltered
    for (const modelName of modelNames) {
      const s = scoreOutcomes(records, "semantic", modelName);
      if (s > best) best = s;
    }
    return best;
  }

  private async _runKeyword(query: string, maxTokens: number): Promise<RetrievedSection[]> {
    const { retrieveSections } = await import("./retriever.ts");
    return retrieveSections(this.cwd, query, maxTokens);
  }

  private async _runSemantic(
    query: string,
    maxTokens: number,
  ): Promise<{ results: RetrievedSection[]; modelName?: EmbeddingModelName }> {
    try {
      const { isOllamaAvailable, semanticSearch } = await import("./embeddings.ts");
      if (!(await isOllamaAvailable())) return { results: [] };

      if (this.useModelRouter) {
        // Route through EmbeddingRouter to select the best model per query complexity
        const { EmbeddingRouter } = await import("./embedding-router.ts");
        const router = new EmbeddingRouter(this.cwd, {
          windowSize: this.windowSize,
          latencyBudgetMs: this.latencyBudgetMs,
          minOutcomesForLearning: this.minOutcomesForLearning,
        });
        const { classifyQueryComplexity, selectEmbeddingModel, loadModelOutcomes } =
          await import("./embedding-router.ts");
        const complexity = classifyQueryComplexity(query);
        const outcomes = await loadModelOutcomes(this.cwd);
        const modelName = selectEmbeddingModel(complexity, outcomes, {
          windowSize: this.windowSize,
          latencyBudgetMs: this.latencyBudgetMs,
          minOutcomesForLearning: this.minOutcomesForLearning,
        });
        void router; // router used for side-effect: outcome tracking done by caller
        const results = await semanticSearch(this.cwd, query, maxTokens, modelName);
        return { results, modelName };
      }

      // Legacy path: use default model
      const results = await semanticSearch(this.cwd, query, maxTokens);
      return { results };
    } catch {
      return { results: [] };
    }
  }
}

// ---------------------------------------------------------------------------
// Convenience factory
// ---------------------------------------------------------------------------

/**
 * Create an AdaptiveRetriever bound to the given working directory.
 */
export function createAdaptiveRetriever(cwd: string, options?: AdaptiveRetrieverOptions): AdaptiveRetriever {
  return new AdaptiveRetriever(cwd, options);
}
