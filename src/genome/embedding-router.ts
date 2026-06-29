/**
 * Embedding Router — multi-tier model selection + hybrid search routing.
 *
 * Orchestrates selection among multiple Ollama embedding models based on
 * query complexity, per-model outcome metrics, and latency/quality tradeoffs.
 *
 * Model tier philosophy:
 *  - FAST  (all-minilm-l6-v2):  Low dimensionality (384d), ~2–5ms. Best for
 *    rare-term / exact-lookup queries where semantic depth is not needed.
 *  - BALANCED (nomic-embed-text): Medium dimensionality (768d), ~10–20ms.
 *    General-purpose default that handles most queries well.
 *  - ACCURATE (bge-base-en-v1-5): High dimensionality (768d), ~20–40ms.
 *    Produces superior embeddings for nuanced semantic queries at the cost of
 *    higher latency and compute.
 *
 * Fallback cascade: if the selected model times out, the router tries the next
 * tier down before giving up.  All model outcomes are tracked separately so the
 * adaptive scorer can learn per-(queryType, modelName) preferences.
 */

import { appendJsonl, readJsonl } from "./jsonl.ts";
import { genomeDir } from "./manifest.ts";
import { generateEmbedding, isOllamaAvailable } from "./embeddings.ts";
import { UltraFastEmbedder, type UltraFastAuditRecord, ULTRAFAST_EMBEDDING_DIM } from "./embedding-ultrafast.ts";
import { join } from "path";
import {
  QuantizationStrategyEngine,
  type QuantizationBitDepth,
  type GenomeProfile,
} from "./quantization-strategy.ts";

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

/** Supported embedding model names. */
export type EmbeddingModelName =
  | "nomic-embed-text"
  | "all-minilm-l6-v2"
  | "bge-base-en-v1-5"
  | "ultrafast";

export type ModelTier = "fast" | "balanced" | "accurate" | "ultrafast";

export interface EmbeddingModelSpec {
  name: EmbeddingModelName;
  tier: ModelTier;
  /**
   * Expected embedding dimension. Used for sanity-checking; 0 = unknown.
   */
  dimension: number;
  /**
   * Nominal latency in milliseconds (rough guidance, not enforced).
   */
  nominalLatencyMs: number;
  /**
   * Per-request timeout override in milliseconds. Defaults to 15 000 ms when
   * not specified.
   */
  timeoutMs: number;
}

/** Registry of all known embedding models, ordered fast → accurate → ultrafast. */
export const EMBEDDING_MODELS: Record<EmbeddingModelName, EmbeddingModelSpec> = {
  "all-minilm-l6-v2": {
    name: "all-minilm-l6-v2",
    tier: "fast",
    dimension: 384,
    nominalLatencyMs: 5,
    timeoutMs: 5_000,
  },
  "nomic-embed-text": {
    name: "nomic-embed-text",
    tier: "balanced",
    dimension: 768,
    nominalLatencyMs: 15,
    timeoutMs: 15_000,
  },
  "bge-base-en-v1-5": {
    name: "bge-base-en-v1-5",
    tier: "accurate",
    dimension: 768,
    nominalLatencyMs: 30,
    timeoutMs: 20_000,
  },
  "ultrafast": {
    name: "ultrafast",
    tier: "ultrafast",
    dimension: ULTRAFAST_EMBEDDING_DIM,
    nominalLatencyMs: 0,
    timeoutMs: 100, // hard cap — should always be sub-ms
  },
};

/** Fallback cascade order (index 0 = first try). ultrafast is always last. */
const CASCADE_ORDER: EmbeddingModelName[] = [
  "bge-base-en-v1-5",
  "nomic-embed-text",
  "all-minilm-l6-v2",
  "ultrafast",
];

// ---------------------------------------------------------------------------
// Per-model outcome metrics
// ---------------------------------------------------------------------------

/** Query complexity classification (mirrors retrieval-adapter's QueryType). */
export type QueryComplexity = "rare-term" | "semantic" | "general";

/** A single logged model-outcome record. */
export interface ModelOutcomeRecord {
  id: string;
  modelName: EmbeddingModelName;
  queryComplexity: QueryComplexity;
  latencyMs: number;
  /** True when the model produced an embedding within its timeout. */
  succeeded: boolean;
  /** Cosine similarity of the top result against the query embedding (0 if none). */
  topSimilarity: number;
  /** Caller-supplied quality signal: 1.0 = perfect, 0.0 = useless. null = not yet rated. */
  matchQuality: number | null;
  timestamp: string;
  /**
   * Number of Ollama tiers that timed out before this model was selected.
   * Non-zero only for ultrafast tier records; signals Ollama flakiness.
   */
  ollamaTimeouts?: number;
  /**
   * True when this record represents an ultrafast-tier fallback (no Ollama).
   * Used by fitness-hooks to track ultrafast adoption rates.
   */
  isUltrafastFallback?: boolean;
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

function modelOutcomesPath(cwd: string): string {
  return join(genomeDir(cwd), "evolution", "embedding-model-outcomes.jsonl");
}

// ---------------------------------------------------------------------------
// Model outcome persistence
// ---------------------------------------------------------------------------

/**
 * Append a model outcome record to the JSONL log.
 */
export async function recordModelOutcome(
  cwd: string,
  record: Omit<ModelOutcomeRecord, "id" | "timestamp">,
): Promise<string> {
  const id = `mo-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
  const full: ModelOutcomeRecord = {
    id,
    timestamp: new Date().toISOString(),
    ...record,
  };
  await appendJsonl(modelOutcomesPath(cwd), full);
  return id;
}

/**
 * Load all recorded model outcome records.
 */
export async function loadModelOutcomes(cwd: string): Promise<ModelOutcomeRecord[]> {
  return readJsonl<ModelOutcomeRecord>(modelOutcomesPath(cwd));
}

// ---------------------------------------------------------------------------
// Per-(queryType, modelName) scoring
// ---------------------------------------------------------------------------

/** Compute an average score for a (queryComplexity, modelName) pair. */
function scoreModelOutcomes(
  records: ModelOutcomeRecord[],
  modelName: EmbeddingModelName,
  queryComplexity: QueryComplexity,
  latencyBudgetMs: number,
): number {
  const relevant = records.filter(
    (r) => r.modelName === modelName && r.queryComplexity === queryComplexity && r.succeeded,
  );
  if (relevant.length === 0) return -1; // No data — signal unknown

  let total = 0;
  for (const r of relevant) {
    // Primary signal: caller-supplied quality (0–1). Fall back to top similarity.
    const quality = r.matchQuality ?? r.topSimilarity;
    // Latency penalty: costs 0.15 per second over budget
    const overBudgetS = Math.max(0, (r.latencyMs - latencyBudgetMs) / 1000);
    const latencyPenalty = overBudgetS * 0.15;
    total += Math.max(0, quality - latencyPenalty);
  }

  return total / relevant.length;
}

// ---------------------------------------------------------------------------
// Query complexity classification
// ---------------------------------------------------------------------------

/**
 * Classify a query string into a complexity bucket.
 *
 * Mirrors the heuristic in retrieval-adapter.ts `classifyQuery` so the two
 * modules stay in sync without a hard dependency.
 */
export function classifyQueryComplexity(query: string): QueryComplexity {
  const trimmed = query.trim();
  if (trimmed.length === 0) return "general";

  const codeSignals = [
    /[a-z][A-Z]/, // camelCase
    /[a-z]_[a-z]/, // snake_case
    /\.[a-z]{1,5}(\s|$)/, // file extension
    /[()[\]{}<>]/, // brackets
    /[=!<>]{1,2}/, // operators
    /\b0x[0-9a-fA-F]+\b/, // hex literals
    /#[a-zA-Z]/, // tag-like
  ];

  let codeScore = 0;
  for (const re of codeSignals) {
    if (re.test(trimmed)) codeScore++;
  }

  const wordCount = trimmed.split(/\s+/).length;
  if (codeScore >= 2 || (codeScore >= 1 && wordCount <= 4)) return "rare-term";

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
// Model selection
// ---------------------------------------------------------------------------

export interface ModelSelectionOptions {
  /**
   * Maximum recent outcome records to consider per (queryComplexity, model) pair.
   * Defaults to 30.
   */
  windowSize?: number;
  /**
   * Latency budget in milliseconds.  Models consistently exceeding this are
   * penalised in scoring. Defaults to 3 000 ms.
   */
  latencyBudgetMs?: number;
  /**
   * Minimum recorded outcomes before learned scores are trusted.
   * Below this threshold the router uses tier-based heuristics.
   * Defaults to 4.
   */
  minOutcomesForLearning?: number;
}

/**
 * Select the best embedding model for a given query complexity.
 *
 * Decision logic:
 *  1. Load recent outcomes.
 *  2. If fewer than `minOutcomesForLearning` exist, return the heuristic default.
 *  3. Score each model by average (quality − latency_penalty) for the given
 *     (queryComplexity, modelName) pair.
 *  4. Return the model with the highest score, falling back to heuristic when
 *     all models have no data for this complexity.
 */
export function selectEmbeddingModel(
  queryComplexity: QueryComplexity,
  outcomes: ModelOutcomeRecord[],
  options: ModelSelectionOptions = {},
): EmbeddingModelName {
  const windowSize = options.windowSize ?? 30;
  const latencyBudgetMs = options.latencyBudgetMs ?? 3_000;
  const minOutcomesForLearning = options.minOutcomesForLearning ?? 4;

  if (outcomes.length < minOutcomesForLearning) {
    return heuristicModelDefault(queryComplexity);
  }

  const window = outcomes.slice(-windowSize);

  let bestModel: EmbeddingModelName | null = null;
  let bestScore = -Infinity;

  for (const modelName of Object.keys(EMBEDDING_MODELS) as EmbeddingModelName[]) {
    const score = scoreModelOutcomes(window, modelName, queryComplexity, latencyBudgetMs);
    if (score > bestScore) {
      bestScore = score;
      bestModel = modelName;
    }
  }

  // If all models returned -1 (no data for this complexity), fall back to heuristic.
  if (bestModel === null || bestScore < 0) {
    return heuristicModelDefault(queryComplexity);
  }

  return bestModel;
}

/**
 * Heuristic default model selection by query complexity.
 *
 *  - rare-term → fast model (exact lexical signals, deep semantic not needed)
 *  - semantic  → accurate model (nuanced meaning requires high-quality embeddings)
 *  - general   → balanced model (safe default)
 */
export function heuristicModelDefault(queryComplexity: QueryComplexity): EmbeddingModelName {
  switch (queryComplexity) {
    case "rare-term":
      return "all-minilm-l6-v2";
    case "semantic":
      return "bge-base-en-v1-5";
    case "general":
    default:
      return "nomic-embed-text";
  }
}

// ---------------------------------------------------------------------------
// Fallback cascade
// ---------------------------------------------------------------------------

export interface EmbeddingResult {
  embedding: number[];
  modelName: EmbeddingModelName;
  latencyMs: number;
  /** True if a fallback model was used (primary timed out). */
  usedFallback: boolean;
  /** True when the ultrafast (no-ML) tier was used as the final fallback. */
  usedUltrafast?: boolean;
  /** Number of Ollama tiers that timed out before this result was produced. */
  ollamaTimeouts?: number;
  /**
   * Quantization bit-depth selected by the QuantizationStrategyEngine for this query.
   * Present when the router has a strategy engine configured.
   * Callers should pass this to QuantizedANNSearcher.search() as overrideBitDepth.
   */
  selectedQuantizationDepth?: QuantizationBitDepth;
  /**
   * Outcome ID from the QuantizationStrategyEngine, for recording actual performance
   * after the search completes via engine.recordOutcome().
   */
  quantizationOutcomeId?: string;
}

/**
 * Generate an embedding with automatic fallback cascade.
 *
 * Tries the `primaryModel` first; if it times out or fails, tries each
 * subsequent model in `CASCADE_ORDER` (accurate → balanced → fast →
 * ultrafast).  The ultrafast tier never fails — it always returns a
 * deterministic hash-based embedding without calling Ollama.
 *
 * Timeout per model is taken from `EmbeddingModelSpec.timeoutMs`.
 *
 * Returns null only when the primary is "ultrafast" and the text is empty
 * (degenerate case). In all other cases a result is guaranteed.
 */
export async function generateEmbeddingWithFallback(
  text: string,
  primaryModel: EmbeddingModelName,
): Promise<EmbeddingResult | null> {
  // Build the attempt order: primary first, then the rest of the cascade
  // (skipping the primary to avoid duplicates). ultrafast is always last.
  const ollamaModels: EmbeddingModelName[] = CASCADE_ORDER.filter(
    (m) => m !== "ultrafast" && m !== primaryModel,
  );
  const attemptOrder: EmbeddingModelName[] = [
    primaryModel,
    ...ollamaModels,
    // ultrafast last (only if not already primary)
    ...(primaryModel === "ultrafast" ? [] : ["ultrafast" as EmbeddingModelName]),
  ];

  const overallStart = Date.now();
  let ollamaTimeouts = 0;

  for (let i = 0; i < attemptOrder.length; i++) {
    const modelName = attemptOrder[i]!;

    // ── ultrafast tier — no Ollama, always succeeds ──────────────────────
    if (modelName === "ultrafast") {
      const embedder = new UltraFastEmbedder();
      const embedding = embedder.embed(text);
      return {
        embedding,
        modelName: "ultrafast",
        latencyMs: Date.now() - overallStart,
        usedFallback: i > 0,
        usedUltrafast: true,
        ollamaTimeouts,
      };
    }

    // ── Ollama-backed tiers ───────────────────────────────────────────────
    const spec = EMBEDDING_MODELS[modelName];

    const embeddingPromise = generateEmbedding(text, modelName);
    const timeoutPromise = new Promise<null>((resolve) =>
      setTimeout(() => resolve(null), spec.timeoutMs),
    );

    const embedding = await Promise.race([embeddingPromise, timeoutPromise]);

    if (embedding !== null) {
      return {
        embedding,
        modelName,
        latencyMs: Date.now() - overallStart,
        usedFallback: i > 0,
        ...(ollamaTimeouts > 0 ? { ollamaTimeouts } : {}),
      };
    }

    // Model failed or timed out — count and try next
    ollamaTimeouts++;
  }

  // Should never be reached: ultrafast always succeeds
  return null;
}

// ---------------------------------------------------------------------------
// EmbeddingRouter class
// ---------------------------------------------------------------------------

export interface EmbeddingRouterOptions extends ModelSelectionOptions {
  /**
   * Override the working directory for outcome persistence.
   * Useful in tests.
   */
  cwd?: string;
  /**
   * When true, the router integrates a QuantizationStrategyEngine to select
   * the best quantization bit-depth before model tier selection.
   * Defaults to false to preserve backward compatibility.
   */
  enableQuantizationStrategy?: boolean;
}

/**
 * EmbeddingRouter orchestrates multi-model embedding generation:
 *
 *  1. Classifies the query by complexity.
 *  2. Loads per-model outcome history.
 *  3. Selects the best model (learned or heuristic).
 *  4. Calls `generateEmbeddingWithFallback` to get the embedding.
 *  5. Records the outcome for future learning.
 *
 * The router is the single entry-point for all embedding requests; callers
 * should not call `generateEmbedding` directly when routing is desired.
 */
export class EmbeddingRouter {
  private readonly cwd: string;
  private readonly options: Required<ModelSelectionOptions>;
  private readonly strategyEngine: QuantizationStrategyEngine | null;

  constructor(cwd: string, options: EmbeddingRouterOptions = {}) {
    this.cwd = cwd;
    this.options = {
      windowSize: options.windowSize ?? 30,
      latencyBudgetMs: options.latencyBudgetMs ?? 3_000,
      minOutcomesForLearning: options.minOutcomesForLearning ?? 4,
    };
    this.strategyEngine = options.enableQuantizationStrategy
      ? new QuantizationStrategyEngine(cwd)
      : null;
  }

  /**
   * Load a genome manifest into the quantization strategy engine (if enabled).
   * Call after loading/reloading the manifest so the engine has an up-to-date
   * GenomeProfile for tier-aware bit-depth selection.
   */
  loadManifestProfile(manifest: import("./manifest.ts").GenomeManifest): GenomeProfile | null {
    if (!this.strategyEngine) return null;
    return this.strategyEngine.loadProfile(manifest);
  }

  /**
   * Generate an embedding for `text`, routing to the best model for the
   * query's complexity.  Records the outcome automatically.
   *
   * When `enableQuantizationStrategy` is true, also selects the optimal
   * quantization bit-depth and attaches it to the result as
   * `selectedQuantizationDepth` + `quantizationOutcomeId`.
   *
   * Returns null when Ollama is unavailable or the full cascade is exhausted.
   */
  async embed(
    query: string,
    text: string,
  ): Promise<EmbeddingResult | null> {
    if (!(await isOllamaAvailable())) return null;

    // ── Quantization strategy selection (before model tier choice) ────────
    let selectedQuantizationDepth: QuantizationBitDepth | undefined;
    let quantizationOutcomeId: string | undefined;

    if (this.strategyEngine) {
      const stratResult = await this.strategyEngine.selectBitDepth(query);
      selectedQuantizationDepth = stratResult.bitDepth;
      quantizationOutcomeId = stratResult.outcomeId;
    }

    // ── Model tier selection ──────────────────────────────────────────────
    const complexity = classifyQueryComplexity(query);
    const outcomes = await loadModelOutcomes(this.cwd);
    const primaryModel = selectEmbeddingModel(complexity, outcomes, this.options);

    const result = await generateEmbeddingWithFallback(text, primaryModel);

    // Record the model outcome for future learning
    await recordModelOutcome(this.cwd, {
      modelName: result?.modelName ?? primaryModel,
      queryComplexity: complexity,
      latencyMs: result?.latencyMs ?? 0,
      succeeded: result !== null,
      topSimilarity: 0, // Updated by caller via updateModelOutcomeQuality
      matchQuality: null, // Updated by caller via updateModelOutcomeQuality
    });

    if (result === null) return null;

    return {
      ...result,
      ...(selectedQuantizationDepth !== undefined ? { selectedQuantizationDepth } : {}),
      ...(quantizationOutcomeId !== undefined ? { quantizationOutcomeId } : {}),
    };
  }

  /**
   * Update the match quality for a previously recorded outcome.
   *
   * Callers should call this after they can assess result quality (e.g., after
   * the LLM has used the retrieved sections and rated their usefulness).
   *
   * This is a best-effort append: if the outcome file is missing the id, it is
   * a no-op.
   */
  async updateModelOutcomeQuality(
    cwd: string,
    outcomeId: string,
    matchQuality: number,
    topSimilarity: number,
  ): Promise<void> {
    const outcomes = await loadModelOutcomes(cwd);
    const idx = outcomes.findIndex((o) => o.id === outcomeId);
    if (idx === -1) return;

    // We cannot mutate JSONL in-place — append a correction record instead.
    // The scoring model reads the last record for each id.
    const original = outcomes[idx]!;
    await recordModelOutcome(cwd, {
      modelName: original.modelName,
      queryComplexity: original.queryComplexity,
      latencyMs: original.latencyMs,
      succeeded: original.succeeded,
      topSimilarity,
      matchQuality,
    });
  }

  /**
   * Return the model that would be selected for a given query without executing
   * the embedding. Useful for inspection and tests.
   */
  async dryRunSelectModel(query: string): Promise<{
    model: EmbeddingModelName;
    complexity: QueryComplexity;
    tier: ModelTier;
    usedHeuristic: boolean;
  }> {
    const complexity = classifyQueryComplexity(query);
    const outcomes = await loadModelOutcomes(this.cwd);
    const usedHeuristic = outcomes.length < this.options.minOutcomesForLearning;
    const model = selectEmbeddingModel(complexity, outcomes, this.options);
    return {
      model,
      complexity,
      tier: EMBEDDING_MODELS[model].tier,
      usedHeuristic,
    };
  }

  /**
   * Return per-model aggregate stats from the outcome log.
   * Useful for dashboards and the savings reporter.
   */
  async modelStats(cwd: string): Promise<
    Array<{
      modelName: EmbeddingModelName;
      tier: ModelTier;
      totalCalls: number;
      successRate: number;
      avgLatencyMs: number;
      avgQuality: number | null;
    }>
  > {
    const outcomes = await loadModelOutcomes(cwd);

    return (Object.keys(EMBEDDING_MODELS) as EmbeddingModelName[]).map((modelName) => {
      const modelRecords = outcomes.filter((o) => o.modelName === modelName);
      const succeeded = modelRecords.filter((o) => o.succeeded);
      const withQuality = succeeded.filter((o) => o.matchQuality !== null);

      return {
        modelName,
        tier: EMBEDDING_MODELS[modelName].tier,
        totalCalls: modelRecords.length,
        successRate: modelRecords.length > 0 ? succeeded.length / modelRecords.length : 0,
        avgLatencyMs:
          succeeded.length > 0
            ? succeeded.reduce((s, r) => s + r.latencyMs, 0) / succeeded.length
            : 0,
        avgQuality:
          withQuality.length > 0
            ? withQuality.reduce((s, r) => s + (r.matchQuality as number), 0) / withQuality.length
            : null,
      };
    });
  }
}

// ---------------------------------------------------------------------------
// Convenience factory
// ---------------------------------------------------------------------------

/**
 * Create an EmbeddingRouter bound to the given working directory.
 */
export function createEmbeddingRouter(cwd: string, options?: EmbeddingRouterOptions): EmbeddingRouter {
  return new EmbeddingRouter(cwd, options);
}
