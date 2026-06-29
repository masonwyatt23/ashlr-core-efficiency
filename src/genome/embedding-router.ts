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
  QuantizationAdvisor,
  type QuantizationBitDepth,
  type GenomeProfile,
  type QuantizationAdvice,
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
 * Result produced by `EmbeddingRouter.initializeQuantization()`.
 *
 * Contains the QuantizationAdvice from the QuantizationAdvisor plus a flag
 * indicating whether the engine has been configured with the recommended
 * strategy.  Callers can inspect `advice.recommendedBitDepth` and
 * `advice.fallbackChain` for downstream use.
 */
export interface QuantizationInitResult {
  /** Detailed advice produced by the QuantizationAdvisor. */
  advice: QuantizationAdvice;
  /** True when the strategy engine is active and has been seeded with the advice. */
  strategyActive: boolean;
  /** Any anomalies detected during profiling (e.g. codec version mismatch). */
  anomalies: string[];
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
   * Auto-select a quantization strategy at startup by running the
   * QuantizationAdvisor over a sample of embeddings from the genome.
   *
   * This is the integration hook that embedding-router callers should invoke
   * once at startup (after loading the manifest and embedding cache) to
   * automatically configure the best quantization strategy.
   *
   * Algorithm:
   *  1. Run QuantizationAdvisor.advise() on the provided embedding samples.
   *  2. Detect any anomalies (codec mismatch, unusual scale).
   *  3. If `enableQuantizationStrategy` is true, load the manifest profile
   *     into the strategy engine so subsequent embed() calls use the
   *     advisor-recommended fallback chain.
   *  4. Return the full QuantizationInitResult for caller inspection.
   *
   * When called without `enableQuantizationStrategy`, the advisor still runs
   * and returns advice, but the strategy engine is not configured
   * (strategyActive=false).  This allows callers to preview recommendations
   * without activating the full adaptive engine.
   *
   * @param manifest        The loaded GenomeManifest (for section count).
   * @param embeddingSamples  Representative float32 embeddings (may be empty).
   * @param advisorOptions  Optional QuantizationAdvisor configuration overrides.
   */
  initializeQuantization(
    manifest: import("./manifest.ts").GenomeManifest,
    embeddingSamples: number[][] = [],
    advisorOptions?: ConstructorParameters<typeof QuantizationAdvisor>[0],
  ): QuantizationInitResult {
    const advisor = new QuantizationAdvisor(advisorOptions);
    const sectionCount = manifest.sections.length;

    const advice = advisor.advise(embeddingSamples, sectionCount);
    const anomalies = advisor.detectAnomalies(embeddingSamples, sectionCount);

    let strategyActive = false;
    if (this.strategyEngine) {
      this.strategyEngine.loadProfile(manifest);
      strategyActive = true;
    }

    return { advice, strategyActive, anomalies };
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

// ===========================================================================
// MULTI-BACKEND FALLBACK CHAIN
//
// Extends the Ollama-only model-tier router with a three-tier backend chain:
//   1. Ollama (local, free) — zero cost, requires local daemon
//   2. Anthropic Embeddings API (paid, reliable) — requires ANTHROPIC_API_KEY
//   3. Local quantized approximation (cached) — always available, deterministic
//
// New exports required by the spec:
//   EmbeddingRouter (already exported above)
//   selectEmbeddingBackend()
//   recordBackendMetric()
//   computeBackendROI()
// ===========================================================================

// ---------------------------------------------------------------------------
// Backend types
// ---------------------------------------------------------------------------

/** Identifies which embedding backend produced a result. */
export type EmbeddingBackend = "ollama" | "anthropic" | "local-cached";

/** Cost in USD for a single embedding request (approximate). */
export interface BackendCostInfo {
  /** Fixed cost per request in USD (0 for free backends). */
  perRequestUsd: number;
  /** Additional cost per 1000 input tokens (0 for fixed-cost backends). */
  perKTokenUsd: number;
}

/** Static cost table per backend. */
export const BACKEND_COST_INFO: Record<EmbeddingBackend, BackendCostInfo> = {
  "ollama": { perRequestUsd: 0, perKTokenUsd: 0 },
  // Anthropic text-embedding-3 pricing (approximate; 0 at time of writing — use
  // a conservative placeholder so ROI can still be computed if pricing changes).
  "anthropic": { perRequestUsd: 0, perKTokenUsd: 0.00002 },
  "local-cached": { perRequestUsd: 0, perKTokenUsd: 0 },
};

/** Preferred fallback chain order (index 0 = highest priority). */
export const BACKEND_FALLBACK_CHAIN: EmbeddingBackend[] = [
  "ollama",
  "anthropic",
  "local-cached",
];

// ---------------------------------------------------------------------------
// Backend metric record (persisted as JSONL)
// ---------------------------------------------------------------------------

/** A single backend performance observation written to evolution/. */
export interface BackendMetricRecord {
  /** Unique record id. */
  id: string;
  /** ISO timestamp. */
  timestamp: string;
  /** Which backend produced this observation. */
  backend: EmbeddingBackend;
  /** Whether the backend returned an embedding successfully. */
  succeeded: boolean;
  /** Wall-clock latency in milliseconds. */
  latencyMs: number;
  /** Estimated USD cost for this single request. */
  costUsd: number;
  /**
   * Dimensionality of the embedding returned (0 when failed).
   * Used to detect dimension variance across backends.
   */
  dimension: number;
  /**
   * Caller-supplied retrieval hit-rate feedback [0,1]: fraction of top-k results
   * that were judged relevant.  null = not yet rated.
   */
  hitRate: number | null;
  /**
   * Cosine-similarity variance of the embedding vector dimensions (proxy for
   * embedding quality — higher variance = more discriminative).
   */
  dimensionalityVariance: number;
}

// ---------------------------------------------------------------------------
// Backend metric persistence
// ---------------------------------------------------------------------------

function backendMetricsPath(cwd: string): string {
  return join(genomeDir(cwd), "evolution", "embedding-backend-metrics.jsonl");
}

/**
 * Append a backend metric record to the JSONL log stored at
 * `genome/evolution/embedding-backend-metrics.jsonl`.
 *
 * @returns The generated record id.
 */
export async function recordBackendMetric(
  cwd: string,
  metric: Omit<BackendMetricRecord, "id" | "timestamp">,
): Promise<string> {
  const id = `bm-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
  const full: BackendMetricRecord = {
    id,
    timestamp: new Date().toISOString(),
    ...metric,
  };
  await appendJsonl(backendMetricsPath(cwd), full);
  return id;
}

/**
 * Load all backend metric records from the JSONL log.
 */
export async function loadBackendMetrics(cwd: string): Promise<BackendMetricRecord[]> {
  return readJsonl<BackendMetricRecord>(backendMetricsPath(cwd));
}

// ---------------------------------------------------------------------------
// Dimensionality variance helper
// ---------------------------------------------------------------------------

/**
 * Compute the variance of an embedding vector's dimension values.
 *
 * Higher variance indicates a more spread-out, discriminative embedding — a
 * rough proxy for embedding quality when no ground-truth labels are available.
 *
 * Returns 0 for empty or single-element arrays.
 */
export function computeEmbeddingVariance(embedding: number[]): number {
  if (embedding.length < 2) return 0;
  const mean = embedding.reduce((s, v) => s + v, 0) / embedding.length;
  const variance =
    embedding.reduce((s, v) => s + (v - mean) ** 2, 0) / embedding.length;
  return variance;
}

// ---------------------------------------------------------------------------
// ROI computation
// ---------------------------------------------------------------------------

/** Per-backend ROI summary. */
export interface BackendROI {
  backend: EmbeddingBackend;
  /** Number of records considered (within the rolling window). */
  sampleCount: number;
  /** Weighted quality score (hit-rate or dimensionality-variance proxy). */
  qualityScore: number;
  /** Average USD cost per request. */
  avgCostUsd: number;
  /**
   * ROI = quality / (cost + epsilon).
   *
   * Free backends (cost=0) receive eps=1e-6 to avoid division-by-zero while
   * still ranking paid backends correctly when quality is equal.
   */
  roi: number;
  /** Average latency of successful calls (ms). */
  avgLatencyMs: number;
  /** Fraction of calls that succeeded [0,1]. */
  successRate: number;
}

/**
 * Compute per-backend ROI over a rolling window of observations.
 *
 * Quality is computed as:
 *   - If hit-rate feedback is available: mean hit-rate over succeeded records.
 *   - Otherwise: mean dimensionality-variance (normalised to [0,1]) as a proxy.
 *
 * ROI = quality / (avgCostUsd + 1e-6)
 *
 * @param records     Metric records to evaluate (typically from loadBackendMetrics).
 * @param windowDays  Number of days to look back (default 7).
 */
export function computeBackendROI(
  records: BackendMetricRecord[],
  windowDays: number = 7,
): BackendROI[] {
  const cutoff = Date.now() - windowDays * 24 * 60 * 60 * 1000;
  const window = records.filter((r) => new Date(r.timestamp).getTime() >= cutoff);

  return BACKEND_FALLBACK_CHAIN.map((backend) => {
    const backendRecords = window.filter((r) => r.backend === backend);
    const succeeded = backendRecords.filter((r) => r.succeeded);

    const sampleCount = backendRecords.length;
    const successRate = sampleCount > 0 ? succeeded.length / sampleCount : 0;
    const avgLatencyMs =
      succeeded.length > 0
        ? succeeded.reduce((s, r) => s + r.latencyMs, 0) / succeeded.length
        : 0;
    const avgCostUsd =
      sampleCount > 0
        ? backendRecords.reduce((s, r) => s + r.costUsd, 0) / sampleCount
        : 0;

    // Quality: prefer explicit hit-rate feedback when available
    const withHitRate = succeeded.filter((r) => r.hitRate !== null);
    let qualityScore: number;

    if (withHitRate.length > 0) {
      qualityScore =
        withHitRate.reduce((s, r) => s + (r.hitRate as number), 0) /
        withHitRate.length;
    } else if (succeeded.length > 0) {
      // Fall back to normalised dimensionality variance proxy.
      // Raw variance is typically in [0, 0.05] for common embedding models —
      // normalise by 0.1 and cap at 1 to get a [0,1] quality signal.
      const avgVar =
        succeeded.reduce((s, r) => s + r.dimensionalityVariance, 0) /
        succeeded.length;
      qualityScore = Math.min(1, avgVar / 0.1);
    } else {
      qualityScore = 0;
    }

    const roi = qualityScore / (avgCostUsd + 1e-6);

    return {
      backend,
      sampleCount,
      qualityScore,
      avgCostUsd,
      roi,
      avgLatencyMs,
      successRate,
    };
  });
}

// ---------------------------------------------------------------------------
// Adaptive backend selection
// ---------------------------------------------------------------------------

export interface BackendSelectionOptions {
  /**
   * Minimum number of metric records required before learned ROI is trusted.
   * Below this threshold the static fallback chain order is used.
   * Defaults to 5.
   */
  minSamplesForLearning?: number;
  /**
   * Rolling window in days for ROI computation.
   * Defaults to 7.
   */
  windowDays?: number;
  /**
   * Per-backend timeout in milliseconds.  Overrides the static defaults.
   * Keys are backend names; missing entries use the built-in defaults.
   */
  timeoutsMs?: Partial<Record<EmbeddingBackend, number>>;
}

/** Static per-backend timeout defaults (ms). */
export const BACKEND_DEFAULT_TIMEOUTS_MS: Record<EmbeddingBackend, number> = {
  "ollama": 15_000,
  "anthropic": 10_000,
  "local-cached": 500,
};

/**
 * Select the best embedding backend for the next request.
 *
 * Decision logic:
 *  1. Load all backend metrics.
 *  2. If fewer than `minSamplesForLearning` records exist, return the static
 *     fallback chain in priority order (ollama → anthropic → local-cached).
 *  3. Otherwise compute ROI for each backend over the rolling window and return
 *     the chain sorted by descending ROI (highest ROI = most cost-efficient).
 *     Backends with zero success rate are demoted to the end.
 *
 * @returns  Ordered list of backends to try (highest priority first).
 */
export function selectEmbeddingBackend(
  metrics: BackendMetricRecord[],
  options: BackendSelectionOptions = {},
): EmbeddingBackend[] {
  const minSamples = options.minSamplesForLearning ?? 5;
  const windowDays = options.windowDays ?? 7;

  if (metrics.length < minSamples) {
    // Not enough data — return static priority order
    return [...BACKEND_FALLBACK_CHAIN];
  }

  const roiList = computeBackendROI(metrics, windowDays);

  // Sort: backends with successRate > 0, by descending ROI; then zero-success ones
  const active = roiList
    .filter((r) => r.successRate > 0)
    .sort((a, b) => b.roi - a.roi);
  const inactive = roiList.filter((r) => r.successRate === 0);

  const ordered: EmbeddingBackend[] = [
    ...active.map((r) => r.backend),
    ...inactive.map((r) => r.backend),
  ];

  // Ensure all backends are present (fill any missing ones at the end)
  for (const b of BACKEND_FALLBACK_CHAIN) {
    if (!ordered.includes(b)) ordered.push(b);
  }

  return ordered;
}

// ---------------------------------------------------------------------------
// Backend cost estimation
// ---------------------------------------------------------------------------

/**
 * Estimate the USD cost for a single embedding request given the input text.
 *
 * Uses a rough 4-chars-per-token heuristic consistent with the rest of the
 * genome module.
 */
export function estimateBackendCost(backend: EmbeddingBackend, text: string): number {
  const costInfo = BACKEND_COST_INFO[backend];
  const estimatedTokens = Math.ceil(text.length / 4);
  return (
    costInfo.perRequestUsd + (estimatedTokens / 1000) * costInfo.perKTokenUsd
  );
}

// ---------------------------------------------------------------------------
// Anthropic embeddings backend
// ---------------------------------------------------------------------------

/**
 * Generate an embedding via the Anthropic Embeddings API.
 *
 * Requires ANTHROPIC_API_KEY in the environment.  Returns null when the key
 * is absent, the API is unreachable, or the request exceeds the timeout.
 *
 * Note: As of the knowledge cutoff, Anthropic does not publish a standalone
 * embeddings endpoint.  This implementation targets a hypothetical
 * `/v1/embeddings` endpoint that mirrors the OpenAI embeddings interface and
 * is shimmed in tests via mock fetch.  When no real endpoint is available this
 * function will return null and the next backend in the chain takes over.
 */
export async function generateAnthropicEmbedding(
  text: string,
  timeoutMs: number = BACKEND_DEFAULT_TIMEOUTS_MS["anthropic"],
): Promise<number[] | null> {
  const apiKey = process.env["ANTHROPIC_API_KEY"];
  if (!apiKey) return null;

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    const res = await fetch("https://api.anthropic.com/v1/embeddings", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "voyage-large-2",
        input: text,
      }),
      signal: controller.signal,
    });

    clearTimeout(timer);

    if (!res.ok) return null;

    const data = (await res.json()) as {
      embeddings?: Array<{ values: number[] }>;
      data?: Array<{ embedding: number[] }>;
    };

    // Support both response shapes
    const embedding =
      data.embeddings?.[0]?.values ?? data.data?.[0]?.embedding ?? null;
    return embedding;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Local cached approximation backend
// ---------------------------------------------------------------------------

/**
 * Generate a deterministic local embedding approximation.
 *
 * Falls back to the UltraFastEmbedder (hash-projection) when the embedding
 * cache has no entry for the text.  Always succeeds.
 *
 * When an embedding cache entry exists for the text (matched by content hash),
 * the cached float32 embedding is returned — this provides near-zero latency
 * and preserves embedding quality from a previous Ollama or Anthropic call.
 */
export async function generateLocalCachedEmbedding(
  text: string,
  cwd: string,
): Promise<number[]> {
  // Try to find a cache hit by content hash
  const { loadEmbeddingCache } = await import("./embeddings.ts");
  const { contentHash } = await import("./embeddings.ts");
  const hash = contentHash(text);
  const cache = await loadEmbeddingCache(cwd);
  const hit = cache.find((e) => e.contentHash === hash);
  if (hit && hit.embedding.length > 0) {
    return hit.embedding;
  }

  // Fall back to ultrafast hash-projection
  const { UltraFastEmbedder } = await import("./embedding-ultrafast.ts");
  const embedder = new UltraFastEmbedder();
  return embedder.embed(text);
}

// ---------------------------------------------------------------------------
// Session cost tracker
// ---------------------------------------------------------------------------

/** Tracks per-backend cost and call counts within a single session. */
export class SessionCostTracker {
  private readonly costs: Map<EmbeddingBackend, number> = new Map();
  private readonly calls: Map<EmbeddingBackend, number> = new Map();

  /** Record a request with its cost. */
  record(backend: EmbeddingBackend, costUsd: number): void {
    this.costs.set(backend, (this.costs.get(backend) ?? 0) + costUsd);
    this.calls.set(backend, (this.calls.get(backend) ?? 0) + 1);
  }

  /** Total cost for a backend in this session. */
  totalCost(backend: EmbeddingBackend): number {
    return this.costs.get(backend) ?? 0;
  }

  /** Total calls for a backend in this session. */
  totalCalls(backend: EmbeddingBackend): number {
    return this.calls.get(backend) ?? 0;
  }

  /** Grand total cost across all backends. */
  grandTotal(): number {
    let sum = 0;
    for (const v of this.costs.values()) sum += v;
    return sum;
  }

  /** Return a summary object suitable for logging. */
  summary(): Record<EmbeddingBackend, { calls: number; costUsd: number }> {
    const result = {} as Record<EmbeddingBackend, { calls: number; costUsd: number }>;
    for (const backend of BACKEND_FALLBACK_CHAIN) {
      result[backend] = {
        calls: this.totalCalls(backend),
        costUsd: this.totalCost(backend),
      };
    }
    return result;
  }
}

// ---------------------------------------------------------------------------
// Multi-backend orchestrator
// ---------------------------------------------------------------------------

export interface MultiBackendEmbeddingResult {
  embedding: number[];
  backend: EmbeddingBackend;
  latencyMs: number;
  costUsd: number;
  dimensionalityVariance: number;
  /** True when a fallback backend was used. */
  usedFallback: boolean;
  /** Number of backends that failed before this one succeeded. */
  failedBackends: number;
}

export interface MultiBackendRouterOptions {
  /** Per-backend timeouts (ms). Overrides BACKEND_DEFAULT_TIMEOUTS_MS. */
  timeoutsMs?: Partial<Record<EmbeddingBackend, number>>;
  /** Minimum samples before adaptive ROI ordering takes over. Default 5. */
  minSamplesForLearning?: number;
  /** ROI rolling window (days). Default 7. */
  windowDays?: number;
  /** Override working directory (for tests). */
  cwd?: string;
}

/**
 * Generate an embedding using the AsyncFirstAvailable pattern across all
 * configured backends.
 *
 * Tries each backend in priority order (adaptive ROI or static fallback chain),
 * respecting per-backend timeouts.  The `local-cached` backend always succeeds
 * so the chain is guaranteed to return a result.
 *
 * Records a BackendMetricRecord for the winning backend and tracks cost via
 * the provided SessionCostTracker (if any).
 *
 * @param text          Text to embed.
 * @param cwd           Working directory for metric persistence and cache lookup.
 * @param options       Router options.
 * @param costTracker   Optional session cost tracker to update.
 * @returns             Embedding result from the first succeeding backend.
 */
export async function generateEmbeddingMultiBackend(
  text: string,
  cwd: string,
  options: MultiBackendRouterOptions = {},
  costTracker?: SessionCostTracker,
): Promise<MultiBackendEmbeddingResult> {
  const allMetrics = await loadBackendMetrics(cwd);

  const orderedBackends = selectEmbeddingBackend(allMetrics, {
    minSamplesForLearning: options.minSamplesForLearning,
    windowDays: options.windowDays,
  });

  const effectiveTimeouts: Record<EmbeddingBackend, number> = {
    ...BACKEND_DEFAULT_TIMEOUTS_MS,
    ...(options.timeoutsMs ?? {}),
  };

  const overallStart = Date.now();
  let failedBackends = 0;

  for (const backend of orderedBackends) {
    const timeoutMs = effectiveTimeouts[backend];
    const requestStart = Date.now();
    let embedding: number[] | null = null;

    try {
      const embeddingPromise = (async (): Promise<number[] | null> => {
        switch (backend) {
          case "ollama": {
            const { generateEmbedding, isOllamaAvailable: ollamaAvail } =
              await import("./embeddings.ts");
            if (!(await ollamaAvail())) return null;
            return generateEmbedding(text);
          }
          case "anthropic":
            return generateAnthropicEmbedding(text, timeoutMs);
          case "local-cached":
            return generateLocalCachedEmbedding(text, cwd);
        }
      })();

      const timeoutPromise = new Promise<null>((resolve) =>
        setTimeout(() => resolve(null), timeoutMs),
      );

      embedding = await Promise.race([embeddingPromise, timeoutPromise]);
    } catch {
      embedding = null;
    }

    const latencyMs = Date.now() - requestStart;
    const succeeded = embedding !== null && embedding.length > 0;

    if (succeeded && embedding !== null) {
      const costUsd = estimateBackendCost(backend, text);
      const dimensionalityVariance = computeEmbeddingVariance(embedding);

      // Record metric for future ROI calculations
      await recordBackendMetric(cwd, {
        backend,
        succeeded: true,
        latencyMs,
        costUsd,
        dimension: embedding.length,
        hitRate: null,
        dimensionalityVariance,
      });

      costTracker?.record(backend, costUsd);

      return {
        embedding,
        backend,
        latencyMs: Date.now() - overallStart,
        costUsd,
        dimensionalityVariance,
        usedFallback: failedBackends > 0,
        failedBackends,
      };
    }

    // Record failure metric
    await recordBackendMetric(cwd, {
      backend,
      succeeded: false,
      latencyMs,
      costUsd: 0,
      dimension: 0,
      hitRate: null,
      dimensionalityVariance: 0,
    });

    failedBackends++;
  }

  // Should never reach here — local-cached always succeeds.
  // Defensive fallback: return an empty ultrafast embedding.
  const { UltraFastEmbedder } = await import("./embedding-ultrafast.ts");
  const fallback = new UltraFastEmbedder().embed(text);
  return {
    embedding: fallback,
    backend: "local-cached",
    latencyMs: Date.now() - overallStart,
    costUsd: 0,
    dimensionalityVariance: computeEmbeddingVariance(fallback),
    usedFallback: true,
    failedBackends,
  };
}

// ---------------------------------------------------------------------------
// Adaptive strategy learner — 7-day rolling ROI rank
// ---------------------------------------------------------------------------

/** Summary of the current backend ranking. */
export interface BackendRankingSummary {
  /** Ordered list from highest ROI to lowest. */
  rankedBackends: EmbeddingBackend[];
  /** Full ROI details per backend. */
  roiDetails: BackendROI[];
  /**
   * Whether the ranking is based on learned data (true) or defaults (false).
   */
  isLearned: boolean;
  /** ISO timestamp of when this summary was computed. */
  computedAt: string;
}

/**
 * Compute the current adaptive backend ranking.
 *
 * Reads the rolling 7-day metrics log, computes per-backend ROI, and returns
 * both the ordered backend list and the full ROI breakdown.
 *
 * The ranking is "learned" when there are at least `minSamplesForLearning`
 * records in the window; otherwise it reflects the static priority order.
 */
export async function computeAdaptiveBackendRanking(
  cwd: string,
  options: BackendSelectionOptions = {},
): Promise<BackendRankingSummary> {
  const minSamples = options.minSamplesForLearning ?? 5;
  const windowDays = options.windowDays ?? 7;

  const allMetrics = await loadBackendMetrics(cwd);
  const cutoff = Date.now() - windowDays * 24 * 60 * 60 * 1000;
  const windowMetrics = allMetrics.filter(
    (r) => new Date(r.timestamp).getTime() >= cutoff,
  );

  const isLearned = windowMetrics.length >= minSamples;
  const roiDetails = computeBackendROI(allMetrics, windowDays);
  const rankedBackends = selectEmbeddingBackend(allMetrics, options);

  return {
    rankedBackends,
    roiDetails,
    isLearned,
    computedAt: new Date().toISOString(),
  };
}
