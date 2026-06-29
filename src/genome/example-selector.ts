/**
 * Adaptive Few-Shot Example Selector — learner-driven, model-agnostic.
 *
 * Three public classes + three module-level helpers:
 *
 *   ExampleSelector        — ranks candidate examples and selects top-K within budget
 *   FewShotInjector        — formats examples into system prompt or user message prefix
 *   OutcomeTracker         — records post-call outcomes and updates success_rate stats
 *   SectionFitnessPredictor — cross-session genome fitness predictor with topic-cluster EMA
 *
 * Module helpers (re-exported from this file):
 *   selectExamples(taskType, provider, model, tokenBudget, cwd?)
 *   injectExamples(prompt, examples, budget?)
 *   recordOutcome(taskType, examples, result, provider)
 *   rerankByFitness(candidates, queryTopic, cwd) — re-rank by combined semantic×utility score
 *
 * No LLM calls are made anywhere in this module.
 */

import { existsSync } from "fs";
import { appendFile, mkdir, readFile, rename, writeFile } from "fs/promises";
import { dirname, join } from "path";
import { homedir } from "os";
import {
  aggregateProfiles,
  createProfile,
  emaSuccessRate,
  loadProfiles,
  loadProfilesFromPath,
  persistProfile,
  rankProfiles,
  type ExampleProfile,
  type RankedExample,
  type TaskType,
} from "./example-profiler.ts";
import { cosineSimilarity } from "./embeddings.ts";

export type { TaskType, ExampleProfile, RankedExample };

// ---------------------------------------------------------------------------
// Example content type
// ---------------------------------------------------------------------------

/**
 * A single few-shot example to be injected into a prompt.
 */
export interface FewShotExample {
  /** Stable ID matching ExampleProfile.example_id */
  id: string;
  /** Human-readable label shown in the prompt header */
  label: string;
  /** The example content (code, tool call, error + fix, etc.) */
  content: string;
  /** Content type hint for formatting */
  kind: "code" | "tool_call" | "error_recovery" | "text";
  /** Language for code blocks (e.g. "typescript", "python") */
  language?: string;
  /**
   * Pre-computed embedding for relevance ranking.
   * When absent, ranking falls back to success_rate only.
   */
  embedding?: number[];
  /** Approximate token cost (computed lazily if not provided) */
  token_cost?: number;
}

// ---------------------------------------------------------------------------
// Token estimation (lightweight, no tiktoken dependency in hot path)
// ---------------------------------------------------------------------------

/**
 * Fast token estimate: ~4 chars per token heuristic.
 * Good enough for budget gating; not billed for.
 */
export function estimateExampleTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

// ---------------------------------------------------------------------------
// ExampleSelector
// ---------------------------------------------------------------------------

export interface ExampleSelectorOptions {
  /**
   * Top-K examples to return (default: 3).
   * The selector may return fewer if budget is tight.
   */
  topK?: number;
  /**
   * Minimum success_rate threshold to include an example (default: 0.0).
   * Set to e.g. 0.4 to skip examples that consistently hurt quality.
   */
  minSuccessRate?: number;
  /**
   * Query text/embedding used for relevance scoring.
   * When absent, ranking is success_rate only.
   */
  queryEmbedding?: number[];
}

/**
 * ExampleSelector — maintains learned rankings per (task_type, provider, model).
 *
 * Usage:
 *   const selector = new ExampleSelector(taskType, provider, model);
 *   await selector.load(candidateExamples);
 *   const selected = selector.select(tokenBudget, options);
 */
export class ExampleSelector {
  private taskType: TaskType;
  private provider: string;
  private model: string;
  private ranked: RankedExample[] = [];
  private examples: Map<string, FewShotExample> = new Map();

  constructor(taskType: TaskType, provider: string, model: string) {
    this.taskType = taskType;
    this.provider = provider;
    this.model = model;
  }

  /**
   * Load candidate examples and their learned profiles from disk.
   *
   * Profiles are loaded from ~/.ashlr/examples-profile.jsonl.
   * For examples with no prior profile, a neutral prior (success_rate=0.5) is used.
   */
  async load(
    candidates: FewShotExample[],
    options: { queryEmbedding?: number[] } = {},
  ): Promise<void> {
    // Index examples
    this.examples.clear();
    for (const ex of candidates) {
      this.examples.set(ex.id, ex);
    }

    // Load and aggregate persisted profiles
    const allProfiles = await loadProfiles();
    const aggregated = aggregateProfiles(allProfiles);

    // Build profile list for ranking — one entry per candidate
    const profileList: ExampleProfile[] = candidates.map((ex) => {
      const key = `${this.taskType}::${this.provider}::${this.model}::${ex.id}`;
      const existing = aggregated.get(key);
      if (existing) return existing;

      // Cold-start: synthesise a neutral profile from the example itself
      return createProfile({
        task_type: this.taskType,
        provider: this.provider,
        model: this.model,
        example_id: ex.id,
        embedding: ex.embedding ?? [],
        token_cost: ex.token_cost ?? estimateExampleTokens(ex.content),
      });
    });

    this.ranked = rankProfiles(profileList, options.queryEmbedding ?? null);
  }

  /**
   * Select top-K examples that fit within tokenBudget.
   *
   * Returns examples in ranked order (best first).
   */
  select(tokenBudget: number, options: ExampleSelectorOptions = {}): FewShotExample[] {
    const { topK = 3, minSuccessRate = 0.0 } = options;

    const selected: FewShotExample[] = [];
    let usedTokens = 0;

    for (const ranked of this.ranked) {
      if (selected.length >= topK) break;
      if (ranked.profile.success_rate < minSuccessRate) continue;

      const ex = this.examples.get(ranked.profile.example_id);
      if (!ex) continue;

      const cost = ex.token_cost ?? estimateExampleTokens(ex.content);
      if (usedTokens + cost > tokenBudget) continue;

      selected.push(ex);
      usedTokens += cost;
    }

    return selected;
  }

  /** Expose current rankings for inspection / testing. */
  getRankings(): RankedExample[] {
    return [...this.ranked];
  }
}

// ---------------------------------------------------------------------------
// FewShotInjector
// ---------------------------------------------------------------------------

export interface InjectionResult {
  /** The modified system prompt with examples prepended */
  systemPrompt: string;
  /** Examples that were actually injected (may be fewer than supplied if budget hit) */
  injectedExamples: FewShotExample[];
  /** Tokens consumed by the injected examples */
  tokensUsed: number;
}

/**
 * FewShotInjector — formats few-shot examples into the system prompt.
 *
 * Examples are rendered as fenced code blocks (code/tool_call) or plain text
 * (error_recovery/text) under a `## Few-Shot Examples` section header.
 *
 * The formatter respects token budget and will stop adding examples once the
 * budget would be exceeded, even if more examples are provided.
 */
export class FewShotInjector {
  /**
   * Format a single example into a markdown block.
   */
  static formatExample(ex: FewShotExample): string {
    const header = `### Example: ${ex.label}`;
    switch (ex.kind) {
      case "code": {
        const lang = ex.language ?? "typescript";
        return `${header}\n\`\`\`${lang}\n${ex.content}\n\`\`\``;
      }
      case "tool_call": {
        return `${header}\n\`\`\`json\n${ex.content}\n\`\`\``;
      }
      case "error_recovery": {
        return `${header}\n${ex.content}`;
      }
      case "text":
      default: {
        return `${header}\n${ex.content}`;
      }
    }
  }

  /**
   * Inject examples into a system prompt.
   *
   * The examples section is prepended before the existing system prompt content
   * so it appears near the top of the context window (high cache ROI position).
   *
   * @param systemPrompt  Base system prompt (will not be mutated)
   * @param examples      Candidates to inject (already ranked)
   * @param tokenBudget   Max tokens to spend on examples (default: 2000)
   */
  static inject(
    systemPrompt: string,
    examples: FewShotExample[],
    tokenBudget = 2000,
  ): InjectionResult {
    if (examples.length === 0) {
      return { systemPrompt, injectedExamples: [], tokensUsed: 0 };
    }

    const blocks: string[] = [];
    const injectedExamples: FewShotExample[] = [];
    let tokensUsed = 0;

    for (const ex of examples) {
      const block = FewShotInjector.formatExample(ex);
      const cost = estimateExampleTokens(block);
      if (tokensUsed + cost > tokenBudget) continue;
      blocks.push(block);
      injectedExamples.push(ex);
      tokensUsed += cost;
    }

    if (blocks.length === 0) {
      return { systemPrompt, injectedExamples: [], tokensUsed: 0 };
    }

    const examplesSection =
      `## Few-Shot Examples\n\n` + blocks.join("\n\n") + `\n\n---\n\n`;

    return {
      systemPrompt: examplesSection + systemPrompt,
      injectedExamples,
      tokensUsed,
    };
  }

  /**
   * Inject examples as a user-message prefix instead of system prompt.
   *
   * Useful for models that don't support system prompts (e.g., some local models).
   */
  static formatAsUserPrefix(examples: FewShotExample[], tokenBudget = 2000): string {
    const blocks: string[] = [];
    let tokensUsed = 0;

    for (const ex of examples) {
      const block = FewShotInjector.formatExample(ex);
      const cost = estimateExampleTokens(block);
      if (tokensUsed + cost > tokenBudget) continue;
      blocks.push(block);
      tokensUsed += cost;
    }

    if (blocks.length === 0) return "";
    return `## Few-Shot Examples\n\n` + blocks.join("\n\n") + `\n\n---\n\n`;
  }
}

// ---------------------------------------------------------------------------
// OutcomeTracker
// ---------------------------------------------------------------------------

export interface LLMCallResult {
  /** Number of tool_use blocks in the response */
  tool_use_count: number;
  /** Number of error markers detected in the response */
  error_count: number;
  /** Total output tokens from the response */
  tokens_output: number;
  /** Whether the task completed successfully (caller-assessed) */
  task_completed: boolean;
  /** Wall-clock latency of the LLM call (ms) */
  latency_ms: number;
  /** Optional quality score (0-1) assigned by caller */
  quality_score?: number;
}

export interface OutcomeRecord {
  task_type: TaskType;
  provider: string;
  model: string;
  example_ids: string[];
  result: LLMCallResult;
  baseline_latency_ms?: number;
  recorded_at: string;
}

function outcomesPath(): string {
  return join(homedir(), ".ashlr", "example-outcomes.jsonl");
}

/**
 * OutcomeTracker — correlates LLM call results with the examples that were injected.
 *
 * After each LLM call, call `record()` with the examples that were injected and the
 * observed outcome. The tracker will:
 *  1. Persist the raw outcome to ~/.ashlr/example-outcomes.jsonl
 *  2. Update each example's ExampleProfile in ~/.ashlr/examples-profile.jsonl
 *     using EMA over success_rate and running mean over latency_impact_ms
 */
export class OutcomeTracker {
  private provider: string;
  private model: string;

  constructor(provider: string, model: string) {
    this.provider = provider;
    this.model = model;
  }

  /**
   * Record a call outcome and update example profiles.
   *
   * success_rate is derived from: task_completed + quality_score + low error_count.
   * latency_impact = actual_latency - baseline_latency (if baseline provided).
   */
  async record(
    taskType: TaskType,
    injectedExamples: FewShotExample[],
    result: LLMCallResult,
    baselineLatencyMs?: number,
  ): Promise<void> {
    if (injectedExamples.length === 0) return;

    // Derive success score (0-1)
    const successScore = deriveSuccessScore(result);

    // Persist raw outcome
    const record: OutcomeRecord = {
      task_type: taskType,
      provider: this.provider,
      model: this.model,
      example_ids: injectedExamples.map((e) => e.id),
      result,
      baseline_latency_ms: baselineLatencyMs,
      recorded_at: new Date().toISOString(),
    };
    await persistOutcome(record);

    // Update each example's profile
    const allProfiles = await loadProfiles();
    const aggregated = aggregateProfiles(allProfiles);

    for (const ex of injectedExamples) {
      const key = `${taskType}::${this.provider}::${this.model}::${ex.id}`;
      const existing = aggregated.get(key);

      const prevSuccessRate = existing?.success_rate ?? 0.5;
      const prevObs = existing?.observation_count ?? 0;
      const newObs = prevObs + 1;

      const updatedSuccessRate = emaSuccessRate(prevSuccessRate, successScore, newObs);
      const latencyImpact =
        baselineLatencyMs !== undefined
          ? result.latency_ms - baselineLatencyMs
          : existing?.latency_impact_ms ?? 0;
      const updatedLatency =
        prevObs === 0
          ? latencyImpact
          : (existing!.latency_impact_ms * prevObs + latencyImpact) / newObs;

      const updatedProfile = createProfile({
        task_type: taskType,
        provider: this.provider,
        model: this.model,
        example_id: ex.id,
        embedding: ex.embedding ?? existing?.embedding ?? [],
        token_cost: ex.token_cost ?? existing?.token_cost ?? estimateExampleTokens(ex.content),
        success_rate: updatedSuccessRate,
        latency_impact_ms: updatedLatency,
        updated_at: new Date().toISOString(),
        observation_count: newObs,
      });

      await persistProfile(updatedProfile);
    }
  }
}

// ---------------------------------------------------------------------------
// Outcome helpers
// ---------------------------------------------------------------------------

/**
 * Compute a 0-1 success score from an LLM call result.
 *
 * Weights:
 *   task_completed: 50%
 *   quality_score:  30% (or 0.5 if not provided)
 *   no errors:      20%
 */
export function deriveSuccessScore(result: LLMCallResult): number {
  const completionScore = result.task_completed ? 1.0 : 0.0;
  const qualityScore = result.quality_score ?? (result.task_completed ? 0.7 : 0.3);
  const errorScore = result.error_count === 0 ? 1.0 : Math.max(0, 1.0 - result.error_count * 0.2);
  return 0.5 * completionScore + 0.3 * qualityScore + 0.2 * errorScore;
}

async function persistOutcome(record: OutcomeRecord): Promise<void> {
  const path = outcomesPath();
  const dir = dirname(path);
  if (!existsSync(dir)) {
    await mkdir(dir, { recursive: true });
  }
  await appendFile(path, JSON.stringify(record) + "\n", "utf-8");
}

// ---------------------------------------------------------------------------
// Module-level helpers (the public API)
// ---------------------------------------------------------------------------

/**
 * Select top-K few-shot examples for a given context.
 *
 * Loads learned profiles, ranks candidates by success_rate × cosine_similarity,
 * and returns examples that fit within tokenBudget. No LLM calls.
 *
 * @param taskType    Task classification
 * @param provider    Provider name (e.g. "anthropic")
 * @param model       Model ID (e.g. "claude-opus-4-5")
 * @param candidates  Pool of candidate examples to select from
 * @param tokenBudget Max tokens to spend on examples
 * @param options     Extra options (topK, minSuccessRate, queryEmbedding)
 */
export async function selectExamples(
  taskType: TaskType,
  provider: string,
  model: string,
  candidates: FewShotExample[],
  tokenBudget: number,
  options: ExampleSelectorOptions & { queryEmbedding?: number[] } = {},
): Promise<FewShotExample[]> {
  const selector = new ExampleSelector(taskType, provider, model);
  await selector.load(candidates, { queryEmbedding: options.queryEmbedding });
  return selector.select(tokenBudget, options);
}

/**
 * Inject examples into a system prompt, respecting token budget.
 *
 * Thin wrapper around FewShotInjector.inject().
 */
export function injectExamples(
  systemPrompt: string,
  examples: FewShotExample[],
  tokenBudget = 2000,
): InjectionResult {
  return FewShotInjector.inject(systemPrompt, examples, tokenBudget);
}

/**
 * Record the outcome of an LLM call and update example profiles.
 *
 * Call after every LLM response to keep success_rate stats up to date.
 */
export async function recordOutcome(
  taskType: TaskType,
  injectedExamples: FewShotExample[],
  result: LLMCallResult,
  provider: string,
  model = "unknown",
  baselineLatencyMs?: number,
): Promise<void> {
  const tracker = new OutcomeTracker(provider, model);
  await tracker.record(taskType, injectedExamples, result, baselineLatencyMs);
}

// ---------------------------------------------------------------------------
// Cross-provider transferability helper
// ---------------------------------------------------------------------------

/**
 * Find examples that performed well on a *different* provider/model and may
 * transfer to the target (task_type matches, success_rate above threshold).
 *
 * Used to bootstrap cold-start rankings for new providers by leveraging
 * observations from similar providers.
 */
export async function findTransferableExamples(
  taskType: TaskType,
  targetProvider: string,
  targetModel: string,
  minSuccessRate = 0.6,
): Promise<ExampleProfile[]> {
  const allProfiles = await loadProfiles();
  const aggregated = aggregateProfiles(allProfiles);

  const transferable: ExampleProfile[] = [];
  for (const profile of aggregated.values()) {
    if (profile.task_type !== taskType) continue;
    if (profile.provider === targetProvider && profile.model === targetModel) continue;
    if (profile.success_rate < minSuccessRate) continue;
    transferable.push(profile);
  }

  return transferable.sort((a, b) => b.success_rate - a.success_rate);
}

// ---------------------------------------------------------------------------
// SectionFitnessPredictor — cross-session genome fitness predictor
// ---------------------------------------------------------------------------

/**
 * A single retrieval-utility event recorded for a genome section.
 */
export interface SectionUsageEvent {
  /** Query text that triggered the retrieval */
  query_topic: string;
  /** Genome section path (relative to genome dir) */
  section_id: string;
  /** Whether the retrieved section proved useful (true = used, false = ignored/discarded) */
  was_useful: boolean;
  /** ISO timestamp */
  recorded_at: string;
}

/**
 * Per-section utility score tracked by the predictor.
 *
 * Stored inside embeddings.json alongside the quantized embeddings so the
 * same cache file serves both semantic search and fitness prediction.
 */
export interface SectionUtilityScore {
  /** Genome section path (relative to genome dir) */
  section_id: string;
  /**
   * EMA utility per query-topic cluster index.
   * Key is the cluster index (0-based string), value is EMA in [0,1].
   */
  cluster_utility: Record<string, number>;
  /** Total number of usage events recorded */
  event_count: number;
  /** ISO timestamp of last update */
  updated_at: string;
}

/**
 * Persisted structure appended to embeddings.json under a top-level
 * "fitness_scores" key so existing embedding entries are untouched.
 */
export interface EmbeddingsCacheWithFitness {
  /** Original embeddings array (pass-through) */
  embeddings?: unknown[];
  /** Fitness utility scores keyed by section_id */
  fitness_scores?: Record<string, SectionUtilityScore>;
}

/** A candidate section ready for re-ranking. */
export interface FitnessCandidate {
  /** Section path (must match section_id in stored scores) */
  section_id: string;
  /** Semantic similarity score from the embedding search (0-1) */
  semantic_score: number;
}

/** Result after re-ranking by combined fitness. */
export interface RankedFitnessCandidate extends FitnessCandidate {
  /** utility_ema for the best-matching topic cluster (0-1) */
  utility_ema: number;
  /** Combined score: semantic_score × utility_ema (0-1) */
  combined_score: number;
}

// ---------------------------------------------------------------------------
// TF-IDF helpers for topic clustering
// ---------------------------------------------------------------------------

const STOP_WORDS = new Set([
  "a", "an", "the", "is", "in", "on", "at", "to", "for", "of", "and", "or",
  "but", "not", "it", "its", "this", "that", "with", "from", "by", "be",
  "as", "are", "was", "were", "has", "have", "had", "do", "did", "does",
  "can", "will", "would", "could", "should", "may", "might", "shall",
]);

/**
 * Tokenise a query into lowercase terms, removing stop words and short tokens.
 */
function tokeniseQuery(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s_-]/g, " ")
    .split(/\s+/)
    .map((t) => t.replace(/^[-_]+|[-_]+$/g, ""))
    .filter((t) => t.length > 2 && !STOP_WORDS.has(t));
}

/**
 * Build a TF-IDF term-frequency vector for a document (query string).
 * Returns a sparse map of term → tf value.
 */
function buildTFVector(tokens: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const t of tokens) {
    counts.set(t, (counts.get(t) ?? 0) + 1);
  }
  const tf = new Map<string, number>();
  for (const [term, count] of counts) {
    tf.set(term, count / tokens.length);
  }
  return tf;
}

/**
 * Cosine similarity between two sparse TF vectors.
 */
function sparseCosine(a: Map<string, number>, b: Map<string, number>): number {
  let dot = 0;
  let normA = 0;
  for (const [term, va] of a) {
    normA += va * va;
    const vb = b.get(term) ?? 0;
    dot += va * vb;
  }
  let normB = 0;
  for (const vb of b.values()) normB += vb * vb;
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

// ---------------------------------------------------------------------------
// Simple k-means clustering on TF-IDF vectors
// ---------------------------------------------------------------------------

const NUM_CLUSTERS = 7; // 5-10 per spec; 7 balances granularity vs. cold-start

/**
 * Assign a document to the nearest centroid using cosine similarity.
 * Returns the cluster index (0-based).
 */
function assignCluster(docVec: Map<string, number>, centroids: Map<string, number>[]): number {
  if (centroids.length === 0) return 0;
  let bestIdx = 0;
  let bestSim = -Infinity;
  for (let i = 0; i < centroids.length; i++) {
    const sim = sparseCosine(docVec, centroids[i]!);
    if (sim > bestSim) {
      bestSim = sim;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/**
 * Run k-means clustering on an array of TF-IDF vectors.
 *
 * Uses cosine similarity as the distance metric. Stops at convergence or
 * maxIter iterations. When the corpus is smaller than k, returns as many
 * clusters as there are documents (graceful degradation for small histories).
 *
 * Returns an array of centroid maps (one per cluster).
 */
function kMeansClusters(
  docs: Map<string, number>[],
  k: number,
  maxIter = 20,
): Map<string, number>[] {
  if (docs.length === 0) return [];
  const effectiveK = Math.min(k, docs.length);

  // Initialise centroids by picking evenly-spaced documents (deterministic).
  const step = Math.max(1, Math.floor(docs.length / effectiveK));
  let centroids: Map<string, number>[] = Array.from(
    { length: effectiveK },
    (_, i) => new Map(docs[Math.min(i * step, docs.length - 1)]!),
  );

  for (let iter = 0; iter < maxIter; iter++) {
    // Assignment step
    const clusters: Map<string, number>[][] = Array.from({ length: effectiveK }, () => []);
    for (const doc of docs) {
      const idx = assignCluster(doc, centroids);
      clusters[idx]!.push(doc);
    }

    // Update step: recompute centroids as mean of assigned docs
    const newCentroids: Map<string, number>[] = centroids.map((prev, i) => {
      const members = clusters[i]!;
      if (members.length === 0) return new Map(prev); // keep old centroid for empty cluster
      const merged = new Map<string, number>();
      for (const doc of members) {
        for (const [term, val] of doc) {
          merged.set(term, (merged.get(term) ?? 0) + val / members.length);
        }
      }
      return merged;
    });

    // Convergence check: all centroids unchanged
    let converged = true;
    for (let i = 0; i < effectiveK; i++) {
      if (sparseCosine(newCentroids[i]!, centroids[i]!) < 0.9999) {
        converged = false;
        break;
      }
    }
    centroids = newCentroids;
    if (converged) break;
  }

  return centroids;
}

// ---------------------------------------------------------------------------
// EMA constant
// ---------------------------------------------------------------------------

const UTILITY_EMA_ALPHA = 0.3;

/**
 * Update a utility EMA value given a new boolean observation.
 *
 * On the first observation the raw value (0 or 1) is used directly.
 * Subsequent observations apply alpha=0.3 EMA so recent outcomes carry
 * ~3× more weight than older ones while retaining full history.
 */
function updateUtilityEma(current: number, wasUseful: boolean, eventCount: number): number {
  const observation = wasUseful ? 1.0 : 0.0;
  if (eventCount <= 1) return observation;
  return UTILITY_EMA_ALPHA * observation + (1 - UTILITY_EMA_ALPHA) * current;
}

// ---------------------------------------------------------------------------
// Persistence helpers for fitness scores inside embeddings.json
// ---------------------------------------------------------------------------

function fitnessScorePath(cwd: string): string {
  // Mirrors the embeddings.json path structure used by the embeddings module.
  return join(cwd, ".ashlrcode", "genome", "evolution", "embeddings.json");
}

/**
 * Load fitness scores from the embeddings.json sidecar.
 *
 * The file may not exist yet (first run) or may be a plain array from the
 * old embeddings module — both cases are handled gracefully.
 */
async function loadFitnessScores(cwd: string): Promise<Record<string, SectionUtilityScore>> {
  const path = fitnessScorePath(cwd);
  if (!existsSync(path)) return {};
  try {
    const raw = await readFile(path, "utf-8");
    const parsed = JSON.parse(raw);
    // Old format: plain array of EmbeddingCache entries — no fitness data yet.
    if (Array.isArray(parsed)) return {};
    const withFitness = parsed as EmbeddingsCacheWithFitness;
    return withFitness.fitness_scores ?? {};
  } catch {
    return {};
  }
}

/**
 * Persist updated fitness scores back into embeddings.json atomically.
 *
 * Preserves the existing embeddings array (if any) so this is backward-
 * compatible with the plain-array format used by embeddings.ts.
 */
async function saveFitnessScores(
  cwd: string,
  scores: Record<string, SectionUtilityScore>,
): Promise<void> {
  const path = fitnessScorePath(cwd);
  const dir = dirname(path);
  if (!existsSync(dir)) {
    await mkdir(dir, { recursive: true });
  }

  // Read existing data to preserve embeddings array
  let existing: unknown = [];
  if (existsSync(path)) {
    try {
      const raw = await readFile(path, "utf-8");
      existing = JSON.parse(raw);
    } catch {
      existing = [];
    }
  }

  let payload: EmbeddingsCacheWithFitness;
  if (Array.isArray(existing)) {
    // Old plain-array format — migrate to object while keeping embeddings intact.
    payload = { embeddings: existing, fitness_scores: scores };
  } else {
    payload = { ...(existing as EmbeddingsCacheWithFitness), fitness_scores: scores };
  }

  const tmp = path + ".fitness.tmp";
  await writeFile(tmp, JSON.stringify(payload, null, 2), "utf-8");
  await rename(tmp, path);
}

// ---------------------------------------------------------------------------
// In-memory lock for concurrent-access safety
// ---------------------------------------------------------------------------

/**
 * Per-cwd mutex map — ensures concurrent calls to recordUsage() for the
 * same workspace are serialised so EMA updates don't race.
 */
const _updateLocks = new Map<string, Promise<void>>();

function withLock(cwd: string, fn: () => Promise<void>): Promise<void> {
  const prev = _updateLocks.get(cwd) ?? Promise.resolve();
  const next = prev.then(fn).catch(() => {});
  _updateLocks.set(cwd, next);
  return next;
}

// ---------------------------------------------------------------------------
// SectionFitnessPredictor
// ---------------------------------------------------------------------------

/**
 * Cross-session genome fitness predictor.
 *
 * Records (query_topic, section_id, was_useful) events from prior sessions
 * and computes per-section utility scores via EMA grouped by query-topic
 * clusters (k-means on TF-IDF vectors, k=7).
 *
 * At retrieval time, call `rerank()` to re-sort candidates by:
 *   combined_score = semantic_score × utility_ema
 *
 * Utility scores are cached in embeddings.json alongside quantized embeddings.
 *
 * Usage:
 *   const predictor = new SectionFitnessPredictor(cwd);
 *   await predictor.recordUsage("how does genome retrieval work", "sections/overview.md", true);
 *   const ranked = await predictor.rerank(candidates, "genome retrieval overview");
 */
export class SectionFitnessPredictor {
  private cwd: string;
  /** Cached centroids rebuilt lazily from recorded events */
  private _centroids: Map<string, number>[] | null = null;
  /** Cached query-topic vectors used to build centroids */
  private _topicVecs: Map<string, number>[] = [];

  constructor(cwd: string) {
    this.cwd = cwd;
  }

  // ── Public API ──────────────────────────────────────────────────────────

  /**
   * Record a retrieval-utility event.
   *
   * This is the primary signal input. Call once per retrieved section per
   * retrieval round, passing `was_useful = true` when the section content was
   * actually used by the LLM or contributed to the answer, and `false` when
   * it was discarded or irrelevant.
   *
   * Concurrent calls for the same `cwd` are serialised via an in-memory lock
   * so EMA updates never race.
   */
  recordUsage(query_topic: string, section_id: string, was_useful: boolean): Promise<void> {
    return withLock(this.cwd, async () => {
      const scores = await loadFitnessScores(this.cwd);

      // Determine cluster assignment for this query topic
      const docVec = buildTFVector(tokeniseQuery(query_topic));
      const centroids = this._getOrRebuildCentroids(docVec);
      const clusterIdx = assignCluster(docVec, centroids);
      const clusterKey = String(clusterIdx);

      // Update score for section
      const existing = scores[section_id];
      const prevEma = existing?.cluster_utility[clusterKey] ?? 0.5; // neutral prior
      const prevCount = existing?.event_count ?? 0;
      const newCount = prevCount + 1;
      const newEma = updateUtilityEma(prevEma, was_useful, newCount);

      scores[section_id] = {
        section_id,
        cluster_utility: {
          ...(existing?.cluster_utility ?? {}),
          [clusterKey]: newEma,
        },
        event_count: newCount,
        updated_at: new Date().toISOString(),
      };

      await saveFitnessScores(this.cwd, scores);

      // Register this topic vector for future centroid rebuilds
      this._topicVecs.push(docVec);
      this._centroids = null; // invalidate cache
    });
  }

  /**
   * Re-rank retrieval candidates by combined fitness score.
   *
   * combined_score = semantic_score × utility_ema
   *
   * Sections with no prior history get a utility_ema of 0.5 (neutral prior)
   * so they are ranked on semantic score alone until data accumulates.
   *
   * Returns candidates sorted descending by combined_score.
   */
  async rerank(
    candidates: FitnessCandidate[],
    queryTopic: string,
  ): Promise<RankedFitnessCandidate[]> {
    if (candidates.length === 0) return [];

    const scores = await loadFitnessScores(this.cwd);
    const docVec = buildTFVector(tokeniseQuery(queryTopic));
    const centroids = this._getOrRebuildCentroids(docVec);
    const clusterIdx = assignCluster(docVec, centroids);
    const clusterKey = String(clusterIdx);

    const ranked: RankedFitnessCandidate[] = candidates.map((c) => {
      const sectionScore = scores[c.section_id];
      const utilityEma = sectionScore?.cluster_utility[clusterKey] ?? 0.5;
      const combinedScore = c.semantic_score * utilityEma;
      return {
        ...c,
        utility_ema: utilityEma,
        combined_score: combinedScore,
      };
    });

    return ranked.sort((a, b) => b.combined_score - a.combined_score);
  }

  /**
   * Load all utility scores from disk (useful for inspection / testing).
   */
  async loadScores(): Promise<Record<string, SectionUtilityScore>> {
    return loadFitnessScores(this.cwd);
  }

  /**
   * Return the number of distinct query-topic clusters currently known.
   * Useful for tests and diagnostics.
   */
  get clusterCount(): number {
    return this._getOrRebuildCentroids(new Map()).length;
  }

  // ── Private helpers ──────────────────────────────────────────────────────

  /**
   * Return current centroids, rebuilding from recorded topic vectors if stale.
   * When no prior vectors exist, the provided `currentDoc` seeds a single cluster.
   */
  private _getOrRebuildCentroids(currentDoc: Map<string, number>): Map<string, number>[] {
    if (this._centroids !== null) return this._centroids;

    const allVecs = this._topicVecs.length > 0 ? this._topicVecs : [currentDoc];
    this._centroids = kMeansClusters(allVecs, NUM_CLUSTERS);
    return this._centroids;
  }
}

// ---------------------------------------------------------------------------
// Module-level re-rank helper
// ---------------------------------------------------------------------------

/**
 * Re-rank genome section candidates by combined (semantic_score × utility_ema).
 *
 * Convenience wrapper around SectionFitnessPredictor for one-shot use.
 *
 * @param candidates  Sections with their semantic similarity scores
 * @param queryTopic  The query used to retrieve these candidates
 * @param cwd         Project root (used to locate embeddings.json)
 */
export async function rerankByFitness(
  candidates: FitnessCandidate[],
  queryTopic: string,
  cwd: string,
): Promise<RankedFitnessCandidate[]> {
  const predictor = new SectionFitnessPredictor(cwd);
  return predictor.rerank(candidates, queryTopic);
}
