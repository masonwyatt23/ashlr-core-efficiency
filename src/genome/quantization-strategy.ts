/**
 * Multi-Tier Embedding Quantization Strategy Engine
 *
 * Provides adaptive per-query quantization bit-depth selection by:
 *  1. Analyzing genome manifest metadata at load time (section count, size distribution, depth)
 *  2. Classifying query complexity via tokenization + IDF entropy
 *  3. Learning per-(complexity_class, quantization_level) outcomes: latency, recall@K, token cost
 *  4. Auto-selecting quantization bit-depth at query time based on complexity + historical perf
 *  5. Persisting an audit trail in evolution/quantization-strategy.jsonl
 *  6. Emitting QuantizationReport showing ROI per strategy
 *
 * Bit-depth semantics:
 *  - 4-bit  (float4):  Experimental; ~8× storage reduction, higher error (~0.02 MAE).
 *                      Suitable only for ultra-high-latency budgets or edge with tiny genomes.
 *  - 8-bit  (int8):    4× storage reduction, ~0.005 MAE. Best balance on large genomes.
 *  - 16-bit (int16):   2× storage reduction, ~0.00002 MAE. Near-lossless; use on small genomes
 *                      or when recall is critical.
 *  - 32-bit (float32): No quantization; baseline for accuracy comparison.
 */

import { appendJsonl, readJsonl } from "./jsonl.ts";
import { genomeDir, type GenomeManifest, type SectionMeta } from "./manifest.ts";
import { join } from "path";
import { homedir } from "os";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Supported quantization bit depths. */
export type QuantizationBitDepth = 4 | 8 | 16 | 32;

/**
 * Query complexity classification produced by QueryComplexityClassifier.
 * Mirrors QueryComplexity from embedding-router.ts but adds "complex" for
 * multi-concept semantic queries.
 */
export type QueryComplexityClass = "rare-term" | "semantic-broad" | "semantic-deep" | "general";

/**
 * Genome size tier derived from manifest metadata analysis.
 *  - tiny   : ≤ 20 sections
 *  - small  : 21–50 sections
 *  - medium : 51–150 sections
 *  - large  : > 150 sections
 */
export type GenomeSizeTier = "tiny" | "small" | "medium" | "large";

/** Static metadata derived from a genome manifest at load time. */
export interface GenomeProfile {
  sectionCount: number;
  avgSectionTokens: number;
  medianSectionTokens: number;
  maxDepth: number;
  /** Fraction of sections that have a parentId (hierarchical) */
  hierarchyRatio: number;
  sizeTier: GenomeSizeTier;
  /** Distribution of section token counts across quartiles [p25, p50, p75, p90] */
  tokenPercentiles: [number, number, number, number];
}

/** Rich output of the query complexity classifier. */
export interface QueryComplexityAnalysis {
  complexityClass: QueryComplexityClass;
  tokenCount: number;
  /** IDF entropy — high value = rare-term heavy; low = common words */
  idfEntropy: number;
  /** True when the query contains code signals (camelCase, operators, etc.) */
  hasCodeSignals: boolean;
  /** Fraction of tokens that are rare (IDF > rare_threshold) */
  rareTermRatio: number;
  /** Detected multi-concept breadth (number of semantic pivot words) */
  conceptCount: number;
}

/** A single outcome record tracked per (complexityClass, bitDepth) pair. */
export interface QuantizationOutcomeRecord {
  id: string;
  timestamp: string;
  complexityClass: QueryComplexityClass;
  bitDepth: QuantizationBitDepth;
  genomeSizeTier: GenomeSizeTier;
  /** Wall-clock search latency in ms */
  latencyMs: number;
  /**
   * Recall@K versus unquantized (float32) baseline.
   * 1.0 = identical top-K results; 0.0 = no overlap.
   * null when baseline comparison was not run.
   */
  recallAtK: number | null;
  /** Estimated token cost of the embedding call (0 for local-only) */
  tokenCost: number;
  /** Whether this search used the bucket (ANN) path or exhaustive fallback */
  usedBucketSearch: boolean;
  /** Caller-provided quality signal [0,1]; null until rated */
  matchQuality: number | null;
}

/** Aggregated per-strategy statistics for reporting. */
export interface StrategyStats {
  complexityClass: QueryComplexityClass;
  bitDepth: QuantizationBitDepth;
  genomeSizeTier: GenomeSizeTier;
  sampleCount: number;
  avgLatencyMs: number;
  /** Latency reduction vs float32 baseline for same class (0–1) */
  latencyReductionFraction: number;
  avgRecallAtK: number | null;
  /** Estimated recall loss vs float32 baseline (0–1, lower is better) */
  recallLoss: number | null;
  avgMatchQuality: number | null;
  /** Composite ROI score: latencyReduction - recallLoss * 3 */
  roiScore: number;
}

/** Per-strategy ROI summary emitted by buildQuantizationReport. */
export interface QuantizationReport {
  generatedAt: string;
  genomeSizeTier: GenomeSizeTier;
  totalOutcomes: number;
  strategies: StrategyStats[];
  /** The single best strategy recommended for the current genome */
  recommendation: {
    complexityClass: QueryComplexityClass;
    bitDepth: QuantizationBitDepth;
    rationale: string;
  };
}

// ---------------------------------------------------------------------------
// Common English stop-words for IDF entropy computation
// ---------------------------------------------------------------------------

const STOP_WORDS = new Set([
  "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
  "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
  "been", "being", "have", "has", "had", "do", "does", "did", "will",
  "would", "could", "should", "may", "might", "shall", "can", "need",
  "how", "what", "why", "when", "where", "which", "who", "that", "this",
  "these", "those", "it", "its", "if", "then", "so", "yet", "both",
  "either", "not", "nor", "such", "while", "since", "unless", "until",
  "find", "show", "list", "explain", "describe", "get", "use", "make",
]);

/** Approximate per-token IDF scores drawn from a general-purpose corpus. */
const COMMON_TOKEN_IDF: Record<string, number> = {
  // Code / technical — high IDF
  "function": 3.2, "class": 3.1, "interface": 3.4, "type": 2.8,
  "import": 3.0, "export": 2.9, "return": 2.7, "const": 3.0,
  "async": 3.3, "await": 3.3, "promise": 3.5, "error": 2.5,
  "config": 2.8, "router": 3.6, "handler": 3.5, "schema": 3.4,
  // Semantic — medium IDF
  "architecture": 4.1, "strategy": 3.9, "pattern": 3.7, "design": 3.5,
  "performance": 3.8, "latency": 4.0, "throughput": 4.2, "memory": 3.6,
  "security": 3.9, "auth": 4.0, "cache": 3.7, "index": 3.1,
  "query": 3.2, "search": 3.0, "retrieval": 4.3, "embedding": 4.5,
  // General — lower IDF
  "code": 2.2, "file": 2.1, "data": 2.0, "value": 1.9, "name": 1.8,
  "path": 2.3, "key": 2.0, "id": 1.7, "list": 1.8, "result": 2.1,
};

const DEFAULT_IDF = 2.5; // Fallback IDF for unknown tokens

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Naive word tokenizer: lowercase, split on non-alphanumeric, filter empties. */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9_]+/)
    .filter((t) => t.length > 1);
}

/** Compute Shannon entropy of a normalized probability distribution. */
function shannonEntropy(probs: number[]): number {
  let h = 0;
  for (const p of probs) {
    if (p > 0) h -= p * Math.log2(p);
  }
  return h;
}

/** Compute a percentile value from a sorted numeric array. */
function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.floor((p / 100) * (sorted.length - 1));
  return sorted[Math.min(idx, sorted.length - 1)]!;
}

// ---------------------------------------------------------------------------
// QueryComplexityClassifier
// ---------------------------------------------------------------------------

/**
 * Classifies query complexity using tokenization + IDF entropy analysis.
 *
 * Unlike the simpler heuristic in embedding-router.ts, this classifier:
 *  1. Computes a per-token IDF score to measure rare-term density.
 *  2. Measures Shannon entropy of the IDF distribution.
 *  3. Counts semantic pivot words (broad multi-concept queries).
 *  4. Distinguishes "semantic-broad" (high concept count) from "semantic-deep"
 *     (few rare terms with high IDF).
 */
export class QueryComplexityClassifier {
  private readonly rareIdfThreshold: number;
  private readonly minSemanticTokens: number;

  constructor(options: { rareIdfThreshold?: number; minSemanticTokens?: number } = {}) {
    this.rareIdfThreshold = options.rareIdfThreshold ?? 3.5;
    this.minSemanticTokens = options.minSemanticTokens ?? 4;
  }

  /**
   * Analyze a query and return a rich complexity breakdown.
   */
  analyze(query: string): QueryComplexityAnalysis {
    const tokens = tokenize(query);
    const tokenCount = tokens.length;

    if (tokenCount === 0) {
      return {
        complexityClass: "general",
        tokenCount: 0,
        idfEntropy: 0,
        hasCodeSignals: false,
        rareTermRatio: 0,
        conceptCount: 0,
      };
    }

    // Code signals
    const hasCodeSignals = this._detectCodeSignals(query);

    // IDF scores for each token (skip stop words)
    const contentTokens = tokens.filter((t) => !STOP_WORDS.has(t));
    const idfScores = contentTokens.map((t) => COMMON_TOKEN_IDF[t] ?? DEFAULT_IDF);

    // Normalize IDF scores to a probability distribution for entropy
    const idfSum = idfScores.reduce((s, v) => s + v, 0);
    const idfProbs = idfSum > 0 ? idfScores.map((v) => v / idfSum) : idfScores.map(() => 0);
    const idfEntropy = shannonEntropy(idfProbs);

    // Rare-term ratio: tokens with IDF above threshold
    const rareCount = idfScores.filter((s) => s >= this.rareIdfThreshold).length;
    const rareTermRatio = contentTokens.length > 0 ? rareCount / contentTokens.length : 0;

    // Concept count: distinct semantic pivot words
    const pivotWords = new Set([
      "how", "why", "what", "when", "where", "which", "compare",
      "difference", "between", "versus", "vs", "tradeoff", "impact",
      "effect", "affect", "relationship", "connection", "interaction",
    ]);
    const conceptCount = tokens.filter((t) => pivotWords.has(t)).length;

    // Classification logic
    const complexityClass = this._classify({
      tokenCount,
      hasCodeSignals,
      rareTermRatio,
      idfEntropy,
      conceptCount,
    });

    return {
      complexityClass,
      tokenCount,
      idfEntropy,
      hasCodeSignals,
      rareTermRatio,
      conceptCount,
    };
  }

  /** Classify based on computed signals. */
  private _classify(signals: {
    tokenCount: number;
    hasCodeSignals: boolean;
    rareTermRatio: number;
    idfEntropy: number;
    conceptCount: number;
  }): QueryComplexityClass {
    const { tokenCount, hasCodeSignals, rareTermRatio, conceptCount } = signals;

    // Short code-heavy queries: exact symbol lookup → rare-term
    if (hasCodeSignals && tokenCount <= 5) return "rare-term";

    // High rare-term ratio with any code signal → rare-term
    if (rareTermRatio >= 0.6 && hasCodeSignals) return "rare-term";

    // Multi-concept queries: broad semantic sweep
    if (conceptCount >= 2 && tokenCount >= this.minSemanticTokens) return "semantic-broad";

    // Long queries with low rare-term ratio: broad search
    if (tokenCount >= this.minSemanticTokens * 2 && rareTermRatio < 0.2) return "semantic-broad";

    // Long queries with high IDF / rare-term density: deep semantic
    if (tokenCount >= this.minSemanticTokens && rareTermRatio >= 0.3) return "semantic-deep";

    // Medium-length natural language queries
    if (tokenCount >= this.minSemanticTokens && !hasCodeSignals) return "semantic-broad";

    return "general";
  }

  /**
   * Detect whether the query contains code signals.
   *
   * Returns true when:
   *  - The query is a single word with any one code pattern (symbol lookup), OR
   *  - The query has at least 2 code patterns (multi-signal code expression).
   */
  private _detectCodeSignals(query: string): boolean {
    const codePatterns = [
      /[a-z][A-Z]/,       // camelCase
      /[a-z]_[a-z]/,      // snake_case
      /\.[a-z]{1,6}(\s|$)/, // file extension
      /[()[\]{}<>]/,      // brackets
      /[=!<>]{1,2}/,      // operators
      /\b0x[0-9a-fA-F]+\b/, // hex literals
      /#[a-zA-Z]/,        // tag/hash
      /\b[A-Z]{2,}\b/,    // UPPER_CASE constants
      /::/,               // namespace separator
    ];
    const matchCount = codePatterns.filter((re) => re.test(query)).length;
    if (matchCount === 0) return false;
    // Single-word queries (no spaces) with any code signal are symbol lookups
    if (!query.includes(" ") && matchCount >= 1) return true;
    return matchCount >= 2;
  }
}

// ---------------------------------------------------------------------------
// GenomeProfiler
// ---------------------------------------------------------------------------

/**
 * Analyzes a GenomeManifest at load time to produce a GenomeProfile.
 * The profile drives tier-aware default quantization choices.
 */
export class GenomeProfiler {
  /**
   * Build a GenomeProfile from a loaded GenomeManifest.
   */
  static profile(manifest: GenomeManifest): GenomeProfile {
    const sections = manifest.sections;
    const sectionCount = sections.length;

    if (sectionCount === 0) {
      return {
        sectionCount: 0,
        avgSectionTokens: 0,
        medianSectionTokens: 0,
        maxDepth: 0,
        hierarchyRatio: 0,
        sizeTier: "tiny",
        tokenPercentiles: [0, 0, 0, 0],
      };
    }

    const tokenCounts = sections.map((s) => s.tokens).sort((a, b) => a - b);
    const avgSectionTokens = tokenCounts.reduce((s, v) => s + v, 0) / sectionCount;
    const medianSectionTokens = percentile(tokenCounts, 50);

    const tokenPercentiles: [number, number, number, number] = [
      percentile(tokenCounts, 25),
      percentile(tokenCounts, 50),
      percentile(tokenCounts, 75),
      percentile(tokenCounts, 90),
    ];

    // Compute max depth by counting path separators
    const maxDepth = sections.reduce((max, s) => {
      const depth = (s.path.match(/\//g) ?? []).length;
      return Math.max(max, depth);
    }, 0);

    const hierarchicalCount = sections.filter((s) => s.parentId != null).length;
    const hierarchyRatio = hierarchicalCount / sectionCount;

    const sizeTier = GenomeProfiler._classifySize(sectionCount);

    return {
      sectionCount,
      avgSectionTokens,
      medianSectionTokens,
      maxDepth,
      hierarchyRatio,
      sizeTier,
      tokenPercentiles,
    };
  }

  private static _classifySize(sectionCount: number): GenomeSizeTier {
    if (sectionCount <= 20) return "tiny";
    if (sectionCount <= 50) return "small";
    if (sectionCount <= 150) return "medium";
    return "large";
  }
}

// ---------------------------------------------------------------------------
// QuantizationStrategyLearner
// ---------------------------------------------------------------------------

/**
 * Tracks per-(complexityClass, bitDepth) outcomes and selects the best
 * quantization depth at query time based on historical performance.
 *
 * Decision policy:
 *  1. If fewer than `minSamples` recorded outcomes for the (class, depth) pair,
 *     fall back to the heuristic default for the genome size tier.
 *  2. Otherwise, compute ROI = avgLatencyReduction - recallLoss * recallWeight
 *     and pick the depth with the highest ROI for the given complexity class.
 */
export class QuantizationStrategyLearner {
  private readonly minSamples: number;
  private readonly recallWeight: number;
  private readonly windowSize: number;

  constructor(options: {
    minSamples?: number;
    recallWeight?: number;
    windowSize?: number;
  } = {}) {
    this.minSamples = options.minSamples ?? 5;
    this.recallWeight = options.recallWeight ?? 3.0;
    this.windowSize = options.windowSize ?? 50;
  }

  /**
   * Select the best quantization bit-depth for the given query complexity
   * and genome size tier, based on recorded outcomes.
   *
   * @param complexityClass  Output of QueryComplexityClassifier
   * @param genomeSizeTier   Output of GenomeProfiler
   * @param outcomes         Recent outcome records (loaded from JSONL)
   * @returns Selected bit depth
   */
  selectBitDepth(
    complexityClass: QueryComplexityClass,
    genomeSizeTier: GenomeSizeTier,
    outcomes: QuantizationOutcomeRecord[],
  ): QuantizationBitDepth {
    const window = outcomes.slice(-this.windowSize);
    const relevant = window.filter(
      (o) => o.complexityClass === complexityClass && o.genomeSizeTier === genomeSizeTier,
    );

    if (relevant.length < this.minSamples) {
      return this._heuristicDefault(complexityClass, genomeSizeTier);
    }

    // Compute ROI per bit depth
    const depths: QuantizationBitDepth[] = [4, 8, 16, 32];
    let bestDepth: QuantizationBitDepth = this._heuristicDefault(complexityClass, genomeSizeTier);
    let bestRoi = -Infinity;

    // Baseline: avg latency for float32 in same class+tier
    const float32Records = relevant.filter((o) => o.bitDepth === 32);
    const baselineLatency =
      float32Records.length > 0
        ? float32Records.reduce((s, o) => s + o.latencyMs, 0) / float32Records.length
        : null;

    for (const depth of depths) {
      const depthRecords = relevant.filter((o) => o.bitDepth === depth);
      if (depthRecords.length < this.minSamples) continue;

      const avgLatency =
        depthRecords.reduce((s, o) => s + o.latencyMs, 0) / depthRecords.length;

      const latencyReduction =
        baselineLatency != null && baselineLatency > 0
          ? (baselineLatency - avgLatency) / baselineLatency
          : 0;

      const withRecall = depthRecords.filter((o) => o.recallAtK !== null);
      const avgRecall =
        withRecall.length > 0
          ? withRecall.reduce((s, o) => s + (o.recallAtK as number), 0) / withRecall.length
          : null;

      const recallLoss = avgRecall !== null ? Math.max(0, 1 - avgRecall) : 0;

      const roi = latencyReduction - recallLoss * this.recallWeight;

      if (roi > bestRoi) {
        bestRoi = roi;
        bestDepth = depth;
      }
    }

    return bestDepth;
  }

  /**
   * Heuristic default quantization level when insufficient data is available.
   *
   * Strategy:
   *  - tiny genome  → int16 (small, so storage savings minor; prefer accuracy)
   *  - small genome → int16 for rare-term; int8 for semantic
   *  - medium/large → int8 always (strong latency gains, acceptable recall loss)
   *  - rare-term on any size → int16 (exact lookups need precision)
   */
  _heuristicDefault(
    complexityClass: QueryComplexityClass,
    genomeSizeTier: GenomeSizeTier,
  ): QuantizationBitDepth {
    if (genomeSizeTier === "tiny") return 16;

    if (complexityClass === "rare-term") {
      // Rare-term lookups need precision — int16 preserves near-exact similarity
      return genomeSizeTier === "small" ? 16 : 8;
    }

    if (complexityClass === "semantic-deep") {
      // Deep semantic needs reasonable recall — int16 on small, int8 on large
      return genomeSizeTier === "small" ? 16 : 8;
    }

    // semantic-broad and general: int8 is fine
    if (genomeSizeTier === "large") return 8;

    return 8;
  }
}

// ---------------------------------------------------------------------------
// Outcome persistence
// ---------------------------------------------------------------------------

function quantizationStrategyPath(cwd: string): string {
  return join(genomeDir(cwd), "evolution", "quantization-strategy.jsonl");
}

/**
 * Append a quantization outcome record to the audit trail.
 */
export async function recordQuantizationOutcome(
  cwd: string,
  record: Omit<QuantizationOutcomeRecord, "id" | "timestamp">,
): Promise<string> {
  const id = `qs-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
  const full: QuantizationOutcomeRecord = {
    id,
    timestamp: new Date().toISOString(),
    ...record,
  };
  await appendJsonl(quantizationStrategyPath(cwd), full);
  return id;
}

/**
 * Load all recorded quantization outcome records.
 */
export async function loadQuantizationOutcomes(
  cwd: string,
): Promise<QuantizationOutcomeRecord[]> {
  return readJsonl<QuantizationOutcomeRecord>(quantizationStrategyPath(cwd));
}

/**
 * Update match quality for a previously recorded outcome (append-only correction).
 */
export async function updateQuantizationOutcomeQuality(
  cwd: string,
  outcomeId: string,
  matchQuality: number,
  recallAtK?: number,
): Promise<void> {
  const outcomes = await loadQuantizationOutcomes(cwd);
  const original = outcomes.find((o) => o.id === outcomeId);
  if (!original) return;

  await recordQuantizationOutcome(cwd, {
    complexityClass: original.complexityClass,
    bitDepth: original.bitDepth,
    genomeSizeTier: original.genomeSizeTier,
    latencyMs: original.latencyMs,
    recallAtK: recallAtK ?? original.recallAtK,
    tokenCost: original.tokenCost,
    usedBucketSearch: original.usedBucketSearch,
    matchQuality,
  });
}

// ---------------------------------------------------------------------------
// QuantizationReport builder
// ---------------------------------------------------------------------------

/**
 * Build a QuantizationReport from recorded outcomes for the given genome tier.
 *
 * Example output line in report:
 *   "int8 saves 40% latency, loses 2% recall on semantic queries > 15 tokens"
 */
export function buildQuantizationReport(
  outcomes: QuantizationOutcomeRecord[],
  genomeSizeTier: GenomeSizeTier,
): QuantizationReport {
  const relevant = outcomes.filter((o) => o.genomeSizeTier === genomeSizeTier);

  const complexityClasses: QueryComplexityClass[] = [
    "rare-term",
    "semantic-broad",
    "semantic-deep",
    "general",
  ];
  const depths: QuantizationBitDepth[] = [4, 8, 16, 32];

  // Baseline (float32) latencies per complexity class
  const baselineLatencies: Partial<Record<QueryComplexityClass, number>> = {};
  for (const cc of complexityClasses) {
    const f32 = relevant.filter((o) => o.complexityClass === cc && o.bitDepth === 32);
    if (f32.length > 0) {
      baselineLatencies[cc] = f32.reduce((s, o) => s + o.latencyMs, 0) / f32.length;
    }
  }

  const strategies: StrategyStats[] = [];

  for (const cc of complexityClasses) {
    for (const depth of depths) {
      const records = relevant.filter(
        (o) => o.complexityClass === cc && o.bitDepth === depth,
      );
      if (records.length === 0) continue;

      const avgLatencyMs = records.reduce((s, o) => s + o.latencyMs, 0) / records.length;
      const baseline = baselineLatencies[cc];
      const latencyReductionFraction =
        baseline != null && baseline > 0
          ? Math.max(0, (baseline - avgLatencyMs) / baseline)
          : 0;

      const withRecall = records.filter((o) => o.recallAtK !== null);
      const avgRecallAtK =
        withRecall.length > 0
          ? withRecall.reduce((s, o) => s + (o.recallAtK as number), 0) / withRecall.length
          : null;

      const recallLoss = avgRecallAtK !== null ? Math.max(0, 1 - avgRecallAtK) : null;

      const withQuality = records.filter((o) => o.matchQuality !== null);
      const avgMatchQuality =
        withQuality.length > 0
          ? withQuality.reduce((s, o) => s + (o.matchQuality as number), 0) / withQuality.length
          : null;

      const roiScore =
        latencyReductionFraction - (recallLoss ?? 0) * 3;

      strategies.push({
        complexityClass: cc,
        bitDepth: depth,
        genomeSizeTier,
        sampleCount: records.length,
        avgLatencyMs,
        latencyReductionFraction,
        avgRecallAtK,
        recallLoss,
        avgMatchQuality,
        roiScore,
      });
    }
  }

  // Sort by ROI descending
  strategies.sort((a, b) => b.roiScore - a.roiScore);

  // Recommendation: best strategy across all complexity classes
  const best = strategies.find((s) => s.sampleCount >= 3) ?? strategies[0];
  const recommendation = best
    ? {
        complexityClass: best.complexityClass,
        bitDepth: best.bitDepth,
        rationale: _buildRationale(best),
      }
    : {
        complexityClass: "general" as QueryComplexityClass,
        bitDepth: 8 as QuantizationBitDepth,
        rationale: "Insufficient data — using default int8 for medium/large genomes.",
      };

  return {
    generatedAt: new Date().toISOString(),
    genomeSizeTier,
    totalOutcomes: relevant.length,
    strategies,
    recommendation,
  };
}

/** Build a human-readable rationale string for a strategy. */
function _buildRationale(s: StrategyStats): string {
  const depthLabel = s.bitDepth === 32 ? "float32 (no quantization)" : `int${s.bitDepth}`;
  const latencyPct = Math.round(s.latencyReductionFraction * 100);
  const recallLossPct =
    s.recallLoss !== null ? Math.round(s.recallLoss * 100) : null;

  const parts: string[] = [];
  if (latencyPct > 0) {
    parts.push(`${depthLabel} saves ${latencyPct}% latency`);
  } else {
    parts.push(`${depthLabel} (no latency gain vs baseline)`);
  }

  if (recallLossPct !== null) {
    parts.push(
      recallLossPct === 0
        ? "with no recall loss"
        : `loses ${recallLossPct}% recall`,
    );
  }

  parts.push(`on ${s.complexityClass} queries`);
  return parts.join(", ") + ".";
}

// ---------------------------------------------------------------------------
// Fallback Chain Types
// ---------------------------------------------------------------------------

/**
 * Model tier labels matching those in embedding-router.ts but defined
 * locally to avoid a circular dependency.
 */
export type QuantizationModelTier = "accurate" | "balanced" | "fast" | "ultrafast";

/**
 * A single step in a QuantizationFallbackChain.
 * Each step specifies a (bitDepth, modelTier, timeoutMs) tuple.
 */
export interface FallbackChainStep {
  bitDepth: QuantizationBitDepth;
  modelTier: QuantizationModelTier;
  /** Hard timeout for this step in ms. If a step exceeds this, the next step is tried. */
  timeoutMs: number;
}

/**
 * An ordered list of fallback steps for quantization selection.
 * Steps are tried from index 0 (preferred, highest quality) to the last
 * (most degraded, guaranteed-fast).
 *
 * Canonical default:
 *   [(16, 'accurate', 5000), (16, 'balanced', 10000), (8, 'fast', 3000), (4, 'ultrafast', 1000)]
 */
export type QuantizationFallbackChain = FallbackChainStep[];

/** Default fallback chain: quality-first with progressive degradation. */
export const DEFAULT_FALLBACK_CHAIN: QuantizationFallbackChain = [
  { bitDepth: 16, modelTier: "accurate",   timeoutMs: 5_000 },
  { bitDepth: 16, modelTier: "balanced",   timeoutMs: 10_000 },
  { bitDepth: 8,  modelTier: "fast",       timeoutMs: 3_000 },
  { bitDepth: 4,  modelTier: "ultrafast",  timeoutMs: 1_000 },
];

/** The result of a successful quantization selection via fallback chain. */
export interface SelectedQuantization {
  bitDepth: QuantizationBitDepth;
  modelTier: QuantizationModelTier;
  /** Index of the chain step that succeeded (0 = preferred). */
  stepIndex: number;
  /** Whether this selection degraded from the preferred step (stepIndex > 0). */
  usedFallback: boolean;
  /** Execution metadata from the chain traversal. */
  fallbackMeta: FallbackExecutionMetadata;
}

/**
 * Per-step execution record emitted during chain traversal.
 * Tracks outcome for each attempted step.
 */
export interface FallbackStepRecord {
  stepIndex: number;
  bitDepth: QuantizationBitDepth;
  modelTier: QuantizationModelTier;
  timeoutMs: number;
  latencyMs: number;
  /** "success" | "timeout" | "error" */
  outcome: "success" | "timeout" | "error";
  errorMessage?: string;
}

/**
 * Full metadata produced by a fallback chain execution.
 * Emitted alongside the SelectedQuantization for callers and audit logging.
 */
export interface FallbackExecutionMetadata {
  id: string;
  timestamp: string;
  profileSizeTier: GenomeSizeTier;
  targetTokens: number;
  stepsAttempted: FallbackStepRecord[];
  finalStepIndex: number;
  finalBitDepth: QuantizationBitDepth;
  finalModelTier: QuantizationModelTier;
  /** Human-readable rationale explaining why this step was selected. */
  selectionRationale: string;
  totalLatencyMs: number;
}

// ---------------------------------------------------------------------------
// Fallback Audit Persistence
// ---------------------------------------------------------------------------

/** Path to the fallback audit JSONL log (~/.ashlr/genome/quantization-fallback-audit.jsonl). */
function fallbackAuditPath(): string {
  return join(homedir(), ".ashlr", "genome", "quantization-fallback-audit.jsonl");
}

/**
 * Append a FallbackExecutionMetadata record to the global audit trail.
 * Best-effort: errors are silently swallowed to avoid disrupting the caller.
 */
async function appendFallbackAudit(meta: FallbackExecutionMetadata): Promise<void> {
  try {
    await appendJsonl(fallbackAuditPath(), meta);
  } catch {
    // Best-effort — audit failure must not break the caller
  }
}

/**
 * Load all fallback audit records from ~/.ashlr/genome/quantization-fallback-audit.jsonl.
 */
export async function loadFallbackAuditRecords(): Promise<FallbackExecutionMetadata[]> {
  return readJsonl<FallbackExecutionMetadata>(fallbackAuditPath());
}

// ---------------------------------------------------------------------------
// QuantizationFallbackExecutor
// ---------------------------------------------------------------------------

/**
 * Evaluates whether a given fallback chain step should be considered
 * acceptable for the profile + targetTokens context.
 *
 * This thin probe allows tests and callers to inject failure scenarios
 * without requiring a real Ollama instance. The default probe always
 * resolves immediately (no I/O), indicating "available".
 *
 * @returns true when the step is available; false when it should be skipped
 *          (treated as a timeout/unavailability failure).
 */
export type FallbackStepProbe = (
  step: FallbackChainStep,
  profile: GenomeProfile,
  targetTokens: number,
) => Promise<boolean>;

/** Default probe: always available (no real I/O). */
const DEFAULT_PROBE: FallbackStepProbe = async () => true;

/**
 * Options for `selectQuantizationWithFallback`.
 */
export interface FallbackSelectionOptions {
  /** Override the fallback chain (defaults to DEFAULT_FALLBACK_CHAIN). */
  chain?: QuantizationFallbackChain;
  /**
   * Probe function that tests whether a given step is currently usable.
   * Inject a custom probe in tests to simulate timeouts or Ollama failures.
   * Defaults to a no-op probe (always available).
   */
  probe?: FallbackStepProbe;
  /** If true, audit records are NOT written to disk. Useful for unit tests. */
  skipAudit?: boolean;
}

/**
 * Select the best quantization configuration for `profile` / `targetTokens`
 * by walking the fallback chain until a step succeeds.
 *
 * Algorithm:
 *  1. For each step in the chain (preferred → degraded):
 *     a. Race `probe(step)` against the step's `timeoutMs`.
 *     b. If the probe resolves true within the timeout → success, stop.
 *     c. If the probe returns false or the timeout fires → record failure, continue.
 *  2. If all steps fail the probe, the last step is used unconditionally
 *     (guaranteed fallback).
 *  3. Record per-step latency + outcome in FallbackExecutionMetadata.
 *  4. Append the metadata to the global audit log (unless skipAudit=true).
 *
 * @param profile       GenomeProfile for the current genome (from GenomeProfiler).
 * @param targetTokens  Token budget for the retrieval operation.
 * @param options       Optional chain override, probe, and audit toggle.
 */
export async function selectQuantizationWithFallback(
  profile: GenomeProfile,
  targetTokens: number,
  options: FallbackSelectionOptions = {},
): Promise<SelectedQuantization> {
  const chain = options.chain ?? DEFAULT_FALLBACK_CHAIN;
  const probe = options.probe ?? DEFAULT_PROBE;

  if (chain.length === 0) {
    throw new Error("QuantizationFallbackChain must contain at least one step.");
  }

  const stepsAttempted: FallbackStepRecord[] = [];
  const overallStart = Date.now();

  for (let i = 0; i < chain.length; i++) {
    const step = chain[i]!;
    const stepStart = Date.now();

    let outcome: FallbackStepRecord["outcome"] = "error";
    let errorMessage: string | undefined;

    try {
      // Race the probe against the step-level timeout
      const probePromise = probe(step, profile, targetTokens).then((ok) => {
        if (!ok) throw new Error("probe returned false");
        return true;
      });
      const timeoutPromise = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error(`step timeout after ${step.timeoutMs}ms`)), step.timeoutMs),
      );

      await Promise.race([probePromise, timeoutPromise]);
      outcome = "success";
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      errorMessage = msg;
      outcome = msg.includes("timeout") ? "timeout" : "error";
    }

    const latencyMs = Date.now() - stepStart;

    stepsAttempted.push({
      stepIndex: i,
      bitDepth: step.bitDepth,
      modelTier: step.modelTier,
      timeoutMs: step.timeoutMs,
      latencyMs,
      outcome,
      ...(errorMessage !== undefined ? { errorMessage } : {}),
    });

    if (outcome === "success") {
      // This step succeeded — build metadata and return
      const meta = _buildFallbackMeta(
        profile,
        targetTokens,
        stepsAttempted,
        i,
        step,
        Date.now() - overallStart,
      );

      if (!options.skipAudit) {
        await appendFallbackAudit(meta);
      }

      return {
        bitDepth: step.bitDepth,
        modelTier: step.modelTier,
        stepIndex: i,
        usedFallback: i > 0,
        fallbackMeta: meta,
      };
    }
  }

  // All steps failed — use the last step unconditionally (guaranteed fallback)
  const lastStep = chain[chain.length - 1]!;
  const lastIdx = chain.length - 1;

  // Correct the last stepsAttempted entry: mark it as success (forced)
  const lastRecord = stepsAttempted[lastIdx];
  if (lastRecord && lastRecord.outcome !== "success") {
    stepsAttempted[lastIdx] = { ...lastRecord, outcome: "success", errorMessage: undefined };
  }

  const meta = _buildFallbackMeta(
    profile,
    targetTokens,
    stepsAttempted,
    lastIdx,
    lastStep,
    Date.now() - overallStart,
    true,
  );

  if (!options.skipAudit) {
    await appendFallbackAudit(meta);
  }

  return {
    bitDepth: lastStep.bitDepth,
    modelTier: lastStep.modelTier,
    stepIndex: lastIdx,
    usedFallback: lastIdx > 0,
    fallbackMeta: meta,
  };
}

/** Build a FallbackExecutionMetadata record from traversal state. */
function _buildFallbackMeta(
  profile: GenomeProfile,
  targetTokens: number,
  stepsAttempted: FallbackStepRecord[],
  finalStepIndex: number,
  finalStep: FallbackChainStep,
  totalLatencyMs: number,
  forcedFallback = false,
): FallbackExecutionMetadata {
  const id = `qfc-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
  const failedCount = stepsAttempted.filter((s) => s.outcome !== "success").length;

  let rationale: string;
  if (finalStepIndex === 0) {
    rationale =
      `Preferred step selected: int${finalStep.bitDepth} / ${finalStep.modelTier} ` +
      `(no fallback needed, genome=${profile.sizeTier}).`;
  } else if (forcedFallback) {
    rationale =
      `All ${stepsAttempted.length} chain steps unavailable; forced last-resort ` +
      `int${finalStep.bitDepth} / ${finalStep.modelTier}.`;
  } else {
    rationale =
      `Degraded to step ${finalStepIndex}: int${finalStep.bitDepth} / ${finalStep.modelTier} ` +
      `after ${failedCount} failed step(s) (genome=${profile.sizeTier}, ` +
      `targetTokens=${targetTokens}).`;
  }

  return {
    id,
    timestamp: new Date().toISOString(),
    profileSizeTier: profile.sizeTier,
    targetTokens,
    stepsAttempted,
    finalStepIndex,
    finalBitDepth: finalStep.bitDepth,
    finalModelTier: finalStep.modelTier,
    selectionRationale: rationale,
    totalLatencyMs,
  };
}

// ---------------------------------------------------------------------------
// Chain builder helper
// ---------------------------------------------------------------------------

/**
 * Build a QuantizationFallbackChain that starts from a given learner-preferred
 * bit depth and degrades through the standard tier order.
 *
 * This lets the adaptive learner's recommendation act as the preferred step
 * while still providing the full fallback safety net.
 *
 * Mapping: 16→accurate, 8→fast, 4→ultrafast, 32→accurate (treated like 16).
 */
function _buildChainFromDepth(depth: QuantizationBitDepth): QuantizationFallbackChain {
  // Full chain anchored to the learner recommendation at the top
  switch (depth) {
    case 16:
      return [
        { bitDepth: 16, modelTier: "accurate",  timeoutMs: 5_000 },
        { bitDepth: 16, modelTier: "balanced",  timeoutMs: 10_000 },
        { bitDepth: 8,  modelTier: "fast",      timeoutMs: 3_000 },
        { bitDepth: 4,  modelTier: "ultrafast", timeoutMs: 1_000 },
      ];
    case 8:
      return [
        { bitDepth: 8,  modelTier: "fast",      timeoutMs: 3_000 },
        { bitDepth: 8,  modelTier: "balanced",  timeoutMs: 10_000 },
        { bitDepth: 4,  modelTier: "ultrafast", timeoutMs: 1_000 },
      ];
    case 4:
      return [
        { bitDepth: 4,  modelTier: "ultrafast", timeoutMs: 1_000 },
      ];
    case 32:
    default:
      return DEFAULT_FALLBACK_CHAIN;
  }
}

// ---------------------------------------------------------------------------
// QuantizationStrategyEngine — main orchestrator
// ---------------------------------------------------------------------------

/**
 * Top-level engine that ties together GenomeProfiler, QueryComplexityClassifier,
 * QuantizationStrategyLearner, outcome persistence, and report generation.
 *
 * Typical usage:
 *   const engine = new QuantizationStrategyEngine(cwd);
 *   await engine.loadProfile(manifest);
 *
 *   // At query time:
 *   const { bitDepth, outcomeId } = await engine.selectBitDepth(query);
 *   // ... run search with bitDepth ...
 *   await engine.recordOutcome(outcomeId, { latencyMs, recallAtK, usedBucketSearch });
 */
export class QuantizationStrategyEngine {
  private readonly cwd: string;
  private readonly classifier: QueryComplexityClassifier;
  private readonly learner: QuantizationStrategyLearner;

  private _profile: GenomeProfile | null = null;

  constructor(
    cwd: string,
    options: {
      classifierOptions?: ConstructorParameters<typeof QueryComplexityClassifier>[0];
      learnerOptions?: ConstructorParameters<typeof QuantizationStrategyLearner>[0];
    } = {},
  ) {
    this.cwd = cwd;
    this.classifier = new QueryComplexityClassifier(options.classifierOptions);
    this.learner = new QuantizationStrategyLearner(options.learnerOptions);
  }

  /**
   * Analyze the manifest and cache the genome profile.
   * Call once at startup / after manifest reload.
   */
  loadProfile(manifest: GenomeManifest): GenomeProfile {
    this._profile = GenomeProfiler.profile(manifest);
    return this._profile;
  }

  /** Return the cached genome profile (null if loadProfile has not been called). */
  get profile(): GenomeProfile | null {
    return this._profile;
  }

  /**
   * Select the best bit-depth for the given query.
   *
   * Internally uses `selectQuantizationWithFallback` to walk the fallback
   * chain, then records a placeholder outcome for later update via
   * `recordOutcome()`.
   *
   * @param query           Natural-language or code query string.
   * @param fallbackOptions Optional chain/probe overrides (useful in tests).
   */
  async selectBitDepth(
    query: string,
    fallbackOptions?: FallbackSelectionOptions,
  ): Promise<{
    bitDepth: QuantizationBitDepth;
    analysis: QueryComplexityAnalysis;
    genomeSizeTier: GenomeSizeTier;
    outcomeId: string;
    fallbackMeta?: FallbackExecutionMetadata;
  }> {
    const profile = this._profile;
    const genomeSizeTier: GenomeSizeTier = profile?.sizeTier ?? "medium";

    const analysis = this.classifier.analyze(query);

    // Derive target token count from query complexity for chain hints
    const targetTokens = analysis.tokenCount;

    // Use learner to determine a candidate depth (no-IO, heuristic-based)
    const outcomes = await loadQuantizationOutcomes(this.cwd);
    const learnerDepth = this.learner.selectBitDepth(
      analysis.complexityClass,
      genomeSizeTier,
      outcomes,
    );

    // Build a profile stub if no manifest has been loaded yet
    const effectiveProfile: GenomeProfile = profile ?? {
      sectionCount: 0,
      avgSectionTokens: 0,
      medianSectionTokens: 0,
      maxDepth: 0,
      hierarchyRatio: 0,
      sizeTier: genomeSizeTier,
      tokenPercentiles: [0, 0, 0, 0],
    };

    // Build a chain that starts from the learner's preferred depth so the
    // fallback chain honours the adaptive learner recommendation.
    const preferredChain = _buildChainFromDepth(learnerDepth);
    const chain = fallbackOptions?.chain ?? preferredChain;

    let bitDepth: QuantizationBitDepth;
    let fallbackMeta: FallbackExecutionMetadata | undefined;

    try {
      const selected = await selectQuantizationWithFallback(effectiveProfile, targetTokens, {
        ...fallbackOptions,
        chain,
        skipAudit: fallbackOptions?.skipAudit ?? false,
      });
      bitDepth = selected.bitDepth;
      fallbackMeta = selected.fallbackMeta;
    } catch {
      // Fallback chain threw unexpectedly — use learner depth as last resort
      bitDepth = learnerDepth;
    }

    // Pre-record with placeholder latency; caller updates via recordOutcome
    const outcomeId = await recordQuantizationOutcome(this.cwd, {
      complexityClass: analysis.complexityClass,
      bitDepth,
      genomeSizeTier,
      latencyMs: 0, // Updated by caller
      recallAtK: null,
      tokenCost: 0,
      usedBucketSearch: false,
      matchQuality: null,
    });

    return { bitDepth, analysis, genomeSizeTier, outcomeId, fallbackMeta };
  }

  /**
   * Alias for `selectBitDepth` — primary entry-point when callers pass a
   * GenomeProfile + targetTokens directly rather than a raw query string.
   *
   * Drives the fallback chain and returns the full SelectedQuantization
   * (including FallbackExecutionMetadata).
   */
  async select(
    profile: GenomeProfile,
    targetTokens: number,
    options?: FallbackSelectionOptions,
  ): Promise<SelectedQuantization> {
    return selectQuantizationWithFallback(profile, targetTokens, options);
  }

  /**
   * Update the outcome record with actual performance data after the search completes.
   */
  async recordOutcome(
    outcomeId: string,
    data: {
      latencyMs: number;
      recallAtK?: number | null;
      tokenCost?: number;
      usedBucketSearch?: boolean;
      matchQuality?: number | null;
    },
  ): Promise<void> {
    const outcomes = await loadQuantizationOutcomes(this.cwd);
    const original = outcomes.find((o) => o.id === outcomeId);
    if (!original) return;

    await recordQuantizationOutcome(this.cwd, {
      complexityClass: original.complexityClass,
      bitDepth: original.bitDepth,
      genomeSizeTier: original.genomeSizeTier,
      latencyMs: data.latencyMs,
      recallAtK: data.recallAtK ?? null,
      tokenCost: data.tokenCost ?? 0,
      usedBucketSearch: data.usedBucketSearch ?? false,
      matchQuality: data.matchQuality ?? null,
    });
  }

  /**
   * Build and return a QuantizationReport for the current genome tier.
   */
  async buildReport(): Promise<QuantizationReport> {
    const outcomes = await loadQuantizationOutcomes(this.cwd);
    const tier = this._profile?.sizeTier ?? "medium";
    return buildQuantizationReport(outcomes, tier);
  }
}
