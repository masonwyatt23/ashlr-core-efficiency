/**
 * Tests for Multi-Tier Embedding Quantization Strategy Engine
 *
 * Covers:
 *  - QueryComplexityClassifier: tokenization, IDF entropy, complexity classification
 *  - GenomeProfiler: manifest metadata analysis, size tier classification
 *  - QuantizationStrategyLearner: heuristic defaults + learned selection
 *  - recordQuantizationOutcome / loadQuantizationOutcomes: JSONL persistence round-trip
 *  - buildQuantizationReport: ROI computation, recommendation, rationale strings
 *  - QuantizationStrategyEngine: end-to-end selectBitDepth + recordOutcome
 *  - QuantizedANNSearcher.search(): overrideBitDepth parameter + selectedBitDepth in result
 *  - EmbeddingRouter: enableQuantizationStrategy option integration
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, rm } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";

import {
  QueryComplexityClassifier,
  GenomeProfiler,
  QuantizationStrategyLearner,
  QuantizationStrategyEngine,
  recordQuantizationOutcome,
  loadQuantizationOutcomes,
  buildQuantizationReport,
  type QueryComplexityClass,
  type QuantizationBitDepth,
  type QuantizationOutcomeRecord,
  type GenomeSizeTier,
} from "../src/genome/quantization-strategy.ts";

import { QuantizedANNSearcher } from "../src/genome/quantized-ann.ts";
import { EmbeddingRouter } from "../src/genome/embedding-router.ts";
import type { GenomeManifest, SectionMeta } from "../src/genome/manifest.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeTmpDir(): string {
  return join(tmpdir(), `ashlr-qstrat-${Date.now()}-${Math.random().toString(36).slice(2)}`);
}

async function setupGenomeDir(cwd: string): Promise<void> {
  await mkdir(join(cwd, ".ashlrcode", "genome", "evolution"), { recursive: true });
}

/** Generate a deterministic unit-normalized embedding. */
function makeEmbedding(dim: number, seed = 1): number[] {
  const vec: number[] = [];
  let x = seed;
  for (let i = 0; i < dim; i++) {
    x = (x * 1664525 + 1013904223) & 0xffffffff;
    vec.push(((x >>> 16) / 32768 - 1) * 0.5);
  }
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  return norm === 0 ? vec : vec.map((v) => v / norm);
}

function makeSection(overrides: Partial<SectionMeta> = {}): SectionMeta {
  return {
    path: "vision/north-star.md",
    title: "North Star",
    summary: "Project vision",
    tags: [],
    tokens: 200,
    updatedAt: new Date().toISOString(),
    ...overrides,
  };
}

function makeManifest(sections: SectionMeta[]): GenomeManifest {
  return {
    version: 1,
    project: "test",
    sections,
    generation: { number: 1, milestone: "M1", startedAt: new Date().toISOString() },
    fitnessHistory: [],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
}

function makeOutcome(
  overrides: Partial<Omit<QuantizationOutcomeRecord, "id" | "timestamp">> = {},
): Omit<QuantizationOutcomeRecord, "id" | "timestamp"> {
  return {
    complexityClass: "semantic-broad",
    bitDepth: 8,
    genomeSizeTier: "large",
    latencyMs: 50,
    recallAtK: 0.95,
    tokenCost: 0,
    usedBucketSearch: true,
    matchQuality: 0.85,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// QueryComplexityClassifier
// ---------------------------------------------------------------------------

describe("QueryComplexityClassifier — complexity detection", () => {
  const clf = new QueryComplexityClassifier();

  test("empty query → general", () => {
    const r = clf.analyze("");
    expect(r.complexityClass).toBe("general");
    expect(r.tokenCount).toBe(0);
  });

  test("camelCase identifier → rare-term", () => {
    const r = clf.analyze("retrieveSections");
    expect(r.complexityClass).toBe("rare-term");
    expect(r.hasCodeSignals).toBe(true);
  });

  test("snake_case + operator → rare-term", () => {
    const r = clf.analyze("genome_dir_path != null");
    expect(r.complexityClass).toBe("rare-term");
  });

  test("short code query ≤ 5 tokens → rare-term", () => {
    const r = clf.analyze("EmbeddingRouter.embed()");
    expect(r.complexityClass).toBe("rare-term");
  });

  test("multi-concept query with pivot words → semantic-broad", () => {
    const r = clf.analyze("what is the difference between int8 and int16 quantization");
    expect(r.complexityClass).toBe("semantic-broad");
    expect(r.conceptCount).toBeGreaterThanOrEqual(2);
  });

  test("long natural-language query, no code → semantic-broad", () => {
    const r = clf.analyze(
      "how does the embedding router select the best model for semantic search",
    );
    expect(r.complexityClass).toBe("semantic-broad");
  });

  test("technical query with rare terms → semantic-deep or rare-term", () => {
    const r = clf.analyze("quantization strategy latency recall tradeoff embedding router");
    // High rare-term ratio → semantic-deep or rare-term
    expect(["semantic-deep", "rare-term"]).toContain(r.complexityClass);
  });

  test("short common query → general", () => {
    const r = clf.analyze("find file");
    expect(r.complexityClass).toBe("general");
  });

  test("analyze returns all expected fields", () => {
    const r = clf.analyze("how does caching work in the router");
    expect(typeof r.tokenCount).toBe("number");
    expect(typeof r.idfEntropy).toBe("number");
    expect(typeof r.hasCodeSignals).toBe("boolean");
    expect(typeof r.rareTermRatio).toBe("number");
    expect(typeof r.conceptCount).toBe("number");
    expect(r.idfEntropy).toBeGreaterThanOrEqual(0);
    expect(r.rareTermRatio).toBeGreaterThanOrEqual(0);
    expect(r.rareTermRatio).toBeLessThanOrEqual(1);
  });

  test("IDF entropy is higher for rare-term queries than stop-word-heavy queries", () => {
    const rareR = clf.analyze("QuantizedANNSearcher buildIndex bucketGranularity");
    const commonR = clf.analyze("how and why the and for with");
    expect(rareR.idfEntropy).toBeGreaterThanOrEqual(commonR.idfEntropy);
  });
});

// ---------------------------------------------------------------------------
// GenomeProfiler
// ---------------------------------------------------------------------------

describe("GenomeProfiler — manifest metadata analysis", () => {
  test("empty manifest → tiny tier, zeros", () => {
    const p = GenomeProfiler.profile(makeManifest([]));
    expect(p.sectionCount).toBe(0);
    expect(p.sizeTier).toBe("tiny");
    expect(p.avgSectionTokens).toBe(0);
  });

  test("≤ 20 sections → tiny", () => {
    const sections = Array.from({ length: 15 }, (_, i) =>
      makeSection({ path: `s${i}.md`, tokens: 100 }),
    );
    const p = GenomeProfiler.profile(makeManifest(sections));
    expect(p.sizeTier).toBe("tiny");
    expect(p.sectionCount).toBe(15);
  });

  test("21–50 sections → small", () => {
    const sections = Array.from({ length: 35 }, (_, i) =>
      makeSection({ path: `s${i}.md`, tokens: 150 }),
    );
    const p = GenomeProfiler.profile(makeManifest(sections));
    expect(p.sizeTier).toBe("small");
  });

  test("51–150 sections → medium", () => {
    const sections = Array.from({ length: 80 }, (_, i) =>
      makeSection({ path: `s${i}.md`, tokens: 200 }),
    );
    const p = GenomeProfiler.profile(makeManifest(sections));
    expect(p.sizeTier).toBe("medium");
  });

  test("> 150 sections → large", () => {
    const sections = Array.from({ length: 200 }, (_, i) =>
      makeSection({ path: `s${i}.md`, tokens: 300 }),
    );
    const p = GenomeProfiler.profile(makeManifest(sections));
    expect(p.sizeTier).toBe("large");
  });

  test("avgSectionTokens computed correctly", () => {
    const sections = [
      makeSection({ path: "a.md", tokens: 100 }),
      makeSection({ path: "b.md", tokens: 200 }),
      makeSection({ path: "c.md", tokens: 300 }),
    ];
    const p = GenomeProfiler.profile(makeManifest(sections));
    expect(p.avgSectionTokens).toBeCloseTo(200, 5);
  });

  test("maxDepth counts path separators", () => {
    const sections = [
      makeSection({ path: "a.md" }),
      makeSection({ path: "a/b.md" }),
      makeSection({ path: "a/b/c.md" }),
    ];
    const p = GenomeProfiler.profile(makeManifest(sections));
    expect(p.maxDepth).toBe(2);
  });

  test("hierarchyRatio computed from parentId presence", () => {
    const sections = [
      makeSection({ path: "a.md", parentId: undefined }),
      makeSection({ path: "b.md", parentId: "a.md" }),
      makeSection({ path: "c.md", parentId: "a.md" }),
      makeSection({ path: "d.md", parentId: undefined }),
    ];
    const p = GenomeProfiler.profile(makeManifest(sections));
    expect(p.hierarchyRatio).toBeCloseTo(0.5, 5);
  });

  test("tokenPercentiles has 4 entries", () => {
    const sections = Array.from({ length: 20 }, (_, i) =>
      makeSection({ path: `s${i}.md`, tokens: (i + 1) * 50 }),
    );
    const p = GenomeProfiler.profile(makeManifest(sections));
    expect(p.tokenPercentiles).toHaveLength(4);
    // p25 ≤ p50 ≤ p75 ≤ p90
    expect(p.tokenPercentiles[0]).toBeLessThanOrEqual(p.tokenPercentiles[1]!);
    expect(p.tokenPercentiles[1]).toBeLessThanOrEqual(p.tokenPercentiles[2]!);
    expect(p.tokenPercentiles[2]).toBeLessThanOrEqual(p.tokenPercentiles[3]!);
  });
});

// ---------------------------------------------------------------------------
// QuantizationStrategyLearner — heuristic defaults
// ---------------------------------------------------------------------------

describe("QuantizationStrategyLearner — heuristic defaults", () => {
  const learner = new QuantizationStrategyLearner();

  test("tiny genome → int16 for any complexity", () => {
    expect(learner._heuristicDefault("rare-term", "tiny")).toBe(16);
    expect(learner._heuristicDefault("semantic-broad", "tiny")).toBe(16);
    expect(learner._heuristicDefault("general", "tiny")).toBe(16);
  });

  test("rare-term + small → int16", () => {
    expect(learner._heuristicDefault("rare-term", "small")).toBe(16);
  });

  test("rare-term + large → int8", () => {
    expect(learner._heuristicDefault("rare-term", "large")).toBe(8);
  });

  test("semantic-broad + large → int8", () => {
    expect(learner._heuristicDefault("semantic-broad", "large")).toBe(8);
  });

  test("semantic-deep + small → int16", () => {
    expect(learner._heuristicDefault("semantic-deep", "small")).toBe(16);
  });

  test("general + medium → int8", () => {
    expect(learner._heuristicDefault("general", "medium")).toBe(8);
  });
});

// ---------------------------------------------------------------------------
// QuantizationStrategyLearner — learned selection
// ---------------------------------------------------------------------------

describe("QuantizationStrategyLearner — learned selection", () => {
  const learner = new QuantizationStrategyLearner({ minSamples: 3 });

  test("fewer than minSamples → falls back to heuristic", () => {
    const outcomes = [makeOutcome({ bitDepth: 8 }), makeOutcome({ bitDepth: 8 })].map(
      (o, i) => ({ ...o, id: `qs-${i}`, timestamp: new Date().toISOString() }),
    );
    // Only 2 outcomes for (semantic-broad, large) — below minSamples=3
    const depth = learner.selectBitDepth("semantic-broad", "large", outcomes);
    // Should return heuristic: int8 for semantic-broad+large
    expect(depth).toBe(8);
  });

  test("int8 with better latency ROI selected over int16", () => {
    // Create 5 int8 outcomes (low latency, high recall) and 5 int32 baselines
    const int32Outcomes = Array.from({ length: 5 }, (_, i) => ({
      ...makeOutcome({ bitDepth: 32, latencyMs: 100, recallAtK: 1.0 }),
      id: `qs-32-${i}`,
      timestamp: new Date().toISOString(),
    }));
    const int8Outcomes = Array.from({ length: 5 }, (_, i) => ({
      ...makeOutcome({ bitDepth: 8, latencyMs: 45, recallAtK: 0.97 }),
      id: `qs-8-${i}`,
      timestamp: new Date().toISOString(),
    }));
    const int16Outcomes = Array.from({ length: 5 }, (_, i) => ({
      ...makeOutcome({ bitDepth: 16, latencyMs: 70, recallAtK: 0.99 }),
      id: `qs-16-${i}`,
      timestamp: new Date().toISOString(),
    }));

    const all = [...int32Outcomes, ...int8Outcomes, ...int16Outcomes];
    const depth = learner.selectBitDepth("semantic-broad", "large", all);
    // int8 reduces latency 55% vs float32 baseline, recall loss 3% → ROI = 0.55 - 0.09 = 0.46
    // int16 reduces 30%, recall loss 1% → ROI = 0.30 - 0.03 = 0.27
    expect(depth).toBe(8);
  });

  test("high recall-loss depth is penalized", () => {
    // int4 (approximated via field value 4) has very poor recall
    const int32Outcomes = Array.from({ length: 5 }, (_, i) => ({
      ...makeOutcome({ bitDepth: 32, latencyMs: 100, recallAtK: 1.0 }),
      id: `qs-32-${i}`,
      timestamp: new Date().toISOString(),
    }));
    const int4Outcomes = Array.from({ length: 5 }, (_, i) => ({
      ...makeOutcome({ bitDepth: 4, latencyMs: 20, recallAtK: 0.60 }),
      id: `qs-4-${i}`,
      timestamp: new Date().toISOString(),
    }));
    const int8Outcomes = Array.from({ length: 5 }, (_, i) => ({
      ...makeOutcome({ bitDepth: 8, latencyMs: 50, recallAtK: 0.97 }),
      id: `qs-8-${i}`,
      timestamp: new Date().toISOString(),
    }));

    const all = [...int32Outcomes, ...int4Outcomes, ...int8Outcomes];
    const depth = learner.selectBitDepth("semantic-broad", "large", all);
    // int4: latency reduction 80%, recall loss 40% → ROI = 0.80 - 1.20 = -0.40
    // int8: latency reduction 50%, recall loss 3% → ROI = 0.50 - 0.09 = 0.41
    expect(depth).toBe(8);
  });
});

// ---------------------------------------------------------------------------
// Outcome persistence
// ---------------------------------------------------------------------------

describe("recordQuantizationOutcome / loadQuantizationOutcomes — JSONL persistence", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("returns empty array when no file exists", async () => {
    const outcomes = await loadQuantizationOutcomes(cwd);
    expect(outcomes).toEqual([]);
  });

  test("roundtrip: record then load", async () => {
    const id = await recordQuantizationOutcome(cwd, makeOutcome());
    const outcomes = await loadQuantizationOutcomes(cwd);
    expect(outcomes).toHaveLength(1);
    expect(outcomes[0]!.id).toBe(id);
    expect(outcomes[0]!.bitDepth).toBe(8);
    expect(outcomes[0]!.complexityClass).toBe("semantic-broad");
    expect(outcomes[0]!.genomeSizeTier).toBe("large");
    expect(outcomes[0]!.recallAtK).toBeCloseTo(0.95, 5);
  });

  test("multiple records append correctly", async () => {
    await recordQuantizationOutcome(cwd, makeOutcome({ bitDepth: 8 }));
    await recordQuantizationOutcome(cwd, makeOutcome({ bitDepth: 16 }));
    await recordQuantizationOutcome(cwd, makeOutcome({ bitDepth: 32 }));
    const outcomes = await loadQuantizationOutcomes(cwd);
    expect(outcomes).toHaveLength(3);
    const depths = outcomes.map((o) => o.bitDepth);
    expect(depths).toContain(8);
    expect(depths).toContain(16);
    expect(depths).toContain(32);
  });

  test("each record has id and timestamp", async () => {
    const id = await recordQuantizationOutcome(cwd, makeOutcome());
    const outcomes = await loadQuantizationOutcomes(cwd);
    expect(outcomes[0]!.id).toBe(id);
    expect(typeof outcomes[0]!.timestamp).toBe("string");
    expect(new Date(outcomes[0]!.timestamp).getTime()).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// buildQuantizationReport
// ---------------------------------------------------------------------------

describe("buildQuantizationReport — ROI computation and recommendations", () => {
  function makeOutcomes(): QuantizationOutcomeRecord[] {
    const classes: QueryComplexityClass[] = ["semantic-broad", "rare-term"];
    const depths: QuantizationBitDepth[] = [8, 16, 32];
    const latencies: Record<number, number> = { 8: 45, 16: 70, 32: 100 };
    const recalls: Record<number, number> = { 8: 0.96, 16: 0.99, 32: 1.0 };

    const outcomes: QuantizationOutcomeRecord[] = [];
    let idx = 0;

    for (const cc of classes) {
      for (const depth of depths) {
        for (let i = 0; i < 5; i++) {
          outcomes.push({
            id: `qs-${idx++}`,
            timestamp: new Date().toISOString(),
            complexityClass: cc,
            bitDepth: depth,
            genomeSizeTier: "large",
            latencyMs: latencies[depth]!,
            recallAtK: recalls[depth]!,
            tokenCost: 0,
            usedBucketSearch: true,
            matchQuality: 0.85,
          });
        }
      }
    }
    return outcomes;
  }

  test("report has expected shape", () => {
    const report = buildQuantizationReport(makeOutcomes(), "large");
    expect(typeof report.generatedAt).toBe("string");
    expect(report.genomeSizeTier).toBe("large");
    expect(typeof report.totalOutcomes).toBe("number");
    expect(Array.isArray(report.strategies)).toBe(true);
    expect(report.recommendation).toBeDefined();
    expect(typeof report.recommendation.rationale).toBe("string");
  });

  test("strategies contain latencyReductionFraction relative to float32 baseline", () => {
    const report = buildQuantizationReport(makeOutcomes(), "large");
    const int8Semantic = report.strategies.find(
      (s) => s.bitDepth === 8 && s.complexityClass === "semantic-broad",
    );
    expect(int8Semantic).toBeDefined();
    // latency 45 vs baseline 100 → 55% reduction
    expect(int8Semantic!.latencyReductionFraction).toBeCloseTo(0.55, 1);
  });

  test("strategies sorted by roiScore descending", () => {
    const report = buildQuantizationReport(makeOutcomes(), "large");
    for (let i = 1; i < report.strategies.length; i++) {
      expect(report.strategies[i - 1]!.roiScore).toBeGreaterThanOrEqual(
        report.strategies[i]!.roiScore,
      );
    }
  });

  test("recommendation rationale mentions the bit depth", () => {
    const report = buildQuantizationReport(makeOutcomes(), "large");
    expect(report.recommendation.rationale.toLowerCase()).toMatch(/int\d+|float32/);
  });

  test("empty outcomes → fallback recommendation with rationale", () => {
    const report = buildQuantizationReport([], "medium");
    expect(report.totalOutcomes).toBe(0);
    expect(report.strategies).toHaveLength(0);
    expect(report.recommendation.rationale).toMatch(/default/i);
  });

  test("recallLoss computed correctly (1 - avgRecallAtK)", () => {
    const report = buildQuantizationReport(makeOutcomes(), "large");
    const int8 = report.strategies.find(
      (s) => s.bitDepth === 8 && s.complexityClass === "semantic-broad",
    );
    expect(int8!.recallLoss).toBeCloseTo(0.04, 2);
  });
});

// ---------------------------------------------------------------------------
// QuantizationStrategyEngine — end-to-end
// ---------------------------------------------------------------------------

describe("QuantizationStrategyEngine — end-to-end", () => {
  let cwd: string;
  let engine: QuantizationStrategyEngine;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
    engine = new QuantizationStrategyEngine(cwd, { learnerOptions: { minSamples: 3 } });
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("loadProfile returns a GenomeProfile", () => {
    const sections = Array.from({ length: 200 }, (_, i) =>
      makeSection({ path: `s${i}.md`, tokens: 300 }),
    );
    const profile = engine.loadProfile(makeManifest(sections));
    expect(profile.sizeTier).toBe("large");
    expect(profile.sectionCount).toBe(200);
    expect(engine.profile).toBe(profile);
  });

  test("selectBitDepth returns a valid bit depth", async () => {
    const { bitDepth, analysis, genomeSizeTier, outcomeId } = await engine.selectBitDepth(
      "how does the embedding router select models",
    );
    expect([4, 8, 16, 32]).toContain(bitDepth);
    expect(typeof analysis.complexityClass).toBe("string");
    expect(typeof genomeSizeTier).toBe("string");
    expect(typeof outcomeId).toBe("string");
    expect(outcomeId.startsWith("qs-")).toBe(true);
  });

  test("selectBitDepth records a placeholder outcome", async () => {
    await engine.selectBitDepth("find the EmbeddingRouter class");
    const outcomes = await loadQuantizationOutcomes(cwd);
    expect(outcomes.length).toBeGreaterThanOrEqual(1);
  });

  test("recordOutcome appends a corrected record", async () => {
    const { outcomeId } = await engine.selectBitDepth("semantic search embeddings latency");
    await engine.recordOutcome(outcomeId, {
      latencyMs: 42,
      recallAtK: 0.97,
      usedBucketSearch: true,
      matchQuality: 0.9,
    });
    const outcomes = await loadQuantizationOutcomes(cwd);
    // Placeholder + correction = at least 2 records (original selectBitDepth + recordOutcome)
    expect(outcomes.length).toBeGreaterThanOrEqual(2);
    const correction = outcomes.at(-1)!;
    expect(correction.latencyMs).toBe(42);
    expect(correction.recallAtK).toBeCloseTo(0.97, 5);
  });

  test("buildReport returns a QuantizationReport", async () => {
    const report = await engine.buildReport();
    expect(typeof report.generatedAt).toBe("string");
    expect(typeof report.totalOutcomes).toBe("number");
    expect(Array.isArray(report.strategies)).toBe(true);
    expect(report.recommendation).toBeDefined();
  });

  test("large genome profile → heuristic selects int8 for semantic-broad", async () => {
    const sections = Array.from({ length: 200 }, (_, i) =>
      makeSection({ path: `s${i}.md`, tokens: 300 }),
    );
    engine.loadProfile(makeManifest(sections));
    // No historical data → falls back to heuristic
    const { bitDepth, analysis } = await engine.selectBitDepth(
      "how does the caching layer integrate with the router",
    );
    expect(analysis.complexityClass).toBe("semantic-broad");
    expect(bitDepth).toBe(8); // large genome + semantic-broad → int8
  });

  test("tiny genome profile → heuristic selects int16", async () => {
    const sections = Array.from({ length: 10 }, (_, i) =>
      makeSection({ path: `s${i}.md`, tokens: 100 }),
    );
    engine.loadProfile(makeManifest(sections));
    const { bitDepth } = await engine.selectBitDepth(
      "how does the caching layer integrate with the router",
    );
    expect(bitDepth).toBe(16); // tiny genome → int16
  });
});

// ---------------------------------------------------------------------------
// QuantizedANNSearcher — overrideBitDepth parameter
// ---------------------------------------------------------------------------

describe("QuantizedANNSearcher.search() — overrideBitDepth", () => {
  function buildSearcher(bitDepth: 8 | 16): QuantizedANNSearcher {
    const searcher = new QuantizedANNSearcher({ bitDepth, numProjections: 8 });
    const entries = Array.from({ length: 10 }, (_, i) => ({
      id: `section-${i}`,
      embedding: makeEmbedding(64, i + 1),
    }));
    searcher.buildIndex(entries);
    return searcher;
  }

  test("search without override returns selectedBitDepth equal to configured depth", () => {
    const searcher = buildSearcher(8);
    const query = makeEmbedding(64, 99);
    const { selectedBitDepth } = searcher.search(query, 3);
    expect(selectedBitDepth).toBe(8);
  });

  test("search with overrideBitDepth=16 returns selectedBitDepth=16", () => {
    const searcher = buildSearcher(8);
    const query = makeEmbedding(64, 99);
    const { selectedBitDepth, results } = searcher.search(query, 3, 16);
    expect(selectedBitDepth).toBe(16);
    expect(results.length).toBeGreaterThanOrEqual(1);
  });

  test("search with overrideBitDepth=8 on int16 index works", () => {
    const searcher = buildSearcher(16);
    const query = makeEmbedding(64, 99);
    const { selectedBitDepth, results } = searcher.search(query, 3, 8);
    expect(selectedBitDepth).toBe(8);
    expect(results.length).toBeGreaterThanOrEqual(1);
  });

  test("empty index with overrideBitDepth returns empty results", () => {
    const searcher = new QuantizedANNSearcher({ bitDepth: 8 });
    const { results, selectedBitDepth } = searcher.search(makeEmbedding(64), 5, 16);
    expect(results).toHaveLength(0);
    expect(selectedBitDepth).toBe(16);
  });

  test("overrideBitDepth does not affect existing entry bitDepth in index", () => {
    const searcher = buildSearcher(8);
    const entries = searcher.getAllEntries();
    // All indexed entries should still use int8 (index-build depth)
    for (const entry of entries) {
      expect(entry.bitDepth).toBe(8);
    }
  });

  test("top-1 result is stable across overrideBitDepth variants", () => {
    // Use a clearly-dominant similar embedding
    const searcher = new QuantizedANNSearcher({ bitDepth: 8, bucketRadius: 2, numProjections: 4 });
    const base = makeEmbedding(64, 1);
    const similar = base.map((v) => v * 0.99 + 0.005);
    const norm = Math.sqrt(similar.reduce((s, v) => s + v * v, 0));
    const normalizedSimilar = similar.map((v) => v / norm);

    const entries = [
      { id: "similar", embedding: normalizedSimilar },
      ...Array.from({ length: 8 }, (_, i) => ({
        id: `other-${i}`,
        embedding: makeEmbedding(64, i + 10),
      })),
    ];
    searcher.buildIndex(entries);

    const r8 = searcher.search(base, 1);
    const r16 = searcher.search(base, 1, 16);

    // Both should find "similar" as top result
    expect(r8.results[0]!.id).toBe("similar");
    expect(r16.results[0]!.id).toBe("similar");
  });
});

// ---------------------------------------------------------------------------
// EmbeddingRouter — enableQuantizationStrategy integration (structural)
// ---------------------------------------------------------------------------

describe("EmbeddingRouter — enableQuantizationStrategy option", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("creates without error when enableQuantizationStrategy=true", () => {
    expect(
      () => new EmbeddingRouter(cwd, { enableQuantizationStrategy: true }),
    ).not.toThrow();
  });

  test("creates without error when enableQuantizationStrategy=false (default)", () => {
    expect(() => new EmbeddingRouter(cwd)).not.toThrow();
  });

  test("loadManifestProfile returns null when strategy disabled", () => {
    const router = new EmbeddingRouter(cwd);
    const manifest = makeManifest([makeSection()]);
    const result = router.loadManifestProfile(manifest);
    expect(result).toBeNull();
  });

  test("loadManifestProfile returns GenomeProfile when strategy enabled", () => {
    const router = new EmbeddingRouter(cwd, { enableQuantizationStrategy: true });
    const sections = Array.from({ length: 200 }, (_, i) =>
      makeSection({ path: `s${i}.md`, tokens: 300 }),
    );
    const profile = router.loadManifestProfile(makeManifest(sections));
    expect(profile).not.toBeNull();
    expect(profile!.sizeTier).toBe("large");
    expect(profile!.sectionCount).toBe(200);
  });
});
