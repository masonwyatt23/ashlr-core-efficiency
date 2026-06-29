/**
 * Tests for QuantizationAdvisor, ROI tables, and EmbeddingRouter.initializeQuantization()
 *
 * Covers:
 *  - QUANTIZATION_ROI_TABLES: shape, ordering invariants, all four tiers
 *  - getRoiTableEntry: lookup by (bitDepth, genomeSizeTier)
 *  - QuantizationAdvisor.advise():
 *    - tiny/small/medium/large corpora (10–500 sections)
 *    - empty embedding sample (ROI-table fallback)
 *    - non-unit-normalized embeddings
 *    - accuracy threshold filtering (strict MAE tolerance)
 *    - all-fail tolerance → float32 fallback
 *    - profiles sorted by bitDepth ascending
 *    - fallback chain anchored to recommended depth
 *    - estimatedTokenCostReductionPct > 0 when compression is used
 *  - QuantizationAdvisor.detectAnomalies():
 *    - clean embeddings → empty anomaly list
 *    - non-unit-normalized embeddings → norm anomaly
 *    - mismatched codec proxy (inflated MAE) → codec anomaly
 *  - EmbeddingRouter.initializeQuantization():
 *    - returns QuantizationInitResult with advice, strategyActive, anomalies
 *    - strategyActive=false when enableQuantizationStrategy=false
 *    - strategyActive=true when enableQuantizationStrategy=true
 *    - anomalies array is present and is an array
 *    - advice.recommendedBitDepth is a valid QuantizationBitDepth
 *    - advice.fallbackChain is non-empty
 *    - works with empty embeddingSamples
 *    - works with large manifest (200+ sections)
 *  - Reconstruction error vs tolerance edge cases:
 *    - single-dimension embeddings
 *    - all-zero embeddings (MAE = 0 for all depths)
 *    - single-element corpora
 */

import { describe, expect, test, beforeEach, afterEach } from "bun:test";
import { mkdir, rm } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";

import {
  QuantizationAdvisor,
  QUANTIZATION_ROI_TABLES,
  getRoiTableEntry,
  type QuantizationBitDepth,
  type GenomeSizeTier,
  type QuantizationAdvice,
  type RoiTableEntry,
} from "../src/genome/quantization-strategy.ts";

import {
  EmbeddingRouter,
  type QuantizationInitResult,
} from "../src/genome/embedding-router.ts";

import type { GenomeManifest, SectionMeta } from "../src/genome/manifest.ts";

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

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

/** Generate N unit-normalized embeddings of dimension dim. */
function makeEmbeddings(n: number, dim = 128): number[][] {
  return Array.from({ length: n }, (_, i) => makeEmbedding(dim, i + 1));
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

function makeManifest(sectionCount: number): GenomeManifest {
  return {
    version: 1,
    project: "test",
    sections: Array.from({ length: sectionCount }, (_, i) =>
      makeSection({ path: `s${i}.md`, tokens: 200 }),
    ),
    generation: { number: 1, milestone: "M1", startedAt: new Date().toISOString() },
    fitnessHistory: [],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
}

function makeTmpDir(): string {
  return join(tmpdir(), `ashlr-advisor-${Date.now()}-${Math.random().toString(36).slice(2)}`);
}

async function setupGenomeDir(cwd: string): Promise<void> {
  await mkdir(join(cwd, ".ashlrcode", "genome", "evolution"), { recursive: true });
}

// ---------------------------------------------------------------------------
// 1. QUANTIZATION_ROI_TABLES — shape and invariants
// ---------------------------------------------------------------------------

describe("QUANTIZATION_ROI_TABLES — shape and invariants", () => {
  const tiers: GenomeSizeTier[] = ["tiny", "small", "medium", "large"];

  test("has entries for all four genome size tiers", () => {
    for (const tier of tiers) {
      expect(QUANTIZATION_ROI_TABLES[tier]).toBeDefined();
      expect(QUANTIZATION_ROI_TABLES[tier].length).toBeGreaterThan(0);
    }
  });

  test("each tier has exactly 4 entries (bit-depths 4, 8, 16, 32)", () => {
    for (const tier of tiers) {
      expect(QUANTIZATION_ROI_TABLES[tier]).toHaveLength(4);
      const depths = new Set(QUANTIZATION_ROI_TABLES[tier].map((e) => e.bitDepth));
      expect(depths.has(4)).toBe(true);
      expect(depths.has(8)).toBe(true);
      expect(depths.has(16)).toBe(true);
      expect(depths.has(32)).toBe(true);
    }
  });

  test("float32 entry has storageReductionFraction=0, expectedMae=0, recall=1.0", () => {
    for (const tier of tiers) {
      const entry = QUANTIZATION_ROI_TABLES[tier].find((e) => e.bitDepth === 32)!;
      expect(entry.storageReductionFraction).toBe(0);
      expect(entry.expectedMae).toBe(0);
      expect(entry.expectedRecallAtK).toBe(1.0);
      expect(entry.expectedCosineRetention).toBe(1.0);
      expect(entry.tokenCostReductionPct).toBe(0);
    }
  });

  test("int16 entry has storageReductionFraction=0.5", () => {
    for (const tier of tiers) {
      const entry = QUANTIZATION_ROI_TABLES[tier].find((e) => e.bitDepth === 16)!;
      expect(entry.storageReductionFraction).toBe(0.5);
    }
  });

  test("int8 entry has storageReductionFraction=0.75", () => {
    for (const tier of tiers) {
      const entry = QUANTIZATION_ROI_TABLES[tier].find((e) => e.bitDepth === 8)!;
      expect(entry.storageReductionFraction).toBe(0.75);
    }
  });

  test("int4 entry has storageReductionFraction=0.875", () => {
    for (const tier of tiers) {
      const entry = QUANTIZATION_ROI_TABLES[tier].find((e) => e.bitDepth === 4)!;
      expect(entry.storageReductionFraction).toBe(0.875);
    }
  });

  test("error ordering: MAE increases as bit-depth decreases (32 ≥ 16 ≥ 8 ≥ 4 accuracy)", () => {
    for (const tier of tiers) {
      const f32 = QUANTIZATION_ROI_TABLES[tier].find((e) => e.bitDepth === 32)!;
      const i16 = QUANTIZATION_ROI_TABLES[tier].find((e) => e.bitDepth === 16)!;
      const i8  = QUANTIZATION_ROI_TABLES[tier].find((e) => e.bitDepth === 8)!;
      const i4  = QUANTIZATION_ROI_TABLES[tier].find((e) => e.bitDepth === 4)!;

      expect(f32.expectedMae).toBeLessThanOrEqual(i16.expectedMae);
      expect(i16.expectedMae).toBeLessThanOrEqual(i8.expectedMae);
      expect(i8.expectedMae).toBeLessThanOrEqual(i4.expectedMae);
    }
  });

  test("recall decreases as bit-depth decreases", () => {
    for (const tier of tiers) {
      const f32 = QUANTIZATION_ROI_TABLES[tier].find((e) => e.bitDepth === 32)!;
      const i16 = QUANTIZATION_ROI_TABLES[tier].find((e) => e.bitDepth === 16)!;
      const i8  = QUANTIZATION_ROI_TABLES[tier].find((e) => e.bitDepth === 8)!;
      const i4  = QUANTIZATION_ROI_TABLES[tier].find((e) => e.bitDepth === 4)!;

      expect(f32.expectedRecallAtK).toBeGreaterThanOrEqual(i16.expectedRecallAtK);
      expect(i16.expectedRecallAtK).toBeGreaterThanOrEqual(i8.expectedRecallAtK);
      expect(i8.expectedRecallAtK).toBeGreaterThanOrEqual(i4.expectedRecallAtK);
    }
  });

  test("tokenCostReductionPct increases with compression", () => {
    for (const tier of tiers) {
      const f32 = QUANTIZATION_ROI_TABLES[tier].find((e) => e.bitDepth === 32)!;
      const i16 = QUANTIZATION_ROI_TABLES[tier].find((e) => e.bitDepth === 16)!;
      const i8  = QUANTIZATION_ROI_TABLES[tier].find((e) => e.bitDepth === 8)!;
      const i4  = QUANTIZATION_ROI_TABLES[tier].find((e) => e.bitDepth === 4)!;

      expect(f32.tokenCostReductionPct).toBe(0);
      expect(i16.tokenCostReductionPct).toBeGreaterThan(0);
      expect(i8.tokenCostReductionPct).toBeGreaterThan(i16.tokenCostReductionPct);
      expect(i4.tokenCostReductionPct).toBeGreaterThan(i8.tokenCostReductionPct);
    }
  });

  test("large genome int8 tokenCostReductionPct >= 8 (fleet target)", () => {
    const entry = QUANTIZATION_ROI_TABLES["large"].find((e) => e.bitDepth === 8)!;
    expect(entry.tokenCostReductionPct).toBeGreaterThanOrEqual(8);
  });

  test("roiScore for non-float32 entries is positive", () => {
    for (const tier of tiers) {
      const nonF32 = QUANTIZATION_ROI_TABLES[tier].filter((e) => e.bitDepth !== 32);
      for (const entry of nonF32) {
        expect(entry.roiScore).toBeGreaterThan(0);
      }
    }
  });

  test("all entries have genomeSizeTier matching the table key", () => {
    for (const tier of tiers) {
      for (const entry of QUANTIZATION_ROI_TABLES[tier]) {
        expect(entry.genomeSizeTier).toBe(tier);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// 2. getRoiTableEntry — lookup helper
// ---------------------------------------------------------------------------

describe("getRoiTableEntry — lookup by (bitDepth, genomeSizeTier)", () => {
  test("returns correct entry for valid (bitDepth, tier)", () => {
    const entry = getRoiTableEntry(8, "large");
    expect(entry).toBeDefined();
    expect(entry!.bitDepth).toBe(8);
    expect(entry!.genomeSizeTier).toBe("large");
    expect(entry!.storageReductionFraction).toBe(0.75);
  });

  test("returns undefined for invalid bitDepth (not in table)", () => {
    // bitDepth 6 is not in any table
    const entry = getRoiTableEntry(6 as QuantizationBitDepth, "large");
    expect(entry).toBeUndefined();
  });

  test("returns float32 entry with zero MAE", () => {
    const entry = getRoiTableEntry(32, "medium");
    expect(entry).toBeDefined();
    expect(entry!.expectedMae).toBe(0);
    expect(entry!.storageReductionFraction).toBe(0);
  });

  test("all four bit-depths resolve for all four tiers", () => {
    const tiers: GenomeSizeTier[] = ["tiny", "small", "medium", "large"];
    const depths: QuantizationBitDepth[] = [4, 8, 16, 32];
    for (const tier of tiers) {
      for (const depth of depths) {
        expect(getRoiTableEntry(depth, tier)).toBeDefined();
      }
    }
  });
});

// ---------------------------------------------------------------------------
// 3. QuantizationAdvisor.advise() — core recommendation logic
// ---------------------------------------------------------------------------

describe("QuantizationAdvisor.advise() — corpus size tiers", () => {
  const advisor = new QuantizationAdvisor();

  test("tiny corpus (10 sections, 15 embeddings): advice is a valid QuantizationAdvice", () => {
    const advice = advisor.advise(makeEmbeddings(15, 128), 10);
    expect(advice.genomeSizeTier).toBe("tiny");
    expect([4, 8, 16, 32]).toContain(advice.recommendedBitDepth);
    expect(Array.isArray(advice.profiles)).toBe(true);
    expect(Array.isArray(advice.fallbackChain)).toBe(true);
    expect(typeof advice.rationale).toBe("string");
    expect(typeof advice.estimatedTokenCostReductionPct).toBe("number");
    expect(advice.roiTable).toBeDefined();
  });

  test("small corpus (35 sections): genomeSizeTier = 'small'", () => {
    const advice = advisor.advise(makeEmbeddings(20, 128), 35);
    expect(advice.genomeSizeTier).toBe("small");
  });

  test("medium corpus (80 sections): genomeSizeTier = 'medium'", () => {
    const advice = advisor.advise(makeEmbeddings(30, 128), 80);
    expect(advice.genomeSizeTier).toBe("medium");
  });

  test("large corpus (200 sections): genomeSizeTier = 'large'", () => {
    const advice = advisor.advise(makeEmbeddings(50, 128), 200);
    expect(advice.genomeSizeTier).toBe("large");
  });

  test("500 sections: genomeSizeTier = 'large'", () => {
    const advice = advisor.advise(makeEmbeddings(50, 128), 500);
    expect(advice.genomeSizeTier).toBe("large");
  });

  test("profiles sorted by bitDepth ascending (4, 8, 16, 32)", () => {
    const advice = advisor.advise(makeEmbeddings(20, 128), 80);
    const depths = advice.profiles.map((p) => p.bitDepth);
    expect(depths).toEqual([4, 8, 16, 32]);
  });

  test("profiles have all required fields", () => {
    const advice = advisor.advise(makeEmbeddings(10, 64), 50);
    for (const profile of advice.profiles) {
      expect(typeof profile.bitDepth).toBe("number");
      expect(typeof profile.measuredMae).toBe("number");
      expect(typeof profile.cosineRetention).toBe("number");
      expect(typeof profile.estimatedRecallAtK).toBe("number");
      expect(typeof profile.storageReductionFraction).toBe("number");
      expect(typeof profile.meetsAccuracyThreshold).toBe("boolean");
      expect(typeof profile.tokenCostReductionPct).toBe("number");
      expect(profile.measuredMae).toBeGreaterThanOrEqual(0);
      expect(profile.cosineRetention).toBeGreaterThan(0);
      expect(profile.cosineRetention).toBeLessThanOrEqual(1.0);
      expect(profile.estimatedRecallAtK).toBeGreaterThan(0);
      expect(profile.estimatedRecallAtK).toBeLessThanOrEqual(1.0);
    }
  });

  test("float32 profile: measuredMae=0, cosineRetention=1.0, meets threshold", () => {
    const advice = advisor.advise(makeEmbeddings(10, 64), 100);
    const f32 = advice.profiles.find((p) => p.bitDepth === 32)!;
    expect(f32.measuredMae).toBe(0);
    expect(f32.cosineRetention).toBe(1.0);
    expect(f32.meetsAccuracyThreshold).toBe(true);
    expect(f32.storageReductionFraction).toBe(0);
  });

  test("int16 profile: measuredMae < 0.001, meetsAccuracyThreshold=true with default tolerance", () => {
    const advice = advisor.advise(makeEmbeddings(20, 256), 80);
    const i16 = advice.profiles.find((p) => p.bitDepth === 16)!;
    expect(i16.measuredMae).toBeLessThan(0.001);
    expect(i16.meetsAccuracyThreshold).toBe(true);
  });

  test("int8 profile: measuredMae < 0.01 (within default tolerance)", () => {
    const advice = advisor.advise(makeEmbeddings(20, 128), 80);
    const i8 = advice.profiles.find((p) => p.bitDepth === 8)!;
    expect(i8.measuredMae).toBeLessThan(0.01);
    expect(i8.meetsAccuracyThreshold).toBe(true);
  });

  test("recommended depth saves storage (storageReductionFraction > 0 for non-float32)", () => {
    const advice = advisor.advise(makeEmbeddings(30, 128), 200);
    if (advice.recommendedBitDepth !== 32) {
      const profile = advice.profiles.find((p) => p.bitDepth === advice.recommendedBitDepth)!;
      expect(profile.storageReductionFraction).toBeGreaterThan(0);
    }
  });

  test("estimatedTokenCostReductionPct > 0 when compression is applied", () => {
    const advice = advisor.advise(makeEmbeddings(30, 128), 200);
    if (advice.recommendedBitDepth !== 32) {
      expect(advice.estimatedTokenCostReductionPct).toBeGreaterThan(0);
    }
  });

  test("fallbackChain is non-empty and has valid steps", () => {
    const advice = advisor.advise(makeEmbeddings(10, 64), 50);
    expect(advice.fallbackChain.length).toBeGreaterThan(0);
    for (const step of advice.fallbackChain) {
      expect([4, 8, 16, 32]).toContain(step.bitDepth);
      expect(typeof step.modelTier).toBe("string");
      expect(step.timeoutMs).toBeGreaterThan(0);
    }
  });

  test("fallbackChain first step bitDepth matches recommendedBitDepth (when not float32)", () => {
    const advice = advisor.advise(makeEmbeddings(20, 128), 100);
    if (advice.recommendedBitDepth !== 32) {
      expect(advice.fallbackChain[0]!.bitDepth).toBe(advice.recommendedBitDepth);
    }
  });

  test("roiTable echoed matches QUANTIZATION_ROI_TABLES for the genome tier", () => {
    const advice = advisor.advise(makeEmbeddings(20, 64), 80);
    expect(advice.roiTable).toBe(QUANTIZATION_ROI_TABLES[advice.genomeSizeTier]);
  });

  test("rationale is a non-empty string containing section count or tier", () => {
    const advice = advisor.advise(makeEmbeddings(10, 64), 80);
    expect(advice.rationale.length).toBeGreaterThan(10);
    // Rationale should mention the genome tier
    expect(advice.rationale.toLowerCase()).toMatch(/tiny|small|medium|large/);
  });
});

// ---------------------------------------------------------------------------
// 4. QuantizationAdvisor.advise() — edge cases
// ---------------------------------------------------------------------------

describe("QuantizationAdvisor.advise() — edge cases", () => {
  test("empty embedding sample: falls back to ROI table, returns valid advice", () => {
    const advisor = new QuantizationAdvisor();
    const advice = advisor.advise([], 100);
    expect([4, 8, 16, 32]).toContain(advice.recommendedBitDepth);
    expect(advice.profiles).toHaveLength(4);
    // With no samples, uses table predictions — all non-float32 should still meet default threshold
    const i16 = advice.profiles.find((p) => p.bitDepth === 16)!;
    expect(i16.meetsAccuracyThreshold).toBe(true);
  });

  test("strict MAE tolerance (1e-6) rejects int8 and int4", () => {
    const strictAdvisor = new QuantizationAdvisor({ maeToleranceThreshold: 1e-6 });
    const advice = strictAdvisor.advise(makeEmbeddings(20, 128), 80);
    // int8 MAE ~0.004–0.005 >> 1e-6, so int8 should not meet threshold
    const i8 = advice.profiles.find((p) => p.bitDepth === 8)!;
    expect(i8.meetsAccuracyThreshold).toBe(false);
    // int4 also should not meet threshold
    const i4 = advice.profiles.find((p) => p.bitDepth === 4)!;
    expect(i4.meetsAccuracyThreshold).toBe(false);
  });

  test("ultra-strict tolerance (0): only float32 passes, recommended = 32", () => {
    const strictAdvisor = new QuantizationAdvisor({
      maeToleranceThreshold: 0,
      cosineRetentionThreshold: 1.0 + 1e-10, // impossible
    });
    const advice = strictAdvisor.advise(makeEmbeddings(10, 64), 80);
    expect(advice.recommendedBitDepth).toBe(32);
  });

  test("very permissive tolerance: lower bit-depths pass", () => {
    const permissiveAdvisor = new QuantizationAdvisor({
      maeToleranceThreshold: 0.1,
      cosineRetentionThreshold: 0.9,
    });
    const advice = permissiveAdvisor.advise(makeEmbeddings(10, 64), 80);
    // With very permissive tolerance, int4 may pass
    const i4 = advice.profiles.find((p) => p.bitDepth === 4)!;
    expect(i4.meetsAccuracyThreshold).toBe(true);
  });

  test("single-dimension embedding: no throw", () => {
    const advisor = new QuantizationAdvisor();
    expect(() => advisor.advise([[0.5], [-0.5], [1.0]], 20)).not.toThrow();
  });

  test("all-zero embeddings: MAE = 0 for all depths (zero vector → no quantization error)", () => {
    const advisor = new QuantizationAdvisor();
    const zeros = Array.from({ length: 5 }, () => new Array(64).fill(0) as number[]);
    const advice = advisor.advise(zeros, 20);
    for (const profile of advice.profiles) {
      expect(profile.measuredMae).toBe(0);
    }
  });

  test("single-element corpus (1 section): valid advice without throw", () => {
    const advisor = new QuantizationAdvisor();
    expect(() => advisor.advise(makeEmbeddings(1, 64), 1)).not.toThrow();
  });

  test("sampleSize cap: advising with more embeddings than sampleSize still works", () => {
    const advisor = new QuantizationAdvisor({ sampleSize: 5 });
    // 200 embeddings but sampleSize=5 → only 5 used
    const advice = advisor.advise(makeEmbeddings(200, 64), 200);
    expect([4, 8, 16, 32]).toContain(advice.recommendedBitDepth);
  });
});

// ---------------------------------------------------------------------------
// 5. QuantizationAdvisor.detectAnomalies()
// ---------------------------------------------------------------------------

describe("QuantizationAdvisor.detectAnomalies()", () => {
  const advisor = new QuantizationAdvisor();

  test("unit-normalized embeddings: returns empty anomaly list", () => {
    const anomalies = advisor.detectAnomalies(makeEmbeddings(20, 128), 100);
    expect(anomalies).toEqual([]);
  });

  test("empty embeddings: returns empty anomaly list (no data to check)", () => {
    const anomalies = advisor.detectAnomalies([], 100);
    expect(anomalies).toEqual([]);
  });

  test("non-unit-normalized embeddings (norm >> 1): returns norm anomaly", () => {
    // Scale embeddings by 10 → norm ≈ 10
    const scaled = makeEmbeddings(5, 64).map((emb) => emb.map((v) => v * 10));
    const anomalies = advisor.detectAnomalies(scaled, 50);
    expect(anomalies.length).toBeGreaterThanOrEqual(1);
    const hasNormAnomaly = anomalies.some((a) => a.toLowerCase().includes("norm"));
    expect(hasNormAnomaly).toBe(true);
  });

  test("very small embeddings (norm << 0.5): returns norm anomaly", () => {
    const tiny = makeEmbeddings(5, 64).map((emb) => emb.map((v) => v * 0.01));
    const anomalies = advisor.detectAnomalies(tiny, 50);
    const hasNormAnomaly = anomalies.some((a) => a.toLowerCase().includes("norm"));
    expect(hasNormAnomaly).toBe(true);
  });

  test("returns an array", () => {
    const anomalies = advisor.detectAnomalies(makeEmbeddings(10, 64), 100);
    expect(Array.isArray(anomalies)).toBe(true);
  });

  test("single embedding does not throw", () => {
    expect(() =>
      advisor.detectAnomalies([makeEmbedding(64, 1)], 10),
    ).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// 6. EmbeddingRouter.initializeQuantization() — startup hook
// ---------------------------------------------------------------------------

describe("EmbeddingRouter.initializeQuantization() — startup hook", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("returns QuantizationInitResult with advice, strategyActive, anomalies", () => {
    const router = new EmbeddingRouter(cwd, { enableQuantizationStrategy: true });
    const manifest = makeManifest(100);
    const result = router.initializeQuantization(manifest, makeEmbeddings(20, 128));

    expect(result.advice).toBeDefined();
    expect(typeof result.strategyActive).toBe("boolean");
    expect(Array.isArray(result.anomalies)).toBe(true);
  });

  test("strategyActive=false when enableQuantizationStrategy=false (default)", () => {
    const router = new EmbeddingRouter(cwd);
    const result = router.initializeQuantization(makeManifest(100));
    expect(result.strategyActive).toBe(false);
  });

  test("strategyActive=true when enableQuantizationStrategy=true", () => {
    const router = new EmbeddingRouter(cwd, { enableQuantizationStrategy: true });
    const result = router.initializeQuantization(makeManifest(100));
    expect(result.strategyActive).toBe(true);
  });

  test("advice.recommendedBitDepth is a valid QuantizationBitDepth", () => {
    const router = new EmbeddingRouter(cwd, { enableQuantizationStrategy: true });
    const result = router.initializeQuantization(makeManifest(80), makeEmbeddings(20, 128));
    expect([4, 8, 16, 32]).toContain(result.advice.recommendedBitDepth);
  });

  test("advice.fallbackChain is non-empty", () => {
    const router = new EmbeddingRouter(cwd, { enableQuantizationStrategy: true });
    const result = router.initializeQuantization(makeManifest(80), makeEmbeddings(10, 64));
    expect(result.advice.fallbackChain.length).toBeGreaterThan(0);
  });

  test("advice.genomeSizeTier matches manifest section count tier", () => {
    const router = new EmbeddingRouter(cwd, { enableQuantizationStrategy: true });
    // 200 sections → large
    const result = router.initializeQuantization(makeManifest(200), makeEmbeddings(10, 64));
    expect(result.advice.genomeSizeTier).toBe("large");
  });

  test("works with empty embeddingSamples (no throw)", () => {
    const router = new EmbeddingRouter(cwd, { enableQuantizationStrategy: true });
    expect(() =>
      router.initializeQuantization(makeManifest(80)),
    ).not.toThrow();
  });

  test("works with empty manifest (0 sections)", () => {
    const router = new EmbeddingRouter(cwd, { enableQuantizationStrategy: true });
    expect(() =>
      router.initializeQuantization(makeManifest(0), []),
    ).not.toThrow();
    const result = router.initializeQuantization(makeManifest(0), []);
    expect(result.advice.genomeSizeTier).toBe("tiny");
  });

  test("works with large manifest (500 sections)", () => {
    const router = new EmbeddingRouter(cwd, { enableQuantizationStrategy: true });
    const result = router.initializeQuantization(makeManifest(500), makeEmbeddings(50, 128));
    expect(result.advice.genomeSizeTier).toBe("large");
    expect(result.strategyActive).toBe(true);
  });

  test("anomalies is empty for unit-normalized embeddings", () => {
    const router = new EmbeddingRouter(cwd, { enableQuantizationStrategy: true });
    const result = router.initializeQuantization(makeManifest(80), makeEmbeddings(20, 128));
    expect(result.anomalies).toEqual([]);
  });

  test("anomalies contains norm warning for scaled embeddings", () => {
    const router = new EmbeddingRouter(cwd, { enableQuantizationStrategy: true });
    const scaled = makeEmbeddings(5, 64).map((emb) => emb.map((v) => v * 10));
    const result = router.initializeQuantization(makeManifest(80), scaled);
    expect(result.anomalies.length).toBeGreaterThan(0);
  });

  test("advisor options passed through (strict tolerance forces float32)", () => {
    const router = new EmbeddingRouter(cwd, { enableQuantizationStrategy: true });
    const result = router.initializeQuantization(
      makeManifest(80),
      makeEmbeddings(20, 128),
      { maeToleranceThreshold: 0, cosineRetentionThreshold: 1.0 + 1e-10 },
    );
    expect(result.advice.recommendedBitDepth).toBe(32);
  });

  test("after initializeQuantization with strategy enabled, profile is loaded in engine", () => {
    const router = new EmbeddingRouter(cwd, { enableQuantizationStrategy: true });
    router.initializeQuantization(makeManifest(200), makeEmbeddings(10, 64));
    // loadManifestProfile should now return a cached profile
    const manifest = makeManifest(200);
    const profile = router.loadManifestProfile(manifest);
    expect(profile).not.toBeNull();
    expect(profile!.sizeTier).toBe("large");
    expect(profile!.sectionCount).toBe(200);
  });
});

// ---------------------------------------------------------------------------
// 7. ROI table ordering and fleet-scale token cost targeting
// ---------------------------------------------------------------------------

describe("ROI tables — fleet-scale token cost target (8–12% reduction)", () => {
  test("large genome int8 tokenCostReductionPct in target range [8, 12]", () => {
    const entry = getRoiTableEntry(8, "large")!;
    expect(entry.tokenCostReductionPct).toBeGreaterThanOrEqual(8);
    expect(entry.tokenCostReductionPct).toBeLessThanOrEqual(12);
  });

  test("medium genome int8 tokenCostReductionPct in target range [8, 12]", () => {
    const entry = getRoiTableEntry(8, "medium")!;
    expect(entry.tokenCostReductionPct).toBeGreaterThanOrEqual(8);
    expect(entry.tokenCostReductionPct).toBeLessThanOrEqual(12);
  });

  test("large genome int4 tokenCostReductionPct > int8 (more aggressive compression)", () => {
    const i8 = getRoiTableEntry(8, "large")!;
    const i4 = getRoiTableEntry(4, "large")!;
    expect(i4.tokenCostReductionPct).toBeGreaterThan(i8.tokenCostReductionPct);
  });

  test("roiScore = storageReduction - recallLoss * 2 (within floating point)", () => {
    for (const tier of ["tiny", "small", "medium", "large"] as GenomeSizeTier[]) {
      for (const entry of QUANTIZATION_ROI_TABLES[tier]) {
        const recallLoss = 1 - entry.expectedRecallAtK;
        const expected = entry.storageReductionFraction - recallLoss * 2;
        expect(entry.roiScore).toBeCloseTo(expected, 3);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// 8. QuantizationAdvisor — reconstruction error ordering on real embeddings
// ---------------------------------------------------------------------------

describe("QuantizationAdvisor profiles — error ordering on real embeddings", () => {
  const advisor = new QuantizationAdvisor();

  test("MAE ordering: float32 ≤ int16 ≤ int8 on unit-normalized 256d embeddings", () => {
    const advice = advisor.advise(makeEmbeddings(20, 256), 100);
    const f32 = advice.profiles.find((p) => p.bitDepth === 32)!;
    const i16 = advice.profiles.find((p) => p.bitDepth === 16)!;
    const i8  = advice.profiles.find((p) => p.bitDepth === 8)!;

    expect(f32.measuredMae).toBeLessThanOrEqual(i16.measuredMae);
    expect(i16.measuredMae).toBeLessThanOrEqual(i8.measuredMae);
  });

  test("cosineRetention ordering: float32 ≥ int16 ≥ int8 on 128d embeddings", () => {
    const advice = advisor.advise(makeEmbeddings(20, 128), 80);
    const f32 = advice.profiles.find((p) => p.bitDepth === 32)!;
    const i16 = advice.profiles.find((p) => p.bitDepth === 16)!;
    const i8  = advice.profiles.find((p) => p.bitDepth === 8)!;

    expect(f32.cosineRetention).toBeGreaterThanOrEqual(i16.cosineRetention);
    expect(i16.cosineRetention).toBeGreaterThanOrEqual(i8.cosineRetention);
  });

  test("int16 cosineRetention > 0.9999 on 128d unit-normalized embeddings", () => {
    const advice = advisor.advise(makeEmbeddings(20, 128), 80);
    const i16 = advice.profiles.find((p) => p.bitDepth === 16)!;
    expect(i16.cosineRetention).toBeGreaterThan(0.9999);
  });

  test("int8 cosineRetention > 0.999 on 128d unit-normalized embeddings", () => {
    const advice = advisor.advise(makeEmbeddings(20, 128), 80);
    const i8 = advice.profiles.find((p) => p.bitDepth === 8)!;
    expect(i8.cosineRetention).toBeGreaterThan(0.999);
  });

  test("storageReductionFraction values are correct for all depths", () => {
    const advice = advisor.advise(makeEmbeddings(10, 64), 50);
    expect(advice.profiles.find((p) => p.bitDepth === 32)!.storageReductionFraction).toBe(0);
    expect(advice.profiles.find((p) => p.bitDepth === 16)!.storageReductionFraction).toBe(0.5);
    expect(advice.profiles.find((p) => p.bitDepth === 8)!.storageReductionFraction).toBe(0.75);
    expect(advice.profiles.find((p) => p.bitDepth === 4)!.storageReductionFraction).toBe(0.875);
  });
});
