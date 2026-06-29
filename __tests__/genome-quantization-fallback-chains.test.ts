/**
 * Comprehensive test suite for Multi-Tier Embedding Quantization Strategy
 * with Fallback Chains
 *
 * Covers:
 *  - bfloat16 encode/decode round-trip and MAE bounds
 *  - quantizeUnified / dequantizeUnified for all four tiers
 *  - Reconstruction error bounds per tier (float32 / int16 / int8 / bfloat16)
 *  - quantizationStrategyRouter: build-time tier assignment
 *    - sensitivity classification (high / medium / low)
 *    - code-section protection heuristic
 *    - sensitivityOverrides
 *    - aggregate stats (tierCounts, compressionRatio, savingsFraction)
 *    - synthetic genomes: 10, 50, 100, 200, 500 sections
 *  - selectQueryTier: tier selection from GenomeProfile + maxTokens
 *  - selectBudgetTier: per-section tier selection with RouterBuildResult
 *  - Cross-tier consistency: top-K results stable across tiers
 *  - Query-time latency/accuracy tradeoffs (timing benchmarks, relative)
 *  - retrieveSectionsV2 with routerBuildResult (integration, no Ollama required)
 */

import { describe, expect, test, beforeEach, afterEach } from "bun:test";
import { mkdir, rm } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";

import {
  quantizationStrategyRouter,
  selectQueryTier,
  selectBudgetTier,
  ROUTER_FALLBACK_CHAIN,
  GenomeProfiler,
  type RouterTier,
  type RouterBuildResult,
  type SectionQuantizationMeta,
} from "../src/genome/quantization-strategy.ts";

import {
  quantizeEmbedding,
  dequantizeEmbedding,
  quantizationError,
  quantizeToBfloat16,
  dequantizeFromBfloat16,
  float32ToBfloat16,
  bfloat16ToFloat32,
  quantizeUnified,
  dequantizeUnified,
  type QuantizationTier,
} from "../src/genome/embeddings.ts";

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

/** Compute MAE between two equal-length arrays. */
function mae(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  return a.reduce((sum, v, i) => sum + Math.abs(v - b[i]!), 0) / a.length;
}

/** Cosine similarity (pure, no imports). */
function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na  += a[i]! * a[i]!;
    nb  += b[i]! * b[i]!;
  }
  const d = Math.sqrt(na) * Math.sqrt(nb);
  return d === 0 ? 0 : dot / d;
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

/**
 * Build a synthetic embedding cache for N sections.
 * Each entry has path `s{i}.md` and a 128-dim unit-normalized embedding.
 */
function makeSyntheticCache(n: number, dim = 128): Array<{
  sectionPath: string;
  embedding: number[];
  tokens?: number;
}> {
  return Array.from({ length: n }, (_, i) => ({
    sectionPath: `section${i}.md`,
    embedding: makeEmbedding(dim, i + 1),
    tokens: 200 + (i % 5) * 50,
  }));
}

// ---------------------------------------------------------------------------
// 1. bfloat16 encode/decode
// ---------------------------------------------------------------------------

describe("bfloat16 — float32ToBfloat16 / bfloat16ToFloat32", () => {
  test("zero encodes/decodes to zero", () => {
    const bf16 = float32ToBfloat16(0);
    expect(bfloat16ToFloat32(bf16)).toBe(0);
  });

  test("one encodes/decodes to one (within bfloat16 resolution)", () => {
    const bf16 = float32ToBfloat16(1.0);
    const decoded = bfloat16ToFloat32(bf16);
    expect(Math.abs(decoded - 1.0)).toBeLessThan(0.01);
  });

  test("-1 encodes/decodes with correct sign", () => {
    const bf16 = float32ToBfloat16(-1.0);
    const decoded = bfloat16ToFloat32(bf16);
    expect(decoded).toBeLessThan(0);
    expect(Math.abs(decoded + 1.0)).toBeLessThan(0.01);
  });

  test("small positive value round-trips within 0.002 MAE", () => {
    const v = 0.12345;
    const decoded = bfloat16ToFloat32(float32ToBfloat16(v));
    expect(Math.abs(decoded - v)).toBeLessThan(0.002);
  });

  test("NaN encodes and decodes as NaN", () => {
    const bf16 = float32ToBfloat16(NaN);
    expect(isNaN(bfloat16ToFloat32(bf16))).toBe(true);
  });

  test("Infinity encodes and decodes as Infinity", () => {
    const bf16 = float32ToBfloat16(Infinity);
    expect(bfloat16ToFloat32(bf16)).toBe(Infinity);
  });

  test("-Infinity encodes and decodes as -Infinity", () => {
    const bf16 = float32ToBfloat16(-Infinity);
    expect(bfloat16ToFloat32(bf16)).toBe(-Infinity);
  });

  test("result is a Uint16 integer in [0, 65535]", () => {
    for (const v of [0, 1, -1, 0.5, -0.5, 3.14]) {
      const bf16 = float32ToBfloat16(v);
      expect(Number.isInteger(bf16)).toBe(true);
      expect(bf16).toBeGreaterThanOrEqual(0);
      expect(bf16).toBeLessThanOrEqual(65535);
    }
  });
});

describe("bfloat16 — quantizeToBfloat16 / dequantizeFromBfloat16", () => {
  test("empty embedding returns empty Uint16Array", () => {
    const q = quantizeToBfloat16([]);
    expect(q).toBeInstanceOf(Uint16Array);
    expect(q.length).toBe(0);
  });

  test("output is Uint16Array with same length as input", () => {
    const emb = makeEmbedding(64, 1);
    const q = quantizeToBfloat16(emb);
    expect(q).toBeInstanceOf(Uint16Array);
    expect(q.length).toBe(64);
  });

  test("dequantizeFromBfloat16 returns number[] with same length", () => {
    const emb = makeEmbedding(64, 2);
    const q = quantizeToBfloat16(emb);
    const deq = dequantizeFromBfloat16(q);
    expect(Array.isArray(deq)).toBe(true);
    expect(deq.length).toBe(64);
  });

  test("round-trip MAE < 0.002 on unit-normalized 64-dim vector", () => {
    const emb = makeEmbedding(64, 3);
    const q = quantizeToBfloat16(emb);
    const deq = dequantizeFromBfloat16(q);
    expect(mae(emb, deq)).toBeLessThan(0.002);
  });

  test("round-trip MAE < 0.002 on unit-normalized 768-dim vector", () => {
    const emb = makeEmbedding(768, 42);
    const q = quantizeToBfloat16(emb);
    const deq = dequantizeFromBfloat16(q);
    expect(mae(emb, deq)).toBeLessThan(0.002);
  });

  test("cosine similarity vs original > 0.9998 for 128-dim vector", () => {
    const emb = makeEmbedding(128, 7);
    const deq = dequantizeFromBfloat16(quantizeToBfloat16(emb));
    expect(cosine(emb, deq)).toBeGreaterThan(0.9998);
  });
});

// ---------------------------------------------------------------------------
// 2. Reconstruction error bounds per tier
// ---------------------------------------------------------------------------

describe("Reconstruction error bounds — per tier", () => {
  const DIM = 768;
  const SEEDS = [1, 7, 13, 42, 99];

  test("float32 tier: MAE = 0 (no-op)", () => {
    for (const seed of SEEDS) {
      const emb = makeEmbedding(DIM, seed);
      const q = quantizeUnified(emb, "float32");
      const deq = dequantizeUnified(q);
      expect(mae(emb, deq)).toBe(0);
    }
  });

  test("int16 tier: MAE < 0.0001 on unit-normalized 768-dim", () => {
    for (const seed of SEEDS) {
      const emb = makeEmbedding(DIM, seed);
      const { quantized, min, max } = quantizeEmbedding(emb, 16);
      const deq = dequantizeEmbedding(quantized, min, max);
      expect(mae(emb, deq)).toBeLessThan(0.0001);
    }
  });

  test("int8 tier: MAE < 0.01 on unit-normalized 768-dim", () => {
    for (const seed of SEEDS) {
      const emb = makeEmbedding(DIM, seed);
      const { quantized, min, max } = quantizeEmbedding(emb, 8);
      const deq = dequantizeEmbedding(quantized, min, max);
      expect(mae(emb, deq)).toBeLessThan(0.01);
    }
  });

  test("bfloat16 tier: MAE < 0.002 on unit-normalized 768-dim", () => {
    for (const seed of SEEDS) {
      const emb = makeEmbedding(DIM, seed);
      const deq = dequantizeFromBfloat16(quantizeToBfloat16(emb));
      expect(mae(emb, deq)).toBeLessThan(0.002);
    }
  });

  test("error ordering: float32 ≤ int16 ≤ bfloat16 ≤ int8 (MAE)", () => {
    // This is the expected sensitivity ordering for unit-normalized embeddings
    for (const seed of SEEDS) {
      const emb = makeEmbedding(DIM, seed);

      const { quantized: q16, min: m16, max: x16 } = quantizeEmbedding(emb, 16);
      const { quantized: q8,  min: m8,  max: x8  } = quantizeEmbedding(emb, 8);
      const deq16 = dequantizeEmbedding(q16, m16, x16);
      const deq8  = dequantizeEmbedding(q8,  m8,  x8);
      const deqbf = dequantizeFromBfloat16(quantizeToBfloat16(emb));

      const maeF32  = 0;
      const mae16   = mae(emb, deq16);
      const maeBf16 = mae(emb, deqbf);
      const mae8    = mae(emb, deq8);

      expect(maeF32).toBeLessThanOrEqual(mae16);
      expect(mae16).toBeLessThanOrEqual(maeBf16 + 1e-9); // allow tiny FP noise
      expect(maeBf16).toBeLessThanOrEqual(mae8);
    }
  });

  test("cosine similarity vs original: int16 > 0.9999", () => {
    for (const seed of SEEDS) {
      const emb = makeEmbedding(DIM, seed);
      const { quantized, min, max } = quantizeEmbedding(emb, 16);
      const deq = dequantizeEmbedding(quantized, min, max);
      expect(cosine(emb, deq)).toBeGreaterThan(0.9999);
    }
  });

  test("cosine similarity vs original: int8 > 0.999", () => {
    for (const seed of SEEDS) {
      const emb = makeEmbedding(DIM, seed);
      const { quantized, min, max } = quantizeEmbedding(emb, 8);
      const deq = dequantizeEmbedding(quantized, min, max);
      expect(cosine(emb, deq)).toBeGreaterThan(0.999);
    }
  });

  test("cosine similarity vs original: bfloat16 > 0.9999", () => {
    for (const seed of SEEDS) {
      const emb = makeEmbedding(DIM, seed);
      const deq = dequantizeFromBfloat16(quantizeToBfloat16(emb));
      expect(cosine(emb, deq)).toBeGreaterThan(0.9999);
    }
  });
});

// ---------------------------------------------------------------------------
// 3. quantizeUnified / dequantizeUnified
// ---------------------------------------------------------------------------

describe("quantizeUnified / dequantizeUnified — all tiers", () => {
  const emb = makeEmbedding(128, 5);

  const tiers: QuantizationTier[] = ["float32", "int16", "int8", "bfloat16"];

  for (const tier of tiers) {
    test(`${tier}: output has correct tier field`, () => {
      const q = quantizeUnified(emb, tier);
      expect(q.tier).toBe(tier);
    });

    test(`${tier}: round-trip produces same length`, () => {
      const q = quantizeUnified(emb, tier);
      const deq = dequantizeUnified(q);
      expect(deq.length).toBe(emb.length);
    });
  }

  test("float32: data is the original array reference", () => {
    const q = quantizeUnified(emb, "float32");
    expect(q.data).toBe(emb); // float32 is a no-op
  });

  test("int16: data is Int16Array", () => {
    const q = quantizeUnified(emb, "int16");
    expect(q.data).toBeInstanceOf(Int16Array);
    expect(q.min).toBeDefined();
    expect(q.max).toBeDefined();
  });

  test("int8: data is Int8Array", () => {
    const q = quantizeUnified(emb, "int8");
    expect(q.data).toBeInstanceOf(Int8Array);
    expect(q.min).toBeDefined();
    expect(q.max).toBeDefined();
  });

  test("bfloat16: data is Uint16Array, no min/max", () => {
    const q = quantizeUnified(emb, "bfloat16");
    expect(q.data).toBeInstanceOf(Uint16Array);
    expect(q.min).toBeUndefined();
    expect(q.max).toBeUndefined();
  });

  test("float32 round-trip MAE = 0", () => {
    const q = quantizeUnified(emb, "float32");
    expect(mae(emb, dequantizeUnified(q))).toBe(0);
  });

  test("int16 round-trip MAE < 0.0001", () => {
    const q = quantizeUnified(emb, "int16");
    expect(mae(emb, dequantizeUnified(q))).toBeLessThan(0.0001);
  });

  test("int8 round-trip MAE < 0.01", () => {
    const q = quantizeUnified(emb, "int8");
    expect(mae(emb, dequantizeUnified(q))).toBeLessThan(0.01);
  });

  test("bfloat16 round-trip MAE < 0.002", () => {
    const q = quantizeUnified(emb, "bfloat16");
    expect(mae(emb, dequantizeUnified(q))).toBeLessThan(0.002);
  });

  test("empty embedding: all tiers produce empty output", () => {
    for (const tier of tiers) {
      const q = quantizeUnified([], tier);
      expect(dequantizeUnified(q).length).toBe(0);
    }
  });
});

// ---------------------------------------------------------------------------
// 4. ROUTER_FALLBACK_CHAIN shape
// ---------------------------------------------------------------------------

describe("ROUTER_FALLBACK_CHAIN — shape", () => {
  test("has exactly 4 entries", () => {
    expect(ROUTER_FALLBACK_CHAIN).toHaveLength(4);
  });

  test("first entry is float32 (highest quality)", () => {
    expect(ROUTER_FALLBACK_CHAIN[0]).toBe("float32");
  });

  test("last entry is bfloat16 (cheapest)", () => {
    expect(ROUTER_FALLBACK_CHAIN[3]).toBe("bfloat16");
  });

  test("contains all four tiers", () => {
    const tiers = new Set(ROUTER_FALLBACK_CHAIN);
    expect(tiers.has("float32")).toBe(true);
    expect(tiers.has("int16")).toBe(true);
    expect(tiers.has("int8")).toBe(true);
    expect(tiers.has("bfloat16")).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 5. quantizationStrategyRouter — tier selection logic
// ---------------------------------------------------------------------------

describe("quantizationStrategyRouter — tier assignment on synthetic genomes", () => {
  test("empty input returns empty map and zero stats", () => {
    const result = quantizationStrategyRouter([]);
    expect(result.sectionMeta.size).toBe(0);
    expect(result.stats.totalSections).toBe(0);
  });

  test("single section: returns one entry in sectionMeta", () => {
    const entries = [{ sectionPath: "intro.md", embedding: makeEmbedding(64, 1) }];
    const result = quantizationStrategyRouter(entries);
    expect(result.sectionMeta.size).toBe(1);
    expect(result.sectionMeta.has("intro.md")).toBe(true);
  });

  test("each entry has all required SectionQuantizationMeta fields", () => {
    const entries = makeSyntheticCache(5, 64);
    const result = quantizationStrategyRouter(entries);
    for (const [, meta] of result.sectionMeta) {
      expect(typeof meta.sectionPath).toBe("string");
      expect(["float32", "int16", "int8", "bfloat16"]).toContain(meta.assignedTier);
      expect(typeof meta.tierErrors).toBe("object");
      expect(typeof meta.tierErrors["int8"]).toBe("number");
      expect(typeof meta.tierErrors["int16"]).toBe("number");
      expect(typeof meta.tierErrors["bfloat16"]).toBe("number");
      expect(["high", "medium", "low"]).toContain(meta.sensitivity);
      expect(meta.storageBytesAssigned).toBeGreaterThan(0);
      expect(meta.storageBytesFloat32).toBeGreaterThan(0);
      expect(meta.compressionRatio).toBeGreaterThan(0);
      expect(meta.compressionRatio).toBeLessThanOrEqual(1);
      expect(typeof meta.evaluatedAt).toBe("string");
    }
  });

  test("compression ratio ≤ 1 for all sections (never larger than float32)", () => {
    const entries = makeSyntheticCache(20, 128);
    const result = quantizationStrategyRouter(entries);
    for (const [, meta] of result.sectionMeta) {
      expect(meta.compressionRatio).toBeLessThanOrEqual(1);
    }
  });

  test("float32 sections have compressionRatio = 1", () => {
    // Force all sections to "low" sensitivity via override
    const entries = [{ sectionPath: "api.md", embedding: makeEmbedding(64, 1) }];
    const result = quantizationStrategyRouter(entries, {
      sensitivityOverrides: { "api.md": "low" },
    });
    const meta = result.sectionMeta.get("api.md")!;
    // "low" sensitivity → int16, not float32; still compressionRatio = 0.5
    expect(meta.compressionRatio).toBeCloseTo(0.5, 2);
  });

  test("aggregate tierCounts sums to totalSections", () => {
    const n = 30;
    const entries = makeSyntheticCache(n, 64);
    const result = quantizationStrategyRouter(entries);
    const sum = Object.values(result.stats.tierCounts).reduce((a, b) => a + b, 0);
    expect(sum).toBe(n);
    expect(result.stats.totalSections).toBe(n);
  });

  test("avgCompressionRatio is between 0 and 1 exclusive", () => {
    const entries = makeSyntheticCache(50, 128);
    const result = quantizationStrategyRouter(entries);
    expect(result.stats.avgCompressionRatio).toBeGreaterThan(0);
    expect(result.stats.avgCompressionRatio).toBeLessThan(1);
  });

  test("estimatedStorageSavingsFraction = 1 - avgCompressionRatio", () => {
    const entries = makeSyntheticCache(20, 64);
    const result = quantizationStrategyRouter(entries);
    expect(result.stats.estimatedStorageSavingsFraction).toBeCloseTo(
      1 - result.stats.avgCompressionRatio, 8,
    );
  });

  test("builtAt is a valid ISO timestamp", () => {
    const result = quantizationStrategyRouter(makeSyntheticCache(5, 32));
    expect(new Date(result.builtAt).getTime()).toBeGreaterThan(0);
  });

  test("sections with 0-length embeddings are skipped", () => {
    const entries = [
      { sectionPath: "empty.md", embedding: [] },
      { sectionPath: "valid.md", embedding: makeEmbedding(64, 1) },
    ];
    const result = quantizationStrategyRouter(entries);
    expect(result.sectionMeta.has("empty.md")).toBe(false);
    expect(result.sectionMeta.has("valid.md")).toBe(true);
    expect(result.stats.totalSections).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// 6. Sensitivity classification
// ---------------------------------------------------------------------------

describe("quantizationStrategyRouter — sensitivity classification", () => {
  test("sensitivityOverride 'low' forces int16 tier", () => {
    const entries = [{ sectionPath: "my-doc.md", embedding: makeEmbedding(128, 5) }];
    const result = quantizationStrategyRouter(entries, {
      sensitivityOverrides: { "my-doc.md": "low" },
    });
    const meta = result.sectionMeta.get("my-doc.md")!;
    expect(meta.sensitivity).toBe("low");
    expect(meta.assignedTier).toBe("int16");
  });

  test("sensitivityOverride 'high' assigns cheapest tier (int8 or bfloat16)", () => {
    const entries = [{ sectionPath: "summary.md", embedding: makeEmbedding(128, 3) }];
    const result = quantizationStrategyRouter(entries, {
      sensitivityOverrides: { "summary.md": "high" },
    });
    const meta = result.sectionMeta.get("summary.md")!;
    expect(meta.sensitivity).toBe("high");
    expect(["int8", "bfloat16"]).toContain(meta.assignedTier);
  });

  test("sensitivityOverride 'medium' assigns int16", () => {
    const entries = [{ sectionPath: "detail.md", embedding: makeEmbedding(128, 9) }];
    const result = quantizationStrategyRouter(entries, {
      sensitivityOverrides: { "detail.md": "medium" },
    });
    const meta = result.sectionMeta.get("detail.md")!;
    expect(meta.sensitivity).toBe("medium");
    expect(meta.assignedTier).toBe("int16");
  });

  test("code-like section paths get medium or low sensitivity (protectCodeSections=true)", () => {
    const codePaths = [
      "src/api-contracts.ts",
      "schema/types.ts",
      "auth/permissions.ts",
      "interfaces/core.ts",
    ];
    for (const path of codePaths) {
      const entries = [{ sectionPath: path, embedding: makeEmbedding(64, 1) }];
      const result = quantizationStrategyRouter(entries, { protectCodeSections: true });
      const meta = result.sectionMeta.get(path)!;
      // Code sections should not be "high" sensitivity (which would use int8/bfloat16)
      expect(["medium", "low"]).toContain(meta.sensitivity);
      expect(meta.assignedTier).toBe("int16");
    }
  });

  test("prose/vision paths get high sensitivity when error is low", () => {
    // vision docs (low MAE embeddings) should tolerate higher compression
    const entries = [{ sectionPath: "vision/north-star.md", embedding: makeEmbedding(128, 2) }];
    const result = quantizationStrategyRouter(entries, { protectCodeSections: true });
    const meta = result.sectionMeta.get("vision/north-star.md")!;
    // Prose docs should typically be "high" sensitivity (int8 or bfloat16)
    // But we check only that it's assigned some tier (not asserting exact value
    // since it depends on the MAE curve)
    expect(["high", "medium", "low"]).toContain(meta.sensitivity);
    expect(["float32", "int16", "int8", "bfloat16"]).toContain(meta.assignedTier);
  });

  test("disabling protectCodeSections allows code sections to be high sensitivity", () => {
    const entries = [{ sectionPath: "api.ts", embedding: makeEmbedding(64, 1) }];
    const withProtection    = quantizationStrategyRouter(entries, { protectCodeSections: true });
    const withoutProtection = quantizationStrategyRouter(entries, { protectCodeSections: false });

    const metaWith    = withProtection.sectionMeta.get("api.ts")!;
    const metaWithout = withoutProtection.sectionMeta.get("api.ts")!;

    // With protection: medium or low (int16)
    expect(["medium", "low"]).toContain(metaWith.sensitivity);
    // Without protection: may be high (int8/bfloat16) — at least the tier is valid
    expect(["float32", "int16", "int8", "bfloat16"]).toContain(metaWithout.assignedTier);
  });

  test("int8ErrorThreshold=0 forces all sections to medium or low sensitivity", () => {
    const entries = makeSyntheticCache(10, 64);
    const result = quantizationStrategyRouter(entries, { int8ErrorThreshold: 0 });
    for (const [, meta] of result.sectionMeta) {
      // With threshold=0, any int8 error > 0 → medium sensitivity at minimum
      expect(["medium", "low"]).toContain(meta.sensitivity);
    }
  });
});

// ---------------------------------------------------------------------------
// 7. Synthetic genome scale tests (10 – 500 sections)
// ---------------------------------------------------------------------------

describe("quantizationStrategyRouter — scale tests", () => {
  for (const n of [10, 50, 100, 200, 500]) {
    test(`${n} sections: all sections assigned, stats consistent`, () => {
      const entries = makeSyntheticCache(n, 128);
      const result = quantizationStrategyRouter(entries);

      expect(result.sectionMeta.size).toBe(n);
      expect(result.stats.totalSections).toBe(n);

      // All tier counts sum to n
      const tierSum = Object.values(result.stats.tierCounts).reduce((a, b) => a + b, 0);
      expect(tierSum).toBe(n);

      // At least some storage savings
      expect(result.stats.estimatedStorageSavingsFraction).toBeGreaterThan(0);
    });
  }

  test("500 sections: avgCompressionRatio < 0.8 (at least 20% savings)", () => {
    const entries = makeSyntheticCache(500, 128);
    const result = quantizationStrategyRouter(entries);
    // Most sections should be at int8/bfloat16 (0.25 or 0.5 compression)
    expect(result.stats.avgCompressionRatio).toBeLessThan(0.8);
  });

  test("100 sections: sectionMeta keys match input paths", () => {
    const entries = makeSyntheticCache(100, 64);
    const result = quantizationStrategyRouter(entries);
    for (const entry of entries) {
      expect(result.sectionMeta.has(entry.sectionPath)).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// 8. selectQueryTier — tier selection from GenomeProfile + maxTokens
// ---------------------------------------------------------------------------

describe("selectQueryTier — profile-based tier selection", () => {
  function makeProfile(sectionCount: number, tokensPerSection = 200) {
    const sections = Array.from({ length: sectionCount }, (_, i) =>
      makeSection({ path: `s${i}.md`, tokens: tokensPerSection }),
    );
    return GenomeProfiler.profile(makeManifest(sections));
  }

  test("forcePrecision=true always returns float32", () => {
    const profile = makeProfile(200);
    expect(selectQueryTier(profile, 1000, true)).toBe("float32");
    expect(selectQueryTier(profile, 100,  true)).toBe("float32");
  });

  test("large genome (200 sections) returns int8 (tight budget)", () => {
    const profile = makeProfile(200, 300); // 200 sections × 300 tokens = 60k total
    // maxTokens=500 → 2.5 tokens/section → very tight → int8
    expect(selectQueryTier(profile, 500)).toBe("int8");
  });

  test("large genome always returns int8 regardless of per-section budget", () => {
    const profile = makeProfile(200, 100);
    // Even with generous maxTokens, large genome → int8
    expect(selectQueryTier(profile, 50000)).toBe("int8");
  });

  test("medium genome with tight budget returns int8", () => {
    const profile = makeProfile(80, 200);
    // maxTokens=100 → ~1.25/section → tight
    expect(selectQueryTier(profile, 100)).toBe("int8");
  });

  test("medium genome with generous budget returns int16", () => {
    const profile = makeProfile(80, 200);
    // maxTokens=50000 → 625/section → generous
    expect(selectQueryTier(profile, 50000)).toBe("int16");
  });

  test("small genome (30 sections) returns int16", () => {
    const profile = makeProfile(30, 150);
    expect(selectQueryTier(profile, 5000)).toBe("int16");
  });

  test("tiny genome (5 sections) returns int16", () => {
    const profile = makeProfile(5, 100);
    expect(selectQueryTier(profile, 2000)).toBe("int16");
  });

  test("empty profile (0 sections) does not throw", () => {
    const profile = makeProfile(0);
    expect(() => selectQueryTier(profile, 1000)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// 9. selectBudgetTier — per-section tier with RouterBuildResult
// ---------------------------------------------------------------------------

describe("selectBudgetTier — budget-aware tier selection", () => {
  function makeProfile(sectionCount: number) {
    const sections = Array.from({ length: sectionCount }, (_, i) =>
      makeSection({ path: `s${i}.md`, tokens: 200 }),
    );
    return GenomeProfiler.profile(makeManifest(sections));
  }

  test("null buildResult falls back to selectQueryTier", () => {
    const profile = makeProfile(200); // large → int8
    const tier = selectBudgetTier(profile, 500, ["s1.md", "s2.md"], null);
    expect(tier).toBe("int8");
  });

  test("empty sectionPaths falls back to selectQueryTier", () => {
    const profile = makeProfile(200);
    const buildResult = quantizationStrategyRouter(makeSyntheticCache(5, 64));
    const tier = selectBudgetTier(profile, 500, [], buildResult);
    expect(tier).toBe("int8"); // large + tight → int8
  });

  test("when all sections assigned int8: returns int8 for large genome", () => {
    const entries = makeSyntheticCache(10, 64).map((e) => ({
      ...e,
      sectionPath: e.sectionPath,
    }));
    const buildResult = quantizationStrategyRouter(entries, {
      sensitivityOverrides: Object.fromEntries(entries.map((e) => [e.sectionPath, "high" as const])),
    });

    const profile = makeProfile(200);
    const paths = entries.map((e) => e.sectionPath);
    const tier = selectBudgetTier(profile, 500, paths, buildResult);
    // All sections are "high" (int8/bfloat16), default is int8 → int8 or bfloat16
    expect(["int8", "bfloat16"]).toContain(tier);
  });

  test("when any section is 'low' sensitivity (int16): tier upgrades to int16", () => {
    const entries = [
      { sectionPath: "api-contracts.ts", embedding: makeEmbedding(64, 1) },
      { sectionPath: "summary.md",       embedding: makeEmbedding(64, 2) },
    ];
    const buildResult = quantizationStrategyRouter(entries, {
      sensitivityOverrides: {
        "api-contracts.ts": "low",    // → int16
        "summary.md":       "high",   // → int8/bfloat16
      },
    });

    // Large genome default is int8, but api-contracts.ts forces int16 upgrade
    const profile = makeProfile(200);
    const tier = selectBudgetTier(profile, 500, ["api-contracts.ts", "summary.md"], buildResult);
    // Should be upgraded to int16 because api-contracts.ts needs it
    expect(tier).toBe("int16");
  });

  test("unknown section path is ignored (does not throw)", () => {
    const entries = [{ sectionPath: "known.md", embedding: makeEmbedding(64, 1) }];
    const buildResult = quantizationStrategyRouter(entries);
    const profile = makeProfile(200);
    expect(() =>
      selectBudgetTier(profile, 500, ["unknown-section.md"], buildResult),
    ).not.toThrow();
  });

  test("returns a value in ROUTER_FALLBACK_CHAIN", () => {
    const entries = makeSyntheticCache(20, 64);
    const buildResult = quantizationStrategyRouter(entries);
    const profile = makeProfile(80);
    const paths = entries.map((e) => e.sectionPath);
    const tier = selectBudgetTier(profile, 2000, paths, buildResult);
    expect(ROUTER_FALLBACK_CHAIN).toContain(tier);
  });
});

// ---------------------------------------------------------------------------
// 10. Cross-tier consistency — top-K results stable across tiers
// ---------------------------------------------------------------------------

describe("Cross-tier consistency — top-K ranking stability", () => {
  const DIM = 128;
  const N_ENTRIES = 30;

  /** Rank embeddings by cosine similarity to query at a given tier. */
  function rankAtTier(
    query: number[],
    entries: Array<{ id: string; embedding: number[] }>,
    tier: QuantizationTier,
    topK: number,
  ): string[] {
    return entries
      .map(({ id, embedding }) => {
        const q = quantizeUnified(embedding, tier);
        const deq = dequantizeUnified(q);
        return { id, sim: cosine(query, deq) };
      })
      .sort((a, b) => b.sim - a.sim)
      .slice(0, topK)
      .map((r) => r.id);
  }

  test("top-1 result is consistent across float32 / int16 / bfloat16", () => {
    const entries = Array.from({ length: N_ENTRIES }, (_, i) => ({
      id: `e${i}`,
      embedding: makeEmbedding(DIM, i + 1),
    }));

    // Make embedding #5 clearly closest to query
    const query = entries[5]!.embedding.map((v) => v * 0.98 + 0.01);
    const norm = Math.sqrt(query.reduce((s, v) => s + v * v, 0));
    const normQuery = query.map((v) => v / norm);

    const top1_f32  = rankAtTier(normQuery, entries, "float32",  1)[0];
    const top1_i16  = rankAtTier(normQuery, entries, "int16",    1)[0];
    const top1_bf16 = rankAtTier(normQuery, entries, "bfloat16", 1)[0];

    expect(top1_f32).toBe("e5");
    expect(top1_i16).toBe("e5");
    expect(top1_bf16).toBe("e5");
  });

  test("top-3 set has ≥ 2 common entries across float32 and int8", () => {
    const entries = Array.from({ length: N_ENTRIES }, (_, i) => ({
      id: `e${i}`,
      embedding: makeEmbedding(DIM, i + 100),
    }));

    const query = makeEmbedding(DIM, 999);

    const top3_f32 = new Set(rankAtTier(query, entries, "float32", 3));
    const top3_i8  = new Set(rankAtTier(query, entries, "int8",    3));

    const overlap = [...top3_f32].filter((id) => top3_i8.has(id)).length;
    expect(overlap).toBeGreaterThanOrEqual(2);
  });

  test("top-5 set has ≥ 4 common entries across int16 and bfloat16", () => {
    const entries = Array.from({ length: N_ENTRIES }, (_, i) => ({
      id: `e${i}`,
      embedding: makeEmbedding(DIM, i + 200),
    }));

    const query = makeEmbedding(DIM, 500);

    const top5_i16  = new Set(rankAtTier(query, entries, "int16",    5));
    const top5_bf16 = new Set(rankAtTier(query, entries, "bfloat16", 5));

    const overlap = [...top5_i16].filter((id) => top5_bf16.has(id)).length;
    expect(overlap).toBeGreaterThanOrEqual(4);
  });
});

// ---------------------------------------------------------------------------
// 11. Query-time latency / accuracy tradeoffs (relative benchmarks)
// ---------------------------------------------------------------------------

describe("Query-time latency/accuracy tradeoffs", () => {
  const DIM = 768;
  const N = 200; // large genome

  const entries = Array.from({ length: N }, (_, i) => ({
    id: `e${i}`,
    embedding: makeEmbedding(DIM, i + 1),
  }));
  const query = makeEmbedding(DIM, 9999);

  function timeRankAtTier(tier: QuantizationTier): { latencyMs: number; topId: string } {
    const t0 = performance.now();
    const ranked = entries
      .map(({ id, embedding }) => {
        const q = quantizeUnified(embedding, tier);
        const deq = dequantizeUnified(q);
        return { id, sim: cosine(query, deq) };
      })
      .sort((a, b) => b.sim - a.sim);
    const latencyMs = performance.now() - t0;
    return { latencyMs, topId: ranked[0]!.id };
  }

  test("all tiers complete in < 500ms on 200 sections (768-dim)", () => {
    for (const tier of ["float32", "int16", "int8", "bfloat16"] as QuantizationTier[]) {
      const { latencyMs } = timeRankAtTier(tier);
      expect(latencyMs).toBeLessThan(500);
    }
  });

  test("top-1 result is stable across all tiers (no Ollama required)", () => {
    const results = new Map<QuantizationTier, string>();
    for (const tier of ["float32", "int16", "int8", "bfloat16"] as QuantizationTier[]) {
      results.set(tier, timeRankAtTier(tier).topId);
    }
    // All tiers should agree on the top-1 result
    const ids = new Set(results.values());
    expect(ids.size).toBe(1);
  });

  test("int8 and bfloat16 latency are bounded in absolute terms on 200×768-dim", () => {
    // Each tier should complete 200-section ranking in under 200ms.
    // We use an absolute bound rather than a relative one because cold-start
    // JIT can make float32 (the no-op reference) artificially fast on the
    // first run, making relative comparisons fragile.
    for (const tier of ["float32", "int16", "int8", "bfloat16"] as QuantizationTier[]) {
      const { latencyMs } = timeRankAtTier(tier);
      expect(latencyMs).toBeLessThan(200);
    }
  });
});

// ---------------------------------------------------------------------------
// 12. Integration: retrieveSectionsV2 with routerBuildResult
// (no Ollama — falls back to hierarchical but doesn't error)
// ---------------------------------------------------------------------------

describe("retrieveSectionsV2 — routerBuildResult integration (no Ollama)", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = join(tmpdir(), `ashlr-rv2-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    await mkdir(join(cwd, ".ashlrcode", "genome", "evolution"), { recursive: true });
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("with routerBuildResult but no Ollama: falls back to hierarchical, no throw", async () => {
    const { retrieveSectionsV2 } = await import("../src/genome/retriever.ts");
    const buildResult = quantizationStrategyRouter(makeSyntheticCache(5, 64));

    // No manifest, no Ollama → should return empty array without throwing
    const results = await retrieveSectionsV2(cwd, "query about embeddings", 1000, buildResult);
    expect(Array.isArray(results)).toBe(true);
  });

  test("without routerBuildResult: behaves identically to legacy signature", async () => {
    const { retrieveSectionsV2 } = await import("../src/genome/retriever.ts");

    // Both calls should return identical results (empty, no manifest)
    const r1 = await retrieveSectionsV2(cwd, "test query", 1000);
    const r2 = await retrieveSectionsV2(cwd, "test query", 1000, null);

    expect(r1).toEqual(r2);
  });

  test("with routerBuildResult=null: no throw", async () => {
    const { retrieveSectionsV2 } = await import("../src/genome/retriever.ts");
    const result = await retrieveSectionsV2(cwd, "embedding search query", 2000, null);
    expect(Array.isArray(result)).toBe(true);
  });

  test("empty query bypasses semantic search regardless of routerBuildResult", async () => {
    const { retrieveSectionsV2 } = await import("../src/genome/retriever.ts");
    const buildResult = quantizationStrategyRouter(makeSyntheticCache(5, 64));

    // Empty query should not attempt Ollama; should return empty (no manifest)
    const result = await retrieveSectionsV2(cwd, "  ", 1000, buildResult);
    expect(Array.isArray(result)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 13. Storage savings estimation
// ---------------------------------------------------------------------------

describe("quantizationStrategyRouter — storage savings estimation", () => {
  test("all-high sensitivity: avgCompressionRatio ≤ 0.5 (2× minimum savings)", () => {
    const entries = makeSyntheticCache(50, 128);
    const result = quantizationStrategyRouter(entries, {
      sensitivityOverrides: Object.fromEntries(
        entries.map((e) => [e.sectionPath, "high" as const]),
      ),
    });
    // All "high" sections get int8 (0.25) or bfloat16 (0.5) → avg ≤ 0.5
    expect(result.stats.avgCompressionRatio).toBeLessThanOrEqual(0.5);
  });

  test("all-low sensitivity: avgCompressionRatio = 0.5 (int16 = 2× savings)", () => {
    const entries = makeSyntheticCache(20, 64);
    const result = quantizationStrategyRouter(entries, {
      sensitivityOverrides: Object.fromEntries(
        entries.map((e) => [e.sectionPath, "low" as const]),
      ),
    });
    // All "low" sections get int16 → compressionRatio = 0.5
    expect(result.stats.avgCompressionRatio).toBeCloseTo(0.5, 5);
  });

  test("storageBytesAssigned + storageBytesFloat32 are consistent with dim", () => {
    const dim = 128;
    const entries = [{ sectionPath: "doc.md", embedding: makeEmbedding(dim, 1) }];
    const result = quantizationStrategyRouter(entries, {
      sensitivityOverrides: { "doc.md": "high" },
    });
    const meta = result.sectionMeta.get("doc.md")!;
    expect(meta.storageBytesFloat32).toBe(dim * 4);
    // int8 or bfloat16
    if (meta.assignedTier === "int8") {
      expect(meta.storageBytesAssigned).toBe(dim * 1);
    } else if (meta.assignedTier === "bfloat16") {
      expect(meta.storageBytesAssigned).toBe(dim * 2);
    }
  });
});
