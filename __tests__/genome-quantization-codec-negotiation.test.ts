/**
 * Tests for Multi-Tier Quantization Fallback with Per-Model Codec Negotiation
 *
 * Covers:
 *  - QuantizationNegotiator: codec selection under memory, latency, and affinity constraints
 *  - selectCodecForRetrieval / recordCodecQuality / getCodecAffinity module-level API
 *  - CodecPredictor: forward pass, predict, SGD update convergence
 *  - NEGOTIATOR_CODEC_CHAIN priority ordering and float32 unconditional fallback
 *  - Reconstruction error (MAE) bounds per codec tier
 *  - NDCG@5 quality tracking via CodecAffinityStore
 *  - Per-section codec affinity: classifySectionType bucketing + getBestCodec
 *  - Codec metadata versioning: CODEC_SCHEMA_VERSION, isCodecVersionStale, upgradeCodecMeta
 *  - serializeEmbeddingVersioned: versioned meta attachment
 *  - Mixed-version cache detection
 */

import { describe, expect, test, beforeEach } from "bun:test";

import {
  QuantizationNegotiator,
  CodecPredictor,
  NEGOTIATOR_CODEC_CHAIN,
  selectCodecForRetrieval,
  recordCodecQuality,
  getCodecAffinity,
  type NegotiatorContext,
  type NegotiatorCodecTier,
  type CodecQualityObservation,
} from "../src/genome/quantization-strategy.ts";

import {
  CodecAffinityStore,
  globalCodecAffinityStore,
  classifySectionType,
  CODEC_SCHEMA_VERSION,
  isCodecVersionStale,
  upgradeCodecMeta,
  serializeEmbeddingVersioned,
  globalCodecRegistry,
  type SectionTypeBucket,
  type CodecAffinityRecord,
} from "../src/genome/embedding-codec.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Deterministic unit-normalized embedding. */
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

function mae(a: number[], b: number[]): number {
  if (a.length === 0 || a.length !== b.length) return 0;
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i]! - b[i]!);
  return sum / a.length;
}

/** Build a NegotiatorContext with sensible defaults. */
function makeCtx(overrides: Partial<NegotiatorContext> = {}): NegotiatorContext {
  return {
    embeddingDims: 384,
    availableHeapBytes: 512 * 1024 * 1024, // 512 MB
    latencyBudgetMs: 500,
    sectionType: "prose",
    candidateCount: 50,
    ...overrides,
  };
}

/** Build a CodecQualityObservation with sensible defaults. */
function makeObs(overrides: Partial<CodecQualityObservation> = {}): CodecQualityObservation {
  return {
    codecFormat: "int8",
    sectionType: "prose",
    reconstructionMae: 0.005,
    ndcgAt5: 0.95,
    latencyMs: 12,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// NEGOTIATOR_CODEC_CHAIN
// ---------------------------------------------------------------------------

describe("NEGOTIATOR_CODEC_CHAIN", () => {
  test("has exactly four tiers", () => {
    expect(NEGOTIATOR_CODEC_CHAIN.length).toBe(4);
  });

  test("priority order is bfloat16 → int8 → int4 → float32", () => {
    expect(NEGOTIATOR_CODEC_CHAIN[0]).toBe("bfloat16");
    expect(NEGOTIATOR_CODEC_CHAIN[1]).toBe("int8");
    expect(NEGOTIATOR_CODEC_CHAIN[2]).toBe("int4");
    expect(NEGOTIATOR_CODEC_CHAIN[3]).toBe("float32");
  });

  test("float32 is the last (unconditional fallback) tier", () => {
    expect(NEGOTIATOR_CODEC_CHAIN[NEGOTIATOR_CODEC_CHAIN.length - 1]).toBe("float32");
  });
});

// ---------------------------------------------------------------------------
// QuantizationNegotiator — baseline selection
// ---------------------------------------------------------------------------

describe("QuantizationNegotiator — codec selection", () => {
  let store: CodecAffinityStore;
  let negotiator: QuantizationNegotiator;

  beforeEach(() => {
    store = new CodecAffinityStore();
    negotiator = new QuantizationNegotiator({ affinityStore: store });
  });

  test("selects bfloat16 when constraints are generous", () => {
    const result = negotiator.selectCodec(makeCtx());
    expect(result.codecFormat).toBe("bfloat16");
    expect(result.tierIndex).toBe(0);
    expect(result.isPreferred).toBe(true);
    expect(result.isFallback).toBe(false);
  });

  test("result has non-empty rationale", () => {
    const result = negotiator.selectCodec(makeCtx());
    expect(result.rationale.length).toBeGreaterThan(0);
  });

  test("estimatedMemoryBytes is dims × bytesPerDim × candidates", () => {
    const ctx = makeCtx({ embeddingDims: 512, candidateCount: 100 });
    const result = negotiator.selectCodec(ctx);
    // bfloat16 = 2 bytes/dim
    expect(result.estimatedMemoryBytes).toBe(512 * 2 * 100);
  });
});

// ---------------------------------------------------------------------------
// QuantizationNegotiator — memory constraint
// ---------------------------------------------------------------------------

describe("QuantizationNegotiator — memory constraint", () => {
  test("downgrades from bfloat16 when heap is very tight", () => {
    const store = new CodecAffinityStore();
    const neg = new QuantizationNegotiator({
      affinityStore: store,
      memorySafetyMargin: 0.1,
    });
    // With 384 dims × 2 bytes × 100 candidates = 76800 bytes needed for bfloat16
    // Force available heap to exactly 70 KB — below the required amount
    const ctx = makeCtx({
      embeddingDims: 384,
      candidateCount: 100,
      availableHeapBytes: 70 * 1024, // 71 680 bytes (safe = 63 512)
    });
    const result = neg.selectCodec(ctx);
    // bfloat16 needs 76800B > safe 63512B → should downgrade
    expect(result.tierIndex).toBeGreaterThan(0);
  });

  test("falls back to float32 when heap is extremely tight", () => {
    const store = new CodecAffinityStore();
    const neg = new QuantizationNegotiator({ affinityStore: store });
    // 1 KB available — all compressed tiers exceed budget, float32 at 0.5 bytes
    // per dim with int4 might still fit; use extreme dims to force float32
    const ctx = makeCtx({
      embeddingDims: 1536,
      candidateCount: 500,
      availableHeapBytes: 100, // essentially nothing
    });
    const result = neg.selectCodec(ctx);
    // float32 always passes (it's the unconditional fallback)
    expect(result.isFallback).toBe(true);
    expect(result.codecFormat).toBe("float32");
  });

  test("isFallback is true only for float32", () => {
    const store = new CodecAffinityStore();
    const neg = new QuantizationNegotiator({ affinityStore: store });
    const result = neg.selectCodec(makeCtx());
    if (result.codecFormat === "float32") {
      expect(result.isFallback).toBe(true);
    } else {
      expect(result.isFallback).toBe(false);
    }
  });
});

// ---------------------------------------------------------------------------
// QuantizationNegotiator — latency constraint
// ---------------------------------------------------------------------------

describe("QuantizationNegotiator — latency constraint", () => {
  test("skips to cheaper codec when latency budget is extremely tight", () => {
    const store = new CodecAffinityStore();
    // Very high cost per dim to force downgrade
    const neg = new QuantizationNegotiator({
      affinityStore: store,
      tierCostMs: { bfloat16: 100, int8: 50, int4: 10, float32: 0.001 },
      latencySlackFactor: 1.0,
    });
    const ctx = makeCtx({
      embeddingDims: 384,
      candidateCount: 50,
      latencyBudgetMs: 5, // very tight
    });
    const result = neg.selectCodec(ctx);
    // bfloat16: 100 * (384/1000) * 50 = 1920ms >> 5ms budget → skip
    // int8: 50 * (384/1000) * 50 = 960ms >> 5ms → skip
    // int4: 10 * (384/1000) * 50 = 192ms >> 5ms → skip
    // float32: 0.001 * (384/1000) * 50 = 0.019ms << 5ms → pass
    expect(result.codecFormat).toBe("float32");
    expect(result.isFallback).toBe(true);
  });

  test("accepts bfloat16 when latency budget is generous", () => {
    const store = new CodecAffinityStore();
    const neg = new QuantizationNegotiator({ affinityStore: store });
    const ctx = makeCtx({ latencyBudgetMs: 10000 });
    const result = neg.selectCodec(ctx);
    expect(result.codecFormat).toBe("bfloat16");
  });
});

// ---------------------------------------------------------------------------
// QuantizationNegotiator — affinity bias
// ---------------------------------------------------------------------------

describe("QuantizationNegotiator — affinity bias", () => {
  test("biases toward affinity-known codec when store has data", () => {
    const store = new CodecAffinityStore();
    const neg = new QuantizationNegotiator({ affinityStore: store, mlpMinObservations: 999 });

    // Seed the affinity store with int8 as the best for "api" sections
    for (let i = 0; i < 5; i++) {
      store.record({
        sectionType: "api",
        codecFormat: "int8",
        reconstructionMae: 0.004,
        ndcgAt5: 0.96,
        latencyMs: 10,
      });
    }

    const ctx = makeCtx({ sectionType: "api" });
    const result = neg.selectCodec(ctx);
    // int8 has index 1 in the chain; the negotiator should start from there
    expect(result.tierIndex).toBeGreaterThanOrEqual(1);
  });

  test("falls back to bfloat16 when affinity store has no data for section type", () => {
    const store = new CodecAffinityStore();
    const neg = new QuantizationNegotiator({ affinityStore: store, mlpMinObservations: 999 });
    const result = neg.selectCodec(makeCtx({ sectionType: "unknown" }));
    expect(result.tierIndex).toBe(0); // no bias → prefer bfloat16
  });
});

// ---------------------------------------------------------------------------
// QuantizationNegotiator — MLP predictor activation
// ---------------------------------------------------------------------------

describe("QuantizationNegotiator — MLP activation", () => {
  test("mlpActive is false before mlpMinObservations are collected", () => {
    const store = new CodecAffinityStore();
    const neg = new QuantizationNegotiator({ affinityStore: store, mlpMinObservations: 5 });
    expect(neg.mlpActive).toBe(false);
    expect(neg.observationCount).toBe(0);
  });

  test("mlpActive becomes true after mlpMinObservations recordObservation calls", () => {
    const store = new CodecAffinityStore();
    const neg = new QuantizationNegotiator({ affinityStore: store, mlpMinObservations: 3 });
    const ctx = makeCtx();
    for (let i = 0; i < 3; i++) {
      neg.recordObservation(makeObs({ codecFormat: "int8" }), ctx);
    }
    expect(neg.mlpActive).toBe(true);
    expect(neg.observationCount).toBe(3);
  });

  test("recordObservation increments observationCount for valid codecs", () => {
    const store = new CodecAffinityStore();
    const neg = new QuantizationNegotiator({ affinityStore: store });
    const ctx = makeCtx();
    neg.recordObservation(makeObs({ codecFormat: "bfloat16" }), ctx);
    neg.recordObservation(makeObs({ codecFormat: "int8" }), ctx);
    expect(neg.observationCount).toBe(2);
  });

  test("recordObservation with unknown codec does not increment count", () => {
    const store = new CodecAffinityStore();
    const neg = new QuantizationNegotiator({ affinityStore: store });
    const ctx = makeCtx();
    neg.recordObservation(makeObs({ codecFormat: "unknown-format" }), ctx);
    expect(neg.observationCount).toBe(0);
  });

  test("MLP selects a valid tier index after sufficient training", () => {
    const store = new CodecAffinityStore();
    const neg = new QuantizationNegotiator({
      affinityStore: store,
      mlpMinObservations: 5,
      mlpLearningRate: 0.1,
    });
    const ctx = makeCtx();
    // Train with int8 as the "correct" choice
    for (let i = 0; i < 5; i++) {
      neg.recordObservation(makeObs({ codecFormat: "int8", ndcgAt5: 0.97 }), ctx);
    }
    expect(neg.mlpActive).toBe(true);
    const result = neg.selectCodec(ctx);
    // Result must be a valid tier (any is acceptable — MLP may not converge in 5 steps)
    expect(NEGOTIATOR_CODEC_CHAIN).toContain(result.codecFormat as NegotiatorCodecTier);
  });
});

// ---------------------------------------------------------------------------
// CodecPredictor — unit tests
// ---------------------------------------------------------------------------

describe("CodecPredictor", () => {
  test("forward() returns 4 probabilities summing to ~1", () => {
    const pred = new CodecPredictor();
    const probs = pred.forward([0.5, 0.8, 0.3, 0.1, 0.2]);
    expect(probs.length).toBe(4);
    const sum = probs.reduce((s, p) => s + p, 0);
    expect(Math.abs(sum - 1.0)).toBeLessThan(1e-6);
  });

  test("forward() all probabilities are in [0, 1]", () => {
    const pred = new CodecPredictor();
    const probs = pred.forward([0.1, 0.5, 0.9, 0.3, 0.7]);
    for (const p of probs) {
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });

  test("predict() returns a valid index in [0, 3]", () => {
    const pred = new CodecPredictor();
    const idx = pred.predict([0.5, 0.8, 0.3, 0.1, 0.2]);
    expect(idx).toBeGreaterThanOrEqual(0);
    expect(idx).toBeLessThanOrEqual(3);
  });

  test("predict() is deterministic for the same input", () => {
    const pred = new CodecPredictor();
    const features = [0.5, 0.8, 0.3, 0.1, 0.2];
    expect(pred.predict(features)).toBe(pred.predict(features));
  });

  test("update() shifts probability toward the target label", () => {
    const pred = new CodecPredictor(0.5); // high LR for fast convergence
    const features = [0.5, 0.8, 0.3, 0.1, 0.2];
    const targetLabel = 1; // int8

    // Multiple update steps
    for (let i = 0; i < 50; i++) {
      pred.update(features, targetLabel);
    }

    const probs = pred.forward(features);
    // After many steps, label 1 should have the highest probability
    const maxIdx = probs.reduce((best, p, i) => (p > probs[best]! ? i : best), 0);
    expect(maxIdx).toBe(targetLabel);
  });

  test("update() changes the prediction for different labels", () => {
    const pred = new CodecPredictor(0.3);
    const features = [0.9, 0.1, 0.5, 0.2, 0.0];

    // Train toward float32 (label 3)
    for (let i = 0; i < 100; i++) {
      pred.update(features, 3);
    }

    const probs = pred.forward(features);
    // float32 (index 3) should have significantly higher probability
    expect(probs[3]).toBeGreaterThan(probs[0]!);
  });
});

// ---------------------------------------------------------------------------
// Reconstruction error (MAE) bounds
// ---------------------------------------------------------------------------

describe("Reconstruction error bounds via codec round-trip", () => {
  const dim = 384;

  function roundTripMae(format: string, embedding: number[]): number {
    const codec = globalCodecRegistry.get(format);
    const serialized = codec.serialize(embedding);
    const reconstructed = codec.deserialize(serialized);
    return mae(embedding, reconstructed);
  }

  test("float32 has near-zero reconstruction error (≤ float rounding)", () => {
    const emb = makeEmbedding(dim, 42);
    expect(roundTripMae("float32", emb)).toBeLessThan(1e-7);
  });

  test("int16 MAE < 0.0001 for unit-normalized embedding", () => {
    const emb = makeEmbedding(dim, 7);
    expect(roundTripMae("int16", emb)).toBeLessThan(0.0001);
  });

  test("int8 MAE < 0.01 for unit-normalized embedding", () => {
    const emb = makeEmbedding(dim, 13);
    expect(roundTripMae("int8", emb)).toBeLessThan(0.01);
  });

  test("int4 MAE < 0.05 for unit-normalized embedding", () => {
    const emb = makeEmbedding(dim, 17);
    expect(roundTripMae("int4", emb)).toBeLessThan(0.05);
  });

  test("int16 MAE << int8 MAE (higher precision codec is more accurate)", () => {
    const emb = makeEmbedding(dim, 99);
    expect(roundTripMae("int16", emb)).toBeLessThan(roundTripMae("int8", emb));
  });

  test("int8 MAE << int4 MAE", () => {
    const emb = makeEmbedding(dim, 55);
    expect(roundTripMae("int8", emb)).toBeLessThan(roundTripMae("int4", emb));
  });
});

// ---------------------------------------------------------------------------
// NDCG@5 quality tracking — CodecAffinityStore
// ---------------------------------------------------------------------------

describe("CodecAffinityStore — NDCG@5 quality tracking", () => {
  test("record() adds an observation with current timestamp", () => {
    const store = new CodecAffinityStore();
    store.record({
      sectionType: "api",
      codecFormat: "int16",
      reconstructionMae: 0.00002,
      ndcgAt5: 0.999,
      latencyMs: 8,
    });
    expect(store.size).toBe(1);
    const recs = store.getRecords("api");
    expect(recs.length).toBe(1);
    expect(recs[0]!.observedAt).toBeTruthy();
  });

  test("getAffinityMap() computes avgNdcgAt5 correctly", () => {
    const store = new CodecAffinityStore();
    const ndcgValues = [0.9, 0.95, 1.0];
    for (const ndcg of ndcgValues) {
      store.record({
        sectionType: "architecture",
        codecFormat: "bfloat16",
        reconstructionMae: 0.0002,
        ndcgAt5: ndcg,
        latencyMs: 5,
      });
    }
    const map = store.getAffinityMap();
    const summaries = map.get("architecture") ?? [];
    const bf16 = summaries.find((s) => s.codecFormat === "bfloat16");
    expect(bf16).toBeDefined();
    const expected = ndcgValues.reduce((s, v) => s + v, 0) / ndcgValues.length;
    expect(Math.abs((bf16!.avgNdcgAt5 ?? 0) - expected)).toBeLessThan(1e-9);
  });

  test("affinityScore = avgNdcgAt5 - avgMae * 10", () => {
    const store = new CodecAffinityStore();
    store.record({
      sectionType: "prose",
      codecFormat: "int8",
      reconstructionMae: 0.005,
      ndcgAt5: 0.95,
      latencyMs: 12,
    });
    const map = store.getAffinityMap();
    const s = map.get("prose")![0]!;
    const expected = 0.95 - 0.005 * 10;
    expect(Math.abs((s.affinityScore ?? 0) - expected)).toBeLessThan(1e-9);
  });

  test("getBestCodec() returns null when no data exists", () => {
    const store = new CodecAffinityStore();
    expect(store.getBestCodec("code")).toBeNull();
  });

  test("getBestCodec() returns the codec with highest affinityScore", () => {
    const store = new CodecAffinityStore();
    // int16: ndcg=0.999, mae=0.00002 → score = 0.999 - 0.0002 = 0.9988
    // int8:  ndcg=0.95,  mae=0.005   → score = 0.95 - 0.05 = 0.90
    for (let i = 0; i < 5; i++) {
      store.record({ sectionType: "api", codecFormat: "int16", reconstructionMae: 0.00002, ndcgAt5: 0.999, latencyMs: 8 });
      store.record({ sectionType: "api", codecFormat: "int8",  reconstructionMae: 0.005,   ndcgAt5: 0.95,  latencyMs: 12 });
    }
    expect(store.getBestCodec("api")).toBe("int16");
  });

  test("getBestCodec() respects minSamples threshold", () => {
    const store = new CodecAffinityStore();
    // Only 2 records — below default minSamples=3
    store.record({ sectionType: "config", codecFormat: "int8", reconstructionMae: 0.005, ndcgAt5: 0.95, latencyMs: 10 });
    store.record({ sectionType: "config", codecFormat: "int8", reconstructionMae: 0.005, ndcgAt5: 0.95, latencyMs: 10 });
    expect(store.getBestCodec("config", 3)).toBeNull();
    expect(store.getBestCodec("config", 2)).toBe("int8");
  });

  test("window eviction: oldest record is dropped when full", () => {
    const store = new CodecAffinityStore(3); // window size = 3
    for (let i = 0; i < 4; i++) {
      store.record({ sectionType: "prose", codecFormat: "int8", reconstructionMae: 0.005, ndcgAt5: null, latencyMs: 10 });
    }
    expect(store.size).toBe(3); // evicted one
  });

  test("avgNdcgAt5 is null when all observations have null ndcgAt5", () => {
    const store = new CodecAffinityStore();
    store.record({ sectionType: "data", codecFormat: "int4", reconstructionMae: 0.02, ndcgAt5: null, latencyMs: 5 });
    const map = store.getAffinityMap();
    const s = map.get("data")![0]!;
    expect(s.avgNdcgAt5).toBeNull();
    expect(s.affinityScore).toBeNull();
  });

  test("clear() empties the store", () => {
    const store = new CodecAffinityStore();
    store.record({ sectionType: "api", codecFormat: "int8", reconstructionMae: 0.005, ndcgAt5: 0.95, latencyMs: 10 });
    store.clear();
    expect(store.size).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Section type bucketing — classifySectionType
// ---------------------------------------------------------------------------

describe("classifySectionType", () => {
  const cases: Array<[string, string | undefined, SectionTypeBucket]> = [
    ["src/api/routes.ts",            undefined,      "api"],
    ["tests/fixtures/embedding.json",undefined,      "test-fixtures"],
    ["__tests__/genome.test.ts",     undefined,      "test-fixtures"],
    ["docs/architecture/overview.md",undefined,      "architecture"],
    ["vision/north-star.md",         "North Star",   "architecture"],
    [".ashlrcode/genome/settings.yaml", undefined,   "config"],
    ["config/env.json",              undefined,      "config"],
    ["src/auth/handler.ts",          undefined,      "code"],
    ["src/genome/embeddings.ts",     undefined,      "code"],
    ["data/seeds/mock-users.json",   undefined,      "data"],
    ["prose/getting-started.md",     undefined,      "prose"],
    ["README.md",                    undefined,      "prose"],
  ];

  for (const [path, title, expected] of cases) {
    test(`"${path}" → "${expected}"`, () => {
      expect(classifySectionType(path, title)).toBe(expected);
    });
  }
});

// ---------------------------------------------------------------------------
// Codec metadata versioning
// ---------------------------------------------------------------------------

describe("CODEC_SCHEMA_VERSION", () => {
  test("is a positive integer", () => {
    expect(Number.isInteger(CODEC_SCHEMA_VERSION)).toBe(true);
    expect(CODEC_SCHEMA_VERSION).toBeGreaterThan(0);
  });
});

describe("isCodecVersionStale", () => {
  test("returns true for undefined meta", () => {
    expect(isCodecVersionStale(undefined)).toBe(true);
  });

  test("returns true for meta without codec_schema_version", () => {
    expect(isCodecVersionStale({ codec_format: "int8", codec_version: 1 })).toBe(true);
  });

  test("returns true for meta with version < CODEC_SCHEMA_VERSION", () => {
    expect(isCodecVersionStale({ codec_schema_version: CODEC_SCHEMA_VERSION - 1 })).toBe(true);
  });

  test("returns false for meta with current version", () => {
    expect(isCodecVersionStale({ codec_schema_version: CODEC_SCHEMA_VERSION })).toBe(false);
  });

  test("returns false for meta with version > CODEC_SCHEMA_VERSION (future compat)", () => {
    expect(isCodecVersionStale({ codec_schema_version: CODEC_SCHEMA_VERSION + 1 })).toBe(false);
  });
});

describe("upgradeCodecMeta", () => {
  test("upgrades undefined to float32 default", () => {
    const m = upgradeCodecMeta(undefined);
    expect(m.codec_format).toBe("float32");
    expect(m.codec_version).toBe(1);
    expect(m.codec_schema_version).toBeUndefined();
  });

  test("preserves codec_format from existing meta", () => {
    const m = upgradeCodecMeta({ codec_format: "int8", codec_version: 1 });
    expect(m.codec_format).toBe("int8");
    expect(m.codec_schema_version).toBe(1); // old v1
  });

  test("preserves codec_version", () => {
    const m = upgradeCodecMeta({ codec_format: "int16", codec_version: 1 });
    expect(m.codec_version).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// serializeEmbeddingVersioned
// ---------------------------------------------------------------------------

describe("serializeEmbeddingVersioned", () => {
  test("attaches codec_schema_version to the returned meta", () => {
    const emb = makeEmbedding(64, 7);
    const { meta } = serializeEmbeddingVersioned(emb, "int8");
    expect(meta.codec_schema_version).toBe(CODEC_SCHEMA_VERSION);
  });

  test("attaches encoded_at ISO timestamp", () => {
    const emb = makeEmbedding(64, 11);
    const { meta } = serializeEmbeddingVersioned(emb, "int16");
    expect(meta.encoded_at).toBeTruthy();
    expect(new Date(meta.encoded_at!).getTime()).toBeLessThanOrEqual(Date.now());
  });

  test("round-trip fidelity is maintained for int8", () => {
    const emb = makeEmbedding(128, 5);
    const { data_b64, meta } = serializeEmbeddingVersioned(emb, "int8");
    const codec = globalCodecRegistry.get(meta.codec_format);
    const buf = Buffer.from(data_b64, "base64");
    const reconstructed = codec.deserialize(new Uint8Array(buf));
    expect(mae(emb, reconstructed)).toBeLessThan(0.01);
  });

  test("round-trip fidelity is maintained for int16", () => {
    const emb = makeEmbedding(128, 3);
    const { data_b64, meta } = serializeEmbeddingVersioned(emb, "int16");
    const codec = globalCodecRegistry.get(meta.codec_format);
    const buf = Buffer.from(data_b64, "base64");
    const reconstructed = codec.deserialize(new Uint8Array(buf));
    expect(mae(emb, reconstructed)).toBeLessThan(0.0001);
  });

  test("isCodecVersionStale returns false for versioned entries", () => {
    const emb = makeEmbedding(32, 2);
    const { meta } = serializeEmbeddingVersioned(emb, "float32");
    expect(isCodecVersionStale(meta)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Mixed-version cache detection
// ---------------------------------------------------------------------------

describe("Mixed-version cache detection", () => {
  test("a v1 entry (no schema version) is detected as stale", () => {
    const v1Meta = { codec_format: "int8", codec_version: 1 as const };
    expect(isCodecVersionStale(v1Meta)).toBe(true);
  });

  test("a v2 entry (with schema version) is not stale", () => {
    const emb = makeEmbedding(64);
    const { meta } = serializeEmbeddingVersioned(emb, "int8");
    expect(isCodecVersionStale(meta)).toBe(false);
  });

  test("mixed cache: can identify which entries need re-encoding", () => {
    const emb = makeEmbedding(64);
    const v1 = { codec_format: "int8", codec_version: 1 as const };
    const { meta: v2 } = serializeEmbeddingVersioned(emb, "int8");
    const cache = [v1, v2, undefined];
    const staleCount = cache.filter((m) => isCodecVersionStale(m)).length;
    expect(staleCount).toBe(2); // v1 and undefined are stale
  });
});

// ---------------------------------------------------------------------------
// getCodecAffinity — module-level API
// ---------------------------------------------------------------------------

describe("getCodecAffinity", () => {
  beforeEach(() => {
    globalCodecAffinityStore.clear();
  });

  test("returns empty map when no observations have been recorded", () => {
    const map = getCodecAffinity();
    expect(map.size).toBe(0);
  });

  test("returns data for a specific section type after recording", () => {
    globalCodecAffinityStore.record({
      sectionType: "api",
      codecFormat: "int16",
      reconstructionMae: 0.00002,
      ndcgAt5: 0.999,
      latencyMs: 8,
    });
    const map = getCodecAffinity("api");
    expect(map.has("api")).toBe(true);
    expect((map.get("api") ?? []).length).toBeGreaterThan(0);
  });

  test("returns all section types when called without argument", () => {
    globalCodecAffinityStore.record({ sectionType: "api", codecFormat: "int16", reconstructionMae: 0.00002, ndcgAt5: 0.99, latencyMs: 8 });
    globalCodecAffinityStore.record({ sectionType: "prose", codecFormat: "int8", reconstructionMae: 0.005, ndcgAt5: 0.95, latencyMs: 12 });
    const map = getCodecAffinity();
    expect(map.has("api")).toBe(true);
    expect(map.has("prose")).toBe(true);
  });

  test("returns empty array for unknown section type", () => {
    const map = getCodecAffinity("unknown");
    const arr = map.get("unknown") ?? [];
    expect(arr).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// selectCodecForRetrieval + recordCodecQuality — module-level singleton API
// ---------------------------------------------------------------------------

describe("selectCodecForRetrieval (module-level singleton)", () => {
  test("returns a valid NegotiatorResult", () => {
    const result = selectCodecForRetrieval(makeCtx());
    expect(NEGOTIATOR_CODEC_CHAIN).toContain(result.codecFormat as NegotiatorCodecTier);
    expect(result.tierIndex).toBeGreaterThanOrEqual(0);
    expect(result.tierIndex).toBeLessThan(NEGOTIATOR_CODEC_CHAIN.length);
    expect(typeof result.rationale).toBe("string");
    expect(result.estimatedMemoryBytes).toBeGreaterThan(0);
  });

  test("recordCodecQuality does not throw", () => {
    expect(() => {
      recordCodecQuality(makeObs({ codecFormat: "bfloat16", ndcgAt5: 0.98 }), makeCtx());
    }).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// End-to-end: negotiate → record → affinity loop
// ---------------------------------------------------------------------------

describe("End-to-end negotiate → record → affinity loop", () => {
  test("affinity improves over repeated observations for a section type", () => {
    const store = new CodecAffinityStore();
    const neg = new QuantizationNegotiator({ affinityStore: store, mlpMinObservations: 999 });
    const ctx = makeCtx({ sectionType: "architecture" });

    // Simulate 10 retrievals where int8 consistently has high NDCG
    for (let i = 0; i < 10; i++) {
      neg.recordObservation(
        { codecFormat: "int8", sectionType: "architecture", reconstructionMae: 0.004, ndcgAt5: 0.97, latencyMs: 11 },
        ctx,
      );
    }

    // The store should now prefer int8 for architecture sections
    expect(store.getBestCodec("architecture")).toBe("int8");
  });

  test("affinity with poor quality codec is outscored by better codec", () => {
    const store = new CodecAffinityStore();
    const neg = new QuantizationNegotiator({ affinityStore: store, mlpMinObservations: 999 });
    const ctx = makeCtx({ sectionType: "test-fixtures" });

    // int4 gives poor NDCG
    for (let i = 0; i < 5; i++) {
      neg.recordObservation(
        { codecFormat: "int4", sectionType: "test-fixtures", reconstructionMae: 0.02, ndcgAt5: 0.72, latencyMs: 6 },
        ctx,
      );
    }

    // bfloat16 gives excellent NDCG
    for (let i = 0; i < 5; i++) {
      neg.recordObservation(
        { codecFormat: "bfloat16", sectionType: "test-fixtures", reconstructionMae: 0.0002, ndcgAt5: 0.99, latencyMs: 9 },
        ctx,
      );
    }

    // bfloat16 has higher affinity score: 0.99 - 0.002 = 0.988 vs int4: 0.72 - 0.2 = 0.52
    expect(store.getBestCodec("test-fixtures")).toBe("bfloat16");
  });

  test("negotiator selection is consistent across multiple calls with same context", () => {
    const store = new CodecAffinityStore();
    const neg = new QuantizationNegotiator({ affinityStore: store, mlpMinObservations: 999 });
    const ctx = makeCtx();
    const r1 = neg.selectCodec(ctx);
    const r2 = neg.selectCodec(ctx);
    expect(r1.codecFormat).toBe(r2.codecFormat);
    expect(r1.tierIndex).toBe(r2.tierIndex);
  });
});
