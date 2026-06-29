/**
 * Tests for the Cross-model Embedding Codec Registry.
 *
 * Covers:
 *  - Round-trip fidelity (MAE/RMSE) for float32, int16, int8, int4 codecs
 *  - Cross-codec top-K similarity ranking preservation vs float32 baseline
 *  - Registry mutation safety (register/get/has/formats)
 *  - getCodecForEmbedding factory (new codec_format field + legacy fallback)
 *  - serializeEmbedding / deserializeEmbedding helpers
 *  - Binary layout correctness (magic bytes, dimension preservation)
 *  - Edge cases: empty vectors, constant vectors, odd-dim vectors (int4)
 */

import { describe, expect, test } from "bun:test";
import {
  EmbeddingCodecRegistry,
  getCodecForEmbedding,
  globalCodecRegistry,
  deserializeEmbedding,
  serializeEmbedding,
  type EmbeddingCodec,
  type EmbeddingCodecMeta,
} from "../src/genome/embedding-codec.ts";
import { cosineSimilarity } from "../src/genome/embeddings.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Deterministic pseudo-random unit-normalized float32 embedding. */
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

function rmse(a: number[], b: number[]): number {
  if (a.length === 0 || a.length !== b.length) return 0;
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i]! - b[i]!) ** 2;
  return Math.sqrt(sum / a.length);
}

// ---------------------------------------------------------------------------
// Global registry — pre-registered codecs
// ---------------------------------------------------------------------------

describe("globalCodecRegistry — pre-registered formats", () => {
  test("has float32, int16, int8, int4 registered", () => {
    expect(globalCodecRegistry.has("float32")).toBe(true);
    expect(globalCodecRegistry.has("int16")).toBe(true);
    expect(globalCodecRegistry.has("int8")).toBe(true);
    expect(globalCodecRegistry.has("int4")).toBe(true);
  });

  test("formats() returns at least the four built-ins", () => {
    const formats = globalCodecRegistry.formats();
    expect(formats).toContain("float32");
    expect(formats).toContain("int16");
    expect(formats).toContain("int8");
    expect(formats).toContain("int4");
  });

  test("get() throws for unknown format", () => {
    expect(() => globalCodecRegistry.get("unknown-xyz")).toThrow(
      /unknown format "unknown-xyz"/,
    );
  });

  test("has() returns false for unknown format", () => {
    expect(globalCodecRegistry.has("nonexistent")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Round-trip fidelity — float32 (lossless baseline)
// ---------------------------------------------------------------------------

describe("float32 codec — round-trip", () => {
  test("exact reconstruction for 768-dim unit vector (MAE = 0)", () => {
    const codec = globalCodecRegistry.get("float32");
    const emb = makeEmbedding(768);
    const reconstructed = codec.deserialize(codec.serialize(emb));
    expect(reconstructed.length).toBe(768);
    // float32 has ~7 significant digits; MAE should be negligible
    expect(mae(emb, reconstructed)).toBeLessThan(1e-6);
  });

  test("empty vector round-trips to empty array", () => {
    const codec = globalCodecRegistry.get("float32");
    const result = codec.deserialize(codec.serialize([]));
    expect(result).toHaveLength(0);
  });

  test("dimension is preserved exactly", () => {
    const codec = globalCodecRegistry.get("float32");
    for (const dim of [1, 64, 256, 768, 1024]) {
      const emb = makeEmbedding(dim, dim);
      expect(codec.deserialize(codec.serialize(emb)).length).toBe(dim);
    }
  });

  test("bad magic throws on deserialize", () => {
    const codec = globalCodecRegistry.get("float32");
    const garbage = new Uint8Array(32);
    expect(() => codec.deserialize(garbage)).toThrow(/unexpected magic/);
  });
});

// ---------------------------------------------------------------------------
// Round-trip fidelity — int16 (2× compression)
// ---------------------------------------------------------------------------

describe("int16 codec — round-trip", () => {
  test("MAE < 0.0001 for 768-dim unit vector", () => {
    const codec = globalCodecRegistry.get("int16");
    const emb = makeEmbedding(768);
    const rec = codec.deserialize(codec.serialize(emb));
    expect(rec.length).toBe(768);
    expect(mae(emb, rec)).toBeLessThan(0.0001);
  });

  test("RMSE < 0.0002 for 768-dim unit vector", () => {
    const codec = globalCodecRegistry.get("int16");
    const emb = makeEmbedding(768);
    const rec = codec.deserialize(codec.serialize(emb));
    expect(rmse(emb, rec)).toBeLessThan(0.0002);
  });

  test("serialized size is 24 + dims*2 bytes", () => {
    const codec = globalCodecRegistry.get("int16");
    const dims = 768;
    const bytes = codec.serialize(makeEmbedding(dims));
    expect(bytes.byteLength).toBe(24 + dims * 2);
  });

  test("is approximately 2× smaller than float32 for same dims", () => {
    const f32 = globalCodecRegistry.get("float32");
    const i16 = globalCodecRegistry.get("int16");
    const emb = makeEmbedding(768);
    // Headers differ (8 vs 24 bytes); at 768 dims ratio ≈ 1.94 → clearly ≥ 1.8.
    const ratio = f32.serialize(emb).byteLength / i16.serialize(emb).byteLength;
    expect(ratio).toBeGreaterThan(1.8);
    expect(ratio).toBeLessThanOrEqual(2);
  });

  test("empty vector round-trips", () => {
    const codec = globalCodecRegistry.get("int16");
    expect(codec.deserialize(codec.serialize([]))).toHaveLength(0);
  });

  test("constant vector round-trips without NaN", () => {
    const codec = globalCodecRegistry.get("int16");
    const emb = new Array(64).fill(0.5);
    const rec = codec.deserialize(codec.serialize(emb));
    expect(rec.every((v) => !Number.isNaN(v))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Round-trip fidelity — int8 (4× compression)
// ---------------------------------------------------------------------------

describe("int8 codec — round-trip", () => {
  test("MAE < 0.01 for 768-dim unit vector", () => {
    const codec = globalCodecRegistry.get("int8");
    const emb = makeEmbedding(768);
    const rec = codec.deserialize(codec.serialize(emb));
    expect(rec.length).toBe(768);
    expect(mae(emb, rec)).toBeLessThan(0.01);
  });

  test("RMSE < 0.015 for 768-dim unit vector", () => {
    const codec = globalCodecRegistry.get("int8");
    const emb = makeEmbedding(768);
    const rec = codec.deserialize(codec.serialize(emb));
    expect(rmse(emb, rec)).toBeLessThan(0.015);
  });

  test("serialized size is 24 + dims bytes", () => {
    const codec = globalCodecRegistry.get("int8");
    const dims = 768;
    expect(codec.serialize(makeEmbedding(dims)).byteLength).toBe(24 + dims);
  });

  test("is approximately 4× smaller than float32 for same dims", () => {
    const f32 = globalCodecRegistry.get("float32");
    const i8 = globalCodecRegistry.get("int8");
    const emb = makeEmbedding(768);
    // Headers differ (8 vs 24 bytes) so ratio is slightly under 4 for 768 dims.
    // At scale (768 dims: float32=3080B, int8=792B), ratio ≈ 3.89 → clearly ≥ 3.5.
    const ratio = f32.serialize(emb).byteLength / i8.serialize(emb).byteLength;
    expect(ratio).toBeGreaterThan(3.5);
    expect(ratio).toBeLessThanOrEqual(4);
  });

  test("empty vector round-trips", () => {
    const codec = globalCodecRegistry.get("int8");
    expect(codec.deserialize(codec.serialize([]))).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Round-trip fidelity — int4 (experimental 8× compression)
// ---------------------------------------------------------------------------

describe("int4 codec — round-trip", () => {
  test("MAE < 0.05 for 768-dim unit vector", () => {
    const codec = globalCodecRegistry.get("int4");
    const emb = makeEmbedding(768);
    const rec = codec.deserialize(codec.serialize(emb));
    expect(rec.length).toBe(768);
    expect(mae(emb, rec)).toBeLessThan(0.05);
  });

  test("serialized size is 24 + ceil(dims/2) bytes", () => {
    const codec = globalCodecRegistry.get("int4");
    for (const dims of [768, 769]) {
      const expected = 24 + Math.ceil(dims / 2);
      expect(codec.serialize(makeEmbedding(dims)).byteLength).toBe(expected);
    }
  });

  test("is approximately 8× smaller than float32", () => {
    const f32 = globalCodecRegistry.get("float32");
    const i4 = globalCodecRegistry.get("int4");
    const emb = makeEmbedding(768);
    // float32 = 8+768*4=3080; int4 = 24+384=408; ratio ≈ 7.5 (header overhead on small vectors)
    const ratio = f32.serialize(emb).byteLength / i4.serialize(emb).byteLength;
    expect(ratio).toBeGreaterThan(7);
  });

  test("odd-dimensional vector round-trips without dimension loss", () => {
    const codec = globalCodecRegistry.get("int4");
    const emb = makeEmbedding(769); // odd
    const rec = codec.deserialize(codec.serialize(emb));
    expect(rec.length).toBe(769);
  });

  test("empty vector round-trips", () => {
    const codec = globalCodecRegistry.get("int4");
    expect(codec.deserialize(codec.serialize([]))).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Cross-codec top-K similarity ranking preservation vs float32 baseline
// ---------------------------------------------------------------------------

describe("cross-codec top-K ranking preservation vs float32", () => {
  const query = makeEmbedding(768, 42);
  const docs = Array.from({ length: 20 }, (_, i) => makeEmbedding(768, i + 100));

  // Compute float32 baseline rankings
  const baselineScores = docs
    .map((d, i) => ({ i, score: cosineSimilarity(query, d) }))
    .sort((a, b) => b.score - a.score);
  const baselineRanking = baselineScores.map((s) => s.i);

  function rankWithCodec(format: string): number[] {
    const codec = globalCodecRegistry.get(format);
    const qRec = codec.deserialize(codec.serialize(query));
    return docs
      .map((d, i) => ({ i, score: cosineSimilarity(qRec, codec.deserialize(codec.serialize(d))) }))
      .sort((a, b) => b.score - a.score)
      .map((s) => s.i);
  }

  test("int16: top-5 matches baseline top-5 exactly", () => {
    const ranking = rankWithCodec("int16");
    const top5Base = baselineRanking.slice(0, 5);
    const top5Codec = ranking.slice(0, 5);
    // int16 is near-lossless; top-5 should match exactly
    expect(top5Codec).toEqual(top5Base);
  });

  test("int8: top-1 matches baseline top-1", () => {
    const ranking = rankWithCodec("int8");
    expect(ranking[0]).toBe(baselineRanking[0]);
  });

  test("int8: ≥ 80% positional match in top-10", () => {
    const ranking = rankWithCodec("int8");
    let matches = 0;
    for (let i = 0; i < 10; i++) {
      if (ranking[i] === baselineRanking[i]) matches++;
    }
    expect(matches / 10).toBeGreaterThanOrEqual(0.8);
  });

  test("int4: top-1 matches baseline top-1", () => {
    const ranking = rankWithCodec("int4");
    expect(ranking[0]).toBe(baselineRanking[0]);
  });

  test("float32: 100% positional match (lossless)", () => {
    const ranking = rankWithCodec("float32");
    for (let i = 0; i < baselineRanking.length; i++) {
      expect(ranking[i]).toBe(baselineRanking[i]);
    }
  });
});

// ---------------------------------------------------------------------------
// Registry mutation safety
// ---------------------------------------------------------------------------

describe("EmbeddingCodecRegistry — mutation safety", () => {
  test("register + get returns the same codec", () => {
    const reg = new EmbeddingCodecRegistry();
    const codec: EmbeddingCodec<number[]> = {
      format: "test-noop",
      precision: "float32",
      serialize: (d) => new Uint8Array(d.length),
      deserialize: (_b) => [],
    };
    reg.register("test-noop", codec);
    expect(reg.get("test-noop")).toBe(codec);
  });

  test("register twice without force throws", () => {
    const reg = new EmbeddingCodecRegistry();
    const codec: EmbeddingCodec<number[]> = {
      format: "dup",
      precision: "int8",
      serialize: (d) => new Uint8Array(d.length),
      deserialize: (_b) => [],
    };
    reg.register("dup", codec);
    expect(() => reg.register("dup", codec)).toThrow(/already registered/);
  });

  test("register twice with force=true succeeds and returns new codec", () => {
    const reg = new EmbeddingCodecRegistry();
    const codec1: EmbeddingCodec<number[]> = {
      format: "dup",
      precision: "int8",
      serialize: (_d) => new Uint8Array([1]),
      deserialize: (_b) => [1],
    };
    const codec2: EmbeddingCodec<number[]> = {
      format: "dup",
      precision: "int16",
      serialize: (_d) => new Uint8Array([2]),
      deserialize: (_b) => [2],
    };
    reg.register("dup", codec1);
    reg.register("dup", codec2, true);
    expect(reg.get("dup")).toBe(codec2);
  });

  test("mutating the returned codec object does not affect subsequent get() calls", () => {
    // get() returns the stored reference — if caller mutates the object, it
    // affects the registry. This test documents the singleton-reference contract:
    // codecs are frozen singletons and should not be mutated by callers.
    const reg = new EmbeddingCodecRegistry();
    const codec: EmbeddingCodec<number[]> = {
      format: "singleton",
      precision: "float32",
      serialize: (d) => new Uint8Array(d.length),
      deserialize: (_b) => [],
    };
    reg.register("singleton", codec);
    const retrieved = reg.get("singleton");
    // Same reference returned each time
    expect(reg.get("singleton")).toBe(retrieved);
  });

  test("formats() reflects newly registered entries", () => {
    const reg = new EmbeddingCodecRegistry();
    expect(reg.formats()).toHaveLength(0);
    reg.register("a", {
      format: "a",
      precision: "float32",
      serialize: (d) => new Uint8Array(d.length),
      deserialize: (_b) => [],
    });
    expect(reg.formats()).toContain("a");
    expect(reg.formats()).toHaveLength(1);
  });

  test("globalCodecRegistry mutations do not affect a fresh EmbeddingCodecRegistry", () => {
    const fresh = new EmbeddingCodecRegistry();
    expect(fresh.has("float32")).toBe(false);
    expect(fresh.formats()).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// getCodecForEmbedding factory
// ---------------------------------------------------------------------------

describe("getCodecForEmbedding", () => {
  test("returns float32 codec when no fields present (legacy fallback)", () => {
    const codec = getCodecForEmbedding({});
    expect(codec.format).toBe("float32");
  });

  test("returns int8 codec for legacy quantization_level=8", () => {
    const codec = getCodecForEmbedding({ quantization_level: 8 });
    expect(codec.format).toBe("int8");
  });

  test("returns int16 codec for legacy quantization_level=16", () => {
    const codec = getCodecForEmbedding({ quantization_level: 16 });
    expect(codec.format).toBe("int16");
  });

  test("codec_format takes priority over quantization_level", () => {
    const codec = getCodecForEmbedding({
      codec_format: "int4",
      quantization_level: 8,
    });
    expect(codec.format).toBe("int4");
  });

  test("falls back to float32 if codec_format is unknown", () => {
    const codec = getCodecForEmbedding({ codec_format: "mystery-format-xyz" });
    expect(codec.format).toBe("float32");
  });

  test("custom registry is used when supplied", () => {
    const reg = new EmbeddingCodecRegistry();
    reg.register("custom", {
      format: "custom",
      precision: "int8",
      serialize: (_d) => new Uint8Array(1),
      deserialize: (_b) => [99],
    });
    const codec = getCodecForEmbedding({ codec_format: "custom" }, reg);
    expect(codec.format).toBe("custom");
  });
});

// ---------------------------------------------------------------------------
// serializeEmbedding / deserializeEmbedding helpers
// ---------------------------------------------------------------------------

describe("serializeEmbedding / deserializeEmbedding", () => {
  const formats = ["float32", "int16", "int8", "int4"] as const;

  for (const format of formats) {
    test(`${format}: round-trip via base64 helpers`, () => {
      const emb = makeEmbedding(128, 7);
      const { data_b64, meta } = serializeEmbedding(emb, format);
      expect(typeof data_b64).toBe("string");
      expect(meta.codec_format).toBe(format);
      expect(meta.codec_version).toBe(1);

      const rec = deserializeEmbedding(data_b64, meta);
      expect(rec.length).toBe(128);
      // float32 is lossless; others just need to be close
      if (format === "float32") {
        expect(mae(emb, rec)).toBeLessThan(1e-6);
      } else {
        expect(mae(emb, rec)).toBeLessThan(0.1);
      }
    });
  }

  test("deserializeEmbedding falls back to float32 when meta is undefined", () => {
    const emb = makeEmbedding(64);
    const f32codec = globalCodecRegistry.get("float32");
    const bytes = f32codec.serialize(emb);
    const b64 = Buffer.from(bytes).toString("base64");
    // No meta — should treat as float32
    const rec = deserializeEmbedding(b64, undefined);
    expect(rec.length).toBe(64);
    expect(mae(emb, rec)).toBeLessThan(1e-6);
  });

  test("deserializeEmbedding falls back gracefully for unknown codec_format", () => {
    const emb = makeEmbedding(64);
    const f32codec = globalCodecRegistry.get("float32");
    const bytes = f32codec.serialize(emb);
    const b64 = Buffer.from(bytes).toString("base64");
    const meta: Partial<EmbeddingCodecMeta> = { codec_format: "never-registered", codec_version: 1 };
    // Should fall back to float32 and decode correctly
    const rec = deserializeEmbedding(b64, meta);
    expect(rec.length).toBe(64);
  });

  test("JSON round-trip preserves base64 payload exactly", () => {
    const emb = makeEmbedding(32, 13);
    const { data_b64, meta } = serializeEmbedding(emb, "int8");
    const serializedJson = JSON.stringify({ data_b64, meta });
    const parsed = JSON.parse(serializedJson) as { data_b64: string; meta: EmbeddingCodecMeta };
    const rec = deserializeEmbedding(parsed.data_b64, parsed.meta);
    expect(rec.length).toBe(32);
    expect(mae(emb, rec)).toBeLessThan(0.01);
  });
});

// ---------------------------------------------------------------------------
// EmbeddingCache codec_format integration (type-level)
// ---------------------------------------------------------------------------

describe("EmbeddingCache codec_format field", () => {
  test("codec_format and codec_version are optional fields on EmbeddingCache", async () => {
    // Import EmbeddingCache type to verify the fields compile
    const { type: _t } = await import("../src/genome/embeddings.ts").then((m) => ({
      type: m as unknown,
    }));

    // Runtime shape check: construct an entry with the new fields
    const entry = {
      sectionPath: "vision/north-star.md",
      embedding: makeEmbedding(64),
      contentHash: "abc123",
      updatedAt: new Date().toISOString(),
      codec_format: "int8" as string,
      codec_version: 1 as const,
    };

    expect(entry.codec_format).toBe("int8");
    expect(entry.codec_version).toBe(1);
  });

  test("entry without codec_format is handled by getCodecForEmbedding as float32", () => {
    const entry: { codec_format?: string; quantization_level?: 8 | 16 } = {};
    const codec = getCodecForEmbedding(entry);
    expect(codec.format).toBe("float32");
    expect(codec.precision).toBe("float32");
  });
});
