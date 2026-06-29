/**
 * Cross-model Embedding Codec Registry
 *
 * Provides a standardized serialize/deserialize interface for embedding vectors
 * across backends (Ollama, API-based, quantized). Each codec encapsulates its
 * own binary layout so EmbeddingCache and embeddings.ts never need to know the
 * encoding details.
 *
 * Pre-registered codecs:
 *  - float32  : Lossless baseline; 4 bytes/dim. Header: magic(4) + dims(4).
 *  - int16    : 2× compression, ~0.00002 MAE; 2 bytes/dim. Header includes min/max.
 *  - int8     : 4× compression, ~0.005 MAE; 1 byte/dim. Header includes min/max.
 *  - int4     : Experimental 8× compression, ~0.02 MAE; 0.5 bytes/dim (packed nibbles).
 *
 * Any new quantization scheme or embedding backend can be plugged in by calling
 * `globalCodecRegistry.register(format, codec)` without touching EmbeddingCache
 * or embeddings.ts.
 */

// ---------------------------------------------------------------------------
// Core interface
// ---------------------------------------------------------------------------

/**
 * A codec encapsulates the binary serialization format for an embedding vector.
 *
 * @template T  The in-memory representation (usually `number[]`).
 */
export interface EmbeddingCodec<T = number[]> {
  /** Unique identifier stored alongside the embedding for auto-detection. */
  readonly format: string;
  /** Memory precision tier. */
  readonly precision: "int4" | "int8" | "int16" | "float32";
  /** Serialize an embedding to bytes. */
  serialize(data: T): Uint8Array;
  /** Deserialize bytes back to the in-memory representation. */
  deserialize(bytes: Uint8Array): T;
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/**
 * Registry mapping format strings to EmbeddingCodec instances.
 *
 * The global singleton `globalCodecRegistry` is pre-populated with the four
 * built-in codecs. Third-party backends register additional codecs here.
 */
export class EmbeddingCodecRegistry {
  private readonly _codecs = new Map<string, EmbeddingCodec<number[]>>();

  /**
   * Register a codec.
   * Throws if the format name is already registered (prevents silent clobber).
   * Use `force: true` to override an existing codec (e.g., testing).
   */
  register(format: string, codec: EmbeddingCodec<number[]>, force = false): void {
    if (!force && this._codecs.has(format)) {
      throw new Error(
        `EmbeddingCodecRegistry: format "${format}" is already registered. ` +
          `Pass force=true to override.`,
      );
    }
    this._codecs.set(format, codec);
  }

  /**
   * Retrieve a codec by format name.
   * Throws if the format is not registered — callers must handle unknown formats
   * by checking `has()` first when dealing with untrusted cache data.
   */
  get(format: string): EmbeddingCodec<number[]> {
    const codec = this._codecs.get(format);
    if (!codec) {
      throw new Error(
        `EmbeddingCodecRegistry: unknown format "${format}". ` +
          `Known formats: ${[...this._codecs.keys()].join(", ") || "(none)"}`,
      );
    }
    return codec;
  }

  /** Returns true if the format is registered. */
  has(format: string): boolean {
    return this._codecs.has(format);
  }

  /** List all registered format names. */
  formats(): string[] {
    return [...this._codecs.keys()];
  }
}

// ---------------------------------------------------------------------------
// Binary helpers
// ---------------------------------------------------------------------------

/** Write a float64 to a DataView at byteOffset (big-endian). */
function writeF64(view: DataView, offset: number, value: number): void {
  view.setFloat64(offset, value, false);
}

/** Read a float64 from a DataView at byteOffset (big-endian). */
function readF64(view: DataView, offset: number): number {
  return view.getFloat64(offset, false);
}

// Magic bytes for each format — 4 ASCII chars encoded as uint32 BE.
// These allow robust format detection from raw bytes.
const MAGIC_FLOAT32 = 0x46333200; // "F32\0"
const MAGIC_INT16 = 0x49313600; // "I16\0"
const MAGIC_INT8 = 0x49303800; // "I08\0"
const MAGIC_INT4 = 0x49303400; // "I04\0"

// ---------------------------------------------------------------------------
// float32 codec (lossless baseline)
// ---------------------------------------------------------------------------
//
// Layout:  [magic:4][dims:4][f32×dims]
//  magic  : uint32 BE = MAGIC_FLOAT32
//  dims   : uint32 BE = number of dimensions
//  data   : dims × float32 LE (IEEE 754)
//
// Total bytes: 8 + dims*4

const float32Codec: EmbeddingCodec<number[]> = {
  format: "float32",
  precision: "float32",

  serialize(data: number[]): Uint8Array {
    const dims = data.length;
    const buf = new ArrayBuffer(8 + dims * 4);
    const view = new DataView(buf);
    view.setUint32(0, MAGIC_FLOAT32, false);
    view.setUint32(4, dims, false);
    for (let i = 0; i < dims; i++) {
      view.setFloat32(8 + i * 4, data[i]!, true); // LE
    }
    return new Uint8Array(buf);
  },

  deserialize(bytes: Uint8Array): number[] {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const magic = view.getUint32(0, false);
    if (magic !== MAGIC_FLOAT32) {
      throw new Error(`float32 codec: unexpected magic 0x${magic.toString(16)}`);
    }
    const dims = view.getUint32(4, false);
    const result: number[] = new Array(dims);
    for (let i = 0; i < dims; i++) {
      result[i] = view.getFloat32(8 + i * 4, true);
    }
    return result;
  },
};

// ---------------------------------------------------------------------------
// int16 codec (2× compression, ~0.00002 MAE)
// ---------------------------------------------------------------------------
//
// Layout:  [magic:4][dims:4][min:8][max:8][i16×dims]
//  magic  : uint32 BE = MAGIC_INT16
//  dims   : uint32 BE
//  min    : float64 BE — min value used for min-max scaling
//  max    : float64 BE — max value
//  data   : dims × int16 LE
//
// Total bytes: 24 + dims*2

const int16Codec: EmbeddingCodec<number[]> = {
  format: "int16",
  precision: "int16",

  serialize(data: number[]): Uint8Array {
    const dims = data.length;
    if (dims === 0) {
      const buf = new ArrayBuffer(24);
      const view = new DataView(buf);
      view.setUint32(0, MAGIC_INT16, false);
      view.setUint32(4, 0, false);
      writeF64(view, 8, 0);
      writeF64(view, 16, 0);
      return new Uint8Array(buf);
    }

    let min = data[0]!;
    let max = data[0]!;
    for (let i = 1; i < dims; i++) {
      const v = data[i]!;
      if (v < min) min = v;
      if (v > max) max = v;
    }

    const range = max - min;
    const intMin = -32768;
    const intMax = 32767;

    const buf = new ArrayBuffer(24 + dims * 2);
    const view = new DataView(buf);
    view.setUint32(0, MAGIC_INT16, false);
    view.setUint32(4, dims, false);
    writeF64(view, 8, min);
    writeF64(view, 16, max);

    for (let i = 0; i < dims; i++) {
      const normalized = range === 0 ? 0 : (data[i]! - min) / range;
      const scaled = Math.round(normalized * (intMax - intMin) + intMin);
      view.setInt16(24 + i * 2, Math.max(intMin, Math.min(intMax, scaled)), true);
    }
    return new Uint8Array(buf);
  },

  deserialize(bytes: Uint8Array): number[] {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const magic = view.getUint32(0, false);
    if (magic !== MAGIC_INT16) {
      throw new Error(`int16 codec: unexpected magic 0x${magic.toString(16)}`);
    }
    const dims = view.getUint32(4, false);
    const min = readF64(view, 8);
    const max = readF64(view, 16);
    const range = max - min;
    const intMin = -32768;
    const intMax = 32767;

    const result: number[] = new Array(dims);
    for (let i = 0; i < dims; i++) {
      const v = view.getInt16(24 + i * 2, true);
      const normalized = (v - intMin) / (intMax - intMin);
      result[i] = normalized * range + min;
    }
    return result;
  },
};

// ---------------------------------------------------------------------------
// int8 codec (4× compression, ~0.005 MAE)
// ---------------------------------------------------------------------------
//
// Layout:  [magic:4][dims:4][min:8][max:8][i8×dims]
//  magic  : uint32 BE = MAGIC_INT8
//  dims   : uint32 BE
//  min    : float64 BE
//  max    : float64 BE
//  data   : dims × int8
//
// Total bytes: 24 + dims*1

const int8Codec: EmbeddingCodec<number[]> = {
  format: "int8",
  precision: "int8",

  serialize(data: number[]): Uint8Array {
    const dims = data.length;
    if (dims === 0) {
      const buf = new ArrayBuffer(24);
      const view = new DataView(buf);
      view.setUint32(0, MAGIC_INT8, false);
      view.setUint32(4, 0, false);
      writeF64(view, 8, 0);
      writeF64(view, 16, 0);
      return new Uint8Array(buf);
    }

    let min = data[0]!;
    let max = data[0]!;
    for (let i = 1; i < dims; i++) {
      const v = data[i]!;
      if (v < min) min = v;
      if (v > max) max = v;
    }

    const range = max - min;
    const intMin = -128;
    const intMax = 127;

    const buf = new ArrayBuffer(24 + dims);
    const view = new DataView(buf);
    view.setUint32(0, MAGIC_INT8, false);
    view.setUint32(4, dims, false);
    writeF64(view, 8, min);
    writeF64(view, 16, max);

    for (let i = 0; i < dims; i++) {
      const normalized = range === 0 ? 0 : (data[i]! - min) / range;
      const scaled = Math.round(normalized * (intMax - intMin) + intMin);
      view.setInt8(24 + i, Math.max(intMin, Math.min(intMax, scaled)));
    }
    return new Uint8Array(buf);
  },

  deserialize(bytes: Uint8Array): number[] {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const magic = view.getUint32(0, false);
    if (magic !== MAGIC_INT8) {
      throw new Error(`int8 codec: unexpected magic 0x${magic.toString(16)}`);
    }
    const dims = view.getUint32(4, false);
    const min = readF64(view, 8);
    const max = readF64(view, 16);
    const range = max - min;
    const intMin = -128;
    const intMax = 127;

    const result: number[] = new Array(dims);
    for (let i = 0; i < dims; i++) {
      const v = view.getInt8(24 + i);
      const normalized = (v - intMin) / (intMax - intMin);
      result[i] = normalized * range + min;
    }
    return result;
  },
};

// ---------------------------------------------------------------------------
// int4 codec (experimental 8× compression, ~0.02 MAE)
// ---------------------------------------------------------------------------
//
// Each dimension is a 4-bit nibble in range [0, 15] mapped from [min, max].
// Two dimensions are packed into one byte: high nibble = dim[2i], low = dim[2i+1].
// Odd-dimensional vectors have a zero-padded final nibble.
//
// Layout:  [magic:4][dims:4][min:8][max:8][packed_nibbles: ceil(dims/2) bytes]
//
// Total bytes: 24 + ceil(dims/2)

const int4Codec: EmbeddingCodec<number[]> = {
  format: "int4",
  precision: "int4",

  serialize(data: number[]): Uint8Array {
    const dims = data.length;
    const packedBytes = Math.ceil(dims / 2);
    const buf = new ArrayBuffer(24 + packedBytes);
    const view = new DataView(buf);

    view.setUint32(0, MAGIC_INT4, false);
    view.setUint32(4, dims, false);

    if (dims === 0) {
      writeF64(view, 8, 0);
      writeF64(view, 16, 0);
      return new Uint8Array(buf);
    }

    let min = data[0]!;
    let max = data[0]!;
    for (let i = 1; i < dims; i++) {
      const v = data[i]!;
      if (v < min) min = v;
      if (v > max) max = v;
    }

    writeF64(view, 8, min);
    writeF64(view, 16, max);

    const range = max - min;

    for (let i = 0; i < packedBytes; i++) {
      const idx0 = i * 2;
      const idx1 = idx0 + 1;

      const normalize = (idx: number): number => {
        if (idx >= dims) return 0;
        const normalized = range === 0 ? 0 : (data[idx]! - min) / range;
        return Math.max(0, Math.min(15, Math.round(normalized * 15)));
      };

      const n0 = normalize(idx0);
      const n1 = normalize(idx1);
      view.setUint8(24 + i, (n0 << 4) | n1);
    }

    return new Uint8Array(buf);
  },

  deserialize(bytes: Uint8Array): number[] {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const magic = view.getUint32(0, false);
    if (magic !== MAGIC_INT4) {
      throw new Error(`int4 codec: unexpected magic 0x${magic.toString(16)}`);
    }
    const dims = view.getUint32(4, false);
    const min = readF64(view, 8);
    const max = readF64(view, 16);
    const range = max - min;

    const result: number[] = new Array(dims);
    for (let i = 0; i < dims; i++) {
      const byteIdx = Math.floor(i / 2);
      const byte = view.getUint8(24 + byteIdx);
      const nibble = i % 2 === 0 ? (byte >> 4) & 0xf : byte & 0xf;
      result[i] = (nibble / 15) * range + min;
    }
    return result;
  },
};

// ---------------------------------------------------------------------------
// Global singleton registry (pre-populated)
// ---------------------------------------------------------------------------

export const globalCodecRegistry = new EmbeddingCodecRegistry();
globalCodecRegistry.register("float32", float32Codec);
globalCodecRegistry.register("int16", int16Codec);
globalCodecRegistry.register("int8", int8Codec);
globalCodecRegistry.register("int4", int4Codec);

// ---------------------------------------------------------------------------
// EmbeddingCache integration helpers
// ---------------------------------------------------------------------------

/**
 * Codec metadata stored alongside an embedding cache entry so old caches
 * can auto-detect their encoding on load.
 */
export interface EmbeddingCodecMeta {
  /** Codec format name (e.g. "int8"). Matches EmbeddingCodec.format. */
  codec_format: string;
  /** Codec version for forward compatibility. Currently always 1. */
  codec_version: 1;
}

/**
 * Serialize an embedding using the named codec and return both the byte
 * payload (as a base64 string for JSON persistence) and the metadata header.
 */
export function serializeEmbedding(
  embedding: number[],
  format: string,
  registry: EmbeddingCodecRegistry = globalCodecRegistry,
): { data_b64: string; meta: EmbeddingCodecMeta } {
  const codec = registry.get(format);
  const bytes = codec.serialize(embedding);
  const data_b64 = Buffer.from(bytes).toString("base64");
  return { data_b64, meta: { codec_format: format, codec_version: 1 } };
}

/**
 * Deserialize an embedding from a base64 payload using the codec identified
 * by the stored metadata. Falls back to float32 if meta is absent (legacy).
 */
export function deserializeEmbedding(
  data_b64: string,
  meta: Partial<EmbeddingCodecMeta> | undefined,
  registry: EmbeddingCodecRegistry = globalCodecRegistry,
): number[] {
  const format = meta?.codec_format ?? "float32";
  // If the format isn't registered (e.g., old cache with no meta) fall back gracefully
  const effectiveFormat = registry.has(format) ? format : "float32";
  const codec = registry.get(effectiveFormat);
  const bytes = Buffer.from(data_b64, "base64");
  return codec.deserialize(new Uint8Array(bytes));
}

/**
 * Factory: return the appropriate codec for an EmbeddingCache entry.
 *
 * Decision logic (in priority order):
 *  1. If the entry carries `codec_format` metadata → use that codec.
 *  2. If the entry has a legacy `quantization_level` field → use int8 or int16.
 *  3. Otherwise → float32 baseline.
 *
 * This ensures old caches (written before the codec registry existed) are
 * handled correctly without any migration step.
 */
export function getCodecForEmbedding(
  cache: {
    codec_format?: string;
    quantization_level?: 8 | 16;
  },
  registry: EmbeddingCodecRegistry = globalCodecRegistry,
): EmbeddingCodec<number[]> {
  if (cache.codec_format && registry.has(cache.codec_format)) {
    return registry.get(cache.codec_format);
  }
  if (cache.quantization_level === 8) return registry.get("int8");
  if (cache.quantization_level === 16) return registry.get("int16");
  return registry.get("float32");
}
