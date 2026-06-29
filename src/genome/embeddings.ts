/**
 * Genome embeddings — semantic search via local Ollama embeddings.
 *
 * Generates embeddings for genome sections using a local Ollama instance,
 * caches them on disk to avoid recomputation, and provides cosine-similarity
 * based semantic search as a retrieval alternative to keyword TF-IDF.
 *
 * Gracefully degrades: if Ollama is not running, all functions return null/empty
 * and the caller falls back to keyword search.
 */

import { createHash } from "crypto";
import { existsSync } from "fs";
import { mkdir, readFile, rename, writeFile } from "fs/promises";
import { dirname, join } from "path";
import { estimateTokens, genomeDir, loadManifest, type SectionMeta } from "./manifest.ts";
import type { RetrievedSection } from "./retriever.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface EmbeddingCache {
  /** Relative section path within genome dir */
  sectionPath: string;
  /** Float32 embedding vector */
  embedding: number[];
  /** MD5 hash of the section content — used to detect stale embeddings */
  contentHash: string;
  /** ISO timestamp of last embedding generation */
  updatedAt: string;
  /**
   * Quantized embedding stored as a plain number[] (serializable) representing
   * Int8 or Int16 values. Reconstructed at query time via dequantizeEmbedding.
   */
  embedding_quantized?: number[];
  /** Bit depth used for quantization (8 or 16) */
  quantization_level?: 8 | 16;
  /** Per-entry min value used for min-max dequantization */
  quantization_min?: number;
  /** Per-entry max value used for min-max dequantization */
  quantization_max?: number;
  /** Mean absolute error between original and reconstructed embedding */
  quantError?: number;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const OLLAMA_BASE_URL = "http://localhost:11434";
export const DEFAULT_EMBEDDING_MODEL = "nomic-embed-text";
const CACHE_FILE = "evolution/embeddings.json";

/** Timeout for Ollama connectivity check (ms) */
const AVAILABILITY_TIMEOUT_MS = 2000;
/** Timeout for individual embedding generation (ms) */
const EMBEDDING_TIMEOUT_MS = 15000;

/**
 * All supported embedding models with their quality/latency tier.
 *
 * Re-exported here for consumers that only import from embeddings.ts
 * and do not need the full EmbeddingRouter orchestration.
 */
export const SUPPORTED_EMBEDDING_MODELS = [
  "nomic-embed-text",
  "all-minilm-l6-v2",
  "bge-base-en-v1-5",
] as const;

export type SupportedEmbeddingModel = (typeof SUPPORTED_EMBEDDING_MODELS)[number];

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

/**
 * Quantize a float32 embedding vector to int8 or int16 using min-max normalization.
 *
 * Maps the float range [min, max] to the integer range of the target type:
 *   int8:  [-128, 127]
 *   int16: [-32768, 32767]
 *
 * Returns the quantized typed array plus the min/max scalars required for
 * accurate dequantization.
 */
export function quantizeEmbedding(
  embedding: number[],
  bit_depth: 8 | 16,
): { quantized: Int8Array | Int16Array; min: number; max: number } {
  if (embedding.length === 0) {
    return {
      quantized: bit_depth === 8 ? new Int8Array(0) : new Int16Array(0),
      min: 0,
      max: 0,
    };
  }

  let min = embedding[0]!;
  let max = embedding[0]!;
  for (let i = 1; i < embedding.length; i++) {
    const v = embedding[i]!;
    if (v < min) min = v;
    if (v > max) max = v;
  }

  const range = max - min;
  const intMax = bit_depth === 8 ? 127 : 32767;
  const intMin = bit_depth === 8 ? -128 : -32768;

  const quantized =
    bit_depth === 8 ? new Int8Array(embedding.length) : new Int16Array(embedding.length);

  for (let i = 0; i < embedding.length; i++) {
    const normalized = range === 0 ? 0 : (embedding[i]! - min) / range; // [0, 1]
    // Scale to [intMin, intMax] and round to nearest integer
    const scaled = Math.round(normalized * (intMax - intMin) + intMin);
    // Clamp to guard against floating-point edge cases
    quantized[i] = Math.max(intMin, Math.min(intMax, scaled));
  }

  return { quantized, min, max };
}

/**
 * Reconstruct a float32 embedding from a quantized representation.
 *
 * Inverts the min-max scaling applied by quantizeEmbedding. The reconstruction
 * is lossy — the mean absolute error is typically <0.005 for int8 and <0.00002
 * for int16 on unit-normalized embedding vectors.
 *
 * Expected overhead: ~2 µs per 768-dim vector on modern hardware.
 */
export function dequantizeEmbedding(
  quantized: Int8Array | Int16Array | number[],
  min: number,
  max: number,
): number[] {
  const intMax = quantized instanceof Int8Array ? 127 : 32767;
  const intMin = quantized instanceof Int8Array ? -128 : -32768;
  const range = max - min;
  const result = new Array<number>(quantized.length);

  for (let i = 0; i < quantized.length; i++) {
    const v = quantized[i]!;
    const normalized = (v - intMin) / (intMax - intMin); // [0, 1]
    result[i] = normalized * range + min;
  }

  return result;
}

/**
 * Compute mean absolute error between an original float32 embedding and its
 * reconstructed version after a quantize/dequantize round-trip.
 */
export function quantizationError(original: number[], reconstructed: number[]): number {
  if (original.length === 0 || original.length !== reconstructed.length) return 0;
  let sum = 0;
  for (let i = 0; i < original.length; i++) {
    sum += Math.abs(original[i]! - reconstructed[i]!);
  }
  return sum / original.length;
}

// ---------------------------------------------------------------------------
// Ollama connectivity
// ---------------------------------------------------------------------------

/**
 * Check whether a local Ollama instance is reachable.
 * Uses a short timeout so it never blocks the critical path.
 */
export async function isOllamaAvailable(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), AVAILABILITY_TIMEOUT_MS);
    const res = await fetch(`${OLLAMA_BASE_URL}/api/tags`, {
      signal: controller.signal,
    });
    clearTimeout(timer);
    return res.ok;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Embedding generation
// ---------------------------------------------------------------------------

/**
 * Generate an embedding vector for a text string via the Ollama embeddings API.
 * Returns null if Ollama is unreachable or the request fails.
 */
export async function generateEmbedding(
  text: string,
  model: string = DEFAULT_EMBEDDING_MODEL,
): Promise<number[] | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), EMBEDDING_TIMEOUT_MS);
    const res = await fetch(`${OLLAMA_BASE_URL}/api/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, prompt: text }),
      signal: controller.signal,
    });
    clearTimeout(timer);

    if (!res.ok) return null;

    const data = (await res.json()) as { embedding?: number[] };
    return data.embedding ?? null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Similarity
// ---------------------------------------------------------------------------

/**
 * Compute cosine similarity between two vectors.
 * Returns 0 for zero-length or mismatched vectors.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || a.length !== b.length) return 0;

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i]! * b[i]!;
    normA += a[i]! * a[i]!;
    normB += b[i]! * b[i]!;
  }

  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom === 0) return 0;

  return dotProduct / denom;
}

// ---------------------------------------------------------------------------
// Content hashing
// ---------------------------------------------------------------------------

/**
 * MD5 hash of a string — used to detect when section content has changed
 * and embeddings need regeneration.
 */
export function contentHash(content: string): string {
  return createHash("md5").update(content).digest("hex");
}

// ---------------------------------------------------------------------------
// Cache persistence
// ---------------------------------------------------------------------------

function cachePath(cwd: string): string {
  return join(genomeDir(cwd), CACHE_FILE);
}

/**
 * Load the embedding cache from disk.
 * Returns an empty array if the file doesn't exist or is corrupt.
 */
export async function loadEmbeddingCache(cwd: string): Promise<EmbeddingCache[]> {
  const path = cachePath(cwd);
  if (!existsSync(path)) return [];

  try {
    const raw = await readFile(path, "utf-8");
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed as EmbeddingCache[];
  } catch {
    return [];
  }
}

/**
 * Save the embedding cache to disk atomically.
 */
export async function saveEmbeddingCache(cwd: string, cache: EmbeddingCache[]): Promise<void> {
  const path = cachePath(cwd);
  const dir = dirname(path);
  if (!existsSync(dir)) {
    await mkdir(dir, { recursive: true });
  }
  // Atomic write: write to temp file then rename (safe on POSIX, mirrors saveManifest)
  const tmp = path + ".tmp";
  try {
    await writeFile(tmp, JSON.stringify(cache, null, 2), "utf-8");
    await rename(tmp, path);
  } catch (e) {
    const { unlink } = await import("fs/promises");
    await unlink(tmp).catch(() => {});
    throw e;
  }
}

// ---------------------------------------------------------------------------
// Update embeddings
// ---------------------------------------------------------------------------

export interface UpdateEmbeddingsOptions {
  /** When true, also store a quantized copy alongside the float32 embedding */
  quantize?: boolean;
  /** Bit depth for quantization (default: 8) */
  bitDepth?: 8 | 16;
}

/**
 * Update embeddings for all genome sections.
 *
 * Reads the manifest, checks each section against the cache (by content hash),
 * and generates new embeddings only for sections whose content has changed.
 *
 * When `options.quantize` is true, also computes and stores a quantized
 * representation (Int8Array serialized as number[]) alongside the float32
 * embedding. The quantization enables 75% storage reduction for disk/mobile
 * deployments while the float32 copy is retained for validation.
 *
 * Returns counts of updated and skipped (already cached) sections, plus
 * quantizationStats when quantization was requested.
 */
export async function updateEmbeddings(
  cwd: string,
  model: string = DEFAULT_EMBEDDING_MODEL,
  options: UpdateEmbeddingsOptions = {},
): Promise<{
  updated: number;
  skipped: number;
  failed: number;
  quantizationStats?: {
    totalSize: number;
    originalSize: number;
    compressionRatio: number;
    avgSimilarityDelta: number;
  };
}> {
  const manifest = await loadManifest(cwd);
  if (!manifest || manifest.sections.length === 0) {
    return { updated: 0, skipped: 0, failed: 0 };
  }

  const cache = await loadEmbeddingCache(cwd);
  const cacheMap = new Map<string, EmbeddingCache>();
  for (const entry of cache) {
    cacheMap.set(entry.sectionPath, entry);
  }

  const dir = genomeDir(cwd);
  let updated = 0;
  let skipped = 0;
  let failed = 0;

  const { quantize = false, bitDepth = 8 } = options;

  // Accumulators for quantization statistics
  let totalOriginalBytes = 0;
  let totalQuantizedBytes = 0;
  let totalSimilarityDelta = 0;
  let quantizedCount = 0;

  for (const section of manifest.sections) {
    const fullPath = join(dir, section.path);
    if (!existsSync(fullPath)) {
      failed++;
      continue;
    }

    const content = await readFile(fullPath, "utf-8");
    const hash = contentHash(content);

    // Check if cached embedding is still fresh (and quantization state matches)
    const cached = cacheMap.get(section.path);
    const quantizationUpToDate =
      !quantize ||
      (cached?.embedding_quantized !== undefined &&
        cached?.quantization_level === bitDepth);
    if (
      cached &&
      cached.contentHash === hash &&
      cached.embedding.length > 0 &&
      quantizationUpToDate
    ) {
      skipped++;
      // Still accumulate stats for skipped entries that have quantization data
      if (quantize && cached.embedding_quantized && cached.quantization_level) {
        const dims = cached.embedding.length;
        totalOriginalBytes += dims * 4; // float32 = 4 bytes
        totalQuantizedBytes += dims * (cached.quantization_level === 8 ? 1 : 2);
        if (cached.quantError !== undefined) {
          totalSimilarityDelta += cached.quantError;
          quantizedCount++;
        }
      }
      continue;
    }

    // Generate new embedding
    const embedding = await generateEmbedding(content, model);
    if (!embedding) {
      failed++;
      continue;
    }

    const entry: EmbeddingCache = {
      sectionPath: section.path,
      embedding,
      contentHash: hash,
      updatedAt: new Date().toISOString(),
    };

    if (quantize) {
      const { quantized, min, max } = quantizeEmbedding(embedding, bitDepth);
      const reconstructed = dequantizeEmbedding(quantized, min, max);
      const err = quantizationError(embedding, reconstructed);
      const simOriginal = cosineSimilarity(embedding, embedding); // = 1.0
      const simReconstructed = cosineSimilarity(embedding, reconstructed);
      const simDelta = Math.abs(simOriginal - simReconstructed);

      entry.embedding_quantized = Array.from(quantized);
      entry.quantization_level = bitDepth;
      entry.quantization_min = min;
      entry.quantization_max = max;
      entry.quantError = err;

      const dims = embedding.length;
      totalOriginalBytes += dims * 4;
      totalQuantizedBytes += dims * (bitDepth === 8 ? 1 : 2);
      totalSimilarityDelta += simDelta;
      quantizedCount++;
    }

    cacheMap.set(section.path, entry);
    updated++;
  }

  // Remove cache entries for sections that no longer exist
  const sectionPaths = new Set(manifest.sections.map((s) => s.path));
  for (const key of cacheMap.keys()) {
    if (!sectionPaths.has(key)) {
      cacheMap.delete(key);
    }
  }

  await saveEmbeddingCache(cwd, Array.from(cacheMap.values()));

  if (quantize && quantizedCount > 0) {
    return {
      updated,
      skipped,
      failed,
      quantizationStats: {
        totalSize: totalQuantizedBytes,
        originalSize: totalOriginalBytes,
        compressionRatio: totalOriginalBytes > 0 ? totalQuantizedBytes / totalOriginalBytes : 1,
        avgSimilarityDelta: totalSimilarityDelta / quantizedCount,
      },
    };
  }

  return { updated, skipped, failed };
}

// ---------------------------------------------------------------------------
// Semantic search
// ---------------------------------------------------------------------------

/**
 * Semantic search: find the most relevant genome sections for a query
 * using embedding cosine similarity.
 *
 * Returns sections ordered by similarity score (descending), packed within
 * the given token budget. Returns an empty array if embeddings are unavailable.
 */
export async function semanticSearch(
  cwd: string,
  query: string,
  maxTokens: number,
  model: string = DEFAULT_EMBEDDING_MODEL,
): Promise<RetrievedSection[]> {
  // Generate embedding for the query
  const queryEmbedding = await generateEmbedding(query, model);
  if (!queryEmbedding) return [];

  // Load cached embeddings
  const cache = await loadEmbeddingCache(cwd);
  if (cache.length === 0) return [];

  // Score all cached sections by cosine similarity
  const scored = cache
    .map((entry) => ({
      path: entry.sectionPath,
      similarity: cosineSimilarity(queryEmbedding, entry.embedding),
    }))
    .filter((s) => s.similarity > 0.3) // Minimum relevance threshold
    .sort((a, b) => b.similarity - a.similarity);

  if (scored.length === 0) return [];

  // Pack sections within token budget
  const dir = genomeDir(cwd);
  const manifest = await loadManifest(cwd);
  const sectionMap = new Map<string, SectionMeta>();
  if (manifest) {
    for (const s of manifest.sections) {
      sectionMap.set(s.path, s);
    }
  }

  const results: RetrievedSection[] = [];
  let usedTokens = 0;

  for (const { path, similarity } of scored) {
    if (usedTokens >= maxTokens) break;

    const fullPath = join(dir, path);
    if (!existsSync(fullPath)) continue;

    const content = await readFile(fullPath, "utf-8");
    const tokens = estimateTokens(content);
    const meta = sectionMap.get(path);
    const title = meta?.title ?? path;

    if (usedTokens + tokens > maxTokens) {
      // Try truncating to fit
      const remaining = maxTokens - usedTokens;
      if (remaining > 200) {
        const truncated = content.slice(0, remaining * 4) + "\n\n[... section truncated ...]";
        results.push({
          path,
          title,
          content: truncated,
          tokens: remaining,
          score: similarity,
        });
        usedTokens += remaining;
      }
      break;
    }

    results.push({
      path,
      title,
      content,
      tokens,
      score: similarity,
    });
    usedTokens += tokens;
  }

  return results;
}
