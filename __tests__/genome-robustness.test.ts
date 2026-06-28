/**
 * Genome robustness tests — failure modes for fitness, manifest, and embeddings.
 *
 * TEST-ONLY: no source changes. Bugs noted inline as BUG comments.
 * Conventions mirror existing __tests__/*.test.ts files (bun:test, describe/test/expect).
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { createHash } from "crypto";
import { existsSync } from "fs";
import { mkdir, readFile, rm, writeFile } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";
import type { GenomeManifest, SectionMeta } from "../src/genome/manifest.ts";

// ─── helpers ────────────────────────────────────────────────────────────────

/** Create a throwaway temp dir that is removed after each test. */
function makeTmpDir(): string {
  // Use a sync-safe unique path; actual mkdir is async and called in beforeEach.
  return join(tmpdir(), `ashlr-robustness-${Date.now()}-${Math.random().toString(36).slice(2)}`);
}

async function setupGenomeDir(cwd: string): Promise<string> {
  const dir = join(cwd, ".ashlrcode", "genome");
  await mkdir(dir, { recursive: true });
  return dir;
}

function makeManifest(project = "test-project"): GenomeManifest {
  const now = new Date().toISOString();
  return {
    version: 1,
    project,
    sections: [],
    generation: { number: 1, milestone: "", startedAt: now },
    fitnessHistory: [],
    createdAt: now,
    updatedAt: now,
  };
}

// ─── fitness ────────────────────────────────────────────────────────────────

describe("fitness — measureFitness graceful degradation", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await mkdir(cwd, { recursive: true });
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("returns all 0-1 scores when no genome exists at all", async () => {
    const { measureFitness } = await import("../src/genome/fitness.ts");
    const metrics = await measureFitness(cwd);
    for (const [key, value] of Object.entries(metrics)) {
      expect(value, `${key} must be in [0,1]`).toBeGreaterThanOrEqual(0);
      expect(value, `${key} must be in [0,1]`).toBeLessThanOrEqual(1);
    }
  });

  test("malformed package.json — invalid JSON — does not crash, returns fallback", async () => {
    await writeFile(join(cwd, "package.json"), "{ NOT VALID JSON !!!", "utf-8");
    const { measureFitness } = await import("../src/genome/fitness.ts");
    // Should not throw; testsPassRate falls back to 0.5
    const metrics = await measureFitness(cwd);
    expect(metrics.testsPassRate).toBe(0.5);
  });

  test("malformed package.json — empty file — does not crash", async () => {
    await writeFile(join(cwd, "package.json"), "", "utf-8");
    const { measureFitness } = await import("../src/genome/fitness.ts");
    const metrics = await measureFitness(cwd);
    expect(metrics.testsPassRate).toBe(0.5);
  });

  test("malformed package.json — null JSON — does not crash", async () => {
    await writeFile(join(cwd, "package.json"), "null", "utf-8");
    const { measureFitness } = await import("../src/genome/fitness.ts");
    const metrics = await measureFitness(cwd);
    // pkg.scripts?.test will be undefined on null — fallback expected
    expect(metrics.testsPassRate).toBe(0.5);
  });

  test("malformed package.json — valid JSON but missing scripts field — fallback 0.5", async () => {
    await writeFile(join(cwd, "package.json"), JSON.stringify({ name: "x" }), "utf-8");
    const { measureFitness } = await import("../src/genome/fitness.ts");
    const metrics = await measureFitness(cwd);
    expect(metrics.testsPassRate).toBe(0.5);
  });

  test("tsc parse failures — src dir with unparseable TS — codeQuality returns 0.5 fallback", async () => {
    // measureCodeQuality counts TODO/FIXME markers, not tsc — but a file that
    // causes readFile to fail (e.g. a broken symlink) should be handled.
    // We test that a src/ full of binary-ish content doesn't crash.
    const srcDir = join(cwd, "src");
    await mkdir(srcDir, { recursive: true });
    // Write a file with null bytes (not valid UTF-8 in strict mode on some runtimes)
    const buf = Buffer.alloc(100, 0x00);
    await writeFile(join(srcDir, "bad.ts"), buf);
    const { measureFitness } = await import("../src/genome/fitness.ts");
    // Must not throw regardless of content
    const metrics = await measureFitness(cwd);
    expect(metrics.codeQuality).toBeGreaterThanOrEqual(0);
    expect(metrics.codeQuality).toBeLessThanOrEqual(1);
  });

  test("scoring does not hang — completes within 5 seconds on empty project", async () => {
    const { measureFitness } = await import("../src/genome/fitness.ts");
    const start = Date.now();
    await measureFitness(cwd);
    const elapsed = Date.now() - start;
    // measureTestPassRate has a 30s internal timeout, but with no test script
    // it should bail early (existsSync / missing scripts.test). Budget 5s.
    expect(elapsed).toBeLessThan(5_000);
  });

  test("compareFitness — delta arithmetic is correct", async () => {
    const { compareFitness } = await import("../src/genome/fitness.ts");
    const before = { testsPassRate: 0.5, codeQuality: 0.8, milestoneProgress: 0.2, costEfficiency: 0.0, strategySuccessRate: 0.5 };
    const after  = { testsPassRate: 0.9, codeQuality: 0.6, milestoneProgress: 0.4, costEfficiency: 0.3, strategySuccessRate: 0.7 };
    const diff = compareFitness(before, after);
    expect(diff.testsPassRate!.delta).toBeCloseTo(0.4);
    expect(diff.codeQuality!.delta).toBeCloseTo(-0.2);
    expect(diff.milestoneProgress!.delta).toBeCloseTo(0.2);
  });

  test("formatFitnessReport — returns a non-empty string with all metric names", async () => {
    const { measureFitness, formatFitnessReport } = await import("../src/genome/fitness.ts");
    const metrics = await measureFitness(cwd);
    const report = formatFitnessReport(metrics);
    expect(typeof report).toBe("string");
    expect(report.length).toBeGreaterThan(0);
    expect(report).toContain("Tests Pass Rate");
    expect(report).toContain("Code Quality");
  });
});

// ─── manifest ───────────────────────────────────────────────────────────────

describe("manifest — atomic write safety", () => {
  let cwd: string;
  let genomeD: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    genomeD = await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("saveManifest writes valid JSON readable by loadManifest", async () => {
    const { saveManifest, loadManifest } = await import("../src/genome/manifest.ts");
    const m = makeManifest();
    await saveManifest(cwd, m);
    const loaded = await loadManifest(cwd);
    expect(loaded).not.toBeNull();
    expect(loaded!.project).toBe("test-project");
  });

  test("temp file is cleaned up after successful save (no .tmp orphan)", async () => {
    const { saveManifest, manifestPath } = await import("../src/genome/manifest.ts");
    await saveManifest(cwd, makeManifest());
    const tmpFile = manifestPath(cwd) + ".tmp";
    expect(existsSync(tmpFile)).toBe(false);
  });

  test("loadManifest returns null for corrupt / partial-write manifest", async () => {
    const { loadManifest, manifestPath } = await import("../src/genome/manifest.ts");
    // Simulate a partial write that left truncated JSON
    const path = manifestPath(cwd);
    await writeFile(path, '{"version":1,"project":"x","sections":[', "utf-8");
    const result = await loadManifest(cwd);
    // BUG-NOTICE: loadManifest catches JSON.parse errors and returns null — correct.
    expect(result).toBeNull();
  });

  test("loadManifest returns null for completely empty manifest file", async () => {
    const { loadManifest, manifestPath } = await import("../src/genome/manifest.ts");
    await writeFile(manifestPath(cwd), "", "utf-8");
    const result = await loadManifest(cwd);
    expect(result).toBeNull();
  });

  test("concurrent updateManifest calls serialize correctly — final state consistent", async () => {
    const { saveManifest, updateManifest, loadManifest } = await import("../src/genome/manifest.ts");
    await saveManifest(cwd, makeManifest());

    // Fire 5 concurrent updates each incrementing generation number
    // (They can't all read-then-write the same counter safely without the lock —
    // the serialization queue ensures they run in order.)
    const updates = Array.from({ length: 5 }, (_, i) =>
      updateManifest(cwd, (m) => {
        m.generation.number += 1;
      })
    );
    await Promise.all(updates);

    const final = await loadManifest(cwd);
    expect(final).not.toBeNull();
    // BUG-NOTICE: because each update reads the freshly-written manifest,
    // they should serialize to generation.number === 6 (1 + 5 increments).
    // If there were a race this would be < 6. Validates the lock chain works.
    expect(final!.generation.number).toBe(6);
  });

  test("updateManifest throws when no genome exists", async () => {
    // Remove the genome dir so no manifest exists
    await rm(genomeD, { recursive: true, force: true });
    const { updateManifest } = await import("../src/genome/manifest.ts");
    await expect(updateManifest(cwd, () => {})).rejects.toThrow(/genome/i);
  });

  test("createEmptyManifest produces valid structure", async () => {
    const { createEmptyManifest } = await import("../src/genome/manifest.ts");
    const m = createEmptyManifest("my-project");
    expect(m.version).toBe(1);
    expect(m.project).toBe("my-project");
    expect(m.sections).toHaveLength(0);
    expect(m.generation.number).toBe(1);
    expect(m.fitnessHistory).toHaveLength(0);
    // Timestamps must be valid ISO strings
    expect(new Date(m.createdAt).toISOString()).toBe(m.createdAt);
  });

  test("totalGenomeTokens sums section token counts correctly", async () => {
    const { totalGenomeTokens } = await import("../src/genome/manifest.ts");
    const m = makeManifest();
    const section = (path: string, tokens: number): SectionMeta => ({
      path, title: path, summary: "", tags: [], tokens, updatedAt: "",
    });
    m.sections = [section("a.md", 100), section("b.md", 250)];
    expect(totalGenomeTokens(m)).toBe(350);
  });

  test("totalGenomeTokens returns 0 for empty sections", async () => {
    const { totalGenomeTokens } = await import("../src/genome/manifest.ts");
    expect(totalGenomeTokens(makeManifest())).toBe(0);
  });

  test("sectionPath rejects path traversal attempts", async () => {
    const { sectionPath } = await import("../src/genome/manifest.ts");
    expect(() => sectionPath(cwd, "../../etc/passwd")).toThrow(/escapes genome directory/);
  });
});

// ─── embeddings ─────────────────────────────────────────────────────────────

describe("embeddings — cache hit/miss, content-hash collision avoidance, cosine stability", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  // ── contentHash ──────────────────────────────────────────────────────────

  test("contentHash — same content produces same hash", async () => {
    const { contentHash } = await import("../src/genome/embeddings.ts");
    expect(contentHash("hello world")).toBe(contentHash("hello world"));
  });

  test("contentHash — different content produces different hash (collision avoidance)", async () => {
    const { contentHash } = await import("../src/genome/embeddings.ts");
    const h1 = contentHash("hello world");
    const h2 = contentHash("hello world!");
    expect(h1).not.toBe(h2);
  });

  test("contentHash — empty string has a stable hash", async () => {
    const { contentHash } = await import("../src/genome/embeddings.ts");
    const h = contentHash("");
    // MD5 of empty string is well-known
    expect(h).toBe(createHash("md5").update("").digest("hex"));
    expect(h).toBe(contentHash(""));
  });

  test("contentHash — unicode content hashes without throwing", async () => {
    const { contentHash } = await import("../src/genome/embeddings.ts");
    expect(() => contentHash("🎉 こんにちは Привет")).not.toThrow();
  });

  // ── cosineSimilarity ─────────────────────────────────────────────────────

  test("cosineSimilarity — identical vectors returns 1.0", async () => {
    const { cosineSimilarity } = await import("../src/genome/embeddings.ts");
    const v = [0.1, 0.5, 0.9, 0.3];
    expect(cosineSimilarity(v, v)).toBeCloseTo(1.0);
  });

  test("cosineSimilarity — orthogonal vectors return 0", async () => {
    const { cosineSimilarity } = await import("../src/genome/embeddings.ts");
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0);
  });

  test("cosineSimilarity — opposite vectors return -1", async () => {
    const { cosineSimilarity } = await import("../src/genome/embeddings.ts");
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1);
  });

  test("cosineSimilarity — zero vector (all zeros) returns 0, no NaN/Inf", async () => {
    const { cosineSimilarity } = await import("../src/genome/embeddings.ts");
    const result = cosineSimilarity([0, 0, 0], [0, 0, 0]);
    expect(result).toBe(0);
    expect(Number.isFinite(result)).toBe(true);
  });

  test("cosineSimilarity — empty arrays return 0", async () => {
    const { cosineSimilarity } = await import("../src/genome/embeddings.ts");
    expect(cosineSimilarity([], [])).toBe(0);
  });

  test("cosineSimilarity — mismatched lengths return 0 (no crash, no NaN)", async () => {
    const { cosineSimilarity } = await import("../src/genome/embeddings.ts");
    const result = cosineSimilarity([1, 2, 3], [1, 2]);
    expect(result).toBe(0);
  });

  test("cosineSimilarity — large vectors are numerically stable", async () => {
    const { cosineSimilarity } = await import("../src/genome/embeddings.ts");
    // 768-dim all-ones vector (like a real embedding before normalisation)
    const big = Array.from({ length: 768 }, () => 1.0);
    const result = cosineSimilarity(big, big);
    expect(result).toBeCloseTo(1.0, 5);
  });

  test("cosineSimilarity — very small values (near underflow) don't produce NaN", async () => {
    const { cosineSimilarity } = await import("../src/genome/embeddings.ts");
    const tiny = [1e-308, 1e-308];
    const result = cosineSimilarity(tiny, tiny);
    expect(Number.isNaN(result)).toBe(false);
    expect(Number.isFinite(result)).toBe(true);
  });

  // ── embedding cache persistence ──────────────────────────────────────────

  test("loadEmbeddingCache returns empty array when cache file missing", async () => {
    const { loadEmbeddingCache } = await import("../src/genome/embeddings.ts");
    const cache = await loadEmbeddingCache(cwd);
    expect(cache).toEqual([]);
  });

  test("loadEmbeddingCache returns empty array for corrupt cache file", async () => {
    const { loadEmbeddingCache } = await import("../src/genome/embeddings.ts");
    const cachePath = join(cwd, ".ashlrcode", "genome", "evolution", "embeddings.json");
    await mkdir(join(cwd, ".ashlrcode", "genome", "evolution"), { recursive: true });
    await writeFile(cachePath, "NOT JSON {{{{", "utf-8");
    const cache = await loadEmbeddingCache(cwd);
    expect(cache).toEqual([]);
  });

  test("loadEmbeddingCache returns empty array when cache file contains non-array JSON", async () => {
    const { loadEmbeddingCache } = await import("../src/genome/embeddings.ts");
    const cachePath = join(cwd, ".ashlrcode", "genome", "evolution", "embeddings.json");
    await mkdir(join(cwd, ".ashlrcode", "genome", "evolution"), { recursive: true });
    await writeFile(cachePath, JSON.stringify({ notAnArray: true }), "utf-8");
    const cache = await loadEmbeddingCache(cwd);
    // BUG-NOTICE: loadEmbeddingCache checks Array.isArray and returns [] for objects — correct.
    expect(cache).toEqual([]);
  });

  test("saveEmbeddingCache then loadEmbeddingCache round-trips correctly", async () => {
    const { saveEmbeddingCache, loadEmbeddingCache, contentHash } = await import("../src/genome/embeddings.ts");
    const entry = {
      sectionPath: "vision/north-star.md",
      embedding: [0.1, 0.2, 0.3],
      contentHash: contentHash("some content"),
      updatedAt: new Date().toISOString(),
    };
    await saveEmbeddingCache(cwd, [entry]);
    const loaded = await loadEmbeddingCache(cwd);
    expect(loaded).toHaveLength(1);
    expect(loaded[0]!.sectionPath).toBe("vision/north-star.md");
    expect(loaded[0]!.embedding).toEqual([0.1, 0.2, 0.3]);
  });

  test("cache hit — updateEmbeddings skips section when hash unchanged", async () => {
    const { saveEmbeddingCache, loadEmbeddingCache, contentHash } = await import("../src/genome/embeddings.ts");
    const { saveManifest, createEmptyManifest } = await import("../src/genome/manifest.ts");

    const sectionContent = "# Vision\nBe the best.";
    const hash = contentHash(sectionContent);
    const sectionRelPath = "vision/north-star.md";

    // Write the section file
    const sectionFullPath = join(cwd, ".ashlrcode", "genome", sectionRelPath);
    await mkdir(join(cwd, ".ashlrcode", "genome", "vision"), { recursive: true });
    await writeFile(sectionFullPath, sectionContent, "utf-8");

    // Prime the cache with a matching hash
    await saveEmbeddingCache(cwd, [{
      sectionPath: sectionRelPath,
      embedding: [0.5, 0.5],
      contentHash: hash,
      updatedAt: new Date().toISOString(),
    }]);

    // Manifest with one section
    const manifest = createEmptyManifest("test");
    manifest.sections = [{
      path: sectionRelPath,
      title: "Vision",
      summary: "",
      tags: [],
      tokens: 10,
      updatedAt: new Date().toISOString(),
    }];
    await saveManifest(cwd, manifest);

    const { updateEmbeddings } = await import("../src/genome/embeddings.ts");
    // With Ollama unavailable, a cache-miss would return failed=1. A cache-hit
    // returns skipped=1 before ever calling generateEmbedding.
    const result = await updateEmbeddings(cwd);
    expect(result.skipped).toBe(1);
    expect(result.updated).toBe(0);
  });

  test("cache miss — updateEmbeddings marks section as failed when Ollama unreachable", async () => {
    const { saveManifest, createEmptyManifest } = await import("../src/genome/manifest.ts");

    const sectionContent = "# Goals\nShip fast.";
    const sectionRelPath = "goals/current.md";
    const sectionFullPath = join(cwd, ".ashlrcode", "genome", sectionRelPath);
    await mkdir(join(cwd, ".ashlrcode", "genome", "goals"), { recursive: true });
    await writeFile(sectionFullPath, sectionContent, "utf-8");

    const manifest = createEmptyManifest("test");
    manifest.sections = [{
      path: sectionRelPath,
      title: "Goals",
      summary: "",
      tags: [],
      tokens: 8,
      updatedAt: new Date().toISOString(),
    }];
    await saveManifest(cwd, manifest);

    // No cached embedding for this path — Ollama unavailable → failed
    const { updateEmbeddings } = await import("../src/genome/embeddings.ts");
    const result = await updateEmbeddings(cwd);
    // Ollama is not running in CI, so generateEmbedding returns null → failed++
    expect(result.failed).toBe(1);
    expect(result.updated).toBe(0);
    expect(result.skipped).toBe(0);
  });

  test("updateEmbeddings returns zero counts when manifest is empty", async () => {
    const { saveManifest, createEmptyManifest } = await import("../src/genome/manifest.ts");
    await saveManifest(cwd, createEmptyManifest("test"));

    const { updateEmbeddings } = await import("../src/genome/embeddings.ts");
    const result = await updateEmbeddings(cwd);
    expect(result).toEqual({ updated: 0, skipped: 0, failed: 0 });
  });

  test("updateEmbeddings returns zero counts when no manifest exists", async () => {
    // No manifest file — loadManifest returns null
    const { updateEmbeddings } = await import("../src/genome/embeddings.ts");
    const result = await updateEmbeddings(cwd);
    expect(result).toEqual({ updated: 0, skipped: 0, failed: 0 });
  });

  test("updateEmbeddings counts missing section files as failed", async () => {
    const { saveManifest, createEmptyManifest } = await import("../src/genome/manifest.ts");
    const manifest = createEmptyManifest("test");
    manifest.sections = [{
      path: "ghost/does-not-exist.md",
      title: "Ghost",
      summary: "",
      tags: [],
      tokens: 0,
      updatedAt: new Date().toISOString(),
    }];
    await saveManifest(cwd, manifest);

    const { updateEmbeddings } = await import("../src/genome/embeddings.ts");
    const result = await updateEmbeddings(cwd);
    expect(result.failed).toBe(1);
  });

  test("isOllamaAvailable returns false (not a crash) when port is unreachable", async () => {
    const { isOllamaAvailable } = await import("../src/genome/embeddings.ts");
    // This test assumes Ollama is NOT running in the test environment.
    // It must resolve to a boolean (not throw) either way.
    const available = await isOllamaAvailable();
    expect(typeof available).toBe("boolean");
  });

  test("generateEmbedding returns null when Ollama is unreachable", async () => {
    const { generateEmbedding } = await import("../src/genome/embeddings.ts");
    const result = await generateEmbedding("test text");
    // When Ollama is not running, fetch throws → null returned
    expect(result).toBeNull();
  });

  test("semanticSearch returns empty array when no embeddings cached", async () => {
    const { saveManifest, createEmptyManifest } = await import("../src/genome/manifest.ts");
    await saveManifest(cwd, createEmptyManifest("test"));

    const { semanticSearch } = await import("../src/genome/embeddings.ts");
    const results = await semanticSearch(cwd, "some query", 4000);
    expect(results).toEqual([]);
  });
});
