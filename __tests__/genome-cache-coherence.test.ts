/**
 * Embedding Cache Coherence tests
 *
 * Covers:
 *  - EmbeddingCacheMetadata type round-trip (save/load)
 *  - hashManifest: stability and change detection
 *  - IndexFreshnessValidator: FRESH / AGE_STALE / HASH_MISMATCH / SECTIONS_ADDED / SECTIONS_REMOVED / NO_METADATA
 *  - IncrementalEmbeddingUpdater: incremental add vs full rebuild, audit records
 *  - CacheCoherence.ensureFresh: auto-rebuild on stale, pass-through on fresh
 *  - CrossAgentCacheSync: record ops, read recent ops, getRecentlyEmbeddedPaths, isIndexFreshFromAgentActivity
 *  - validateIndexFreshness: exported API (no manifest / with manifest)
 *  - rebuildIfStale: force rebuild, fresh skip
 *  - getIndexStats: metadata present / absent
 *  - genome/index.ts: new symbols are exported
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, rm } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeTmpDir(): string {
  return join(tmpdir(), `ashlr-cache-coh-${Date.now()}-${Math.random().toString(36).slice(2)}`);
}

async function setupGenomeDir(cwd: string): Promise<void> {
  await mkdir(join(cwd, ".ashlrcode", "genome", "evolution"), { recursive: true });
}

function makeManifest(
  sections: Array<{ path: string; title?: string; summary?: string; tags?: string[]; tokens?: number; updatedAt?: string }> = [],
) {
  return {
    version: 1 as const,
    schemaVersion: 3,
    project: "test-project",
    sections: sections.map((s) => ({
      path: s.path,
      title: s.title ?? "Test Section",
      summary: s.summary ?? "A test section",
      tags: s.tags ?? ["test"],
      tokens: s.tokens ?? 100,
      updatedAt: s.updatedAt ?? "2026-01-01T00:00:00.000Z",
    })),
    generation: { number: 1, milestone: "test", startedAt: "2026-01-01T00:00:00.000Z" },
    fitnessHistory: [],
    createdAt: "2026-01-01T00:00:00.000Z",
    updatedAt: "2026-01-01T00:00:00.000Z",
  };
}

function makeMetadata(
  overrides: Partial<import("../src/genome/embedding-cache-coherence.ts").EmbeddingCacheMetadata> = {},
): import("../src/genome/embedding-cache-coherence.ts").EmbeddingCacheMetadata {
  return {
    builtAt: new Date(Date.now() - 1000).toISOString(), // 1 second ago by default
    manifestHash: "abc123",
    sectionCount: 2,
    quantDepth: 8,
    modelTier: "balanced",
    indexedSectionPaths: ["arch/overview.md", "vision/north-star.md"],
    ...overrides,
  };
}

/** Minimal ANNSearcher mock */
function makeSearcherMock() {
  const addedEntries: Array<{ id: string; embedding: number[] }> = [];
  const builtEntries: Array<{ id: string; embedding: number[] }> = [];
  return {
    addEntry(id: string, embedding: number[]) {
      addedEntries.push({ id, embedding });
    },
    buildIndex(entries: Array<{ id: string; embedding: number[] }>) {
      builtEntries.splice(0, builtEntries.length, ...entries);
    },
    addedEntries,
    builtEntries,
  };
}

const DUMMY_EMBED_FN = async (_path: string, _content: string) => [0.1, 0.2, 0.3];
const DUMMY_READ_SECTION = async (_cwd: string, _path: string) => "section content";
const DUMMY_EMBED_ALL = async () => [
  { id: "arch/overview.md", embedding: [0.1, 0.2, 0.3] },
  { id: "vision/north-star.md", embedding: [0.4, 0.5, 0.6] },
];

// ---------------------------------------------------------------------------
// EmbeddingCacheMetadata — round-trip persistence
// ---------------------------------------------------------------------------

describe("EmbeddingCacheMetadata — round-trip (save/load)", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("saveCacheMetadata + loadCacheMetadata round-trip", async () => {
    const { saveCacheMetadata, loadCacheMetadata } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const meta = makeMetadata();
    await saveCacheMetadata(cwd, meta);
    const loaded = await loadCacheMetadata(cwd);
    expect(loaded).not.toBeNull();
    expect(loaded!.manifestHash).toBe(meta.manifestHash);
    expect(loaded!.sectionCount).toBe(meta.sectionCount);
    expect(loaded!.quantDepth).toBe(8);
    expect(loaded!.modelTier).toBe("balanced");
  });

  test("loadCacheMetadata returns null when no file exists", async () => {
    const { loadCacheMetadata } = await import("../src/genome/embedding-cache-coherence.ts");
    const result = await loadCacheMetadata(cwd);
    expect(result).toBeNull();
  });

  test("indexedSectionPaths round-trips correctly", async () => {
    const { saveCacheMetadata, loadCacheMetadata } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const paths = ["arch/overview.md", "vision/north-star.md", "api/contracts.md"];
    await saveCacheMetadata(cwd, makeMetadata({ indexedSectionPaths: paths }));
    const loaded = await loadCacheMetadata(cwd);
    expect(loaded!.indexedSectionPaths).toEqual(paths);
  });

  test("createCacheMetadata derives hash and count from manifest", async () => {
    const { createCacheMetadata, hashManifest } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const manifest = makeManifest([{ path: "arch/overview.md" }, { path: "vision/north-star.md" }]);
    const meta = createCacheMetadata(manifest, 16, "accurate");
    expect(meta.sectionCount).toBe(2);
    expect(meta.quantDepth).toBe(16);
    expect(meta.modelTier).toBe("accurate");
    expect(meta.manifestHash).toBe(hashManifest(manifest));
    expect(meta.indexedSectionPaths).toEqual(["arch/overview.md", "vision/north-star.md"]);
    expect(typeof meta.builtAt).toBe("string");
  });
});

// ---------------------------------------------------------------------------
// hashManifest
// ---------------------------------------------------------------------------

describe("hashManifest — stability and change detection", () => {
  test("same manifest produces same hash", async () => {
    const { hashManifest } = await import("../src/genome/embedding-cache-coherence.ts");
    const m = makeManifest([{ path: "arch/overview.md" }]);
    expect(hashManifest(m)).toBe(hashManifest(m));
  });

  test("adding a section changes the hash", async () => {
    const { hashManifest } = await import("../src/genome/embedding-cache-coherence.ts");
    const m1 = makeManifest([{ path: "arch/overview.md" }]);
    const m2 = makeManifest([{ path: "arch/overview.md" }, { path: "api/contracts.md" }]);
    expect(hashManifest(m1)).not.toBe(hashManifest(m2));
  });

  test("changing section summary changes the hash", async () => {
    const { hashManifest } = await import("../src/genome/embedding-cache-coherence.ts");
    const m1 = makeManifest([{ path: "arch/overview.md", summary: "original" }]);
    const m2 = makeManifest([{ path: "arch/overview.md", summary: "updated" }]);
    expect(hashManifest(m1)).not.toBe(hashManifest(m2));
  });

  test("changing updatedAt of top-level manifest does NOT change hash (stabilisation)", async () => {
    const { hashManifest } = await import("../src/genome/embedding-cache-coherence.ts");
    const m1 = makeManifest([{ path: "arch/overview.md" }]);
    const m2 = { ...m1, updatedAt: "2099-12-31T23:59:59.000Z" };
    // Top-level updatedAt should not be included in hash
    expect(hashManifest(m1)).toBe(hashManifest(m2));
  });

  test("section order does not affect hash (canonical sort)", async () => {
    const { hashManifest } = await import("../src/genome/embedding-cache-coherence.ts");
    const m1 = makeManifest([{ path: "aaa.md" }, { path: "zzz.md" }]);
    const m2 = makeManifest([{ path: "zzz.md" }, { path: "aaa.md" }]);
    expect(hashManifest(m1)).toBe(hashManifest(m2));
  });

  test("hash is a 64-char hex string (SHA-256)", async () => {
    const { hashManifest } = await import("../src/genome/embedding-cache-coherence.ts");
    const hash = hashManifest(makeManifest([{ path: "arch/overview.md" }]));
    expect(hash).toMatch(/^[0-9a-f]{64}$/);
  });
});

// ---------------------------------------------------------------------------
// IndexFreshnessValidator
// ---------------------------------------------------------------------------

describe("IndexFreshnessValidator — staleness detection", () => {
  test("NO_METADATA when metadata is null", async () => {
    const { IndexFreshnessValidator } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const validator = new IndexFreshnessValidator();
    const result = validator.validate(makeManifest([]), null);
    expect(result.reason).toBe("NO_METADATA");
    expect(result.isFresh).toBe(false);
  });

  test("FRESH when hash matches, no age exceeded, no section changes", async () => {
    const { IndexFreshnessValidator, hashManifest } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const manifest = makeManifest([{ path: "arch/overview.md" }]);
    const hash = hashManifest(manifest);
    const meta = makeMetadata({
      manifestHash: hash,
      sectionCount: 1,
      indexedSectionPaths: ["arch/overview.md"],
      builtAt: new Date(Date.now() - 100).toISOString(), // 100ms ago
    });
    const validator = new IndexFreshnessValidator(60 * 60 * 1000);
    const result = validator.validate(manifest, meta);
    expect(result.reason).toBe("FRESH");
    expect(result.isFresh).toBe(true);
    expect(result.newSections).toHaveLength(0);
    expect(result.removedSectionPaths).toHaveLength(0);
  });

  test("AGE_STALE when index is older than maxAgeMs", async () => {
    const { IndexFreshnessValidator, hashManifest } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const manifest = makeManifest([{ path: "arch/overview.md" }]);
    const hash = hashManifest(manifest);
    const meta = makeMetadata({
      manifestHash: hash,
      sectionCount: 1,
      indexedSectionPaths: ["arch/overview.md"],
      builtAt: new Date(Date.now() - 7200 * 1000).toISOString(), // 2 hours ago
    });
    const validator = new IndexFreshnessValidator(3600 * 1000); // 1 hour max
    const result = validator.validate(manifest, meta);
    expect(result.reason).toBe("AGE_STALE");
    expect(result.isFresh).toBe(false);
    expect(result.ageMs).toBeGreaterThan(3600 * 1000);
  });

  test("HASH_MISMATCH when manifest content changed", async () => {
    const { IndexFreshnessValidator } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const manifest = makeManifest([{ path: "arch/overview.md", summary: "new summary" }]);
    const meta = makeMetadata({
      manifestHash: "stale_hash_value",
      sectionCount: 1,
      indexedSectionPaths: ["arch/overview.md"],
      builtAt: new Date(Date.now() - 100).toISOString(),
    });
    const validator = new IndexFreshnessValidator(3600 * 1000);
    const result = validator.validate(manifest, meta);
    expect(result.reason).toBe("HASH_MISMATCH");
    expect(result.isFresh).toBe(false);
  });

  test("SECTIONS_ADDED when manifest has new sections", async () => {
    const { IndexFreshnessValidator, hashManifest } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const oldManifest = makeManifest([{ path: "arch/overview.md" }]);
    const newManifest = makeManifest([
      { path: "arch/overview.md" },
      { path: "api/contracts.md" },
    ]);
    const meta = makeMetadata({
      manifestHash: hashManifest(oldManifest),
      sectionCount: 1,
      indexedSectionPaths: ["arch/overview.md"],
      builtAt: new Date(Date.now() - 100).toISOString(),
    });
    const validator = new IndexFreshnessValidator(3600 * 1000);
    const result = validator.validate(newManifest, meta);
    expect(result.reason).toBe("SECTIONS_ADDED");
    expect(result.isFresh).toBe(false);
    expect(result.newSections).toHaveLength(1);
    expect(result.newSections[0]!.path).toBe("api/contracts.md");
  });

  test("SECTIONS_REMOVED when manifest lost sections", async () => {
    const { IndexFreshnessValidator, hashManifest } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const newManifest = makeManifest([{ path: "arch/overview.md" }]);
    const meta = makeMetadata({
      manifestHash: hashManifest(newManifest),
      sectionCount: 2,
      indexedSectionPaths: ["arch/overview.md", "deleted-section.md"],
      builtAt: new Date(Date.now() - 100).toISOString(),
    });
    const validator = new IndexFreshnessValidator(3600 * 1000);
    const result = validator.validate(newManifest, meta);
    expect(result.reason).toBe("SECTIONS_REMOVED");
    expect(result.isFresh).toBe(false);
    expect(result.removedSectionPaths).toContain("deleted-section.md");
  });

  test("SECTIONS_REMOVED takes priority over AGE_STALE", async () => {
    const { IndexFreshnessValidator } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const manifest = makeManifest([{ path: "arch/overview.md" }]);
    const meta = makeMetadata({
      manifestHash: "any",
      sectionCount: 2,
      indexedSectionPaths: ["arch/overview.md", "deleted.md"],
      builtAt: new Date(Date.now() - 7200 * 1000).toISOString(), // also age-stale
    });
    const validator = new IndexFreshnessValidator(3600 * 1000);
    const result = validator.validate(manifest, meta);
    expect(result.reason).toBe("SECTIONS_REMOVED"); // removals take priority
  });

  test("ageMs is accurate", async () => {
    const { IndexFreshnessValidator, hashManifest } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const manifest = makeManifest([{ path: "arch/overview.md" }]);
    const builtAt = Date.now() - 5000; // 5 seconds ago
    const meta = makeMetadata({
      manifestHash: hashManifest(manifest),
      sectionCount: 1,
      indexedSectionPaths: ["arch/overview.md"],
      builtAt: new Date(builtAt).toISOString(),
    });
    const validator = new IndexFreshnessValidator(3600 * 1000);
    const nowMs = Date.now();
    const result = validator.validate(manifest, meta, nowMs);
    expect(result.ageMs).toBeGreaterThanOrEqual(4900);
    expect(result.ageMs).toBeLessThan(10000);
  });

  test("nowMs parameter controls age calculation", async () => {
    const { IndexFreshnessValidator, hashManifest } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const manifest = makeManifest([{ path: "arch/overview.md" }]);
    const builtAt = 1000000; // epoch + 1000s
    const meta = makeMetadata({
      manifestHash: hashManifest(manifest),
      sectionCount: 1,
      indexedSectionPaths: ["arch/overview.md"],
      builtAt: new Date(builtAt).toISOString(),
    });
    const validator = new IndexFreshnessValidator(3600 * 1000);
    // Inject nowMs = builtAt + 2 hours → should be stale
    const result = validator.validate(manifest, meta, builtAt + 2 * 3600 * 1000);
    expect(result.reason).toBe("AGE_STALE");
    expect(result.ageMs).toBeCloseTo(2 * 3600 * 1000, -2);
  });
});

// ---------------------------------------------------------------------------
// IncrementalEmbeddingUpdater
// ---------------------------------------------------------------------------

describe("IncrementalEmbeddingUpdater — incremental add vs full rebuild", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("adds new sections incrementally when count <= threshold", async () => {
    const { IncrementalEmbeddingUpdater, IndexFreshnessValidator, hashManifest } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const oldManifest = makeManifest([{ path: "arch/overview.md" }]);
    const newManifest = makeManifest([
      { path: "arch/overview.md" },
      { path: "api/contracts.md" },
      { path: "vision/north-star.md" },
    ]);

    const meta = {
      builtAt: new Date(Date.now() - 100).toISOString(),
      manifestHash: hashManifest(oldManifest),
      sectionCount: 1,
      quantDepth: 8 as const,
      modelTier: "balanced" as const,
      indexedSectionPaths: ["arch/overview.md"],
    };

    const validator = new IndexFreshnessValidator(3600 * 1000);
    const freshness = validator.validate(newManifest, meta);
    expect(freshness.reason).toBe("SECTIONS_ADDED");

    const searcher = makeSearcherMock();
    const updater = new IncrementalEmbeddingUpdater(5);

    const audit = await updater.applyUpdates(
      cwd,
      searcher,
      freshness,
      newManifest,
      DUMMY_EMBED_FN,
      DUMMY_READ_SECTION,
      DUMMY_EMBED_ALL,
      8,
      "balanced",
    );

    expect(audit.type).toBe("incremental_add");
    expect(audit.sectionsAdded).toBe(2);
    expect(searcher.addedEntries).toHaveLength(2);
    expect(searcher.builtEntries).toHaveLength(0); // no full rebuild
  });

  test("triggers full rebuild when new sections exceed threshold", async () => {
    const { IncrementalEmbeddingUpdater, IndexFreshnessValidator, hashManifest } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const oldManifest = makeManifest([{ path: "arch/overview.md" }]);
    const newSections = Array.from({ length: 6 }, (_, i) => ({ path: `new/section-${i}.md` }));
    const newManifest = makeManifest([{ path: "arch/overview.md" }, ...newSections]);

    const meta = {
      builtAt: new Date(Date.now() - 100).toISOString(),
      manifestHash: hashManifest(oldManifest),
      sectionCount: 1,
      quantDepth: 8 as const,
      modelTier: "balanced" as const,
      indexedSectionPaths: ["arch/overview.md"],
    };

    const validator = new IndexFreshnessValidator(3600 * 1000);
    const freshness = validator.validate(newManifest, meta);

    const searcher = makeSearcherMock();
    const updater = new IncrementalEmbeddingUpdater(5); // threshold = 5, 6 new → rebuild

    const audit = await updater.applyUpdates(
      cwd,
      searcher,
      freshness,
      newManifest,
      DUMMY_EMBED_FN,
      DUMMY_READ_SECTION,
      DUMMY_EMBED_ALL,
      8,
      "balanced",
    );

    expect(audit.type).toBe("full_rebuild");
    expect(searcher.builtEntries.length).toBeGreaterThan(0);
  });

  test("triggers full rebuild when sections removed", async () => {
    const { IncrementalEmbeddingUpdater, IndexFreshnessValidator } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const newManifest = makeManifest([{ path: "arch/overview.md" }]);
    const meta = {
      builtAt: new Date(Date.now() - 100).toISOString(),
      manifestHash: "any",
      sectionCount: 2,
      quantDepth: 8 as const,
      modelTier: "balanced" as const,
      indexedSectionPaths: ["arch/overview.md", "deleted.md"],
    };

    const validator = new IndexFreshnessValidator(3600 * 1000);
    const freshness = validator.validate(newManifest, meta);
    expect(freshness.reason).toBe("SECTIONS_REMOVED");

    const searcher = makeSearcherMock();
    const updater = new IncrementalEmbeddingUpdater(5);

    const audit = await updater.applyUpdates(
      cwd,
      searcher,
      freshness,
      newManifest,
      DUMMY_EMBED_FN,
      DUMMY_READ_SECTION,
      DUMMY_EMBED_ALL,
      8,
      "balanced",
    );

    expect(audit.type).toBe("full_rebuild");
    expect(audit.sectionsRemoved).toBe(1);
  });

  test("audit record is persisted to JSONL", async () => {
    const { IncrementalEmbeddingUpdater, IndexFreshnessValidator, hashManifest } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const { readJsonl } = await import("../src/genome/jsonl.ts");

    const manifest = makeManifest([{ path: "arch/overview.md" }, { path: "api/contracts.md" }]);
    const meta = {
      builtAt: new Date(Date.now() - 100).toISOString(),
      manifestHash: hashManifest(makeManifest([{ path: "arch/overview.md" }])),
      sectionCount: 1,
      quantDepth: 8 as const,
      modelTier: "balanced" as const,
      indexedSectionPaths: ["arch/overview.md"],
    };

    const validator = new IndexFreshnessValidator(3600 * 1000);
    const freshness = validator.validate(manifest, meta);
    const searcher = makeSearcherMock();
    const updater = new IncrementalEmbeddingUpdater(5);

    await updater.applyUpdates(
      cwd,
      searcher,
      freshness,
      manifest,
      DUMMY_EMBED_FN,
      DUMMY_READ_SECTION,
      DUMMY_EMBED_ALL,
      8,
      "balanced",
    );

    const auditPath = join(cwd, ".ashlrcode", "genome", "evolution", "embedding-incremental-audit.jsonl");
    const records = await readJsonl<{ type: string; id: string }>(auditPath);
    expect(records).toHaveLength(1);
    expect(records[0]!.type).toBe("incremental_add");
    expect(records[0]!.id).toMatch(/^icu-/);
  });

  test("audit record has correct elapsedMs", async () => {
    const { IncrementalEmbeddingUpdater, IndexFreshnessValidator, hashManifest } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const manifest = makeManifest([{ path: "arch/overview.md" }, { path: "api/contracts.md" }]);
    const meta = {
      builtAt: new Date(Date.now() - 100).toISOString(),
      manifestHash: hashManifest(makeManifest([{ path: "arch/overview.md" }])),
      sectionCount: 1,
      quantDepth: 8 as const,
      modelTier: "balanced" as const,
      indexedSectionPaths: ["arch/overview.md"],
    };

    const validator = new IndexFreshnessValidator(3600 * 1000);
    const freshness = validator.validate(manifest, meta);
    const searcher = makeSearcherMock();
    const updater = new IncrementalEmbeddingUpdater(5);

    const audit = await updater.applyUpdates(
      cwd, searcher, freshness, manifest,
      DUMMY_EMBED_FN, DUMMY_READ_SECTION, DUMMY_EMBED_ALL, 8, "balanced",
    );

    expect(audit.elapsedMs).toBeGreaterThanOrEqual(0);
    expect(audit.elapsedMs).toBeLessThan(5000);
  });
});

// ---------------------------------------------------------------------------
// CacheCoherence.ensureFresh
// ---------------------------------------------------------------------------

describe("CacheCoherence.ensureFresh — auto-rebuild", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("passes through when index is fresh", async () => {
    const { CacheCoherence, saveCacheMetadata, hashManifest } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const manifest = makeManifest([{ path: "arch/overview.md" }]);
    await saveCacheMetadata(cwd, {
      builtAt: new Date(Date.now() - 100).toISOString(),
      manifestHash: hashManifest(manifest),
      sectionCount: 1,
      quantDepth: 8,
      modelTier: "balanced",
      indexedSectionPaths: ["arch/overview.md"],
    });

    const searcher = makeSearcherMock();
    const coherence = new CacheCoherence({ maxAgeMs: 3600 * 1000 });

    const result = await coherence.ensureFresh(
      cwd, manifest, searcher, DUMMY_EMBED_FN, DUMMY_READ_SECTION, DUMMY_EMBED_ALL,
    );

    expect(result.freshness.reason).toBe("FRESH");
    expect(result.updated).toBe(false);
    expect(result.auditRecord).toBeUndefined();
    expect(searcher.addedEntries).toHaveLength(0);
    expect(searcher.builtEntries).toHaveLength(0);
  });

  test("rebuilds when index is stale by age", async () => {
    const { CacheCoherence, saveCacheMetadata, hashManifest } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const manifest = makeManifest([{ path: "arch/overview.md" }]);
    await saveCacheMetadata(cwd, {
      builtAt: new Date(Date.now() - 7200 * 1000).toISOString(), // 2 hours ago
      manifestHash: hashManifest(manifest),
      sectionCount: 1,
      quantDepth: 8,
      modelTier: "balanced",
      indexedSectionPaths: ["arch/overview.md"],
    });

    const searcher = makeSearcherMock();
    const coherence = new CacheCoherence({ maxAgeMs: 3600 * 1000 });

    const result = await coherence.ensureFresh(
      cwd, manifest, searcher, DUMMY_EMBED_FN, DUMMY_READ_SECTION, DUMMY_EMBED_ALL,
    );

    expect(result.freshness.reason).toBe("AGE_STALE");
    expect(result.updated).toBe(true);
    expect(result.auditRecord).toBeDefined();
  });

  test("force=true rebuilds even when index is fresh", async () => {
    const { CacheCoherence, saveCacheMetadata, hashManifest } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const manifest = makeManifest([{ path: "arch/overview.md" }]);
    await saveCacheMetadata(cwd, {
      builtAt: new Date(Date.now() - 100).toISOString(),
      manifestHash: hashManifest(manifest),
      sectionCount: 1,
      quantDepth: 8,
      modelTier: "balanced",
      indexedSectionPaths: ["arch/overview.md"],
    });

    const searcher = makeSearcherMock();
    const coherence = new CacheCoherence({ maxAgeMs: 3600 * 1000, force: true });

    const result = await coherence.ensureFresh(
      cwd, manifest, searcher, DUMMY_EMBED_FN, DUMMY_READ_SECTION, DUMMY_EMBED_ALL,
    );

    expect(result.updated).toBe(true);
  });

  test("incremental add on SECTIONS_ADDED within threshold", async () => {
    const { CacheCoherence, saveCacheMetadata, hashManifest } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const oldManifest = makeManifest([{ path: "arch/overview.md" }]);
    const newManifest = makeManifest([
      { path: "arch/overview.md" },
      { path: "api/contracts.md" },
    ]);
    await saveCacheMetadata(cwd, {
      builtAt: new Date(Date.now() - 100).toISOString(),
      manifestHash: hashManifest(oldManifest),
      sectionCount: 1,
      quantDepth: 8,
      modelTier: "balanced",
      indexedSectionPaths: ["arch/overview.md"],
    });

    const searcher = makeSearcherMock();
    const coherence = new CacheCoherence({ maxAgeMs: 3600 * 1000, incrementalThreshold: 5 });

    const result = await coherence.ensureFresh(
      cwd, newManifest, searcher, DUMMY_EMBED_FN, DUMMY_READ_SECTION, DUMMY_EMBED_ALL,
    );

    expect(result.updated).toBe(true);
    expect(result.auditRecord?.type).toBe("incremental_add");
    expect(searcher.addedEntries).toHaveLength(1);
    expect(searcher.addedEntries[0]!.id).toBe("api/contracts.md");
  });
});

// ---------------------------------------------------------------------------
// CrossAgentCacheSync
// ---------------------------------------------------------------------------

describe("CrossAgentCacheSync — cross-agent coordination", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("recordEmbeddingOp returns a string id starting with aeo-", async () => {
    const { CrossAgentCacheSync } = await import("../src/genome/embedding-cache-coherence.ts");
    const sync = new CrossAgentCacheSync("agent-A");
    const id = await sync.recordEmbeddingOp(cwd, ["arch/overview.md"], "balanced", 8);
    expect(id).toMatch(/^aeo-/);
  });

  test("readRecentOps returns records within time window", async () => {
    const { CrossAgentCacheSync } = await import("../src/genome/embedding-cache-coherence.ts");
    const sync = new CrossAgentCacheSync("agent-A");
    const now = Date.now();

    await sync.recordEmbeddingOp(cwd, ["arch/overview.md"], "balanced", 8);
    const ops = await sync.readRecentOps(cwd, 10 * 60 * 1000, now + 5000);
    expect(ops).toHaveLength(1);
    expect(ops[0]!.agentId).toBe("agent-A");
    expect(ops[0]!.sectionPaths).toContain("arch/overview.md");
  });

  test("readRecentOps excludes stale records beyond maxAgeMs", async () => {
    const { CrossAgentCacheSync } = await import("../src/genome/embedding-cache-coherence.ts");
    const sync = new CrossAgentCacheSync("agent-A");

    await sync.recordEmbeddingOp(cwd, ["arch/overview.md"], "balanced", 8);
    // Read with nowMs far in the future so record is beyond 1-minute window
    const ops = await sync.readRecentOps(cwd, 60 * 1000, Date.now() + 2 * 60 * 1000);
    expect(ops).toHaveLength(0);
  });

  test("getRecentlyEmbeddedPaths returns paths embedded by any agent", async () => {
    const { CrossAgentCacheSync } = await import("../src/genome/embedding-cache-coherence.ts");
    const agentA = new CrossAgentCacheSync("agent-A");
    const agentB = new CrossAgentCacheSync("agent-B");
    const now = Date.now();

    await agentA.recordEmbeddingOp(cwd, ["arch/overview.md", "api/contracts.md"], "balanced", 8);

    const recentlyEmbedded = await agentB.getRecentlyEmbeddedPaths(
      cwd,
      ["arch/overview.md", "api/contracts.md", "vision/north-star.md"],
      10 * 60 * 1000,
      now + 5000,
    );

    expect(recentlyEmbedded).toContain("arch/overview.md");
    expect(recentlyEmbedded).toContain("api/contracts.md");
    expect(recentlyEmbedded).not.toContain("vision/north-star.md");
  });

  test("isIndexFreshFromAgentActivity returns true when all sections recently embedded", async () => {
    const { CrossAgentCacheSync } = await import("../src/genome/embedding-cache-coherence.ts");
    const agentA = new CrossAgentCacheSync("agent-A");
    const agentB = new CrossAgentCacheSync("agent-B");
    const now = Date.now();

    await agentA.recordEmbeddingOp(
      cwd,
      ["arch/overview.md", "api/contracts.md"],
      "balanced",
      8,
    );

    const isFresh = await agentB.isIndexFreshFromAgentActivity(
      cwd,
      ["arch/overview.md", "api/contracts.md"],
      10 * 60 * 1000,
      now + 5000,
    );

    expect(isFresh).toBe(true);
  });

  test("isIndexFreshFromAgentActivity returns false when some sections not embedded", async () => {
    const { CrossAgentCacheSync } = await import("../src/genome/embedding-cache-coherence.ts");
    const agentA = new CrossAgentCacheSync("agent-A");
    const agentB = new CrossAgentCacheSync("agent-B");
    const now = Date.now();

    await agentA.recordEmbeddingOp(cwd, ["arch/overview.md"], "balanced", 8);

    const isFresh = await agentB.isIndexFreshFromAgentActivity(
      cwd,
      ["arch/overview.md", "missing.md"],
      10 * 60 * 1000,
      now + 5000,
    );

    expect(isFresh).toBe(false);
  });

  test("isIndexFreshFromAgentActivity returns true for empty section list", async () => {
    const { CrossAgentCacheSync } = await import("../src/genome/embedding-cache-coherence.ts");
    const sync = new CrossAgentCacheSync("agent-A");
    const isFresh = await sync.isIndexFreshFromAgentActivity(cwd, []);
    expect(isFresh).toBe(true);
  });

  test("agentId is returned by .id getter", async () => {
    const { CrossAgentCacheSync } = await import("../src/genome/embedding-cache-coherence.ts");
    const sync = new CrossAgentCacheSync("my-agent");
    expect(sync.id).toBe("my-agent");
  });

  test("multiple agents write independently to the session log", async () => {
    const { CrossAgentCacheSync } = await import("../src/genome/embedding-cache-coherence.ts");
    const agentA = new CrossAgentCacheSync("agent-A");
    const agentB = new CrossAgentCacheSync("agent-B");
    const now = Date.now();

    await agentA.recordEmbeddingOp(cwd, ["arch/overview.md"], "fast", 8);
    await agentB.recordEmbeddingOp(cwd, ["api/contracts.md"], "accurate", 16);

    const ops = await agentA.readRecentOps(cwd, 10 * 60 * 1000, now + 5000);
    expect(ops).toHaveLength(2);
    const agentIds = ops.map((o) => o.agentId);
    expect(agentIds).toContain("agent-A");
    expect(agentIds).toContain("agent-B");
  });
});

// ---------------------------------------------------------------------------
// validateIndexFreshness — exported API
// ---------------------------------------------------------------------------

describe("validateIndexFreshness — exported convenience function", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("returns NO_METADATA when no metadata file exists", async () => {
    const { validateIndexFreshness } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const result = await validateIndexFreshness(cwd);
    expect(result.reason).toBe("NO_METADATA");
    expect(result.isFresh).toBe(false);
  });

  test("returns FRESH when metadata is present and within maxAgeMs", async () => {
    const { validateIndexFreshness, saveCacheMetadata } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    await saveCacheMetadata(cwd, makeMetadata({ builtAt: new Date(Date.now() - 100).toISOString() }));
    const result = await validateIndexFreshness(cwd, 3600 * 1000);
    expect(result.reason).toBe("FRESH");
    expect(result.isFresh).toBe(true);
  });

  test("returns AGE_STALE when metadata is older than maxAgeMs", async () => {
    const { validateIndexFreshness, saveCacheMetadata } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    await saveCacheMetadata(
      cwd,
      makeMetadata({ builtAt: new Date(Date.now() - 7200 * 1000).toISOString() }),
    );
    const result = await validateIndexFreshness(cwd, 3600 * 1000);
    expect(result.reason).toBe("AGE_STALE");
    expect(result.isFresh).toBe(false);
  });

  test("validates against manifest when provided", async () => {
    const { validateIndexFreshness, saveCacheMetadata, hashManifest } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const manifest = makeManifest([{ path: "arch/overview.md" }]);
    await saveCacheMetadata(cwd, {
      builtAt: new Date(Date.now() - 100).toISOString(),
      manifestHash: hashManifest(manifest),
      sectionCount: 1,
      quantDepth: 8,
      modelTier: "balanced",
      indexedSectionPaths: ["arch/overview.md"],
    });
    const result = await validateIndexFreshness(cwd, 3600 * 1000, manifest);
    expect(result.reason).toBe("FRESH");
    expect(result.isFresh).toBe(true);
  });

  test("detects HASH_MISMATCH via manifest validation", async () => {
    const { validateIndexFreshness, saveCacheMetadata } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const manifest = makeManifest([{ path: "arch/overview.md", summary: "changed" }]);
    await saveCacheMetadata(cwd, {
      builtAt: new Date(Date.now() - 100).toISOString(),
      manifestHash: "outdated_hash",
      sectionCount: 1,
      quantDepth: 8,
      modelTier: "balanced",
      indexedSectionPaths: ["arch/overview.md"],
    });
    const result = await validateIndexFreshness(cwd, 3600 * 1000, manifest);
    expect(result.reason).toBe("HASH_MISMATCH");
  });
});

// ---------------------------------------------------------------------------
// rebuildIfStale
// ---------------------------------------------------------------------------

describe("rebuildIfStale — exported convenience function", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("returns rebuilt=false when no metadata and no manifest provided", async () => {
    const { rebuildIfStale } = await import("../src/genome/embedding-cache-coherence.ts");
    const result = await rebuildIfStale(cwd);
    expect(result.rebuilt).toBe(false);
    expect(result.reason).toBe("NO_METADATA");
  });

  test("force=true triggers rebuild with all dependencies", async () => {
    const { rebuildIfStale, saveCacheMetadata, hashManifest } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const manifest = makeManifest([{ path: "arch/overview.md" }]);
    await saveCacheMetadata(cwd, {
      builtAt: new Date(Date.now() - 100).toISOString(),
      manifestHash: hashManifest(manifest),
      sectionCount: 1,
      quantDepth: 8,
      modelTier: "balanced",
      indexedSectionPaths: ["arch/overview.md"],
    });

    const searcher = makeSearcherMock();
    const result = await rebuildIfStale(
      cwd,
      true, // force
      manifest,
      searcher,
      DUMMY_EMBED_FN,
      DUMMY_READ_SECTION,
      DUMMY_EMBED_ALL,
    );

    expect(result.rebuilt).toBe(true);
  });

  test("skips rebuild when fresh (no force)", async () => {
    const { rebuildIfStale, saveCacheMetadata, hashManifest } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const manifest = makeManifest([{ path: "arch/overview.md" }]);
    await saveCacheMetadata(cwd, {
      builtAt: new Date(Date.now() - 100).toISOString(),
      manifestHash: hashManifest(manifest),
      sectionCount: 1,
      quantDepth: 8,
      modelTier: "balanced",
      indexedSectionPaths: ["arch/overview.md"],
    });

    const searcher = makeSearcherMock();
    const result = await rebuildIfStale(
      cwd,
      false, // no force
      manifest,
      searcher,
      DUMMY_EMBED_FN,
      DUMMY_READ_SECTION,
      DUMMY_EMBED_ALL,
    );

    expect(result.rebuilt).toBe(false);
    expect(result.reason).toBe("FRESH");
  });
});

// ---------------------------------------------------------------------------
// getIndexStats
// ---------------------------------------------------------------------------

describe("getIndexStats — diagnostic statistics", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("returns hasMetadata=false when no file exists", async () => {
    const { getIndexStats } = await import("../src/genome/embedding-cache-coherence.ts");
    const stats = await getIndexStats(cwd);
    expect(stats.hasMetadata).toBe(false);
    expect(stats.builtAt).toBeNull();
    expect(stats.sectionCount).toBe(0);
    expect(stats.quantDepth).toBeNull();
    expect(stats.modelTier).toBeNull();
    expect(stats.ageMs).toBeNull();
    expect(stats.indexedSectionPaths).toHaveLength(0);
  });

  test("returns correct stats when metadata exists", async () => {
    const { getIndexStats, saveCacheMetadata } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );
    const meta = makeMetadata({
      builtAt: new Date(Date.now() - 5000).toISOString(),
      sectionCount: 3,
      quantDepth: 16,
      modelTier: "accurate",
      indexedSectionPaths: ["a.md", "b.md", "c.md"],
    });
    await saveCacheMetadata(cwd, meta);

    const stats = await getIndexStats(cwd);
    expect(stats.hasMetadata).toBe(true);
    expect(stats.sectionCount).toBe(3);
    expect(stats.quantDepth).toBe(16);
    expect(stats.modelTier).toBe("accurate");
    expect(stats.indexedSectionPaths).toHaveLength(3);
    expect(stats.ageMs).toBeGreaterThan(4000);
  });

  test("auditRecordCount reflects number of incremental update records", async () => {
    const { getIndexStats, IncrementalEmbeddingUpdater, IndexFreshnessValidator, hashManifest, saveCacheMetadata } = await import(
      "../src/genome/embedding-cache-coherence.ts"
    );

    const manifest = makeManifest([{ path: "arch/overview.md" }, { path: "api/contracts.md" }]);
    const oldMeta = {
      builtAt: new Date(Date.now() - 100).toISOString(),
      manifestHash: hashManifest(makeManifest([{ path: "arch/overview.md" }])),
      sectionCount: 1,
      quantDepth: 8 as const,
      modelTier: "balanced" as const,
      indexedSectionPaths: ["arch/overview.md"],
    };
    await saveCacheMetadata(cwd, oldMeta);

    const validator = new IndexFreshnessValidator(3600 * 1000);
    const freshness = validator.validate(manifest, oldMeta);
    const searcher = makeSearcherMock();
    const updater = new IncrementalEmbeddingUpdater(5);

    await updater.applyUpdates(
      cwd, searcher, freshness, manifest,
      DUMMY_EMBED_FN, DUMMY_READ_SECTION, DUMMY_EMBED_ALL, 8, "balanced",
    );

    const stats = await getIndexStats(cwd);
    expect(stats.auditRecordCount).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// genome/index.ts exports
// ---------------------------------------------------------------------------

describe("genome/index.ts — embedding-cache-coherence symbols are exported", () => {
  test("IndexFreshnessValidator is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.IndexFreshnessValidator).toBe("function");
  });

  test("IncrementalEmbeddingUpdater is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.IncrementalEmbeddingUpdater).toBe("function");
  });

  test("CacheCoherence is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.CacheCoherence).toBe("function");
  });

  test("CrossAgentCacheSync is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.CrossAgentCacheSync).toBe("function");
  });

  test("validateIndexFreshness is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.validateIndexFreshness).toBe("function");
  });

  test("rebuildIfStale is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.rebuildIfStale).toBe("function");
  });

  test("getIndexStats is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.getIndexStats).toBe("function");
  });

  test("hashManifest is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.hashManifest).toBe("function");
  });

  test("loadCacheMetadata is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.loadCacheMetadata).toBe("function");
  });

  test("saveCacheMetadata is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.saveCacheMetadata).toBe("function");
  });

  test("createCacheMetadata is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.createCacheMetadata).toBe("function");
  });
});
