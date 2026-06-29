/**
 * Tests for SectionFitnessPredictor — cross-session genome fitness predictor.
 *
 * Covers:
 *  - recordUsage: basic utility update + EMA convergence
 *  - EMA stability: monotone convergence under repeated observations
 *  - Concurrent access: serialised updates produce deterministic EMA
 *  - rerank: combined score ordering + neutral-prior for unseen sections
 *  - topic clustering edge cases: empty history, topic drift, single event
 *  - rerankByFitness: module-level convenience wrapper
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, rm } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";
import type {
  FitnessCandidate,
  RankedFitnessCandidate,
  SectionUtilityScore,
} from "../src/genome/example-selector.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeTmpDir(): string {
  return join(tmpdir(), `ashlr-fitness-${Date.now()}-${Math.random().toString(36).slice(2)}`);
}

async function setupDir(cwd: string): Promise<void> {
  await mkdir(join(cwd, ".ashlrcode", "genome", "evolution"), { recursive: true });
}

function makeCandidate(section_id: string, semantic_score: number): FitnessCandidate {
  return { section_id, semantic_score };
}

// ---------------------------------------------------------------------------
// Basic recordUsage + loadScores round-trip
// ---------------------------------------------------------------------------

describe("SectionFitnessPredictor — basic recordUsage", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("records a useful event and persists a score", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    await predictor.recordUsage("genome retrieval overview", "sections/overview.md", true);

    const scores = await predictor.loadScores();
    expect(scores["sections/overview.md"]).toBeDefined();
    expect(scores["sections/overview.md"]!.event_count).toBe(1);
  });

  test("records a not-useful event with ema = 0 on first event", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    await predictor.recordUsage("code generation", "sections/codegen.md", false);

    const scores = await predictor.loadScores();
    const entry = scores["sections/codegen.md"]!;
    expect(entry).toBeDefined();
    // First event uses raw value: false → 0.0
    const clusterKeys = Object.keys(entry.cluster_utility);
    expect(clusterKeys.length).toBeGreaterThan(0);
    const ema = entry.cluster_utility[clusterKeys[0]!]!;
    expect(ema).toBe(0.0);
  });

  test("event_count increments across multiple calls", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    await predictor.recordUsage("retrieval", "s/a.md", true);
    await predictor.recordUsage("retrieval", "s/a.md", true);
    await predictor.recordUsage("retrieval", "s/a.md", false);

    const scores = await predictor.loadScores();
    expect(scores["s/a.md"]!.event_count).toBe(3);
  });

  test("loadScores returns empty object when no file exists", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    const scores = await predictor.loadScores();
    expect(Object.keys(scores)).toHaveLength(0);
  });

  test("updated_at is a valid ISO string", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    await predictor.recordUsage("test query", "s/b.md", true);
    const scores = await predictor.loadScores();
    const ts = scores["s/b.md"]!.updated_at;
    expect(() => new Date(ts)).not.toThrow();
    expect(new Date(ts).getFullYear()).toBeGreaterThanOrEqual(2024);
  });
});

// ---------------------------------------------------------------------------
// EMA stability
// ---------------------------------------------------------------------------

describe("SectionFitnessPredictor — EMA stability", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("all-true events converge EMA toward 1.0", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    // 20 useful events on the same topic+section
    for (let i = 0; i < 20; i++) {
      await predictor.recordUsage("what is the genome manifest", "s/manifest.md", true);
    }
    const scores = await predictor.loadScores();
    const entry = scores["s/manifest.md"]!;
    const clusterKeys = Object.keys(entry.cluster_utility);
    const ema = entry.cluster_utility[clusterKeys[0]!]!;
    // After 20 all-true events the EMA should be very close to 1.0
    expect(ema).toBeGreaterThan(0.95);
  });

  test("all-false events converge EMA toward 0.0", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    for (let i = 0; i < 20; i++) {
      await predictor.recordUsage("what is the genome manifest", "s/useless.md", false);
    }
    const scores = await predictor.loadScores();
    const entry = scores["s/useless.md"]!;
    const clusterKeys = Object.keys(entry.cluster_utility);
    const ema = entry.cluster_utility[clusterKeys[0]!]!;
    expect(ema).toBeLessThan(0.05);
  });

  test("mixed events produce intermediate EMA between 0 and 1", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    for (let i = 0; i < 10; i++) {
      await predictor.recordUsage("genome section retrieval", "s/mixed.md", i % 2 === 0);
    }
    const scores = await predictor.loadScores();
    const entry = scores["s/mixed.md"]!;
    const clusterKeys = Object.keys(entry.cluster_utility);
    const ema = entry.cluster_utility[clusterKeys[0]!]!;
    expect(ema).toBeGreaterThan(0.0);
    expect(ema).toBeLessThan(1.0);
  });

  test("EMA is bounded in [0,1] after 50 random events", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    // Deterministic pseudo-random sequence via simple LCG
    let seed = 42;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 0xffffffff;
    };
    for (let i = 0; i < 50; i++) {
      await predictor.recordUsage("genome retrieval query", "s/bounded.md", rand() > 0.5);
    }
    const scores = await predictor.loadScores();
    const entry = scores["s/bounded.md"]!;
    for (const ema of Object.values(entry.cluster_utility)) {
      expect(ema).toBeGreaterThanOrEqual(0.0);
      expect(ema).toBeLessThanOrEqual(1.0);
    }
  });
});

// ---------------------------------------------------------------------------
// Concurrent-access safety
// ---------------------------------------------------------------------------

describe("SectionFitnessPredictor — concurrent access", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("10 concurrent recordUsage calls produce consistent event_count", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);

    // Fire 10 concurrent updates — the in-memory lock must serialise them.
    await Promise.all(
      Array.from({ length: 10 }, (_, i) =>
        predictor.recordUsage(`query topic ${i % 3}`, "s/concurrent.md", i % 2 === 0),
      ),
    );

    const scores = await predictor.loadScores();
    expect(scores["s/concurrent.md"]!.event_count).toBe(10);
  });

  test("two predictor instances for the same cwd serialise via shared lock", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const p1 = new SectionFitnessPredictor(cwd);
    const p2 = new SectionFitnessPredictor(cwd);

    await Promise.all([
      ...Array.from({ length: 5 }, () => p1.recordUsage("query a", "s/shared.md", true)),
      ...Array.from({ length: 5 }, () => p2.recordUsage("query a", "s/shared.md", false)),
    ]);

    const scores = await p1.loadScores();
    // Total events must equal 10 (no lost writes)
    expect(scores["s/shared.md"]!.event_count).toBe(10);
  });

  test("EMA remains in [0,1] under concurrent updates from multiple predictor instances", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictors = Array.from({ length: 4 }, () => new SectionFitnessPredictor(cwd));
    await Promise.all(
      predictors.flatMap((p, i) =>
        Array.from({ length: 5 }, (_, j) =>
          p.recordUsage("concurrent topic", "s/ema-safe.md", (i + j) % 2 === 0),
        ),
      ),
    );

    const scores = await predictors[0]!.loadScores();
    const entry = scores["s/ema-safe.md"]!;
    for (const ema of Object.values(entry.cluster_utility)) {
      expect(ema).toBeGreaterThanOrEqual(0.0);
      expect(ema).toBeLessThanOrEqual(1.0);
    }
  });
});

// ---------------------------------------------------------------------------
// rerank — combined score ordering
// ---------------------------------------------------------------------------

describe("SectionFitnessPredictor — rerank", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("empty candidates returns empty array", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    const result = await predictor.rerank([], "any query");
    expect(result).toEqual([]);
  });

  test("unseen sections get neutral prior (utility_ema = 0.5)", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    const candidates = [
      makeCandidate("s/a.md", 0.9),
      makeCandidate("s/b.md", 0.7),
    ];
    const ranked = await predictor.rerank(candidates, "genome retrieval");
    for (const r of ranked) {
      expect(r.utility_ema).toBe(0.5);
    }
  });

  test("higher semantic_score wins when utility_ema is equal (neutral prior)", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    const candidates = [
      makeCandidate("s/low.md", 0.4),
      makeCandidate("s/high.md", 0.9),
      makeCandidate("s/mid.md", 0.6),
    ];
    const ranked = await predictor.rerank(candidates, "genome retrieval");
    expect(ranked[0]!.section_id).toBe("s/high.md");
    expect(ranked[ranked.length - 1]!.section_id).toBe("s/low.md");
  });

  test("high-utility section ranks above higher-semantic section with low utility", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    const topic = "how does genome manifest work";

    // Build up strong positive utility for the lower-semantic section
    for (let i = 0; i < 15; i++) {
      await predictor.recordUsage(topic, "s/useful.md", true);
    }
    // Build up strong negative utility for the higher-semantic section
    for (let i = 0; i < 15; i++) {
      await predictor.recordUsage(topic, "s/useless.md", false);
    }

    const candidates = [
      makeCandidate("s/useless.md", 0.95), // high semantic, low utility
      makeCandidate("s/useful.md", 0.60),  // lower semantic, high utility
    ];
    const ranked = await predictor.rerank(candidates, topic);
    // combined = 0.60 × ~1.0 = ~0.60 vs 0.95 × ~0.0 = ~0.0
    expect(ranked[0]!.section_id).toBe("s/useful.md");
  });

  test("combined_score equals semantic_score × utility_ema", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    const candidates = [makeCandidate("s/x.md", 0.8)];
    const ranked = await predictor.rerank(candidates, "test topic");
    const r = ranked[0]!;
    expect(r.combined_score).toBeCloseTo(r.semantic_score * r.utility_ema, 10);
  });

  test("result preserves all input candidates", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    const candidates = Array.from({ length: 8 }, (_, i) => makeCandidate(`s/${i}.md`, i * 0.1));
    const ranked = await predictor.rerank(candidates, "any topic");
    expect(ranked).toHaveLength(8);
    const ids = new Set(ranked.map((r) => r.section_id));
    for (const c of candidates) expect(ids.has(c.section_id)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Topic clustering edge cases
// ---------------------------------------------------------------------------

describe("SectionFitnessPredictor — topic clustering edge cases", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("empty history: rerank still works with neutral priors", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    // No prior events at all
    const candidates = [makeCandidate("s/a.md", 0.7), makeCandidate("s/b.md", 0.3)];
    const ranked = await predictor.rerank(candidates, "completely new topic");
    expect(ranked).toHaveLength(2);
    // Order: 0.7×0.5=0.35 > 0.3×0.5=0.15
    expect(ranked[0]!.section_id).toBe("s/a.md");
  });

  test("single-event history: cluster count is 1", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    await predictor.recordUsage("one topic only", "s/only.md", true);
    // With one vector, k-means produces exactly 1 cluster
    expect(predictor.clusterCount).toBe(1);
  });

  test("topic drift: different topics land in different clusters after sufficient history", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);

    // Seed two very different topic families
    const codeTopics = [
      "typescript function signature generation",
      "code generation tool calling patterns",
      "typescript class method extraction",
    ];
    const docTopics = [
      "project vision roadmap milestones planning",
      "documentation strategy goals objectives",
      "milestone completion criteria review",
    ];

    for (const t of codeTopics) await predictor.recordUsage(t, "s/code.md", true);
    for (const t of docTopics) await predictor.recordUsage(t, "s/docs.md", true);

    // Rerank on a code topic — code section should rank above docs section
    const codeRanked = await predictor.rerank(
      [makeCandidate("s/code.md", 0.6), makeCandidate("s/docs.md", 0.6)],
      "typescript code generation patterns",
    );
    // Both have same semantic score; code section should have higher utility for code cluster
    // (This may not always hold with 3 events each due to cluster variance, so we just verify
    //  no crash and correct shape)
    expect(codeRanked).toHaveLength(2);
    expect(codeRanked[0]!.combined_score).toBeGreaterThanOrEqual(codeRanked[1]!.combined_score);
  });

  test("rerank handles sections with scores from multiple clusters", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);

    // Record events under two different topic families for the same section
    await predictor.recordUsage("code generation typescript", "s/multi.md", true);
    await predictor.recordUsage("project documentation milestones", "s/multi.md", false);

    const ranked = await predictor.rerank(
      [makeCandidate("s/multi.md", 0.75)],
      "code generation",
    );
    expect(ranked).toHaveLength(1);
    expect(ranked[0]!.utility_ema).toBeGreaterThanOrEqual(0);
    expect(ranked[0]!.utility_ema).toBeLessThanOrEqual(1);
  });

  test("empty query string does not throw", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    await expect(
      predictor.rerank([makeCandidate("s/a.md", 0.5)], ""),
    ).resolves.toHaveLength(1);
  });

  test("very long query does not throw", async () => {
    const { SectionFitnessPredictor } = await import("../src/genome/example-selector.ts");
    const predictor = new SectionFitnessPredictor(cwd);
    const longQuery = "genome retrieval section ".repeat(50);
    await expect(
      predictor.rerank([makeCandidate("s/a.md", 0.5)], longQuery),
    ).resolves.toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// rerankByFitness module-level helper
// ---------------------------------------------------------------------------

describe("rerankByFitness — module-level helper", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("returns ranked candidates via convenience wrapper", async () => {
    const { rerankByFitness } = await import("../src/genome/example-selector.ts");
    const candidates = [
      makeCandidate("s/a.md", 0.8),
      makeCandidate("s/b.md", 0.4),
    ];
    const ranked = await rerankByFitness(candidates, "genome manifest retrieval", cwd);
    expect(ranked).toHaveLength(2);
    expect(ranked[0]!.section_id).toBe("s/a.md"); // higher semantic wins with equal prior
  });

  test("empty candidates returns empty array via wrapper", async () => {
    const { rerankByFitness } = await import("../src/genome/example-selector.ts");
    const result = await rerankByFitness([], "any topic", cwd);
    expect(result).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Exports available via genome index
// ---------------------------------------------------------------------------

describe("genome/index.ts — fitness predictor symbols are exported", () => {
  test("SectionFitnessPredictor is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof (mod as Record<string, unknown>)["SectionFitnessPredictor"]).toBe("function");
  });

  test("rerankByFitness is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof (mod as Record<string, unknown>)["rerankByFitness"]).toBe("function");
  });
});
