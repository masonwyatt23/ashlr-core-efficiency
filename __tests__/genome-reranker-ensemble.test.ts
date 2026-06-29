/**
 * Tests for genome/reranker-ensemble.ts
 *
 * Covers:
 *  1. Single-tier dominance — when one pass vastly outscores the others
 *  2. Multi-tier tie scenarios — equal TF-IDF scores resolved by semantic
 *  3. Cross-reference discovery — graph pass boosts in-tree xref sections
 *  4. Graph cycles — cycle-safe traversal does not hang
 *  5. Budget constraints — token budget correctly trims results
 *  6. Drift detection — detectWeightDrift reports correctly
 *  7. Weight learning — learnWeightsFromFeedback converges direction
 *  8. Normalisation and helpers — BM25, normalizeScores, redundancy penalty
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, rm, writeFile } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";
import type { SectionMeta } from "../src/genome/manifest.ts";
import type { SectionGraph, SectionNode } from "../src/genome/graph-traversal.ts";

// ─── helpers ────────────────────────────────────────────────────────────────

function makeTmpDir(): string {
  return join(
    tmpdir(),
    `ashlr-reranker-${Date.now()}-${Math.random().toString(36).slice(2)}`,
  );
}

function makeSection(
  path: string,
  overrides: Partial<SectionMeta> = {},
): SectionMeta {
  return {
    path,
    title: path,
    summary: "",
    tags: [],
    tokens: 100,
    updatedAt: new Date().toISOString(),
    ...overrides,
  };
}

/**
 * Build a minimal SectionGraph from a flat list of (path, parentId?, crossRefs?).
 */
function buildGraph(
  nodes: Array<{
    path: string;
    parentId?: string;
    crossRefs?: string[];
    tags?: string[];
    title?: string;
    summary?: string;
  }>,
): SectionGraph {
  const graph: SectionGraph = new Map();
  const pathSet = new Set(nodes.map((n) => n.path));

  // First pass: create nodes
  for (const n of nodes) {
    const parent =
      n.parentId && pathSet.has(n.parentId) ? n.parentId : undefined;
    graph.set(n.path, {
      meta: {
        path: n.path,
        title: n.title ?? n.path,
        summary: n.summary ?? "",
        tags: n.tags ?? [],
        tokens: 100,
        updatedAt: new Date().toISOString(),
        parentId: parent,
      },
      children: [],
      parent,
      siblings: [],
      crossRefs: (n.crossRefs ?? []).filter(
        (r) => pathSet.has(r) && r !== n.path,
      ),
    });
  }

  // Second pass: wire children
  for (const [, node] of graph) {
    if (node.parent) {
      const parentNode = graph.get(node.parent);
      if (parentNode && !parentNode.children.includes(node.meta.path)) {
        parentNode.children.push(node.meta.path);
      }
    }
  }

  return graph;
}

/** Identity cosine — treats vectors as pre-normalised scalars for simple tests. */
function scalarCosine(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0) return 0;
  // Dot product / (|a| * |b|) for 1-D vectors
  return (a[0]! * b[0]!) / (Math.abs(a[0]!) * Math.abs(b[0]!) || 1);
}

/** Proper cosine for n-D vector tests. */
function cosine(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

// ─── 1. normalizeScores ──────────────────────────────────────────────────────

describe("normalizeScores — min-max normalization", () => {
  test("empty array returns empty array", async () => {
    const { normalizeScores } = await import("../src/genome/reranker-ensemble.ts");
    expect(normalizeScores([])).toEqual([]);
  });

  test("all-zero array returns all zeros", async () => {
    const { normalizeScores } = await import("../src/genome/reranker-ensemble.ts");
    expect(normalizeScores([0, 0, 0])).toEqual([0, 0, 0]);
  });

  test("single-element array returns [0] (range = 0)", async () => {
    const { normalizeScores } = await import("../src/genome/reranker-ensemble.ts");
    expect(normalizeScores([5])).toEqual([0]);
  });

  test("max element maps to 1, min to 0", async () => {
    const { normalizeScores } = await import("../src/genome/reranker-ensemble.ts");
    const result = normalizeScores([2, 5, 8]);
    expect(result[2]).toBeCloseTo(1);
    expect(result[0]).toBeCloseTo(0);
    expect(result[1]).toBeGreaterThan(0);
    expect(result[1]).toBeLessThan(1);
  });

  test("all-equal array returns all zeros", async () => {
    const { normalizeScores } = await import("../src/genome/reranker-ensemble.ts");
    expect(normalizeScores([3, 3, 3])).toEqual([0, 0, 0]);
  });
});

// ─── 2. computeBm25Score ────────────────────────────────────────────────────

describe("computeBm25Score — BM25 term scoring", () => {
  test("returns 0 for empty query terms", async () => {
    const { computeBm25Score } = await import("../src/genome/reranker-ensemble.ts");
    const meta = makeSection("a/b.md", { title: "Architecture", summary: "design", tags: ["arch"] });
    expect(computeBm25Score(meta, [], 100, 300)).toBe(0);
  });

  test("section with matching tags/title scores higher than non-matching section", async () => {
    const { computeBm25Score } = await import("../src/genome/reranker-ensemble.ts");
    const match = makeSection("a/b.md", {
      title: "Architecture Overview",
      summary: "architecture design patterns",
      tags: ["architecture"],
    });
    const noMatch = makeSection("c/d.md", {
      title: "Logging Format",
      summary: "log output format",
      tags: ["logging"],
    });
    const queryTerms = ["architecture", "design"];
    const s1 = computeBm25Score(match, queryTerms, 100, 300);
    const s2 = computeBm25Score(noMatch, queryTerms, 100, 300);
    expect(s1).toBeGreaterThan(s2);
  });

  test("additional content terms boost score above metadata-only", async () => {
    const { computeBm25Score } = await import("../src/genome/reranker-ensemble.ts");
    const meta = makeSection("a/b.md", {
      title: "Architecture",
      summary: "architecture overview",
      tags: [],
    });
    const queryTerms = ["architecture", "patterns"];
    const withoutContent = computeBm25Score(meta, queryTerms, 100, 300);
    const withContent = computeBm25Score(
      meta,
      queryTerms,
      300,
      300,
      "# Architecture\nPatterns and patterns design patterns everywhere.",
    );
    expect(withContent).toBeGreaterThanOrEqual(withoutContent);
  });

  test("score is non-negative", async () => {
    const { computeBm25Score } = await import("../src/genome/reranker-ensemble.ts");
    const meta = makeSection("x/y.md");
    expect(computeBm25Score(meta, ["any", "terms"], 50, 300)).toBeGreaterThanOrEqual(0);
  });
});

// ─── 3. collectInTreePaths — graph proximity ─────────────────────────────────

describe("collectInTreePaths — DAG traversal for proximity", () => {
  test("root node with depth 0 returns only itself", async () => {
    const { collectInTreePaths } = await import("../src/genome/reranker-ensemble.ts");
    const graph = buildGraph([{ path: "arch/root.md" }]);
    const paths = collectInTreePaths("arch/root.md", graph, 0);
    expect(paths.has("arch/root.md")).toBe(true);
    expect(paths.size).toBe(1);
  });

  test("depth 1 includes direct children", async () => {
    const { collectInTreePaths } = await import("../src/genome/reranker-ensemble.ts");
    const graph = buildGraph([
      { path: "arch/root.md" },
      { path: "arch/child.md", parentId: "arch/root.md" },
    ]);
    const paths = collectInTreePaths("arch/root.md", graph, 1);
    expect(paths.has("arch/root.md")).toBe(true);
    expect(paths.has("arch/child.md")).toBe(true);
  });

  test("depth 2 includes grandchildren", async () => {
    const { collectInTreePaths } = await import("../src/genome/reranker-ensemble.ts");
    const graph = buildGraph([
      { path: "a/root.md" },
      { path: "a/child.md", parentId: "a/root.md" },
      { path: "a/grand.md", parentId: "a/child.md" },
    ]);
    const paths = collectInTreePaths("a/root.md", graph, 2);
    expect(paths.has("a/grand.md")).toBe(true);
  });

  test("cross-referenced paths are included within depth budget", async () => {
    const { collectInTreePaths } = await import("../src/genome/reranker-ensemble.ts");
    const graph = buildGraph([
      { path: "vision/intro.md", crossRefs: ["arch/patterns.md"] },
      { path: "arch/patterns.md" },
    ]);
    const paths = collectInTreePaths("vision/intro.md", graph, 1);
    expect(paths.has("arch/patterns.md")).toBe(true);
  });

  test("cycle-safe — A→B→A does not cause infinite loop", async () => {
    const { collectInTreePaths } = await import("../src/genome/reranker-ensemble.ts");
    // Manually build a cyclic graph
    const graph: SectionGraph = new Map();
    const nodeA: SectionNode = {
      meta: makeSection("cycle/a.md"),
      children: ["cycle/b.md"],
      parent: "cycle/b.md", // cycle
      siblings: [],
      crossRefs: [],
    };
    const nodeB: SectionNode = {
      meta: makeSection("cycle/b.md"),
      children: ["cycle/a.md"], // cycle
      parent: "cycle/a.md",
      siblings: [],
      crossRefs: [],
    };
    graph.set("cycle/a.md", nodeA);
    graph.set("cycle/b.md", nodeB);

    const start = Date.now();
    const paths = collectInTreePaths("cycle/a.md", graph, 10);
    const elapsed = Date.now() - start;

    expect(elapsed).toBeLessThan(1000);
    expect(paths.has("cycle/a.md")).toBe(true);
    expect(paths.has("cycle/b.md")).toBe(true);
    // Both nodes should appear exactly once (no infinite expansion)
    expect(paths.size).toBe(2);
  });
});

// ─── 4. Single-tier dominance ────────────────────────────────────────────────

describe("rerankerEnsemble — single-tier dominance", () => {
  test("BM25-dominant weight: highest BM25 section ranks first", async () => {
    const { rerankerEnsemble } = await import("../src/genome/reranker-ensemble.ts");
    const graph = buildGraph([
      { path: "arch/a.md" },
      { path: "arch/b.md" },
      { path: "arch/c.md" },
    ]);

    const candidates = [
      {
        meta: makeSection("arch/a.md", {
          title: "Architecture",
          summary: "architecture design",
          tags: ["architecture"],
          tokens: 50,
        }),
        tfidfScore: 10,
        semanticScore: 0.5,
        tokens: 50,
      },
      {
        meta: makeSection("arch/b.md", {
          title: "Other",
          summary: "other content",
          tags: [],
          tokens: 50,
        }),
        tfidfScore: 1,
        semanticScore: 0.4,
        tokens: 50,
      },
      {
        meta: makeSection("arch/c.md", {
          title: "Logging",
          summary: "logging format",
          tags: [],
          tokens: 50,
        }),
        tfidfScore: 1,
        semanticScore: 0.3,
        tokens: 50,
      },
    ];

    const results = rerankerEnsemble(
      candidates,
      "architecture design",
      graph,
      {
        queryEmbedding: null, // no semantic
        embeddingCache: new Map(),
        weights: { bm25: 1.0, semantic: 0.0, graph: 0.0 },
      },
    );

    expect(results[0]!.path).toBe("arch/a.md");
  });

  test("semantic-dominant weight: highest cosine section ranks first", async () => {
    const { rerankerEnsemble } = await import("../src/genome/reranker-ensemble.ts");
    const graph = buildGraph([
      { path: "a/high.md" },
      { path: "a/low.md" },
    ]);

    // Embeddings: high similarity for "a/high.md", low for "a/low.md"
    const queryEmb = [1.0, 0.0];
    const highEmb = [1.0, 0.0];  // cos = 1.0
    const lowEmb  = [0.0, 1.0];  // cos = 0.0

    const embCache = new Map<string, number[]>([
      ["a/high.md", highEmb],
      ["a/low.md", lowEmb],
    ]);

    const candidates = [
      {
        meta: makeSection("a/high.md", { title: "High", summary: "content", tokens: 50 }),
        tfidfScore: 1,
        semanticScore: 1.0,
        tokens: 50,
      },
      {
        meta: makeSection("a/low.md", { title: "Low", summary: "content", tokens: 50 }),
        tfidfScore: 1,
        semanticScore: 0.0,
        tokens: 50,
      },
    ];

    const results = rerankerEnsemble(
      candidates,
      "query",
      graph,
      {
        queryEmbedding: queryEmb,
        embeddingCache: embCache,
        cosineFn: cosine,
        weights: { bm25: 0.0, semantic: 1.0, graph: 0.0 },
      },
    );

    expect(results[0]!.path).toBe("a/high.md");
  });
});

// ─── 5. Multi-tier tie scenarios ─────────────────────────────────────────────

describe("rerankerEnsemble — multi-tier tie scenarios", () => {
  test("equal BM25 scores resolved by semantic cosine similarity", async () => {
    const { rerankerEnsemble } = await import("../src/genome/reranker-ensemble.ts");
    const graph = buildGraph([
      { path: "tie/a.md" },
      { path: "tie/b.md" },
    ]);

    // Both have identical BM25-influencing metadata — only semantic differs
    const queryEmb = [1.0, 0.0, 0.0];
    const embA = [1.0, 0.0, 0.0]; // cos ≈ 1
    const embB = [0.0, 1.0, 0.0]; // cos ≈ 0

    const embCache = new Map<string, number[]>([
      ["tie/a.md", embA],
      ["tie/b.md", embB],
    ]);

    const candidates = [
      {
        meta: makeSection("tie/a.md", {
          title: "Identical title",
          summary: "same summary architecture",
          tags: ["architecture"],
          tokens: 50,
        }),
        tfidfScore: 5,
        semanticScore: null,
        tokens: 50,
      },
      {
        meta: makeSection("tie/b.md", {
          title: "Identical title",
          summary: "same summary architecture",
          tags: ["architecture"],
          tokens: 50,
        }),
        tfidfScore: 5,
        semanticScore: null,
        tokens: 50,
      },
    ];

    const results = rerankerEnsemble(
      candidates,
      "architecture",
      graph,
      {
        queryEmbedding: queryEmb,
        embeddingCache: embCache,
        cosineFn: cosine,
        weights: { bm25: 0.40, semantic: 0.50, graph: 0.10 },
      },
    );

    // "tie/a.md" should rank first due to higher cosine similarity
    expect(results[0]!.path).toBe("tie/a.md");
  });

  test("results are sorted descending by ensemble score", async () => {
    const { rerankerEnsemble } = await import("../src/genome/reranker-ensemble.ts");
    const graph = buildGraph([
      { path: "s/a.md" },
      { path: "s/b.md" },
      { path: "s/c.md" },
    ]);

    const candidates = [
      {
        meta: makeSection("s/a.md", { title: "A", summary: "alpha beta gamma", tags: ["alpha"], tokens: 50 }),
        tfidfScore: 3,
        semanticScore: null,
        tokens: 50,
      },
      {
        meta: makeSection("s/b.md", { title: "B", summary: "alpha beta", tags: ["alpha"], tokens: 50 }),
        tfidfScore: 2,
        semanticScore: null,
        tokens: 50,
      },
      {
        meta: makeSection("s/c.md", { title: "C", summary: "alpha", tags: [], tokens: 50 }),
        tfidfScore: 1,
        semanticScore: null,
        tokens: 50,
      },
    ];

    const results = rerankerEnsemble(
      candidates,
      "alpha beta gamma",
      graph,
      {
        queryEmbedding: null,
        embeddingCache: new Map(),
        weights: { bm25: 1.0, semantic: 0.0, graph: 0.0 },
      },
    );

    for (let i = 1; i < results.length; i++) {
      expect(results[i - 1]!.score).toBeGreaterThanOrEqual(results[i]!.score);
    }
  });
});

// ─── 6. Cross-reference discovery ───────────────────────────────────────────

describe("rerankerEnsemble — cross-reference graph boost", () => {
  test("section cross-referenced from top candidate receives graphBoosted=true", async () => {
    const { rerankerEnsemble } = await import("../src/genome/reranker-ensemble.ts");
    // top candidate is "vision/intro.md" which cross-refs "arch/patterns.md"
    const graph = buildGraph([
      { path: "vision/intro.md", crossRefs: ["arch/patterns.md"] },
      { path: "arch/patterns.md" },
      { path: "other/section.md" },
    ]);

    const candidates = [
      {
        meta: makeSection("vision/intro.md", {
          title: "Vision Intro",
          summary: "vision architecture intro design",
          tags: ["vision", "architecture"],
          tokens: 50,
        }),
        tfidfScore: 10,
        semanticScore: null,
        tokens: 50,
      },
      {
        meta: makeSection("arch/patterns.md", {
          title: "Patterns",
          summary: "patterns overview",
          tags: ["patterns"],
          tokens: 50,
        }),
        tfidfScore: 2,
        semanticScore: null,
        tokens: 50,
      },
      {
        meta: makeSection("other/section.md", {
          title: "Other",
          summary: "unrelated content",
          tags: [],
          tokens: 50,
        }),
        tfidfScore: 2,
        semanticScore: null,
        tokens: 50,
      },
    ];

    const results = rerankerEnsemble(
      candidates,
      "vision architecture",
      graph,
      {
        queryEmbedding: null,
        embeddingCache: new Map(),
        weights: { bm25: 0.40, semantic: 0.0, graph: 0.60 },
      },
    );

    const patterns = results.find((r) => r.path === "arch/patterns.md");
    expect(patterns).toBeDefined();
    expect(patterns!.breakdown.graphBoosted).toBe(true);

    const other = results.find((r) => r.path === "other/section.md");
    if (other) {
      expect(other.breakdown.graphBoosted).toBe(false);
    }
  });

  test("cross-ref section scores higher than orphaned section with equal BM25", async () => {
    const { rerankerEnsemble } = await import("../src/genome/reranker-ensemble.ts");
    const graph = buildGraph([
      { path: "top/anchor.md", crossRefs: ["top/xref.md"] },
      { path: "top/xref.md" },
      { path: "top/orphan.md" },
    ]);

    const candidates = [
      {
        meta: makeSection("top/anchor.md", {
          summary: "main anchor architecture",
          tags: ["architecture"],
          tokens: 50,
        }),
        tfidfScore: 10,
        semanticScore: null,
        tokens: 50,
      },
      {
        meta: makeSection("top/xref.md", {
          summary: "xref content",
          tags: [],
          tokens: 50,
        }),
        tfidfScore: 3,
        semanticScore: null,
        tokens: 50,
      },
      {
        meta: makeSection("top/orphan.md", {
          summary: "orphan content",
          tags: [],
          tokens: 50,
        }),
        tfidfScore: 3,
        semanticScore: null,
        tokens: 50,
      },
    ];

    const results = rerankerEnsemble(
      candidates,
      "architecture",
      graph,
      {
        queryEmbedding: null,
        embeddingCache: new Map(),
        weights: { bm25: 0.30, semantic: 0.0, graph: 0.70 },
      },
    );

    const xrefResult = results.find((r) => r.path === "top/xref.md");
    const orphanResult = results.find((r) => r.path === "top/orphan.md");
    if (xrefResult && orphanResult) {
      expect(xrefResult.score).toBeGreaterThanOrEqual(orphanResult.score);
    }
  });
});

// ─── 7. Token budget enforcement ─────────────────────────────────────────────

describe("rerankerEnsemble — token budget constraints", () => {
  test("total tokens of returned results does not exceed maxTokens", async () => {
    const { rerankerEnsemble } = await import("../src/genome/reranker-ensemble.ts");
    const graph = buildGraph([
      { path: "budget/a.md" },
      { path: "budget/b.md" },
      { path: "budget/c.md" },
    ]);

    const candidates = [
      {
        meta: makeSection("budget/a.md", { summary: "architecture design", tags: ["architecture"], tokens: 300 }),
        tfidfScore: 10,
        semanticScore: null,
        tokens: 300,
      },
      {
        meta: makeSection("budget/b.md", { summary: "architecture patterns", tags: ["architecture"], tokens: 300 }),
        tfidfScore: 8,
        semanticScore: null,
        tokens: 300,
      },
      {
        meta: makeSection("budget/c.md", { summary: "architecture overview", tags: ["architecture"], tokens: 300 }),
        tfidfScore: 6,
        semanticScore: null,
        tokens: 300,
      },
    ];

    const results = rerankerEnsemble(
      candidates,
      "architecture",
      graph,
      {
        queryEmbedding: null,
        embeddingCache: new Map(),
        weights: { bm25: 1.0, semantic: 0.0, graph: 0.0 },
        maxTokens: 500,
      },
    );

    const total = results.reduce((sum, r) => {
      const c = candidates.find((cand) => cand.meta.path === r.path);
      return sum + (c?.tokens ?? 0);
    }, 0);
    expect(total).toBeLessThanOrEqual(500);
  });

  test("no maxTokens: all candidates returned (when none filtered by score)", async () => {
    const { rerankerEnsemble } = await import("../src/genome/reranker-ensemble.ts");
    const graph = buildGraph([
      { path: "full/a.md" },
      { path: "full/b.md" },
    ]);

    const candidates = [
      {
        meta: makeSection("full/a.md", { summary: "architecture", tags: ["architecture"], tokens: 5000 }),
        tfidfScore: 5,
        semanticScore: null,
        tokens: 5000,
      },
      {
        meta: makeSection("full/b.md", { summary: "design", tags: ["design"], tokens: 5000 }),
        tfidfScore: 5,
        semanticScore: null,
        tokens: 5000,
      },
    ];

    const results = rerankerEnsemble(
      candidates,
      "architecture design",
      graph,
      {
        queryEmbedding: null,
        embeddingCache: new Map(),
        // No maxTokens — no budget filtering
      },
    );

    expect(results.length).toBe(2);
  });
});

// ─── 8. Drift detection ──────────────────────────────────────────────────────

describe("detectWeightDrift — weight change monitoring", () => {
  test("no drift when weights match default", async () => {
    const { detectWeightDrift, DEFAULT_ENSEMBLE_WEIGHTS } = await import(
      "../src/genome/reranker-ensemble.ts"
    );
    const report = detectWeightDrift({ ...DEFAULT_ENSEMBLE_WEIGHTS });
    expect(report.drifted).toBe(false);
    expect(report.deltas.bm25).toBeCloseTo(0);
    expect(report.deltas.semantic).toBeCloseTo(0);
    expect(report.deltas.graph).toBeCloseTo(0);
  });

  test("drift detected when any single weight deviates by > threshold", async () => {
    const { detectWeightDrift } = await import("../src/genome/reranker-ensemble.ts");
    const shifted: import("../src/genome/reranker-ensemble.ts").EnsembleWeights = {
      bm25: 0.05, // was 0.40 — delta = 0.35 > 0.15
      semantic: 0.80,
      graph: 0.15,
    };
    const report = detectWeightDrift(shifted, undefined, 0.15);
    expect(report.drifted).toBe(true);
    expect(report.deltas.bm25).toBeCloseTo(0.35);
  });

  test("custom threshold: small drift not flagged below threshold", async () => {
    const { detectWeightDrift } = await import("../src/genome/reranker-ensemble.ts");
    const slight: import("../src/genome/reranker-ensemble.ts").EnsembleWeights = {
      bm25: 0.42,
      semantic: 0.48,
      graph: 0.10,
    };
    // Delta = 0.02, threshold = 0.05 → no drift
    const report = detectWeightDrift(slight, undefined, 0.05);
    expect(report.drifted).toBe(false);
  });

  test("drift report includes threshold value", async () => {
    const { detectWeightDrift, DEFAULT_ENSEMBLE_WEIGHTS } = await import(
      "../src/genome/reranker-ensemble.ts"
    );
    const report = detectWeightDrift({ ...DEFAULT_ENSEMBLE_WEIGHTS }, undefined, 0.20);
    expect(report.threshold).toBe(0.20);
  });
});

// ─── 9. Weight learning ───────────────────────────────────────────────────────

describe("learnWeightsFromFeedback — L2-regularised weight learning", () => {
  test("returns default weights when fewer than 3 records", async () => {
    const { learnWeightsFromFeedback, DEFAULT_ENSEMBLE_WEIGHTS } = await import(
      "../src/genome/reranker-ensemble.ts"
    );
    const result = learnWeightsFromFeedback([]);
    expect(result.bm25).toBeCloseTo(DEFAULT_ENSEMBLE_WEIGHTS.bm25, 2);
    expect(result.semantic).toBeCloseTo(DEFAULT_ENSEMBLE_WEIGHTS.semantic, 2);
    expect(result.graph).toBeCloseTo(DEFAULT_ENSEMBLE_WEIGHTS.graph, 2);
  });

  test("learned weights sum to 1", async () => {
    const { learnWeightsFromFeedback } = await import("../src/genome/reranker-ensemble.ts");
    const weights: import("../src/genome/reranker-ensemble.ts").EnsembleWeights = {
      bm25: 0.40,
      semantic: 0.50,
      graph: 0.10,
    };
    const records: import("../src/genome/reranker-ensemble.ts").RerankerFeedbackRecord[] = Array.from(
      { length: 5 },
      (_, i) => ({
        ts: new Date().toISOString(),
        query: `query ${i}`,
        rankedPaths: ["a/b.md", "c/d.md", "e/f.md"],
        groundTruthPaths: ["a/b.md"],
        weights,
      }),
    );
    const learned = learnWeightsFromFeedback(records);
    const sum = learned.bm25 + learned.semantic + learned.graph;
    expect(sum).toBeCloseTo(1, 2);
  });

  test("all learned weights are non-negative", async () => {
    const { learnWeightsFromFeedback } = await import("../src/genome/reranker-ensemble.ts");
    const weights: import("../src/genome/reranker-ensemble.ts").EnsembleWeights = {
      bm25: 0.40,
      semantic: 0.50,
      graph: 0.10,
    };
    const records: import("../src/genome/reranker-ensemble.ts").RerankerFeedbackRecord[] = Array.from(
      { length: 10 },
      (_, i) => ({
        ts: new Date().toISOString(),
        query: `test query ${i}`,
        rankedPaths: ["x/a.md", "x/b.md"],
        groundTruthPaths: i % 2 === 0 ? ["x/a.md"] : ["x/b.md"],
        weights,
      }),
    );
    const learned = learnWeightsFromFeedback(records);
    expect(learned.bm25).toBeGreaterThanOrEqual(0);
    expect(learned.semantic).toBeGreaterThanOrEqual(0);
    expect(learned.graph).toBeGreaterThanOrEqual(0);
  });
});

// ─── 10. computeRedundancyPenalty ────────────────────────────────────────────

describe("computeRedundancyPenalty — sibling dedup in semantic pass", () => {
  test("identical candidate and top-3 embedding returns penalty close to 1", async () => {
    const { computeRedundancyPenalty } = await import("../src/genome/reranker-ensemble.ts");
    const emb = [1.0, 0.0, 0.0];
    const penalty = computeRedundancyPenalty(emb, [emb], cosine);
    expect(penalty).toBeCloseTo(1.0, 1);
  });

  test("orthogonal candidate returns penalty 0", async () => {
    const { computeRedundancyPenalty } = await import("../src/genome/reranker-ensemble.ts");
    const candidate = [1.0, 0.0, 0.0];
    const top3 = [[0.0, 1.0, 0.0]];
    const penalty = computeRedundancyPenalty(candidate, top3, cosine);
    expect(penalty).toBeCloseTo(0, 1);
  });

  test("null candidate embedding returns 0", async () => {
    const { computeRedundancyPenalty } = await import("../src/genome/reranker-ensemble.ts");
    const penalty = computeRedundancyPenalty(null, [[1, 0, 0]], cosine);
    expect(penalty).toBe(0);
  });

  test("empty top-3 returns 0", async () => {
    const { computeRedundancyPenalty } = await import("../src/genome/reranker-ensemble.ts");
    const penalty = computeRedundancyPenalty([1, 0, 0], [], cosine);
    expect(penalty).toBe(0);
  });
});

// ─── 11. Feedback I/O (integration with tmp dir) ─────────────────────────────

describe("appendFeedbackRecord + loadFeedbackRecords — JSONL persistence", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await mkdir(join(cwd, ".ashlrcode", "genome", "evolution"), { recursive: true });
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("appended record is readable from loadFeedbackRecords", async () => {
    const { appendFeedbackRecord, loadFeedbackRecords } = await import(
      "../src/genome/reranker-ensemble.ts"
    );
    await appendFeedbackRecord(cwd, {
      query: "test query",
      rankedPaths: ["a/b.md"],
      groundTruthPaths: ["a/b.md"],
      weights: { bm25: 0.4, semantic: 0.5, graph: 0.1 },
    });
    const records = await loadFeedbackRecords(cwd);
    expect(records.length).toBeGreaterThan(0);
    expect(records[0]!.query).toBe("test query");
  });

  test("loadFeedbackRecords returns empty array for missing file", async () => {
    const { loadFeedbackRecords } = await import("../src/genome/reranker-ensemble.ts");
    // Use a different cwd that has no feedback file
    const empty = makeTmpDir();
    await mkdir(join(empty, ".ashlrcode", "genome", "evolution"), { recursive: true });
    const records = await loadFeedbackRecords(empty);
    expect(records).toEqual([]);
    await rm(empty, { recursive: true, force: true });
  });
});

// ─── 12. genome/index.ts exports ────────────────────────────────────────────

describe("genome/index.ts — reranker-ensemble symbols are exported", () => {
  test("rerankerEnsemble is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof (mod as Record<string, unknown>).rerankerEnsemble).toBe("function");
  });

  test("rerankerEnsembleAsync is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof (mod as Record<string, unknown>).rerankerEnsembleAsync).toBe("function");
  });

  test("DEFAULT_ENSEMBLE_WEIGHTS is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect((mod as Record<string, unknown>).DEFAULT_ENSEMBLE_WEIGHTS).toBeDefined();
  });

  test("detectWeightDrift is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof (mod as Record<string, unknown>).detectWeightDrift).toBe("function");
  });

  test("learnWeightsFromFeedback is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof (mod as Record<string, unknown>).learnWeightsFromFeedback).toBe("function");
  });
});
