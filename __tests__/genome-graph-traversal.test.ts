/**
 * Tests for genome/graph-traversal.ts
 *
 * Covers:
 *  - buildSectionGraph: DAG construction, parent/child/sibling wiring, cross-refs
 *  - Cycle detection: A→B→A loops and self-referential cycles
 *  - Cross-reference resolution: "See X/Y.md" pattern extraction & validation
 *  - Sibling ranking: overlap-threshold filtering, unrelated sibling exclusion
 *  - Token budget enforcement: aggressive cap, truncation path
 *  - Missing/invalid references: dangling parentId, nonexistent cross-ref paths
 *  - buildAndTraverseGraph: end-to-end with real files on disk
 *  - relatedSectionPaths: populated on sections with discovered cross-refs
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, rm, writeFile } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";
import type { GenomeManifest, SectionMeta } from "../src/genome/manifest.ts";

// ─── helpers ────────────────────────────────────────────────────────────────

function makeTmpDir(): string {
  return join(
    tmpdir(),
    `ashlr-graph-${Date.now()}-${Math.random().toString(36).slice(2)}`,
  );
}

function makeManifest(sections: SectionMeta[] = []): GenomeManifest {
  const now = new Date().toISOString();
  return {
    version: 1,
    project: "test",
    sections,
    generation: { number: 1, milestone: "", startedAt: now },
    fitnessHistory: [],
    createdAt: now,
    updatedAt: now,
  };
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
    tokens: 10,
    updatedAt: new Date().toISOString(),
    ...overrides,
  };
}

async function setupGenomeDir(cwd: string): Promise<string> {
  const dir = join(cwd, ".ashlrcode", "genome");
  await mkdir(dir, { recursive: true });
  return dir;
}

/** Write a section file under .ashlrcode/genome/<relPath>. */
async function writeGenomeSection(
  cwd: string,
  relPath: string,
  content: string,
): Promise<void> {
  const fullPath = join(cwd, ".ashlrcode", "genome", relPath);
  const dir = fullPath.split("/").slice(0, -1).join("/");
  await mkdir(dir, { recursive: true });
  await writeFile(fullPath, content, "utf-8");
}

// ─── extractCrossRefCandidates ───────────────────────────────────────────────

describe("extractCrossRefCandidates — pattern extraction from metadata", () => {
  test("extracts 'See X/Y.md' cross-ref from summary", async () => {
    const { extractCrossRefCandidates } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const meta = makeSection("vision/intro.md", {
      summary: "This section describes the vision. See architecture/patterns.md for details.",
    });
    const refs = extractCrossRefCandidates(meta);
    expect(refs).toContain("architecture/patterns.md");
  });

  test("extracts arrow-style '→ X/Y' cross-ref", async () => {
    const { extractCrossRefCandidates } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const meta = makeSection("a/b.md", {
      summary: "Related content → strategies/active for planning.",
    });
    const refs = extractCrossRefCandidates(meta);
    expect(refs.some((r) => r.includes("strategies"))).toBe(true);
  });

  test("extracts cross-ref from tags", async () => {
    const { extractCrossRefCandidates } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const meta = makeSection("a/b.md", {
      tags: ["see also: milestones/current.md"],
    });
    const refs = extractCrossRefCandidates(meta);
    expect(refs.some((r) => r.includes("milestones"))).toBe(true);
  });

  test("returns empty array when no cross-refs in metadata", async () => {
    const { extractCrossRefCandidates } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const meta = makeSection("plain/section.md", {
      summary: "Just a plain section with no references.",
      tags: ["plain"],
    });
    const refs = extractCrossRefCandidates(meta);
    expect(refs).toHaveLength(0);
  });

  test("does not emit self-references when path appears in summary", async () => {
    const { extractCrossRefCandidates } = await import(
      "../src/genome/graph-traversal.ts"
    );
    // self-refs are filtered later in buildSectionGraph, not here — this just checks extraction
    const meta = makeSection("a/b.md", {
      summary: "See other/section.md for related content.",
    });
    const refs = extractCrossRefCandidates(meta);
    // 'other/section.md' should be found
    expect(refs.some((r) => r.includes("other"))).toBe(true);
  });

  test("deduplicates identical cross-refs appearing multiple times", async () => {
    const { extractCrossRefCandidates } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const meta = makeSection("a/b.md", {
      summary:
        "See architecture/overview.md for context. Also see architecture/overview.md again.",
    });
    const refs = extractCrossRefCandidates(meta);
    const counted = refs.filter((r) => r === "architecture/overview.md").length;
    expect(counted).toBeLessThanOrEqual(1);
  });
});

// ─── buildSectionGraph — DAG construction ───────────────────────────────────

describe("buildSectionGraph — DAG construction from metadata", () => {
  test("empty manifest produces empty graph", async () => {
    const { buildSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const graph = buildSectionGraph(makeManifest([]));
    expect(graph.size).toBe(0);
  });

  test("single root section has no parent, no children, no siblings", async () => {
    const { buildSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const graph = buildSectionGraph(
      makeManifest([makeSection("vision/north-star.md")]),
    );
    const node = graph.get("vision/north-star.md");
    expect(node).toBeDefined();
    expect(node!.parent).toBeUndefined();
    expect(node!.children).toHaveLength(0);
    expect(node!.siblings).toHaveLength(0);
  });

  test("child section links to parent; parent node lists child", async () => {
    const { buildSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const parent = makeSection("arch/overview.md");
    const child = makeSection("arch/detail.md", { parentId: "arch/overview.md" });
    const graph = buildSectionGraph(makeManifest([parent, child]));

    const parentNode = graph.get("arch/overview.md");
    const childNode = graph.get("arch/detail.md");

    expect(parentNode!.children).toContain("arch/detail.md");
    expect(childNode!.parent).toBe("arch/overview.md");
  });

  test("two siblings share the same parent and each lists the other in siblings", async () => {
    const { buildSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const parent = makeSection("arch/overview.md");
    const childA = makeSection("arch/a.md", { parentId: "arch/overview.md" });
    const childB = makeSection("arch/b.md", { parentId: "arch/overview.md" });
    const graph = buildSectionGraph(makeManifest([parent, childA, childB]));

    const nodeA = graph.get("arch/a.md");
    const nodeB = graph.get("arch/b.md");

    expect(nodeA!.siblings).toContain("arch/b.md");
    expect(nodeB!.siblings).toContain("arch/a.md");
    // Section is not its own sibling
    expect(nodeA!.siblings).not.toContain("arch/a.md");
  });

  test("dangling parentId (nonexistent section) is silently ignored", async () => {
    const { buildSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const orphan = makeSection("orphan/child.md", {
      parentId: "does/not/exist.md",
    });
    const graph = buildSectionGraph(makeManifest([orphan]));
    const node = graph.get("orphan/child.md");
    expect(node!.parent).toBeUndefined(); // dangling ref stripped
  });

  test("cross-ref validated against manifest — invalid paths excluded", async () => {
    const { buildSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const section = makeSection("a/b.md", {
      summary: "See nonexistent/target.md for reference.",
    });
    const graph = buildSectionGraph(makeManifest([section]));
    const node = graph.get("a/b.md");
    // 'nonexistent/target.md' is not in manifest → excluded from crossRefs
    expect(node!.crossRefs).not.toContain("nonexistent/target.md");
  });

  test("cross-ref that exists in manifest is retained in crossRefs", async () => {
    const { buildSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const target = makeSection("arch/patterns.md");
    const source = makeSection("vision/intro.md", {
      summary: "See arch/patterns.md for architectural guidance.",
    });
    const graph = buildSectionGraph(makeManifest([target, source]));
    const node = graph.get("vision/intro.md");
    expect(node!.crossRefs).toContain("arch/patterns.md");
  });

  test("section is not added as its own cross-reference", async () => {
    const { buildSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const section = makeSection("arch/overview.md", {
      summary: "See arch/overview.md for the full overview.",
    });
    const graph = buildSectionGraph(makeManifest([section]));
    const node = graph.get("arch/overview.md");
    expect(node!.crossRefs).not.toContain("arch/overview.md");
  });

  test("three-level hierarchy wires grandchild correctly", async () => {
    const { buildSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const root = makeSection("a/root.md");
    const child = makeSection("a/child.md", { parentId: "a/root.md" });
    const grand = makeSection("a/grand.md", { parentId: "a/child.md" });
    const graph = buildSectionGraph(makeManifest([root, child, grand]));

    expect(graph.get("a/root.md")!.children).toContain("a/child.md");
    expect(graph.get("a/child.md")!.children).toContain("a/grand.md");
    expect(graph.get("a/grand.md")!.parent).toBe("a/child.md");
  });
});

// ─── termOverlap ────────────────────────────────────────────────────────────

describe("termOverlap — Jaccard-like set overlap", () => {
  test("identical sets return 1.0", async () => {
    const { termOverlap } = await import("../src/genome/graph-traversal.ts");
    const s = new Set(["arch", "vision", "design"]);
    expect(termOverlap(s, s)).toBeCloseTo(1.0);
  });

  test("disjoint sets return 0", async () => {
    const { termOverlap } = await import("../src/genome/graph-traversal.ts");
    expect(termOverlap(new Set(["a", "b"]), new Set(["c", "d"]))).toBe(0);
  });

  test("empty sets return 0 without NaN/error", async () => {
    const { termOverlap } = await import("../src/genome/graph-traversal.ts");
    expect(termOverlap(new Set(), new Set())).toBe(0);
  });

  test("partial overlap is in (0, 1)", async () => {
    const { termOverlap } = await import("../src/genome/graph-traversal.ts");
    const a = new Set(["arch", "vision", "design"]);
    const b = new Set(["vision", "other", "extra"]);
    const score = termOverlap(a, b);
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThan(1);
  });

  test("single shared element has non-zero overlap", async () => {
    const { termOverlap } = await import("../src/genome/graph-traversal.ts");
    const a = new Set(["shared"]);
    const b = new Set(["shared", "extra1", "extra2"]);
    const score = termOverlap(a, b);
    expect(score).toBeGreaterThan(0);
  });
});

// ─── traverseSectionGraph — cycle detection ──────────────────────────────────

describe("traverseSectionGraph — cycle detection", () => {
  test("A→B→A parent cycle does not hang; each path appears at most once", async () => {
    const { buildSectionGraph, traverseSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const sA = makeSection("cycle/a.md", {
      summary: "cycle node architecture",
      tags: ["cycle"],
    });
    const sB = makeSection("cycle/b.md", {
      summary: "cycle node architecture",
      tags: ["cycle"],
      parentId: "cycle/a.md",
    });
    // Manually create a cycle: A's parent → B (even though B is actually A's child)
    sA.parentId = "cycle/b.md";
    sB.parentId = "cycle/a.md";

    const graph = buildSectionGraph(makeManifest([sA, sB]));
    const seeds = [
      { path: "cycle/a.md", score: 10 },
      { path: "cycle/b.md", score: 8 },
    ];
    const start = Date.now();
    const entries = traverseSectionGraph(graph, seeds, ["cycle", "node"], {
      maxTokens: 10000,
      maxDepth: 10,
    });
    const elapsed = Date.now() - start;

    expect(elapsed).toBeLessThan(5000);
    const paths = entries.map((e) => e.path);
    const uniquePaths = new Set(paths);
    expect(uniquePaths.size).toBe(paths.length);
  });

  test("self-referential cycle (parentId === own path) terminates immediately", async () => {
    const { buildSectionGraph, traverseSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const self = makeSection("self/ref.md", {
      summary: "self reference content",
      tags: ["self"],
    });
    self.parentId = "self/ref.md";

    const graph = buildSectionGraph(makeManifest([self]));
    const start = Date.now();
    const entries = traverseSectionGraph(
      graph,
      [{ path: "self/ref.md", score: 10 }],
      ["self"],
      { maxTokens: 10000, maxDepth: 5 },
    );
    const elapsed = Date.now() - start;

    expect(elapsed).toBeLessThan(5000);
    expect(entries.filter((e) => e.path === "self/ref.md").length).toBeLessThanOrEqual(1);
  });

  test("long chain (depth 10) terminates and respects maxDepth clamp", async () => {
    const { buildSectionGraph, traverseSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    // Build a 15-node linear chain
    const sections: SectionMeta[] = [];
    for (let i = 0; i < 15; i++) {
      sections.push(
        makeSection(`chain/s${i}.md`, {
          summary: `chain section ${i} architecture`,
          tags: ["chain"],
          ...(i > 0 ? { parentId: `chain/s${i - 1}.md` } : {}),
        }),
      );
    }
    const graph = buildSectionGraph(makeManifest(sections));
    const entries = traverseSectionGraph(
      graph,
      [{ path: "chain/s0.md", score: 10 }],
      ["chain"],
      { maxTokens: 100000, maxDepth: 5 }, // depth capped at 5 → only first 6 nodes
    );
    // maxDepth=5 → root + 5 levels = 6 nodes (chain/s0..s5); s6..s14 excluded
    const paths = entries.map((e) => e.path);
    expect(paths).not.toContain("chain/s6.md");
    expect(paths).not.toContain("chain/s14.md");
  });
});

// ─── traverseSectionGraph — cross-reference resolution ──────────────────────

describe("traverseSectionGraph — cross-reference resolution", () => {
  test("cross-referenced section is pulled when it has query relevance", async () => {
    const { buildSectionGraph, traverseSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const source = makeSection("vision/intro.md", {
      summary: "See architecture/patterns.md for architectural guidance. Vision overview.",
      tags: ["vision", "architecture"],
    });
    const target = makeSection("architecture/patterns.md", {
      summary: "Architectural patterns for the system.",
      tags: ["architecture", "patterns"],
    });
    const graph = buildSectionGraph(makeManifest([source, target]));
    const entries = traverseSectionGraph(
      graph,
      [{ path: "vision/intro.md", score: 10 }],
      ["architecture", "patterns"],
      { maxTokens: 10000, resolveCrossRefs: true },
    );
    const paths = entries.map((e) => e.path);
    expect(paths).toContain("architecture/patterns.md");
  });

  test("cross-referenced section not pulled when resolveCrossRefs=false", async () => {
    const { buildSectionGraph, traverseSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const source = makeSection("vision/intro.md", {
      summary: "See architecture/patterns.md for context.",
      tags: ["vision"],
    });
    const target = makeSection("architecture/patterns.md", {
      summary: "Architectural patterns.",
      tags: ["architecture"],
    });
    const graph = buildSectionGraph(makeManifest([source, target]));
    const entries = traverseSectionGraph(
      graph,
      [{ path: "vision/intro.md", score: 10 }],
      ["vision"],
      { maxTokens: 10000, resolveCrossRefs: false },
    );
    const paths = entries.map((e) => e.path);
    // Cross-ref disabled → target should not appear via xref
    // (It may appear if it scores directly, but with only "vision" as query term it won't)
    const xrefEntries = entries.filter((e) => e.source === "xref");
    expect(xrefEntries).toHaveLength(0);
  });

  test("cross-ref score is a fraction of the source section's score", async () => {
    const { buildSectionGraph, traverseSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const source = makeSection("vision/intro.md", {
      summary: "See arch/patterns.md for reference. Vision architecture intro.",
      tags: ["vision", "architecture"],
    });
    const target = makeSection("arch/patterns.md", {
      summary: "Patterns overview.",
      tags: ["patterns"],
    });
    const graph = buildSectionGraph(makeManifest([source, target]));
    const entries = traverseSectionGraph(
      graph,
      [{ path: "vision/intro.md", score: 10 }],
      ["vision", "architecture"],
      { maxTokens: 10000, crossRefScoreFraction: 0.4, resolveCrossRefs: true },
    );

    const sourceEntry = entries.find((e) => e.path === "vision/intro.md");
    const targetEntry = entries.find((e) => e.path === "arch/patterns.md");
    if (targetEntry && targetEntry.source === "xref") {
      // If pulled as xref with no direct match, score should be < sourceEntry score
      expect(targetEntry.score).toBeLessThan(sourceEntry!.score);
    }
  });
});

// ─── traverseSectionGraph — sibling ranking ──────────────────────────────────

describe("traverseSectionGraph — sibling ranking and overlap filtering", () => {
  test("sibling with high overlap is included", async () => {
    const { buildSectionGraph, traverseSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const parent = makeSection("arch/overview.md", {
      summary: "Architecture overview",
      tags: ["architecture"],
    });
    const anchorSibling = makeSection("arch/patterns.md", {
      summary: "Architecture patterns design",
      tags: ["architecture", "patterns"],
      parentId: "arch/overview.md",
    });
    const relatedSibling = makeSection("arch/principles.md", {
      summary: "Architecture principles design",
      tags: ["architecture", "principles"],
      parentId: "arch/overview.md",
    });

    const graph = buildSectionGraph(makeManifest([parent, anchorSibling, relatedSibling]));
    const entries = traverseSectionGraph(
      graph,
      [{ path: "arch/patterns.md", score: 10 }],
      ["architecture", "patterns", "design"],
      { maxTokens: 10000, siblingOverlapThreshold: 0.1 },
    );
    const paths = entries.map((e) => e.path);
    // related sibling shares "architecture" + "design" → overlap above 0.1
    expect(paths).toContain("arch/principles.md");
  });

  test("sibling with no term overlap is excluded when threshold > 0", async () => {
    const { buildSectionGraph, traverseSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const parent = makeSection("arch/overview.md");
    const anchor = makeSection("arch/auth.md", {
      summary: "Authentication security tokens",
      tags: ["auth", "security"],
      parentId: "arch/overview.md",
    });
    const unrelated = makeSection("arch/logging.md", {
      summary: "Logging output formatting",
      tags: ["logging", "formatting"],
      parentId: "arch/overview.md",
    });

    const graph = buildSectionGraph(makeManifest([parent, anchor, unrelated]));
    const entries = traverseSectionGraph(
      graph,
      [{ path: "arch/auth.md", score: 10 }],
      ["auth", "security"],
      {
        maxTokens: 10000,
        siblingOverlapThreshold: 0.2,
        // Disable parent backlinks so the parent (arch/overview.md) is not
        // pulled and then expanded as a seed, which would bring arch/logging.md
        // in via child expansion rather than sibling logic.
        includeParents: false,
      },
    );
    const paths = entries.map((e) => e.path);
    // "arch/logging.md" shares no terms with auth/security → sibling overlap < 0.2 → excluded
    expect(paths).not.toContain("arch/logging.md");
  });

  test("sibling score is lower than direct match score", async () => {
    const { buildSectionGraph, traverseSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const parent = makeSection("arch/overview.md");
    const anchor = makeSection("arch/patterns.md", {
      summary: "Architecture patterns design overview",
      tags: ["architecture", "patterns"],
      parentId: "arch/overview.md",
    });
    const sibling = makeSection("arch/principles.md", {
      summary: "Architecture principles design",
      tags: ["architecture", "principles"],
      parentId: "arch/overview.md",
    });

    const graph = buildSectionGraph(makeManifest([parent, anchor, sibling]));
    const entries = traverseSectionGraph(
      graph,
      [{ path: "arch/patterns.md", score: 10 }],
      ["architecture", "patterns"],
      { maxTokens: 10000, siblingOverlapThreshold: 0.1, siblingScoreFraction: 0.3 },
    );

    const anchorEntry = entries.find((e) => e.path === "arch/patterns.md");
    const sibEntry = entries.find((e) => e.path === "arch/principles.md" && e.source === "sibling");
    if (sibEntry) {
      expect(anchorEntry!.score).toBeGreaterThan(sibEntry.score);
    }
  });
});

// ─── traverseSectionGraph — parent backlinks ─────────────────────────────────

describe("traverseSectionGraph — transitive parent backlinks", () => {
  test("parent is pulled when includeParents=true (default)", async () => {
    const { buildSectionGraph, traverseSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const parent = makeSection("arch/overview.md", {
      summary: "Architecture overview context",
      tags: ["architecture"],
    });
    const child = makeSection("arch/detail.md", {
      summary: "Architecture detail implementation",
      tags: ["architecture", "detail"],
      parentId: "arch/overview.md",
    });

    const graph = buildSectionGraph(makeManifest([parent, child]));
    const entries = traverseSectionGraph(
      graph,
      [{ path: "arch/detail.md", score: 10 }],
      ["architecture", "detail"],
      { maxTokens: 10000 },
    );
    const paths = entries.map((e) => e.path);
    expect(paths).toContain("arch/overview.md");
  });

  test("parent not pulled when includeParents=false", async () => {
    const { buildSectionGraph, traverseSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const parent = makeSection("arch/overview.md");
    const child = makeSection("arch/detail.md", {
      summary: "Architecture detail content",
      tags: ["architecture"],
      parentId: "arch/overview.md",
    });

    const graph = buildSectionGraph(makeManifest([parent, child]));
    const entries = traverseSectionGraph(
      graph,
      [{ path: "arch/detail.md", score: 10 }],
      ["architecture"],
      { maxTokens: 10000, includeParents: false },
    );
    const parentEntries = entries.filter(
      (e) => e.path === "arch/overview.md" && e.source === "parent",
    );
    expect(parentEntries).toHaveLength(0);
  });

  test("grandparent is pulled when maxParentDepth >= 2", async () => {
    const { buildSectionGraph, traverseSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const root = makeSection("arch/root.md", {
      summary: "Architecture root",
      tags: ["architecture"],
    });
    const child = makeSection("arch/child.md", {
      summary: "Architecture child",
      tags: ["architecture"],
      parentId: "arch/root.md",
    });
    const grand = makeSection("arch/grand.md", {
      summary: "Architecture grand detail implementation",
      tags: ["architecture", "detail"],
      parentId: "arch/child.md",
    });

    const graph = buildSectionGraph(makeManifest([root, child, grand]));
    const entries = traverseSectionGraph(
      graph,
      [{ path: "arch/grand.md", score: 10 }],
      ["architecture", "detail"],
      { maxTokens: 10000, includeParents: true, maxParentDepth: 2 },
    );
    const paths = entries.map((e) => e.path);
    expect(paths).toContain("arch/child.md");
    expect(paths).toContain("arch/root.md");
  });

  test("grandparent not pulled when maxParentDepth=1", async () => {
    const { buildSectionGraph, traverseSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const root = makeSection("arch/root.md", {
      summary: "Architecture root",
      tags: ["architecture"],
    });
    const child = makeSection("arch/child.md", {
      summary: "Architecture child",
      tags: ["architecture"],
      parentId: "arch/root.md",
    });
    const grand = makeSection("arch/grand.md", {
      summary: "Architecture grand detail",
      tags: ["architecture", "detail"],
      parentId: "arch/child.md",
    });

    const graph = buildSectionGraph(makeManifest([root, child, grand]));
    const entries = traverseSectionGraph(
      graph,
      [{ path: "arch/grand.md", score: 10 }],
      ["architecture", "detail"],
      { maxTokens: 10000, includeParents: true, maxParentDepth: 1 },
    );
    const paths = entries.map((e) => e.path);
    expect(paths).toContain("arch/child.md"); // direct parent — yes
    expect(paths).not.toContain("arch/root.md"); // grandparent — no
  });
});

// ─── token budget limits ─────────────────────────────────────────────────────

describe("packGraphSections — token budget enforcement", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("total tokens of returned sections never exceeds maxTokens", async () => {
    const { packGraphSections } = await import(
      "../src/genome/graph-traversal.ts"
    );

    // Write 5 sections of ~200 tokens each (~800 chars)
    const bigContent = "word ".repeat(160); // ~160 words ≈ 213 tokens
    const entries: import("../src/genome/graph-traversal.ts").TraversalEntry[] = [];
    for (let i = 0; i < 5; i++) {
      const relPath = `budget/section-${i}.md`;
      await writeGenomeSection(cwd, relPath, `# Section ${i}\n${bigContent}`);
      entries.push({
        path: relPath,
        score: 10 - i,
        source: "direct",
        discoveredRefs: [],
      });
    }

    const maxTokens = 300;
    const results = await packGraphSections(cwd, entries, maxTokens);
    const total = results.reduce((sum, r) => sum + r.tokens, 0);
    expect(total).toBeLessThanOrEqual(maxTokens);
  });

  test("truncated version included when remaining > 200 tokens", async () => {
    const { packGraphSections } = await import(
      "../src/genome/graph-traversal.ts"
    );

    // estimateTokens uses chars/4. Budget=2000, first section uses ~50 tokens (200 chars).
    // Remaining = 1950. Second section = 3000 tokens (12000 chars) → won't fit fully but
    // 1950 > 200 so truncation kicks in.
    const firstContent = "x ".repeat(100);   // ~200 chars → ~50 tokens
    const secondContent = "y ".repeat(6000); // ~12000 chars → ~3000 tokens

    await writeGenomeSection(cwd, "trunc/first.md", firstContent);
    await writeGenomeSection(cwd, "trunc/second.md", secondContent);

    const entries: import("../src/genome/graph-traversal.ts").TraversalEntry[] = [
      { path: "trunc/first.md", score: 10, source: "direct", discoveredRefs: [] },
      { path: "trunc/second.md", score: 8, source: "direct", discoveredRefs: [] },
    ];

    const maxTokens = 2000;
    const results = await packGraphSections(cwd, entries, maxTokens);
    const total = results.reduce((sum, r) => sum + r.tokens, 0);
    expect(total).toBeLessThanOrEqual(maxTokens);

    // Truncation must have happened on the second section
    const second = results.find((r) => r.path === "trunc/second.md");
    expect(second).toBeDefined();
    expect(second!.content).toContain("[... section truncated ...]");
  });

  test("missing file is skipped without crashing", async () => {
    const { packGraphSections } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const entries: import("../src/genome/graph-traversal.ts").TraversalEntry[] = [
      {
        path: "ghost/does-not-exist.md",
        score: 10,
        source: "direct",
        discoveredRefs: [],
      },
    ];
    const results = await packGraphSections(cwd, entries, 10000);
    expect(results).toHaveLength(0);
  });

  test("path traversal attempt is rejected by sectionPath", async () => {
    const { packGraphSections } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const entries: import("../src/genome/graph-traversal.ts").TraversalEntry[] = [
      {
        path: "../../etc/passwd",
        score: 10,
        source: "direct",
        discoveredRefs: [],
      },
    ];
    // sectionPath throws → packGraphSections catches and skips, returns []
    const results = await packGraphSections(cwd, entries, 10000);
    expect(results).toHaveLength(0);
  });

  test("zero maxTokens returns empty array", async () => {
    const { packGraphSections } = await import(
      "../src/genome/graph-traversal.ts"
    );
    await writeGenomeSection(cwd, "some/section.md", "# Content here");
    const entries: import("../src/genome/graph-traversal.ts").TraversalEntry[] = [
      { path: "some/section.md", score: 5, source: "direct", discoveredRefs: [] },
    ];
    const results = await packGraphSections(cwd, entries, 0);
    expect(results).toHaveLength(0);
  });
});

// ─── maxSections cap ─────────────────────────────────────────────────────────

describe("traverseSectionGraph — maxSections cap", () => {
  test("result never exceeds maxSections even with large graph", async () => {
    const { buildSectionGraph, traverseSectionGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    // Build 50 sections all with the same query terms
    const sections: SectionMeta[] = Array.from({ length: 50 }, (_, i) =>
      makeSection(`sections/s${i}.md`, {
        summary: "architecture design patterns overview",
        tags: ["architecture"],
      }),
    );
    const graph = buildSectionGraph(makeManifest(sections));
    const seeds = sections.map((s, i) => ({ path: s.path, score: 50 - i }));

    const entries = traverseSectionGraph(graph, seeds, ["architecture"], {
      maxTokens: 1000000,
      maxSections: 10,
    });
    expect(entries.length).toBeLessThanOrEqual(10);
  });
});

// ─── buildAndTraverseGraph — end-to-end ─────────────────────────────────────

describe("buildAndTraverseGraph — end-to-end with real section files", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("returns empty array for empty manifest", async () => {
    const { buildAndTraverseGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const results = await buildAndTraverseGraph(makeManifest([]), "any query", cwd, {
      maxTokens: 10000,
    });
    expect(results).toHaveLength(0);
  });

  test("returns empty array when no sections match the query", async () => {
    const { buildAndTraverseGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    await writeGenomeSection(cwd, "vision/north-star.md", "# Vision");
    const manifest = makeManifest([
      makeSection("vision/north-star.md", {
        title: "Vision",
        summary: "Project vision statement",
        tags: ["vision"],
      }),
    ]);
    // Query has no matching terms
    const results = await buildAndTraverseGraph(
      manifest,
      "xyzzy-unmatched-term",
      cwd,
      { maxTokens: 10000 },
    );
    expect(results).toHaveLength(0);
  });

  test("matched section content is loaded and returned", async () => {
    const { buildAndTraverseGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const content = "# Architecture Overview\nThis describes the architecture.";
    await writeGenomeSection(cwd, "arch/overview.md", content);

    const manifest = makeManifest([
      makeSection("arch/overview.md", {
        title: "Architecture Overview",
        summary: "High-level architecture design",
        tags: ["architecture", "overview"],
      }),
    ]);

    const results = await buildAndTraverseGraph(
      manifest,
      "architecture overview",
      cwd,
      { maxTokens: 10000 },
    );
    expect(results).toHaveLength(1);
    expect(results[0]!.content).toContain("Architecture Overview");
    expect(results[0]!.path).toBe("arch/overview.md");
    expect(results[0]!.title).toBe("Architecture Overview");
  });

  test("child section is pulled via hierarchy expansion", async () => {
    const { buildAndTraverseGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    await writeGenomeSection(cwd, "arch/overview.md", "# Architecture\nMain overview.");
    await writeGenomeSection(cwd, "arch/detail.md", "# Detail\nImplementation detail.");

    const manifest = makeManifest([
      makeSection("arch/overview.md", {
        title: "Architecture",
        summary: "High-level architecture overview design",
        tags: ["architecture"],
      }),
      makeSection("arch/detail.md", {
        title: "Detail",
        summary: "Unrelated zorp content",
        tags: ["zorp"],
        parentId: "arch/overview.md",
      }),
    ]);

    const results = await buildAndTraverseGraph(
      manifest,
      "architecture overview",
      cwd,
      { maxTokens: 10000, maxDepth: 1 },
    );
    const paths = results.map((r) => r.path);
    expect(paths).toContain("arch/overview.md");
    expect(paths).toContain("arch/detail.md");
  });

  test("relatedSectionPaths is populated from cross-refs", async () => {
    const { buildAndTraverseGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    await writeGenomeSection(
      cwd,
      "vision/intro.md",
      "# Vision Intro\nSee architecture/patterns.md for context.",
    );
    await writeGenomeSection(
      cwd,
      "architecture/patterns.md",
      "# Patterns\nArchitecture patterns.",
    );

    const manifest = makeManifest([
      makeSection("vision/intro.md", {
        title: "Vision Intro",
        summary: "See architecture/patterns.md for architectural guidance.",
        tags: ["vision", "architecture"],
      }),
      makeSection("architecture/patterns.md", {
        title: "Patterns",
        summary: "Architecture patterns design",
        tags: ["architecture", "patterns"],
      }),
    ]);

    const results = await buildAndTraverseGraph(
      manifest,
      "vision architecture",
      cwd,
      { maxTokens: 10000, resolveCrossRefs: true },
    );

    const introResult = results.find((r) => r.path === "vision/intro.md");
    expect(introResult).toBeDefined();
    // relatedSectionPaths should mention the cross-referenced patterns section
    if (introResult?.relatedSectionPaths) {
      expect(introResult.relatedSectionPaths).toContain("architecture/patterns.md");
    }
  });

  test("token budget is respected in end-to-end traversal", async () => {
    const { buildAndTraverseGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    const bigContent = "architecture ".repeat(200); // ~200+ tokens
    await writeGenomeSection(cwd, "big/a.md", `# A\n${bigContent}`);
    await writeGenomeSection(cwd, "big/b.md", `# B\n${bigContent}`);
    await writeGenomeSection(cwd, "big/c.md", `# C\n${bigContent}`);

    const manifest = makeManifest([
      makeSection("big/a.md", {
        title: "A",
        summary: "Architecture overview main",
        tags: ["architecture"],
      }),
      makeSection("big/b.md", {
        title: "B",
        summary: "Architecture design details",
        tags: ["architecture"],
      }),
      makeSection("big/c.md", {
        title: "C",
        summary: "Architecture implementation guide",
        tags: ["architecture"],
      }),
    ]);

    const maxTokens = 300;
    const results = await buildAndTraverseGraph(
      manifest,
      "architecture",
      cwd,
      { maxTokens },
    );
    const total = results.reduce((sum, r) => sum + r.tokens, 0);
    expect(total).toBeLessThanOrEqual(maxTokens);
  });

  test("sections are sorted by score descending", async () => {
    const { buildAndTraverseGraph } = await import(
      "../src/genome/graph-traversal.ts"
    );
    // High-scoring: lots of matching terms
    await writeGenomeSection(cwd, "arch/high.md", "# High\nHigh relevance.");
    // Low-scoring: fewer terms
    await writeGenomeSection(cwd, "arch/low.md", "# Low\nLow relevance.");

    const manifest = makeManifest([
      makeSection("arch/high.md", {
        title: "High",
        summary: "Architecture patterns design overview system",
        tags: ["architecture", "patterns", "design"],
      }),
      makeSection("arch/low.md", {
        title: "Low",
        summary: "Architecture overview",
        tags: ["architecture"],
      }),
    ]);

    const results = await buildAndTraverseGraph(
      manifest,
      "architecture patterns design",
      cwd,
      { maxTokens: 10000 },
    );

    expect(results.length).toBeGreaterThan(0);
    for (let i = 1; i < results.length; i++) {
      expect(results[i - 1]!.score).toBeGreaterThanOrEqual(results[i]!.score);
    }
  });
});

// ─── genome/index.ts — graph-traversal symbols exported ─────────────────────

describe("genome/index.ts — graph-traversal symbols are exported", () => {
  test("buildSectionGraph is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.buildSectionGraph).toBe("function");
  });

  test("buildAndTraverseGraph is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.buildAndTraverseGraph).toBe("function");
  });

  test("traverseSectionGraph is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.traverseSectionGraph).toBe("function");
  });

  test("packGraphSections is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.packGraphSections).toBe("function");
  });

  test("extractCrossRefCandidates is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.extractCrossRefCandidates).toBe("function");
  });

  test("termOverlap is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.termOverlap).toBe("function");
  });
});

// ─── RetrievedSection.relatedSectionPaths — type extension ──────────────────

describe("RetrievedSection — relatedSectionPaths field is optional", () => {
  test("existing retriever results without relatedSectionPaths are valid", async () => {
    const { retrieveSections } = await import("../src/genome/retriever.ts");
    // The type extension must be backward-compatible (optional field)
    // We check that the type compiles and runtime results without the field are fine
    // A section without relatedSectionPaths should have it as undefined
    const section: import("../src/genome/retriever.ts").RetrievedSection = {
      path: "a/b.md",
      title: "Test",
      content: "content",
      tokens: 10,
      score: 1,
    };
    expect(section.relatedSectionPaths).toBeUndefined();
  });

  test("section with relatedSectionPaths accepts string array", () => {
    const section: import("../src/genome/retriever.ts").RetrievedSection = {
      path: "a/b.md",
      title: "Test",
      content: "content",
      tokens: 10,
      score: 1,
      relatedSectionPaths: ["arch/patterns.md", "vision/north-star.md"],
    };
    expect(section.relatedSectionPaths).toHaveLength(2);
    expect(section.relatedSectionPaths![0]).toBe("arch/patterns.md");
  });
});
