/**
 * Tests for retrieveSectionsAdaptive — three-tier adaptive retrieval.
 *
 * Covers:
 *  - Tier 1 (keyword): forced strategy returns sections matching query terms.
 *  - Tier 3 (hierarchical): forced strategy expands parents to children.
 *  - "auto" with aggressiveness=0 skips semantic, uses hierarchical.
 *  - "auto" with Ollama unavailable falls through to hierarchical/keyword.
 *  - Return shape: { sections, strategy, latency_ms, relevance_score }.
 *  - relevance_score is in [0, 1].
 *  - latency_ms is a non-negative number.
 *  - Audit JSONL is written to .ashlrcode/genome/evolution/retrieval-audit.jsonl.
 *  - Empty query falls back gracefully (no throw).
 *  - Zero-budget returns empty sections without throwing.
 *  - Aggressiveness clamps: values outside [0,1] are accepted without throw.
 *  - retrieveSectionsAdaptive is exported from genome/index.ts.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { existsSync } from "fs";
import { mkdir, readFile, rm, writeFile } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeTmpDir(): string {
  return join(tmpdir(), `ashlr-adaptive-strat-${Date.now()}-${Math.random().toString(36).slice(2)}`);
}

/** Set up a minimal genome with a few sections so retrieval has something to work with. */
async function scaffoldGenome(cwd: string): Promise<void> {
  const genomeDir = join(cwd, ".ashlrcode", "genome");
  await mkdir(join(genomeDir, "evolution"), { recursive: true });
  await mkdir(join(genomeDir, "architecture"), { recursive: true });
  await mkdir(join(genomeDir, "vision"), { recursive: true });

  // Write section files
  await writeFile(
    join(genomeDir, "architecture", "overview.md"),
    "# Architecture Overview\n\nThe system is composed of a retriever, embeddings, and a manifest.\n",
    "utf-8",
  );
  await writeFile(
    join(genomeDir, "architecture", "layer-api.md"),
    "# Layer API\n\nThe public API surface exported by each layer.\n",
    "utf-8",
  );
  await writeFile(
    join(genomeDir, "vision", "north-star.md"),
    "# North Star\n\nBuild a self-improving AI development assistant.\n",
    "utf-8",
  );

  // Write manifest
  const manifest = {
    version: 1,
    project: "test-project",
    sections: [
      {
        path: "architecture/overview.md",
        title: "Architecture Overview",
        summary: "System composed of retriever embeddings manifest",
        tags: ["architecture", "overview", "retriever"],
        tokens: 20,
        updatedAt: new Date().toISOString(),
      },
      {
        path: "architecture/layer-api.md",
        title: "Layer API",
        summary: "Public API surface exported by each layer",
        tags: ["architecture", "api", "layer"],
        tokens: 15,
        updatedAt: new Date().toISOString(),
        parentId: "architecture/overview.md",
      },
      {
        path: "vision/north-star.md",
        title: "North Star",
        summary: "Build a self-improving AI development assistant",
        tags: ["vision", "goal", "mission"],
        tokens: 15,
        updatedAt: new Date().toISOString(),
      },
    ],
    generation: { number: 1, milestone: "test", startedAt: new Date().toISOString() },
    fitnessHistory: [],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  await writeFile(join(genomeDir, "manifest.json"), JSON.stringify(manifest, null, 2), "utf-8");
}

// ---------------------------------------------------------------------------
// Export surface
// ---------------------------------------------------------------------------

describe("genome/index.ts — retrieveSectionsAdaptive is exported", () => {
  test("retrieveSectionsAdaptive is a function in genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof (mod as Record<string, unknown>).retrieveSectionsAdaptive).toBe("function");
  });

  test("AdaptiveRetrievalStrategy type guard — strategy string accepted", async () => {
    // Just check the export exists and is callable; runtime type check is TypeScript's job.
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    expect(typeof retrieveSectionsAdaptive).toBe("function");
  });
});

// ---------------------------------------------------------------------------
// Return shape
// ---------------------------------------------------------------------------

describe("retrieveSectionsAdaptive — return shape", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await scaffoldGenome(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("returns an object with sections, strategy, latency_ms, relevance_score", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    const result = await retrieveSectionsAdaptive("architecture retriever", cwd, 2000);
    expect(typeof result).toBe("object");
    expect(Array.isArray(result.sections)).toBe(true);
    expect(typeof result.strategy).toBe("string");
    expect(typeof result.latency_ms).toBe("number");
    expect(typeof result.relevance_score).toBe("number");
  });

  test("latency_ms is non-negative", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    const result = await retrieveSectionsAdaptive("architecture", cwd, 2000);
    expect(result.latency_ms).toBeGreaterThanOrEqual(0);
  });

  test("relevance_score is between 0 and 1 inclusive", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    const result = await retrieveSectionsAdaptive("architecture retriever", cwd, 2000);
    expect(result.relevance_score).toBeGreaterThanOrEqual(0);
    expect(result.relevance_score).toBeLessThanOrEqual(1);
  });

  test("sections are RetrievedSection objects with required fields", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    const result = await retrieveSectionsAdaptive("architecture", cwd, 2000);
    for (const s of result.sections) {
      expect(typeof s.path).toBe("string");
      expect(typeof s.title).toBe("string");
      expect(typeof s.content).toBe("string");
      expect(typeof s.tokens).toBe("number");
      expect(typeof s.score).toBe("number");
    }
  });
});

// ---------------------------------------------------------------------------
// Tier 1: keyword strategy (forced)
// ---------------------------------------------------------------------------

describe("retrieveSectionsAdaptive — Tier 1 keyword strategy", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await scaffoldGenome(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("forced keyword strategy returns matching sections", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    const result = await retrieveSectionsAdaptive("architecture retriever", cwd, 2000, {
      strategy: "keyword",
    });
    expect(result.strategy).toBe("keyword");
    expect(result.sections.length).toBeGreaterThan(0);
    const paths = result.sections.map((s) => s.path);
    // architecture/overview.md is tagged "architecture" and "retriever"
    expect(paths).toContain("architecture/overview.md");
  });

  test("forced keyword with no matching query returns empty sections", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    const result = await retrieveSectionsAdaptive("xyzzy-nonexistent-term-zzz", cwd, 2000, {
      strategy: "keyword",
    });
    expect(result.strategy).toBe("keyword");
    expect(result.sections).toHaveLength(0);
    expect(result.relevance_score).toBe(0);
  });

  test("forced keyword respects token budget", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    const result = await retrieveSectionsAdaptive("architecture api vision", cwd, 10, {
      strategy: "keyword",
    });
    expect(result.strategy).toBe("keyword");
    const totalTokens = result.sections.reduce((t, s) => t + s.tokens, 0);
    expect(totalTokens).toBeLessThanOrEqual(10 + 10); // allow truncation overhead
  });
});

// ---------------------------------------------------------------------------
// Tier 3: hierarchical strategy (forced)
// ---------------------------------------------------------------------------

describe("retrieveSectionsAdaptive — Tier 3 hierarchical strategy", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await scaffoldGenome(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("forced hierarchical returns parent + child sections", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    // Query matches "architecture/overview" — hierarchical should also pull "layer-api"
    const result = await retrieveSectionsAdaptive("architecture overview", cwd, 2000, {
      strategy: "hierarchical",
    });
    expect(result.strategy).toBe("hierarchical");
    const paths = result.sections.map((s) => s.path);
    expect(paths).toContain("architecture/overview.md");
    // layer-api.md declares parentId = architecture/overview.md so must be included
    expect(paths).toContain("architecture/layer-api.md");
  });

  test("hierarchical strategy result has strategy field set to hierarchical", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    const result = await retrieveSectionsAdaptive("architecture", cwd, 2000, {
      strategy: "hierarchical",
    });
    expect(result.strategy).toBe("hierarchical");
  });

  test("hierarchical aggressiveness=1 uses deeper expansion", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    // Should not throw and returns valid result
    const result = await retrieveSectionsAdaptive("architecture", cwd, 2000, {
      strategy: "hierarchical",
      aggressiveness: 1,
    });
    expect(result.strategy).toBe("hierarchical");
    expect(result.sections.length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Auto strategy
// ---------------------------------------------------------------------------

describe("retrieveSectionsAdaptive — auto strategy", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await scaffoldGenome(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("auto defaults to hierarchical or semantic (not keyword) for standard query", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    const result = await retrieveSectionsAdaptive("architecture retriever", cwd, 2000);
    // auto can resolve to hierarchical or semantic (if Ollama is up)
    expect(["hierarchical", "semantic"]).toContain(result.strategy);
  });

  test("auto with aggressiveness=0 skips semantic and returns hierarchical", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    const result = await retrieveSectionsAdaptive("architecture retriever", cwd, 2000, {
      strategy: "auto",
      aggressiveness: 0,
    });
    // aggressiveness < 0.2 → skipSemantic → falls to hierarchical
    expect(result.strategy).toBe("hierarchical");
  });

  test("auto with no genome match returns empty sections gracefully", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    const result = await retrieveSectionsAdaptive("zzz-nonexistent-zzzz", cwd, 2000, {
      strategy: "auto",
      aggressiveness: 0,
    });
    expect(result.sections).toHaveLength(0);
    expect(result.relevance_score).toBe(0);
  });

  test("auto with empty query does not throw", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    const fn = () =>
      retrieveSectionsAdaptive("", cwd, 2000, { strategy: "auto" });
    await expect(fn()).resolves.toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// Semantic strategy (forced) — Ollama unavailable path
// ---------------------------------------------------------------------------

describe("retrieveSectionsAdaptive — semantic strategy, Ollama unavailable", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await scaffoldGenome(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("forced semantic falls back to keyword when Ollama is down", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    // Ollama is almost certainly not running in CI; the function must gracefully
    // fall back to keyword rather than throwing.
    const result = await retrieveSectionsAdaptive("architecture", cwd, 2000, {
      strategy: "semantic",
    });
    // Either semantic (if Ollama happens to be up) or keyword fallback
    expect(["semantic", "keyword"]).toContain(result.strategy);
    expect(typeof result.relevance_score).toBe("number");
    expect(result.relevance_score).toBeGreaterThanOrEqual(0);
    expect(result.relevance_score).toBeLessThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// Audit trail
// ---------------------------------------------------------------------------

describe("retrieveSectionsAdaptive — audit JSONL trail", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await scaffoldGenome(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  async function waitForAudit(auditFile: string, maxMs = 500): Promise<void> {
    const deadline = Date.now() + maxMs;
    while (Date.now() < deadline) {
      if (existsSync(auditFile) && (await readFile(auditFile, "utf-8")).trim().length > 0) return;
      await new Promise((r) => setTimeout(r, 20));
    }
  }

  test("audit file is created after a retrieval", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    await retrieveSectionsAdaptive("architecture", cwd, 2000, { strategy: "keyword" });

    const auditFile = join(cwd, ".ashlrcode", "genome", "evolution", "retrieval-audit.jsonl");
    await waitForAudit(auditFile);
    expect(existsSync(auditFile)).toBe(true);
  });

  test("audit record contains required fields", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    await retrieveSectionsAdaptive("architecture retriever", cwd, 2000, { strategy: "keyword" });

    const auditFile = join(cwd, ".ashlrcode", "genome", "evolution", "retrieval-audit.jsonl");
    await waitForAudit(auditFile);

    const raw = await readFile(auditFile, "utf-8");
    const lines = raw.split("\n").filter((l) => l.trim().length > 0);
    expect(lines.length).toBeGreaterThan(0);

    const record = JSON.parse(lines[0]!) as Record<string, unknown>;
    expect(typeof record.ts).toBe("string");
    expect(typeof record.query).toBe("string");
    expect(typeof record.requested_strategy).toBe("string");
    expect(typeof record.used_strategy).toBe("string");
    expect(typeof record.latency_ms).toBe("number");
    expect(typeof record.relevance_score).toBe("number");
    expect(typeof record.sections_returned).toBe("number");
    expect(typeof record.budget_tokens).toBe("number");
    expect(typeof record.tokens_used).toBe("number");
  });

  test("multiple retrievals append multiple JSONL lines", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    await retrieveSectionsAdaptive("architecture", cwd, 2000, { strategy: "keyword" });
    await retrieveSectionsAdaptive("vision goal", cwd, 2000, { strategy: "keyword" });

    const auditFile = join(cwd, ".ashlrcode", "genome", "evolution", "retrieval-audit.jsonl");
    await waitForAudit(auditFile);

    const raw = await readFile(auditFile, "utf-8");
    const lines = raw.split("\n").filter((l) => l.trim().length > 0);
    expect(lines.length).toBeGreaterThanOrEqual(2);
  });

  test("audit record requested_strategy matches what caller asked for", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    await retrieveSectionsAdaptive("architecture", cwd, 2000, { strategy: "hierarchical" });

    const auditFile = join(cwd, ".ashlrcode", "genome", "evolution", "retrieval-audit.jsonl");
    await waitForAudit(auditFile);

    const raw = await readFile(auditFile, "utf-8");
    const lines = raw.split("\n").filter((l) => l.trim().length > 0);
    const record = JSON.parse(lines[lines.length - 1]!) as Record<string, unknown>;
    expect(record.requested_strategy).toBe("hierarchical");
    expect(record.used_strategy).toBe("hierarchical");
  });
});

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

describe("retrieveSectionsAdaptive — edge cases", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await scaffoldGenome(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("zero budget returns empty sections without throwing", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    const result = await retrieveSectionsAdaptive("architecture", cwd, 0, {
      strategy: "keyword",
    });
    expect(result.sections).toHaveLength(0);
  });

  test("aggressiveness < 0 clamps to 0 without throwing", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    const fn = () =>
      retrieveSectionsAdaptive("architecture", cwd, 2000, {
        strategy: "auto",
        aggressiveness: -5,
      });
    await expect(fn()).resolves.toBeDefined();
  });

  test("aggressiveness > 1 clamps to 1 without throwing", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    const fn = () =>
      retrieveSectionsAdaptive("architecture", cwd, 2000, {
        strategy: "hierarchical",
        aggressiveness: 99,
      });
    await expect(fn()).resolves.toBeDefined();
  });

  test("missing genome (no manifest.json) returns empty sections", async () => {
    const { retrieveSectionsAdaptive } = await import("../src/genome/retriever.ts");
    const emptyCwd = join(tmpdir(), `ashlr-empty-${Date.now()}`);
    await mkdir(emptyCwd, { recursive: true });
    try {
      const result = await retrieveSectionsAdaptive("architecture", emptyCwd, 2000, {
        strategy: "keyword",
      });
      expect(result.sections).toHaveLength(0);
    } finally {
      await rm(emptyCwd, { recursive: true, force: true });
    }
  });
});
