/**
 * Adaptive retrieval tests — A/B testing & adaptive strategy selection.
 *
 * Covers:
 *  - classifyQuery: rare-term / semantic / general heuristics
 *  - recordRetrievalOutcome: JSONL persistence round-trip
 *  - AdaptiveRetriever.selectBestStrategy:
 *      * keyword preferred for rare-term queries with no prior data
 *      * semantic preferred for semantic queries with no prior data
 *      * learned preference from seeded outcome records
 *      * latency-pressure fallback forces keyword when semantic is slow
 *  - createAdaptiveRetriever: convenience factory
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, rm } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";

// ─── helpers ────────────────────────────────────────────────────────────────

function makeTmpDir(): string {
  return join(tmpdir(), `ashlr-adaptive-${Date.now()}-${Math.random().toString(36).slice(2)}`);
}

async function setupGenomeDir(cwd: string): Promise<void> {
  await mkdir(join(cwd, ".ashlrcode", "genome", "evolution"), { recursive: true });
}

/** Build a minimal StrategyMetrics object. */
function makeMetrics(
  overrides: Partial<import("../src/genome/retrieval-adapter.ts").StrategyMetrics> = {},
): import("../src/genome/retrieval-adapter.ts").StrategyMetrics {
  return {
    latencyMs: 100,
    sectionsReturned: 3,
    totalRelevanceScore: 2.5,
    budgetFit: true,
    matchQuality: 0.8,
    ...overrides,
  };
}

// ─── classifyQuery ──────────────────────────────────────────────────────────

describe("classifyQuery — heuristic query type detection", () => {
  test("camelCase identifier is classified as rare-term", async () => {
    const { classifyQuery } = await import("../src/genome/retrieval-adapter.ts");
    expect(classifyQuery("retrieveSections")).toBe("rare-term");
  });

  test("snake_case identifier is classified as rare-term", async () => {
    const { classifyQuery } = await import("../src/genome/retrieval-adapter.ts");
    expect(classifyQuery("genome_manifest_path")).toBe("rare-term");
  });

  test("file extension token is classified as rare-term", async () => {
    const { classifyQuery } = await import("../src/genome/retrieval-adapter.ts");
    expect(classifyQuery("embeddings.ts update logic")).toBe("rare-term");
  });

  test("natural-language question is classified as semantic", async () => {
    const { classifyQuery } = await import("../src/genome/retrieval-adapter.ts");
    expect(classifyQuery("how does the genome retrieval system find sections")).toBe("semantic");
  });

  test("describe-style phrase is classified as semantic", async () => {
    const { classifyQuery } = await import("../src/genome/retrieval-adapter.ts");
    expect(classifyQuery("what should the vision for the project be")).toBe("semantic");
  });

  test("very short query is classified as general", async () => {
    const { classifyQuery } = await import("../src/genome/retrieval-adapter.ts");
    expect(classifyQuery("embed")).toBe("general");
  });

  test("empty query is classified as general", async () => {
    const { classifyQuery } = await import("../src/genome/retrieval-adapter.ts");
    expect(classifyQuery("")).toBe("general");
  });

  test("whitespace-only query is classified as general", async () => {
    const { classifyQuery } = await import("../src/genome/retrieval-adapter.ts");
    expect(classifyQuery("   ")).toBe("general");
  });

  test("query with brackets is classified as rare-term", async () => {
    const { classifyQuery } = await import("../src/genome/retrieval-adapter.ts");
    expect(classifyQuery("SectionMeta[]")).toBe("rare-term");
  });
});

// ─── recordRetrievalOutcome ─────────────────────────────────────────────────

describe("recordRetrievalOutcome — JSONL persistence", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("returns a non-empty id string", async () => {
    const { recordRetrievalOutcome } = await import("../src/genome/retrieval-adapter.ts");
    const id = await recordRetrievalOutcome(cwd, "test query", "keyword", makeMetrics());
    expect(typeof id).toBe("string");
    expect(id.length).toBeGreaterThan(0);
  });

  test("persisted record is readable via loadRetrievalOutcomes", async () => {
    const { recordRetrievalOutcome, loadRetrievalOutcomes } = await import(
      "../src/genome/retrieval-adapter.ts"
    );
    await recordRetrievalOutcome(cwd, "search for embeddings", "semantic", makeMetrics({ matchQuality: 0.9 }));
    const records = await loadRetrievalOutcomes(cwd);
    expect(records).toHaveLength(1);
    expect(records[0]!.strategy).toBe("semantic");
    expect(records[0]!.metrics.matchQuality).toBe(0.9);
    expect(records[0]!.query).toBe("search for embeddings");
  });

  test("multiple records accumulate in order", async () => {
    const { recordRetrievalOutcome, loadRetrievalOutcomes } = await import(
      "../src/genome/retrieval-adapter.ts"
    );
    await recordRetrievalOutcome(cwd, "query A", "keyword", makeMetrics());
    await recordRetrievalOutcome(cwd, "query B", "semantic", makeMetrics());
    await recordRetrievalOutcome(cwd, "query C", "keyword", makeMetrics());
    const records = await loadRetrievalOutcomes(cwd);
    expect(records).toHaveLength(3);
    expect(records[0]!.query).toBe("query A");
    expect(records[1]!.strategy).toBe("semantic");
    expect(records[2]!.query).toBe("query C");
  });

  test("record includes correct queryType classification", async () => {
    const { recordRetrievalOutcome, loadRetrievalOutcomes } = await import(
      "../src/genome/retrieval-adapter.ts"
    );
    await recordRetrievalOutcome(cwd, "retrieveSections function body", "keyword", makeMetrics());
    const records = await loadRetrievalOutcomes(cwd);
    expect(records[0]!.queryType).toBe("rare-term");
  });

  test("loadRetrievalOutcomes returns empty array when no file exists", async () => {
    const { loadRetrievalOutcomes } = await import("../src/genome/retrieval-adapter.ts");
    const records = await loadRetrievalOutcomes(cwd);
    expect(records).toEqual([]);
  });
});

// ─── AdaptiveRetriever.selectBestStrategy — heuristic (no prior data) ───────

describe("AdaptiveRetriever.selectBestStrategy — heuristic defaults (no prior outcomes)", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("rare-term query selects keyword strategy when no data", async () => {
    const { AdaptiveRetriever } = await import("../src/genome/retrieval-adapter.ts");
    const retriever = new AdaptiveRetriever(cwd);
    // camelCase identifier → rare-term → keyword
    const strategy = await retriever.selectBestStrategy("retrieveSectionsV2");
    expect(strategy).toBe("keyword");
  });

  test("snake_case rare-term query selects keyword strategy", async () => {
    const { AdaptiveRetriever } = await import("../src/genome/retrieval-adapter.ts");
    const retriever = new AdaptiveRetriever(cwd);
    const strategy = await retriever.selectBestStrategy("genome_dir_path");
    expect(strategy).toBe("keyword");
  });

  test("semantic natural-language query selects semantic strategy when no data", async () => {
    const { AdaptiveRetriever } = await import("../src/genome/retrieval-adapter.ts");
    const retriever = new AdaptiveRetriever(cwd);
    const strategy = await retriever.selectBestStrategy(
      "what should the project vision describe for the roadmap",
    );
    expect(strategy).toBe("semantic");
  });

  test("general short query selects keyword strategy when no data", async () => {
    const { AdaptiveRetriever } = await import("../src/genome/retrieval-adapter.ts");
    const retriever = new AdaptiveRetriever(cwd);
    const strategy = await retriever.selectBestStrategy("embed");
    expect(strategy).toBe("keyword");
  });
});

// ─── AdaptiveRetriever.selectBestStrategy — learned preference ───────────────

describe("AdaptiveRetriever.selectBestStrategy — learned preference from outcomes", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  /** Seed N outcome records for a given strategy with the given matchQuality. */
  async function seedOutcomes(
    strategy: "keyword" | "semantic",
    count: number,
    matchQuality: number,
    latencyMs = 100,
  ): Promise<void> {
    const { recordRetrievalOutcome } = await import("../src/genome/retrieval-adapter.ts");
    for (let i = 0; i < count; i++) {
      await recordRetrievalOutcome(
        cwd,
        `how does the system find relevant sections for queries iteration ${i}`,
        strategy,
        makeMetrics({ matchQuality, latencyMs }),
      );
    }
  }

  test("keyword preferred when keyword outcomes have higher matchQuality than semantic", async () => {
    // Seed 5 keyword outcomes with high quality and 5 semantic with low quality.
    await seedOutcomes("keyword", 5, 0.9);
    await seedOutcomes("semantic", 5, 0.3);

    const { AdaptiveRetriever } = await import("../src/genome/retrieval-adapter.ts");
    const retriever = new AdaptiveRetriever(cwd, { minOutcomesForLearning: 4 });
    const strategy = await retriever.selectBestStrategy(
      "how does the system find relevant sections for queries",
    );
    expect(strategy).toBe("keyword");
  });

  test("semantic preferred when semantic outcomes have higher matchQuality than keyword", async () => {
    // Seed 5 semantic outcomes with high quality and 5 keyword with low quality.
    await seedOutcomes("semantic", 5, 0.95);
    await seedOutcomes("keyword", 5, 0.2);

    const { AdaptiveRetriever } = await import("../src/genome/retrieval-adapter.ts");
    const retriever = new AdaptiveRetriever(cwd, { minOutcomesForLearning: 4 });
    const strategy = await retriever.selectBestStrategy(
      "how does the system find relevant sections for queries",
    );
    expect(strategy).toBe("semantic");
  });

  test("latency-pressure fallback: semantic with avg latency > budget falls back to keyword", async () => {
    // Seed semantic outcomes with good quality but very high latency.
    await seedOutcomes("semantic", 5, 0.95, 5000); // 5s avg latency
    await seedOutcomes("keyword", 5, 0.4, 80);

    const { AdaptiveRetriever } = await import("../src/genome/retrieval-adapter.ts");
    // Set latency budget to 3000ms — semantic at 5000ms exceeds it.
    const retriever = new AdaptiveRetriever(cwd, {
      minOutcomesForLearning: 4,
      latencyBudgetMs: 3000,
    });
    const strategy = await retriever.selectBestStrategy(
      "how does the system find relevant sections for queries",
    );
    // Despite semantic having higher quality, latency pressure forces keyword.
    expect(strategy).toBe("keyword");
  });

  test("semantic within latency budget is not downgraded", async () => {
    // Semantic is fast (200ms) and high quality — should stay selected.
    await seedOutcomes("semantic", 5, 0.9, 200);
    await seedOutcomes("keyword", 5, 0.3, 80);

    const { AdaptiveRetriever } = await import("../src/genome/retrieval-adapter.ts");
    const retriever = new AdaptiveRetriever(cwd, {
      minOutcomesForLearning: 4,
      latencyBudgetMs: 3000,
    });
    const strategy = await retriever.selectBestStrategy(
      "how does the system find relevant sections for queries",
    );
    expect(strategy).toBe("semantic");
  });

  test("keyword preferred for rare-term queries even when semantic has some data", async () => {
    // Seed sparse data — fewer than minOutcomesForLearning for rare-term type,
    // so heuristic kicks in → keyword for rare-term.
    const { recordRetrievalOutcome } = await import("../src/genome/retrieval-adapter.ts");
    // Only 2 records (below minOutcomesForLearning=4 threshold for this type)
    // but overall count is 2, also below threshold → pure heuristic
    await recordRetrievalOutcome(cwd, "retrieveSections camelCase", "semantic", makeMetrics({ matchQuality: 0.9 }));
    await recordRetrievalOutcome(cwd, "snake_case_token body", "semantic", makeMetrics({ matchQuality: 0.9 }));

    const { AdaptiveRetriever } = await import("../src/genome/retrieval-adapter.ts");
    const retriever = new AdaptiveRetriever(cwd, { minOutcomesForLearning: 4 });
    // Below threshold → heuristic → rare-term → keyword
    const strategy = await retriever.selectBestStrategy("retrieveSections");
    expect(strategy).toBe("keyword");
  });
});

// ─── createAdaptiveRetriever ─────────────────────────────────────────────────

describe("createAdaptiveRetriever — factory function", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("returns an AdaptiveRetriever instance", async () => {
    const { createAdaptiveRetriever, AdaptiveRetriever } = await import(
      "../src/genome/retrieval-adapter.ts"
    );
    const retriever = createAdaptiveRetriever(cwd);
    expect(retriever).toBeInstanceOf(AdaptiveRetriever);
  });

  test("factory-created retriever uses heuristic default for rare-term with no data", async () => {
    const { createAdaptiveRetriever } = await import("../src/genome/retrieval-adapter.ts");
    const retriever = createAdaptiveRetriever(cwd);
    const strategy = await retriever.selectBestStrategy("SectionMeta[]");
    expect(strategy).toBe("keyword");
  });
});

// ─── exports via genome index ─────────────────────────────────────────────────

describe("genome/index.ts — retrieval-adapter symbols are exported", () => {
  test("AdaptiveRetriever is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.AdaptiveRetriever).toBe("function");
  });

  test("createAdaptiveRetriever is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.createAdaptiveRetriever).toBe("function");
  });

  test("classifyQuery is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.classifyQuery).toBe("function");
  });

  test("recordRetrievalOutcome is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.recordRetrievalOutcome).toBe("function");
  });

  test("loadRetrievalOutcomes is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.loadRetrievalOutcomes).toBe("function");
  });
});
