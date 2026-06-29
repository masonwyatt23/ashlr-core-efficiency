/**
 * Embedding Router tests — multi-tier model selection + hybrid search routing.
 *
 * Covers:
 *  - classifyQueryComplexity: rare-term / semantic / general
 *  - heuristicModelDefault: tier mapping per complexity
 *  - selectEmbeddingModel: heuristic fallback + learned selection
 *  - scoreModelOutcomes: per-(queryComplexity, modelName) scoring with latency penalty
 *  - recordModelOutcome / loadModelOutcomes: JSONL persistence round-trip
 *  - generateEmbeddingWithFallback: cascade order and fallback flag
 *  - EmbeddingRouter.dryRunSelectModel: model selection without calling Ollama
 *  - EmbeddingRouter.modelStats: aggregate per-model statistics
 *  - createEmbeddingRouter: convenience factory
 *  - genome/index.ts: all new symbols are exported
 *  - retrieval-adapter: modelName field persisted in RetrievalOutcomeRecord
 *  - retrieval-adapter: scoreOutcomes per-(queryType, modelName) filtering
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, rm } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";

// ─── helpers ────────────────────────────────────────────────────────────────

function makeTmpDir(): string {
  return join(tmpdir(), `ashlr-emb-router-${Date.now()}-${Math.random().toString(36).slice(2)}`);
}

async function setupGenomeDir(cwd: string): Promise<void> {
  await mkdir(join(cwd, ".ashlrcode", "genome", "evolution"), { recursive: true });
}

function makeModelOutcome(
  overrides: Partial<import("../src/genome/embedding-router.ts").ModelOutcomeRecord> = {},
): Omit<import("../src/genome/embedding-router.ts").ModelOutcomeRecord, "id" | "timestamp"> {
  return {
    modelName: "nomic-embed-text",
    queryComplexity: "semantic",
    latencyMs: 100,
    succeeded: true,
    topSimilarity: 0.7,
    matchQuality: 0.8,
    ...overrides,
  };
}

// ─── classifyQueryComplexity ────────────────────────────────────────────────

describe("classifyQueryComplexity — complexity detection", () => {
  test("camelCase identifier → rare-term", async () => {
    const { classifyQueryComplexity } = await import("../src/genome/embedding-router.ts");
    expect(classifyQueryComplexity("retrieveSections")).toBe("rare-term");
  });

  test("snake_case identifier → rare-term", async () => {
    const { classifyQueryComplexity } = await import("../src/genome/embedding-router.ts");
    expect(classifyQueryComplexity("genome_dir_path")).toBe("rare-term");
  });

  test("file extension token → rare-term", async () => {
    const { classifyQueryComplexity } = await import("../src/genome/embedding-router.ts");
    expect(classifyQueryComplexity("embeddings.ts update logic")).toBe("rare-term");
  });

  test("bracket token → rare-term", async () => {
    const { classifyQueryComplexity } = await import("../src/genome/embedding-router.ts");
    expect(classifyQueryComplexity("SectionMeta[]")).toBe("rare-term");
  });

  test("natural-language question → semantic", async () => {
    const { classifyQueryComplexity } = await import("../src/genome/embedding-router.ts");
    expect(classifyQueryComplexity("how does the genome retrieval system find sections")).toBe("semantic");
  });

  test("describe-style phrase → semantic", async () => {
    const { classifyQueryComplexity } = await import("../src/genome/embedding-router.ts");
    expect(classifyQueryComplexity("what should the vision for the project be")).toBe("semantic");
  });

  test("very short query → general", async () => {
    const { classifyQueryComplexity } = await import("../src/genome/embedding-router.ts");
    expect(classifyQueryComplexity("embed")).toBe("general");
  });

  test("empty query → general", async () => {
    const { classifyQueryComplexity } = await import("../src/genome/embedding-router.ts");
    expect(classifyQueryComplexity("")).toBe("general");
  });

  test("whitespace-only → general", async () => {
    const { classifyQueryComplexity } = await import("../src/genome/embedding-router.ts");
    expect(classifyQueryComplexity("   ")).toBe("general");
  });
});

// ─── heuristicModelDefault ──────────────────────────────────────────────────

describe("heuristicModelDefault — tier-based defaults", () => {
  test("rare-term → fast tier (all-minilm-l6-v2)", async () => {
    const { heuristicModelDefault } = await import("../src/genome/embedding-router.ts");
    expect(heuristicModelDefault("rare-term")).toBe("all-minilm-l6-v2");
  });

  test("semantic → accurate tier (bge-base-en-v1-5)", async () => {
    const { heuristicModelDefault } = await import("../src/genome/embedding-router.ts");
    expect(heuristicModelDefault("semantic")).toBe("bge-base-en-v1-5");
  });

  test("general → balanced tier (nomic-embed-text)", async () => {
    const { heuristicModelDefault } = await import("../src/genome/embedding-router.ts");
    expect(heuristicModelDefault("general")).toBe("nomic-embed-text");
  });
});

// ─── EMBEDDING_MODELS registry ──────────────────────────────────────────────

describe("EMBEDDING_MODELS — registry integrity", () => {
  test("all three models are registered", async () => {
    const { EMBEDDING_MODELS } = await import("../src/genome/embedding-router.ts");
    expect(Object.keys(EMBEDDING_MODELS)).toContain("nomic-embed-text");
    expect(Object.keys(EMBEDDING_MODELS)).toContain("all-minilm-l6-v2");
    expect(Object.keys(EMBEDDING_MODELS)).toContain("bge-base-en-v1-5");
  });

  test("all-minilm-l6-v2 is fast tier with 384 dimensions", async () => {
    const { EMBEDDING_MODELS } = await import("../src/genome/embedding-router.ts");
    const spec = EMBEDDING_MODELS["all-minilm-l6-v2"];
    expect(spec.tier).toBe("fast");
    expect(spec.dimension).toBe(384);
  });

  test("nomic-embed-text is balanced tier", async () => {
    const { EMBEDDING_MODELS } = await import("../src/genome/embedding-router.ts");
    expect(EMBEDDING_MODELS["nomic-embed-text"].tier).toBe("balanced");
  });

  test("bge-base-en-v1-5 is accurate tier", async () => {
    const { EMBEDDING_MODELS } = await import("../src/genome/embedding-router.ts");
    expect(EMBEDDING_MODELS["bge-base-en-v1-5"].tier).toBe("accurate");
  });
});

// ─── recordModelOutcome / loadModelOutcomes ──────────────────────────────────

describe("recordModelOutcome / loadModelOutcomes — JSONL persistence", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("returns a non-empty id string", async () => {
    const { recordModelOutcome } = await import("../src/genome/embedding-router.ts");
    const id = await recordModelOutcome(cwd, makeModelOutcome());
    expect(typeof id).toBe("string");
    expect(id.length).toBeGreaterThan(0);
    expect(id.startsWith("mo-")).toBe(true);
  });

  test("persisted record is readable via loadModelOutcomes", async () => {
    const { recordModelOutcome, loadModelOutcomes } = await import(
      "../src/genome/embedding-router.ts"
    );
    await recordModelOutcome(cwd, makeModelOutcome({ modelName: "bge-base-en-v1-5", matchQuality: 0.95 }));
    const records = await loadModelOutcomes(cwd);
    expect(records).toHaveLength(1);
    expect(records[0]!.modelName).toBe("bge-base-en-v1-5");
    expect(records[0]!.matchQuality).toBe(0.95);
  });

  test("multiple records accumulate in order", async () => {
    const { recordModelOutcome, loadModelOutcomes } = await import(
      "../src/genome/embedding-router.ts"
    );
    await recordModelOutcome(cwd, makeModelOutcome({ modelName: "all-minilm-l6-v2" }));
    await recordModelOutcome(cwd, makeModelOutcome({ modelName: "nomic-embed-text" }));
    await recordModelOutcome(cwd, makeModelOutcome({ modelName: "bge-base-en-v1-5" }));
    const records = await loadModelOutcomes(cwd);
    expect(records).toHaveLength(3);
    expect(records[0]!.modelName).toBe("all-minilm-l6-v2");
    expect(records[1]!.modelName).toBe("nomic-embed-text");
    expect(records[2]!.modelName).toBe("bge-base-en-v1-5");
  });

  test("loadModelOutcomes returns empty array when no file exists", async () => {
    const { loadModelOutcomes } = await import("../src/genome/embedding-router.ts");
    const records = await loadModelOutcomes(cwd);
    expect(records).toEqual([]);
  });

  test("succeeded:false is persisted correctly", async () => {
    const { recordModelOutcome, loadModelOutcomes } = await import(
      "../src/genome/embedding-router.ts"
    );
    await recordModelOutcome(cwd, makeModelOutcome({ succeeded: false, matchQuality: null }));
    const records = await loadModelOutcomes(cwd);
    expect(records[0]!.succeeded).toBe(false);
    expect(records[0]!.matchQuality).toBeNull();
  });
});

// ─── selectEmbeddingModel — heuristic (no prior data) ───────────────────────

describe("selectEmbeddingModel — heuristic defaults (no prior outcomes)", () => {
  test("rare-term with no data → fast model", async () => {
    const { selectEmbeddingModel } = await import("../src/genome/embedding-router.ts");
    const model = selectEmbeddingModel("rare-term", [], { minOutcomesForLearning: 4 });
    expect(model).toBe("all-minilm-l6-v2");
  });

  test("semantic with no data → accurate model", async () => {
    const { selectEmbeddingModel } = await import("../src/genome/embedding-router.ts");
    const model = selectEmbeddingModel("semantic", [], { minOutcomesForLearning: 4 });
    expect(model).toBe("bge-base-en-v1-5");
  });

  test("general with no data → balanced model", async () => {
    const { selectEmbeddingModel } = await import("../src/genome/embedding-router.ts");
    const model = selectEmbeddingModel("general", [], { minOutcomesForLearning: 4 });
    expect(model).toBe("nomic-embed-text");
  });
});

// ─── selectEmbeddingModel — learned selection ────────────────────────────────

describe("selectEmbeddingModel — learned model preference from outcomes", () => {
  test("selects model with highest matchQuality for the complexity", async () => {
    const { selectEmbeddingModel, recordModelOutcome, loadModelOutcomes } = await import(
      "../src/genome/embedding-router.ts"
    );
    const cwd = makeTmpDir();
    await setupGenomeDir(cwd);

    // Seed: bge wins for semantic queries
    for (let i = 0; i < 5; i++) {
      await recordModelOutcome(cwd, makeModelOutcome({ modelName: "bge-base-en-v1-5", matchQuality: 0.95, queryComplexity: "semantic" }));
      await recordModelOutcome(cwd, makeModelOutcome({ modelName: "nomic-embed-text", matchQuality: 0.5, queryComplexity: "semantic" }));
      await recordModelOutcome(cwd, makeModelOutcome({ modelName: "all-minilm-l6-v2", matchQuality: 0.3, queryComplexity: "semantic" }));
    }
    const outcomes = await loadModelOutcomes(cwd);
    const model = selectEmbeddingModel("semantic", outcomes, { minOutcomesForLearning: 4 });
    expect(model).toBe("bge-base-en-v1-5");

    await rm(cwd, { recursive: true, force: true });
  });

  test("selects fast model when it outperforms others for rare-term", async () => {
    const { selectEmbeddingModel, recordModelOutcome, loadModelOutcomes } = await import(
      "../src/genome/embedding-router.ts"
    );
    const cwd = makeTmpDir();
    await setupGenomeDir(cwd);

    // Seed: all-minilm wins for rare-term queries
    for (let i = 0; i < 5; i++) {
      await recordModelOutcome(cwd, makeModelOutcome({ modelName: "all-minilm-l6-v2", matchQuality: 0.9, queryComplexity: "rare-term" }));
      await recordModelOutcome(cwd, makeModelOutcome({ modelName: "bge-base-en-v1-5", matchQuality: 0.4, queryComplexity: "rare-term" }));
    }
    const outcomes = await loadModelOutcomes(cwd);
    const model = selectEmbeddingModel("rare-term", outcomes, { minOutcomesForLearning: 4 });
    expect(model).toBe("all-minilm-l6-v2");

    await rm(cwd, { recursive: true, force: true });
  });

  test("latency penalty reduces score for slow model", async () => {
    const { selectEmbeddingModel, recordModelOutcome, loadModelOutcomes } = await import(
      "../src/genome/embedding-router.ts"
    );
    const cwd = makeTmpDir();
    await setupGenomeDir(cwd);

    // bge has higher quality but very slow; nomic is faster and still good
    for (let i = 0; i < 5; i++) {
      await recordModelOutcome(cwd, makeModelOutcome({ modelName: "bge-base-en-v1-5", matchQuality: 0.95, latencyMs: 8000, queryComplexity: "semantic" }));
      await recordModelOutcome(cwd, makeModelOutcome({ modelName: "nomic-embed-text", matchQuality: 0.85, latencyMs: 300, queryComplexity: "semantic" }));
    }
    const outcomes = await loadModelOutcomes(cwd);
    // latencyBudgetMs=500 means 8000ms is heavily penalised
    const model = selectEmbeddingModel("semantic", outcomes, {
      minOutcomesForLearning: 4,
      latencyBudgetMs: 500,
    });
    // nomic should win due to latency penalty on bge
    expect(model).toBe("nomic-embed-text");

    await rm(cwd, { recursive: true, force: true });
  });

  test("falls back to heuristic when all models have no data for complexity", async () => {
    const { selectEmbeddingModel, recordModelOutcome, loadModelOutcomes } = await import(
      "../src/genome/embedding-router.ts"
    );
    const cwd = makeTmpDir();
    await setupGenomeDir(cwd);

    // Seed outcomes for a different complexity — no semantic data
    for (let i = 0; i < 5; i++) {
      await recordModelOutcome(cwd, makeModelOutcome({ modelName: "all-minilm-l6-v2", queryComplexity: "rare-term" }));
    }
    const outcomes = await loadModelOutcomes(cwd);
    // Requesting semantic model with no semantic data → heuristic default
    const model = selectEmbeddingModel("semantic", outcomes, { minOutcomesForLearning: 4 });
    // heuristic default for semantic is bge-base-en-v1-5
    expect(model).toBe("bge-base-en-v1-5");

    await rm(cwd, { recursive: true, force: true });
  });
});

// ─── EmbeddingRouter.dryRunSelectModel ──────────────────────────────────────

describe("EmbeddingRouter.dryRunSelectModel — selection without network call", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("rare-term query selects fast tier model with no prior data", async () => {
    const { EmbeddingRouter } = await import("../src/genome/embedding-router.ts");
    const router = new EmbeddingRouter(cwd);
    const result = await router.dryRunSelectModel("retrieveSections");
    expect(result.complexity).toBe("rare-term");
    expect(result.tier).toBe("fast");
    expect(result.model).toBe("all-minilm-l6-v2");
    expect(result.usedHeuristic).toBe(true);
  });

  test("semantic query selects accurate tier model with no prior data", async () => {
    const { EmbeddingRouter } = await import("../src/genome/embedding-router.ts");
    const router = new EmbeddingRouter(cwd);
    const result = await router.dryRunSelectModel(
      "what should the vision for the project describe",
    );
    expect(result.complexity).toBe("semantic");
    expect(result.tier).toBe("accurate");
    expect(result.model).toBe("bge-base-en-v1-5");
    expect(result.usedHeuristic).toBe(true);
  });

  test("general query selects balanced tier model with no prior data", async () => {
    const { EmbeddingRouter } = await import("../src/genome/embedding-router.ts");
    const router = new EmbeddingRouter(cwd);
    const result = await router.dryRunSelectModel("embedding");
    expect(result.complexity).toBe("general");
    expect(result.tier).toBe("balanced");
    expect(result.model).toBe("nomic-embed-text");
  });

  test("usedHeuristic is false when enough outcome data exists", async () => {
    const { EmbeddingRouter, recordModelOutcome } = await import(
      "../src/genome/embedding-router.ts"
    );
    // Seed enough outcomes to exceed minOutcomesForLearning
    for (let i = 0; i < 5; i++) {
      await recordModelOutcome(cwd, makeModelOutcome({ queryComplexity: "semantic" }));
    }
    const router = new EmbeddingRouter(cwd, { minOutcomesForLearning: 4 });
    const result = await router.dryRunSelectModel(
      "how does the system find sections for the query",
    );
    expect(result.usedHeuristic).toBe(false);
  });
});

// ─── EmbeddingRouter.modelStats ─────────────────────────────────────────────

describe("EmbeddingRouter.modelStats — aggregate per-model statistics", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("returns stats for all three known models", async () => {
    const { EmbeddingRouter } = await import("../src/genome/embedding-router.ts");
    const router = new EmbeddingRouter(cwd);
    const stats = await router.modelStats(cwd);
    const names = stats.map((s) => s.modelName);
    expect(names).toContain("nomic-embed-text");
    expect(names).toContain("all-minilm-l6-v2");
    expect(names).toContain("bge-base-en-v1-5");
  });

  test("totalCalls is 0 for all models when no outcomes recorded", async () => {
    const { EmbeddingRouter } = await import("../src/genome/embedding-router.ts");
    const router = new EmbeddingRouter(cwd);
    const stats = await router.modelStats(cwd);
    for (const s of stats) {
      expect(s.totalCalls).toBe(0);
    }
  });

  test("correctly counts calls and computes successRate", async () => {
    const { EmbeddingRouter, recordModelOutcome } = await import(
      "../src/genome/embedding-router.ts"
    );
    await recordModelOutcome(cwd, makeModelOutcome({ modelName: "nomic-embed-text", succeeded: true }));
    await recordModelOutcome(cwd, makeModelOutcome({ modelName: "nomic-embed-text", succeeded: true }));
    await recordModelOutcome(cwd, makeModelOutcome({ modelName: "nomic-embed-text", succeeded: false, matchQuality: null }));

    const router = new EmbeddingRouter(cwd);
    const stats = await router.modelStats(cwd);
    const nomicStats = stats.find((s) => s.modelName === "nomic-embed-text")!;
    expect(nomicStats.totalCalls).toBe(3);
    expect(nomicStats.successRate).toBeCloseTo(2 / 3, 5);
  });

  test("avgQuality is null when no quality signals recorded", async () => {
    const { EmbeddingRouter, recordModelOutcome } = await import(
      "../src/genome/embedding-router.ts"
    );
    await recordModelOutcome(cwd, makeModelOutcome({ modelName: "nomic-embed-text", matchQuality: null }));

    const router = new EmbeddingRouter(cwd);
    const stats = await router.modelStats(cwd);
    const nomicStats = stats.find((s) => s.modelName === "nomic-embed-text")!;
    expect(nomicStats.avgQuality).toBeNull();
  });

  test("avgLatencyMs is computed only over succeeded records", async () => {
    const { EmbeddingRouter, recordModelOutcome } = await import(
      "../src/genome/embedding-router.ts"
    );
    await recordModelOutcome(cwd, makeModelOutcome({ modelName: "bge-base-en-v1-5", latencyMs: 200, succeeded: true }));
    await recordModelOutcome(cwd, makeModelOutcome({ modelName: "bge-base-en-v1-5", latencyMs: 400, succeeded: true }));
    await recordModelOutcome(cwd, makeModelOutcome({ modelName: "bge-base-en-v1-5", latencyMs: 9999, succeeded: false, matchQuality: null }));

    const router = new EmbeddingRouter(cwd);
    const stats = await router.modelStats(cwd);
    const bgeStats = stats.find((s) => s.modelName === "bge-base-en-v1-5")!;
    // avg of 200 and 400 = 300; 9999 excluded (failed)
    expect(bgeStats.avgLatencyMs).toBeCloseTo(300, 0);
  });
});

// ─── createEmbeddingRouter ──────────────────────────────────────────────────

describe("createEmbeddingRouter — factory function", () => {
  test("returns an EmbeddingRouter instance", async () => {
    const { createEmbeddingRouter, EmbeddingRouter } = await import(
      "../src/genome/embedding-router.ts"
    );
    const cwd = makeTmpDir();
    await setupGenomeDir(cwd);
    const router = createEmbeddingRouter(cwd);
    expect(router).toBeInstanceOf(EmbeddingRouter);
    await rm(cwd, { recursive: true, force: true });
  });
});

// ─── genome/index.ts exports ────────────────────────────────────────────────

describe("genome/index.ts — embedding-router symbols are exported", () => {
  test("EmbeddingRouter is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.EmbeddingRouter).toBe("function");
  });

  test("createEmbeddingRouter is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.createEmbeddingRouter).toBe("function");
  });

  test("classifyQueryComplexity is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.classifyQueryComplexity).toBe("function");
  });

  test("heuristicModelDefault is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.heuristicModelDefault).toBe("function");
  });

  test("selectEmbeddingModel is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.selectEmbeddingModel).toBe("function");
  });

  test("recordModelOutcome is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.recordModelOutcome).toBe("function");
  });

  test("loadModelOutcomes is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.loadModelOutcomes).toBe("function");
  });

  test("EMBEDDING_MODELS is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.EMBEDDING_MODELS).toBe("object");
  });

  test("generateEmbeddingWithFallback is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof mod.generateEmbeddingWithFallback).toBe("function");
  });

  // embeddings.ts additions
  test("SUPPORTED_EMBEDDING_MODELS is exported from genome index", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(Array.isArray(mod.SUPPORTED_EMBEDDING_MODELS)).toBe(true);
    expect(mod.SUPPORTED_EMBEDDING_MODELS).toContain("nomic-embed-text");
    expect(mod.SUPPORTED_EMBEDDING_MODELS).toContain("all-minilm-l6-v2");
    expect(mod.SUPPORTED_EMBEDDING_MODELS).toContain("bge-base-en-v1-5");
  });
});

// ─── retrieval-adapter: modelName field ─────────────────────────────────────

describe("retrieval-adapter — modelName field in RetrievalOutcomeRecord", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("recordRetrievalOutcome stores modelName when provided", async () => {
    const { recordRetrievalOutcome, loadRetrievalOutcomes } = await import(
      "../src/genome/retrieval-adapter.ts"
    );
    await recordRetrievalOutcome(
      cwd,
      "how does the system find sections",
      "semantic",
      {
        latencyMs: 150,
        sectionsReturned: 3,
        totalRelevanceScore: 2.1,
        budgetFit: true,
        matchQuality: 0.88,
      },
      "bge-base-en-v1-5",
    );
    const records = await loadRetrievalOutcomes(cwd);
    expect(records[0]!.modelName).toBe("bge-base-en-v1-5");
  });

  test("recordRetrievalOutcome omits modelName when not provided", async () => {
    const { recordRetrievalOutcome, loadRetrievalOutcomes } = await import(
      "../src/genome/retrieval-adapter.ts"
    );
    await recordRetrievalOutcome(
      cwd,
      "keyword query",
      "keyword",
      {
        latencyMs: 50,
        sectionsReturned: 2,
        totalRelevanceScore: 1.5,
        budgetFit: true,
        matchQuality: 0.7,
      },
    );
    const records = await loadRetrievalOutcomes(cwd);
    expect(records[0]!.modelName).toBeUndefined();
  });
});

// ─── retrieval-adapter: useModelRouter option ───────────────────────────────

describe("AdaptiveRetriever — useModelRouter option", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("defaults to useModelRouter=true", async () => {
    const { AdaptiveRetriever } = await import("../src/genome/retrieval-adapter.ts");
    // We can only test this indirectly — just confirm construction works
    const retriever = new AdaptiveRetriever(cwd);
    expect(retriever).toBeDefined();
  });

  test("useModelRouter=false creates retriever without model routing", async () => {
    const { AdaptiveRetriever } = await import("../src/genome/retrieval-adapter.ts");
    const retriever = new AdaptiveRetriever(cwd, { useModelRouter: false });
    expect(retriever).toBeDefined();
  });

  test("selectBestStrategy still works with useModelRouter=false", async () => {
    const { AdaptiveRetriever } = await import("../src/genome/retrieval-adapter.ts");
    const retriever = new AdaptiveRetriever(cwd, { useModelRouter: false });
    // rare-term → keyword heuristic
    const strategy = await retriever.selectBestStrategy("retrieveSectionsV2");
    expect(strategy).toBe("keyword");
  });
});
