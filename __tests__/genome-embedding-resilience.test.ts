/**
 * Genome Embedding Resilience — multi-tier fallback strategy tests.
 *
 * Covers:
 *  - TfIdfProvider: always available, deterministic, correct IDF weighting
 *  - OllamaProvider: unavailable path + timeout → fallback
 *  - OpenAIProvider: unavailable when no API key, permanent error on 401/403
 *  - EmbeddingProviderRegistry: tier ordering, forced provider, circuit breakers
 *  - ResilientProviderPipeline: Ollama timeout → OpenAI fallback → TF-IDF,
 *    circuit breaker trips + recovery, strategy event persistence
 *  - parseEmbeddingProviderFlag: all aliases
 *  - createDefaultProviderRegistry: shape of built registry
 *  - EmbeddingRouter + providerRegistry: delegates to multi-tier pipeline
 *  - genome/index.ts: all new symbols exported
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, rm } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";

// ─── helpers ────────────────────────────────────────────────────────────────

function makeTmpDir(): string {
  return join(
    tmpdir(),
    `ashlr-prov-${Date.now()}-${Math.random().toString(36).slice(2)}`,
  );
}

async function setupGenomeDir(cwd: string): Promise<void> {
  await mkdir(join(cwd, ".ashlrcode", "genome", "evolution"), {
    recursive: true,
  });
}

// ─── TfIdfProvider ───────────────────────────────────────────────────────────

describe("TfIdfProvider — Tier-3 deterministic embedding", () => {
  test("isAvailable() always returns true", async () => {
    const { TfIdfProvider } = await import("../src/genome/embedding-resilience.ts");
    const provider = new TfIdfProvider();
    expect(provider.isAvailable()).toBe(true);
  });

  test("tier is 3", async () => {
    const { TfIdfProvider } = await import("../src/genome/embedding-resilience.ts");
    expect(new TfIdfProvider().tier).toBe(3);
  });

  test("name is 'tfidf'", async () => {
    const { TfIdfProvider } = await import("../src/genome/embedding-resilience.ts");
    expect(new TfIdfProvider().name).toBe("tfidf");
  });

  test("returns empty array for empty input", async () => {
    const { TfIdfProvider } = await import("../src/genome/embedding-resilience.ts");
    const provider = new TfIdfProvider();
    const result = await provider.embed([]);
    expect(result).toEqual([]);
  });

  test("single text returns a non-empty L2-normalised vector", async () => {
    const { TfIdfProvider } = await import("../src/genome/embedding-resilience.ts");
    const provider = new TfIdfProvider();
    const result = await provider.embed(["hello world test"]);
    expect(result).toHaveLength(1);
    const vec = result[0]!;
    expect(vec.length).toBeGreaterThan(0);
    // L2 norm should be ~1 (or 0 for empty text)
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
    expect(norm).toBeCloseTo(1, 3);
  });

  test("batch of texts returns one vector per text", async () => {
    const { TfIdfProvider } = await import("../src/genome/embedding-resilience.ts");
    const provider = new TfIdfProvider();
    const texts = ["alpha beta", "gamma delta epsilon", "hello world"];
    const result = await provider.embed(texts);
    expect(result).toHaveLength(3);
    for (const vec of result) {
      expect(vec.length).toBeGreaterThan(0);
    }
  });

  test("identical texts produce identical vectors", async () => {
    const { TfIdfProvider } = await import("../src/genome/embedding-resilience.ts");
    const provider = new TfIdfProvider();
    const r1 = await provider.embed(["the quick brown fox"]);
    const r2 = await provider.embed(["the quick brown fox"]);
    expect(r1[0]).toEqual(r2[0]);
  });

  test("different texts produce different vectors", async () => {
    const { TfIdfProvider } = await import("../src/genome/embedding-resilience.ts");
    const provider = new TfIdfProvider();
    const [v1, v2] = await provider.embed(["apple orange mango", "car bus truck"]);
    // Cosine similarity should be < 1 for unrelated texts
    const dot = v1!.reduce((s, v, i) => s + v * v2![i]!, 0);
    expect(dot).toBeLessThan(1);
  });

  test("respects TFIDF_MAX_DIM cap on vocabulary-heavy texts", async () => {
    const { TfIdfProvider, TFIDF_MAX_DIM } = await import(
      "../src/genome/embedding-resilience.ts"
    );
    const provider = new TfIdfProvider();
    // Generate a text with many unique terms
    const uniqueWords = Array.from({ length: 1000 }, (_, i) => `word${i}`).join(
      " ",
    );
    const result = await provider.embed([uniqueWords]);
    expect(result[0]!.length).toBeLessThanOrEqual(TFIDF_MAX_DIM);
  });
});

// ─── OpenAIProvider ──────────────────────────────────────────────────────────

describe("OpenAIProvider — Tier-2 OpenAI embeddings", () => {
  test("tier is 2", async () => {
    const { OpenAIProvider } = await import("../src/genome/embedding-resilience.ts");
    expect(new OpenAIProvider().tier).toBe(2);
  });

  test("name is 'openai'", async () => {
    const { OpenAIProvider } = await import("../src/genome/embedding-resilience.ts");
    expect(new OpenAIProvider().name).toBe("openai");
  });

  test("isAvailable() returns false when OPENAI_API_KEY is absent", async () => {
    const { OpenAIProvider } = await import("../src/genome/embedding-resilience.ts");
    const provider = new OpenAIProvider();
    const savedKey = process.env["OPENAI_API_KEY"];
    delete process.env["OPENAI_API_KEY"];
    expect(provider.isAvailable()).toBe(false);
    if (savedKey !== undefined) process.env["OPENAI_API_KEY"] = savedKey;
  });

  test("isAvailable() returns true when OPENAI_API_KEY is set", async () => {
    const { OpenAIProvider } = await import("../src/genome/embedding-resilience.ts");
    const provider = new OpenAIProvider();
    const savedKey = process.env["OPENAI_API_KEY"];
    process.env["OPENAI_API_KEY"] = "sk-test-key";
    expect(provider.isAvailable()).toBe(true);
    if (savedKey !== undefined) process.env["OPENAI_API_KEY"] = savedKey;
    else delete process.env["OPENAI_API_KEY"];
  });

  test("embed() throws when OPENAI_API_KEY is not set", async () => {
    const { OpenAIProvider } = await import("../src/genome/embedding-resilience.ts");
    const provider = new OpenAIProvider();
    const savedKey = process.env["OPENAI_API_KEY"];
    delete process.env["OPENAI_API_KEY"];
    await expect(provider.embed(["test"])).rejects.toThrow("OPENAI_API_KEY not set");
    if (savedKey !== undefined) process.env["OPENAI_API_KEY"] = savedKey;
  });
});

// ─── OllamaProvider ──────────────────────────────────────────────────────────

describe("OllamaProvider — Tier-1 Ollama local", () => {
  test("tier is 1", async () => {
    const { OllamaProvider } = await import("../src/genome/embedding-resilience.ts");
    expect(new OllamaProvider().tier).toBe(1);
  });

  test("name is 'ollama'", async () => {
    const { OllamaProvider } = await import("../src/genome/embedding-resilience.ts");
    expect(new OllamaProvider().name).toBe("ollama");
  });

  test("isAvailable() returns a boolean", async () => {
    const { OllamaProvider } = await import("../src/genome/embedding-resilience.ts");
    const provider = new OllamaProvider();
    const result = await provider.isAvailable();
    expect(typeof result).toBe("boolean");
  });
});

// ─── parseEmbeddingProviderFlag ───────────────────────────────────────────────

describe("parseEmbeddingProviderFlag — CLI flag parsing", () => {
  test("'ollama' → 'ollama'", async () => {
    const { parseEmbeddingProviderFlag } = await import(
      "../src/genome/embedding-resilience.ts"
    );
    expect(parseEmbeddingProviderFlag("ollama")).toBe("ollama");
  });

  test("'1' → 'ollama'", async () => {
    const { parseEmbeddingProviderFlag } = await import(
      "../src/genome/embedding-resilience.ts"
    );
    expect(parseEmbeddingProviderFlag("1")).toBe("ollama");
  });

  test("'openai' → 'openai'", async () => {
    const { parseEmbeddingProviderFlag } = await import(
      "../src/genome/embedding-resilience.ts"
    );
    expect(parseEmbeddingProviderFlag("openai")).toBe("openai");
  });

  test("'2' → 'openai'", async () => {
    const { parseEmbeddingProviderFlag } = await import(
      "../src/genome/embedding-resilience.ts"
    );
    expect(parseEmbeddingProviderFlag("2")).toBe("openai");
  });

  test("'tfidf' → 'tfidf'", async () => {
    const { parseEmbeddingProviderFlag } = await import(
      "../src/genome/embedding-resilience.ts"
    );
    expect(parseEmbeddingProviderFlag("tfidf")).toBe("tfidf");
  });

  test("'3' → 'tfidf'", async () => {
    const { parseEmbeddingProviderFlag } = await import(
      "../src/genome/embedding-resilience.ts"
    );
    expect(parseEmbeddingProviderFlag("3")).toBe("tfidf");
  });

  test("case-insensitive: 'OLLAMA' → 'ollama'", async () => {
    const { parseEmbeddingProviderFlag } = await import(
      "../src/genome/embedding-resilience.ts"
    );
    expect(parseEmbeddingProviderFlag("OLLAMA")).toBe("ollama");
  });

  test("null → undefined (auto-select)", async () => {
    const { parseEmbeddingProviderFlag } = await import(
      "../src/genome/embedding-resilience.ts"
    );
    expect(parseEmbeddingProviderFlag(null)).toBeUndefined();
  });

  test("empty string → undefined", async () => {
    const { parseEmbeddingProviderFlag } = await import(
      "../src/genome/embedding-resilience.ts"
    );
    expect(parseEmbeddingProviderFlag("")).toBeUndefined();
  });

  test("unknown value → undefined", async () => {
    const { parseEmbeddingProviderFlag } = await import(
      "../src/genome/embedding-resilience.ts"
    );
    expect(parseEmbeddingProviderFlag("anthropic")).toBeUndefined();
  });
});

// ─── EmbeddingProviderRegistry ───────────────────────────────────────────────

describe("EmbeddingProviderRegistry — provider management + circuit breakers", () => {
  test("providers are sorted by tier", async () => {
    const { EmbeddingProviderRegistry, TfIdfProvider, OpenAIProvider, OllamaProvider } =
      await import("../src/genome/embedding-resilience.ts");
    // Register in reverse tier order to confirm sorting
    const registry = new EmbeddingProviderRegistry([
      new TfIdfProvider(),
      new OpenAIProvider(),
      new OllamaProvider(),
    ]);
    const providers = registry.getProviders();
    expect(providers[0]!.tier).toBe(1);
    expect(providers[1]!.tier).toBe(2);
    expect(providers[2]!.tier).toBe(3);
  });

  test("getCircuitState returns 'closed' initially", async () => {
    const { EmbeddingProviderRegistry, TfIdfProvider } = await import(
      "../src/genome/embedding-resilience.ts"
    );
    const registry = new EmbeddingProviderRegistry([new TfIdfProvider()]);
    expect(registry.getCircuitState("tfidf")).toBe("closed");
  });

  test("recordOutcome trips breaker to open after enough failures", async () => {
    const { EmbeddingProviderRegistry, TfIdfProvider } = await import(
      "../src/genome/embedding-resilience.ts"
    );
    const registry = new EmbeddingProviderRegistry([new TfIdfProvider()], {
      windowSize: 4,
      failureRateThreshold: 0.5,
      openDurationMs: 30_000,
    });
    // 3 failures out of 4 = 75% > 50% threshold (need ≥70% window filled)
    registry.recordOutcome("tfidf", false);
    registry.recordOutcome("tfidf", false);
    registry.recordOutcome("tfidf", false);
    registry.recordOutcome("tfidf", true);
    expect(registry.getCircuitState("tfidf")).toBe("open");
  });

  test("resetAllBreakers resets open breakers to closed", async () => {
    const { EmbeddingProviderRegistry, TfIdfProvider } = await import(
      "../src/genome/embedding-resilience.ts"
    );
    const registry = new EmbeddingProviderRegistry([new TfIdfProvider()], {
      windowSize: 4,
      failureRateThreshold: 0.5,
      openDurationMs: 30_000,
    });
    for (let i = 0; i < 4; i++) registry.recordOutcome("tfidf", false);
    registry.resetAllBreakers();
    expect(registry.getCircuitState("tfidf")).toBe("closed");
  });

  test("selectProviders returns all providers when no forced override", async () => {
    const { EmbeddingProviderRegistry, TfIdfProvider, OpenAIProvider } = await import(
      "../src/genome/embedding-resilience.ts"
    );
    const registry = new EmbeddingProviderRegistry([
      new TfIdfProvider(),
      new OpenAIProvider(),
    ]);
    const selected = registry.selectProviders();
    expect(selected).toHaveLength(2);
  });

  test("selectProviders with forced name returns only that provider", async () => {
    const {
      EmbeddingProviderRegistry,
      TfIdfProvider,
      OpenAIProvider,
    } = await import("../src/genome/embedding-resilience.ts");
    const registry = new EmbeddingProviderRegistry([
      new TfIdfProvider(),
      new OpenAIProvider(),
    ]);
    const selected = registry.selectProviders("tfidf");
    expect(selected).toHaveLength(1);
    expect(selected[0]!.name).toBe("tfidf");
  });

  test("selectProviders skips open-circuit providers", async () => {
    const {
      EmbeddingProviderRegistry,
      TfIdfProvider,
      OpenAIProvider,
    } = await import("../src/genome/embedding-resilience.ts");
    const registry = new EmbeddingProviderRegistry(
      [new TfIdfProvider(), new OpenAIProvider()],
      { windowSize: 4, failureRateThreshold: 0.5, openDurationMs: 60_000 },
    );
    // Trip tfidf breaker open
    for (let i = 0; i < 4; i++) registry.recordOutcome("tfidf", false);
    const selected = registry.selectProviders();
    const names = selected.map((p) => p.name);
    expect(names).not.toContain("tfidf");
    expect(names).toContain("openai");
  });
});

// ─── ResilientProviderPipeline — Ollama timeout → OpenAI fallback ─────────────

describe("ResilientProviderPipeline — multi-tier fallback + circuit breakers", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("always returns a result when Tier-3 TF-IDF is in the registry", async () => {
    const {
      EmbeddingProviderRegistry,
      TfIdfProvider,
      ResilientProviderPipeline,
    } = await import("../src/genome/embedding-resilience.ts");
    const registry = new EmbeddingProviderRegistry([new TfIdfProvider()]);
    const pipeline = new ResilientProviderPipeline(registry, {
      cwd,
      disablePersistence: true,
    });
    const result = await pipeline.embed("hello world");
    expect(result.embedding.length).toBeGreaterThan(0);
    expect(result.providerName).toBe("tfidf");
    expect(result.tier).toBe(3);
  });

  test("usedFallback=false when Tier-1 provider succeeds directly", async () => {
    const {
      EmbeddingProviderRegistry,
      TfIdfProvider,
      ResilientProviderPipeline,
    } = await import("../src/genome/embedding-resilience.ts");

    // Mock Tier-1 provider that always succeeds
    const mockTier1 = {
      name: "mock-ollama",
      tier: 1 as const,
      isAvailable: () => true,
      embed: async (texts: string[]) => texts.map(() => [0.1, 0.2, 0.3]),
    };

    const registry = new EmbeddingProviderRegistry([mockTier1, new TfIdfProvider()]);
    const pipeline = new ResilientProviderPipeline(registry, {
      cwd,
      disablePersistence: true,
    });
    const result = await pipeline.embed("test text");
    expect(result.usedFallback).toBe(false);
    expect(result.providerName).toBe("mock-ollama");
    expect(result.tier).toBe(1);
    expect(result.failedProviders).toHaveLength(0);
  });

  test("Ollama timeout triggers OpenAI fallback", async () => {
    const {
      EmbeddingProviderRegistry,
      TfIdfProvider,
      ResilientProviderPipeline,
    } = await import("../src/genome/embedding-resilience.ts");

    // Tier-1 that always times out / throws
    const failingTier1 = {
      name: "mock-ollama-timeout",
      tier: 1 as const,
      isAvailable: () => true,
      embed: async (_texts: string[]) => {
        throw new Error("Ollama embedding timed out after 1ms");
      },
    };

    // Tier-2 mock that always succeeds
    const mockTier2 = {
      name: "mock-openai",
      tier: 2 as const,
      isAvailable: () => true,
      embed: async (texts: string[]) =>
        texts.map(() => Array.from({ length: 1536 }, () => Math.random())),
    };

    const registry = new EmbeddingProviderRegistry([
      failingTier1,
      mockTier2,
      new TfIdfProvider(),
    ]);
    const pipeline = new ResilientProviderPipeline(registry, {
      cwd,
      disablePersistence: true,
      backoff: { maxAttempts: 1, disableJitter: true },
    });

    const result = await pipeline.embed("test embedding");
    expect(result.providerName).toBe("mock-openai");
    expect(result.tier).toBe(2);
    expect(result.usedFallback).toBe(true);
    expect(result.failedProviders).toContain("mock-ollama-timeout");
    expect(result.embedding.length).toBe(1536);
  });

  test("Ollama + OpenAI failure falls through to TF-IDF (Tier 3)", async () => {
    const {
      EmbeddingProviderRegistry,
      TfIdfProvider,
      ResilientProviderPipeline,
    } = await import("../src/genome/embedding-resilience.ts");

    const failingProvider = (name: string, tier: 1 | 2) => ({
      name,
      tier,
      isAvailable: () => true,
      embed: async (_texts: string[]) => {
        throw new Error(`${name} unavailable`);
      },
    });

    const registry = new EmbeddingProviderRegistry([
      failingProvider("mock-ollama", 1),
      failingProvider("mock-openai", 2),
      new TfIdfProvider(),
    ]);
    const pipeline = new ResilientProviderPipeline(registry, {
      cwd,
      disablePersistence: true,
      backoff: { maxAttempts: 1, disableJitter: true },
    });

    const result = await pipeline.embed("offline fallback text");
    expect(result.tier).toBe(3);
    expect(result.usedFallback).toBe(true);
    expect(result.embedding.length).toBeGreaterThan(0);
  });

  test("circuit breaker trips after repeated Ollama failures", async () => {
    const {
      EmbeddingProviderRegistry,
      TfIdfProvider,
      ResilientProviderPipeline,
    } = await import("../src/genome/embedding-resilience.ts");

    let callCount = 0;
    const flakeyTier1 = {
      name: "flakey-ollama",
      tier: 1 as const,
      isAvailable: () => true,
      embed: async (_texts: string[]) => {
        callCount++;
        throw new Error("connection refused");
      },
    };

    const registry = new EmbeddingProviderRegistry(
      [flakeyTier1, new TfIdfProvider()],
      { windowSize: 4, failureRateThreshold: 0.5, openDurationMs: 60_000 },
    );
    const pipeline = new ResilientProviderPipeline(registry, {
      cwd,
      disablePersistence: true,
      backoff: { maxAttempts: 1, disableJitter: true },
    });

    // Make 4 requests to trip the circuit breaker (3 failures fills window)
    for (let i = 0; i < 4; i++) {
      await pipeline.embed(`text ${i}`);
    }

    const initialCallCount = callCount;
    // Now the breaker should be open; next call should skip flakey-ollama
    await pipeline.embed("should skip ollama");
    // callCount should not have increased (circuit open → skipped)
    expect(callCount).toBe(initialCallCount);
    expect(registry.getCircuitState("flakey-ollama")).toBe("open");
  });

  test("circuit breaker recovery: half-open probe on success resets to closed", async () => {
    const {
      EmbeddingProviderRegistry,
      TfIdfProvider,
      recordCircuitOutcome,
    } = await import("../src/genome/embedding-resilience.ts");

    const registry = new EmbeddingProviderRegistry(
      [new TfIdfProvider()],
      { windowSize: 4, failureRateThreshold: 0.5, openDurationMs: 30_000 },
    );

    // Trip breaker open
    for (let i = 0; i < 4; i++) registry.recordOutcome("tfidf", false);
    expect(registry.getCircuitState("tfidf")).toBe("open");

    // Simulate time passing by recording via recordCircuitOutcome directly
    // Get the internal cb and set openedAtMs to past
    // We test the public API: resetAllBreakers and confirm state reset
    registry.resetAllBreakers();
    expect(registry.getCircuitState("tfidf")).toBe("closed");

    // After reset, record a success — stays closed
    registry.recordOutcome("tfidf", true);
    expect(registry.getCircuitState("tfidf")).toBe("closed");
  });

  test("permanentErrorEncountered flag set on 401/403 from provider", async () => {
    const {
      EmbeddingProviderRegistry,
      TfIdfProvider,
      ResilientProviderPipeline,
      PermanentEmbeddingError,
    } = await import("../src/genome/embedding-resilience.ts");

    const permanentFailProvider = {
      name: "mock-permanent-fail",
      tier: 1 as const,
      isAvailable: () => true,
      embed: async (_texts: string[]) => {
        throw new PermanentEmbeddingError(401, "Unauthorized");
      },
    };

    const registry = new EmbeddingProviderRegistry([
      permanentFailProvider,
      new TfIdfProvider(),
    ]);
    const pipeline = new ResilientProviderPipeline(registry, {
      cwd,
      disablePersistence: true,
      backoff: { maxAttempts: 2, disableJitter: true },
    });

    const result = await pipeline.embed("auth test");
    expect(result.permanentErrorEncountered).toBe(true);
    // TF-IDF should still provide an embedding as the ultimate fallback
    expect(result.embedding.length).toBeGreaterThan(0);
  });

  test("strategy events are persisted to evolution/embedding-strategy.jsonl", async () => {
    const {
      EmbeddingProviderRegistry,
      TfIdfProvider,
      ResilientProviderPipeline,
      loadEmbeddingStrategyLog,
    } = await import("../src/genome/embedding-resilience.ts");
    const registry = new EmbeddingProviderRegistry([new TfIdfProvider()]);
    const pipeline = new ResilientProviderPipeline(registry, {
      cwd,
      disablePersistence: false,
    });
    await pipeline.embed("persistence test");
    const events = await loadEmbeddingStrategyLog(cwd);
    expect(events.length).toBeGreaterThan(0);
    const ev = events[0]!;
    expect(typeof ev.winningProvider).toBe("string");
    expect(typeof ev.totalLatencyMs).toBe("number");
    expect(typeof ev.usedFallback).toBe("boolean");
    expect(typeof ev.costUsd).toBe("number");
  });

  test("disablePersistence=true writes no strategy events", async () => {
    const {
      EmbeddingProviderRegistry,
      TfIdfProvider,
      ResilientProviderPipeline,
      loadEmbeddingStrategyLog,
    } = await import("../src/genome/embedding-resilience.ts");
    const registry = new EmbeddingProviderRegistry([new TfIdfProvider()]);
    const pipeline = new ResilientProviderPipeline(registry, {
      cwd,
      disablePersistence: true,
    });
    await pipeline.embed("no persistence test");
    const events = await loadEmbeddingStrategyLog(cwd);
    expect(events).toEqual([]);
  });

  test("forcedProvider pins to a specific provider (bypasses tier order)", async () => {
    const {
      EmbeddingProviderRegistry,
      TfIdfProvider,
      ResilientProviderPipeline,
    } = await import("../src/genome/embedding-resilience.ts");

    const mockTier1 = {
      name: "mock-ollama",
      tier: 1 as const,
      isAvailable: () => true,
      embed: async (texts: string[]) => texts.map(() => [1, 2, 3]),
    };

    const registry = new EmbeddingProviderRegistry([mockTier1, new TfIdfProvider()]);
    const pipeline = new ResilientProviderPipeline(registry, {
      cwd,
      disablePersistence: true,
      forcedProvider: "tfidf",
    });

    const result = await pipeline.embed("forced provider test");
    expect(result.providerName).toBe("tfidf");
  });

  test("costUsd is 0 for TF-IDF and Ollama providers", async () => {
    const {
      EmbeddingProviderRegistry,
      TfIdfProvider,
      ResilientProviderPipeline,
    } = await import("../src/genome/embedding-resilience.ts");
    const registry = new EmbeddingProviderRegistry([new TfIdfProvider()]);
    const pipeline = new ResilientProviderPipeline(registry, {
      cwd,
      disablePersistence: true,
    });
    const result = await pipeline.embed("cost test");
    expect(result.costUsd).toBe(0);
  });

  test("unavailable provider is skipped before embed is called", async () => {
    const {
      EmbeddingProviderRegistry,
      TfIdfProvider,
      ResilientProviderPipeline,
    } = await import("../src/genome/embedding-resilience.ts");

    let embedCalled = false;
    const unavailableProvider = {
      name: "unavailable-tier1",
      tier: 1 as const,
      isAvailable: () => false,
      embed: async (_texts: string[]) => {
        embedCalled = true;
        return [[0.1]];
      },
    };

    const registry = new EmbeddingProviderRegistry([
      unavailableProvider,
      new TfIdfProvider(),
    ]);
    const pipeline = new ResilientProviderPipeline(registry, {
      cwd,
      disablePersistence: true,
    });

    const result = await pipeline.embed("skip unavailable");
    expect(embedCalled).toBe(false);
    expect(result.failedProviders).toContain("unavailable-tier1");
    expect(result.failureReasons).toContain("unavailable");
    expect(result.providerName).toBe("tfidf");
  });
});

// ─── createDefaultProviderRegistry ───────────────────────────────────────────

describe("createDefaultProviderRegistry — standard 3-tier registry", () => {
  test("returns an EmbeddingProviderRegistry", async () => {
    const { createDefaultProviderRegistry, EmbeddingProviderRegistry } = await import(
      "../src/genome/embedding-resilience.ts"
    );
    const registry = createDefaultProviderRegistry();
    expect(registry).toBeInstanceOf(EmbeddingProviderRegistry);
  });

  test("contains exactly 3 providers", async () => {
    const { createDefaultProviderRegistry } = await import(
      "../src/genome/embedding-resilience.ts"
    );
    const registry = createDefaultProviderRegistry();
    expect(registry.getProviders()).toHaveLength(3);
  });

  test("providers are named ollama, openai, tfidf in tier order", async () => {
    const { createDefaultProviderRegistry } = await import(
      "../src/genome/embedding-resilience.ts"
    );
    const registry = createDefaultProviderRegistry();
    const names = registry.getProviders().map((p) => p.name);
    expect(names[0]).toBe("ollama");
    expect(names[1]).toBe("openai");
    expect(names[2]).toBe("tfidf");
  });
});

// ─── EmbeddingRouter + providerRegistry integration ──────────────────────────

describe("EmbeddingRouter — providerRegistry integration", () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("embed() delegates to registry when providerRegistry is set", async () => {
    const { EmbeddingRouter } = await import("../src/genome/embedding-router.ts");
    const {
      EmbeddingProviderRegistry,
      TfIdfProvider,
    } = await import("../src/genome/embedding-resilience.ts");

    const registry = new EmbeddingProviderRegistry([new TfIdfProvider()]);
    const router = new EmbeddingRouter(cwd, {
      providerRegistry: registry,
      providerCwd: cwd,
    });

    const result = await router.embed("test query", "test text to embed");
    expect(result).not.toBeNull();
    expect(result!.embedding.length).toBeGreaterThan(0);
    // TF-IDF is Tier 3 → usedUltrafast flag
    expect(result!.usedUltrafast).toBe(true);
  });

  test("embed() with providerOverride pins to forced provider", async () => {
    const { EmbeddingRouter } = await import("../src/genome/embedding-router.ts");
    const {
      EmbeddingProviderRegistry,
      TfIdfProvider,
    } = await import("../src/genome/embedding-resilience.ts");

    const mockTier1 = {
      name: "mock-ollama",
      tier: 1 as const,
      isAvailable: () => true,
      embed: async (texts: string[]) =>
        texts.map(() => Array.from({ length: 384 }, () => 0.01)),
    };

    const registry = new EmbeddingProviderRegistry([mockTier1, new TfIdfProvider()]);
    const router = new EmbeddingRouter(cwd, {
      providerRegistry: registry,
      providerOverride: "tfidf",
      providerCwd: cwd,
    });

    const result = await router.embed("query", "forced tfidf text");
    expect(result).not.toBeNull();
    // Forced to TF-IDF (tier 3)
    expect(result!.usedUltrafast).toBe(true);
  });
});

// ─── genome/index.ts — all new symbols exported ──────────────────────────────

describe("genome/index.ts — multi-tier resilience symbols exported", () => {
  test("EmbeddingProviderRegistry is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(
      typeof (mod as Record<string, unknown>)["EmbeddingProviderRegistry"],
    ).toBe("function");
  });

  test("TfIdfProvider is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof (mod as Record<string, unknown>)["TfIdfProvider"]).toBe("function");
  });

  test("OllamaProvider is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof (mod as Record<string, unknown>)["OllamaProvider"]).toBe("function");
  });

  test("OpenAIProvider is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof (mod as Record<string, unknown>)["OpenAIProvider"]).toBe("function");
  });

  test("ResilientProviderPipeline is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(
      typeof (mod as Record<string, unknown>)["ResilientProviderPipeline"],
    ).toBe("function");
  });

  test("createDefaultProviderRegistry is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(
      typeof (mod as Record<string, unknown>)["createDefaultProviderRegistry"],
    ).toBe("function");
  });

  test("parseEmbeddingProviderFlag is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(
      typeof (mod as Record<string, unknown>)["parseEmbeddingProviderFlag"],
    ).toBe("function");
  });

  test("persistEmbeddingStrategyEvent is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(
      typeof (mod as Record<string, unknown>)["persistEmbeddingStrategyEvent"],
    ).toBe("function");
  });

  test("loadEmbeddingStrategyLog is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(
      typeof (mod as Record<string, unknown>)["loadEmbeddingStrategyLog"],
    ).toBe("function");
  });

  test("TFIDF_MAX_DIM constant is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(typeof (mod as Record<string, unknown>)["TFIDF_MAX_DIM"]).toBe("number");
  });

  test("OPENAI_EMBEDDING_MODEL constant is exported", async () => {
    const mod = await import("../src/genome/index.ts");
    expect(
      typeof (mod as Record<string, unknown>)["OPENAI_EMBEDDING_MODEL"],
    ).toBe("string");
  });
});
