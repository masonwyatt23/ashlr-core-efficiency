/**
 * Tests for MultiProviderAdapter & Unified Cost Oracle extensions.
 *
 * Coverage:
 *   - Anthropic (baseline): explicit breakpoints, correct multipliers
 *   - OpenAI GPT-4: implicit caching, no breakpoints returned, savings estimated
 *   - OpenAI o1: no cache support → CacheDegradation
 *   - OpenAI o3: no cache support → CacheDegradation
 *   - Gemini: explicit breakpoints, 32k minimum token threshold
 *   - normalizeProviderKey: model-ID patterns
 *   - resolveProviderStrategy: correct strategy selection
 *   - ProviderCostOracle.getCacheStrategy & getCalibratedCacheCost extensions
 */

import { describe, expect, test, beforeEach, afterEach } from "bun:test";
import { mkdtemp, rm } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";

import {
  MultiProviderAdapter,
  recommendProviderCacheStrategy,
  normalizeProviderKey,
  resolveProviderStrategy,
  PROVIDER_CACHE_STRATEGIES,
  type AdapterOutcome,
  type CacheAdapterResult,
  type CacheDegradation,
} from "../src/anthropic/multi-provider-adapter.ts";
import { ProviderCostOracle } from "../src/budget/provider-cost-oracle.ts";
import type { Message } from "../src/anthropic/cache-breakpoint-optimizer.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a minimal message array with a target approximate token count. */
function makeMessages(approxTokens: number, count = 3): Message[] {
  const charsPerMessage = Math.ceil((approxTokens * 4) / count);
  const msgs: Message[] = [];
  for (let i = 0; i < count; i++) {
    msgs.push({
      role: i % 2 === 0 ? "user" : "assistant",
      content: "x".repeat(charsPerMessage),
    });
  }
  return msgs;
}

let tmpDir: string;

beforeEach(async () => {
  tmpDir = await mkdtemp(join(tmpdir(), "mpa-test-"));
});

afterEach(async () => {
  await rm(tmpDir, { recursive: true, force: true });
});

// ---------------------------------------------------------------------------
// 1. normalizeProviderKey
// ---------------------------------------------------------------------------

describe("normalizeProviderKey", () => {
  test("anthropic model IDs → 'anthropic'", () => {
    expect(normalizeProviderKey("claude-sonnet-4")).toBe("anthropic");
    expect(normalizeProviderKey("claude-opus-4")).toBe("anthropic");
    expect(normalizeProviderKey("anthropic")).toBe("anthropic");
  });

  test("OpenAI GPT model IDs → 'openai'", () => {
    expect(normalizeProviderKey("gpt-4o")).toBe("openai");
    expect(normalizeProviderKey("gpt-4-turbo")).toBe("openai");
    expect(normalizeProviderKey("openai")).toBe("openai");
  });

  test("OpenAI o1 variants → 'openai-o1'", () => {
    expect(normalizeProviderKey("o1")).toBe("openai-o1");
    expect(normalizeProviderKey("o1-preview")).toBe("openai-o1");
    expect(normalizeProviderKey("o1-mini")).toBe("openai-o1");
    expect(normalizeProviderKey("openai-o1")).toBe("openai-o1");
  });

  test("OpenAI o3 variants → 'openai-o3'", () => {
    expect(normalizeProviderKey("o3")).toBe("openai-o3");
    expect(normalizeProviderKey("o3-mini")).toBe("openai-o3");
    expect(normalizeProviderKey("openai-o3")).toBe("openai-o3");
  });

  test("Gemini model IDs → 'gemini'", () => {
    expect(normalizeProviderKey("gemini-1.5-pro")).toBe("gemini");
    expect(normalizeProviderKey("gemini-flash")).toBe("gemini");
    expect(normalizeProviderKey("google")).toBe("gemini");
  });

  test("unknown provider → 'anthropic' fallback", () => {
    expect(normalizeProviderKey("some-unknown-llm")).toBe("anthropic");
  });
});

// ---------------------------------------------------------------------------
// 2. resolveProviderStrategy — correct strategy fields
// ---------------------------------------------------------------------------

describe("resolveProviderStrategy", () => {
  test("Anthropic: explicit breakpoints, 1.25× write, 0.10× read", () => {
    const s = resolveProviderStrategy("anthropic");
    expect(s.supported).toBe(true);
    expect(s.explicitBreakpoints).toBe(true);
    expect(s.writeCostMultiplier).toBe(1.25);
    expect(s.readCostMultiplier).toBe(0.10);
    expect(s.maxBreakpoints).toBe(4);
    expect(s.minCacheableTokens).toBe(1024);
  });

  test("OpenAI GPT-4: implicit, no explicit breakpoints, 0.50× read", () => {
    const s = resolveProviderStrategy("gpt-4o");
    expect(s.supported).toBe(true);
    expect(s.explicitBreakpoints).toBe(false);
    expect(s.readCostMultiplier).toBe(0.50);
    expect(s.maxBreakpoints).toBe(0);
  });

  test("OpenAI o1: unsupported", () => {
    const s = resolveProviderStrategy("o1");
    expect(s.supported).toBe(false);
  });

  test("OpenAI o3: unsupported", () => {
    const s = resolveProviderStrategy("o3");
    expect(s.supported).toBe(false);
  });

  test("Gemini: 1 breakpoint, 32k minimum, 0.25× read", () => {
    const s = resolveProviderStrategy("gemini-1.5-pro");
    expect(s.supported).toBe(true);
    expect(s.explicitBreakpoints).toBe(true);
    expect(s.maxBreakpoints).toBe(1);
    expect(s.minCacheableTokens).toBe(32768);
    expect(s.readCostMultiplier).toBe(0.25);
  });
});

// ---------------------------------------------------------------------------
// 3. MultiProviderAdapter — Anthropic (baseline)
// ---------------------------------------------------------------------------

describe("MultiProviderAdapter — Anthropic", () => {
  test("returns supported outcome with correct strategy", async () => {
    const adapter = new MultiProviderAdapter({ cwd: tmpDir });
    const messages = makeMessages(2000);
    const outcome = await adapter.recommendCacheStrategy("anthropic", messages, {
      sessionLogDir: join(tmpDir, "sessions"),
      auditPath: join(tmpDir, "audit.jsonl"),
    });

    expect(outcome.type).toBe("supported");
    const result = outcome as CacheAdapterResult;
    expect(result.strategy.provider).toBe("anthropic");
    expect(result.strategy.explicitBreakpoints).toBe(true);
    expect(result.calibratedRate.inputPerMToken).toBeGreaterThan(0);
    // recommendations may be empty (no session data) but must be an array
    expect(Array.isArray(result.recommendations)).toBe(true);
  });

  test("computeCacheCostMicroUsd uses 1.25× write + 0.10× read multipliers", async () => {
    const adapter = new MultiProviderAdapter({ cwd: tmpDir });
    // 1M cached + 1M written at $3.00/M input rate (anthropic default)
    // read cost: 1M × $3 × 0.10 = $0.30 = 300_000 μUSD
    // write cost: 1M × $3 × 1.25 = $3.75 = 3_750_000 μUSD
    // total: 4_050_000 μUSD
    const cost = await adapter.computeCacheCostMicroUsd("anthropic", 1_000_000, 1_000_000);
    expect(cost).toBeGreaterThan(0);
    // Using default static rate $3.00/M
    expect(cost).toBe(4_050_000);
  });
});

// ---------------------------------------------------------------------------
// 4. MultiProviderAdapter — OpenAI GPT-4 (implicit cache)
// ---------------------------------------------------------------------------

describe("MultiProviderAdapter — OpenAI GPT-4", () => {
  test("returns supported with empty recommendations (implicit caching)", async () => {
    const adapter = new MultiProviderAdapter({ cwd: tmpDir });
    const messages = makeMessages(3000);
    const outcome = await adapter.recommendCacheStrategy("gpt-4o", messages);

    expect(outcome.type).toBe("supported");
    const result = outcome as CacheAdapterResult;
    expect(result.strategy.explicitBreakpoints).toBe(false);
    expect(result.recommendations).toHaveLength(0);
    // Still estimates savings (30% hit-rate heuristic)
    expect(result.estimatedSavingsMicroUsd).toBeGreaterThanOrEqual(0);
  });

  test("computeCacheCostMicroUsd uses 1.0× write + 0.50× read for OpenAI", async () => {
    const adapter = new MultiProviderAdapter({ cwd: tmpDir });
    // 1M cached + 0 written at $2.50/M input rate (openai default)
    // read cost: 1M × $2.50 × 0.50 = $1.25 = 1_250_000 μUSD
    const cost = await adapter.computeCacheCostMicroUsd("openai", 1_000_000, 0);
    expect(cost).toBe(1_250_000);
  });
});

// ---------------------------------------------------------------------------
// 5. MultiProviderAdapter — OpenAI o1 (no cache, graceful degradation)
// ---------------------------------------------------------------------------

describe("MultiProviderAdapter — OpenAI o1 (no cache)", () => {
  test("returns CacheDegradation for o1", async () => {
    const adapter = new MultiProviderAdapter({ cwd: tmpDir });
    const messages = makeMessages(5000);
    const outcome = await adapter.recommendCacheStrategy("o1", messages);

    expect(outcome.type).toBe("unsupported");
    const degradation = outcome as CacheDegradation;
    expect(degradation.provider).toBe("o1");
    expect(typeof degradation.reason).toBe("string");
    expect(degradation.reason.length).toBeGreaterThan(0);
  });

  test("returns CacheDegradation for o1-preview", async () => {
    const adapter = new MultiProviderAdapter({ cwd: tmpDir });
    const outcome = await adapter.recommendCacheStrategy("o1-preview", makeMessages(2000));
    expect(outcome.type).toBe("unsupported");
  });

  test("computeCacheCostMicroUsd returns 0 for o1", async () => {
    const adapter = new MultiProviderAdapter({ cwd: tmpDir });
    const cost = await adapter.computeCacheCostMicroUsd("o1", 1_000_000, 500_000);
    expect(cost).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// 6. MultiProviderAdapter — Gemini (token minimum threshold)
// ---------------------------------------------------------------------------

describe("MultiProviderAdapter — Gemini", () => {
  test("returns supported with correct strategy", async () => {
    const adapter = new MultiProviderAdapter({ cwd: tmpDir });
    // Small messages (below 32k Gemini minimum) → eligible list may be empty
    const smallMessages = makeMessages(1000);
    const outcome = await adapter.recommendCacheStrategy("gemini-1.5-pro", smallMessages, {
      sessionLogDir: join(tmpDir, "sessions"),
      auditPath: join(tmpDir, "audit.jsonl"),
    });

    expect(outcome.type).toBe("supported");
    const result = outcome as CacheAdapterResult;
    expect(result.strategy.provider).toBe("gemini");
    expect(result.strategy.minCacheableTokens).toBe(32768);
    // With small messages below min threshold, no breakpoints recommended
    expect(result.recommendations).toHaveLength(0);
  });

  test("Gemini: 0.25× read multiplier in cost calculation", async () => {
    const adapter = new MultiProviderAdapter({ cwd: tmpDir });
    // Gemini doesn't have a static entry in PROVIDER_PRICING so oracle falls
    // back to mid-range ($3/M). read cost = 1M × $3 × 0.25 = $0.75 = 750_000 μUSD
    const cost = await adapter.computeCacheCostMicroUsd("gemini", 1_000_000, 0);
    expect(cost).toBe(750_000);
  });
});

// ---------------------------------------------------------------------------
// 7. CacheBreakpointOptimizer — providerHint integration
// ---------------------------------------------------------------------------

describe("CacheBreakpointOptimizer providerHint", () => {
  test("o1 providerHint → empty recommendations (no cache)", async () => {
    const { CacheBreakpointOptimizer } = await import(
      "../src/anthropic/cache-breakpoint-optimizer.ts"
    );
    const optimizer = new CacheBreakpointOptimizer({
      sessionLogDir: join(tmpDir, "sessions"),
      auditPath: join(tmpDir, "audit.jsonl"),
    });
    const messages = makeMessages(4000);
    const recs = await optimizer.recommendBreakpoints(messages, { providerHint: "o1" });
    expect(recs).toHaveLength(0);
  });

  test("openai providerHint (implicit) → empty recommendations", async () => {
    const { CacheBreakpointOptimizer } = await import(
      "../src/anthropic/cache-breakpoint-optimizer.ts"
    );
    const optimizer = new CacheBreakpointOptimizer({
      sessionLogDir: join(tmpDir, "sessions"),
      auditPath: join(tmpDir, "audit.jsonl"),
    });
    const messages = makeMessages(4000);
    const recs = await optimizer.recommendBreakpoints(messages, { providerHint: "gpt-4o" });
    expect(recs).toHaveLength(0);
  });

  test("anthropic providerHint → maxBreakpoints clamped to 4", async () => {
    const { CacheBreakpointOptimizer } = await import(
      "../src/anthropic/cache-breakpoint-optimizer.ts"
    );
    const optimizer = new CacheBreakpointOptimizer({
      sessionLogDir: join(tmpDir, "sessions"),
      auditPath: join(tmpDir, "audit.jsonl"),
    });
    const messages = makeMessages(4000);
    // Requesting 10 breakpoints — must be clamped to Anthropic's limit of 4
    const recs = await optimizer.recommendBreakpoints(messages, {
      providerHint: "anthropic",
      maxBreakpoints: 10,
    });
    // Each recommendation's breakpointIndices must have ≤ 4 entries
    for (const rec of recs) {
      expect(rec.breakpointIndices.length).toBeLessThanOrEqual(4);
    }
  });
});

// ---------------------------------------------------------------------------
// 8. ProviderCostOracle extensions — getCacheStrategy / getCalibratedCacheCost
// ---------------------------------------------------------------------------

describe("ProviderCostOracle cache strategy extensions", () => {
  test("getCacheStrategy returns correct strategy for anthropic", async () => {
    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();
    const strategy = await oracle.getCacheStrategy("anthropic");
    expect(strategy.supported).toBe(true);
    expect(strategy.writeCostMultiplier).toBe(1.25);
    expect(strategy.readCostMultiplier).toBe(0.10);
  });

  test("getCacheStrategy returns unsupported for o1", async () => {
    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();
    const strategy = await oracle.getCacheStrategy("o1");
    expect(strategy.supported).toBe(false);
  });

  test("getCalibratedCacheCost returns 0 for o1 (no cache)", async () => {
    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();
    const cost = await oracle.getCalibratedCacheCost("o1", 1_000_000, 500_000);
    expect(cost).toBe(0);
  });

  test("getCalibratedCacheCost computes correctly for anthropic", async () => {
    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();
    // Static anthropic rate: $3.00/M input
    // 500k cached: 500k × $3 × 0.10 = $0.15 = 150_000 μUSD
    // 200k written: 200k × $3 × 1.25 = $0.75 = 750_000 μUSD
    // total = 900_000 μUSD
    const cost = await oracle.getCalibratedCacheCost("anthropic", 500_000, 200_000);
    expect(cost).toBe(900_000);
  });

  test("getCalibratedCacheCost uses 0.50× read multiplier for openai", async () => {
    const oracle = new ProviderCostOracle(tmpDir);
    await oracle.initialize();
    // Static openai rate: $2.50/M input
    // 1M cached: 1M × $2.50 × 0.50 = $1.25 = 1_250_000 μUSD
    // 0 written
    const cost = await oracle.getCalibratedCacheCost("openai", 1_000_000, 0);
    expect(cost).toBe(1_250_000);
  });
});

// ---------------------------------------------------------------------------
// 9. recommendProviderCacheStrategy module-level convenience
// ---------------------------------------------------------------------------

describe("recommendProviderCacheStrategy", () => {
  test("convenience wrapper returns same shape as adapter", async () => {
    const messages = makeMessages(2000);
    const outcome = await recommendProviderCacheStrategy("anthropic", messages, {
      cwd: tmpDir,
      sessionLogDir: join(tmpDir, "sessions"),
      auditPath: join(tmpDir, "audit.jsonl"),
    });
    expect(outcome.type).toBe("supported");
  });

  test("convenience wrapper degrades for o1", async () => {
    const messages = makeMessages(2000);
    const outcome = await recommendProviderCacheStrategy("o1", messages, { cwd: tmpDir });
    expect(outcome.type).toBe("unsupported");
  });
});

// ---------------------------------------------------------------------------
// 10. PROVIDER_CACHE_STRATEGIES registry completeness
// ---------------------------------------------------------------------------

describe("PROVIDER_CACHE_STRATEGIES registry", () => {
  test("all required providers present", () => {
    const required = ["anthropic", "openai", "openai-o1", "openai-o3", "gemini"];
    for (const provider of required) {
      expect(PROVIDER_CACHE_STRATEGIES[provider]).toBeDefined();
    }
  });

  test("supported providers have positive read multiplier < 1", () => {
    for (const [key, strategy] of Object.entries(PROVIDER_CACHE_STRATEGIES)) {
      if (strategy.supported) {
        expect(strategy.readCostMultiplier).toBeGreaterThan(0);
        expect(strategy.readCostMultiplier).toBeLessThan(1);
      }
    }
  });
});
