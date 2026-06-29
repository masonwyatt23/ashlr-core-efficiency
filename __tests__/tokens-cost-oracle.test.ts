/**
 * Tests for the USD cost oracle — costUsdFromUsage / costUsdFromResponse.
 *
 * These tie the static PROVIDER_RATES table to the cache multipliers and
 * produce an actual dollar figure (including output tokens), which no prior
 * helper did: recordActualTokenUsage yields effective *tokens* and omits output.
 */

import { describe, expect, test } from "bun:test";
import {
  costUsdFromUsage,
  costUsdFromResponse,
  PROVIDER_RATES,
  CACHE_WRITE_MULTIPLIER,
  CACHE_READ_MULTIPLIER,
  DEFAULT_PROVIDER_RATE,
} from "../src/tokens/index.ts";
import { countTokensWithCache } from "../src/tokens/index.ts";
import type { MessageResponse, CacheBlockMetadata, Message } from "../src/types/index.ts";

describe("costUsdFromUsage", () => {
  test("input-only cost uses the input rate at 1×", () => {
    const c = costUsdFromUsage({ inputTokens: 1_000_000, outputTokens: 0 }, "claude-3-5-sonnet");
    expect(c.inputCost).toBeCloseTo(3.0, 10); // $3.00 / 1M input
    expect(c.outputCost).toBe(0);
    expect(c.cacheWriteCost).toBe(0);
    expect(c.cacheReadCost).toBe(0);
    expect(c.totalCost).toBeCloseTo(3.0, 10);
  });

  test("output tokens are billed at the output rate (was previously ignored)", () => {
    const c = costUsdFromUsage({ inputTokens: 0, outputTokens: 1_000_000 }, "claude-3-5-sonnet");
    expect(c.outputCost).toBeCloseTo(15.0, 10); // $15.00 / 1M output
    expect(c.totalCost).toBeCloseTo(15.0, 10);
  });

  test("cache-write tokens billed at 1.25× input rate", () => {
    const rate = PROVIDER_RATES["claude-3-5-sonnet"];
    const c = costUsdFromUsage(
      { inputTokens: 0, outputTokens: 0, cacheCreationTokens: 1000 },
      "claude-3-5-sonnet",
    );
    expect(c.cacheWriteCost).toBeCloseTo(1000 * rate.inputRate * CACHE_WRITE_MULTIPLIER, 12);
  });

  test("cache-read tokens billed at 0.1× input rate", () => {
    const rate = PROVIDER_RATES["claude-3-5-sonnet"];
    const c = costUsdFromUsage(
      { inputTokens: 0, outputTokens: 0, cacheReadTokens: 1000 },
      "claude-3-5-sonnet",
    );
    expect(c.cacheReadCost).toBeCloseTo(1000 * rate.inputRate * CACHE_READ_MULTIPLIER, 12);
  });

  test("totalCost is the sum of all four components", () => {
    const c = costUsdFromUsage(
      {
        inputTokens: 5000,
        outputTokens: 2000,
        cacheCreationTokens: 3000,
        cacheReadTokens: 10000,
      },
      "claude-3-5-sonnet",
    );
    const sum = c.inputCost + c.cacheWriteCost + c.cacheReadCost + c.outputCost;
    expect(c.totalCost).toBeCloseTo(sum, 12);
    expect(c.totalCost).toBeGreaterThan(0);
  });

  test("negative token counts are clamped to zero", () => {
    const c = costUsdFromUsage(
      { inputTokens: -100, outputTokens: -50, cacheCreationTokens: -1, cacheReadTokens: -1 },
      "claude-3-5-sonnet",
    );
    expect(c.totalCost).toBe(0);
  });

  test("versioned slug resolves via prefix match", () => {
    const c = costUsdFromUsage(
      { inputTokens: 1_000_000, outputTokens: 0 },
      "claude-3-5-sonnet-20241022",
    );
    expect(c.rate).toBe(PROVIDER_RATES["claude-3-5-sonnet"]);
    expect(c.inputCost).toBeCloseTo(3.0, 10);
  });

  test("unknown provider falls back to the default rate", () => {
    const c = costUsdFromUsage({ inputTokens: 1000, outputTokens: 0 }, "totally-made-up-model");
    expect(c.rate).toBe(DEFAULT_PROVIDER_RATE);
  });

  test("haiku is cheaper than opus for identical usage", () => {
    const usage = { inputTokens: 100_000, outputTokens: 50_000 };
    const haiku = costUsdFromUsage(usage, "claude-3-haiku");
    const opus = costUsdFromUsage(usage, "claude-3-opus");
    expect(haiku.totalCost).toBeLessThan(opus.totalCost);
  });
});

describe("costUsdFromResponse", () => {
  test("maps a MessageResponse into a UsdCostBreakdown", () => {
    const resp: MessageResponse = {
      input_tokens: 1000,
      output_tokens: 500,
      cache_creation_input_tokens: 2000,
      cache_read_input_tokens: 8000,
      stop_reason: "end_turn",
    };
    const c = costUsdFromResponse(resp, "claude-3-5-sonnet");
    const expected = costUsdFromUsage(
      { inputTokens: 1000, outputTokens: 500, cacheCreationTokens: 2000, cacheReadTokens: 8000 },
      "claude-3-5-sonnet",
    );
    expect(c.totalCost).toBeCloseTo(expected.totalCost, 12);
  });

  test("missing optional cache fields default to zero", () => {
    const resp: MessageResponse = { input_tokens: 1000, output_tokens: 500 };
    const c = costUsdFromResponse(resp, "claude-3-5-sonnet");
    expect(c.cacheWriteCost).toBe(0);
    expect(c.cacheReadCost).toBe(0);
    expect(c.totalCost).toBeCloseTo(c.inputCost + c.outputCost, 12);
  });

  test("a cache-read-heavy turn is far cheaper than the same tokens uncached", () => {
    const cached = costUsdFromResponse(
      { input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 100_000 },
      "claude-3-5-sonnet",
    );
    const uncached = costUsdFromResponse(
      { input_tokens: 100_000, output_tokens: 0 },
      "claude-3-5-sonnet",
    );
    // cache read is 0.1× — should be roughly a 10× saving.
    expect(cached.totalCost).toBeCloseTo(uncached.totalCost * CACHE_READ_MULTIPLIER, 10);
  });
});

describe("countTokensWithCache — multi-read-block regression", () => {
  const msgs: Message[] = [{ role: "user", content: "hello world this is a message" }];

  test("multiple read blocks (no created_token_count) do not double-count the last message", async () => {
    const oneRead: CacheBlockMetadata[] = [{ cache_type: "ephemeral" }];
    const fiveReads: CacheBlockMetadata[] = Array.from({ length: 5 }, () => ({
      cache_type: "ephemeral" as const,
    }));

    const single = await countTokensWithCache(msgs, oneRead, "claude-3-5-sonnet");
    const many = await countTokensWithCache(msgs, fiveReads, "claude-3-5-sonnet");

    // Before the fix, five read blocks counted the last message 5× and could
    // exceed rawTokens. The fallback now applies at most once.
    expect(many.cacheReadTokens).toBe(single.cacheReadTokens);
    expect(many.cacheReadTokens).toBeLessThanOrEqual(many.rawTokens);
  });

  test("explicit write blocks still accumulate per block", async () => {
    const blocks: CacheBlockMetadata[] = [
      { cache_type: "ephemeral", created_token_count: 100 },
      { cache_type: "static", created_token_count: 200 },
    ];
    const r = await countTokensWithCache(msgs, blocks, "claude-3-5-sonnet");
    expect(r.cacheWriteTokens).toBe(300);
    expect(r.cacheReadTokens).toBe(0);
  });
});
