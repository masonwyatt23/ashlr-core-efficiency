/**
 * Multi-Provider Token Counter tests.
 *
 * Covers:
 *   - getEncodingForModel: correct encoding per provider family
 *   - estimateTokensWithModel: provider-aware estimates + fallback chain
 *   - Fallback chain: unknown models don't crash, pick sensible defaults
 *   - Batch consistency: deterministic across 10 calls
 *   - Edge cases: empty string, very long message, mixed content blocks
 *   - Per-provider multipliers: Llama / Qwen apply ≥1× multiplier
 */

import { describe, expect, test, beforeEach } from "bun:test";
import {
  getEncodingForModel,
  estimateTokensWithModel,
  countTokensAccurate,
  _resetTokenizerCache,
  _forceLoaderFailure,
} from "../src/tokens/index.ts";
import type { Message } from "../src/types/index.ts";

beforeEach(() => {
  _resetTokenizerCache();
});

// ---------------------------------------------------------------------------
// getEncodingForModel — mapping correctness
// ---------------------------------------------------------------------------

describe("getEncodingForModel — encoding selection", () => {
  // Anthropic: modern claude-3+ → o200k_base
  test("claude-3-5-sonnet-20241022 → o200k_base", () => {
    expect(getEncodingForModel("claude-3-5-sonnet-20241022")).toBe("o200k_base");
  });

  test("claude-3-opus-20240229 → o200k_base", () => {
    expect(getEncodingForModel("claude-3-opus-20240229")).toBe("o200k_base");
  });

  test("claude-3-haiku → o200k_base", () => {
    expect(getEncodingForModel("claude-3-haiku")).toBe("o200k_base");
  });

  // Anthropic: legacy claude-2 / claude-1 → cl100k_base
  test("claude-2.1 → cl100k_base", () => {
    expect(getEncodingForModel("claude-2.1")).toBe("cl100k_base");
  });

  test("claude-instant-1.2 → cl100k_base", () => {
    expect(getEncodingForModel("claude-instant-1.2")).toBe("cl100k_base");
  });

  // OpenAI exact slugs
  test("gpt-4o → o200k_base", () => {
    expect(getEncodingForModel("gpt-4o")).toBe("o200k_base");
  });

  test("gpt-4o-mini → o200k_base", () => {
    expect(getEncodingForModel("gpt-4o-mini")).toBe("o200k_base");
  });

  test("gpt-4-turbo → cl100k_base", () => {
    expect(getEncodingForModel("gpt-4-turbo")).toBe("cl100k_base");
  });

  test("gpt-3.5-turbo → cl100k_base", () => {
    expect(getEncodingForModel("gpt-3.5-turbo")).toBe("cl100k_base");
  });

  test("o1 → o200k_base", () => {
    expect(getEncodingForModel("o1")).toBe("o200k_base");
  });

  // Meta Llama
  test("meta-llama/Llama-3-8b → cl100k_base", () => {
    expect(getEncodingForModel("meta-llama/Llama-3-8b")).toBe("cl100k_base");
  });

  test("llama2-70b → cl100k_base", () => {
    expect(getEncodingForModel("llama2-70b")).toBe("cl100k_base");
  });

  // Qwen
  test("qwen-7b → cl100k_base", () => {
    expect(getEncodingForModel("qwen-7b")).toBe("cl100k_base");
  });

  test("qwen2.5-72b → cl100k_base", () => {
    expect(getEncodingForModel("qwen2.5-72b")).toBe("cl100k_base");
  });

  // Gemini
  test("gemini-1.5-pro → o200k_base", () => {
    expect(getEncodingForModel("gemini-1.5-pro")).toBe("o200k_base");
  });

  // Mistral
  test("mistral-large → cl100k_base", () => {
    expect(getEncodingForModel("mistral-large")).toBe("cl100k_base");
  });

  // Unknown → cl100k_base (safe fallback)
  test("totally-unknown-model-xyz → cl100k_base", () => {
    expect(getEncodingForModel("totally-unknown-model-xyz")).toBe("cl100k_base");
  });

  test("empty string → cl100k_base", () => {
    expect(getEncodingForModel("")).toBe("cl100k_base");
  });
});

// ---------------------------------------------------------------------------
// Fallback chain: unknown model does not crash
// ---------------------------------------------------------------------------

describe("estimateTokensWithModel — fallback chain (tiktoken available)", () => {
  test("unknown model string does not throw", async () => {
    const n = await estimateTokensWithModel("hello world", "totally-unknown-model-v99");
    expect(n).toBeGreaterThan(0);
  });

  test("empty model string does not throw", async () => {
    const n = await estimateTokensWithModel("hello world", "");
    expect(n).toBeGreaterThan(0);
  });

  test("unknown model with 1.1× multiplier: result ≥ bare cl100k_base count", async () => {
    // Unknown model applies a 1.1× safety margin, so result should be ≥ the
    // raw tiktoken count for the same text.
    const unknownCount = await estimateTokensWithModel(
      "the quick brown fox",
      "some-hypothetical-future-model",
    );
    const baseCount = await countTokensAccurate("the quick brown fox", "gpt-4");
    // Unknown ≥ base (safety multiplier must never reduce the estimate).
    expect(unknownCount).toBeGreaterThanOrEqual(baseCount);
  });

  test("Llama estimate ≥ gpt-4 estimate (1.05× multiplier)", async () => {
    const text = "function fibonacci(n) { return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2); }";
    const llamaCount = await estimateTokensWithModel(text, "llama2-7b");
    const gptCount   = await countTokensAccurate(text, "gpt-4");
    expect(llamaCount).toBeGreaterThanOrEqual(gptCount);
  });

  test("Qwen estimate ≥ gpt-4 estimate (1.08× multiplier)", async () => {
    const text = "你好，世界。This is mixed Chinese and English content.";
    const qwenCount = await estimateTokensWithModel(text, "qwen-7b");
    const gptCount  = await countTokensAccurate(text, "gpt-4");
    expect(qwenCount).toBeGreaterThanOrEqual(gptCount);
  });
});

// ---------------------------------------------------------------------------
// Fallback chain: tiktoken unavailable
// ---------------------------------------------------------------------------

describe("estimateTokensWithModel — fallback chain (tiktoken unavailable)", () => {
  beforeEach(() => {
    _forceLoaderFailure();
  });

  test("unknown model: falls back to heuristic without throwing", async () => {
    const n = await estimateTokensWithModel("hello world", "totally-unknown-model-v99");
    // Heuristic: ceil(11/4) = 3, but with 1.1× safety: ceil(3*1.1) = 4 (or higher).
    expect(n).toBeGreaterThan(0);
  });

  test("claude-3-5: heuristic result is positive", async () => {
    const text = "the quick brown fox jumps over the lazy dog";
    const n = await estimateTokensWithModel(text, "claude-3-5");
    // No multiplier for claude (1.0×), so result = heuristic = ceil(len/4)
    expect(n).toBe(Math.ceil(text.length / 4));
  });

  test("llama2: heuristic × 1.05 applied", async () => {
    const text = "the quick brown fox jumps over the lazy dog";
    const n = await estimateTokensWithModel(text, "llama2");
    const base = Math.ceil(text.length / 4);
    // ceil(base * 1.05)
    expect(n).toBe(Math.ceil(base * 1.05));
  });

  test("qwen: heuristic × 1.08 applied", async () => {
    const text = "the quick brown fox jumps over the lazy dog";
    const n = await estimateTokensWithModel(text, "qwen-7b");
    const base = Math.ceil(text.length / 4);
    expect(n).toBe(Math.ceil(base * 1.08));
  });
});

// ---------------------------------------------------------------------------
// Per-provider accuracy: ±2% error margin vs known counts
// ---------------------------------------------------------------------------

describe("per-provider accuracy — ±2% error margin", () => {
  // Build a deterministic ~500-token sample (~2000 chars).
  const SAMPLE_500 = "The quick brown fox jumps over the lazy dog. ".repeat(45); // ~45×44=1980 chars

  test("claude-3-5 (o200k_base proxy): estimate within ±5% of baseline", async () => {
    const claudeCount = await estimateTokensWithModel(SAMPLE_500, "claude-3-5");
    // Compare against direct o200k_base count (same encoding, so must be identical).
    const baseCount = await countTokensAccurate(SAMPLE_500, "claude-3-5");
    const errorPct = Math.abs((claudeCount - baseCount) / baseCount) * 100;
    expect(errorPct).toBeLessThanOrEqual(5);
  });

  test("gpt-4 (cl100k_base exact): estimate within ±2% of baseline", async () => {
    const gptCount  = await estimateTokensWithModel(SAMPLE_500, "gpt-4");
    const baseCount = await countTokensAccurate(SAMPLE_500, "gpt-4");
    const errorPct  = Math.abs((gptCount - baseCount) / baseCount) * 100;
    expect(errorPct).toBeLessThanOrEqual(2);
  });

  test("gpt-4o (o200k_base exact): estimate within ±2% of baseline", async () => {
    const gptCount  = await estimateTokensWithModel(SAMPLE_500, "gpt-4o");
    const baseCount = await countTokensAccurate(SAMPLE_500, "gpt-4o");
    const errorPct  = Math.abs((gptCount - baseCount) / baseCount) * 100;
    expect(errorPct).toBeLessThanOrEqual(2);
  });

  test("llama2 estimate is within ±10% of gpt-4 estimate (proxy)", async () => {
    // Llama uses cl100k_base × 1.05 — verify it doesn't balloon unreasonably.
    const llamaCount = await estimateTokensWithModel(SAMPLE_500, "llama2");
    const gptCount   = await countTokensAccurate(SAMPLE_500, "gpt-4");
    const errorPct   = Math.abs((llamaCount - gptCount) / gptCount) * 100;
    expect(errorPct).toBeLessThanOrEqual(10);
  });
});

// ---------------------------------------------------------------------------
// Batch consistency: deterministic across 10 calls
// ---------------------------------------------------------------------------

describe("batch consistency — deterministic", () => {
  const MODELS: string[] = [
    "claude-3-5",
    "gpt-4",
    "gpt-4o",
    "llama2",
    "qwen-7b",
    "totally-unknown-model",
  ];
  const TEXT = "Batch determinism test: same input must always yield same token count.";

  for (const model of MODELS) {
    test(`10× calls with model="${model}" all return the same count`, async () => {
      const counts: number[] = [];
      for (let i = 0; i < 10; i++) {
        counts.push(await estimateTokensWithModel(TEXT, model));
      }
      // All 10 must be identical.
      expect(new Set(counts).size).toBe(1);
      expect(counts[0]).toBeGreaterThan(0);
    });
  }
});

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

describe("edge cases", () => {
  test("empty string → 0 for all provider models", async () => {
    const models = ["claude-3-5", "gpt-4", "gpt-4o", "llama2", "qwen", "default"];
    for (const model of models) {
      const n = await estimateTokensWithModel("", model);
      expect(n).toBe(0);
    }
  });

  test("very long message (~10K tokens) does not throw or return 0", async () => {
    // ~40 000 chars → ~10 000 tokens at 4 chars/token.
    const longText = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(700);
    const n = await estimateTokensWithModel(longText, "claude-3-5");
    expect(n).toBeGreaterThan(1000);
  });

  test("very long message: llama estimate ≥ gpt-4 estimate", async () => {
    const longText = "a ".repeat(20_000);
    const llamaCount = await estimateTokensWithModel(longText, "llama2");
    const gptCount   = await countTokensAccurate(longText, "gpt-4");
    expect(llamaCount).toBeGreaterThanOrEqual(gptCount);
  });

  test("mixed content blocks — all providers return positive count", async () => {
    const msgs: Message[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "What is this image?" },
          { type: "image_url", image_url: { url: "https://example.com/img.png" } },
        ],
      },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "The user wants me to describe an image.", signature: "sig" },
          { type: "text", text: "It appears to be a landscape photo." },
        ],
      },
      {
        role: "assistant",
        content: [
          {
            type: "tool_use",
            id: "tu_1",
            name: "describe_image",
            input: { url: "https://example.com/img.png" },
          },
        ],
      },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "tu_1", content: "A serene mountain lake." },
        ],
      },
    ];

    const models = ["claude-3-5", "gpt-4", "gpt-4o", "llama2", "qwen-7b", "totally-unknown"];
    for (const model of models) {
      const n = await estimateTokensWithModel(msgs, model);
      expect(n).toBeGreaterThan(0);
    }
  });

  test("single character string → positive token count for all providers", async () => {
    for (const model of ["claude-3-5", "gpt-4", "llama2", "qwen"]) {
      const n = await estimateTokensWithModel("x", model);
      expect(n).toBeGreaterThan(0);
    }
  });

  test("unicode-heavy content: CJK text gets appropriate count per provider", async () => {
    const cjkText = "你好世界，这是一段中文文本，用于测试不同提供商的令牌计数。".repeat(20);
    const qwenCount   = await estimateTokensWithModel(cjkText, "qwen-7b");
    const claudeCount = await estimateTokensWithModel(cjkText, "claude-3-5");
    // Both should be positive; Qwen may differ due to its multiplier.
    expect(qwenCount).toBeGreaterThan(0);
    expect(claudeCount).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// TokenizerModel shorthand back-compat
// ---------------------------------------------------------------------------

describe("TokenizerModel shorthands remain backward-compatible", () => {
  const shorthands = [
    "claude-3-5",
    "claude-2",
    "gpt-4",
    "gpt-4o",
    "gpt-3.5",
    "llama2",
    "qwen",
    "default",
  ] as const;

  for (const model of shorthands) {
    test(`shorthand "${model}" works with estimateTokensWithModel`, async () => {
      const n = await estimateTokensWithModel("hello world", model);
      expect(n).toBeGreaterThan(0);
    });
  }

  test("countTokensAccurate still accepts TokenizerModel shorthands", async () => {
    const n = await countTokensAccurate("hello world", "claude-2");
    expect(n).toBeGreaterThan(0);
  });
});
