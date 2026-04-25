import { expect, test, describe, beforeEach } from "bun:test";
import {
  estimateTokens,
  countTokensAccurate,
  primeTokenizer,
  _resetTokenizerCache,
  _forceLoaderFailure,
} from "../src/tokens/index.ts";

describe("estimateTokens (heuristic — back-compat)", () => {
  test("string: chars/4", () => {
    expect(estimateTokens("hello world")).toBe(3); // 11/4 = 2.75 → 3
  });

  test("empty string → 0", () => {
    expect(estimateTokens("")).toBe(0);
  });

  test("messages with string content", () => {
    const msgs = [
      { role: "user" as const, content: "hi" },
      { role: "assistant" as const, content: "hello there" },
    ];
    expect(estimateTokens(msgs)).toBe(Math.ceil((2 + 11) / 4));
  });

  test("messages with content blocks", () => {
    const msgs = [
      {
        role: "assistant" as const,
        content: [
          { type: "text" as const, text: "block one" },
          { type: "text" as const, text: "block two" },
        ],
      },
    ];
    expect(estimateTokens(msgs)).toBe(Math.ceil(18 / 4));
  });
});

describe("countTokensAccurate (tiktoken-backed)", () => {
  beforeEach(() => {
    _resetTokenizerCache();
  });

  test("matches published cl100k_base counts within ±2", async () => {
    // Known: "hello world" → 2 tokens in cl100k_base.
    const n = await countTokensAccurate("hello world", "gpt-4");
    expect(Math.abs(n - 2)).toBeLessThanOrEqual(2);
  });

  test("counts tool_use blocks (name + JSON args)", async () => {
    const msgs = [
      {
        role: "assistant" as const,
        content: [
          { type: "text" as const, text: "calling tool" },
          {
            type: "tool_use" as const,
            id: "tu_1",
            name: "search",
            input: { query: "typescript tokenizer" },
          },
        ],
      },
    ];
    const n = await countTokensAccurate(msgs, "gpt-4");
    // Should be > heuristic floor and non-trivial.
    expect(n).toBeGreaterThan(5);
    expect(n).toBeLessThan(50);
  });

  test("falls back to chars/4 when tokenizer can't load", async () => {
    _forceLoaderFailure();
    const text = "the quick brown fox jumps over the lazy dog";
    const n = await countTokensAccurate(text, "gpt-4");
    expect(n).toBe(Math.ceil(text.length / 4));
  });

  test("fallback doesn't throw on messages either", async () => {
    _forceLoaderFailure();
    const msgs = [{ role: "user" as const, content: "hi there" }];
    const n = await countTokensAccurate(msgs, "default");
    expect(n).toBe(estimateTokens(msgs));
  });

  test("claude-3-5 maps to o200k_base and still returns a count", async () => {
    const n = await countTokensAccurate("function foo() { return 42; }", "claude-3-5");
    expect(n).toBeGreaterThan(0);
  });

  test("performance: 1 MB text counts within budget", async () => {
    // Prime once (wasm load counted here).
    const primed = await primeTokenizer("default");
    expect(primed).toBe(true);

    const oneMB = "const x = 42;\n".repeat(Math.ceil(1_000_000 / 14));
    // First call post-prime (tokenizer cached).
    const t1 = performance.now();
    const n1 = await countTokensAccurate(oneMB, "default");
    const elapsed1 = performance.now() - t1;
    expect(n1).toBeGreaterThan(0);
    // Budget covers the "first real encode" path. Windows CI runners are
    // ~3× slower than macOS/Linux; bumped from 500 → 1500 ms to keep the
    // matrix green without losing regression sensitivity.
    expect(elapsed1).toBeLessThan(1500);

    // Second call (fully cached).
    const t2 = performance.now();
    await countTokensAccurate(oneMB, "default");
    const elapsed2 = performance.now() - t2;
    expect(elapsed2).toBeLessThan(1500); // same encoder; should be comparable. Windows runner allowance.
  });
});
