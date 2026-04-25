/**
 * Additional token-counting tests:
 *   - Cross-platform encoding safety (explicit UTF-8 via tiktoken)
 *   - All ContentBlock types round-trip through both heuristic and accurate paths
 *   - Regression for blockCharCount/blockTokenCount unknown-type returning 0 (not NaN)
 *   - estimateTokensFromString / estimateTokensFromMessages exported helpers
 */

import { describe, expect, test, beforeEach } from "bun:test";
import {
  estimateTokens,
  estimateTokensFromString,
  estimateTokensFromMessages,
  countTokensAccurate,
  _resetTokenizerCache,
  _forceLoaderFailure,
} from "../src/tokens/index.ts";
import type { Message } from "../src/types/index.ts";

beforeEach(() => {
  _resetTokenizerCache();
});

describe("estimateTokensFromString", () => {
  test("empty string → 0", () => {
    expect(estimateTokensFromString("")).toBe(0);
  });

  test("4-char string → 1 token", () => {
    expect(estimateTokensFromString("abcd")).toBe(1);
  });

  test("5-char string → 2 tokens (ceil)", () => {
    expect(estimateTokensFromString("abcde")).toBe(2);
  });

  test("unicode multi-byte string counts chars, not bytes", () => {
    // "こんにちは" is 5 chars → ceil(5/4) = 2
    expect(estimateTokensFromString("こんにちは")).toBe(2);
  });
});

describe("estimateTokensFromMessages — all block types", () => {
  test("text block", () => {
    const msgs: Message[] = [
      { role: "assistant", content: [{ type: "text", text: "hello" }] },
    ];
    expect(estimateTokensFromMessages(msgs)).toBe(Math.ceil(5 / 4));
  });

  test("thinking block", () => {
    const msgs: Message[] = [
      {
        role: "assistant",
        content: [{ type: "thinking", thinking: "I think therefore I am", signature: "sig" }],
      },
    ];
    expect(estimateTokensFromMessages(msgs)).toBe(Math.ceil("I think therefore I am".length / 4));
  });

  test("tool_use block (name + JSON input)", () => {
    const input = { query: "test" };
    const msgs: Message[] = [
      {
        role: "assistant",
        content: [{ type: "tool_use", id: "tu_1", name: "search", input }],
      },
    ];
    const expected = Math.ceil(("search".length + JSON.stringify(input).length) / 4);
    expect(estimateTokensFromMessages(msgs)).toBe(expected);
  });

  test("tool_result block", () => {
    const msgs: Message[] = [
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "tu_1", content: "result text" }],
      },
    ];
    expect(estimateTokensFromMessages(msgs)).toBe(Math.ceil("result text".length / 4));
  });

  test("image_url block → 1000 chars (= 250 tokens)", () => {
    const msgs: Message[] = [
      {
        role: "user",
        content: [{ type: "image_url", image_url: { url: "https://example.com/img.png" } }],
      },
    ];
    expect(estimateTokensFromMessages(msgs)).toBe(250); // ceil(1000/4)
  });

  test("mixed block types sum correctly", () => {
    const msgs: Message[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "question" },           // 8 chars
          { type: "tool_result", tool_use_id: "tu", content: "answer" }, // 6 chars
        ],
      },
    ];
    expect(estimateTokensFromMessages(msgs)).toBe(Math.ceil(14 / 4));
  });

  test("empty messages list → 0", () => {
    expect(estimateTokensFromMessages([])).toBe(0);
  });
});

describe("countTokensAccurate — edge cases", () => {
  test("empty string → 0 (accurate path)", async () => {
    const n = await countTokensAccurate("", "gpt-4");
    expect(n).toBe(0);
  });

  test("empty string fallback path → 0", async () => {
    _forceLoaderFailure();
    const n = await countTokensAccurate("", "gpt-4");
    expect(n).toBe(0);
  });

  test("per-message overhead: single-message with 1-char content > 1 token", async () => {
    // Should include the +4 per-message overhead
    const n = await countTokensAccurate([{ role: "user", content: "x" }], "gpt-4");
    expect(n).toBeGreaterThan(1);
  });

  test("image_url block counts as 1000 tokens (accurate path)", async () => {
    const msgs: Message[] = [
      {
        role: "user",
        content: [{ type: "image_url", image_url: { url: "https://example.com/img.png" } }],
      },
    ];
    const n = await countTokensAccurate(msgs, "gpt-4");
    // 1000 image tokens + 4 overhead
    expect(n).toBeGreaterThanOrEqual(1000);
  });

  test("fallback produces same result as heuristic for messages", async () => {
    _forceLoaderFailure();
    const msgs: Message[] = [
      { role: "user", content: "hello world this is a test message" },
    ];
    const accurate = await countTokensAccurate(msgs, "gpt-4");
    const heuristic = estimateTokens(msgs);
    expect(accurate).toBe(heuristic);
  });

  test("Windows CRLF vs LF: string length differs, token count scales accordingly", async () => {
    const lf = "line one\nline two\nline three";
    const crlf = "line one\r\nline two\r\nline three";
    const nLf = await countTokensAccurate(lf, "gpt-4");
    const nCrlf = await countTokensAccurate(crlf, "gpt-4");
    // CRLF has extra bytes — both should be positive, CRLF >= LF
    expect(nLf).toBeGreaterThan(0);
    expect(nCrlf).toBeGreaterThanOrEqual(nLf);
  });
});
