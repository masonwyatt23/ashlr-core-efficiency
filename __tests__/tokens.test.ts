import { expect, test, describe } from "bun:test";
import { estimateTokens } from "../src/tokens/index.ts";

describe("estimateTokens", () => {
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
