import { expect, test, describe } from "bun:test";
import {
  PROVIDER_CONTEXT_LIMITS,
  getProviderContextLimit,
  systemPromptBudget,
} from "../src/budget/index.ts";

describe("PROVIDER_CONTEXT_LIMITS", () => {
  test("covers all known providers", () => {
    for (const p of ["xai", "anthropic", "openai", "ollama", "groq", "deepseek", "lmstudio", "openrouter"]) {
      expect(PROVIDER_CONTEXT_LIMITS[p]).toBeGreaterThan(0);
    }
  });

  test("xai has 2M context", () => {
    expect(PROVIDER_CONTEXT_LIMITS.xai).toBe(2_000_000);
  });
});

describe("getProviderContextLimit", () => {
  test("case-insensitive lookup", () => {
    expect(getProviderContextLimit("Anthropic")).toBe(200_000);
    expect(getProviderContextLimit("ANTHROPIC")).toBe(200_000);
  });

  test("falls back to 128K for unknown providers", () => {
    expect(getProviderContextLimit("made-up-provider")).toBe(128_000);
  });
});

describe("systemPromptBudget", () => {
  test("5% of xai 2M caps at 50K", () => {
    expect(systemPromptBudget("xai")).toBe(50_000);
  });

  test("5% of Anthropic 200K is 10K", () => {
    expect(systemPromptBudget("anthropic")).toBe(10_000);
  });

  test("5% of ollama 32K is 1.6K", () => {
    expect(systemPromptBudget("ollama")).toBe(1_600);
  });

  test("custom ratio respected", () => {
    expect(systemPromptBudget("anthropic", 0.1)).toBe(20_000);
  });

  test("custom cap respected", () => {
    expect(systemPromptBudget("xai", 0.05, 10_000)).toBe(10_000);
  });
});
