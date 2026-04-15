// Phase B: extract from ashlrcode/src/agent/context.ts:28-47 and cli.ts:274-281

export const SYSTEM_PROMPT_BUDGET_RATIO = 0.05;
export const SYSTEM_PROMPT_BUDGET_CAP = 50_000;

export const PROVIDER_CONTEXT_LIMITS: Record<string, number> = {
  xai: 2_000_000,
  anthropic: 200_000,
  openai: 128_000,
  ollama: 32_000,
  groq: 128_000,
  deepseek: 128_000,
  // Added for local-ai-stack integration:
  lmstudio: 262_144, // Qwen3-Coder-30B max advertised context
  openrouter: 1_000_000, // varies per model; this is the ceiling
};

export function getProviderContextLimit(providerName: string): number {
  const key = providerName.toLowerCase();
  return PROVIDER_CONTEXT_LIMITS[key] ?? 128_000;
}

export function systemPromptBudget(
  providerName: string,
  ratio: number = SYSTEM_PROMPT_BUDGET_RATIO,
  cap: number = SYSTEM_PROMPT_BUDGET_CAP,
): number {
  const limit = getProviderContextLimit(providerName);
  return Math.min(Math.floor(limit * ratio), cap);
}
