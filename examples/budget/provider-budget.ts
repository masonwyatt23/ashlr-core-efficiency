/**
 * provider-budget.ts — demonstrates @ashlr/core-efficiency/budget
 *
 * Subpath: @ashlr/core-efficiency/budget
 *
 * Shows how to query context window limits per provider and compute a
 * safe system-prompt budget with a percentage floor and an absolute cap.
 *
 * Run:
 *   bun run examples/budget/provider-budget.ts
 *
 * Windows note: no shell-outs; runs identically on macOS / Linux / Windows.
 */
import {
  getProviderContextLimit,
  systemPromptBudget,
} from "../../src/budget/index.ts";

const providers = ["anthropic", "openai", "ollama"] as const;

for (const provider of providers) {
  const limit = getProviderContextLimit(provider);
  // 5% floor, 50 000-token hard cap.
  const budget = systemPromptBudget(provider, 0.05, 50_000);
  console.log(
    `${provider.padEnd(10)} context=${limit.toLocaleString().padStart(9)} tokens  ` +
      `system-prompt budget=${budget.toLocaleString().padStart(8)} tokens`,
  );
}
