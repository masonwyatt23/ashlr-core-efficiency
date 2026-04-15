# @ashlr/core-efficiency

Token-efficiency primitives extracted from [ashlrcode](https://github.com/...). Designed to be consumed by both `ashlrcode` (the CLI) and `ashlr-plugin` (the Claude Code plugin — WOZCODE-style but open-source).

## Modules

| Module | What it does | Status |
|--------|--------------|--------|
| `@ashlr/core-efficiency/tokens` | Single-impl token estimation (chars/4 heuristic, consolidates 3 duplicates) | placeholder |
| `@ashlr/core-efficiency/genome` | Self-evolving project specs via RAG + scribe protocol + fitness-based strategy evolution | placeholder |
| `@ashlr/core-efficiency/compression` | 3-tier context compression (autoCompact LLM-summarize → snipCompact truncate → contextCollapse dedupe) + `SystemPromptBuilder` | placeholder |
| `@ashlr/core-efficiency/budget` | Provider-aware prompt budgeting — `getProviderContextLimit`, `systemPromptBudget(provider, 0.05, 50k cap)` | placeholder |

## Extraction source

See `~/Desktop/ashlr-core-efficiency-extraction-map.md` for the concrete file-by-file, symbol-by-symbol extraction plan from ashlrcode. Top-level: ~3,410 LOC of real code + ~1,571 LOC of tests to port.

## Build / test

```bash
bun install
bun test
bun run typecheck
```

## Status

**Scaffold only.** Real module code lands in Phase B of the ashlr-plugin build plan.

Prerequisites in ashlrcode (Phase A prep refactors) before any code moves here:
1. Abstract `ProviderRouter` → `interface LLMSummarizer` so genome/scribe + compression/autoCompact don't hard-couple to the router.
2. Consolidate 3 copies of `estimateTokens` to one util.
3. Replace 11 magic-number priorities with a `PromptPriority` enum.
4. Add missing tests for `PROVIDER_CONTEXT_LIMITS` edge cases.
