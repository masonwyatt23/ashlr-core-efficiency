# @ashlr/core-efficiency/local

Context management for small-context local models (e.g., Qwen3-Coder at 32K).

## Problem

Local models have 32K context vs Claude's 200K. After reading 3-4 files, the context is full and the model hallucinates or crashes. Standard compaction strategies (designed for 200K) are too late — by the time you're at 90% of 32K, there's no room to maneuver.

## Solution

### `LocalContextWindow`

Manages token budget with aggressive compaction at 75% utilization (not 90% like Claude). Three-tier compaction, cheapest first:

1. **snipCompact** — truncate tool results >2KB (zero cost, applied on every `add()`)
2. **contextCollapse** — drop short/duplicate older messages (zero cost)
3. **autoCompact** — LLM-summarize old messages (only when the first two tiers aren't enough, and only if a summarizer is provided)

Genome sections are injected at startup as the highest-value context per token.

```typescript
import { LocalContextWindow } from "@ashlr/core-efficiency/local";

const ctx = new LocalContextWindow({ maxTokens: 28_000 }); // 32K - 4K response reserve
ctx.setSystemPrompt("You are a coding assistant.");
await ctx.injectGenome(process.cwd());

// In your agent loop:
const fits = ctx.add(userMessage);
if (!fits) await ctx.compact(optionalSummarizer);
const messages = ctx.getMessages();
const systemPrompt = ctx.getSystemPrompt();
```

### `buildResumeContext`

Rebuilds session context from the shared session-log + genome state. Gives the local model enough orientation (~500-1000 tokens) to continue productively after a context reset or agent switch.

```typescript
import { buildResumeContext } from "@ashlr/core-efficiency/local";

const resume = await buildResumeContext(process.cwd());
// resume.recentActivity   — last 10 session-log entries
// resume.genomeContext     — top genome sections
// resume.lastSessionSummary — 1-paragraph summary
```

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `maxTokens` | 28,000 | Usable context (model max minus response reserve) |
| `compactThreshold` | 0.75 | Compact when this fraction is used |
| `genomeTokenBudget` | 2,000 | Tokens reserved for genome in system prompt |
| `systemPromptBudget` | 1,000 | Tokens reserved for base system prompt |
