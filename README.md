# @ashlr/core-efficiency

Token-efficiency primitives for AI coding agents. Extracted from [ashlrcode](https://github.com/masonwyatt23/ashlrcode) and consumed by both the `ashlrcode` CLI and the [ashlr-plugin](https://github.com/masonwyatt23/ashlr-plugin) for Claude Code.

**One library, multiple consumers.** Evolution happens in one place.

## What's inside

| Module | Size | Purpose |
|--------|------|---------|
| [`/genome`](./src/genome) | ~2,342 LOC | Self-evolving project specs via RAG + scribe protocol. Manifest CRUD, TF-IDF/Ollama retrieval, fitness-based strategy evolution, mutation audit trail. |
| [`/compression`](./src/compression) | ~470 LOC | 3-tier context compression: `autoCompact` (LLM summarize old turns), `snipCompact` (truncate tool results > 2KB), `contextCollapse` (drop short/dup). Plus `PromptPriority` enum. |
| [`/budget`](./src/budget) | ~50 LOC | Provider-aware prompt budgeting. `getProviderContextLimit`, `systemPromptBudget(provider, 0.05, 50K cap)`. |
| [`/tokens`](./src/tokens) | ~50 LOC | Single-impl token estimation (chars/4 heuristic). |
| [`/types`](./src/types) | ~60 LOC | `Message`, `ContentBlock`, `LLMSummarizer`, `StreamEvent`, `ProviderRequest`. |

## Install

```bash
bun add @ashlr/core-efficiency
# or from the repo directly during development:
bun add file:../ashlr-core-efficiency
```

## Use

```typescript
import {
  autoCompact,
  snipCompact,
  contextCollapse,
  PromptPriority,
} from "@ashlr/core-efficiency/compression";

import {
  getProviderContextLimit,
  systemPromptBudget,
} from "@ashlr/core-efficiency/budget";

import {
  retrieveSectionsV2,
  injectGenomeContext,
  genomeExists,
} from "@ashlr/core-efficiency/genome";

import { estimateTokensFromString } from "@ashlr/core-efficiency/tokens";
import type { Message, LLMSummarizer } from "@ashlr/core-efficiency/types";
```

Or import everything from the root barrel:

```typescript
import {
  autoCompact,
  getProviderContextLimit,
  retrieveSectionsV2,
  estimateTokensFromString,
} from "@ashlr/core-efficiency";
```

## Test

```bash
bun install
bun test       # ~17 tests (budget + tokens); genome/compression tests live in ashlrcode
bun run typecheck
```

Integration tests live in the [ashlrcode repo](https://github.com/masonwyatt23/ashlrcode) — that's where all 746 tests run against a real-world consumer.

## Design notes

- **`LLMSummarizer` interface**: `autoCompact` and genome `scribe` depend on a minimal `{ stream(ProviderRequest): AsyncGenerator<StreamEvent> }` contract, not a concrete router. Consumers inject their own provider. ashlrcode's `ProviderRouter` structurally satisfies it.
- **`PromptPriority` enum**: 12 named slots (Core=0 → Undercover=95). Numeric values are stable so raw-int callers still work.
- **`estimateTokens`**: previously duplicated in three places in ashlrcode. Now one implementation, two entry points: `FromString` and `FromMessages` (walks `ContentBlock[]` including tool_use/tool_result).
- **Genome `commands.ts`**: deliberately kept in ashlrcode (CLI layer, not library code).

## Anthropic SDK integration

Any app built on `@anthropic-ai/sdk` (Messages API) or
`@anthropic-ai/claude-agent-sdk` (stdio MCP) can pull in ashlr's tools,
genome RAG, and prompt caching in 2-3 lines:

```typescript
import {
  ashlrMcpConfig,   // auto-detect ashlr-plugin → stdio MCP server list
  withGenome,       // prepend genome RAG context to a system prompt
  cacheBreakpoints, // add ephemeral cache markers at static/dynamic seams
} from "@ashlr/core-efficiency/anthropic";
```

### Quick start

```typescript
import Anthropic from "@anthropic-ai/sdk";
import { withGenome, cacheBreakpoints } from "@ashlr/core-efficiency/anthropic";

const client = new Anthropic();
const system = await withGenome("You are a senior engineer.", process.cwd());

const req = cacheBreakpoints({
  system,
  messages: [
    { role: "user", content: projectContext, cache: true },
    { role: "user", content: "what does login.ts do?" },
  ],
});

await client.messages.create({ ...req, model: "claude-sonnet-4-5", max_tokens: 1024 });
```

For stdio MCP tools via the Agent SDK:

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";
import { ashlrMcpConfigRecord } from "@ashlr/core-efficiency/anthropic";

const mcpServers = ashlrMcpConfigRecord();  // all 10 plugins, auto-detected
for await (const msg of query({ prompt: "...", options: { mcpServers } })) { /* ... */ }
```

`ashlrMcpConfig` auto-detects the plugin root in this order:
`ASHLR_PLUGIN_ROOT` → `~/.claude/plugins/cache/ashlr-marketplace/ashlr/<latest>/`
→ `~/Desktop/ashlr-plugin`. Throws `AshlrPluginNotFoundError` with
remediation steps if nothing is found.

See [`src/anthropic/README.md`](./src/anthropic/README.md) for the full API
and [`examples/anthropic-sdk/`](./examples/anthropic-sdk/) for runnable
scenarios (basic MCP, genome injection, prompt caching with hit-rate
reporting).

`@anthropic-ai/sdk` is an **optional peer dependency** — callers bring
their own version.

## License

MIT — see [LICENSE](./LICENSE).
