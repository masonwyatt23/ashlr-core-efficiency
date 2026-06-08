# @ashlr/core-efficiency

Token-efficiency primitives for AI coding agents — genome RAG, multi-tier context compression, provider-aware budgeting, and prompt-caching helpers.

Extracted from [ashlrcode](https://github.com/ashlrai/ashlrcode) and shared by the `ashlrcode` CLI and the [ashlr-plugin](https://github.com/ashlrai/ashlr-plugin) for Claude Code.

**Platform support:** macOS / Linux / Windows on Bun >= 1.0 and Node >= 20.

---

## Modules

| Subpath | LOC | Purpose |
|---------|-----|---------|
| [`/genome`](./src/genome) | ~2,342 | Self-evolving project specs via RAG + scribe protocol. Manifest CRUD, TF-IDF/Ollama retrieval, fitness-based strategy evolution, mutation audit trail. |
| [`/compression`](./src/compression) | ~470 | 3-tier context compression: `autoCompact` (LLM summarize old turns), `snipCompact` (truncate tool results > 2 KB), `contextCollapse` (drop short/dup). `PromptPriority` enum. |
| [`/budget`](./src/budget) | ~50 | Provider-aware prompt budgeting. `getProviderContextLimit`, `systemPromptBudget`. |
| [`/tokens`](./src/tokens) | ~50 | Token estimation: `estimateTokensFromString`, `estimateTokensFromMessages`. |
| [`/anthropic`](./src/anthropic) | ~200 | Anthropic SDK helpers: `withGenome`, `cacheBreakpoints`, `ashlrMcpConfig`. |
| [`/session-log`](./src/session-log) | ~150 | Structured session event log (tool calls, costs, savings). |
| [`/local`](./src/local) | ~120 | Context-window manager for small-context local models. |
| [`/types`](./src/types) | ~60 | Shared types: `Message`, `ContentBlock`, `LLMSummarizer`, `StreamEvent`. |

---

## Install

```bash
# Bun (primary runtime)
bun add @ashlr/core-efficiency

# npm / pnpm / yarn
npm install @ashlr/core-efficiency
```

For local development against a checkout:

```bash
bun add file:../ashlr-core-efficiency
```

The package ships TypeScript source in `src/`. Bun runs it directly. For Node.js, compile first with `bun run build` (outputs to `dist/`).

---

## Quickstart

### compression

```typescript
import {
  autoCompact,
  snipCompact,
  contextCollapse,
  PromptPriority,
} from "@ashlr/core-efficiency/compression";

// Truncate any tool result that exceeds 2 KB (head + tail elided middle).
const trimmed = snipCompact(messages, { maxBytes: 2048 });

// Drop short or duplicate messages to reduce prompt size.
const collapsed = contextCollapse(messages);

// LLM-summarize old turns when approaching the context limit.
const compacted = await autoCompact(messages, summarizer, {
  targetTokens: 50_000,
  priority: PromptPriority.High,
});
```

### budget

```typescript
import {
  getProviderContextLimit,
  systemPromptBudget,
} from "@ashlr/core-efficiency/budget";

const limit = getProviderContextLimit("anthropic");     // 200_000
const budget = systemPromptBudget("anthropic", 0.05, 50_000);  // 5% floor, 50K cap
```

### tokens

```typescript
import {
  estimateTokensFromString,
  estimateTokensFromMessages,
} from "@ashlr/core-efficiency/tokens";

const n = estimateTokensFromString("Hello, world!");
const total = estimateTokensFromMessages(messages);  // walks ContentBlock[] incl. tool results
```

### genome

```typescript
import {
  retrieveSectionsV2,
  injectGenomeContext,
  genomeExists,
} from "@ashlr/core-efficiency/genome";

if (genomeExists(process.cwd())) {
  // retrieveSectionsV2(cwd, query, maxTokens) → relevant sections within budget.
  const sections = await retrieveSectionsV2(
    process.cwd(),
    "architecture overview",
    2000,
  );

  // Or inject directly into a prompt builder:
  // injectGenomeContext(builder, cwd, taskDescription, maxTokens) → tokens added.
  const added = await injectGenomeContext(
    builder,
    process.cwd(),
    "architecture overview",
    2000,
  );
}
```

### anthropic

```typescript
import Anthropic from "@anthropic-ai/sdk";
import { withGenome, cacheBreakpoints } from "@ashlr/core-efficiency/anthropic";

const client = new Anthropic();
const system = await withGenome("You are a senior engineer.", process.cwd());

const req = cacheBreakpoints({
  system,
  messages: [
    { role: "user", content: projectContext, cache: true },
    { role: "user", content: "What does login.ts do?" },
  ],
});

await client.messages.create({ ...req, model: "claude-sonnet-4-6", max_tokens: 1024 });
```

For stdio MCP tools via the Agent SDK:

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";
import { ashlrMcpConfigRecord } from "@ashlr/core-efficiency/anthropic";

const mcpServers = ashlrMcpConfigRecord({ plugins: ["efficiency"] });
for await (const msg of query({ prompt: "...", options: { mcpServers } })) {
  // ...
}
```

See [`examples/anthropic-sdk/`](./examples/anthropic-sdk/) for runnable scenarios.

### session-log

```typescript
import { append, read, tail, rotate } from "@ashlr/core-efficiency/session-log";

// Append one entry (ts is auto-filled if omitted).
await append({
  agent: "claude-code",
  event: "tool_call",
  tool: "ashlr__read",
  meta: { inputTokens: 120 },
});

// Read the most recent entries (most-recent-first).
const recent = read({ limit: 10 });

// Follow new entries as they're appended (async iterator).
for await (const entry of tail({ agent: "claude-code" })) {
  console.log(entry.event, entry.tool);
  break;
}

// Force a rotation (renames the log to <path>.1 and starts fresh).
rotate();
```

### local

```typescript
import { LocalContextWindow, buildResumeContext } from "@ashlr/core-efficiency/local";

// Configure for a small-context model (e.g. a 32K model with 4K reserved).
const ctx = new LocalContextWindow({ maxTokens: 28_000, compactThreshold: 0.75 });

// Add messages; add() returns false once compaction is needed.
for (const message of allMessages) {
  if (!ctx.add(message)) {
    await ctx.compact(summarizer);  // summarizer optional; tiers snip/collapse first
  }
}

const messages = ctx.getMessages();  // messages ready for the next LLM call
console.log(ctx.stats());            // { totalTokens, used, utilization, messageCount, compactions }

// Rebuild a compact orientation block when resuming a session.
const resume = await buildResumeContext(process.cwd());
```

Root barrel export (all subpaths re-exported):

```typescript
import {
  autoCompact,
  getProviderContextLimit,
  retrieveSectionsV2,
  estimateTokensFromString,
} from "@ashlr/core-efficiency";
```

---

## Compatibility

| Runtime | macOS | Linux | Windows |
|---------|-------|-------|---------|
| Bun >= 1.0 | Yes | Yes | Yes |
| Node >= 20 | Yes (compile first) | Yes (compile first) | Yes (compile first) |

Path separators are normalized internally; no Unix-only assumptions.

---

## Development

```bash
bun install
bun test          # ~17 unit tests (budget + tokens)
bun run typecheck
bun run build     # emit dist/ for Node consumers
```

Integration tests live in the [ashlrcode repo](https://github.com/ashlrai/ashlrcode) where 700+ tests run against real-world consumers.

---

## Design notes

- **`LLMSummarizer` interface**: `autoCompact` and genome `scribe` depend on a minimal `{ stream(ProviderRequest): AsyncGenerator<StreamEvent> }` contract, not a concrete router. Consumers inject their own provider.
- **`PromptPriority` enum**: 12 named slots (`Core=0` through `Undercover=95`). Numeric values are stable — raw-int callers continue to work across versions.
- **`estimateTokens`**: previously duplicated in three places. Now one implementation, two entry points: `FromString` and `FromMessages` (walks `ContentBlock[]` including `tool_use`/`tool_result`).
- **Source-first exports**: `main` and `exports` point to `src/`. Bun resolves `.ts` imports directly. For Node.js, run `bun run build` and consume from `dist/`. A `module` field mirrors `main` for bundlers that inspect it.

---

## Versioning

Follows semver. Breaking changes (removed exports, changed interfaces) go to major versions. Additive exports and bug fixes are minor/patch.

---

## License

MIT — see [LICENSE](./LICENSE).
