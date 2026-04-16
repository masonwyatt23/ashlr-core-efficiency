# @ashlr/core-efficiency/anthropic

First-class Anthropic SDK integration for the ashlr toolchain. Bring
ashlr's token-saving MCP tools, genome RAG, and prompt-caching discipline
to any app built on `@anthropic-ai/sdk` or `@anthropic-ai/claude-agent-sdk`
in 2-3 lines.

## Install

```bash
bun add @ashlr/core-efficiency @anthropic-ai/sdk
# or, for stdio MCP callers:
bun add @ashlr/core-efficiency @anthropic-ai/claude-agent-sdk
```

`@anthropic-ai/sdk` is a peer dependency — bring your own version.

You also need the [ashlr-plugin](https://github.com/ashlrai/ashlr-plugin)
checked out somewhere `ashlrMcpConfig()` can find it (Claude Code install,
`~/Desktop/ashlr-plugin`, or `ASHLR_PLUGIN_ROOT`).

## API

### `ashlrMcpConfig(options?)`

Returns an array of stdio MCP server configs — one per ashlr plugin you
opt in to. Suitable for the Claude Agent SDK's `mcpServers` option.

```ts
import { ashlrMcpConfig } from "@ashlr/core-efficiency/anthropic";

const servers = ashlrMcpConfig();
// [
//   { type: 'stdio', name: 'ashlr-efficiency', command: 'bun', args: ['run', '/abs/path/servers/efficiency-server.ts'] },
//   { type: 'stdio', name: 'ashlr-genome',     command: 'bun', args: ['run', '/abs/path/servers/genome-server.ts']     },
//   ...
// ]
```

Options:

| Option | Default | Notes |
|---|---|---|
| `plugins` | all 10 | `['efficiency', 'genome', 'bash', 'diff', 'github', 'http', 'logs', 'orient', 'sql', 'tree']` |
| `pluginRoot` | auto-detect | Absolute path to an `ashlr-plugin` checkout. |
| `runtime` | `"bun"` | Command used to exec each server. Use `"tsx"` or `"node --import=tsx"` if you prefer. |

Auto-detect order:

1. `options.pluginRoot` if passed.
2. `ASHLR_PLUGIN_ROOT` environment variable.
3. `~/.claude/plugins/cache/ashlr-marketplace/ashlr/<latest-semver>/` (Claude Code plugin install).
4. `~/Desktop/ashlr-plugin` (developer default).

If none resolve, `AshlrPluginNotFoundError` is thrown with remediation
instructions.

There's also `ashlrMcpConfigRecord()` which returns the same data keyed by
name — handy for the Agent SDK's `Record<string, McpServerConfig>` shape.

### `withGenome(systemPrompt, cwd, opts?)`

Reads a project genome at `{cwd}/.ashlrcode/genome/` (walking up if
necessary), retrieves the top relevant sections, and returns a new system
prompt with those sections stitched above the caller's original prompt.
Uses the same `retrieveSectionsV2` used by ashlrcode — Ollama-backed
semantic search when available, TF-IDF fallback otherwise.

```ts
import { withGenome } from "@ashlr/core-efficiency/anthropic";

const system = await withGenome(
  "You are a senior engineer. Answer precisely.",
  process.cwd(),
  { query: "authentication flow", maxTokens: 1500 },
);
```

Options:

| Option | Default | Notes |
|---|---|---|
| `query` | `""` | Empty query selects the canonical "core" sections (north-star, current milestone, active strategies). |
| `maxTokens` | `2000` | Retrieval token budget. |
| `walkUp` | `true` | If no genome at `cwd`, walk up the tree. |

Returns the original prompt unchanged when no genome exists or retrieval
fails — safe to use unconditionally.

### `cacheBreakpoints(req)`

Adds `cache_control: { type: 'ephemeral' }` markers at the logical
static→dynamic boundaries of an Anthropic Messages request. Input is not
mutated.

```ts
import { cacheBreakpoints } from "@ashlr/core-efficiency/anthropic";

const req = cacheBreakpoints({
  system: longSystemPrompt,
  tools: toolDefinitions,
  messages: [
    { role: "user", content: bigProjectContext, cache: true }, // static
    { role: "user", content: "what does login.ts do?" },       // dynamic
  ],
});
await client.messages.create({ ...req, model: "claude-opus-4-5", max_tokens: 1024 });
```

Cache markers are placed on:

- The last `text` block of `system` (string prompts are promoted to a
  `{ type: 'text', text, cache_control }` block).
- The last tool definition in `tools` (covers the full tool list).
- The last `content` block of the last message flagged with `cache: true`
  (the static-prefix boundary).

Reference: [Anthropic prompt caching docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).

## Examples

See [`examples/anthropic-sdk/`](../../examples/anthropic-sdk/):

- `basic.ts` — minimal Agent-SDK stream with ashlr MCP tools.
- `with-genome.ts` — prepend genome context to the system prompt.
- `prompt-caching.ts` — add cache breakpoints and read the hit counters.

Each example runs with `bun run examples/anthropic-sdk/<name>.ts`.

## Design notes

- No hard dependency on `@anthropic-ai/sdk`. This module defines its own
  structural types; importing the SDK is the caller's concern.
- `ashlrMcpConfig()` is synchronous. Path resolution is cheap; MCP process
  spawn is the Agent SDK's job.
- `withGenome` fails open — a missing or corrupted genome returns the
  original prompt rather than throwing.
- `cacheBreakpoints` never emits more than 4 markers (Anthropic's cap):
  system, tools, static messages boundary. Per-message-content caching is
  opt-in via `cache: true` on the message.
