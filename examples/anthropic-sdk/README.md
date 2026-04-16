# Anthropic SDK examples

Three runnable examples showing how to wire `@ashlr/core-efficiency/anthropic`
into an Anthropic app.

## Setup

```bash
export ANTHROPIC_API_KEY=sk-ant-...
bun install @anthropic-ai/sdk @anthropic-ai/claude-agent-sdk
```

`basic.ts` uses the **Claude Agent SDK** (stdio MCP). The other two use
the **Messages SDK** directly. Install whichever you need.

## Examples

### `basic.ts`
Minimal integration. Asks Claude to read `package.json` via
`ashlr__read` (the token-saving snipCompact-truncating read).

```bash
bun run examples/anthropic-sdk/basic.ts
```

Requires the ashlr-plugin to be discoverable (installed in Claude Code,
checked out to `~/Desktop/ashlr-plugin`, or pointed at via
`ASHLR_PLUGIN_ROOT`).

### `with-genome.ts`
Prepends genome context to the system prompt. Run from a directory that
has `.ashlrcode/genome/` populated (or a parent of one):

```bash
cd /path/to/your/project
bun run /path/to/ashlr-core-efficiency/examples/anthropic-sdk/with-genome.ts
```

If no genome is found, `withGenome` returns the base prompt unchanged —
so this example is safe to run anywhere.

### `prompt-caching.ts`
Two back-to-back calls with a shared static prefix. The second call
should log `cache_read=<non-zero>`, proving the prefix was served from
cache. Cache markers are placed automatically by `cacheBreakpoints`.

```bash
bun run examples/anthropic-sdk/prompt-caching.ts
```

Expected output:
```
[call 1 (cold)] in=... cache_create=~6000 cache_read=0
[call 2 (warm)] in=... cache_create=0    cache_read=~6000
```
