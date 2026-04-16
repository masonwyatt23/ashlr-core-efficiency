# @ashlr/core-efficiency/session-log

Cross-agent shared activity log. A single JSONL file at
`~/.ashlr/session-log.jsonl` that every Ashlr-ecosystem coding agent
(Claude Code, OpenHands, Goose, Aider, ashlrcode) appends to — so when the
user switches tools mid-task, the new tool can see what the previous one did.

## Why

The biggest multi-agent UX problem is context loss at the agent boundary.
Each agent keeps its own private transcript, so switching from (say)
Claude Code to Goose mid-feature loses everything the first agent learned.
This module is the minimum-viable shared substrate: one append-only JSONL
that each agent writes to on tool-calls / file-edits / observations, and
that any agent can read or tail to recover the shared history.

## Install / import

For now, import via the full subpath (a short-form subpath export will be
added in a follow-up `package.json` change):

```ts
import { append, read, tail, rotate } from "@ashlr/core-efficiency/src/session-log";
// or, relative from inside the monorepo:
import { append } from "./src/session-log/index.ts";
```

## API

```ts
// Constants
SESSION_LOG_PATH: string           // ~/.ashlr/session-log.jsonl (or env override)
SESSION_LOG_MAX_BYTES = 10 * 1024 * 1024

// Ops
append(entry): Promise<void>       // ts auto-filled; never throws
read(opts?):   SessionLogEntry[]   // most-recent-first; supports limit/agent/since
tail(opts?):   AsyncIterable<...>  // streams new entries; honors opts.signal
rotate():      void                // rename → .1 and start fresh
```

### `append`

```ts
await append({
  agent: "claude-code",
  event: "tool_call",
  tool: "Read",
  path: "/Users/me/project/src/foo.ts",
  summary: "Read src/foo.ts (1.2KB)",
  cwd: process.cwd(),
});
```

- `ts` is auto-filled with `new Date().toISOString()` if omitted.
- Serialized line size should stay **under 4KB** — POSIX append atomicity is
  only guaranteed for writes smaller than `PIPE_BUF` (4KB on macOS/Linux).
  Keep `meta` small.
- If `ASHLR_SESSION_LOG=0` is set, `append` is a no-op.
- Swallows all I/O errors and logs to `console.error` — a broken log will
  never break the agent.

### `read`

```ts
const recent = read({ limit: 20 });                        // last 20 entries
const mine   = read({ agent: "goose", limit: 50 });        // filter
const today  = read({ since: new Date(Date.now() - 864e5) });
```

Returns most-recent-first. Parses defensively — corrupted lines are
skipped, not thrown.

### `tail`

```ts
const ac = new AbortController();
for await (const entry of tail({ signal: ac.signal })) {
  console.log(entry.agent, entry.summary);
}
// somewhere else: ac.abort();
```

Seeks to EOF, watches the file, yields new entries as they appear.
Handles rotation transparently (resets to offset 0 when the file shrinks).

### `rotate`

Renames the current log to `<path>.1` (overwriting any prior `.1`) and
starts a fresh empty file. Called automatically when the file exceeds
`SESSION_LOG_MAX_BYTES`; exposed for manual use.

## Environment variables

| Var                       | Effect                                           |
|---------------------------|--------------------------------------------------|
| `ASHLR_SESSION_LOG=0`     | Disable all writes (reads still work).           |
| `ASHLR_SESSION_LOG_PATH`  | Override the default path (e.g. for tests).      |

## Concurrency

Multiple agents writing at the same time is the common case. We rely on
POSIX `O_APPEND` atomicity: `fs.appendFile` with `flag: 'a'` performs one
`write(2)` syscall per call, and the kernel guarantees that offset +
bytes-written is a single atomic step when the payload is under
`PIPE_BUF` (4KB). That's why entries must stay small.

On network filesystems (NFS, SMB) this guarantee does not hold — we do
not protect against that case. Keep the log on a local filesystem.

## Schema

See `types.ts`. The contract is **additive only**: new optional fields are
fine, existing fields never change meaning. Consumers must ignore unknown
fields.

## Zero dependencies

`node:fs`, `node:os`, `node:path`. That's it. Importing this module must
not pull in any third-party runtime dep, because every agent in the
ecosystem must be able to call it with minimum friction.
