/**
 * Cross-agent shared session log.
 *
 * Any Ashlr-ecosystem coding agent (Claude Code, OpenHands, Goose, Aider,
 * ashlrcode) can `append` to a single JSONL file so that when the user
 * switches tools mid-task, the next tool can `read`/`tail` what the previous
 * one did. This solves the biggest multi-agent UX problem: context loss at
 * the agent boundary.
 *
 * Design constraints:
 *  - Zero runtime deps (Node/Bun stdlib only).
 *  - Concurrent-write safe via POSIX append-mode atomicity for writes
 *    smaller than PIPE_BUF (4KB on macOS/Linux).
 *  - Never throws on I/O errors — writes are best-effort so a broken log
 *    never breaks an agent.
 *  - Self-rotating at 10MB so the file can't grow unbounded.
 *  - `ASHLR_SESSION_LOG=0` disables all write ops (reads still work).
 *  - `ASHLR_SESSION_LOG_PATH` overrides the default location.
 */

import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  statSync,
  watch,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

import type {
  ReadOptions,
  SessionLogEntry,
  TailOptions,
} from "./types.ts";

export type {
  KnownAgent,
  KnownEvent,
  ReadOptions,
  SessionLogEntry,
  TailOptions,
} from "./types.ts";

// ---------- Configuration ------------------------------------------------

/**
 * Resolved session-log path. Respects `ASHLR_SESSION_LOG_PATH`, else
 * `$HOME/.ashlr/session-log.jsonl`.
 *
 * Evaluated once at import time. If a caller mutates `process.env` later,
 * they must use {@link resolveLogPath} explicitly.
 */
export const SESSION_LOG_PATH: string = resolveLogPath();

/** Size threshold (bytes) above which the log is rotated on next append. */
export const SESSION_LOG_MAX_BYTES = 10 * 1024 * 1024;

/** Internal: compute the configured log path from env + $HOME. */
function resolveLogPath(): string {
  const override = process.env.ASHLR_SESSION_LOG_PATH;
  if (override && override.length > 0) return override;
  return join(homedir(), ".ashlr", "session-log.jsonl");
}

/** Internal: read the path live so env changes after import are honored. */
function currentLogPath(): string {
  // Prefer the frozen constant when env hasn't changed; re-resolve otherwise
  // so tests that set ASHLR_SESSION_LOG_PATH work without reloading the module.
  const override = process.env.ASHLR_SESSION_LOG_PATH;
  if (override && override.length > 0) return override;
  return SESSION_LOG_PATH;
}

function writesDisabled(): boolean {
  return process.env.ASHLR_SESSION_LOG === "0";
}

// ---------- Append -------------------------------------------------------

/**
 * Append one entry to the shared session log.
 *
 * Behavior:
 *  1. If `ASHLR_SESSION_LOG=0` → no-op.
 *  2. Ensure the parent dir exists (`mkdir -p`).
 *  3. If the current file is larger than {@link SESSION_LOG_MAX_BYTES},
 *     rotate it first so this write starts a fresh file.
 *  4. Append exactly one JSON line.
 *  5. Auto-fill `ts` with `new Date().toISOString()` if the caller omitted it.
 *  6. Swallow all errors — a broken log must never break the agent.
 *
 * The write is atomic with respect to concurrent appenders on POSIX
 * filesystems as long as the serialized line stays under PIPE_BUF (4KB).
 * Keep `meta` small and you're safe.
 */
export async function append(
  entry: Omit<SessionLogEntry, "ts"> & { ts?: string },
): Promise<void> {
  if (writesDisabled()) return;
  try {
    const path = currentLogPath();
    ensureDir(path);
    maybeRotate(path);

    const filled: SessionLogEntry = {
      ...entry,
      ts: entry.ts ?? new Date().toISOString(),
    } as SessionLogEntry;

    const line = JSON.stringify(filled) + "\n";

    // `flag: 'a'` → POSIX O_APPEND. Kernel guarantees the offset is advanced
    // atomically under PIPE_BUF so concurrent writers never interleave bytes.
    appendFileSync(path, line, { flag: "a" });
  } catch (err) {
    // Never throw: log-and-move-on.
    // eslint-disable-next-line no-console
    console.error("[session-log]", err);
  }
}

// ---------- Read ---------------------------------------------------------

/**
 * Synchronously read entries from the log. Most-recent-first.
 *
 * Parses defensively: lines that fail `JSON.parse` are skipped, not thrown.
 * Empty / missing file returns `[]`.
 */
export function read(opts: ReadOptions = {}): SessionLogEntry[] {
  const path = currentLogPath();
  let raw: string;
  try {
    raw = readFileSync(path, "utf8");
  } catch {
    // File missing, unreadable, etc. — treat as empty.
    return [];
  }

  const lines = raw.split("\n");
  const out: SessionLogEntry[] = [];

  // Walk bottom-up so we can short-circuit on `limit`.
  const sinceIso = opts.since ? opts.since.toISOString() : undefined;
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i];
    if (!line) continue;
    let entry: SessionLogEntry;
    try {
      entry = JSON.parse(line) as SessionLogEntry;
    } catch {
      continue; // defensive: skip corrupted lines
    }
    if (!entry || typeof entry !== "object") continue;
    if (opts.agent && entry.agent !== opts.agent) continue;
    if (sinceIso && entry.ts < sinceIso) continue;
    out.push(entry);
    if (opts.limit && out.length >= opts.limit) break;
  }

  return out;
}

// ---------- Tail ---------------------------------------------------------

/**
 * Async iterator that yields new entries as they're appended.
 *
 * Implementation:
 *  - Seek to the current end-of-file offset.
 *  - `fs.watch` the file for changes.
 *  - On each change, read the bytes past our offset, split on newlines,
 *    parse, filter, yield.
 *  - Honors `opts.signal` for cooperative cancellation.
 *
 * Not bullet-proof on network filesystems or across rotation events. After
 * rotation, the watch fires and we transparently reset to offset 0 on the
 * new file.
 */
export async function* tail(
  opts: TailOptions = {},
): AsyncIterable<SessionLogEntry> {
  const path = currentLogPath();
  ensureDir(path);
  // Make sure the file exists so `fs.watch` has something to bind to.
  if (!existsSync(path)) writeFileSync(path, "");

  let offset = safeSize(path);

  // Backpressure channel: watcher pushes events, consumer pulls.
  const queue: Array<() => void> = [];
  let pending = 0;
  let done = false;

  const wakeup = () => {
    pending++;
    const resolve = queue.shift();
    if (resolve) resolve();
  };

  const watcher = watch(path, { persistent: false }, () => {
    wakeup();
  });

  const abortHandler = () => {
    done = true;
    wakeup();
  };
  if (opts.signal) {
    if (opts.signal.aborted) {
      watcher.close();
      return;
    }
    opts.signal.addEventListener("abort", abortHandler, { once: true });
  }

  try {
    while (!done) {
      // Wait for a change signal unless one is already buffered.
      if (pending === 0) {
        await new Promise<void>((resolve) => queue.push(resolve));
      }
      pending = Math.max(0, pending - 1);
      if (done) break;

      const size = safeSize(path);
      if (size < offset) {
        // File shrank → rotation occurred. Reset to start of new file.
        offset = 0;
      }
      if (size === offset) continue;

      const slice = readSlice(path, offset, size);
      offset = size;

      for (const line of slice.split("\n")) {
        if (!line) continue;
        let entry: SessionLogEntry;
        try {
          entry = JSON.parse(line) as SessionLogEntry;
        } catch {
          continue;
        }
        if (opts.agent && entry.agent !== opts.agent) continue;
        yield entry;
      }
    }
  } finally {
    watcher.close();
    if (opts.signal) opts.signal.removeEventListener("abort", abortHandler);
  }
}

// ---------- Rotate -------------------------------------------------------

/**
 * Force a rotation. Renames the current log to `<path>.1` (overwriting any
 * prior `.1`) and creates a fresh empty file at the original path.
 *
 * Idempotent: safe to call when the log doesn't exist (no-op).
 */
export function rotate(): void {
  const path = currentLogPath();
  try {
    if (!existsSync(path)) return;
    const prev = path + ".1";
    renameSync(path, prev);
    writeFileSync(path, "");
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error("[session-log] rotate failed:", err);
  }
}

// ---------- Internals ----------------------------------------------------

function ensureDir(path: string): void {
  const dir = dirname(path);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

function maybeRotate(path: string): void {
  const size = safeSize(path);
  if (size > SESSION_LOG_MAX_BYTES) rotate();
}

function safeSize(path: string): number {
  try {
    return statSync(path).size;
  } catch {
    return 0;
  }
}

function readSlice(path: string, start: number, end: number): string {
  if (end <= start) return "";
  try {
    const buf = readFileSync(path);
    return buf.slice(start, end).toString("utf8");
  } catch {
    return "";
  }
}
