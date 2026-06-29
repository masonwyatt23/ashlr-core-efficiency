/**
 * Strict types for the cross-agent shared session log.
 *
 * The log lives at `~/.ashlr/session-log.jsonl` and is appended to by any
 * coding agent in the Ashlr ecosystem (Claude Code, OpenHands, Goose, Aider,
 * ashlrcode). A single entry is ONE JSON object per line (JSONL).
 *
 * Keep this contract additive: new optional fields OK, never rename or
 * repurpose an existing field. Consumers must ignore unknown fields.
 */

/** Known agent identifiers. Additional string values are allowed for forward-compat. */
export type KnownAgent =
  | "claude-code"
  | "openhands"
  | "goose"
  | "aider"
  | "ashlrcode";

/** Known event types. Additional string values are allowed for forward-compat. */
export type KnownEvent =
  | "session_start"
  | "session_end"
  | "tool_call"
  | "file_edit"
  | "message"
  | "observation";

/**
 * A single entry in the shared session log.
 *
 * Field semantics:
 * - `ts`:     ISO-8601 UTC timestamp (e.g. `"2026-04-16T05:58:00.123Z"`).
 *             Auto-filled by `append` if omitted.
 * - `agent`:  Which agent produced the entry. See {@link KnownAgent}.
 * - `event`:  Category of event. See {@link KnownEvent}.
 * - `cwd`:    Absolute working directory at the time of the event. Optional.
 * - `session`: Opaque per-agent session id for grouping related entries.
 * - `tool`:   Name of the tool invoked (for `tool_call` / `file_edit`).
 * - `path`:   File path (for `file_edit`, `observation`, etc.).
 * - `summary`: Short (≤120 char) human-readable description for CLI tails.
 * - `meta`:   Free-form structured payload. Keep it small — entries should
 *             stay well under 4KB so POSIX append stays atomic.
 */
export interface SessionLogEntry {
  ts: string;
  agent: KnownAgent | (string & {});
  event: KnownEvent | (string & {});
  cwd?: string;
  session?: string;
  tool?: string;
  path?: string;
  summary?: string;
  meta?: Record<string, unknown>;
}

/**
 * Optional outcome telemetry fields that can be attached to a session_end entry.
 *
 * Callers set these on a `session_end` event so the shared log carries enough
 * context for `SessionOutcomeRecorder` to reconstruct the full trace without
 * reading a separate file.
 *
 * All fields are optional so existing log writers are not broken.
 */
export interface SessionOutcomeMeta {
  /** Genome generation number active during the session. */
  genome_version?: number;
  /** Compression tier active for the session (1–4). */
  compression_tier?: 1 | 2 | 3 | 4;
  /** Actual USD cost incurred. */
  cost_actual?: number;
  /** Budget (USD) the session was given. */
  cost_budget?: number;
  /** Cache savings realized (USD). */
  cache_savings_usd?: number;
  /** Actual wall-clock latency of the session (ms). */
  latency_actual_ms?: number;
  /** Latency SLA budget (ms). */
  latency_budget_ms?: number;
  /** Whether the test suite passed at session end. */
  tests_passed?: boolean;
}

/** Options for {@link read}. */
export interface ReadOptions {
  /** Cap on number of entries returned. Most-recent-first. */
  limit?: number;
  /** Only return entries where `entry.agent === agent`. */
  agent?: string;
  /** Only return entries with `ts >= since.toISOString()`. */
  since?: Date;
}

/** Options for {@link tail}. */
export interface TailOptions {
  /** Only yield entries where `entry.agent === agent`. */
  agent?: string;
  /** Aborts the async iterator when fired. */
  signal?: AbortSignal;
}
