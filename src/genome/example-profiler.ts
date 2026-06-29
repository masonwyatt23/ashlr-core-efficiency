/**
 * Example Profiler — persistence + ranking substrate for the adaptive few-shot
 * example selector (example-selector.ts).
 *
 * Responsibilities (no LLM calls, no network):
 *   - Define the on-disk ExampleProfile record and its JSONL persistence.
 *   - Aggregate the append-only JSONL log into a latest-wins view keyed by
 *     (task_type, provider, model, example_id).
 *   - Provide an EMA update for success_rate that down-weights early noise.
 *   - Rank candidate profiles by a blend of learned utility (success_rate,
 *     latency, recency) and optional semantic relevance to a query embedding.
 *
 * Profiles live at ~/.ashlr/examples-profile.jsonl (append-only). The selector
 * loads the whole log and folds it; persistProfile appends one line per update.
 */

import { existsSync } from "fs";
import { appendFile, mkdir, readFile } from "fs/promises";
import { dirname, join } from "path";
import { homedir } from "os";
import { cosineSimilarity } from "./embeddings.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Coarse task categories used to scope example rankings. Free-form strings are
 * also accepted (the union is open via the `string` member) so callers can
 * introduce new task types without a code change.
 */
export type TaskType =
  | "code_generation"
  | "code_review"
  | "refactor"
  | "debug"
  | "test_writing"
  | "tool_call"
  | "error_recovery"
  | "explanation"
  | "general"
  | (string & {});

/**
 * A learned profile for a single few-shot example under a specific
 * (task_type, provider, model) context. Persisted as one JSONL line.
 */
export interface ExampleProfile {
  /** Task category this profile applies to. */
  task_type: TaskType;
  /** Provider id (e.g. "anthropic", "openai", "ollama"). */
  provider: string;
  /** Model id within the provider. */
  model: string;
  /** Stable example identifier (matches FewShotExample.id). */
  example_id: string;
  /** Pre-computed example embedding for relevance ranking (may be empty). */
  embedding: number[];
  /** Approximate token cost of injecting this example. */
  token_cost: number;
  /** EMA of observed success (0..1). Cold-start default is 0.5. */
  success_rate: number;
  /** Mean latency delta (ms) attributable to including this example. */
  latency_impact_ms: number;
  /** Number of post-call outcomes folded into this profile. */
  observation_count: number;
  /** ISO-8601 timestamp of last update. */
  updated_at: string;
}

/** A profile paired with its computed ranking score and components. */
export interface RankedExample {
  profile: ExampleProfile;
  /** Combined ranking score (higher = better). */
  score: number;
  /** Semantic relevance component (0 when no query embedding). */
  relevance: number;
  /** Learned utility component derived from success_rate/latency/recency. */
  utility: number;
}

/** The composite key uniquely identifying a profile. */
export function profileKey(p: {
  task_type: TaskType;
  provider: string;
  model: string;
  example_id: string;
}): string {
  return `${p.task_type}::${p.provider}::${p.model}::${p.example_id}`;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

export interface CreateProfileInput {
  task_type: TaskType;
  provider: string;
  model: string;
  example_id: string;
  embedding?: number[];
  token_cost?: number;
  success_rate?: number;
  latency_impact_ms?: number;
  observation_count?: number;
  updated_at?: string;
}

/**
 * Build an ExampleProfile, filling neutral cold-start defaults for any field
 * not supplied. Pure — performs no I/O.
 */
export function createProfile(input: CreateProfileInput): ExampleProfile {
  return {
    task_type: input.task_type,
    provider: input.provider,
    model: input.model,
    example_id: input.example_id,
    embedding: input.embedding ?? [],
    token_cost: input.token_cost ?? 0,
    success_rate: clamp01(input.success_rate ?? 0.5),
    latency_impact_ms: input.latency_impact_ms ?? 0,
    observation_count: Math.max(0, input.observation_count ?? 0),
    updated_at: input.updated_at ?? new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// EMA success-rate update
// ---------------------------------------------------------------------------

/**
 * Update a running success-rate EMA.
 *
 * The smoothing factor adapts to how many observations have been seen: early
 * observations move the estimate a lot (fast cold-start), later observations
 * move it little (stable long-run). alpha = 1 / min(observationCount, CAP),
 * where CAP bounds the effective window so the estimate keeps tracking drift.
 *
 * @param prev    previous success_rate (0..1)
 * @param outcome the new observation (0..1; e.g. 1 = success, 0 = failure)
 * @param observationCount the new total observation count (>= 1)
 */
export function emaSuccessRate(prev: number, outcome: number, observationCount: number): number {
  const CAP = 20;
  const n = Math.max(1, observationCount);
  const alpha = 1 / Math.min(n, CAP);
  return clamp01(prev * (1 - alpha) + clamp01(outcome) * alpha);
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

const PROFILE_FILE = "examples-profile.jsonl";

function defaultProfilePath(): string {
  return join(homedir(), ".ashlr", PROFILE_FILE);
}

/**
 * Load all profile records from the default path
 * (~/.ashlr/examples-profile.jsonl). Returns [] when the file is absent or
 * corrupt — never throws on read.
 */
export async function loadProfiles(): Promise<ExampleProfile[]> {
  return loadProfilesFromPath(defaultProfilePath());
}

/**
 * Load all profile records from an explicit JSONL path. Malformed lines are
 * skipped rather than failing the whole load.
 */
export async function loadProfilesFromPath(path: string): Promise<ExampleProfile[]> {
  if (!existsSync(path)) return [];
  let raw: string;
  try {
    raw = await readFile(path, "utf-8");
  } catch {
    return [];
  }

  const out: ExampleProfile[] = [];
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const parsed = JSON.parse(trimmed);
      if (isExampleProfile(parsed)) out.push(parsed);
    } catch {
      // skip malformed line
    }
  }
  return out;
}

/**
 * Append a profile update to the default JSONL log. The log is append-only;
 * aggregateProfiles folds it into a latest-wins view on read. Creates the
 * parent directory if needed.
 */
export async function persistProfile(profile: ExampleProfile): Promise<void> {
  const path = defaultProfilePath();
  await mkdir(dirname(path), { recursive: true });
  await appendFile(path, JSON.stringify(profile) + "\n", "utf-8");
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

/**
 * Fold an append-only profile log into a latest-wins map keyed by
 * profileKey(). Later entries (by array order, then updated_at) win, so the
 * most recent observation for each (task, provider, model, example) survives.
 */
export function aggregateProfiles(profiles: ExampleProfile[]): Map<string, ExampleProfile> {
  const map = new Map<string, ExampleProfile>();
  for (const p of profiles) {
    const key = profileKey(p);
    const existing = map.get(key);
    if (!existing) {
      map.set(key, p);
      continue;
    }
    // Prefer the record with the higher observation_count; tie-break on
    // updated_at so a later write wins.
    if (
      p.observation_count > existing.observation_count ||
      (p.observation_count === existing.observation_count && p.updated_at >= existing.updated_at)
    ) {
      map.set(key, p);
    }
  }
  return map;
}

// ---------------------------------------------------------------------------
// Ranking
// ---------------------------------------------------------------------------

/**
 * Rank profiles best-first by a blend of learned utility and (optional)
 * semantic relevance to `queryEmbedding`.
 *
 *   utility   = success_rate, mildly penalized by positive latency impact and
 *               discounted toward the 0.5 prior when observation_count is low
 *   relevance = cosine(query, profile.embedding) mapped to 0..1; 0 when no
 *               query embedding or no profile embedding is available
 *   score     = relevance present ? 0.6*relevance + 0.4*utility : utility
 */
export function rankProfiles(
  profiles: ExampleProfile[],
  queryEmbedding: number[] | null,
): RankedExample[] {
  const hasQuery = Array.isArray(queryEmbedding) && queryEmbedding.length > 0;

  const ranked: RankedExample[] = profiles.map((profile) => {
    const utility = computeUtility(profile);

    let relevance = 0;
    if (hasQuery && profile.embedding.length === queryEmbedding!.length) {
      // cosine in [-1, 1] -> [0, 1]
      relevance = (cosineSimilarity(queryEmbedding!, profile.embedding) + 1) / 2;
    }

    const score = hasQuery ? 0.6 * relevance + 0.4 * utility : utility;
    return { profile, score, relevance, utility };
  });

  ranked.sort((a, b) => b.score - a.score);
  return ranked;
}

/**
 * Learned utility of a profile in [0, 1]. Blends EMA success_rate with a small
 * latency penalty, then shrinks toward the 0.5 prior in proportion to how few
 * observations back the estimate (confidence weighting).
 */
function computeUtility(profile: ExampleProfile): number {
  const latencyPenalty = profile.latency_impact_ms > 0
    ? Math.min(0.2, profile.latency_impact_ms / 10_000)
    : 0;
  const raw = clamp01(profile.success_rate - latencyPenalty);

  // Confidence shrinkage: with n observations, weight = n / (n + K).
  const K = 5;
  const n = Math.max(0, profile.observation_count);
  const confidence = n / (n + K);
  return clamp01(0.5 * (1 - confidence) + raw * confidence);
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

function clamp01(x: number): number {
  if (!Number.isFinite(x)) return 0;
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function isExampleProfile(v: unknown): v is ExampleProfile {
  if (typeof v !== "object" || v === null) return false;
  const o = v as Record<string, unknown>;
  return (
    typeof o.task_type === "string" &&
    typeof o.provider === "string" &&
    typeof o.model === "string" &&
    typeof o.example_id === "string" &&
    Array.isArray(o.embedding) &&
    typeof o.token_cost === "number" &&
    typeof o.success_rate === "number" &&
    typeof o.observation_count === "number"
  );
}
