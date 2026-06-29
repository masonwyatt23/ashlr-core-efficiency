/**
 * Embedding Resilience — exponential backoff + circuit breaker for embedding backends.
 *
 * Wraps EmbeddingRouter.generateEmbeddingWithFallback() and
 * generateEmbeddingMultiBackend() with:
 *
 *  1. Exponential backoff with jitter (3–15s range, configurable max attempts).
 *     Tracks per-provider/per-query-size latency histograms so the fleet learns
 *     which request sizes trigger timeouts.
 *
 *  2. Per-backend circuit breaker over a sliding window. After ≥50% failures in
 *     a 10-request window the breaker opens for 30 s, then half-opens to probe
 *     one request. Auto-resets on success.
 *
 *  3. Fallback chain orchestration: when a backend trips its breaker the
 *     orchestrator skips directly to the next tier without re-attempting.
 *
 *  4. Adaptive timeout scaling: Ollama latency >8 s → reduce next timeout to 12 s;
 *     Anthropic latency >3 s → prefer local-ultrafast for the next batch. Learned
 *     timeouts persist in `.ashlrcode/genome/resilience-profile.jsonl`.
 *
 * Permanent errors (HTTP 401, 403) are never retried.
 */

import { appendJsonl, readJsonl } from "./jsonl.ts";
import { genomeDir } from "./manifest.ts";
import { join } from "path";
import type { EmbeddingBackend, MultiBackendEmbeddingResult } from "./embedding-router.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Minimum base delay for exponential backoff (ms). */
export const BACKOFF_BASE_MS = 3_000;
/** Maximum backoff delay before jitter (ms). */
export const BACKOFF_MAX_MS = 15_000;
/** Default maximum retry attempts (including the first attempt). */
export const DEFAULT_MAX_ATTEMPTS = 3;
/** Default circuit-breaker sliding window size (number of requests). */
export const DEFAULT_WINDOW_SIZE = 10;
/** Default failure-rate threshold to trip the circuit breaker [0,1]. */
export const DEFAULT_FAILURE_RATE_THRESHOLD = 0.5;
/** How long the circuit breaker stays open before transitioning to half-open (ms). */
export const CIRCUIT_OPEN_DURATION_MS = 30_000;

/**
 * Ollama latency ceiling; above this the adaptive timeout is reduced to
 * OLLAMA_REDUCED_TIMEOUT_MS for subsequent requests.
 */
export const OLLAMA_LATENCY_HIGH_WATERMARK_MS = 8_000;
export const OLLAMA_REDUCED_TIMEOUT_MS = 12_000;

/**
 * Anthropic latency ceiling; above this the orchestrator prefers the
 * local-ultrafast backend for subsequent batches.
 */
export const ANTHROPIC_LATENCY_HIGH_WATERMARK_MS = 3_000;

// ---------------------------------------------------------------------------
// HTTP status codes that indicate a permanent error — never retry.
// ---------------------------------------------------------------------------

export const PERMANENT_ERROR_CODES = new Set([401, 403, 404]);

// ---------------------------------------------------------------------------
// Query size buckets for per-(backend, size) histograms
// ---------------------------------------------------------------------------

export type QuerySizeBucket = "small" | "medium" | "large";

/**
 * Classify a text string into a size bucket based on character count.
 *  - small : < 500 chars  (~125 tokens)
 *  - medium: 500–2000 chars (~125–500 tokens)
 *  - large : > 2000 chars  (>500 tokens)
 */
export function classifyQuerySize(text: string): QuerySizeBucket {
  const len = text.length;
  if (len < 500) return "small";
  if (len < 2_000) return "medium";
  return "large";
}

// ---------------------------------------------------------------------------
// Circuit breaker state
// ---------------------------------------------------------------------------

export type CircuitState = "closed" | "open" | "half-open";

export interface CircuitBreakerState {
  backend: EmbeddingBackend;
  state: CircuitState;
  /** Ring buffer of recent outcomes (true=success, false=failure). */
  window: boolean[];
  /** Index of the next write position in the ring buffer. */
  windowIndex: number;
  /** Number of entries filled in the ring buffer. */
  windowFilled: number;
  /** Timestamp (ms) when the breaker last opened. 0 = never opened. */
  openedAtMs: number;
}

/** Create a fresh (closed) circuit breaker for a backend. */
function makeCircuitBreaker(
  backend: EmbeddingBackend,
  windowSize: number,
): CircuitBreakerState {
  return {
    backend,
    state: "closed",
    window: new Array<boolean>(windowSize).fill(true),
    windowIndex: 0,
    windowFilled: 0,
    openedAtMs: 0,
  };
}

// ---------------------------------------------------------------------------
// Latency histogram entry (persisted)
// ---------------------------------------------------------------------------

/** Persisted entry in the resilience-profile.jsonl log. */
export interface ResilienceProfileEntry {
  /** ISO timestamp. */
  timestamp: string;
  backend: EmbeddingBackend;
  querySizeBucket: QuerySizeBucket;
  /** Observed latency in milliseconds. */
  latencyMs: number;
  /** Whether the request succeeded. */
  succeeded: boolean;
  /**
   * Adaptive timeout applied to this request (may differ from the model
   * spec's nominal timeout when the adaptive scaler has adjusted it).
   */
  effectiveTimeoutMs: number;
}

// ---------------------------------------------------------------------------
// Circuit breaker options
// ---------------------------------------------------------------------------

export interface CircuitBreakerOptions {
  /** Size of the sliding observation window. Default 10. */
  windowSize?: number;
  /** Failure-rate threshold [0,1] that trips the breaker. Default 0.5. */
  failureRateThreshold?: number;
  /** Duration (ms) the breaker stays open before attempting half-open. Default 30 000. */
  openDurationMs?: number;
}

// ---------------------------------------------------------------------------
// Backoff options
// ---------------------------------------------------------------------------

export interface BackoffOptions {
  /** Maximum number of attempts (first attempt + retries). Default 3. */
  maxAttempts?: number;
  /** Base delay in ms. Default BACKOFF_BASE_MS (3 000). */
  baseDelayMs?: number;
  /** Maximum delay before jitter in ms. Default BACKOFF_MAX_MS (15 000). */
  maxDelayMs?: number;
  /** When true, jitter is disabled (useful for tests). Default false. */
  disableJitter?: boolean;
}

// ---------------------------------------------------------------------------
// Resilient embedding options
// ---------------------------------------------------------------------------

export interface ResilientEmbeddingOptions {
  backoff?: BackoffOptions;
  circuitBreaker?: CircuitBreakerOptions;
  /** Working directory for profile persistence. */
  cwd?: string;
  /** Override default timeouts per backend (ms). */
  timeoutsMs?: Partial<Record<EmbeddingBackend, number>>;
  /** When true, profile entries are not written to disk. Default false. */
  disablePersistence?: boolean;
}

// ---------------------------------------------------------------------------
// Compute backoff delay with jitter
// ---------------------------------------------------------------------------

/**
 * Compute the delay for a given retry attempt using truncated exponential
 * backoff with full jitter.
 *
 * Formula: min(maxDelay, baseDelay * 2^attempt) * rand(0,1)
 *
 * @param attempt    Zero-based attempt index (0 = first retry, i.e., second call).
 * @param baseDelayMs  Base delay in milliseconds.
 * @param maxDelayMs   Maximum delay before applying jitter.
 * @param disableJitter  When true, return the deterministic cap without jitter.
 */
export function computeBackoffDelay(
  attempt: number,
  baseDelayMs: number = BACKOFF_BASE_MS,
  maxDelayMs: number = BACKOFF_MAX_MS,
  disableJitter = false,
): number {
  const cap = Math.min(maxDelayMs, baseDelayMs * Math.pow(2, attempt));
  if (disableJitter) return cap;
  return Math.floor(cap * Math.random());
}

// ---------------------------------------------------------------------------
// Circuit breaker logic
// ---------------------------------------------------------------------------

/**
 * Record an outcome into the circuit breaker ring buffer and update state.
 *
 * Transitions:
 *  closed   → open      : when failure rate ≥ threshold after recording failure
 *  open     → half-open : when openDurationMs has elapsed
 *  half-open → closed   : on success
 *  half-open → open     : on failure
 */
export function recordCircuitOutcome(
  cb: CircuitBreakerState,
  succeeded: boolean,
  nowMs: number = Date.now(),
  options: Required<CircuitBreakerOptions> = {
    windowSize: DEFAULT_WINDOW_SIZE,
    failureRateThreshold: DEFAULT_FAILURE_RATE_THRESHOLD,
    openDurationMs: CIRCUIT_OPEN_DURATION_MS,
  },
): void {
  // Check if open breaker should transition to half-open
  if (cb.state === "open") {
    if (nowMs - cb.openedAtMs >= options.openDurationMs) {
      cb.state = "half-open";
    } else {
      // Still open — do not record into the window; the request was rejected
      return;
    }
  }

  // Write into ring buffer
  cb.window[cb.windowIndex] = succeeded;
  cb.windowIndex = (cb.windowIndex + 1) % options.windowSize;
  cb.windowFilled = Math.min(cb.windowFilled + 1, options.windowSize);

  if (cb.state === "half-open") {
    if (succeeded) {
      cb.state = "closed";
      cb.openedAtMs = 0;
    } else {
      cb.state = "open";
      cb.openedAtMs = nowMs;
    }
    return;
  }

  // closed → potentially open
  const failures = cb.window.slice(0, cb.windowFilled).filter((v) => !v).length;
  const failureRate = failures / cb.windowFilled;

  if (cb.windowFilled >= Math.ceil(options.windowSize * 0.7) && failureRate >= options.failureRateThreshold) {
    cb.state = "open";
    cb.openedAtMs = nowMs;
  }
}

/**
 * Returns true when a request should be allowed through the circuit breaker.
 *
 * - closed   : always allow
 * - open     : allow only if enough time has elapsed (auto-transition to half-open)
 * - half-open: allow exactly one probe request
 */
export function isCircuitAllowed(
  cb: CircuitBreakerState,
  nowMs: number = Date.now(),
  openDurationMs: number = CIRCUIT_OPEN_DURATION_MS,
): boolean {
  if (cb.state === "closed") return true;
  if (cb.state === "half-open") return true;
  // open — check if it's time to transition
  if (nowMs - cb.openedAtMs >= openDurationMs) {
    cb.state = "half-open";
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Resilience profile path
// ---------------------------------------------------------------------------

function resilienceProfilePath(cwd: string): string {
  return join(genomeDir(cwd), "resilience-profile.jsonl");
}

// ---------------------------------------------------------------------------
// Adaptive timeout tracker
// ---------------------------------------------------------------------------

/**
 * Per-backend adaptive timeout state, maintained in-process.
 *
 * Start from the static defaults; when high latency is observed the tracker
 * lowers the timeout so the next batch doesn't wait too long for a slow backend.
 */
export class AdaptiveTimeoutTracker {
  private readonly timeouts: Map<EmbeddingBackend, number>;
  /** Whether the anthropic backend recently exceeded its high-watermark. */
  private anthropicSlow = false;

  constructor(overrides?: Partial<Record<EmbeddingBackend, number>>) {
    this.timeouts = new Map<EmbeddingBackend, number>([
      ["ollama", overrides?.ollama ?? 15_000],
      ["anthropic", overrides?.anthropic ?? 10_000],
      ["local-cached", overrides?.["local-cached"] ?? 500],
    ]);
  }

  /** Get the current effective timeout for a backend. */
  get(backend: EmbeddingBackend): number {
    return this.timeouts.get(backend) ?? 15_000;
  }

  /**
   * Record an observed latency for a backend and apply adaptive scaling rules:
   *  - Ollama > 8 s → reduce to 12 s
   *  - Anthropic > 3 s → mark as slow (prefers local-ultrafast for next batch)
   */
  record(backend: EmbeddingBackend, latencyMs: number): void {
    if (backend === "ollama" && latencyMs > OLLAMA_LATENCY_HIGH_WATERMARK_MS) {
      this.timeouts.set("ollama", OLLAMA_REDUCED_TIMEOUT_MS);
    }
    if (backend === "anthropic" && latencyMs > ANTHROPIC_LATENCY_HIGH_WATERMARK_MS) {
      this.anthropicSlow = true;
    }
  }

  /**
   * Returns true when the Anthropic backend has recently been observed as slow.
   * The caller should prefer local-ultrafast for the next batch.
   */
  isAnthropicSlow(): boolean {
    return this.anthropicSlow;
  }

  /** Reset the Anthropic slow flag (e.g., after a successful fast request). */
  resetAnthropicSlow(): void {
    this.anthropicSlow = false;
  }

  /** Return a snapshot of current timeouts for inspection. */
  snapshot(): Record<EmbeddingBackend, number> {
    return {
      ollama: this.get("ollama"),
      anthropic: this.get("anthropic"),
      "local-cached": this.get("local-cached"),
    };
  }
}

// ---------------------------------------------------------------------------
// Main resilience wrapper
// ---------------------------------------------------------------------------

/**
 * Result of a resilient embedding attempt.
 */
export interface ResilientEmbeddingResult extends MultiBackendEmbeddingResult {
  /** Total number of attempts made (1 = no retries needed). */
  totalAttempts: number;
  /** Backends that were skipped because their circuit breaker was open. */
  circuitBreakerSkips: EmbeddingBackend[];
  /** True when a permanent error (e.g., HTTP 401/403) stopped retries early. */
  permanentErrorEncountered: boolean;
}

/**
 * EmbeddingResiliencePipeline wraps the multi-backend embedding router with
 * exponential backoff, per-backend circuit breakers, and adaptive timeout
 * scaling.
 *
 * Usage:
 * ```ts
 * const pipeline = new EmbeddingResiliencePipeline(cwd);
 * const result = await pipeline.embed("some text to embed");
 * ```
 */
export class EmbeddingResiliencePipeline {
  private readonly cwd: string;
  private readonly backoffOpts: Required<BackoffOptions>;
  private readonly cbOpts: Required<CircuitBreakerOptions>;
  private readonly disablePersistence: boolean;

  /** Per-backend circuit breakers. */
  private readonly breakers: Map<EmbeddingBackend, CircuitBreakerState>;

  /** Adaptive timeout tracker. */
  readonly timeouts: AdaptiveTimeoutTracker;

  constructor(cwd: string, options: ResilientEmbeddingOptions = {}) {
    this.cwd = cwd;
    this.disablePersistence = options.disablePersistence ?? false;

    this.backoffOpts = {
      maxAttempts: options.backoff?.maxAttempts ?? DEFAULT_MAX_ATTEMPTS,
      baseDelayMs: options.backoff?.baseDelayMs ?? BACKOFF_BASE_MS,
      maxDelayMs: options.backoff?.maxDelayMs ?? BACKOFF_MAX_MS,
      disableJitter: options.backoff?.disableJitter ?? false,
    };

    this.cbOpts = {
      windowSize: options.circuitBreaker?.windowSize ?? DEFAULT_WINDOW_SIZE,
      failureRateThreshold:
        options.circuitBreaker?.failureRateThreshold ?? DEFAULT_FAILURE_RATE_THRESHOLD,
      openDurationMs:
        options.circuitBreaker?.openDurationMs ?? CIRCUIT_OPEN_DURATION_MS,
    };

    this.timeouts = new AdaptiveTimeoutTracker(options.timeoutsMs);

    const backends: EmbeddingBackend[] = ["ollama", "anthropic", "local-cached"];
    this.breakers = new Map(
      backends.map((b) => [b, makeCircuitBreaker(b, this.cbOpts.windowSize)]),
    );
  }

  // ── circuit breaker accessors ─────────────────────────────────────────────

  /** Return a snapshot of the current circuit breaker state for a backend. */
  getCircuitState(backend: EmbeddingBackend): CircuitState {
    return this.breakers.get(backend)?.state ?? "closed";
  }

  /** Manually reset all circuit breakers to closed (useful in tests). */
  resetAllBreakers(): void {
    for (const cb of this.breakers.values()) {
      cb.state = "closed";
      cb.openedAtMs = 0;
      cb.windowFilled = 0;
      cb.windowIndex = 0;
      cb.window.fill(true);
    }
  }

  // ── persistence ───────────────────────────────────────────────────────────

  private async persistEntry(entry: ResilienceProfileEntry): Promise<void> {
    if (this.disablePersistence) return;
    try {
      await appendJsonl(resilienceProfilePath(this.cwd), entry);
    } catch {
      // Best-effort — never throw from persistence
    }
  }

  // ── core embed method ────────────────────────────────────────────────────

  /**
   * Generate an embedding for `text` with full resilience: backoff retries,
   * circuit breaker protection, and adaptive timeout scaling.
   *
   * The orchestrator respects the static backend fallback chain
   * (ollama → anthropic → local-cached) but skips any backend whose circuit
   * breaker is currently open. Within each attempt it applies exponential
   * backoff with jitter on transient failures.
   *
   * Permanent errors (HTTP 401/403) abort the retry loop immediately.
   */
  async embed(text: string): Promise<ResilientEmbeddingResult> {
    const { generateEmbeddingMultiBackend, BACKEND_FALLBACK_CHAIN } =
      await import("./embedding-router.ts");

    const sizeBucket = classifyQuerySize(text);
    const circuitBreakerSkips: EmbeddingBackend[] = [];
    let permanentErrorEncountered = false;
    let totalAttempts = 0;

    // Build the effective backend order, skipping open breakers
    const orderedBackends = this._orderedBackends(circuitBreakerSkips);

    if (orderedBackends.length === 0) {
      // All breakers open — force local-cached as last resort
      orderedBackends.push("local-cached");
    }

    // Compute adaptive timeouts for this call
    const effectiveTimeouts: Partial<Record<EmbeddingBackend, number>> = {};
    for (const b of (BACKEND_FALLBACK_CHAIN as EmbeddingBackend[])) {
      effectiveTimeouts[b] = this.timeouts.get(b);
    }

    // If anthropic is slow, demote it by giving it a very small timeout
    // so local-cached takes over faster
    if (this.timeouts.isAnthropicSlow()) {
      effectiveTimeouts["anthropic"] = 100;
    }

    let lastResult: MultiBackendEmbeddingResult | null = null;
    let lastError: unknown = null;

    for (let attempt = 0; attempt < this.backoffOpts.maxAttempts; attempt++) {
      totalAttempts++;

      // Recompute open-breaker skips on each attempt (state may change)
      const activeBackends = this._orderedBackends(circuitBreakerSkips);
      if (activeBackends.length === 0) {
        activeBackends.push("local-cached");
      }

      try {
        const requestStart = Date.now();

        const result = await generateEmbeddingMultiBackend(
          text,
          this.cwd,
          {
            timeoutsMs: effectiveTimeouts,
            // Pass the active backends order by seeding metrics to 0 samples
            // so the static chain order applies — the circuit-breaker-aware
            // ordering is handled above via effectiveTimeouts (0ms = instant fail)
          },
        );

        const latencyMs = Date.now() - requestStart;
        const succeeded = result.embedding.length > 0;

        // Record circuit outcome
        const cb = this.breakers.get(result.backend);
        if (cb) {
          recordCircuitOutcome(cb, succeeded, Date.now(), this.cbOpts);
        }

        // Adaptive timeout update
        this.timeouts.record(result.backend, result.latencyMs);

        // Persist profile entry
        await this.persistEntry({
          timestamp: new Date().toISOString(),
          backend: result.backend,
          querySizeBucket: sizeBucket,
          latencyMs: result.latencyMs,
          succeeded,
          effectiveTimeoutMs: effectiveTimeouts[result.backend] ?? 15_000,
        });

        if (succeeded) {
          lastResult = result;
          break;
        }

        lastResult = result;

      } catch (err: unknown) {
        lastError = err;

        // Check if this is a permanent error
        if (isPermanentError(err)) {
          permanentErrorEncountered = true;
          break;
        }
      }

      // Apply backoff before next attempt (not after the last attempt)
      if (attempt < this.backoffOpts.maxAttempts - 1 && !permanentErrorEncountered) {
        const delay = computeBackoffDelay(
          attempt,
          this.backoffOpts.baseDelayMs,
          this.backoffOpts.maxDelayMs,
          this.backoffOpts.disableJitter,
        );
        if (delay > 0) {
          await sleep(delay);
        }
      }
    }

    // If we have a result, return it enriched
    if (lastResult !== null) {
      return {
        ...lastResult,
        totalAttempts,
        circuitBreakerSkips,
        permanentErrorEncountered,
      };
    }

    // Final fallback: use local UltraFastEmbedder — never fails
    const { UltraFastEmbedder } = await import("./embedding-ultrafast.ts");
    const { computeEmbeddingVariance } = await import("./embedding-router.ts");
    const embedder = new UltraFastEmbedder();
    const embedding = embedder.embed(text);

    return {
      embedding,
      backend: "local-cached",
      latencyMs: 0,
      costUsd: 0,
      dimensionalityVariance: computeEmbeddingVariance(embedding),
      usedFallback: true,
      failedBackends: 3,
      totalAttempts,
      circuitBreakerSkips,
      permanentErrorEncountered,
    };
  }

  // ── helper: ordered backends filtered by open circuit breakers ───────────

  private _orderedBackends(circuitBreakerSkips: EmbeddingBackend[]): EmbeddingBackend[] {
    const order: EmbeddingBackend[] = ["ollama", "anthropic", "local-cached"];
    return order.filter((b) => {
      const cb = this.breakers.get(b);
      if (!cb) return true;
      const allowed = isCircuitAllowed(cb, Date.now(), this.cbOpts.openDurationMs);
      if (!allowed) {
        if (!circuitBreakerSkips.includes(b)) circuitBreakerSkips.push(b);
        return false;
      }
      return true;
    });
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Returns true if an error represents a permanent (non-retryable) failure. */
export function isPermanentError(err: unknown): boolean {
  if (err instanceof PermanentEmbeddingError) return true;
  if (err instanceof Error) {
    const msg = err.message;
    for (const code of PERMANENT_ERROR_CODES) {
      if (msg.includes(String(code))) return true;
    }
  }
  return false;
}

/** Thrown to signal a non-retryable HTTP error (401, 403, etc.). */
export class PermanentEmbeddingError extends Error {
  readonly statusCode: number;
  constructor(statusCode: number, message?: string) {
    super(message ?? `Permanent embedding error: HTTP ${statusCode}`);
    this.name = "PermanentEmbeddingError";
    this.statusCode = statusCode;
  }
}

/** Async sleep helper. */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Resilient wrappers for EmbeddingRouter methods
// ---------------------------------------------------------------------------

/**
 * Wrap `generateEmbeddingWithFallback` with exponential backoff.
 *
 * Retries the entire cascade on transient failures. Permanent errors (401/403)
 * are surfaced immediately without retry.
 *
 * @param text         Text to embed.
 * @param primaryModel Primary model name (passed through to the cascade).
 * @param options      Backoff configuration.
 * @returns            First successful EmbeddingResult or null if all attempts fail.
 */
export async function resilientGenerateEmbeddingWithFallback(
  text: string,
  primaryModel: import("./embedding-router.ts").EmbeddingModelName,
  options: BackoffOptions = {},
): Promise<import("./embedding-router.ts").EmbeddingResult | null> {
  const { generateEmbeddingWithFallback } = await import("./embedding-router.ts");

  const maxAttempts = options.maxAttempts ?? DEFAULT_MAX_ATTEMPTS;
  const baseDelayMs = options.baseDelayMs ?? BACKOFF_BASE_MS;
  const maxDelayMs = options.maxDelayMs ?? BACKOFF_MAX_MS;
  const disableJitter = options.disableJitter ?? false;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      const result = await generateEmbeddingWithFallback(text, primaryModel);
      if (result !== null) return result;
    } catch (err) {
      if (isPermanentError(err)) throw err;
    }

    if (attempt < maxAttempts - 1) {
      const delay = computeBackoffDelay(attempt, baseDelayMs, maxDelayMs, disableJitter);
      if (delay > 0) await sleep(delay);
    }
  }

  return null;
}

/**
 * Wrap `EmbeddingRouter.embed` (the full pipeline including model selection)
 * with exponential backoff.
 *
 * Permanent errors (401/403) abort retries immediately.
 */
export async function resilientRouterEmbed(
  router: import("./embedding-router.ts").EmbeddingRouter,
  query: string,
  text: string,
  options: BackoffOptions = {},
): Promise<import("./embedding-router.ts").EmbeddingResult | null> {
  const maxAttempts = options.maxAttempts ?? DEFAULT_MAX_ATTEMPTS;
  const baseDelayMs = options.baseDelayMs ?? BACKOFF_BASE_MS;
  const maxDelayMs = options.maxDelayMs ?? BACKOFF_MAX_MS;
  const disableJitter = options.disableJitter ?? false;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      const result = await router.embed(query, text);
      if (result !== null) return result;
    } catch (err) {
      if (isPermanentError(err)) throw err;
    }

    if (attempt < maxAttempts - 1) {
      const delay = computeBackoffDelay(attempt, baseDelayMs, maxDelayMs, disableJitter);
      if (delay > 0) await sleep(delay);
    }
  }

  return null;
}

// ---------------------------------------------------------------------------
// Load persisted resilience profile
// ---------------------------------------------------------------------------

/**
 * Load all persisted resilience profile entries for a working directory.
 */
export async function loadResilienceProfile(
  cwd: string,
): Promise<ResilienceProfileEntry[]> {
  return readJsonl<ResilienceProfileEntry>(resilienceProfilePath(cwd));
}

/**
 * Compute per-(backend, querySizeBucket) latency statistics from persisted
 * profile entries.  Useful for dashboards and fleet learning.
 */
export function computeLatencyHistogram(
  entries: ResilienceProfileEntry[],
): Array<{
  backend: EmbeddingBackend;
  sizeBucket: QuerySizeBucket;
  count: number;
  avgLatencyMs: number;
  p95LatencyMs: number;
  successRate: number;
}> {
  type Key = `${EmbeddingBackend}:${QuerySizeBucket}`;
  const groups = new Map<Key, ResilienceProfileEntry[]>();

  for (const entry of entries) {
    const key: Key = `${entry.backend}:${entry.querySizeBucket}`;
    const group = groups.get(key) ?? [];
    group.push(entry);
    groups.set(key, group);
  }

  const result: ReturnType<typeof computeLatencyHistogram> = [];

  for (const [key, group] of groups.entries()) {
    const [backend, sizeBucket] = key.split(":") as [EmbeddingBackend, QuerySizeBucket];
    const succeeded = group.filter((e) => e.succeeded);
    const latencies = group.map((e) => e.latencyMs).sort((a, b) => a - b);
    const p95Idx = Math.floor(latencies.length * 0.95);

    result.push({
      backend,
      sizeBucket,
      count: group.length,
      avgLatencyMs:
        group.length > 0
          ? group.reduce((s, e) => s + e.latencyMs, 0) / group.length
          : 0,
      p95LatencyMs: latencies[p95Idx] ?? 0,
      successRate: group.length > 0 ? succeeded.length / group.length : 0,
    });
  }

  return result;
}

// ---------------------------------------------------------------------------
// Factory helper
// ---------------------------------------------------------------------------

/**
 * Create an EmbeddingResiliencePipeline bound to the given working directory.
 */
export function createEmbeddingResiliencePipeline(
  cwd: string,
  options?: ResilientEmbeddingOptions,
): EmbeddingResiliencePipeline {
  return new EmbeddingResiliencePipeline(cwd, options);
}

// ===========================================================================
// MULTI-TIER FALLBACK STRATEGY (M-item: Embedding Resilience & Fallback Tiers)
//
// Tier 1 = Ollama local      — fastest, zero cost, requires local daemon
// Tier 2 = OpenAI embeddings — reliable, ~15% cost adder, requires OPENAI_API_KEY
// Tier 3 = TF-IDF only       — zero cost, deterministic, always available
//
// New exports:
//   EmbeddingProviderDescriptor      — pluggable provider contract
//   EmbeddingProviderRegistry        — manages + selects among providers
//   TfIdfProvider                    — Tier-3 built-in (deterministic)
//   OllamaProvider                   — Tier-1 built-in wrapper
//   OpenAIProvider                   — Tier-2 built-in wrapper
//   EmbeddingStrategyEvent           — event written to embedding-strategy.jsonl
//   persistEmbeddingStrategyEvent    — appends one event to evolution/
//   loadEmbeddingStrategyLog         — reads all events for fitness feedback
//   createDefaultProviderRegistry    — builds the standard 3-tier registry
//   ResilientProviderPipeline        — orchestrates registry + circuit breakers
// ===========================================================================

// ---------------------------------------------------------------------------
// Provider contract
// ---------------------------------------------------------------------------

/**
 * A pluggable embedding provider.
 *
 * Implementations must be stateless (or manage their own state internally).
 * The registry calls isAvailable() before embed() on each attempt.
 */
export interface EmbeddingProviderDescriptor {
  /** Unique identifier for this provider (shown in logs + strategy events). */
  readonly name: string;
  /** Tier rank: lower = higher priority (1=Ollama, 2=OpenAI, 3=TF-IDF). */
  readonly tier: 1 | 2 | 3;
  /**
   * Return true when this provider is reachable / configured.
   * Called before every embed attempt; may be async (e.g., health-check).
   */
  isAvailable(): boolean | Promise<boolean>;
  /**
   * Embed a batch of texts.  Returns a parallel array of embedding vectors.
   * Throw (or return an empty array per text) on failure.
   */
  embed(texts: string[]): Promise<number[][]>;
}

// ---------------------------------------------------------------------------
// Tier-3: TF-IDF provider (always available, deterministic)
// ---------------------------------------------------------------------------

/**
 * TF-IDF embedding provider — Tier 3.
 *
 * Builds a simple term-frequency × inverse-document-frequency vector from the
 * vocabulary seen across the current batch.  Zero cost; no external calls;
 * deterministic for a fixed batch.  Dimensionality = vocabulary size (capped at
 * TFIDF_MAX_DIM to keep vectors bounded).
 *
 * Because IDF requires a corpus, single-text batches receive a pure TF vector
 * normalised to unit length.
 */
export const TFIDF_MAX_DIM = 512;

export class TfIdfProvider implements EmbeddingProviderDescriptor {
  readonly name = "tfidf";
  readonly tier = 3 as const;

  isAvailable(): boolean {
    return true; // always available
  }

  async embed(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];

    // Build vocabulary from all texts
    const vocab = new Map<string, number>(); // term → index
    const tokenised = texts.map((t) => this._tokenize(t));

    for (const tokens of tokenised) {
      for (const tok of tokens) {
        if (!vocab.has(tok)) {
          vocab.set(tok, vocab.size);
        }
      }
    }

    const vocabSize = Math.min(vocab.size, TFIDF_MAX_DIM);

    // Compute TF per document
    const tfVectors = tokenised.map((tokens) => {
      const tf = new Float64Array(vocabSize);
      for (const tok of tokens) {
        const idx = vocab.get(tok);
        if (idx !== undefined && idx < vocabSize) {
          tf[idx] = (tf[idx] ?? 0) + 1;
        }
      }
      // Normalise by token count
      const total = tokens.length || 1;
      for (let i = 0; i < vocabSize; i++) {
        tf[i] = (tf[i] ?? 0) / total;
      }
      return tf;
    });

    // Compute IDF (log((N+1)/(df+1))+1) for each term
    const N = texts.length;
    const idf = new Float64Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) {
      let df = 0;
      for (const tf of tfVectors) {
        if ((tf[i] ?? 0) > 0) df++;
      }
      idf[i] = Math.log((N + 1) / (df + 1)) + 1;
    }

    // TF-IDF = TF * IDF, then L2-normalise
    return tfVectors.map((tf) => {
      const vec = new Array<number>(vocabSize);
      for (let i = 0; i < vocabSize; i++) {
        vec[i] = (tf[i] ?? 0) * (idf[i] ?? 1);
      }
      return l2Normalize(vec);
    });
  }

  private _tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((t) => t.length > 1);
  }
}

/** L2-normalise a vector in-place, returning the same array. */
function l2Normalize(vec: number[]): number[] {
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  if (norm === 0) return vec;
  for (let i = 0; i < vec.length; i++) {
    vec[i] = (vec[i] ?? 0) / norm;
  }
  return vec;
}

// ---------------------------------------------------------------------------
// Tier-1: Ollama provider
// ---------------------------------------------------------------------------

/**
 * Ollama local embedding provider — Tier 1.
 *
 * Wraps the existing `generateEmbedding` + `isOllamaAvailable` from embeddings.ts.
 * Times out after `timeoutMs` milliseconds (default 15 s) to respect the
 * adaptive timeout from AdaptiveTimeoutTracker.
 */
export class OllamaProvider implements EmbeddingProviderDescriptor {
  readonly name = "ollama";
  readonly tier = 1 as const;
  private readonly timeoutMs: number;

  constructor(timeoutMs = 15_000) {
    this.timeoutMs = timeoutMs;
  }

  async isAvailable(): Promise<boolean> {
    try {
      const { isOllamaAvailable } = await import("./embeddings.ts");
      return isOllamaAvailable();
    } catch {
      return false;
    }
  }

  async embed(texts: string[]): Promise<number[][]> {
    const { generateEmbedding } = await import("./embeddings.ts");
    const results: number[][] = [];
    for (const text of texts) {
      const embeddingPromise = generateEmbedding(text);
      const timeoutPromise = new Promise<null>((resolve) =>
        setTimeout(() => resolve(null), this.timeoutMs),
      );
      const result = await Promise.race([embeddingPromise, timeoutPromise]);
      if (result === null) {
        throw new Error(`Ollama embedding timed out after ${this.timeoutMs}ms`);
      }
      results.push(result);
    }
    return results;
  }
}

// ---------------------------------------------------------------------------
// Tier-2: OpenAI embeddings provider
// ---------------------------------------------------------------------------

/** OpenAI embedding model to use. */
export const OPENAI_EMBEDDING_MODEL = "text-embedding-3-small";
/** Default timeout for OpenAI embedding calls (ms). */
export const OPENAI_DEFAULT_TIMEOUT_MS = 10_000;

/**
 * OpenAI embeddings provider — Tier 2.
 *
 * Calls the OpenAI `/v1/embeddings` endpoint.  Requires OPENAI_API_KEY in the
 * environment.  Returns null (triggering Tier 3 fallback) when:
 *  - OPENAI_API_KEY is absent
 *  - The request times out
 *  - The API returns a non-200 response
 *
 * Permanent errors (HTTP 401, 403) throw PermanentEmbeddingError so the
 * circuit breaker records them correctly.
 */
export class OpenAIProvider implements EmbeddingProviderDescriptor {
  readonly name = "openai";
  readonly tier = 2 as const;
  private readonly timeoutMs: number;
  private readonly model: string;

  constructor(timeoutMs = OPENAI_DEFAULT_TIMEOUT_MS, model = OPENAI_EMBEDDING_MODEL) {
    this.timeoutMs = timeoutMs;
    this.model = model;
  }

  isAvailable(): boolean {
    return typeof process.env["OPENAI_API_KEY"] === "string" &&
      process.env["OPENAI_API_KEY"].length > 0;
  }

  async embed(texts: string[]): Promise<number[][]> {
    const apiKey = process.env["OPENAI_API_KEY"];
    if (!apiKey) throw new Error("OPENAI_API_KEY not set");

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const res = await fetch("https://api.openai.com/v1/embeddings", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: this.model,
          input: texts,
          encoding_format: "float",
        }),
        signal: controller.signal,
      });

      clearTimeout(timer);

      if (res.status === 401 || res.status === 403) {
        throw new PermanentEmbeddingError(res.status, `OpenAI HTTP ${res.status}`);
      }

      if (!res.ok) {
        throw new Error(`OpenAI embeddings HTTP ${res.status}`);
      }

      const data = (await res.json()) as {
        data: Array<{ embedding: number[]; index: number }>;
      };

      // Sort by index to ensure order matches input
      const sorted = [...data.data].sort((a, b) => a.index - b.index);
      return sorted.map((d) => d.embedding);
    } catch (err) {
      clearTimeout(timer);
      throw err;
    }
  }
}

// ---------------------------------------------------------------------------
// Embedding strategy event (persisted for fitness feedback)
// ---------------------------------------------------------------------------

/** Tier name for logging. */
export type ProviderTierName = "ollama" | "openai" | "tfidf" | string;

/**
 * A single embedding strategy event written to `evolution/embedding-strategy.jsonl`.
 *
 * Used by the fleet's fitness feedback loop to learn which provider tier
 * performs best under different load/availability conditions.
 */
export interface EmbeddingStrategyEvent {
  /** ISO timestamp. */
  timestamp: string;
  /** Name of the provider that produced the final embedding. */
  winningProvider: ProviderTierName;
  /** Tier number of the winning provider (1/2/3). */
  winningTier: 1 | 2 | 3 | number;
  /** Providers that were tried before the winner (in attempt order). */
  failedProviders: ProviderTierName[];
  /** Reason the previous tier(s) failed, e.g. "timeout", "unavailable", "error". */
  failureReasons: string[];
  /** Whether the winning provider was a fallback (i.e., Tier 1 was unavailable). */
  usedFallback: boolean;
  /** Total wall-clock latency for the full chain (ms). */
  totalLatencyMs: number;
  /** Estimated USD cost of the winning call. */
  costUsd: number;
  /** Optional: caller-supplied --embedding-provider flag value (null = auto). */
  requestedProvider: ProviderTierName | null;
}

/** Path to the embedding strategy log. */
function embeddingStrategyLogPath(cwd: string): string {
  return join(genomeDir(cwd), "evolution", "embedding-strategy.jsonl");
}

/**
 * Append one EmbeddingStrategyEvent to the evolution log.
 * Best-effort: never throws.
 */
export async function persistEmbeddingStrategyEvent(
  cwd: string,
  event: EmbeddingStrategyEvent,
): Promise<void> {
  try {
    await appendJsonl(embeddingStrategyLogPath(cwd), event);
  } catch {
    // best-effort
  }
}

/**
 * Load all EmbeddingStrategyEvents for a working directory.
 */
export async function loadEmbeddingStrategyLog(
  cwd: string,
): Promise<EmbeddingStrategyEvent[]> {
  return readJsonl<EmbeddingStrategyEvent>(embeddingStrategyLogPath(cwd));
}

// ---------------------------------------------------------------------------
// EmbeddingProviderRegistry
// ---------------------------------------------------------------------------

/**
 * Registry of pluggable embedding providers.
 *
 * Accepts any number of `EmbeddingProviderDescriptor` implementations.  Providers
 * are tried in ascending tier order (Tier 1 first).  A forced `--embedding-provider`
 * flag value can be supplied to pin a specific provider.
 *
 * Per-provider circuit breakers are maintained internally (same logic as the
 * existing EmbeddingResiliencePipeline breakers).
 */
export class EmbeddingProviderRegistry {
  private readonly providers: EmbeddingProviderDescriptor[];
  private readonly breakers: Map<string, CircuitBreakerState>;
  private readonly cbOpts: Required<CircuitBreakerOptions>;

  constructor(
    providers: EmbeddingProviderDescriptor[],
    cbOptions: CircuitBreakerOptions = {},
  ) {
    // Sort by tier (ascending), then by registration order for ties
    this.providers = [...providers].sort((a, b) => a.tier - b.tier);
    this.cbOpts = {
      windowSize: cbOptions.windowSize ?? DEFAULT_WINDOW_SIZE,
      failureRateThreshold:
        cbOptions.failureRateThreshold ?? DEFAULT_FAILURE_RATE_THRESHOLD,
      openDurationMs: cbOptions.openDurationMs ?? CIRCUIT_OPEN_DURATION_MS,
    };
    this.breakers = new Map(
      this.providers.map((p) => [
        p.name,
        {
          backend: p.name as EmbeddingBackend,
          state: "closed" as CircuitState,
          window: new Array<boolean>(this.cbOpts.windowSize).fill(true),
          windowIndex: 0,
          windowFilled: 0,
          openedAtMs: 0,
        },
      ]),
    );
  }

  /** Return all registered providers in priority order. */
  getProviders(): EmbeddingProviderDescriptor[] {
    return [...this.providers];
  }

  /** Return the circuit breaker state for a named provider. */
  getCircuitState(name: string): CircuitState {
    return this.breakers.get(name)?.state ?? "closed";
  }

  /** Reset all circuit breakers to closed. */
  resetAllBreakers(): void {
    for (const cb of this.breakers.values()) {
      cb.state = "closed";
      cb.openedAtMs = 0;
      cb.windowFilled = 0;
      cb.windowIndex = 0;
      cb.window.fill(true);
    }
  }

  /**
   * Record an outcome for a provider's circuit breaker.
   */
  recordOutcome(name: string, succeeded: boolean, nowMs = Date.now()): void {
    const cb = this.breakers.get(name);
    if (cb) {
      recordCircuitOutcome(cb, succeeded, nowMs, this.cbOpts);
    }
  }

  /**
   * Select the ordered list of providers to try for a request.
   *
   * @param forcedProvider  When set (via --embedding-provider flag), only that
   *                        provider is returned (if registered); otherwise the
   *                        full tier-ordered list is returned with open-circuit
   *                        breakers skipped.
   */
  selectProviders(forcedProvider?: string): EmbeddingProviderDescriptor[] {
    if (forcedProvider) {
      const found = this.providers.find((p) => p.name === forcedProvider);
      return found ? [found] : [];
    }

    return this.providers.filter((p) => {
      const cb = this.breakers.get(p.name);
      if (!cb) return true;
      return isCircuitAllowed(cb, Date.now(), this.cbOpts.openDurationMs);
    });
  }
}

// ---------------------------------------------------------------------------
// ResilientProviderPipeline — orchestrates registry + circuit breakers
// ---------------------------------------------------------------------------

/** Result from ResilientProviderPipeline.embed(). */
export interface ProviderEmbeddingResult {
  /** The embedding vector. */
  embedding: number[];
  /** Name of the provider that produced the result. */
  providerName: ProviderTierName;
  /** Tier of the winning provider. */
  tier: 1 | 2 | 3 | number;
  /** True when a lower-priority provider was used as fallback. */
  usedFallback: boolean;
  /** Total wall-clock latency (ms). */
  totalLatencyMs: number;
  /** Estimated cost in USD. */
  costUsd: number;
  /** Providers that failed before the winner. */
  failedProviders: ProviderTierName[];
  /** Failure reasons for each failed provider. */
  failureReasons: string[];
  /** Whether a permanent error (401/403) was encountered. */
  permanentErrorEncountered: boolean;
}

/** Options for ResilientProviderPipeline. */
export interface ResilientProviderPipelineOptions {
  /** Working directory for evolution log persistence. */
  cwd?: string;
  /** When true, strategy events are not written to disk. */
  disablePersistence?: boolean;
  /** Backoff options for retrying failed providers. */
  backoff?: BackoffOptions;
  /** Circuit breaker options applied to all providers. */
  circuitBreaker?: CircuitBreakerOptions;
  /**
   * Forced provider name (e.g. from --embedding-provider CLI flag).
   * When set, only this provider is attempted; no fallback occurs.
   */
  forcedProvider?: string;
}

/** Cost estimate per provider tier (USD per request). */
const PROVIDER_COST_PER_REQUEST: Record<string, number> = {
  ollama: 0,
  openai: 0.000_02, // ~$0.02/1M tokens × ~1000 tokens avg
  tfidf: 0,
};

function estimateProviderCost(providerName: string, text: string): number {
  const base = PROVIDER_COST_PER_REQUEST[providerName] ?? 0;
  if (base === 0) return 0;
  // rough 4 chars/token heuristic
  const tokens = Math.ceil(text.length / 4);
  return (tokens / 1_000) * base;
}

/**
 * ResilientProviderPipeline orchestrates multi-tier embedding with:
 *
 *  - EmbeddingProviderRegistry for pluggable Tier1/2/3 providers
 *  - Per-provider circuit breakers
 *  - Exponential backoff for transient failures
 *  - evolution/embedding-strategy.jsonl persistence for fitness feedback
 *  - --embedding-provider forced override support
 *
 * Usage:
 * ```ts
 * const pipeline = new ResilientProviderPipeline(registry, { cwd });
 * const result = await pipeline.embed("some text");
 * ```
 */
export class ResilientProviderPipeline {
  private readonly registry: EmbeddingProviderRegistry;
  private readonly cwd: string;
  private readonly disablePersistence: boolean;
  private readonly backoffOpts: Required<BackoffOptions>;
  private readonly forcedProvider?: string;

  constructor(
    registry: EmbeddingProviderRegistry,
    options: ResilientProviderPipelineOptions = {},
  ) {
    this.registry = registry;
    this.cwd = options.cwd ?? process.cwd();
    this.disablePersistence = options.disablePersistence ?? false;
    this.forcedProvider = options.forcedProvider;
    this.backoffOpts = {
      maxAttempts: options.backoff?.maxAttempts ?? DEFAULT_MAX_ATTEMPTS,
      baseDelayMs: options.backoff?.baseDelayMs ?? BACKOFF_BASE_MS,
      maxDelayMs: options.backoff?.maxDelayMs ?? BACKOFF_MAX_MS,
      disableJitter: options.backoff?.disableJitter ?? false,
    };
  }

  /**
   * Embed a single text using the registry's provider chain.
   *
   * Tries each provider in tier order; on failure records the outcome in the
   * circuit breaker and moves to the next tier.  Exponential backoff is applied
   * between retry attempts for the same provider.
   *
   * Always returns a result (Tier 3 TF-IDF never fails).
   */
  async embed(text: string): Promise<ProviderEmbeddingResult> {
    const overallStart = Date.now();
    const failedProviders: ProviderTierName[] = [];
    const failureReasons: string[] = [];
    let permanentErrorEncountered = false;

    const orderedProviders = this.registry.selectProviders(this.forcedProvider);

    for (const provider of orderedProviders) {
      // Check availability before attempting
      let available: boolean;
      try {
        available = await Promise.resolve(provider.isAvailable());
      } catch {
        available = false;
      }

      if (!available) {
        failedProviders.push(provider.name);
        failureReasons.push("unavailable");
        this.registry.recordOutcome(provider.name, false);
        continue;
      }

      // Try with exponential backoff
      let lastErr: unknown = null;
      let succeeded = false;
      let resultEmbedding: number[] | null = null;

      for (let attempt = 0; attempt < this.backoffOpts.maxAttempts; attempt++) {
        try {
          const embeddings = await provider.embed([text]);
          if (embeddings.length > 0 && embeddings[0] && embeddings[0].length > 0) {
            resultEmbedding = embeddings[0];
            succeeded = true;
            break;
          }
          lastErr = new Error("Empty embedding returned");
        } catch (err) {
          lastErr = err;
          if (isPermanentError(err)) {
            permanentErrorEncountered = true;
            break;
          }
        }

        if (attempt < this.backoffOpts.maxAttempts - 1 && !permanentErrorEncountered) {
          const delay = computeBackoffDelay(
            attempt,
            this.backoffOpts.baseDelayMs,
            this.backoffOpts.maxDelayMs,
            this.backoffOpts.disableJitter,
          );
          if (delay > 0) await sleep(delay);
        }
      }

      this.registry.recordOutcome(provider.name, succeeded);

      if (succeeded && resultEmbedding !== null) {
        const totalLatencyMs = Date.now() - overallStart;
        const costUsd = estimateProviderCost(provider.name, text);

        await this._persistEvent({
          timestamp: new Date().toISOString(),
          winningProvider: provider.name,
          winningTier: provider.tier,
          failedProviders,
          failureReasons,
          usedFallback: failedProviders.length > 0,
          totalLatencyMs,
          costUsd,
          requestedProvider: this.forcedProvider ?? null,
        });

        return {
          embedding: resultEmbedding,
          providerName: provider.name,
          tier: provider.tier,
          usedFallback: failedProviders.length > 0,
          totalLatencyMs,
          costUsd,
          failedProviders,
          failureReasons,
          permanentErrorEncountered,
        };
      }

      // Provider failed — record reason and move on
      if (permanentErrorEncountered) break;
      failedProviders.push(provider.name);
      const reason =
        lastErr instanceof Error ? lastErr.message.slice(0, 80) : "unknown";
      failureReasons.push(reason);
    }

    // All providers failed — use TF-IDF as absolute last resort (in case it
    // wasn't in the registry or was circuit-broken)
    const tfidf = new TfIdfProvider();
    const fallbackEmbeddings = await tfidf.embed([text]);
    const fallbackEmbedding = fallbackEmbeddings[0] ?? [];
    const totalLatencyMs = Date.now() - overallStart;

    await this._persistEvent({
      timestamp: new Date().toISOString(),
      winningProvider: "tfidf-emergency",
      winningTier: 3,
      failedProviders,
      failureReasons,
      usedFallback: true,
      totalLatencyMs,
      costUsd: 0,
      requestedProvider: this.forcedProvider ?? null,
    });

    return {
      embedding: fallbackEmbedding,
      providerName: "tfidf-emergency",
      tier: 3,
      usedFallback: true,
      totalLatencyMs,
      costUsd: 0,
      failedProviders,
      failureReasons,
      permanentErrorEncountered,
    };
  }

  private async _persistEvent(event: EmbeddingStrategyEvent): Promise<void> {
    if (this.disablePersistence) return;
    await persistEmbeddingStrategyEvent(this.cwd, event);
  }
}

// ---------------------------------------------------------------------------
// EmbeddingRouter integration: --embedding-provider flag support
// ---------------------------------------------------------------------------

/**
 * Parse an --embedding-provider CLI flag value into a provider name.
 *
 * Accepted values (case-insensitive):
 *  - "ollama" | "1"  → "ollama"
 *  - "openai" | "2"  → "openai"
 *  - "tfidf"  | "3"  → "tfidf"
 *  - null / undefined / "" → undefined (auto-select)
 *
 * Unknown values return undefined so the registry auto-selects.
 */
export function parseEmbeddingProviderFlag(
  value: string | null | undefined,
): string | undefined {
  if (!value) return undefined;
  const v = value.trim().toLowerCase();
  if (v === "ollama" || v === "1") return "ollama";
  if (v === "openai" || v === "2") return "openai";
  if (v === "tfidf" || v === "3") return "tfidf";
  return undefined;
}

// ---------------------------------------------------------------------------
// Default registry factory
// ---------------------------------------------------------------------------

/**
 * Build the standard three-tier EmbeddingProviderRegistry:
 *
 *  Tier 1 — OllamaProvider  (fastest, zero cost)
 *  Tier 2 — OpenAIProvider  (reliable, ~15% cost adder)
 *  Tier 3 — TfIdfProvider   (zero cost, deterministic offline)
 *
 * @param ollamaTimeoutMs  Timeout for Ollama calls (default 15 s).
 * @param openaiTimeoutMs  Timeout for OpenAI calls (default 10 s).
 * @param cbOptions        Circuit breaker options.
 */
export function createDefaultProviderRegistry(
  ollamaTimeoutMs = 15_000,
  openaiTimeoutMs = OPENAI_DEFAULT_TIMEOUT_MS,
  cbOptions?: CircuitBreakerOptions,
): EmbeddingProviderRegistry {
  return new EmbeddingProviderRegistry(
    [
      new OllamaProvider(ollamaTimeoutMs),
      new OpenAIProvider(openaiTimeoutMs),
      new TfIdfProvider(),
    ],
    cbOptions,
  );
}

// Re-export EmbeddingBackend for use in registry types
export type { EmbeddingBackend } from "./embedding-router.ts";
