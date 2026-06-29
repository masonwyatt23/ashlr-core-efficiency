/**
 * RecalibrationEngine — per-provider/model live token-count calibration.
 *
 * Maintains per-bin correction factors (text, tool_result, tool_use, thinking,
 * image, mixed) and per-provider correction factors, updated via EMA on each
 * real API response. Persists calibration state to
 * ~/.ashlr/token-calibration.jsonl.
 *
 * Usage:
 *   const engine = RecalibrationEngine.instance;
 *   // After an API call:
 *   engine.record({ provider, messages, estimated, actual });
 *   // Before a call:
 *   const factor = engine.correctionFactor(provider, bin);
 */

import { homedir } from "os";
import { join } from "path";
import { existsSync, mkdirSync } from "fs";
import { appendFile } from "fs/promises";
import type { Message } from "../types/index.ts";
import { CodecRegistry, classifyMessage, type MessageBin } from "./provider-codec.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** EMA smoothing factor for per-bin correction updates. */
export const RECAL_EMA_ALPHA = 0.25;

/** Hard bounds on correction factors to prevent instability. */
export const RECAL_FACTOR_MIN = 0.70;
export const RECAL_FACTOR_MAX = 2.50;

/** Minimum number of samples before we trust the per-bin correction factor. */
export const RECAL_MIN_SAMPLES = 3;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface RecalibrationRecord {
  /** ISO timestamp. */
  timestamp: string;
  /** Full model/provider slug. */
  provider: string;
  /** Bin classification of the dominant message in the request. */
  bin: MessageBin;
  /** Token count we estimated before the call. */
  estimated: number;
  /** Billing-equivalent actual token count from the API response. */
  actual: number;
}

export interface BinStats {
  /** Number of samples collected for this bin. */
  sampleCount: number;
  /** Current EMA correction factor. */
  correctionFactor: number;
  /** Running mean absolute percentage error (%). */
  mape: number;
}

export interface ProviderCalibrationState {
  /** Provider/model slug. */
  provider: string;
  /** Per-bin stats. */
  bins: Record<MessageBin, BinStats>;
  /** Overall correction factor across all bins. */
  overallFactor: number;
  /** Total samples recorded for this provider. */
  totalSamples: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function defaultBinStats(): BinStats {
  return { sampleCount: 0, correctionFactor: 1.0, mape: 0 };
}

function allBins(): Record<MessageBin, BinStats> {
  return {
    text:        defaultBinStats(),
    tool_result: defaultBinStats(),
    tool_use:    defaultBinStats(),
    thinking:    defaultBinStats(),
    image:       defaultBinStats(),
    mixed:       defaultBinStats(),
  };
}

/** Resolve the canonical provider key for storage/lookup. */
function canonicalKey(provider: string): string {
  return provider.toLowerCase().trim();
}

function calibrationFilePath(): string {
  const ashlrDir = join(homedir(), ".ashlr");
  if (!existsSync(ashlrDir)) {
    mkdirSync(ashlrDir, { recursive: true });
  }
  return join(ashlrDir, "token-calibration.jsonl");
}

// ---------------------------------------------------------------------------
// RecalibrationEngine
// ---------------------------------------------------------------------------

/**
 * Singleton engine that tracks estimation accuracy per provider × bin,
 * applies EMA-smoothed correction factors, and persists records for offline
 * analysis.
 */
export class RecalibrationEngine {
  private static _instance: RecalibrationEngine | null = null;

  static get instance(): RecalibrationEngine {
    if (!RecalibrationEngine._instance) {
      RecalibrationEngine._instance = new RecalibrationEngine();
    }
    return RecalibrationEngine._instance;
  }

  /** Replace singleton (used in tests). */
  static resetInstance(): void {
    RecalibrationEngine._instance = null;
  }

  /** In-memory provider → calibration state. */
  private readonly _state = new Map<string, ProviderCalibrationState>();

  private constructor() {}

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Record a real (estimated, actual) feedback pair and update per-bin factors.
   *
   * @param args.provider   Full model/provider slug from the API response.
   * @param args.messages   The messages that were sent (for bin classification).
   * @param args.estimated  Token count we estimated before the call.
   * @param args.actual     Billing-equivalent actual token count.
   * @param args.bin        Optional override for bin classification.
   */
  record(args: {
    provider: string;
    messages: Message[];
    estimated: number;
    actual: number;
    bin?: MessageBin;
  }): void {
    const { provider, messages, estimated, actual } = args;
    if (estimated <= 0 || actual <= 0) return;

    // Classify request bin (use the last user message, or the most common type).
    const bin: MessageBin = args.bin ?? this._classifyRequest(messages);

    const key = canonicalKey(provider);
    const state = this._getOrCreate(key, provider);

    // Update per-bin stats.
    this._updateBin(state.bins[bin], estimated, actual);

    // Update overall factor via EMA across all bins.
    const ratio = actual / estimated;
    const alpha = RECAL_EMA_ALPHA;
    const updated = state.overallFactor + alpha * (ratio - state.overallFactor);
    state.overallFactor = Math.max(RECAL_FACTOR_MIN, Math.min(RECAL_FACTOR_MAX, updated));
    state.totalSamples += 1;

    // Update the codec's correction factor so countTokensAccurate benefits.
    const codec = CodecRegistry.instance.getCodec(provider);
    codec.calibrate(estimated, actual, provider);

    // Fire-and-forget persistence.
    this._persist({
      timestamp: new Date().toISOString(),
      provider,
      bin,
      estimated,
      actual,
    }).catch(() => { /* non-fatal */ });
  }

  /**
   * Get the correction factor to apply for a given provider + bin.
   *
   * Returns 1.0 when we don't yet have enough samples for the bin (falls back
   * gracefully to the heuristic path).
   */
  correctionFactor(provider: string, bin: MessageBin = "text"): number {
    const state = this._state.get(canonicalKey(provider));
    if (!state) return 1.0;

    const binStats = state.bins[bin];
    if (binStats.sampleCount < RECAL_MIN_SAMPLES) {
      // Not enough data for this bin: fall back to overall factor.
      if (state.totalSamples < RECAL_MIN_SAMPLES) return 1.0;
      return state.overallFactor;
    }
    return binStats.correctionFactor;
  }

  /**
   * Get the overall correction factor for a provider (ignores bin).
   * Returns 1.0 when no data is available.
   */
  overallFactor(provider: string): number {
    return this._state.get(canonicalKey(provider))?.overallFactor ?? 1.0;
  }

  /**
   * Get full calibration state for a provider, or undefined if unknown.
   */
  getState(provider: string): ProviderCalibrationState | undefined {
    return this._state.get(canonicalKey(provider));
  }

  /**
   * Return all provider keys with recorded state.
   */
  knownProviders(): string[] {
    return Array.from(this._state.keys());
  }

  /**
   * MAPE for a specific provider + bin. Returns 0 when no data.
   */
  mape(provider: string, bin?: MessageBin): number {
    const state = this._state.get(canonicalKey(provider));
    if (!state) return 0;
    if (!bin) {
      // Compute weighted average MAPE across all bins.
      let totalWeight = 0;
      let weightedMape = 0;
      for (const binKey of Object.keys(state.bins) as MessageBin[]) {
        const bs = state.bins[binKey];
        if (bs.sampleCount > 0) {
          weightedMape += bs.mape * bs.sampleCount;
          totalWeight += bs.sampleCount;
        }
      }
      return totalWeight > 0 ? weightedMape / totalWeight : 0;
    }
    return state.bins[bin].mape;
  }

  /** Reset all in-memory state (used in tests). */
  reset(): void {
    this._state.clear();
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  private _getOrCreate(key: string, provider: string): ProviderCalibrationState {
    let state = this._state.get(key);
    if (!state) {
      state = {
        provider,
        bins: allBins(),
        overallFactor: 1.0,
        totalSamples: 0,
      };
      this._state.set(key, state);
    }
    return state;
  }

  private _updateBin(bin: BinStats, estimated: number, actual: number): void {
    const error = Math.abs((actual - estimated) / estimated);
    const alpha = RECAL_EMA_ALPHA;

    // Update MAPE via EMA.
    if (bin.sampleCount === 0) {
      bin.mape = error * 100;
    } else {
      bin.mape = bin.mape + alpha * (error * 100 - bin.mape);
    }

    // Target correction factor: actual/estimated (raw ratio).
    const target = actual / estimated;
    if (bin.sampleCount === 0) {
      // First sample: blend halfway (conservative).
      bin.correctionFactor = 1.0 + alpha * (target - 1.0);
    } else {
      bin.correctionFactor = bin.correctionFactor + alpha * (target - bin.correctionFactor);
    }

    // Clamp.
    bin.correctionFactor = Math.max(
      RECAL_FACTOR_MIN,
      Math.min(RECAL_FACTOR_MAX, bin.correctionFactor),
    );
    bin.sampleCount += 1;
  }

  private _classifyRequest(messages: Message[]): MessageBin {
    if (messages.length === 0) return "text";
    // Use the last message as representative.
    const last = messages[messages.length - 1];
    if (!last) return "text";
    return classifyMessage(last);
  }

  private async _persist(record: RecalibrationRecord): Promise<void> {
    try {
      const line = JSON.stringify(record) + "\n";
      await appendFile(calibrationFilePath(), line, "utf8");
    } catch {
      // Non-fatal.
    }
  }
}
