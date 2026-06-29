/**
 * Cross-Provider Token Estimation Validator — Live Calibration Against Actual LLM Usage
 *
 * Records (estimated, actual) token pairs per model, persists them in a
 * sliding-window JSONL file at ~/.ashlr/token-calibration.jsonl, and
 * computes per-quartile MAPE to detect when a model's nominal multiplier
 * has drifted more than 5% from observed reality.
 *
 * Workflow:
 *   1. Call `appendCalibrationRecord()` after each LLM response (hooked via
 *      `recordActualTokenUsage` in index.ts).
 *   2. Call `validateAndRecalibrate(model)` periodically (e.g. every 20 calls)
 *      to compute drift and get a `RecalibratedEncodingConfig`.
 *   3. The token counter checks `getActiveCalibratedMultiplier(model)` to apply
 *      the latest calibrated multiplier instead of the static table value.
 */

import { homedir } from "os";
import { join } from "path";
import { existsSync, mkdirSync } from "fs";
import { readFile, writeFile, appendFile } from "fs/promises";

// ---------- public types ----------

export type EncodingName = "cl100k_base" | "o200k_base";

/**
 * A single record of one LLM call: the estimate we made before the call and
 * the actual token usage billed by the provider.
 */
export interface CalibrationRecord {
  /** ISO timestamp of the observation. */
  timestamp: string;
  /** Full or partial model name (e.g. "claude-3-5-sonnet-20241022"). */
  model: string;
  /** Token count estimated before the call. */
  estimated: number;
  /** Billing-equivalent actual token count (includes cache multipliers). */
  actual: number;
}

/**
 * Per-quartile MAPE breakdown so callers can see where estimation is worst.
 */
export interface MapeBreakdown {
  /** Overall MAPE across all records for this model. */
  overall: number;
  /** MAPE for the bottom-25% of actual token counts (short prompts). */
  q1: number;
  /** MAPE for the 25th–50th percentile of actual token counts. */
  q2: number;
  /** MAPE for the 50th–75th percentile of actual token counts. */
  q3: number;
  /** MAPE for the top-25% of actual token counts (long prompts). */
  q4: number;
}

/**
 * Result returned by `validateAndRecalibrate`. Contains diagnostic stats and,
 * when drift is detected, a new `RecalibratedEncodingConfig` to apply.
 */
export interface CalibrationResult {
  model: string;
  /** Number of observations used in this calibration run. */
  sampleSize: number;
  /** Per-quartile MAPE (mean absolute percentage error). */
  mape: MapeBreakdown;
  /**
   * Ratio of mean(actual) / mean(estimated). Values > 1 mean we under-estimate;
   * values < 1 mean we over-estimate.
   */
  biasFactor: number;
  /**
   * True when the overall MAPE exceeds the drift threshold (default 5%).
   * When true, `recalibrated` is populated with a suggested new config.
   */
  driftDetected: boolean;
  /** Suggested new encoding config when drift is detected. */
  recalibrated?: RecalibratedEncodingConfig;
  /** ISO timestamp of this calibration run. */
  calibratedAt: string;
}

/**
 * A new encoding config produced by the calibration run. The token counter
 * applies this on subsequent calls until the next calibration cycle.
 */
export interface RecalibratedEncodingConfig {
  model: string;
  encoding: EncodingName;
  /**
   * New multiplier, clamped to [0.80, 2.00] to prevent runaway corrections.
   * Computed as: nominal_multiplier × biasFactor, then smoothed with EMA.
   */
  multiplier: number;
  /** The prior nominal multiplier this replaces. */
  priorMultiplier: number;
  /** Overall MAPE that triggered this recalibration. */
  triggerMape: number;
}

// ---------- constants ----------

/** Sliding window size: keep the most recent N records per model. */
export const WINDOW_SIZE = 100;

/** MAPE threshold (%) above which we flag drift and emit a new config. */
export const DRIFT_THRESHOLD_PCT = 5;

/** EMA smoothing factor for multiplier updates (0 = no update, 1 = full jump). */
export const EMA_ALPHA = 0.3;

/** Hard bounds on the calibrated multiplier to prevent instability. */
export const MULTIPLIER_MIN = 0.80;
export const MULTIPLIER_MAX = 2.00;

// ---------- file path ----------

function calibrationFilePath(): string {
  const ashlrDir = join(homedir(), ".ashlr");
  if (!existsSync(ashlrDir)) {
    mkdirSync(ashlrDir, { recursive: true });
  }
  return join(ashlrDir, "token-calibration.jsonl");
}

// ---------- in-memory cache of active calibrated multipliers ----------

// model → calibrated multiplier (applied on top of tiktoken count)
const _calibratedMultipliers = new Map<string, number>();

/**
 * Return the currently active calibrated multiplier for `model`, or `null`
 * if no calibration has been applied yet (use the static table value instead).
 */
export function getActiveCalibratedMultiplier(model: string): number | null {
  const lower = model.toLowerCase();
  // Exact match first
  if (_calibratedMultipliers.has(lower)) return _calibratedMultipliers.get(lower)!;
  // Prefix match (the calibration key may be a prefix or vice-versa)
  for (const [key, mult] of _calibratedMultipliers) {
    if (lower.startsWith(key) || key.startsWith(lower)) return mult;
  }
  return null;
}

/** Exposed for tests: clear all in-memory calibrated multipliers. */
export function _resetCalibratedMultipliers(): void {
  _calibratedMultipliers.clear();
}

// ---------- I/O helpers ----------

/**
 * Append a single `CalibrationRecord` to the JSONL file.
 * Fire-and-forget safe: errors are swallowed so a disk issue never kills the
 * main execution path.
 */
export async function appendCalibrationRecord(
  record: CalibrationRecord,
): Promise<void> {
  try {
    const line = JSON.stringify(record) + "\n";
    await appendFile(calibrationFilePath(), line, "utf8");
  } catch {
    // Non-fatal — calibration is best-effort.
  }
}

/**
 * Read all records from the JSONL file. Returns an empty array on any error
 * (missing file, parse failure, etc.).
 */
export async function readAllCalibrationRecords(): Promise<CalibrationRecord[]> {
  try {
    const raw = await readFile(calibrationFilePath(), "utf8");
    const records: CalibrationRecord[] = [];
    for (const line of raw.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        records.push(JSON.parse(trimmed) as CalibrationRecord);
      } catch {
        // Skip malformed lines.
      }
    }
    return records;
  } catch {
    return [];
  }
}

/**
 * Read records for a specific model (prefix match), limited to the most
 * recent `WINDOW_SIZE` entries.
 */
export async function readModelRecords(model: string): Promise<CalibrationRecord[]> {
  const all = await readAllCalibrationRecords();
  const lower = model.toLowerCase();
  const modelRecords = all.filter((r) => {
    const rLower = r.model.toLowerCase();
    return rLower === lower || rLower.startsWith(lower) || lower.startsWith(rLower);
  });
  // Keep the most recent WINDOW_SIZE records.
  return modelRecords.slice(-WINDOW_SIZE);
}

/**
 * Trim the JSONL file by rewriting it with only the most recent `WINDOW_SIZE`
 * records per model. Called automatically after `appendCalibrationRecord` when
 * the total line count would exceed 10× WINDOW_SIZE.
 */
export async function trimCalibrationFile(): Promise<void> {
  try {
    const all = await readAllCalibrationRecords();
    if (all.length === 0) return;

    // Group by normalised model key
    const byModel = new Map<string, CalibrationRecord[]>();
    for (const r of all) {
      const key = r.model.toLowerCase();
      const bucket = byModel.get(key) ?? [];
      bucket.push(r);
      byModel.set(key, bucket);
    }

    const trimmed: CalibrationRecord[] = [];
    for (const records of byModel.values()) {
      trimmed.push(...records.slice(-WINDOW_SIZE));
    }

    // Re-sort by timestamp to keep the file roughly chronological.
    trimmed.sort((a, b) => a.timestamp.localeCompare(b.timestamp));

    const lines = trimmed.map((r) => JSON.stringify(r)).join("\n") + "\n";
    await writeFile(calibrationFilePath(), lines, "utf8");
  } catch {
    // Non-fatal.
  }
}

// ---------- MAPE computation ----------

function mapeForSlice(records: CalibrationRecord[]): number {
  if (records.length === 0) return 0;
  let sum = 0;
  for (const r of records) {
    if (r.estimated <= 0) continue;
    sum += Math.abs((r.actual - r.estimated) / r.estimated);
  }
  return (sum / records.length) * 100;
}

function computeMapeBreakdown(records: CalibrationRecord[]): MapeBreakdown {
  if (records.length === 0) {
    return { overall: 0, q1: 0, q2: 0, q3: 0, q4: 0 };
  }

  // Sort by actual token count for quartile split.
  const sorted = [...records].sort((a, b) => a.actual - b.actual);
  const n = sorted.length;
  const q1End = Math.floor(n * 0.25);
  const q2End = Math.floor(n * 0.50);
  const q3End = Math.floor(n * 0.75);

  return {
    overall: mapeForSlice(sorted),
    q1: mapeForSlice(sorted.slice(0, q1End)),
    q2: mapeForSlice(sorted.slice(q1End, q2End)),
    q3: mapeForSlice(sorted.slice(q2End, q3End)),
    q4: mapeForSlice(sorted.slice(q3End)),
  };
}

// ---------- nominal multiplier lookup (mirrors index.ts table) ----------

/**
 * Nominal (static) multiplier for a model string.
 * Mirrors the MODEL_ENCODING_MAP in index.ts so we can compute drift relative
 * to the baseline without importing the full index (avoids circular deps).
 */
export function getNominalMultiplier(model: string): number {
  const lower = model.toLowerCase();
  if (lower.includes("llama")) return 1.05;
  if (lower.includes("qwen")) return 1.08;
  if (lower.includes("mistral") || lower.includes("mixtral")) return 1.02;
  return 1.0;
}

/**
 * Nominal tiktoken encoding for a model string.
 * Used when constructing a `RecalibratedEncodingConfig`.
 */
export function getNominalEncoding(model: string): EncodingName {
  const lower = model.toLowerCase();
  if (
    lower.includes("claude-3") ||
    lower.includes("gpt-4o") ||
    lower.includes("o1") ||
    lower.includes("o3") ||
    lower.includes("gemini")
  ) {
    return "o200k_base";
  }
  return "cl100k_base";
}

// ---------- main calibration entry point ----------

/**
 * Read the sliding-window history for `model`, compute per-quartile MAPE,
 * flag drift when MAPE > DRIFT_THRESHOLD_PCT, and emit a new
 * `RecalibratedEncodingConfig` when drift is detected.
 *
 * The new multiplier is computed as:
 *   rawMultiplier  = nominalMultiplier × biasFactor
 *   smoothed       = EMA(currentCalibrated, rawMultiplier, EMA_ALPHA)
 *   clamped        = clamp(smoothed, MULTIPLIER_MIN, MULTIPLIER_MAX)
 *
 * @param model   Full or partial model name string.
 * @param options Optional overrides for drift threshold and window size.
 */
export async function validateAndRecalibrate(
  model: string,
  options?: {
    driftThresholdPct?: number;
    windowSize?: number;
    filePath?: string;
  },
): Promise<CalibrationResult> {
  const threshold = options?.driftThresholdPct ?? DRIFT_THRESHOLD_PCT;

  const records = options?.filePath
    ? await _readModelRecordsFromFile(model, options.filePath, options.windowSize ?? WINDOW_SIZE)
    : await readModelRecords(model);

  const sampleSize = records.length;

  if (sampleSize === 0) {
    return {
      model,
      sampleSize: 0,
      mape: { overall: 0, q1: 0, q2: 0, q3: 0, q4: 0 },
      biasFactor: 1.0,
      driftDetected: false,
      calibratedAt: new Date().toISOString(),
    };
  }

  const mape = computeMapeBreakdown(records);

  // Bias factor: mean(actual) / mean(estimated)
  const sumActual = records.reduce((s, r) => s + r.actual, 0);
  const sumEstimated = records.reduce((s, r) => s + r.estimated, 0);
  const biasFactor = sumEstimated > 0 ? sumActual / sumEstimated : 1.0;

  const driftDetected = mape.overall > threshold;

  let recalibrated: RecalibratedEncodingConfig | undefined;
  if (driftDetected) {
    const nominalMultiplier = getNominalMultiplier(model);
    const rawNewMultiplier = nominalMultiplier * biasFactor;
    // EMA smoothing: blend current active multiplier with new raw value.
    const currentActive = getActiveCalibratedMultiplier(model) ?? nominalMultiplier;
    const smoothed = currentActive + EMA_ALPHA * (rawNewMultiplier - currentActive);
    const clamped = Math.max(MULTIPLIER_MIN, Math.min(MULTIPLIER_MAX, smoothed));

    recalibrated = {
      model,
      encoding: getNominalEncoding(model),
      multiplier: Math.round(clamped * 10000) / 10000, // 4 decimal places
      priorMultiplier: currentActive,
      triggerMape: Math.round(mape.overall * 100) / 100,
    };

    // Apply in-memory so the next countTokensAccurate call picks it up.
    _calibratedMultipliers.set(model.toLowerCase(), clamped);
  }

  return {
    model,
    sampleSize,
    mape,
    biasFactor: Math.round(biasFactor * 10000) / 10000,
    driftDetected,
    recalibrated,
    calibratedAt: new Date().toISOString(),
  };
}

// ---------- internal helper for tests (injectable file path) ----------

async function _readModelRecordsFromFile(
  model: string,
  filePath: string,
  windowSize: number,
): Promise<CalibrationRecord[]> {
  try {
    const raw = await readFile(filePath, "utf8");
    const records: CalibrationRecord[] = [];
    for (const line of raw.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        records.push(JSON.parse(trimmed) as CalibrationRecord);
      } catch {
        // Skip malformed lines.
      }
    }
    const lower = model.toLowerCase();
    const modelRecords = records.filter((r) => {
      const rLower = r.model.toLowerCase();
      return rLower === lower || rLower.startsWith(lower) || lower.startsWith(rLower);
    });
    return modelRecords.slice(-windowSize);
  } catch {
    return [];
  }
}
