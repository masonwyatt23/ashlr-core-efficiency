/**
 * Cross-Provider Token Estimation Benchmarking & Calibration Report Generator
 *
 * Analyzes the state of RecalibrationEngine + CalibrationEngine across all
 * tracked providers and produces a structured TokenCalibrationReport for CI/DevOps
 * observability, fitness tracking, and proactive recalibration decisions.
 *
 * Usage:
 *   const report = await generateCalibrationReport(["claude-3-5", "gpt-4o"]);
 *   console.log(formatCalibrationReport(report));
 *   await persistCalibrationReport(report, process.cwd());
 *
 * CI integration:
 *   - Emits GitHub Actions step summary via GITHUB_STEP_SUMMARY env var.
 *   - Optionally posts to a Slack webhook via CALIBRATION_SLACK_WEBHOOK env var.
 */

import { join } from "path";
import { mkdirSync, existsSync } from "fs";
import { appendFile } from "fs/promises";
import { RecalibrationEngine, RECAL_MIN_SAMPLES } from "./recalibration-engine.ts";
import { CalibrationEngine, CE_MIN_SAMPLES } from "./calibration-engine.ts";
import type { MessageBin } from "./provider-codec.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** High-confidence threshold: >= this many samples. */
export const CONFIDENCE_HIGH_THRESHOLD = 50;

/** Low-confidence threshold: below this many samples. */
export const CONFIDENCE_LOW_THRESHOLD = 5;

/** MAPE threshold (%) above which we flag drift risk in recommendations. */
export const DRIFT_WARN_THRESHOLD_PCT = 8;

/** MAPE threshold (%) above which we flag critical drift. */
export const DRIFT_CRITICAL_THRESHOLD_PCT = 15;

/** Filename for the persisted calibration report JSONL. */
export const CALIBRATION_REPORT_FILENAME = "token-calibration-report.jsonl";

/** Path within cwd for genome evolution tracking. */
export const GENOME_EVOLUTION_DIR = ".ashlrcode/genome/evolution";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type Days = number;

/** Confidence level based on sample count and variance. */
export type ConfidenceLevel = "high" | "medium" | "low";

/** Per-bin breakdown within a provider report entry. */
export interface BinBreakdown {
  bin: MessageBin;
  samplesCount: number;
  correctionFactor: number;
  drift_pct: number;
}

/** Per-provider calibration assessment. */
export interface ProviderCalibrationEntry {
  /** Provider/model slug. */
  provider: string;
  /** Total observations recorded for this provider. */
  samplesCount: number;
  /**
   * Current live correction multiplier, or null when below MIN_SAMPLES
   * (engine has not yet converged; pre-trained baseline is used instead).
   */
  activeFactor: number | null;
  /**
   * Absolute percentage error from samples (weighted MAPE across all bins).
   * Represents the average estimation error observed.
   */
  drift_pct: number;
  /** Confidence level based on sample count and variance. */
  confidence: ConfidenceLevel;
  /** Timestamp of the most recent recorded sample. */
  lastUpdated: Date;
  /** Actionable recommendations for this provider. */
  recommendations: string[];
  /** Per-bin breakdown (only bins with at least 1 sample). */
  bins: BinBreakdown[];
}

/** Full cross-provider calibration report. */
export interface TokenCalibrationReport {
  /** ISO timestamp when this report was generated. */
  generatedAt: string;
  /** Providers included in this report. */
  providers: ProviderCalibrationEntry[];
  /** Summary statistics across all providers. */
  summary: CalibrationSummary;
  /** Raw JSON lines for machine-readable output. */
  jsonLines: string;
  /** Human-readable formatted text. */
  formattedText: string;
}

/** Aggregate statistics across all providers in the report. */
export interface CalibrationSummary {
  totalProviders: number;
  highConfidenceCount: number;
  mediumConfidenceCount: number;
  lowConfidenceCount: number;
  averageDrift_pct: number;
  providersWithCriticalDrift: string[];
  providersWithWarningDrift: string[];
  totalSamples: number;
}

// ---------------------------------------------------------------------------
// Core implementation
// ---------------------------------------------------------------------------

/**
 * Build a confidence level from sample count.
 * High: >= CONFIDENCE_HIGH_THRESHOLD samples
 * Medium: >= CONFIDENCE_LOW_THRESHOLD samples (and < high)
 * Low: < CONFIDENCE_LOW_THRESHOLD samples
 */
function computeConfidence(samplesCount: number): ConfidenceLevel {
  if (samplesCount >= CONFIDENCE_HIGH_THRESHOLD) return "high";
  if (samplesCount >= CONFIDENCE_LOW_THRESHOLD) return "medium";
  return "low";
}

/**
 * Generate actionable recommendations for a provider entry.
 */
function buildRecommendations(
  provider: string,
  samplesCount: number,
  activeFactor: number | null,
  drift_pct: number,
  confidence: ConfidenceLevel,
): string[] {
  const recs: string[] = [];

  // Sample collection recommendations
  if (confidence === "low") {
    const needed = CONFIDENCE_LOW_THRESHOLD - samplesCount;
    if (needed > 0) {
      recs.push(`Collect ${needed} more sample(s) to reach minimum confidence threshold.`);
    } else {
      recs.push(`Collect ${CONFIDENCE_HIGH_THRESHOLD - samplesCount} more samples to reach high confidence.`);
    }
  } else if (confidence === "medium") {
    const needed = CONFIDENCE_HIGH_THRESHOLD - samplesCount;
    recs.push(`Collect ${needed} more samples to reach high confidence (currently medium).`);
  }

  // Factor-based recommendations
  if (activeFactor !== null) {
    if (activeFactor > 1.15) {
      recs.push(
        `Factor suggests ${activeFactor.toFixed(3)}x multiplier — token estimates are running low by ~${((activeFactor - 1) * 100).toFixed(1)}%. Consider raising budget buffers for ${provider}.`,
      );
    } else if (activeFactor < 0.90) {
      recs.push(
        `Factor suggests ${activeFactor.toFixed(3)}x multiplier — token estimates are running high by ~${((1 - activeFactor) * 100).toFixed(1)}%. Estimates may be conservative for ${provider}.`,
      );
    } else if (Math.abs(activeFactor - 1.0) < 0.03) {
      recs.push(`Factor ${activeFactor.toFixed(3)}x is near-ideal — calibration is healthy for ${provider}.`);
    }
  } else {
    recs.push(`No live factor yet (below ${RECAL_MIN_SAMPLES} samples) — pre-trained baseline in use.`);
  }

  // Drift-based recommendations
  if (drift_pct >= DRIFT_CRITICAL_THRESHOLD_PCT) {
    recs.push(
      `Critical drift detected (${drift_pct.toFixed(1)}% MAPE > ${DRIFT_CRITICAL_THRESHOLD_PCT}%). Trigger recalibration immediately for ${provider}.`,
    );
  } else if (drift_pct >= DRIFT_WARN_THRESHOLD_PCT) {
    recs.push(
      `Elevated drift (${drift_pct.toFixed(1)}% MAPE > ${DRIFT_WARN_THRESHOLD_PCT}%). Monitor closely and schedule recalibration for ${provider}.`,
    );
  } else if (drift_pct > 0 && drift_pct < DRIFT_WARN_THRESHOLD_PCT) {
    recs.push(`Drift within acceptable range (${drift_pct.toFixed(1)}% MAPE). No action needed.`);
  }

  return recs;
}

/**
 * Collect state from both RecalibrationEngine and CalibrationEngine for
 * a single provider slug. Returns null if no data exists for the provider
 * in either engine.
 */
function collectProviderEntry(
  provider: string,
  recalEngine: RecalibrationEngine,
  calEngine: CalibrationEngine,
): ProviderCalibrationEntry | null {
  const recalState = recalEngine.getState(provider);
  const calState = calEngine.getState(provider);

  // Use whichever engine has data; prefer CalibrationEngine (more sophisticated)
  const totalSamples =
    calState?.totalSamples ?? recalState?.totalSamples ?? 0;

  if (totalSamples === 0 && !recalState && !calState) {
    return null;
  }

  // Active factor: use CalibrationEngine overall factor if enough samples,
  // else RecalibrationEngine overall factor, else null
  let activeFactor: number | null = null;
  const minSamples = Math.max(CE_MIN_SAMPLES, RECAL_MIN_SAMPLES);
  if (totalSamples >= minSamples) {
    activeFactor = calState
      ? calEngine.overallFactor(provider)
      : recalEngine.overallFactor(provider);
  }

  // Drift: weighted MAPE from CalibrationEngine, falling back to RecalibrationEngine
  const drift_pct = calState
    ? calEngine.mape(provider)
    : recalEngine.mape(provider);

  const confidence = computeConfidence(totalSamples);

  // Per-bin breakdown from CalibrationEngine bins (highest fidelity)
  const bins: BinBreakdown[] = [];
  const binSource = calState?.bins ?? recalState?.bins;
  if (binSource) {
    for (const [binName, binStats] of Object.entries(binSource)) {
      if (binStats.sampleCount > 0) {
        bins.push({
          bin: binName as MessageBin,
          samplesCount: binStats.sampleCount,
          correctionFactor: binStats.correctionFactor,
          drift_pct: binStats.mape,
        });
      }
    }
  }

  // Use current timestamp as lastUpdated approximation
  // (engines don't track per-provider timestamps, so we use now)
  const lastUpdated = new Date();

  const recommendations = buildRecommendations(
    provider,
    totalSamples,
    activeFactor,
    drift_pct,
    confidence,
  );

  return {
    provider,
    samplesCount: totalSamples,
    activeFactor,
    drift_pct,
    confidence,
    lastUpdated,
    recommendations,
    bins,
  };
}

/**
 * Generate a comprehensive calibration report for the given providers.
 *
 * Analyzes both RecalibrationEngine and CalibrationEngine state to produce
 * a unified per-provider assessment with recommendations.
 *
 * @param providers  List of provider/model slugs to include. If empty, all
 *                   known providers from both engines are included.
 * @param _maxAge    Optional: filter to providers updated within this many days
 *                   (currently used for forward-compat; engines don't track
 *                   per-provider update timestamps yet).
 */
export async function generateCalibrationReport(
  providers: string[],
  _maxAge?: Days,
): Promise<TokenCalibrationReport> {
  const recalEngine = RecalibrationEngine.instance;
  const calEngine = CalibrationEngine.instance;

  // Resolve provider list: if empty, union of all known providers from both engines
  let resolvedProviders = providers.length > 0 ? providers : [
    ...recalEngine.knownProviders(),
    ...calEngine.knownProviders(),
  ].filter((v, i, arr) => arr.indexOf(v) === i); // deduplicate

  // Collect entries
  const entries: ProviderCalibrationEntry[] = [];
  for (const provider of resolvedProviders) {
    const entry = collectProviderEntry(provider, recalEngine, calEngine);
    if (entry) {
      entries.push(entry);
    }
  }

  // Build summary
  const summary: CalibrationSummary = buildSummary(entries);

  // Build output formats
  const generatedAt = new Date().toISOString();
  const jsonLines = buildJsonLines(entries, generatedAt);
  const formattedText = buildFormattedText(entries, summary, generatedAt);

  return {
    generatedAt,
    providers: entries,
    summary,
    jsonLines,
    formattedText,
  };
}

// ---------------------------------------------------------------------------
// Summary builder
// ---------------------------------------------------------------------------

function buildSummary(entries: ProviderCalibrationEntry[]): CalibrationSummary {
  let highConfidenceCount = 0;
  let mediumConfidenceCount = 0;
  let lowConfidenceCount = 0;
  let totalDrift = 0;
  let totalSamples = 0;
  const providersWithCriticalDrift: string[] = [];
  const providersWithWarningDrift: string[] = [];

  for (const entry of entries) {
    switch (entry.confidence) {
      case "high":   highConfidenceCount++;   break;
      case "medium": mediumConfidenceCount++; break;
      case "low":    lowConfidenceCount++;    break;
    }
    totalDrift += entry.drift_pct;
    totalSamples += entry.samplesCount;

    if (entry.drift_pct >= DRIFT_CRITICAL_THRESHOLD_PCT) {
      providersWithCriticalDrift.push(entry.provider);
    } else if (entry.drift_pct >= DRIFT_WARN_THRESHOLD_PCT) {
      providersWithWarningDrift.push(entry.provider);
    }
  }

  return {
    totalProviders: entries.length,
    highConfidenceCount,
    mediumConfidenceCount,
    lowConfidenceCount,
    averageDrift_pct:
      entries.length > 0
        ? Math.round((totalDrift / entries.length) * 100) / 100
        : 0,
    providersWithCriticalDrift,
    providersWithWarningDrift,
    totalSamples,
  };
}

// ---------------------------------------------------------------------------
// Output formatters
// ---------------------------------------------------------------------------

/** Build newline-delimited JSON lines for machine-readable output. */
function buildJsonLines(
  entries: ProviderCalibrationEntry[],
  generatedAt: string,
): string {
  const lines = entries.map((entry) =>
    JSON.stringify({
      generatedAt,
      provider: entry.provider,
      samplesCount: entry.samplesCount,
      activeFactor: entry.activeFactor,
      drift_pct: entry.drift_pct,
      confidence: entry.confidence,
      lastUpdated: entry.lastUpdated.toISOString(),
      recommendations: entry.recommendations,
      bins: entry.bins,
    }),
  );
  return lines.join("\n");
}

/** Build a human-readable text summary for logs. */
function buildFormattedText(
  entries: ProviderCalibrationEntry[],
  summary: CalibrationSummary,
  generatedAt: string,
): string {
  const lines: string[] = [];

  lines.push("=".repeat(72));
  lines.push("  TOKEN CALIBRATION REPORT");
  lines.push(`  Generated: ${generatedAt}`);
  lines.push("=".repeat(72));
  lines.push("");

  // Summary block
  lines.push("SUMMARY");
  lines.push("-".repeat(40));
  lines.push(`  Providers analyzed:   ${summary.totalProviders}`);
  lines.push(`  Total samples:        ${summary.totalSamples}`);
  lines.push(`  Average drift (MAPE): ${summary.averageDrift_pct.toFixed(2)}%`);
  lines.push(`  Confidence breakdown: high=${summary.highConfidenceCount}  medium=${summary.mediumConfidenceCount}  low=${summary.lowConfidenceCount}`);
  if (summary.providersWithCriticalDrift.length > 0) {
    lines.push(`  CRITICAL DRIFT:       ${summary.providersWithCriticalDrift.join(", ")}`);
  }
  if (summary.providersWithWarningDrift.length > 0) {
    lines.push(`  Warning drift:        ${summary.providersWithWarningDrift.join(", ")}`);
  }
  lines.push("");

  // Per-provider details
  for (const entry of entries) {
    lines.push(`PROVIDER: ${entry.provider}`);
    lines.push("-".repeat(40));
    lines.push(`  Samples:       ${entry.samplesCount}`);
    lines.push(`  Active factor: ${entry.activeFactor !== null ? entry.activeFactor.toFixed(4) : "null (pre-trained baseline)"}`);
    lines.push(`  Drift (MAPE):  ${entry.drift_pct.toFixed(2)}%`);
    lines.push(`  Confidence:    ${entry.confidence.toUpperCase()}`);

    if (entry.bins.length > 0) {
      lines.push("  Bins:");
      for (const bin of entry.bins) {
        lines.push(
          `    ${bin.bin.padEnd(12)} samples=${bin.samplesCount}  factor=${bin.correctionFactor.toFixed(4)}  mape=${bin.drift_pct.toFixed(2)}%`,
        );
      }
    }

    if (entry.recommendations.length > 0) {
      lines.push("  Recommendations:");
      for (const rec of entry.recommendations) {
        lines.push(`    - ${rec}`);
      }
    }
    lines.push("");
  }

  if (entries.length === 0) {
    lines.push("  No provider data available. Record calibration samples first.");
    lines.push("");
  }

  lines.push("=".repeat(72));
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

/**
 * Persist a calibration report to the genome evolution JSONL file for
 * fitness tracking. Appends one JSON line per provider entry.
 *
 * File path: `<cwd>/.ashlrcode/genome/evolution/token-calibration-report.jsonl`
 *
 * @param report  The report to persist.
 * @param cwd     Working directory (defaults to process.cwd()).
 */
export async function persistCalibrationReport(
  report: TokenCalibrationReport,
  cwd: string = process.cwd(),
): Promise<void> {
  const evolutionDir = join(cwd, GENOME_EVOLUTION_DIR);
  try {
    if (!existsSync(evolutionDir)) {
      mkdirSync(evolutionDir, { recursive: true });
    }
    const filePath = join(evolutionDir, CALIBRATION_REPORT_FILENAME);
    await appendFile(filePath, report.jsonLines + "\n", "utf8");
  } catch {
    // Non-fatal — persistence is best-effort.
  }
}

// ---------------------------------------------------------------------------
// CI / webhook hooks
// ---------------------------------------------------------------------------

/**
 * Emit the report as a GitHub Actions step summary.
 * No-op when GITHUB_STEP_SUMMARY is not set (i.e. not running in CI).
 *
 * @param report  The calibration report to emit.
 */
export async function emitGitHubActionsSummary(
  report: TokenCalibrationReport,
): Promise<void> {
  const summaryPath = process.env["GITHUB_STEP_SUMMARY"];
  if (!summaryPath) return;
  try {
    const { appendFile: appendFileImpl } = await import("fs/promises");
    const md = buildMarkdownSummary(report);
    await appendFileImpl(summaryPath, md, "utf8");
  } catch {
    // Non-fatal.
  }
}

/** Build a Markdown summary suitable for GitHub Actions step summaries. */
export function buildMarkdownSummary(report: TokenCalibrationReport): string {
  const { summary } = report;
  const lines: string[] = [];

  lines.push("## Token Calibration Report");
  lines.push(`_Generated: ${report.generatedAt}_`);
  lines.push("");
  lines.push("### Summary");
  lines.push(`| Metric | Value |`);
  lines.push(`|--------|-------|`);
  lines.push(`| Providers | ${summary.totalProviders} |`);
  lines.push(`| Total samples | ${summary.totalSamples} |`);
  lines.push(`| Avg MAPE | ${summary.averageDrift_pct.toFixed(2)}% |`);
  lines.push(`| High confidence | ${summary.highConfidenceCount} |`);
  lines.push(`| Medium confidence | ${summary.mediumConfidenceCount} |`);
  lines.push(`| Low confidence | ${summary.lowConfidenceCount} |`);
  if (summary.providersWithCriticalDrift.length > 0) {
    lines.push(`| Critical drift | ${summary.providersWithCriticalDrift.join(", ")} |`);
  }
  lines.push("");

  if (report.providers.length > 0) {
    lines.push("### Per-Provider Details");
    lines.push("| Provider | Samples | Factor | MAPE | Confidence |");
    lines.push("|----------|---------|--------|------|------------|");
    for (const entry of report.providers) {
      const factor =
        entry.activeFactor !== null
          ? entry.activeFactor.toFixed(4)
          : "baseline";
      lines.push(
        `| ${entry.provider} | ${entry.samplesCount} | ${factor} | ${entry.drift_pct.toFixed(2)}% | ${entry.confidence} |`,
      );
    }
    lines.push("");
  }

  return lines.join("\n") + "\n";
}

/**
 * Post the report to a Slack webhook.
 * Reads webhook URL from `options.webhookUrl` or the CALIBRATION_SLACK_WEBHOOK
 * environment variable. No-op when neither is set.
 *
 * @param report   The calibration report to post.
 * @param options  Optional overrides.
 */
export async function postSlackWebhook(
  report: TokenCalibrationReport,
  options?: { webhookUrl?: string },
): Promise<void> {
  const webhookUrl =
    options?.webhookUrl ?? process.env["CALIBRATION_SLACK_WEBHOOK"];
  if (!webhookUrl) return;

  const { summary } = report;
  const status =
    summary.providersWithCriticalDrift.length > 0
      ? "CRITICAL"
      : summary.providersWithWarningDrift.length > 0
        ? "WARNING"
        : "OK";

  const payload = {
    text: `Token Calibration Report — Status: *${status}*`,
    blocks: [
      {
        type: "section",
        text: {
          type: "mrkdwn",
          text: [
            `*Token Calibration Report* — \`${report.generatedAt}\``,
            `Status: *${status}*`,
            `Providers: ${summary.totalProviders}  |  Samples: ${summary.totalSamples}  |  Avg MAPE: ${summary.averageDrift_pct.toFixed(2)}%`,
            summary.providersWithCriticalDrift.length > 0
              ? `:rotating_light: Critical drift: ${summary.providersWithCriticalDrift.join(", ")}`
              : "",
            summary.providersWithWarningDrift.length > 0
              ? `:warning: Warning drift: ${summary.providersWithWarningDrift.join(", ")}`
              : "",
          ]
            .filter(Boolean)
            .join("\n"),
        },
      },
    ],
  };

  try {
    await fetch(webhookUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    // Non-fatal — webhook delivery is best-effort.
  }
}
