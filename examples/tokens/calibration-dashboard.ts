/**
 * calibration-dashboard.ts — Cross-Provider Token Calibration Dashboard
 *
 * Demonstrates how to use generateCalibrationReport() to load calibration
 * state, pretty-print a human-readable report, and emit a Slack webhook
 * payload. Designed for CI pipelines and local DevOps inspection.
 *
 * Usage (dry-run with synthetic data):
 *   bun run examples/tokens/calibration-dashboard.ts
 *
 * Usage (against live ~/.ashlr data after warming up engines):
 *   LOAD_LIVE=1 bun run examples/tokens/calibration-dashboard.ts
 *
 * CI usage (posts GitHub Actions step summary):
 *   bun run examples/tokens/calibration-dashboard.ts
 *   # GITHUB_STEP_SUMMARY is picked up automatically in Actions runners.
 *
 * Slack usage:
 *   CALIBRATION_SLACK_WEBHOOK=https://hooks.slack.com/... bun run examples/tokens/calibration-dashboard.ts
 */

import { RecalibrationEngine } from "../../src/tokens/recalibration-engine.ts";
import { CalibrationEngine } from "../../src/tokens/calibration-engine.ts";
import {
  generateCalibrationReport,
  persistCalibrationReport,
  buildMarkdownSummary,
  postSlackWebhook,
  emitGitHubActionsSummary,
  CONFIDENCE_HIGH_THRESHOLD,
  CONFIDENCE_LOW_THRESHOLD,
  type TokenCalibrationReport,
  type ProviderCalibrationEntry,
} from "../../src/tokens/calibration-report.ts";
import type { Message } from "../../src/types/index.ts";

// ---------------------------------------------------------------------------
// Synthetic data seed helpers
// ---------------------------------------------------------------------------

function txt(text: string): Message {
  return { role: "user", content: text };
}

function toolResultMsg(): Message {
  return {
    role: "user",
    content: [{ type: "tool_result", tool_use_id: "t1", content: "result" }],
  };
}

function thinkingMsg(): Message {
  return {
    role: "user",
    content: [{ type: "thinking", thinking: "let me reason step by step..." }],
  };
}

/**
 * Seed the CalibrationEngine with realistic synthetic observations.
 * Each provider represents a common fleet scenario.
 */
function seedSyntheticData(): void {
  const engine = CalibrationEngine.instance;

  // claude-3-5-sonnet: high confidence, near-perfect calibration
  // 60 samples, actual ≈ estimated (±2%)
  for (let i = 0; i < 60; i++) {
    const jitter = 1 + (Math.sin(i) * 0.02); // ±2% noise
    engine.record({
      provider: "claude-3-5-sonnet",
      messages: [txt("hello world, analyze this document carefully")],
      estimated: 120,
      actual: Math.round(120 * jitter),
    });
  }
  // Also seed tool_use bin for claude-3-5
  for (let i = 0; i < 20; i++) {
    engine.record({
      provider: "claude-3-5-sonnet",
      messages: [toolResultMsg()],
      estimated: 200,
      actual: Math.round(200 * 1.04), // 4% under-estimate on tool_result
    });
  }

  // gpt-4o: medium confidence, slight under-estimation
  // 25 samples, actual ≈ 1.07× estimated
  for (let i = 0; i < 25; i++) {
    engine.record({
      provider: "gpt-4o",
      messages: [txt("summarize the following text: " + "x".repeat(100))],
      estimated: 180,
      actual: Math.round(180 * 1.07),
    });
  }

  // llama3-70b: medium confidence, moderate under-estimation (Llama tokenizer drift)
  // 18 samples, actual ≈ 1.12× estimated
  for (let i = 0; i < 18; i++) {
    engine.record({
      provider: "llama3-70b",
      messages: [txt("generate code for the following function: " + "z".repeat(80))],
      estimated: 250,
      actual: Math.round(250 * 1.12),
    });
  }

  // qwen-7b: low confidence — only 3 samples, high drift
  // 3 samples, actual ≈ 1.35× estimated (CJK tokenizer mismatch)
  for (let i = 0; i < 3; i++) {
    engine.record({
      provider: "qwen-7b",
      messages: [txt("分析这段文字的语义和结构特点。")], // Chinese text
      estimated: 80,
      actual: Math.round(80 * 1.35),
    });
  }

  // mistral-7b: high confidence, thinking bin with elevated drift
  // 55 samples, actual ≈ 1.20× estimated (heavy thinking overhead)
  for (let i = 0; i < 55; i++) {
    engine.record({
      provider: "mistral-7b",
      messages: [thinkingMsg()],
      estimated: 300,
      actual: Math.round(300 * 1.20),
    });
  }
}

// ---------------------------------------------------------------------------
// Pretty-print helpers
// ---------------------------------------------------------------------------

function printBanner(title: string): void {
  const width = 72;
  console.log("\n" + "=".repeat(width));
  console.log(`  ${title}`);
  console.log("=".repeat(width));
}

function printEntry(entry: ProviderCalibrationEntry): void {
  const confidenceIcon =
    entry.confidence === "high" ? "✓" :
    entry.confidence === "medium" ? "~" : "!";

  const factorStr =
    entry.activeFactor !== null
      ? entry.activeFactor.toFixed(4)
      : "null (pre-trained baseline)";

  const driftStatus =
    entry.drift_pct === 0 ? "OK" :
    entry.drift_pct >= 15 ? "CRITICAL" :
    entry.drift_pct >= 8 ? "WARNING" : "OK";

  console.log(`\n[${confidenceIcon}] ${entry.provider}`);
  console.log(`    Samples:       ${entry.samplesCount}  (confidence: ${entry.confidence.toUpperCase()})`);
  console.log(`    Active factor: ${factorStr}`);
  console.log(`    Drift (MAPE):  ${entry.drift_pct.toFixed(2)}%  [${driftStatus}]`);

  if (entry.bins.length > 0) {
    console.log(`    Bins:`);
    for (const bin of entry.bins) {
      console.log(
        `      ${bin.bin.padEnd(12)} n=${bin.samplesCount}  ` +
        `factor=${bin.correctionFactor.toFixed(4)}  mape=${bin.drift_pct.toFixed(2)}%`,
      );
    }
  }

  if (entry.recommendations.length > 0) {
    console.log(`    Recommendations:`);
    for (const rec of entry.recommendations) {
      console.log(`      > ${rec}`);
    }
  }
}

function printSummaryTable(report: TokenCalibrationReport): void {
  const { summary } = report;
  console.log("\nSUMMARY TABLE");
  console.log("-".repeat(48));
  console.log(`  Total providers:  ${summary.totalProviders}`);
  console.log(`  Total samples:    ${summary.totalSamples}`);
  console.log(`  Avg MAPE:         ${summary.averageDrift_pct.toFixed(2)}%`);
  console.log(`  Confidence:       high=${summary.highConfidenceCount}  medium=${summary.mediumConfidenceCount}  low=${summary.lowConfidenceCount}`);

  if (summary.providersWithCriticalDrift.length > 0) {
    console.log(`  CRITICAL DRIFT:   ${summary.providersWithCriticalDrift.join(", ")}`);
  }
  if (summary.providersWithWarningDrift.length > 0) {
    console.log(`  Warning drift:    ${summary.providersWithWarningDrift.join(", ")}`);
  }
  console.log("-".repeat(48));
}

// ---------------------------------------------------------------------------
// Webhook demo
// ---------------------------------------------------------------------------

function printWebhookDemoPayload(report: TokenCalibrationReport): void {
  const { summary } = report;
  const status =
    summary.providersWithCriticalDrift.length > 0
      ? "CRITICAL"
      : summary.providersWithWarningDrift.length > 0
        ? "WARNING"
        : "OK";

  const payload = {
    text: `Token Calibration Report — Status: ${status}`,
    blocks: [
      {
        type: "section",
        text: {
          type: "mrkdwn",
          text: [
            `*Token Calibration Report* — \`${report.generatedAt}\``,
            `Status: *${status}*`,
            `Providers: ${summary.totalProviders}  |  Samples: ${summary.totalSamples}  |  Avg MAPE: ${summary.averageDrift_pct.toFixed(2)}%`,
          ].join("\n"),
        },
      },
    ],
  };

  printBanner("SLACK WEBHOOK DEMO PAYLOAD");
  console.log(JSON.stringify(payload, null, 2));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const loadLive = process.env["LOAD_LIVE"] === "1";

  if (!loadLive) {
    // Seed with synthetic data for demonstration
    printBanner("SEEDING SYNTHETIC CALIBRATION DATA");
    console.log("  Seeding 5 providers with synthetic observations...");
    seedSyntheticData();
    console.log(`  Done. High-confidence threshold: ${CONFIDENCE_HIGH_THRESHOLD} samples`);
    console.log(`  Min-confidence threshold:        ${CONFIDENCE_LOW_THRESHOLD} samples`);
  } else {
    // Load live data from disk (warm up engines from existing JSONL files)
    console.log("  Loading live calibration data from ~/.ashlr/...");
    await CalibrationEngine.instance.loadFromFile();
    console.log("  Live data loaded.");
  }

  // Generate the report for all known providers (empty list = auto-discover)
  console.log("\n  Generating calibration report...");
  const report = await generateCalibrationReport([]);

  // Print human-readable report
  printBanner("CALIBRATION DASHBOARD");
  for (const entry of report.providers) {
    printEntry(entry);
  }

  printSummaryTable(report);

  // Print formatted text summary (same as what logs would show)
  printBanner("FULL FORMATTED LOG OUTPUT");
  console.log(report.formattedText);

  // Show GitHub Actions Markdown summary format
  printBanner("GITHUB ACTIONS MARKDOWN SUMMARY (preview)");
  console.log(buildMarkdownSummary(report));

  // Demo Slack webhook payload format (no actual send unless env var is set)
  printWebhookDemoPayload(report);

  // Persist to genome evolution dir (uses process.cwd())
  printBanner("PERSISTING REPORT");
  const cwd = process.cwd();
  await persistCalibrationReport(report, cwd);
  console.log(`  Report appended to: ${cwd}/.ashlrcode/genome/evolution/token-calibration-report.jsonl`);

  // Emit GitHub Actions step summary (no-op if not in CI)
  await emitGitHubActionsSummary(report);

  // Post Slack webhook (no-op if CALIBRATION_SLACK_WEBHOOK not set)
  await postSlackWebhook(report);

  printBanner("DONE");
  console.log("  Calibration dashboard complete.\n");
}

main().catch((err) => {
  console.error("calibration-dashboard error:", err);
  process.exit(1);
});
