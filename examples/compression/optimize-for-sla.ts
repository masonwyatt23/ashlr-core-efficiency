/**
 * optimize-for-sla.ts — SLA-driven compression tier selection
 *
 * Demonstrates how to use the Multi-Tier Compression ROI Optimizer to pick
 * a compression tier that satisfies explicit latency + cost SLA constraints.
 *
 * Use case: "pick the tier that keeps latency < 200 ms AND cost < $0.50
 * per call, while maximising token savings."
 *
 * Run (no API key required — uses synthetic session-log data):
 *   bun run examples/compression/optimize-for-sla.ts
 *
 * Windows note: no shell-outs; runs identically on macOS / Linux / Windows.
 */

import { BackpropagationEngine } from "../../src/session-log/backpropagation.ts";
import {
  analyzeLatencyROI,
  recommendTierForLatencyBudget,
  getModelLatencyCoefficient,
  getLatencyDashboard,
} from "../../src/compression/latency-roi-analyzer.ts";
import type { CompressionTier } from "../../src/compression/context.ts";

// ---------------------------------------------------------------------------
// 1. Bootstrap a BackpropagationEngine with synthetic session data
// ---------------------------------------------------------------------------

const engine = new BackpropagationEngine();

/**
 * Simulate historical session data by injecting synthetic attribution
 * records for each tier. In production these come from `attributeSessionCost`
 * being called after each real LLM response.
 *
 * Tier semantics (fastest → slowest, cheapest → most savings):
 *   Tier 4 — tree prune:      fast, minimal savings
 *   Tier 3 — context collapse: moderate latency, moderate savings
 *   Tier 2 — snip compact:    moderate latency, good savings
 *   Tier 1 — LLM summarise:   slowest, maximum savings (LLM overhead)
 */
async function populateSyntheticHistory(model: string): Promise<void> {
  // Tier 4: very fast, small token removal
  for (let i = 0; i < 8; i++) {
    await engine.record(4, 20, model, 50_000, 48_000, 0.06, 45 + i * 2);
  }

  // Tier 3: fast, moderate removal
  for (let i = 0; i < 8; i++) {
    await engine.record(3, 20, model, 50_000, 42_000, 0.05, 90 + i * 5);
  }

  // Tier 2: moderate, good removal
  for (let i = 0; i < 8; i++) {
    await engine.record(2, 20, model, 50_000, 35_000, 0.04, 160 + i * 10);
  }

  // Tier 1: slow (LLM summarisation call), maximum removal
  for (let i = 0; i < 8; i++) {
    await engine.record(1, 20, model, 50_000, 10_000, 0.03, 800 + i * 50);
  }
}

// ---------------------------------------------------------------------------
// 2. Helper: describe a tier recommendation
// ---------------------------------------------------------------------------

function describeTier(tier: CompressionTier): string {
  const descriptions: Record<CompressionTier, string> = {
    1: "Tier 1 — autoCompact (LLM summarisation, max savings, slowest)",
    2: "Tier 2 — snipCompact (truncate large tool results, good savings)",
    3: "Tier 3 — contextCollapse (remove short/dup turns, moderate savings)",
    4: "Tier 4 — treeCompact (prune low-value subtrees, minimal savings, fastest)",
  };
  return descriptions[tier];
}

// ---------------------------------------------------------------------------
// 3. SLA-driven selection examples
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const model = "claude-3-5-sonnet";

  console.log("=== SLA-Driven Compression Tier Selection ===\n");
  console.log(`Model: ${model}`);
  console.log(`Latency coefficient: ${getModelLatencyCoefficient(model)} ms/tier-step\n`);

  // Seed the engine with synthetic history.
  await populateSyntheticHistory(model);

  // Build the latency-ROI profile from session history.
  const profile = await analyzeLatencyROI(model, ".", engine);

  console.log("--- Per-Tier Statistics (from session history) ---");
  for (const tier of profile.tiers) {
    console.log(
      `  Tier ${tier.tier}: ` +
      `latency P50=${tier.p50LatencyMs.toFixed(0)}ms P95=${tier.p95LatencyMs.toFixed(0)}ms ` +
      `cost=$${tier.meanCostUsd.toFixed(4)} ` +
      `saved=$${tier.meanUsdSaved.toFixed(4)} ` +
      `quality_loss=${tier.qualityLossPct.toFixed(1)}% ` +
      `samples=${tier.sampleCount}`,
    );
  }
  console.log();

  console.log(`Pareto-efficient tiers: [${profile.paretoFrontier.join(", ")}]\n`);

  // ---------------------------------------------------------------------------
  // SLA scenario A: Real-time chat — latency < 200 ms, cost < $0.50
  // ---------------------------------------------------------------------------
  const slaA_latency = 200; // ms
  const slaA_cost = 0.50;   // USD
  const recA = recommendTierForLatencyBudget(slaA_latency, slaA_cost, profile);
  console.log(`--- SLA A: Real-time chat (latency < ${slaA_latency}ms, cost < $${slaA_cost}) ---`);
  console.log(`  Recommended: ${describeTier(recA)}`);
  const tierStatsA = profile.tiers.find((t) => t.tier === recA)!;
  if (tierStatsA) {
    console.log(`  Expected latency: ~${tierStatsA.meanLatencyMs.toFixed(0)}ms`);
    console.log(`  Expected cost:    ~$${tierStatsA.meanCostUsd.toFixed(4)}`);
    console.log(`  Expected savings: ~$${tierStatsA.meanUsdSaved.toFixed(4)}`);
  }
  console.log();

  // ---------------------------------------------------------------------------
  // SLA scenario B: Batch processing — latency < 1000 ms, cost < $0.05
  // ---------------------------------------------------------------------------
  const slaB_latency = 1000; // ms
  const slaB_cost = 0.05;    // USD
  const recB = recommendTierForLatencyBudget(slaB_latency, slaB_cost, profile);
  console.log(`--- SLA B: Batch processing (latency < ${slaB_latency}ms, cost < $${slaB_cost}) ---`);
  console.log(`  Recommended: ${describeTier(recB)}`);
  const tierStatsB = profile.tiers.find((t) => t.tier === recB)!;
  if (tierStatsB) {
    console.log(`  Expected latency: ~${tierStatsB.meanLatencyMs.toFixed(0)}ms`);
    console.log(`  Expected cost:    ~$${tierStatsB.meanCostUsd.toFixed(4)}`);
    console.log(`  Expected savings: ~$${tierStatsB.meanUsdSaved.toFixed(4)}`);
  }
  console.log();

  // ---------------------------------------------------------------------------
  // SLA scenario C: Cost-sensitive background task — latency < 5000ms, cost < $0.04
  // ---------------------------------------------------------------------------
  const slaC_latency = 5000; // ms
  const slaC_cost = 0.04;    // USD
  const recC = recommendTierForLatencyBudget(slaC_latency, slaC_cost, profile);
  console.log(`--- SLA C: Cost-sensitive task (latency < ${slaC_latency}ms, cost < $${slaC_cost}) ---`);
  console.log(`  Recommended: ${describeTier(recC)}`);
  const tierStatsC = profile.tiers.find((t) => t.tier === recC)!;
  if (tierStatsC) {
    console.log(`  Expected latency: ~${tierStatsC.meanLatencyMs.toFixed(0)}ms`);
    console.log(`  Expected cost:    ~$${tierStatsC.meanCostUsd.toFixed(4)}`);
    console.log(`  Expected savings: ~$${tierStatsC.meanUsdSaved.toFixed(4)}`);
  }
  console.log();

  // ---------------------------------------------------------------------------
  // SLA scenario D: Impossible SLA — latency < 10ms (no tier can satisfy it)
  // ---------------------------------------------------------------------------
  const slaD_latency = 10;  // ms — no tier will fit
  const slaD_cost = 0.001;  // USD
  const recD = recommendTierForLatencyBudget(slaD_latency, slaD_cost, profile);
  console.log(`--- SLA D: Impossible SLA (latency < ${slaD_latency}ms, cost < $${slaD_cost}) ---`);
  console.log(`  No tier satisfies both constraints. Least-bad option:`);
  console.log(`  Recommended: ${describeTier(recD)}`);
  console.log();

  // ---------------------------------------------------------------------------
  // 4. Multi-model fleet dashboard
  // ---------------------------------------------------------------------------

  // Add a second model (gpt-4o) to the engine for comparison.
  await populateSyntheticHistoryGpt();

  const dashboard = await getLatencyDashboard(engine);
  console.log("--- Fleet-Wide Latency Dashboard ---");
  for (const [mdl, stats] of Object.entries(dashboard.fastestTierByModel)) {
    const cheapest = dashboard.cheapestTierByModel[mdl] ?? "N/A";
    console.log(
      `  ${mdl}: fastest=Tier ${stats}  cheapest=Tier ${cheapest}`,
    );
  }
  console.log();

  console.log("Done. In production, replace synthetic records with:");
  console.log("  attributeSessionCost(tier, msgCount, model, estTokens, actTokens, cost, latency)");
  console.log("after every real LLM call. The profile updates incrementally.\n");
}

/** Inject synthetic GPT-4o history for fleet comparison. */
async function populateSyntheticHistoryGpt(): Promise<void> {
  const model = "gpt-4o";
  for (let i = 0; i < 8; i++) {
    await engine.record(4, 20, model, 50_000, 48_000, 0.07, 30 + i * 2);
  }
  for (let i = 0; i < 8; i++) {
    await engine.record(3, 20, model, 50_000, 42_000, 0.06, 70 + i * 3);
  }
  for (let i = 0; i < 8; i++) {
    await engine.record(2, 20, model, 50_000, 35_000, 0.05, 120 + i * 8);
  }
  for (let i = 0; i < 8; i++) {
    await engine.record(1, 20, model, 50_000, 10_000, 0.04, 600 + i * 40);
  }
}

await main();
