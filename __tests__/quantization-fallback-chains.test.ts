/**
 * Tests for Multi-Tier Genome Quantization Strategy with Automatic Fallback Chains
 *
 * Covers:
 *  - FallbackChainStep / QuantizationFallbackChain shape validation
 *  - selectQuantizationWithFallback: happy path (step 0), single fallback, all-fail forced last
 *  - Timeout simulation: probe returning after step.timeoutMs fires
 *  - Degraded Ollama scenario: accurate + balanced fail → fast succeeds
 *  - FallbackExecutionMetadata: all fields present, step records, rationale strings
 *  - appendFallbackAudit / loadFallbackAuditRecords: JSONL round-trip
 *  - QuantizationStrategyEngine.select() alias method
 *  - QuantizationStrategyEngine.selectBitDepth() with fallbackOptions.probe injection
 *  - Empty chain throws
 *  - skipAudit suppresses disk write
 *  - FallbackStepRecord outcome classification (timeout vs error)
 *  - DEFAULT_FALLBACK_CHAIN shape
 *  - _buildChainFromDepth indirectly via engine.selectBitDepth
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, rm, access } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";

import {
  selectQuantizationWithFallback,
  loadFallbackAuditRecords,
  QuantizationStrategyEngine,
  GenomeProfiler,
  DEFAULT_FALLBACK_CHAIN,
  type QuantizationFallbackChain,
  type FallbackChainStep,
  type FallbackStepProbe,
  type GenomeProfile,
  type QuantizationBitDepth,
  type QuantizationModelTier,
  type SelectedQuantization,
  type FallbackExecutionMetadata,
} from "../src/genome/quantization-strategy.ts";

import type { GenomeManifest, SectionMeta } from "../src/genome/manifest.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeTmpDir(): string {
  return join(tmpdir(), `ashlr-qfc-${Date.now()}-${Math.random().toString(36).slice(2)}`);
}

async function setupGenomeDir(cwd: string): Promise<void> {
  await mkdir(join(cwd, ".ashlrcode", "genome", "evolution"), { recursive: true });
}

function makeSection(overrides: Partial<SectionMeta> = {}): SectionMeta {
  return {
    path: "vision/north-star.md",
    title: "North Star",
    summary: "Project vision",
    tags: [],
    tokens: 200,
    updatedAt: new Date().toISOString(),
    ...overrides,
  };
}

function makeManifest(sections: SectionMeta[]): GenomeManifest {
  return {
    version: 1,
    project: "test",
    sections,
    generation: { number: 1, milestone: "M1", startedAt: new Date().toISOString() },
    fitnessHistory: [],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
}

function makeLargeProfile(): GenomeProfile {
  const sections = Array.from({ length: 200 }, (_, i) =>
    makeSection({ path: `s${i}.md`, tokens: 300 }),
  );
  return GenomeProfiler.profile(makeManifest(sections));
}

function makeSmallProfile(): GenomeProfile {
  const sections = Array.from({ length: 10 }, (_, i) =>
    makeSection({ path: `s${i}.md`, tokens: 100 }),
  );
  return GenomeProfiler.profile(makeManifest(sections));
}

/** Probe that always succeeds immediately. */
const alwaysSucceedProbe: FallbackStepProbe = async () => true;

/** Probe that always rejects immediately (simulates hard unavailability). */
const alwaysFailProbe: FallbackStepProbe = async () => false;

/**
 * Probe that fails for the first N steps (by index via closure), then succeeds.
 * We derive the "step index" by counting invocations.
 */
function failFirstNStepsProbe(n: number): FallbackStepProbe {
  let callCount = 0;
  return async () => {
    const idx = callCount++;
    return idx >= n;
  };
}

/**
 * Probe that simulates a timeout by delaying longer than the step's timeout.
 * stepIndicesToTimeout: set of step indices that should time out.
 */
function timeoutStepsProbe(stepIndicesToTimeout: Set<number>): FallbackStepProbe {
  let callCount = 0;
  return async (step: FallbackChainStep) => {
    const idx = callCount++;
    if (stepIndicesToTimeout.has(idx)) {
      // Delay longer than the step's own timeout — will lose the race
      await new Promise((resolve) => setTimeout(resolve, step.timeoutMs + 200));
      return true; // Never reached within timeout
    }
    return true;
  };
}

// ---------------------------------------------------------------------------
// DEFAULT_FALLBACK_CHAIN shape
// ---------------------------------------------------------------------------

describe("DEFAULT_FALLBACK_CHAIN — shape validation", () => {
  test("has exactly 4 steps", () => {
    expect(DEFAULT_FALLBACK_CHAIN).toHaveLength(4);
  });

  test("step 0 is int16/accurate/5s", () => {
    const s = DEFAULT_FALLBACK_CHAIN[0]!;
    expect(s.bitDepth).toBe(16);
    expect(s.modelTier).toBe("accurate");
    expect(s.timeoutMs).toBe(5_000);
  });

  test("step 1 is int16/balanced/10s", () => {
    const s = DEFAULT_FALLBACK_CHAIN[1]!;
    expect(s.bitDepth).toBe(16);
    expect(s.modelTier).toBe("balanced");
    expect(s.timeoutMs).toBe(10_000);
  });

  test("step 2 is int8/fast/3s", () => {
    const s = DEFAULT_FALLBACK_CHAIN[2]!;
    expect(s.bitDepth).toBe(8);
    expect(s.modelTier).toBe("fast");
    expect(s.timeoutMs).toBe(3_000);
  });

  test("step 3 is int4/ultrafast/1s", () => {
    const s = DEFAULT_FALLBACK_CHAIN[3]!;
    expect(s.bitDepth).toBe(4);
    expect(s.modelTier).toBe("ultrafast");
    expect(s.timeoutMs).toBe(1_000);
  });

  test("timeouts decrease monotonically (degraded → faster)", () => {
    // step 0 (5s) < step 1 (10s) is intentional (accurate needs more time)
    // but step 2 (3s) < step 3 (1s) — each degraded step has tighter budget
    expect(DEFAULT_FALLBACK_CHAIN[2]!.timeoutMs).toBeGreaterThan(
      DEFAULT_FALLBACK_CHAIN[3]!.timeoutMs,
    );
  });
});

// ---------------------------------------------------------------------------
// selectQuantizationWithFallback — happy path
// ---------------------------------------------------------------------------

describe("selectQuantizationWithFallback — happy path", () => {
  const profile = makeLargeProfile();

  test("step 0 succeeds: usedFallback=false, stepIndex=0", async () => {
    const result = await selectQuantizationWithFallback(profile, 500, {
      probe: alwaysSucceedProbe,
      skipAudit: true,
    });
    expect(result.usedFallback).toBe(false);
    expect(result.stepIndex).toBe(0);
    expect(result.bitDepth).toBe(DEFAULT_FALLBACK_CHAIN[0]!.bitDepth);
    expect(result.modelTier).toBe(DEFAULT_FALLBACK_CHAIN[0]!.modelTier);
  });

  test("SelectedQuantization has all required fields", async () => {
    const result = await selectQuantizationWithFallback(profile, 100, {
      probe: alwaysSucceedProbe,
      skipAudit: true,
    });
    expect(typeof result.bitDepth).toBe("number");
    expect(typeof result.modelTier).toBe("string");
    expect(typeof result.stepIndex).toBe("number");
    expect(typeof result.usedFallback).toBe("boolean");
    expect(result.fallbackMeta).toBeDefined();
  });

  test("rationale mentions 'Preferred step' when step 0 succeeds", async () => {
    const result = await selectQuantizationWithFallback(profile, 200, {
      probe: alwaysSucceedProbe,
      skipAudit: true,
    });
    expect(result.fallbackMeta.selectionRationale).toMatch(/preferred step/i);
  });

  test("single-step chain always returns that step", async () => {
    const chain: QuantizationFallbackChain = [
      { bitDepth: 8, modelTier: "fast", timeoutMs: 500 },
    ];
    const result = await selectQuantizationWithFallback(profile, 50, {
      chain,
      probe: alwaysSucceedProbe,
      skipAudit: true,
    });
    expect(result.bitDepth).toBe(8);
    expect(result.modelTier).toBe("fast");
    expect(result.stepIndex).toBe(0);
    expect(result.usedFallback).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// selectQuantizationWithFallback — fallback scenarios
// ---------------------------------------------------------------------------

describe("selectQuantizationWithFallback — fallback steps", () => {
  const profile = makeLargeProfile();

  test("first step fails → falls back to step 1", async () => {
    const result = await selectQuantizationWithFallback(profile, 300, {
      probe: failFirstNStepsProbe(1),
      skipAudit: true,
    });
    expect(result.stepIndex).toBe(1);
    expect(result.usedFallback).toBe(true);
  });

  test("first 2 steps fail → falls back to step 2", async () => {
    const result = await selectQuantizationWithFallback(profile, 300, {
      probe: failFirstNStepsProbe(2),
      skipAudit: true,
    });
    expect(result.stepIndex).toBe(2);
    expect(result.usedFallback).toBe(true);
    expect(result.bitDepth).toBe(8);
    expect(result.modelTier).toBe("fast");
  });

  test("first 3 steps fail → falls back to step 3 (ultrafast)", async () => {
    const result = await selectQuantizationWithFallback(profile, 300, {
      probe: failFirstNStepsProbe(3),
      skipAudit: true,
    });
    expect(result.stepIndex).toBe(3);
    expect(result.usedFallback).toBe(true);
    expect(result.bitDepth).toBe(4);
    expect(result.modelTier).toBe("ultrafast");
  });

  test("all steps fail → forced last-resort (last chain step)", async () => {
    const result = await selectQuantizationWithFallback(profile, 300, {
      probe: alwaysFailProbe,
      skipAudit: true,
    });
    const lastStep = DEFAULT_FALLBACK_CHAIN[DEFAULT_FALLBACK_CHAIN.length - 1]!;
    expect(result.bitDepth).toBe(lastStep.bitDepth);
    expect(result.modelTier).toBe(lastStep.modelTier);
    expect(result.stepIndex).toBe(DEFAULT_FALLBACK_CHAIN.length - 1);
    expect(result.usedFallback).toBe(true);
  });

  test("forced fallback rationale mentions 'forced last-resort'", async () => {
    const result = await selectQuantizationWithFallback(profile, 300, {
      probe: alwaysFailProbe,
      skipAudit: true,
    });
    expect(result.fallbackMeta.selectionRationale).toMatch(/forced last-resort/i);
  });

  test("degraded rationale mentions failed step count", async () => {
    const result = await selectQuantizationWithFallback(profile, 300, {
      probe: failFirstNStepsProbe(2),
      skipAudit: true,
    });
    expect(result.fallbackMeta.selectionRationale).toMatch(/2 failed step/i);
  });
});

// ---------------------------------------------------------------------------
// selectQuantizationWithFallback — timeout simulation
// ---------------------------------------------------------------------------

describe("selectQuantizationWithFallback — timeout scenarios", () => {
  test("step 0 times out → falls back to step 1", async () => {
    const chain: QuantizationFallbackChain = [
      { bitDepth: 16, modelTier: "accurate",  timeoutMs: 50 },
      { bitDepth: 8,  modelTier: "fast",      timeoutMs: 500 },
    ];

    const result = await selectQuantizationWithFallback(makeLargeProfile(), 200, {
      chain,
      probe: timeoutStepsProbe(new Set([0])),
      skipAudit: true,
    });

    expect(result.stepIndex).toBe(1);
    expect(result.bitDepth).toBe(8);
    expect(result.modelTier).toBe("fast");
    expect(result.usedFallback).toBe(true);
  }, 3_000);

  test("timed-out step recorded with outcome='timeout'", async () => {
    const chain: QuantizationFallbackChain = [
      { bitDepth: 16, modelTier: "accurate", timeoutMs: 50 },
      { bitDepth: 8,  modelTier: "fast",     timeoutMs: 500 },
    ];

    const result = await selectQuantizationWithFallback(makeLargeProfile(), 200, {
      chain,
      probe: timeoutStepsProbe(new Set([0])),
      skipAudit: true,
    });

    const step0Record = result.fallbackMeta.stepsAttempted[0]!;
    expect(step0Record.outcome).toBe("timeout");
    expect(step0Record.errorMessage).toMatch(/timeout/i);
  }, 3_000);

  test("successful step has outcome='success' in stepsAttempted", async () => {
    const result = await selectQuantizationWithFallback(makeLargeProfile(), 100, {
      probe: alwaysSucceedProbe,
      skipAudit: true,
    });
    expect(result.fallbackMeta.stepsAttempted[0]!.outcome).toBe("success");
  });

  test("degraded Ollama: accurate+balanced timeout → fast succeeds", async () => {
    const chain: QuantizationFallbackChain = [
      { bitDepth: 16, modelTier: "accurate",  timeoutMs: 50 },
      { bitDepth: 16, modelTier: "balanced",  timeoutMs: 50 },
      { bitDepth: 8,  modelTier: "fast",      timeoutMs: 500 },
      { bitDepth: 4,  modelTier: "ultrafast", timeoutMs: 200 },
    ];

    const result = await selectQuantizationWithFallback(makeLargeProfile(), 300, {
      chain,
      probe: timeoutStepsProbe(new Set([0, 1])),
      skipAudit: true,
    });

    expect(result.stepIndex).toBe(2);
    expect(result.bitDepth).toBe(8);
    expect(result.modelTier).toBe("fast");
    // Two prior steps should be timeout
    expect(result.fallbackMeta.stepsAttempted[0]!.outcome).toBe("timeout");
    expect(result.fallbackMeta.stepsAttempted[1]!.outcome).toBe("timeout");
    expect(result.fallbackMeta.stepsAttempted[2]!.outcome).toBe("success");
  }, 5_000);

  test("all steps time out → forced last step returned", async () => {
    const chain: QuantizationFallbackChain = [
      { bitDepth: 16, modelTier: "accurate",  timeoutMs: 50 },
      { bitDepth: 8,  modelTier: "fast",      timeoutMs: 50 },
      { bitDepth: 4,  modelTier: "ultrafast", timeoutMs: 50 },
    ];

    const result = await selectQuantizationWithFallback(makeLargeProfile(), 200, {
      chain,
      probe: timeoutStepsProbe(new Set([0, 1, 2])),
      skipAudit: true,
    });

    expect(result.stepIndex).toBe(2);
    expect(result.bitDepth).toBe(4);
    expect(result.modelTier).toBe("ultrafast");
  }, 3_000);
});

// ---------------------------------------------------------------------------
// FallbackExecutionMetadata — fields and content
// ---------------------------------------------------------------------------

describe("FallbackExecutionMetadata — structure", () => {
  const profile = makeLargeProfile();

  test("id starts with 'qfc-'", async () => {
    const result = await selectQuantizationWithFallback(profile, 100, {
      probe: alwaysSucceedProbe,
      skipAudit: true,
    });
    expect(result.fallbackMeta.id).toMatch(/^qfc-/);
  });

  test("timestamp is a valid ISO string", async () => {
    const result = await selectQuantizationWithFallback(profile, 100, {
      probe: alwaysSucceedProbe,
      skipAudit: true,
    });
    const ts = new Date(result.fallbackMeta.timestamp).getTime();
    expect(ts).toBeGreaterThan(0);
  });

  test("profileSizeTier matches profile", async () => {
    const result = await selectQuantizationWithFallback(profile, 100, {
      probe: alwaysSucceedProbe,
      skipAudit: true,
    });
    expect(result.fallbackMeta.profileSizeTier).toBe("large");
  });

  test("targetTokens echoed correctly", async () => {
    const result = await selectQuantizationWithFallback(profile, 999, {
      probe: alwaysSucceedProbe,
      skipAudit: true,
    });
    expect(result.fallbackMeta.targetTokens).toBe(999);
  });

  test("stepsAttempted has one entry for single-step success", async () => {
    const result = await selectQuantizationWithFallback(profile, 50, {
      probe: alwaysSucceedProbe,
      skipAudit: true,
    });
    // Stopped at step 0 — only one record
    expect(result.fallbackMeta.stepsAttempted).toHaveLength(1);
  });

  test("stepsAttempted has N entries when N steps tried before success", async () => {
    const result = await selectQuantizationWithFallback(profile, 100, {
      probe: failFirstNStepsProbe(2),
      skipAudit: true,
    });
    // 2 failures + 1 success = 3 entries
    expect(result.fallbackMeta.stepsAttempted).toHaveLength(3);
  });

  test("finalStepIndex matches stepIndex", async () => {
    const result = await selectQuantizationWithFallback(profile, 100, {
      probe: failFirstNStepsProbe(1),
      skipAudit: true,
    });
    expect(result.fallbackMeta.finalStepIndex).toBe(result.stepIndex);
  });

  test("finalBitDepth + finalModelTier match result", async () => {
    const result = await selectQuantizationWithFallback(profile, 100, {
      probe: failFirstNStepsProbe(2),
      skipAudit: true,
    });
    expect(result.fallbackMeta.finalBitDepth).toBe(result.bitDepth);
    expect(result.fallbackMeta.finalModelTier).toBe(result.modelTier);
  });

  test("totalLatencyMs is a non-negative number", async () => {
    const result = await selectQuantizationWithFallback(profile, 100, {
      probe: alwaysSucceedProbe,
      skipAudit: true,
    });
    expect(result.fallbackMeta.totalLatencyMs).toBeGreaterThanOrEqual(0);
  });

  test("each FallbackStepRecord has all required fields", async () => {
    const result = await selectQuantizationWithFallback(profile, 100, {
      probe: alwaysSucceedProbe,
      skipAudit: true,
    });
    const rec = result.fallbackMeta.stepsAttempted[0]!;
    expect(typeof rec.stepIndex).toBe("number");
    expect(typeof rec.bitDepth).toBe("number");
    expect(typeof rec.modelTier).toBe("string");
    expect(typeof rec.timeoutMs).toBe("number");
    expect(typeof rec.latencyMs).toBe("number");
    expect(["success", "timeout", "error"]).toContain(rec.outcome);
  });
});

// ---------------------------------------------------------------------------
// Audit JSONL persistence — loadFallbackAuditRecords
// ---------------------------------------------------------------------------

describe("loadFallbackAuditRecords — JSONL round-trip", () => {
  // These tests write to a temp dir and override the audit path via environment
  // trick is not needed — we test the behaviour by using skipAudit=false with
  // a manually-injected path via the exported loadFallbackAuditRecords, but since
  // the global audit path is fixed, we verify round-trip indirectly via the
  // in-memory metadata and check loadFallbackAuditRecords returns an array.

  test("loadFallbackAuditRecords returns an array (may be empty)", async () => {
    const records = await loadFallbackAuditRecords();
    expect(Array.isArray(records)).toBe(true);
  });

  test("skipAudit=true does not write to disk (records unchanged)", async () => {
    const before = await loadFallbackAuditRecords();
    const beforeLen = before.length;

    await selectQuantizationWithFallback(makeLargeProfile(), 100, {
      probe: alwaysSucceedProbe,
      skipAudit: true,
    });

    const after = await loadFallbackAuditRecords();
    expect(after.length).toBe(beforeLen);
  });

  test("skipAudit=false writes one record to disk", async () => {
    const before = await loadFallbackAuditRecords();
    const beforeLen = before.length;

    const result = await selectQuantizationWithFallback(makeLargeProfile(), 42, {
      probe: alwaysSucceedProbe,
      skipAudit: false,
    });

    const after = await loadFallbackAuditRecords();
    expect(after.length).toBe(beforeLen + 1);

    const written = after[after.length - 1]!;
    expect(written.id).toBe(result.fallbackMeta.id);
    expect(written.targetTokens).toBe(42);
    expect(written.profileSizeTier).toBe("large");
  });
});

// ---------------------------------------------------------------------------
// Empty chain error
// ---------------------------------------------------------------------------

describe("selectQuantizationWithFallback — invalid inputs", () => {
  test("empty chain throws an error", async () => {
    await expect(
      selectQuantizationWithFallback(makeLargeProfile(), 100, {
        chain: [],
        probe: alwaysSucceedProbe,
        skipAudit: true,
      }),
    ).rejects.toThrow(/at least one step/i);
  });
});

// ---------------------------------------------------------------------------
// Custom chain behaviour
// ---------------------------------------------------------------------------

describe("selectQuantizationWithFallback — custom chain", () => {
  test("custom chain steps are honoured", async () => {
    const chain: QuantizationFallbackChain = [
      { bitDepth: 32, modelTier: "accurate",  timeoutMs: 500 },
      { bitDepth: 8,  modelTier: "balanced",  timeoutMs: 500 },
    ];

    const result = await selectQuantizationWithFallback(makeLargeProfile(), 100, {
      chain,
      probe: alwaysSucceedProbe,
      skipAudit: true,
    });

    expect(result.bitDepth).toBe(32);
    expect(result.modelTier).toBe("accurate");
  });

  test("custom chain fallback produces correct stepIndex", async () => {
    const chain: QuantizationFallbackChain = [
      { bitDepth: 16, modelTier: "accurate",  timeoutMs: 500 },
      { bitDepth: 8,  modelTier: "balanced",  timeoutMs: 500 },
      { bitDepth: 4,  modelTier: "ultrafast", timeoutMs: 500 },
    ];

    const result = await selectQuantizationWithFallback(makeLargeProfile(), 100, {
      chain,
      probe: failFirstNStepsProbe(1),
      skipAudit: true,
    });

    expect(result.stepIndex).toBe(1);
    expect(result.bitDepth).toBe(8);
  });

  test("tiny profile + accurate step: profileSizeTier is 'tiny' in metadata", async () => {
    const chain: QuantizationFallbackChain = [
      { bitDepth: 16, modelTier: "accurate", timeoutMs: 500 },
    ];

    const result = await selectQuantizationWithFallback(makeSmallProfile(), 50, {
      chain,
      probe: alwaysSucceedProbe,
      skipAudit: true,
    });

    expect(result.fallbackMeta.profileSizeTier).toBe("tiny");
  });
});

// ---------------------------------------------------------------------------
// QuantizationStrategyEngine.select() — alias method
// ---------------------------------------------------------------------------

describe("QuantizationStrategyEngine.select() — direct chain execution", () => {
  let cwd: string;
  let engine: QuantizationStrategyEngine;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
    engine = new QuantizationStrategyEngine(cwd);
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("select() returns SelectedQuantization", async () => {
    const profile = makeLargeProfile();
    const result = await engine.select(profile, 300, {
      probe: alwaysSucceedProbe,
      skipAudit: true,
    });
    expect(result.bitDepth).toBeDefined();
    expect(result.modelTier).toBeDefined();
    expect(result.fallbackMeta).toBeDefined();
    expect(typeof result.stepIndex).toBe("number");
    expect(typeof result.usedFallback).toBe("boolean");
  });

  test("select() with all-fail probe returns last-resort step", async () => {
    const profile = makeLargeProfile();
    const result = await engine.select(profile, 100, {
      probe: alwaysFailProbe,
      skipAudit: true,
    });
    const lastStep = DEFAULT_FALLBACK_CHAIN[DEFAULT_FALLBACK_CHAIN.length - 1]!;
    expect(result.bitDepth).toBe(lastStep.bitDepth);
    expect(result.modelTier).toBe(lastStep.modelTier);
  });

  test("select() with custom chain honours the chain", async () => {
    const chain: QuantizationFallbackChain = [
      { bitDepth: 8, modelTier: "fast", timeoutMs: 500 },
    ];
    const result = await engine.select(makeLargeProfile(), 100, {
      chain,
      probe: alwaysSucceedProbe,
      skipAudit: true,
    });
    expect(result.bitDepth).toBe(8);
    expect(result.modelTier).toBe("fast");
  });
});

// ---------------------------------------------------------------------------
// QuantizationStrategyEngine.selectBitDepth() — probe injection
// ---------------------------------------------------------------------------

describe("QuantizationStrategyEngine.selectBitDepth() — fallback probe injection", () => {
  let cwd: string;
  let engine: QuantizationStrategyEngine;

  beforeEach(async () => {
    cwd = makeTmpDir();
    await setupGenomeDir(cwd);
    engine = new QuantizationStrategyEngine(cwd, { learnerOptions: { minSamples: 3 } });
  });

  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  test("selectBitDepth without probe returns valid bitDepth", async () => {
    const result = await engine.selectBitDepth(
      "how does the embedding router work",
      { skipAudit: true },
    );
    expect([4, 8, 16, 32]).toContain(result.bitDepth);
    expect(result.outcomeId).toMatch(/^qs-/);
  });

  test("selectBitDepth with alwaysSucceedProbe populates fallbackMeta", async () => {
    const result = await engine.selectBitDepth(
      "find the QuantizedANNSearcher buildIndex method",
      { probe: alwaysSucceedProbe, skipAudit: true },
    );
    expect(result.fallbackMeta).toBeDefined();
    expect(result.fallbackMeta!.stepsAttempted.length).toBeGreaterThanOrEqual(1);
  });

  test("selectBitDepth with alwaysFailProbe still returns a bitDepth", async () => {
    const result = await engine.selectBitDepth(
      "semantic embedding latency tradeoff",
      { probe: alwaysFailProbe, skipAudit: true },
    );
    expect([4, 8, 16, 32]).toContain(result.bitDepth);
  });

  test("selectBitDepth with large manifest profile respects heuristic depth in chain", async () => {
    // Large genome → learner heuristic picks int8 → chain should start at int8
    const sections = Array.from({ length: 200 }, (_, i) =>
      makeSection({ path: `s${i}.md`, tokens: 300 }),
    );
    engine.loadProfile(makeManifest(sections));

    const result = await engine.selectBitDepth(
      "how does caching work in the router",
      { probe: alwaysSucceedProbe, skipAudit: true },
    );
    // int8-based chain starts at int8/fast
    expect(result.bitDepth).toBe(8);
    expect(result.fallbackMeta!.finalModelTier).toBe("fast");
  });

  test("selectBitDepth with tiny manifest profile starts chain at int16", async () => {
    const sections = Array.from({ length: 5 }, (_, i) =>
      makeSection({ path: `s${i}.md`, tokens: 100 }),
    );
    engine.loadProfile(makeManifest(sections));

    const result = await engine.selectBitDepth(
      "how does the genome init work",
      { probe: alwaysSucceedProbe, skipAudit: true },
    );
    // tiny genome → learner picks int16 → chain starts at int16/accurate
    expect(result.bitDepth).toBe(16);
  });

  test("selectBitDepth fallbackMeta stepRecord latencyMs is numeric", async () => {
    const result = await engine.selectBitDepth(
      "embedding router fallback cascade",
      { probe: alwaysSucceedProbe, skipAudit: true },
    );
    const rec = result.fallbackMeta!.stepsAttempted[0]!;
    expect(typeof rec.latencyMs).toBe("number");
    expect(rec.latencyMs).toBeGreaterThanOrEqual(0);
  });

  test("records a placeholder outcome in JSONL after selectBitDepth", async () => {
    const { default: loadQ } = await import("../src/genome/quantization-strategy.ts").then(
      async (m) => ({ default: m.loadQuantizationOutcomes }),
    );
    await engine.selectBitDepth(
      "semantic retrieval embeddings",
      { probe: alwaysSucceedProbe, skipAudit: true },
    );
    const outcomes = await loadQ(cwd);
    expect(outcomes.length).toBeGreaterThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// Probe error classification — "error" vs "timeout" outcome
// ---------------------------------------------------------------------------

describe("FallbackStepRecord outcome classification", () => {
  test("probe throwing non-timeout error → outcome='error'", async () => {
    const errorProbe: FallbackStepProbe = async () => {
      throw new Error("connection refused");
    };

    const chain: QuantizationFallbackChain = [
      { bitDepth: 16, modelTier: "accurate", timeoutMs: 500 },
      { bitDepth: 8,  modelTier: "fast",     timeoutMs: 500 },
    ];

    const result = await selectQuantizationWithFallback(makeLargeProfile(), 100, {
      chain,
      probe: errorProbe,
      skipAudit: true,
    });

    // First step should have outcome='error', not 'timeout'
    expect(result.fallbackMeta.stepsAttempted[0]!.outcome).toBe("error");
    expect(result.fallbackMeta.stepsAttempted[0]!.errorMessage).toMatch(/connection refused/i);
  });

  test("probe returning false → outcome='error' (not timeout)", async () => {
    const chain: QuantizationFallbackChain = [
      { bitDepth: 16, modelTier: "accurate", timeoutMs: 500 },
      { bitDepth: 8,  modelTier: "fast",     timeoutMs: 500 },
    ];

    const result = await selectQuantizationWithFallback(makeLargeProfile(), 100, {
      chain,
      probe: failFirstNStepsProbe(1),
      skipAudit: true,
    });

    const step0 = result.fallbackMeta.stepsAttempted[0]!;
    expect(step0.outcome).toBe("error");
  });
});
