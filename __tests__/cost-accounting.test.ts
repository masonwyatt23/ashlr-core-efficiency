/**
 * Tests for src/session-log/cost-accounting.ts
 *
 * Covers:
 *   - recordCompressionCost() — cost math per tier, amnesia flagging
 *   - summarizeSessionROI()   — aggregation, toCSV()
 *   - getAmnesiaFlags() / hasAmnesiaFlags()
 *   - PROVIDER_RATES / defaultProviderRate() from tokens/index.ts
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  recordCompressionCost,
  summarizeSessionROI,
  getAmnesiaFlags,
  hasAmnesiaFlags,
  _getRecords,
  _resetRecords,
  type CompressionCostRecord,
  type SessionROI,
} from "../src/session-log/cost-accounting.ts";
import {
  PROVIDER_RATES,
  DEFAULT_PROVIDER_RATE,
  defaultProviderRate,
} from "../src/tokens/index.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

beforeEach(() => _resetRecords());
afterEach(() => _resetRecords());

// ---------------------------------------------------------------------------
// PROVIDER_RATES — pricing table sanity checks
// ---------------------------------------------------------------------------

describe("PROVIDER_RATES — pricing table", () => {
  test("claude-3-5-sonnet has correct rates", () => {
    const r = PROVIDER_RATES["claude-3-5-sonnet"];
    expect(r.inputRate).toBeCloseTo(3.0 / 1_000_000, 10);
    expect(r.outputRate).toBeCloseTo(15.0 / 1_000_000, 10);
  });

  test("claude-3-opus has correct rates", () => {
    const r = PROVIDER_RATES["claude-3-opus"];
    expect(r.inputRate).toBeCloseTo(15.0 / 1_000_000, 10);
    expect(r.outputRate).toBeCloseTo(75.0 / 1_000_000, 10);
  });

  test("gpt-4o has correct rates", () => {
    const r = PROVIDER_RATES["gpt-4o"];
    expect(r.inputRate).toBeCloseTo(5.0 / 1_000_000, 10);
    expect(r.outputRate).toBeCloseTo(15.0 / 1_000_000, 10);
  });

  test("all providers have positive rates", () => {
    for (const [name, rate] of Object.entries(PROVIDER_RATES)) {
      expect(rate.inputRate).toBeGreaterThan(0);
      expect(rate.outputRate).toBeGreaterThan(0);
    }
  });
});

// ---------------------------------------------------------------------------
// defaultProviderRate — fallback logic
// ---------------------------------------------------------------------------

describe("defaultProviderRate()", () => {
  test("returns known provider rate for exact match", () => {
    const r = defaultProviderRate("claude-3-haiku");
    expect(r.inputRate).toBeCloseTo(0.25 / 1_000_000, 10);
  });

  test("returns matching rate for versioned slug (prefix match)", () => {
    const r = defaultProviderRate("claude-3-5-sonnet-20241022");
    expect(r.inputRate).toBeCloseTo(PROVIDER_RATES["claude-3-5-sonnet"].inputRate, 10);
  });

  test("returns DEFAULT_PROVIDER_RATE for unknown provider", () => {
    const r = defaultProviderRate("unknown-llm-xyz");
    expect(r.inputRate).toBe(DEFAULT_PROVIDER_RATE.inputRate);
    expect(r.outputRate).toBe(DEFAULT_PROVIDER_RATE.outputRate);
  });

  test("DEFAULT_PROVIDER_RATE equals claude-3-5-sonnet", () => {
    expect(DEFAULT_PROVIDER_RATE.inputRate).toBe(PROVIDER_RATES["claude-3-5-sonnet"].inputRate);
  });
});

// ---------------------------------------------------------------------------
// recordCompressionCost — tier 1 (autoCompact)
// ---------------------------------------------------------------------------

describe("recordCompressionCost — tier 1 (autoCompact)", () => {
  test("returns a record with correct fields", () => {
    const rec = recordCompressionCost(1, 10_000, 250, true, "claude-3-5-sonnet");
    expect(rec.tier).toBe(1);
    expect(rec.tokensRemoved).toBe(10_000);
    expect(rec.timeMs).toBe(250);
    expect(rec.success).toBe(true);
    expect(rec.provider).toBe("claude-3-5-sonnet");
    expect(typeof rec.costSaved).toBe("number");
    expect(typeof rec.llmCallCost).toBe("number");
    expect(rec.llmCallCost).toBeGreaterThan(0);
    expect(typeof rec.recordedAt).toBe("string");
  });

  test("llmCallCost is positive (covers summary LLM call)", () => {
    const rec = recordCompressionCost(1, 50_000, 500, true, "claude-3-5-sonnet");
    expect(rec.llmCallCost).toBeGreaterThan(0);
  });

  test("costSaved = grossSavings - llmCallCost", () => {
    const tokensRemoved = 20_000;
    const rate = PROVIDER_RATES["claude-3-5-sonnet"];
    const summaryInputCost = tokensRemoved * rate.inputRate * 0.1;
    const summaryOutputCost = 200 * rate.outputRate;
    const expectedLlmCallCost = summaryInputCost + summaryOutputCost;
    const grossSavings = tokensRemoved * 0.1 * rate.inputRate;
    const expectedCostSaved = grossSavings - expectedLlmCallCost;

    const rec = recordCompressionCost(1, tokensRemoved, 100, true, "claude-3-5-sonnet");
    expect(rec.llmCallCost).toBeCloseTo(expectedLlmCallCost, 10);
    expect(rec.costSaved).toBeCloseTo(expectedCostSaved, 10);
  });

  test("no amnesia flag when estimationErrorPct is within ±10%", () => {
    const rec = recordCompressionCost(1, 10_000, 100, true, "claude-3-5-sonnet", 9.9);
    expect(rec.amnesiaFlag).toBe(false);
    const rec2 = recordCompressionCost(1, 10_000, 100, true, "claude-3-5-sonnet", -9.9);
    expect(rec2.amnesiaFlag).toBe(false);
  });

  test("amnesia flag set when estimationErrorPct > 10%", () => {
    const rec = recordCompressionCost(1, 10_000, 100, true, "claude-3-5-sonnet", 10.1);
    expect(rec.amnesiaFlag).toBe(true);
  });

  test("amnesia flag set when estimationErrorPct < -10%", () => {
    const rec = recordCompressionCost(1, 10_000, 100, true, "claude-3-5-sonnet", -10.1);
    expect(rec.amnesiaFlag).toBe(true);
  });

  test("estimationErrorPct is stored on tier-1 records", () => {
    const rec = recordCompressionCost(1, 5_000, 50, true, "claude-3-5-sonnet", 15.5);
    expect(rec.estimationErrorPct).toBeCloseTo(15.5);
  });
});

// ---------------------------------------------------------------------------
// recordCompressionCost — tier 2 (snipCompact)
// ---------------------------------------------------------------------------

describe("recordCompressionCost — tier 2 (snipCompact)", () => {
  test("llmCallCost is zero (no LLM call)", () => {
    const rec = recordCompressionCost(2, 30_000, 80, true, "claude-3-5-sonnet");
    expect(rec.llmCallCost).toBe(0);
  });

  test("costSaved = tokensRemoved × inputRate", () => {
    const tokensRemoved = 30_000;
    const rate = PROVIDER_RATES["claude-3-5-sonnet"];
    const expected = tokensRemoved * rate.inputRate;
    const rec = recordCompressionCost(2, tokensRemoved, 80, true, "claude-3-5-sonnet");
    expect(rec.costSaved).toBeCloseTo(expected, 10);
  });

  test("amnesiaFlag is always false for tier 2", () => {
    const rec = recordCompressionCost(2, 10_000, 50, true, "claude-3-5-sonnet", 99);
    expect(rec.amnesiaFlag).toBe(false);
  });

  test("estimationErrorPct is always 0 for tier 2", () => {
    const rec = recordCompressionCost(2, 10_000, 50, true, "claude-3-5-sonnet", 99);
    expect(rec.estimationErrorPct).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// recordCompressionCost — tier 3 (contextCollapse)
// ---------------------------------------------------------------------------

describe("recordCompressionCost — tier 3 (contextCollapse)", () => {
  test("llmCallCost is zero", () => {
    const rec = recordCompressionCost(3, 50_000, 5, true, "claude-3-haiku");
    expect(rec.llmCallCost).toBe(0);
  });

  test("costSaved = tokensRemoved × inputRate for haiku", () => {
    const tokensRemoved = 50_000;
    const rate = PROVIDER_RATES["claude-3-haiku"];
    const expected = tokensRemoved * rate.inputRate;
    const rec = recordCompressionCost(3, tokensRemoved, 5, true, "claude-3-haiku");
    expect(rec.costSaved).toBeCloseTo(expected, 10);
  });

  test("amnesiaFlag is always false for tier 3", () => {
    const rec = recordCompressionCost(3, 10_000, 3, false, "gpt-4o", 50);
    expect(rec.amnesiaFlag).toBe(false);
  });

  test("failed operations are still recorded", () => {
    const rec = recordCompressionCost(3, 10_000, 3, false, "gpt-4o");
    expect(rec.success).toBe(false);
    expect(_getRecords().length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// recordCompressionCost — multi-provider
// ---------------------------------------------------------------------------

describe("recordCompressionCost — provider pricing differences", () => {
  test("opus costs more per token than haiku", () => {
    const tokens = 10_000;
    const opus = recordCompressionCost(3, tokens, 10, true, "claude-3-opus");
    _resetRecords();
    const haiku = recordCompressionCost(3, tokens, 10, true, "claude-3-haiku");
    expect(opus.costSaved).toBeGreaterThan(haiku.costSaved);
  });

  test("unknown provider uses default (sonnet) rates", () => {
    const tokens = 10_000;
    const unknown = recordCompressionCost(3, tokens, 10, true, "unknown-model");
    _resetRecords();
    const sonnet = recordCompressionCost(3, tokens, 10, true, "claude-3-5-sonnet");
    expect(unknown.costSaved).toBeCloseTo(sonnet.costSaved, 10);
  });
});

// ---------------------------------------------------------------------------
// _getRecords — in-process store
// ---------------------------------------------------------------------------

describe("_getRecords()", () => {
  test("starts empty after reset", () => {
    expect(_getRecords().length).toBe(0);
  });

  test("accumulates records across calls", () => {
    recordCompressionCost(1, 1000, 100, true, "claude-3-5-sonnet");
    recordCompressionCost(2, 2000, 50, true, "claude-3-5-sonnet");
    recordCompressionCost(3, 3000, 10, true, "claude-3-5-sonnet");
    expect(_getRecords().length).toBe(3);
  });

  test("returned snapshot reflects all tiers", () => {
    recordCompressionCost(1, 500, 200, true, "gpt-4o");
    recordCompressionCost(2, 1000, 30, true, "gpt-4o");
    const recs = _getRecords();
    expect(recs[0]!.tier).toBe(1);
    expect(recs[1]!.tier).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// summarizeSessionROI()
// ---------------------------------------------------------------------------

describe("summarizeSessionROI()", () => {
  test("returns zeroed ROI when no records exist", () => {
    const roi = summarizeSessionROI();
    expect(roi.totalTokensSaved).toBe(0);
    expect(roi.totalCostSaved).toBe(0);
    expect(roi.compressionTimeMs).toBe(0);
    expect(roi.amnesiaCount).toBe(0);
    for (const t of [1, 2, 3] as const) {
      expect(roi.tierBreakdown[t].callCount).toBe(0);
      expect(roi.tierBreakdown[t].tokensRemoved).toBe(0);
      expect(roi.tierBreakdown[t].costSaved).toBe(0);
    }
  });

  test("aggregates tokens and cost correctly across tiers", () => {
    recordCompressionCost(2, 10_000, 50, true, "claude-3-5-sonnet");
    recordCompressionCost(3, 20_000, 10, true, "claude-3-5-sonnet");
    const roi = summarizeSessionROI();
    expect(roi.totalTokensSaved).toBe(30_000);
    expect(roi.tierBreakdown[2].callCount).toBe(1);
    expect(roi.tierBreakdown[3].callCount).toBe(1);
    expect(roi.tierBreakdown[2].tokensRemoved).toBe(10_000);
    expect(roi.tierBreakdown[3].tokensRemoved).toBe(20_000);
  });

  test("totalCostSaved is sum of individual costSaved values", () => {
    const r1 = recordCompressionCost(2, 10_000, 50, true, "claude-3-5-sonnet");
    const r2 = recordCompressionCost(3, 20_000, 10, true, "claude-3-5-sonnet");
    const roi = summarizeSessionROI();
    expect(roi.totalCostSaved).toBeCloseTo(r1.costSaved + r2.costSaved, 10);
  });

  test("compressionTimeMs sums all operation times", () => {
    recordCompressionCost(1, 5000, 300, true, "claude-3-5-sonnet");
    recordCompressionCost(2, 5000, 80, true, "claude-3-5-sonnet");
    recordCompressionCost(3, 5000, 15, true, "claude-3-5-sonnet");
    const roi = summarizeSessionROI();
    expect(roi.compressionTimeMs).toBe(395);
  });

  test("amnesiaCount counts flagged tier-1 records", () => {
    recordCompressionCost(1, 10_000, 200, true, "claude-3-5-sonnet", 15.0);   // amnesia
    recordCompressionCost(1, 10_000, 200, true, "claude-3-5-sonnet", 5.0);    // no amnesia
    recordCompressionCost(1, 10_000, 200, true, "claude-3-5-sonnet", -12.0);  // amnesia
    const roi = summarizeSessionROI();
    expect(roi.amnesiaCount).toBe(2);
  });

  test("generatedAt is a valid ISO timestamp", () => {
    const roi = summarizeSessionROI();
    expect(() => new Date(roi.generatedAt)).not.toThrow();
    expect(isNaN(new Date(roi.generatedAt).getTime())).toBe(false);
  });

  test("multiple calls on same tier accumulate callCount", () => {
    recordCompressionCost(2, 5000, 30, true, "claude-3-5-sonnet");
    recordCompressionCost(2, 5000, 30, true, "claude-3-5-sonnet");
    recordCompressionCost(2, 5000, 30, true, "claude-3-5-sonnet");
    const roi = summarizeSessionROI();
    expect(roi.tierBreakdown[2].callCount).toBe(3);
    expect(roi.tierBreakdown[2].tokensRemoved).toBe(15_000);
  });
});

// ---------------------------------------------------------------------------
// summarizeSessionROI().toCSV()
// ---------------------------------------------------------------------------

describe("summarizeSessionROI().toCSV()", () => {
  test("returns a CSV string with header row", () => {
    recordCompressionCost(2, 10_000, 50, true, "claude-3-5-sonnet");
    const csv = summarizeSessionROI().toCSV();
    expect(csv).toContain("recordedAt,tier,tokensRemoved,timeMs,success,provider,costSaved,llmCallCost,amnesiaFlag,estimationErrorPct");
  });

  test("CSV has N+1 lines for N records (header + data rows)", () => {
    recordCompressionCost(1, 5000, 100, true, "claude-3-5-sonnet", 5.0);
    recordCompressionCost(2, 8000, 40, true, "claude-3-5-sonnet");
    recordCompressionCost(3, 12000, 8, false, "claude-3-haiku");
    const csv = summarizeSessionROI().toCSV();
    const lines = csv.split("\n").filter(Boolean);
    expect(lines.length).toBe(4); // header + 3 data rows
  });

  test("CSV data row contains correct tier value", () => {
    recordCompressionCost(3, 20_000, 12, true, "claude-3-haiku");
    const csv = summarizeSessionROI().toCSV();
    const lines = csv.split("\n");
    const dataRow = lines[1]!;
    const parts = dataRow.split(",");
    // tier is the 2nd column (index 1)
    expect(parts[1]).toBe("3");
  });

  test("CSV is empty (header only) when no records", () => {
    const csv = summarizeSessionROI().toCSV();
    const lines = csv.split("\n").filter(Boolean);
    expect(lines.length).toBe(1); // just header
  });
});

// ---------------------------------------------------------------------------
// getAmnesiaFlags() / hasAmnesiaFlags()
// ---------------------------------------------------------------------------

describe("getAmnesiaFlags() / hasAmnesiaFlags()", () => {
  test("returns empty array when no records", () => {
    expect(getAmnesiaFlags().length).toBe(0);
  });

  test("hasAmnesiaFlags() is false when no records", () => {
    expect(hasAmnesiaFlags()).toBe(false);
  });

  test("returns only records with amnesiaFlag=true", () => {
    recordCompressionCost(1, 10_000, 100, true, "claude-3-5-sonnet", 5.0);   // no amnesia
    recordCompressionCost(1, 10_000, 100, true, "claude-3-5-sonnet", 15.0);  // amnesia
    recordCompressionCost(2, 10_000, 50, true, "claude-3-5-sonnet");         // tier 2, never amnesia
    const flags = getAmnesiaFlags();
    expect(flags.length).toBe(1);
    expect(flags[0]!.amnesiaFlag).toBe(true);
    expect(flags[0]!.estimationErrorPct).toBeCloseTo(15.0);
  });

  test("hasAmnesiaFlags() is true when any tier-1 record has amnesia", () => {
    recordCompressionCost(1, 10_000, 100, true, "claude-3-5-sonnet", 20.0);
    expect(hasAmnesiaFlags()).toBe(true);
  });

  test("tier 2/3 records never appear in amnesia flags", () => {
    recordCompressionCost(2, 5000, 20, true, "claude-3-5-sonnet", 99);
    recordCompressionCost(3, 5000, 5, true, "claude-3-5-sonnet", 99);
    expect(getAmnesiaFlags().length).toBe(0);
    expect(hasAmnesiaFlags()).toBe(false);
  });

  test("multiple amnesia records are all returned", () => {
    recordCompressionCost(1, 10_000, 100, true, "claude-3-5-sonnet", 11.0);
    recordCompressionCost(1, 10_000, 100, true, "claude-3-5-sonnet", -11.0);
    recordCompressionCost(1, 10_000, 100, true, "claude-3-5-sonnet", 25.0);
    expect(getAmnesiaFlags().length).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// End-to-end: mixed session simulation
// ---------------------------------------------------------------------------

describe("end-to-end: mixed session simulation", () => {
  test("realistic session produces positive totalCostSaved for tier 2/3", () => {
    // 10 tier-3 calls (cheap, fast, always positive savings)
    for (let i = 0; i < 10; i++) {
      recordCompressionCost(3, 5_000, 5, true, "claude-3-5-sonnet");
    }
    // 5 tier-2 calls
    for (let i = 0; i < 5; i++) {
      recordCompressionCost(2, 8_000, 30, true, "claude-3-5-sonnet");
    }
    const roi = summarizeSessionROI();
    expect(roi.totalCostSaved).toBeGreaterThan(0);
    expect(roi.tierBreakdown[3].callCount).toBe(10);
    expect(roi.tierBreakdown[2].callCount).toBe(5);
    expect(roi.tierBreakdown[1].callCount).toBe(0);
  });

  test("ROI JSON is serialisable (no circular refs)", () => {
    recordCompressionCost(2, 10_000, 50, true, "claude-3-5-sonnet");
    const roi = summarizeSessionROI();
    // Strip the toCSV function before serialising (it's a method, not data)
    const { toCSV, ...roiData } = roi;
    expect(() => JSON.stringify(roiData)).not.toThrow();
  });

  test("tier 1 with many amnesia flags increments amnesiaCount correctly", () => {
    for (let i = 0; i < 5; i++) {
      recordCompressionCost(1, 10_000, 300, true, "claude-3-5-sonnet", 20.0);  // amnesia
    }
    for (let i = 0; i < 3; i++) {
      recordCompressionCost(1, 10_000, 300, true, "claude-3-5-sonnet", 2.0);   // clean
    }
    const roi = summarizeSessionROI();
    expect(roi.amnesiaCount).toBe(5);
    expect(roi.tierBreakdown[1].callCount).toBe(8);
  });
});
