/**
 * Tests for CacheBreakpointOptimizer and related helpers.
 *
 * Covers:
 *  - estimateTokenCount heuristic (via recommendBreakpoints output)
 *  - heuristic fallback when no session data exists
 *  - session-log ingestion and prefix-repetition analysis
 *  - ROI computation: cost_saved vs write_overhead
 *  - recommendBreakpoints: returns correct indices and ROI fields
 *  - audit trail: evolution JSONL is appended after each call
 *  - cacheBreakpoints() integration with optimizerRecommendation option
 *  - edge cases: empty messages, single message, maxBreakpoints cap
 */

import { describe, expect, test, beforeEach, afterEach } from "bun:test";
import { mkdtemp, rm, mkdir, writeFile } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";

import {
  CacheBreakpointOptimizer,
  recommendBreakpoints,
  MAX_BREAKPOINTS,
  type Message,
  type OptimizerOptions,
  type BreakpointRecommendation,
  type BreakpointEvolutionRecord,
} from "../src/anthropic/cache-breakpoint-optimizer.ts";

import {
  cacheBreakpoints,
  type CacheableMessage,
  type CacheableRequest,
} from "../src/anthropic/prompt-cache.ts";

import { readJsonl } from "../src/genome/jsonl.ts";

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

async function makeTmpDir(): Promise<string> {
  return mkdtemp(join(tmpdir(), "bp-opt-test-"));
}

async function cleanup(dir: string) {
  await rm(dir, { recursive: true, force: true });
}

/** Build a minimal Message. */
function msg(
  role: Message["role"],
  content: string,
  tokenCount?: number,
): Message {
  return { role, content, ...(tokenCount !== undefined ? { tokenCount } : {}) };
}

/** Build a session JSONL line with cache_read_input_tokens set. */
function sessionHit(tokens: number): string {
  return JSON.stringify({
    role: "user",
    content: "x".repeat(tokens * 4),
    cache_read_input_tokens: tokens,
    input_tokens: tokens,
  });
}

/** Build a session JSONL line with no cache hit. */
function sessionMiss(tokens: number): string {
  return JSON.stringify({
    role: "user",
    content: "x".repeat(tokens * 4),
    cache_read_input_tokens: 0,
    input_tokens: tokens,
  });
}

/**
 * Write a fake session JSONL file into `dir/sessions/<name>.jsonl`.
 * `lines` is an array of strings, each a valid JSON object per line.
 */
async function writeSessionFile(
  dir: string,
  name: string,
  lines: string[],
): Promise<void> {
  const sessionDir = join(dir, "sessions");
  await mkdir(sessionDir, { recursive: true });
  await writeFile(join(sessionDir, name), lines.join("\n") + "\n", "utf-8");
}

// ---------------------------------------------------------------------------
// Helpers for building test options
// ---------------------------------------------------------------------------

function makeOptions(
  tmpDir: string,
  extra: Partial<OptimizerOptions> = {},
): OptimizerOptions {
  return {
    sessionLogDir: join(tmpDir, "sessions"),
    auditPath:     join(tmpDir, "evolution.jsonl"),
    ...extra,
  };
}

// ---------------------------------------------------------------------------
// MAX_BREAKPOINTS constant
// ---------------------------------------------------------------------------

describe("MAX_BREAKPOINTS", () => {
  test("equals 4 (Anthropic API limit)", () => {
    expect(MAX_BREAKPOINTS).toBe(4);
  });
});

// ---------------------------------------------------------------------------
// Empty / trivial inputs
// ---------------------------------------------------------------------------

describe("recommendBreakpoints — empty inputs", () => {
  let tmpDir: string;
  beforeEach(async () => { tmpDir = await makeTmpDir(); });
  afterEach(async () => { await cleanup(tmpDir); });

  test("returns empty array for empty message list", async () => {
    const result = await recommendBreakpoints([], makeOptions(tmpDir));
    expect(result).toEqual([]);
  });

  test("returns empty array for single message", async () => {
    const result = await recommendBreakpoints(
      [msg("user", "hello")],
      makeOptions(tmpDir),
    );
    // A single message has no valid split point (need at least 2 for a prefix)
    expect(result).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Heuristic fallback (no session data)
// ---------------------------------------------------------------------------

describe("recommendBreakpoints — heuristic fallback", () => {
  let tmpDir: string;
  beforeEach(async () => { tmpDir = await makeTmpDir(); });
  afterEach(async () => { await cleanup(tmpDir); });

  test("returns recommendation for 2-message input (no sessions)", async () => {
    const messages: Message[] = [
      msg("system", "You are a helpful assistant."),
      msg("user",   "What is 2+2?"),
    ];
    const [rec] = await recommendBreakpoints(messages, makeOptions(tmpDir));
    expect(rec).toBeDefined();
    expect(rec!.learnedFromSessions).toBe(false);
    expect(rec!.breakpointIndices.length).toBeGreaterThan(0);
    expect(rec!.breakpointIndices[0]).toBe(0); // after system message
  });

  test("heuristic adds midpoint breakpoint for ≥8 messages", async () => {
    const messages: Message[] = Array.from({ length: 10 }, (_, i) =>
      msg("user", `Turn ${i}: some content here for testing purposes.`),
    );
    const [rec] = await recommendBreakpoints(
      messages,
      makeOptions(tmpDir, { maxBreakpoints: 3 }),
    );
    expect(rec).toBeDefined();
    expect(rec!.learnedFromSessions).toBe(false);
    // Should include index 0 and a midpoint
    expect(rec!.breakpointIndices).toContain(0);
    expect(rec!.breakpointIndices.length).toBeGreaterThanOrEqual(2);
  });

  test("heuristic respects maxBreakpoints cap", async () => {
    const messages: Message[] = Array.from({ length: 10 }, (_, i) =>
      msg("user", `Turn ${i}`),
    );
    const [rec] = await recommendBreakpoints(
      messages,
      makeOptions(tmpDir, { maxBreakpoints: 1 }),
    );
    expect(rec).toBeDefined();
    expect(rec!.breakpointIndices.length).toBeLessThanOrEqual(1);
  });

  test("heuristic recommendation has ROI array matching breakpointIndices length", async () => {
    const messages: Message[] = Array.from({ length: 6 }, (_, i) =>
      msg("user", `Turn ${i}: ` + "x".repeat(200)),
    );
    const [rec] = await recommendBreakpoints(messages, makeOptions(tmpDir));
    expect(rec).toBeDefined();
    expect(rec!.roi.length).toBe(rec!.breakpointIndices.length);
  });
});

// ---------------------------------------------------------------------------
// Session-log ingestion and learned breakpoints
// ---------------------------------------------------------------------------

describe("recommendBreakpoints — session-log learning", () => {
  let tmpDir: string;
  beforeEach(async () => { tmpDir = await makeTmpDir(); });
  afterEach(async () => { await cleanup(tmpDir); });

  test("learns from session hits and marks learnedFromSessions=true", async () => {
    // Write 6 sessions each with a hit at position 0 (minRepeatTurns default=5)
    for (let i = 0; i < 6; i++) {
      await writeSessionFile(tmpDir, `s${i}.jsonl`, [
        sessionHit(500),
        sessionMiss(200),
        sessionMiss(200),
      ]);
    }

    const messages: Message[] = [
      msg("system", "x".repeat(2000), 500),
      msg("user", "question 1"),
      msg("user", "question 2"),
    ];

    const [rec] = await recommendBreakpoints(
      messages,
      makeOptions(tmpDir, { minRepeatTurns: 5 }),
    );

    expect(rec).toBeDefined();
    expect(rec!.learnedFromSessions).toBe(true);
    expect(rec!.breakpointIndices).toContain(0);
  });

  test("does not recommend breakpoint when repeat count < minRepeatTurns", async () => {
    // Only 3 sessions — below default minRepeatTurns of 5
    for (let i = 0; i < 3; i++) {
      await writeSessionFile(tmpDir, `s${i}.jsonl`, [
        sessionHit(500),
        sessionMiss(200),
      ]);
    }

    const messages: Message[] = [
      msg("system", "x".repeat(2000), 500),
      msg("user", "question"),
    ];

    const [rec] = await recommendBreakpoints(
      messages,
      makeOptions(tmpDir, { minRepeatTurns: 5 }),
    );

    // With only 3 sessions below threshold, should fall back to heuristic
    if (rec) {
      expect(rec.learnedFromSessions).toBe(false);
    }
  });

  test("breakpointIndices are in ascending order", async () => {
    for (let i = 0; i < 6; i++) {
      await writeSessionFile(tmpDir, `s${i}.jsonl`, [
        sessionHit(500),
        sessionHit(300),
        sessionMiss(200),
        sessionMiss(200),
      ]);
    }

    const messages: Message[] = Array.from({ length: 6 }, (_, j) =>
      msg("user", "x".repeat(200), 50),
    );

    const [rec] = await recommendBreakpoints(
      messages,
      makeOptions(tmpDir, { minRepeatTurns: 5, maxBreakpoints: 3 }),
    );

    if (rec && rec.breakpointIndices.length > 1) {
      for (let i = 1; i < rec.breakpointIndices.length; i++) {
        expect(rec.breakpointIndices[i]!).toBeGreaterThan(rec.breakpointIndices[i - 1]!);
      }
    }
  });

  test("never recommends more than maxBreakpoints", async () => {
    for (let i = 0; i < 8; i++) {
      await writeSessionFile(tmpDir, `s${i}.jsonl`, [
        sessionHit(500), sessionHit(400), sessionHit(300), sessionHit(200),
        sessionMiss(100),
      ]);
    }

    const messages: Message[] = Array.from({ length: 8 }, (_, j) =>
      msg("user", "x".repeat(200), 50),
    );

    const [rec] = await recommendBreakpoints(
      messages,
      makeOptions(tmpDir, { minRepeatTurns: 5, maxBreakpoints: 2 }),
    );

    if (rec) {
      expect(rec.breakpointIndices.length).toBeLessThanOrEqual(2);
    }
  });

  test("never recommends more than MAX_BREAKPOINTS even if maxBreakpoints > 4", async () => {
    for (let i = 0; i < 8; i++) {
      await writeSessionFile(tmpDir, `s${i}.jsonl`, [
        sessionHit(500), sessionHit(400), sessionHit(300), sessionHit(200), sessionHit(100),
      ]);
    }

    const messages: Message[] = Array.from({ length: 8 }, (_, j) =>
      msg("user", "x".repeat(200), 50),
    );

    const [rec] = await recommendBreakpoints(
      messages,
      makeOptions(tmpDir, { minRepeatTurns: 5, maxBreakpoints: 10 }),
    );

    if (rec) {
      expect(rec.breakpointIndices.length).toBeLessThanOrEqual(MAX_BREAKPOINTS);
    }
  });
});

// ---------------------------------------------------------------------------
// ROI fields
// ---------------------------------------------------------------------------

describe("BreakpointROI fields", () => {
  let tmpDir: string;
  beforeEach(async () => { tmpDir = await makeTmpDir(); });
  afterEach(async () => { await cleanup(tmpDir); });

  test("ROI entry has required numeric fields", async () => {
    const messages: Message[] = [
      msg("system", "x".repeat(4000), 1000),
      msg("user", "question here"),
    ];

    const [rec] = await recommendBreakpoints(messages, makeOptions(tmpDir));
    expect(rec).toBeDefined();
    for (const roi of rec!.roi) {
      expect(typeof roi.messageIndex).toBe("number");
      expect(typeof roi.prefixTokens).toBe("number");
      expect(typeof roi.estimatedHitRate).toBe("number");
      expect(typeof roi.estimatedTokenSavings).toBe("number");
      expect(typeof roi.writeCostTokens).toBe("number");
      expect(typeof roi.netTokenSavings).toBe("number");
      expect(typeof roi.netRoiMicroUsd).toBe("number");
      expect(roi.prefixTokens).toBeGreaterThan(0);
      expect(roi.estimatedHitRate).toBeGreaterThanOrEqual(0);
      expect(roi.estimatedHitRate).toBeLessThanOrEqual(1);
    }
  });

  test("totalNetRoiMicroUsd equals sum of individual roi entries", async () => {
    const messages: Message[] = Array.from({ length: 8 }, (_, i) =>
      msg("user", "x".repeat(400), 100),
    );

    const [rec] = await recommendBreakpoints(
      messages,
      makeOptions(tmpDir, { maxBreakpoints: 3 }),
    );

    if (rec && rec.roi.length > 0) {
      const sum = rec.roi.reduce((s, r) => s + r.netRoiMicroUsd, 0);
      expect(rec.totalNetRoiMicroUsd).toBe(sum);
    }
  });

  test("ROI messageIndex values match breakpointIndices", async () => {
    const messages: Message[] = Array.from({ length: 6 }, (_, i) =>
      msg("user", "x".repeat(400), 100),
    );

    const [rec] = await recommendBreakpoints(messages, makeOptions(tmpDir));

    if (rec) {
      const roiIndices = rec.roi.map((r) => r.messageIndex).sort((a, b) => a - b);
      const bpIndices  = [...rec.breakpointIndices].sort((a, b) => a - b);
      expect(roiIndices).toEqual(bpIndices);
    }
  });
});

// ---------------------------------------------------------------------------
// Audit trail (evolution JSONL)
// ---------------------------------------------------------------------------

describe("evolution audit trail", () => {
  let tmpDir: string;
  beforeEach(async () => { tmpDir = await makeTmpDir(); });
  afterEach(async () => { await cleanup(tmpDir); });

  test("appends a record after recommendBreakpoints is called", async () => {
    const messages: Message[] = [
      msg("system", "x".repeat(2000), 500),
      msg("user", "question"),
    ];
    const opts = makeOptions(tmpDir);

    await recommendBreakpoints(messages, opts);

    const records = await readJsonl<BreakpointEvolutionRecord>(opts.auditPath!);
    expect(records.length).toBe(1);
  });

  test("appends one record per call", async () => {
    const messages: Message[] = [
      msg("system", "x".repeat(2000), 500),
      msg("user", "question"),
    ];
    const opts = makeOptions(tmpDir);

    await recommendBreakpoints(messages, opts);
    await recommendBreakpoints(messages, opts);
    await recommendBreakpoints(messages, opts);

    const records = await readJsonl<BreakpointEvolutionRecord>(opts.auditPath!);
    expect(records.length).toBe(3);
  });

  test("audit record has required fields", async () => {
    const messages: Message[] = [
      msg("system", "x".repeat(2000), 500),
      msg("user", "question"),
    ];
    const opts = makeOptions(tmpDir);

    await recommendBreakpoints(messages, opts);

    const records = await readJsonl<BreakpointEvolutionRecord>(opts.auditPath!);
    const rec = records[0]!;

    expect(typeof rec.id).toBe("string");
    expect(rec.id.startsWith("bpe-")).toBe(true);
    expect(typeof rec.timestamp).toBe("string");
    expect(Array.isArray(rec.breakpointIndices)).toBe(true);
    expect(typeof rec.totalNetRoiMicroUsd).toBe("number");
    expect(typeof rec.sessionFilesAnalyzed).toBe("number");
    expect(typeof rec.learnedFromSessions).toBe("boolean");
    expect(Array.isArray(rec.roiSnapshot)).toBe(true);
  });

  test("loadEvolutionAudit returns all persisted records", async () => {
    const messages: Message[] = [
      msg("system", "system prompt content"),
      msg("user", "user message"),
    ];
    const opts = makeOptions(tmpDir);
    const optimizer = new CacheBreakpointOptimizer({
      sessionLogDir: opts.sessionLogDir,
      auditPath:     opts.auditPath,
    });

    await optimizer.recommendBreakpoints(messages, opts);
    await optimizer.recommendBreakpoints(messages, opts);

    const all = await optimizer.loadEvolutionAudit();
    expect(all.length).toBe(2);
  });

  test("audit records have unique IDs", async () => {
    const messages: Message[] = [
      msg("system", "x".repeat(2000), 500),
      msg("user", "question"),
    ];
    const opts = makeOptions(tmpDir);

    await recommendBreakpoints(messages, opts);
    await recommendBreakpoints(messages, opts);

    const records = await readJsonl<BreakpointEvolutionRecord>(opts.auditPath!);
    const ids = records.map((r) => r.id);
    const uniqueIds = new Set(ids);
    expect(uniqueIds.size).toBe(ids.length);
  });
});

// ---------------------------------------------------------------------------
// cacheBreakpoints() integration with optimizerRecommendation
// ---------------------------------------------------------------------------

describe("cacheBreakpoints() — optimizerRecommendation integration", () => {
  test("applies cache markers at recommended indices", () => {
    const messages: CacheableMessage[] = [
      { role: "user", content: "static context about the project" },
      { role: "user", content: "more static info" },
      { role: "user", content: "dynamic question" },
    ];

    const req: CacheableRequest = { messages };
    const result = cacheBreakpoints(req, {
      optimizerRecommendation: { breakpointIndices: [0, 1] },
    });

    // Messages at index 0 and 1 should have cache_control on their content
    const m0 = result.messages[0]!;
    const m1 = result.messages[1]!;
    const m2 = result.messages[2]!;

    // Check that content blocks at indices 0 and 1 have cache_control
    const getLastBlockCacheControl = (m: CacheableMessage) => {
      if (Array.isArray(m.content)) {
        const last = m.content[m.content.length - 1];
        return last?.cache_control;
      }
      return undefined;
    };

    expect(getLastBlockCacheControl(m0 as CacheableMessage)).toEqual({ type: "ephemeral" });
    expect(getLastBlockCacheControl(m1 as CacheableMessage)).toEqual({ type: "ephemeral" });
    // Message 2 (dynamic) should NOT have cache_control from the optimizer
    const m2Cc = getLastBlockCacheControl(m2 as CacheableMessage);
    expect(m2Cc).toBeUndefined();
  });

  test("no optimizer recommendation → behaves like original cacheBreakpoints", () => {
    const messages: CacheableMessage[] = [
      { role: "user", content: "static", cache: true },
      { role: "user", content: "dynamic" },
    ];
    const req: CacheableRequest = { messages };

    const withOpt   = cacheBreakpoints(req, {});
    const withoutOpt = cacheBreakpoints(req);

    expect(JSON.stringify(withOpt)).toBe(JSON.stringify(withoutOpt));
  });

  test("optimizer indices out of range are safely ignored", () => {
    const messages: CacheableMessage[] = [
      { role: "user", content: "message 0" },
    ];
    const req: CacheableRequest = { messages };

    // Index 99 is out of range — should not throw
    expect(() =>
      cacheBreakpoints(req, {
        optimizerRecommendation: { breakpointIndices: [99] },
      }),
    ).not.toThrow();
  });

  test("empty breakpointIndices → same as no option", () => {
    const messages: CacheableMessage[] = [
      { role: "user", content: "message" },
    ];
    const req: CacheableRequest = { messages };

    const withEmpty  = cacheBreakpoints(req, { optimizerRecommendation: { breakpointIndices: [] } });
    const withoutOpt = cacheBreakpoints(req);

    expect(JSON.stringify(withEmpty)).toBe(JSON.stringify(withoutOpt));
  });

  test("system and tools markers are still applied alongside optimizer markers", () => {
    const messages: CacheableMessage[] = [
      { role: "user", content: "static context" },
      { role: "user", content: "dynamic question" },
    ];
    const req: CacheableRequest = {
      system: "You are a helpful assistant.",
      tools: [{ name: "search", input_schema: {} }],
      messages,
    };

    const result = cacheBreakpoints(req, {
      optimizerRecommendation: { breakpointIndices: [0] },
    });

    // System should be promoted to blocks with cache_control
    expect(Array.isArray(result.system)).toBe(true);
    const sysBlocks = result.system as Array<{ cache_control?: unknown }>;
    expect(sysBlocks[sysBlocks.length - 1]?.cache_control).toEqual({ type: "ephemeral" });

    // Tools last entry should have cache_control
    const lastTool = result.tools![result.tools!.length - 1];
    expect(lastTool?.cache_control).toEqual({ type: "ephemeral" });
  });
});

// ---------------------------------------------------------------------------
// CacheBreakpointOptimizer class — direct API
// ---------------------------------------------------------------------------

describe("CacheBreakpointOptimizer class", () => {
  let tmpDir: string;
  beforeEach(async () => { tmpDir = await makeTmpDir(); });
  afterEach(async () => { await cleanup(tmpDir); });

  test("recommendBreakpoints returns array", async () => {
    const optimizer = new CacheBreakpointOptimizer({
      sessionLogDir: join(tmpDir, "sessions"),
      auditPath:     join(tmpDir, "evolution.jsonl"),
    });

    const messages: Message[] = [
      msg("system", "system prompt"),
      msg("user", "user message"),
    ];

    const result = await optimizer.recommendBreakpoints(messages);
    expect(Array.isArray(result)).toBe(true);
  });

  test("recommendation summary is a non-empty string", async () => {
    const optimizer = new CacheBreakpointOptimizer({
      sessionLogDir: join(tmpDir, "sessions"),
      auditPath:     join(tmpDir, "evolution.jsonl"),
    });

    const messages: Message[] = [
      msg("system", "system prompt"),
      msg("user", "user message"),
    ];

    const [rec] = await optimizer.recommendBreakpoints(messages);
    if (rec) {
      expect(typeof rec.summary).toBe("string");
      expect(rec.summary.length).toBeGreaterThan(0);
    }
  });

  test("generatedAt is a valid ISO-8601 timestamp", async () => {
    const optimizer = new CacheBreakpointOptimizer({
      sessionLogDir: join(tmpDir, "sessions"),
      auditPath:     join(tmpDir, "evolution.jsonl"),
    });

    const messages: Message[] = [
      msg("system", "system prompt"),
      msg("user", "user message"),
    ];

    const [rec] = await optimizer.recommendBreakpoints(messages);
    if (rec) {
      expect(() => new Date(rec.generatedAt)).not.toThrow();
      expect(isNaN(new Date(rec.generatedAt).getTime())).toBe(false);
    }
  });
});

// ---------------------------------------------------------------------------
// Token count estimation
// ---------------------------------------------------------------------------

describe("token count estimation", () => {
  let tmpDir: string;
  beforeEach(async () => { tmpDir = await makeTmpDir(); });
  afterEach(async () => { await cleanup(tmpDir); });

  test("uses provided tokenCount when available", async () => {
    // With an explicit tokenCount=1000 on the first message, prefixTokens
    // in the ROI entry at index 0 should be 1000.
    const messages: Message[] = [
      msg("system", "x", 1000),
      msg("user", "question"),
    ];

    const [rec] = await recommendBreakpoints(messages, makeOptions(tmpDir));
    if (rec && rec.roi.length > 0) {
      const firstRoi = rec.roi.find((r) => r.messageIndex === 0);
      if (firstRoi) {
        expect(firstRoi.prefixTokens).toBe(1000);
      }
    }
  });

  test("falls back to content-length heuristic when tokenCount is absent", async () => {
    // 400 chars / 4 = 100 tokens
    const messages: Message[] = [
      msg("system", "x".repeat(400)),
      msg("user", "question"),
    ];

    const [rec] = await recommendBreakpoints(messages, makeOptions(tmpDir));
    if (rec && rec.roi.length > 0) {
      const firstRoi = rec.roi.find((r) => r.messageIndex === 0);
      if (firstRoi) {
        expect(firstRoi.prefixTokens).toBe(100);
      }
    }
  });
});
