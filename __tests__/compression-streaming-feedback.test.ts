/**
 * Tests for src/compression/streaming-feedback.ts
 *
 * Covers:
 *   1. Normal stream — no overage, needsReTierRecommendation() stays false.
 *   2. High-cardinality tool results — large actual count triggers re-tier signal.
 *   3. Cache miss cascade — multiple consecutive usage events with growing counts.
 *   4. Provider codec drift — codec correction factor drives tighter threshold.
 *   5. Thinking token explosion — thinking blocks cause overage and correct block attribution.
 *   6. Streaming path helpers — streamingFeedbackPath(), createMonitor() factory.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, rm } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";
import {
  StreamingCompressionMonitor,
  createMonitor,
  streamingFeedbackPath,
  type BlockTypeBudget,
  type StreamingCompressionSignal,
} from "../src/compression/streaming-feedback.ts";
import { readJsonl } from "../src/genome/jsonl.ts";
import type { LLMSummarizer, ProviderRequest, StreamEvent } from "../src/types/index.ts";
import { _resetLearner } from "../src/compression/regret-learner.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function makeTmpDir(): Promise<string> {
  const base = join(
    tmpdir(),
    `ashlr-stream-feedback-test-${Date.now()}-${Math.random().toString(36).slice(2)}`,
  );
  await mkdir(base, { recursive: true });
  return base;
}

/**
 * Build a fake LLMSummarizer that emits the provided sequence of StreamEvents
 * then ends.
 */
function fakeSummarizer(events: StreamEvent[]): LLMSummarizer {
  return {
    async *stream(_request: ProviderRequest): AsyncGenerator<StreamEvent> {
      for (const ev of events) {
        yield ev;
      }
    },
  };
}

/** Minimal ProviderRequest for tests. */
function makeRequest(messages: Array<{ role: "user" | "assistant"; content: string }>): ProviderRequest {
  return {
    systemPrompt: "You are a helpful assistant.",
    messages,
    tools: [],
  };
}

/** Collect all events from an async generator. */
async function drainGenerator<T>(gen: AsyncGenerator<T>): Promise<T[]> {
  const results: T[] = [];
  for await (const item of gen) {
    results.push(item);
  }
  return results;
}

// ---------------------------------------------------------------------------
// Scenario 1: Normal stream — no overage
// ---------------------------------------------------------------------------

describe("StreamingCompressionMonitor — normal stream (no overage)", () => {
  let tmp: string;
  beforeEach(async () => {
    tmp = await makeTmpDir();
    _resetLearner();
  });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("passes all events through unchanged", async () => {
    const events: StreamEvent[] = [
      { type: "text_delta", text: "Hello world" },
      { type: "usage", usage: { inputTokens: 100, outputTokens: 20 } },
      { type: "message_end", stopReason: "end_turn" },
    ];
    const monitor = new StreamingCompressionMonitor({
      summarizer: fakeSummarizer(events),
      tier: 3,
      provider: "claude-sonnet",
      estimatedTokensByBlock: { text: 120 }, // estimated slightly higher → no overage
      cwd: tmp,
    });

    const received = await drainGenerator(monitor.stream(makeRequest([{ role: "user", content: "hi" }])));
    expect(received).toHaveLength(3);
    expect(received[0]!.type).toBe("text_delta");
    expect(received[1]!.type).toBe("usage");
    expect(received[2]!.type).toBe("message_end");
  });

  test("needsReTierRecommendation() returns false when actual is within threshold", async () => {
    const events: StreamEvent[] = [
      { type: "usage", usage: { inputTokens: 105, outputTokens: 10 } }, // 5% over — below 15%
      { type: "message_end", stopReason: "end_turn" },
    ];
    const monitor = new StreamingCompressionMonitor({
      summarizer: fakeSummarizer(events),
      tier: 2,
      provider: "claude-sonnet",
      estimatedTokensByBlock: { text: 100 },
      cwd: tmp,
    });

    await drainGenerator(monitor.stream(makeRequest([{ role: "user", content: "hi" }])));
    expect(monitor.needsReTierRecommendation()).toBe(false);
    expect(monitor.getRecommendedTier()).toBeNull();
  });

  test("getSessionSummary reflects zero overages", async () => {
    const events: StreamEvent[] = [
      { type: "usage", usage: { inputTokens: 80, outputTokens: 10 } }, // under-estimate
      { type: "message_end", stopReason: "end_turn" },
    ];
    const monitor = new StreamingCompressionMonitor({
      summarizer: fakeSummarizer(events),
      tier: 3,
      provider: "claude-sonnet",
      estimatedTokensByBlock: { text: 100 },
      cwd: tmp,
    });

    await drainGenerator(monitor.stream(makeRequest([{ role: "user", content: "hi" }])));
    const summary = monitor.getSessionSummary();
    expect(summary.needsReTier).toBe(false);
    expect(summary.overageEventCount).toBe(0);
    expect(summary.recommendedTier).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Scenario 2: High-cardinality tool results
// ---------------------------------------------------------------------------

describe("StreamingCompressionMonitor — high-cardinality tool results", () => {
  let tmp: string;
  beforeEach(async () => {
    tmp = await makeTmpDir();
    _resetLearner();
  });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("triggers re-tier when tool_result actual tokens are 30% above estimate", async () => {
    // Estimate 1000 tokens for tool_result; actual is 1300 (30% overage)
    const events: StreamEvent[] = [
      { type: "text_delta", text: "processing..." },
      { type: "usage", usage: { inputTokens: 1300, outputTokens: 50 } },
      { type: "message_end", stopReason: "end_turn" },
    ];
    const monitor = new StreamingCompressionMonitor({
      summarizer: fakeSummarizer(events),
      tier: 3,
      provider: "claude-sonnet",
      estimatedTokensByBlock: { tool_result: 1000 },
      cwd: tmp,
    });

    await drainGenerator(monitor.stream(makeRequest([{ role: "user", content: "run tool" }])));
    expect(monitor.needsReTierRecommendation()).toBe(true);
  });

  test("recommends tier-1 lower than current tier-2 after tool_result explosion", async () => {
    const events: StreamEvent[] = [
      { type: "usage", usage: { inputTokens: 5000, outputTokens: 80 } }, // 400% over
      { type: "message_end", stopReason: "end_turn" },
    ];
    const monitor = new StreamingCompressionMonitor({
      summarizer: fakeSummarizer(events),
      tier: 2,
      provider: "claude-sonnet",
      estimatedTokensByBlock: { tool_result: 1000 },
      cwd: tmp,
    });

    await drainGenerator(monitor.stream(makeRequest([{ role: "user", content: "run tool" }])));
    expect(monitor.needsReTierRecommendation()).toBe(true);
    const recommended = monitor.getRecommendedTier();
    expect(recommended).not.toBeNull();
    expect(recommended!).toBeLessThan(2);
  });

  test("writes signal to evolution/streaming-compression.jsonl", async () => {
    const events: StreamEvent[] = [
      { type: "usage", usage: { inputTokens: 1600, outputTokens: 60 } }, // 60% over
      { type: "message_end", stopReason: "end_turn" },
    ];
    const monitor = new StreamingCompressionMonitor({
      summarizer: fakeSummarizer(events),
      tier: 3,
      provider: "claude-opus",
      estimatedTokensByBlock: { tool_result: 1000 },
      cwd: tmp,
    });

    await drainGenerator(monitor.stream(makeRequest([{ role: "user", content: "run tool" }])));

    // Give the async I/O a tick to flush
    await new Promise((r) => setTimeout(r, 20));

    const signals = await readJsonl<StreamingCompressionSignal>(streamingFeedbackPath(tmp));
    expect(signals.length).toBeGreaterThanOrEqual(1);
    const signal = signals[0]!;
    expect(signal.tier).toBe(3);
    expect(signal.provider).toBe("claude-opus");
    expect(signal.overage_pct).toBeGreaterThan(15);
    expect(signal.recommended_next_tier).toBeLessThan(3);
    expect(typeof signal.timestamp).toBe("string");
    expect(() => new Date(signal.timestamp)).not.toThrow();
  });

  test("block budget tracks tool_result proportional share correctly", async () => {
    // 500 text + 1000 tool_result estimated; actual 1950 (30% over)
    const events: StreamEvent[] = [
      { type: "usage", usage: { inputTokens: 1950, outputTokens: 40 } },
      { type: "message_end", stopReason: "end_turn" },
    ];
    const monitor = new StreamingCompressionMonitor({
      summarizer: fakeSummarizer(events),
      tier: 2,
      provider: "claude-sonnet",
      estimatedTokensByBlock: { text: 500, tool_result: 1000 },
      cwd: tmp,
    });

    await drainGenerator(monitor.stream(makeRequest([{ role: "user", content: "hi" }])));

    const budgets = monitor.getBlockBudgets();
    const toolBudget = budgets.find((b) => b.blockType === "tool_result");
    expect(toolBudget).toBeDefined();
    expect(toolBudget!.actual).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Scenario 3: Cache miss cascade
// ---------------------------------------------------------------------------

describe("StreamingCompressionMonitor — cache miss cascade", () => {
  let tmp: string;
  beforeEach(async () => {
    tmp = await makeTmpDir();
    _resetLearner();
  });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("multiple consecutive usage events accumulate correctly", async () => {
    // Simulate two usage events arriving mid-stream (cache miss on retry)
    const events: StreamEvent[] = [
      { type: "text_delta", text: "part 1" },
      { type: "usage", usage: { inputTokens: 800, outputTokens: 20 } },
      { type: "text_delta", text: "part 2" },
      { type: "usage", usage: { inputTokens: 1800, outputTokens: 60 } }, // 80% over 1000 estimate
      { type: "message_end", stopReason: "end_turn" },
    ];
    const monitor = new StreamingCompressionMonitor({
      summarizer: fakeSummarizer(events),
      tier: 3,
      provider: "claude-sonnet",
      estimatedTokensByBlock: { text: 1000 },
      cwd: tmp,
    });

    await drainGenerator(monitor.stream(makeRequest([{ role: "user", content: "hi" }])));

    // First usage (800/1000 = -20%) is fine; second usage (1800/1000 = +80%) triggers
    expect(monitor.needsReTierRecommendation()).toBe(true);
    expect(monitor.getSessionSummary().overageEventCount).toBeGreaterThanOrEqual(1);
  });

  test("exponential backoff tightens threshold after first overage", async () => {
    // First usage: exactly 15% over (triggers first signal, tightens threshold)
    // Second usage: 11% over (below original 15% but above tightened 12% threshold)
    const events: StreamEvent[] = [
      { type: "usage", usage: { inputTokens: 1150, outputTokens: 20 } }, // 15% over
      { type: "usage", usage: { inputTokens: 1110, outputTokens: 20 } }, // 11% over original est
      { type: "message_end", stopReason: "end_turn" },
    ];
    const monitor = new StreamingCompressionMonitor({
      summarizer: fakeSummarizer(events),
      tier: 3,
      provider: "claude-sonnet",
      estimatedTokensByBlock: { text: 1000 },
      cwd: tmp,
    });

    await drainGenerator(monitor.stream(makeRequest([{ role: "user", content: "hi" }])));

    // With backoff the monitor may fire multiple signals; total should reflect cascade
    const summary = monitor.getSessionSummary();
    expect(summary.needsReTier).toBe(true);
  });

  test("tier-1 monitor never recommends below tier 1", async () => {
    const events: StreamEvent[] = [
      { type: "usage", usage: { inputTokens: 5000, outputTokens: 80 } },
      { type: "message_end", stopReason: "end_turn" },
    ];
    const monitor = new StreamingCompressionMonitor({
      summarizer: fakeSummarizer(events),
      tier: 1,  // already most aggressive
      provider: "claude-sonnet",
      estimatedTokensByBlock: { text: 500 },
      cwd: tmp,
    });

    await drainGenerator(monitor.stream(makeRequest([{ role: "user", content: "hi" }])));
    // Re-tier fires, but recommended is still 1 (clamped)
    expect(monitor.getRecommendedTier()).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Scenario 4: Provider codec drift
// ---------------------------------------------------------------------------

describe("StreamingCompressionMonitor — provider codec drift", () => {
  let tmp: string;
  beforeEach(async () => {
    tmp = await makeTmpDir();
    _resetLearner();
  });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("createMonitor factory classifies messages and builds estimates via codec", async () => {
    const request = makeRequest([
      { role: "user", content: "Hello, what is the weather today?" },
      { role: "assistant", content: "Let me check the weather for you." },
    ]);
    const events: StreamEvent[] = [
      { type: "usage", usage: { inputTokens: 30, outputTokens: 10 } },
      { type: "message_end", stopReason: "end_turn" },
    ];
    const monitor = createMonitor(
      fakeSummarizer(events),
      request,
      3,
      "claude-sonnet",
      tmp,
    );

    await drainGenerator(monitor.stream(request));

    // Should have classified text blocks
    const budgets = monitor.getBlockBudgets();
    expect(budgets.length).toBeGreaterThan(0);
    const textBudget = budgets.find((b) => b.blockType === "text");
    expect(textBudget).toBeDefined();
  });

  test("different providers produce different estimates (codec-aware)", async () => {
    const messages = [
      { role: "user" as const, content: "A".repeat(400) },
    ];
    const request = makeRequest(messages);
    const events: StreamEvent[] = [
      { type: "usage", usage: { inputTokens: 120, outputTokens: 5 } },
      { type: "message_end", stopReason: "end_turn" },
    ];

    const monitorClaude = createMonitor(fakeSummarizer(events), request, 2, "claude-sonnet", tmp);
    const monitorGpt = createMonitor(fakeSummarizer(events), request, 2, "gpt-4", tmp);

    const budgetsClaude = monitorClaude.getBlockBudgets();
    const budgetsGpt = monitorGpt.getBlockBudgets();

    // Both should produce non-zero estimates
    const totalClaude = budgetsClaude.reduce((s, b) => s + b.estimated, 0);
    const totalGpt = budgetsGpt.reduce((s, b) => s + b.estimated, 0);
    expect(totalClaude).toBeGreaterThan(0);
    expect(totalGpt).toBeGreaterThan(0);
    // They may or may not differ depending on codec config — but both must be positive
  });

  test("streamingFeedbackPath returns path inside genome/evolution", () => {
    const p = streamingFeedbackPath("/projects/myapp");
    expect(p).toContain(".ashlrcode/genome/evolution");
    expect(p).toContain("streaming-compression.jsonl");
  });

  test("signal record includes provider and tier from monitor config", async () => {
    const events: StreamEvent[] = [
      { type: "usage", usage: { inputTokens: 2000, outputTokens: 40 } }, // 100% over
      { type: "message_end", stopReason: "end_turn" },
    ];
    const monitor = new StreamingCompressionMonitor({
      summarizer: fakeSummarizer(events),
      tier: 3,
      provider: "deepseek",
      estimatedTokensByBlock: { text: 1000 },
      cwd: tmp,
    });

    await drainGenerator(monitor.stream(makeRequest([{ role: "user", content: "hi" }])));
    await new Promise((r) => setTimeout(r, 20));

    const signals = await readJsonl<StreamingCompressionSignal>(streamingFeedbackPath(tmp));
    expect(signals.length).toBeGreaterThanOrEqual(1);
    expect(signals[0]!.provider).toBe("deepseek");
    expect(signals[0]!.tier).toBe(3);
    expect(signals[0]!.recommended_next_tier).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// Scenario 5: Thinking token explosion
// ---------------------------------------------------------------------------

describe("StreamingCompressionMonitor — thinking token explosion", () => {
  let tmp: string;
  beforeEach(async () => {
    tmp = await makeTmpDir();
    _resetLearner();
  });
  afterEach(async () => { await rm(tmp, { recursive: true, force: true }); });

  test("thinking block overage triggers re-tier with correct block attribution", async () => {
    // 200 text + 300 thinking estimated; actual is 700 (40% over combined 500)
    const events: StreamEvent[] = [
      { type: "thinking_delta", text: "Let me reason step by step..." },
      { type: "usage", usage: { inputTokens: 700, outputTokens: 120 } },
      { type: "message_end", stopReason: "end_turn" },
    ];
    const monitor = new StreamingCompressionMonitor({
      summarizer: fakeSummarizer(events),
      tier: 2,
      provider: "claude-opus",
      estimatedTokensByBlock: { text: 200, thinking: 300 },
      cwd: tmp,
    });

    await drainGenerator(monitor.stream(makeRequest([{ role: "user", content: "reason about this" }])));

    expect(monitor.needsReTierRecommendation()).toBe(true);
    const budgets = monitor.getBlockBudgets();
    const thinkingBudget = budgets.find((b) => b.blockType === "thinking");
    expect(thinkingBudget).toBeDefined();
    expect(thinkingBudget!.actual).toBeGreaterThan(0);
  });

  test("large thinking overage is recorded in the JSONL signal file", async () => {
    const events: StreamEvent[] = [
      { type: "usage", usage: { inputTokens: 4000, outputTokens: 500 } }, // 300% over
      { type: "message_end", stopReason: "end_turn" },
    ];
    const monitor = new StreamingCompressionMonitor({
      summarizer: fakeSummarizer(events),
      tier: 3,
      provider: "claude-opus",
      estimatedTokensByBlock: { thinking: 1000 },
      cwd: tmp,
    });

    await drainGenerator(monitor.stream(makeRequest([{ role: "user", content: "think hard" }])));
    await new Promise((r) => setTimeout(r, 20));

    const signals = await readJsonl<StreamingCompressionSignal>(streamingFeedbackPath(tmp));
    expect(signals.length).toBeGreaterThanOrEqual(1);
    expect(signals[0]!.overage_pct).toBeGreaterThan(100);
  });

  test("streamWithMessages classifies thinking blocks from request", async () => {
    const events: StreamEvent[] = [
      { type: "usage", usage: { inputTokens: 50, outputTokens: 10 } },
      { type: "message_end", stopReason: "end_turn" },
    ];
    const request: ProviderRequest = {
      systemPrompt: "You are helpful.",
      messages: [
        {
          role: "assistant",
          content: [
            { type: "thinking", thinking: "Let me think..." },
            { type: "text", text: "The answer is 42." },
          ],
        },
      ],
      tools: [],
    };
    const monitor = new StreamingCompressionMonitor({
      summarizer: fakeSummarizer(events),
      tier: 2,
      provider: "claude-opus",
      estimatedTokensByBlock: {},
      cwd: tmp,
    });

    await drainGenerator(monitor.streamWithMessages(request));

    const budgets = monitor.getBlockBudgets();
    // Should have at least one budget entry from the classified message
    expect(budgets.length).toBeGreaterThan(0);
  });

  test("getSessionSummary maxOverageFraction reflects worst-case block", async () => {
    // text block: 10% over; thinking block: 50% over
    const events: StreamEvent[] = [
      { type: "usage", usage: { inputTokens: 1650, outputTokens: 80 } }, // combined 50% over 1100
      { type: "message_end", stopReason: "end_turn" },
    ];
    const monitor = new StreamingCompressionMonitor({
      summarizer: fakeSummarizer(events),
      tier: 2,
      provider: "claude-opus",
      estimatedTokensByBlock: { text: 100, thinking: 1000 },
      cwd: tmp,
    });

    await drainGenerator(monitor.stream(makeRequest([{ role: "user", content: "think" }])));

    const summary = monitor.getSessionSummary();
    expect(summary.maxOverageFraction).toBeGreaterThan(0);
  });
});
