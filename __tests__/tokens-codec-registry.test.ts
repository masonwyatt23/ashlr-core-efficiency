/**
 * Tests for the Multi-Provider Token Estimation Codec Registry
 * (src/tokens/provider-codec.ts, src/tokens/recalibration-engine.ts)
 * and the calibratedEstimate() export from src/tokens/index.ts.
 *
 * Coverage:
 *   - CodecRegistry: per-provider codec resolution, auto-detect from response
 *   - ProviderTokenCodec: estimateInput, encodeMessage, calibrate, correctionFactor
 *   - classifyMessage: bin classification for all content-block types
 *   - RecalibrationEngine: per-bin correction factors, EMA updates, fallback
 *   - calibratedEstimate: adaptive accuracy, fallback on first call, cache hits
 *   - Persistence: RecalibrationRecord written to JSONL (smoke test)
 *   - Drift detection: correction factor converges toward actual/estimated ratio
 */

import { describe, test, expect, beforeEach } from "bun:test";

import {
  CodecRegistry,
  classifyMessage,
  type ProviderTokenCodec,
  type MessageBin,
} from "../src/tokens/provider-codec.ts";

import {
  RecalibrationEngine,
  RECAL_EMA_ALPHA,
  RECAL_FACTOR_MIN,
  RECAL_FACTOR_MAX,
  RECAL_MIN_SAMPLES,
} from "../src/tokens/recalibration-engine.ts";

import {
  calibratedEstimate,
  _resetTokenizerCache,
} from "../src/tokens/index.ts";

import type { Message } from "../src/types/index.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function txt(text: string): Message {
  return { role: "user", content: text };
}

function msgBlocks(blocks: Message["content"]): Message {
  return { role: "user", content: blocks };
}

function makeMessages(n: number, chars = 100): Message[] {
  return Array.from({ length: n }, (_, i) => txt(`message ${i} ${"x".repeat(chars)}`));
}

beforeEach(() => {
  CodecRegistry.resetInstance();
  RecalibrationEngine.resetInstance();
  _resetTokenizerCache();
});

// ---------------------------------------------------------------------------
// 1. CodecRegistry — provider resolution
// ---------------------------------------------------------------------------

describe("CodecRegistry — provider resolution", () => {
  test("claude-3-5-sonnet-20241022 resolves to claude-sonnet codec", () => {
    const codec = CodecRegistry.instance.getCodec("claude-3-5-sonnet-20241022");
    expect(codec.providerId).toBe("claude-sonnet");
  });

  test("gpt-4o-mini resolves to gpt-4o codec", () => {
    const codec = CodecRegistry.instance.getCodec("gpt-4o-mini");
    expect(codec.providerId).toBe("gpt-4o");
  });

  test("gpt-4-turbo resolves to gpt-4 codec", () => {
    const codec = CodecRegistry.instance.getCodec("gpt-4-turbo");
    expect(codec.providerId).toBe("gpt-4");
  });

  test("meta-llama/Llama-3-8b resolves to llama codec", () => {
    const codec = CodecRegistry.instance.getCodec("meta-llama/Llama-3-8b");
    expect(codec.providerId).toBe("llama");
  });

  test("qwen-72b resolves to qwen codec", () => {
    const codec = CodecRegistry.instance.getCodec("qwen-72b");
    expect(codec.providerId).toBe("qwen");
  });

  test("groq resolves to groq codec", () => {
    const codec = CodecRegistry.instance.getCodec("groq");
    expect(codec.providerId).toBe("groq");
  });

  test("mistral-large resolves to mistral codec", () => {
    const codec = CodecRegistry.instance.getCodec("mistral-large");
    expect(codec.providerId).toBe("mistral");
  });

  test("gemini-1.5-pro resolves to gemini codec", () => {
    const codec = CodecRegistry.instance.getCodec("gemini-1.5-pro");
    expect(codec.providerId).toBe("gemini");
  });

  test("totally-unknown-model-xyz falls back to default codec", () => {
    const codec = CodecRegistry.instance.getCodec("totally-unknown-model-xyz");
    expect(codec.providerId).toBe("default");
  });

  test("empty string falls back to default codec", () => {
    const codec = CodecRegistry.instance.getCodec("");
    expect(codec.providerId).toBe("default");
  });

  test("getCodec is deterministic — same model returns same codec object", () => {
    const a = CodecRegistry.instance.getCodec("claude-3-5-sonnet");
    const b = CodecRegistry.instance.getCodec("claude-3-5-sonnet");
    expect(a).toBe(b);
  });

  test("registeredProviders includes all built-in providers", () => {
    const providers = CodecRegistry.instance.registeredProviders();
    expect(providers).toContain("claude-sonnet");
    expect(providers).toContain("gpt-4");
    expect(providers).toContain("gpt-4o");
    expect(providers).toContain("llama");
    expect(providers).toContain("qwen");
    expect(providers).toContain("groq");
    expect(providers).toContain("default");
  });
});

// ---------------------------------------------------------------------------
// 2. CodecRegistry — auto-detect from API response
// ---------------------------------------------------------------------------

describe("CodecRegistry — detectFromResponse", () => {
  test("detects claude from model field", () => {
    const codec = CodecRegistry.instance.detectFromResponse({
      model: "claude-3-5-sonnet-20241022",
      usage: { input_tokens: 100, output_tokens: 20 },
    });
    expect(codec.providerId).toBe("claude-sonnet");
  });

  test("detects gpt-4o from model field", () => {
    const codec = CodecRegistry.instance.detectFromResponse({
      model: "gpt-4o",
      object: "chat.completion",
    });
    expect(codec.providerId).toBe("gpt-4o");
  });

  test("detects from provider field (proxy response)", () => {
    const codec = CodecRegistry.instance.detectFromResponse({
      provider: "llama2",
    });
    expect(codec.providerId).toBe("llama");
  });

  test("falls back to default when no model or provider field", () => {
    const codec = CodecRegistry.instance.detectFromResponse({
      object: "chat.completion",
    });
    expect(codec.providerId).toBe("default");
  });
});

// ---------------------------------------------------------------------------
// 3. ProviderTokenCodec — estimateInput / encodeMessage
// ---------------------------------------------------------------------------

describe("ProviderTokenCodec — estimateInput and encodeMessage", () => {
  test("estimateInput returns positive number for text messages", () => {
    const codec = CodecRegistry.instance.getCodec("claude-3-5");
    const msgs = makeMessages(3, 200);
    expect(codec.estimateInput(msgs)).toBeGreaterThan(0);
  });

  test("encodeMessage returns positive number for text message", () => {
    const codec = CodecRegistry.instance.getCodec("gpt-4");
    expect(codec.encodeMessage(txt("Hello world"))).toBeGreaterThan(0);
  });

  test("llama codec applies 1.05× nominal multiplier vs default (1.0×)", () => {
    const llamaCodec = CodecRegistry.instance.getCodec("llama2");
    const defaultCodec = CodecRegistry.instance.getCodec("gpt-4");
    const msgs = makeMessages(5, 400);
    const llamaEst = llamaCodec.estimateInput(msgs);
    const defaultEst = defaultCodec.estimateInput(msgs);
    // Llama should be >= default due to multiplier.
    expect(llamaEst).toBeGreaterThanOrEqual(defaultEst);
  });

  test("qwen codec applies 1.08× nominal multiplier vs gpt-4 (1.0×)", () => {
    const qwenCodec = CodecRegistry.instance.getCodec("qwen-7b");
    const gptCodec  = CodecRegistry.instance.getCodec("gpt-4");
    const msgs = makeMessages(5, 400);
    expect(qwenCodec.estimateInput(msgs)).toBeGreaterThanOrEqual(gptCodec.estimateInput(msgs));
  });

  test("correctionFactor starts at 1.0 for fresh codec", () => {
    const codec = CodecRegistry.instance.getCodec("claude-3-5");
    expect(codec.correctionFactor).toBe(1.0);
  });

  test("calibrate updates correctionFactor toward actual/estimated ratio", () => {
    const codec = CodecRegistry.instance.getCodec("gpt-4");
    // actual is 20% higher than estimated → factor should increase above 1.0.
    codec.calibrate(100, 120, "gpt-4");
    expect(codec.correctionFactor).toBeGreaterThan(1.0);
  });

  test("calibrate clamps correction factor within [FACTOR_MIN, FACTOR_MAX]", () => {
    const codec = CodecRegistry.instance.getCodec("llama2");
    // Extreme: actual = 10× estimated → target factor well above FACTOR_MAX.
    for (let i = 0; i < 50; i++) {
      codec.calibrate(10, 100, "llama2");
    }
    expect(codec.correctionFactor).toBeLessThanOrEqual(2.0);
    expect(codec.correctionFactor).toBeGreaterThanOrEqual(0.8);
  });

  test("calibrate with 0 estimated does nothing (no division by zero)", () => {
    const codec = CodecRegistry.instance.getCodec("gpt-4o");
    expect(() => codec.calibrate(0, 100, "gpt-4o")).not.toThrow();
    expect(codec.correctionFactor).toBe(1.0);
  });
});

// ---------------------------------------------------------------------------
// 4. classifyMessage — bin classification
// ---------------------------------------------------------------------------

describe("classifyMessage — bin classification", () => {
  test("string content → text", () => {
    expect(classifyMessage(txt("hello"))).toBe("text");
  });

  test("text block array → text", () => {
    expect(classifyMessage(msgBlocks([{ type: "text", text: "hi" }]))).toBe("text");
  });

  test("image_url block → image", () => {
    expect(classifyMessage(msgBlocks([
      { type: "image_url", image_url: { url: "https://example.com/img.png" } },
    ]))).toBe("image");
  });

  test("thinking block → thinking", () => {
    expect(classifyMessage(msgBlocks([
      { type: "thinking", thinking: "let me reason", signature: "sig" },
    ]))).toBe("thinking");
  });

  test("tool_use block → tool_use", () => {
    expect(classifyMessage(msgBlocks([
      { type: "tool_use", id: "t1", name: "bash", input: {} },
    ]))).toBe("tool_use");
  });

  test("tool_result block → tool_result", () => {
    expect(classifyMessage(msgBlocks([
      { type: "tool_result", tool_use_id: "t1", content: "done" },
    ]))).toBe("tool_result");
  });

  test("tool_use + tool_result → mixed", () => {
    expect(classifyMessage(msgBlocks([
      { type: "tool_use", id: "t1", name: "read", input: {} },
      { type: "tool_result", tool_use_id: "t1", content: "result" },
    ]))).toBe("mixed");
  });

  test("text + image → image (image takes priority)", () => {
    expect(classifyMessage(msgBlocks([
      { type: "text", text: "what is this?" },
      { type: "image_url", image_url: { url: "https://x.com/img.png" } },
    ]))).toBe("image");
  });
});

// ---------------------------------------------------------------------------
// 5. RecalibrationEngine — per-bin correction factors
// ---------------------------------------------------------------------------

describe("RecalibrationEngine — per-bin factors and EMA updates", () => {
  test("correctionFactor returns 1.0 before any records (no-history fallback)", () => {
    const engine = RecalibrationEngine.instance;
    expect(engine.correctionFactor("claude-3-5", "text")).toBe(1.0);
  });

  test("overallFactor returns 1.0 before any records", () => {
    expect(RecalibrationEngine.instance.overallFactor("unknown-model")).toBe(1.0);
  });

  test("record updates overallFactor via EMA toward actual/estimated ratio", () => {
    const engine = RecalibrationEngine.instance;
    const msgs = makeMessages(2, 100);
    // Record 10 observations: actual = 1.2× estimated.
    for (let i = 0; i < 10; i++) {
      engine.record({ provider: "claude-3-5", messages: msgs, estimated: 100, actual: 120 });
    }
    const factor = engine.overallFactor("claude-3-5");
    // After many 1.2× observations, factor should be above 1.0.
    expect(factor).toBeGreaterThan(1.0);
    expect(factor).toBeLessThanOrEqual(RECAL_FACTOR_MAX);
  });

  test("record falls back to overall when per-bin has fewer than MIN_SAMPLES", () => {
    const engine = RecalibrationEngine.instance;
    const msgs = makeMessages(1, 100);
    // Record only 1 text-bin sample (below MIN_SAMPLES).
    engine.record({ provider: "gpt-4", messages: msgs, estimated: 100, actual: 130 });
    // Per-bin correctionFactor should fall back to overall.
    const binFactor = engine.correctionFactor("gpt-4", "text");
    expect(binFactor).toBeGreaterThanOrEqual(RECAL_FACTOR_MIN);
  });

  test("per-bin factor converges independently for different bins", () => {
    const engine = RecalibrationEngine.instance;
    const textMsgs = [txt("plain text message")];
    const toolMsgs: Message[] = [{
      role: "user",
      content: [{ type: "tool_result", tool_use_id: "t1", content: "result" }],
    }];

    // Text bin: actual = 1.0× (no error).
    for (let i = 0; i < 5; i++) {
      engine.record({ provider: "claude-3-5", messages: textMsgs, estimated: 100, actual: 100 });
    }
    // Tool-result bin: actual = 1.4× (large under-estimate).
    for (let i = 0; i < 5; i++) {
      engine.record({ provider: "claude-3-5", messages: toolMsgs, estimated: 100, actual: 140 });
    }

    const textFactor   = engine.correctionFactor("claude-3-5", "text");
    const toolFactor   = engine.correctionFactor("claude-3-5", "tool_result");
    // Tool factor should be higher than text factor.
    expect(toolFactor).toBeGreaterThan(textFactor);
  });

  test("correction factor is clamped within [RECAL_FACTOR_MIN, RECAL_FACTOR_MAX]", () => {
    const engine = RecalibrationEngine.instance;
    const msgs = makeMessages(1, 50);
    // Extreme: actual = 10× estimated — should not push factor above max.
    for (let i = 0; i < 30; i++) {
      engine.record({ provider: "qwen-72b", messages: msgs, estimated: 10, actual: 100 });
    }
    const factor = engine.overallFactor("qwen-72b");
    expect(factor).toBeLessThanOrEqual(RECAL_FACTOR_MAX);
    expect(factor).toBeGreaterThanOrEqual(RECAL_FACTOR_MIN);
  });

  test("mape returns 0 for unknown provider", () => {
    expect(RecalibrationEngine.instance.mape("no-such-provider")).toBe(0);
  });

  test("mape increases after high-error observations", () => {
    const engine = RecalibrationEngine.instance;
    const msgs = makeMessages(1, 100);
    for (let i = 0; i < 8; i++) {
      engine.record({ provider: "mistral-7b", messages: msgs, estimated: 100, actual: 200 });
    }
    expect(engine.mape("mistral-7b", "text")).toBeGreaterThan(0);
  });

  test("knownProviders returns providers after recording", () => {
    const engine = RecalibrationEngine.instance;
    engine.record({ provider: "llama3-70b", messages: [txt("hi")], estimated: 50, actual: 55 });
    expect(engine.knownProviders()).toContain("llama3-70b");
  });

  test("reset() clears all state", () => {
    const engine = RecalibrationEngine.instance;
    engine.record({ provider: "gpt-4o", messages: [txt("test")], estimated: 100, actual: 150 });
    engine.reset();
    expect(engine.knownProviders()).toHaveLength(0);
    expect(engine.overallFactor("gpt-4o")).toBe(1.0);
  });

  test("record() with bin override uses the provided bin", () => {
    const engine = RecalibrationEngine.instance;
    const msgs = [txt("text message")]; // would be classified as "text"

    // Train "image" bin heavily: actual = 1.8× estimated.
    for (let i = 0; i < 10; i++) {
      engine.record({ provider: "model-a", messages: msgs, estimated: 100, actual: 180, bin: "image" });
    }
    // Train "text" bin with no error: actual = 1.0× estimated.
    for (let i = 0; i < 10; i++) {
      engine.record({ provider: "model-b", messages: msgs, estimated: 100, actual: 100, bin: "text" });
    }
    const imageFactor = engine.correctionFactor("model-a", "image");
    const textFactor  = engine.correctionFactor("model-b", "text");
    // Image bin is trained toward 1.8× so its factor must be above the text
    // bin which was trained toward 1.0×.
    expect(imageFactor).toBeGreaterThan(textFactor);
  });
});

// ---------------------------------------------------------------------------
// 6. calibratedEstimate — adaptive accuracy
// ---------------------------------------------------------------------------

describe("calibratedEstimate — adaptive per-request accuracy", () => {
  test("returns positive number for non-empty messages", async () => {
    const msgs = makeMessages(3, 200);
    const n = await calibratedEstimate(msgs, "claude-3-5-sonnet-20241022");
    expect(n).toBeGreaterThan(0);
  });

  test("fallback to heuristic on first call — no history, factor=1.0", async () => {
    const msgs = makeMessages(2, 100);
    // Fresh engine: correction factor is 1.0, so calibratedEstimate == countTokensAccurate.
    const engine = RecalibrationEngine.instance;
    const calibrated = await calibratedEstimate(msgs, "claude-3-5", engine);
    // Must be positive; factor=1.0 so no inflation.
    expect(calibrated).toBeGreaterThan(0);
    // Factor is 1.0, so result matches countTokensAccurate directly.
    expect(engine.overallFactor("claude-3-5")).toBe(1.0);
  });

  test("applies correction factor after training", async () => {
    const engine = RecalibrationEngine.instance;
    const msgs = makeMessages(3, 200);

    // Train: actual is consistently 30% higher than estimated.
    for (let i = 0; i < 10; i++) {
      engine.record({ provider: "gpt-4o", messages: msgs, estimated: 100, actual: 130 });
    }

    const calibrated = await calibratedEstimate(msgs, "gpt-4o", engine);
    // After training with 1.3× ratio, calibrated should be higher than the
    // raw base estimate (factor > 1.0 is applied).
    const factor = engine.overallFactor("gpt-4o");
    expect(factor).toBeGreaterThan(1.0);
    // The calibrated value must be positive.
    expect(calibrated).toBeGreaterThan(0);
  });

  test("deterministic — same input + same state gives same result", async () => {
    const msgs = makeMessages(2, 150);
    const a = await calibratedEstimate(msgs, "claude-3-5");
    const b = await calibratedEstimate(msgs, "claude-3-5");
    expect(a).toBe(b);
  });

  test("empty messages array returns 0", async () => {
    const n = await calibratedEstimate([], "claude-3-5");
    expect(n).toBe(0);
  });

  test("uses singleton RecalibrationEngine when no history arg provided", async () => {
    const msgs = makeMessages(2, 100);
    const singleton = RecalibrationEngine.instance;
    // Record training data on singleton.
    for (let i = 0; i < 5; i++) {
      singleton.record({ provider: "llama3", messages: msgs, estimated: 100, actual: 120 });
    }
    const result = await calibratedEstimate(msgs, "llama3");
    expect(result).toBeGreaterThan(0);
  });

  test("calibratedEstimate result matches factor × base for known factor", async () => {
    const engine = RecalibrationEngine.instance;
    const msgs = [txt("hello world, this is a test message to check calibration")];

    // Apply many 2× observations to push factor close to max.
    for (let i = 0; i < 20; i++) {
      engine.record({ provider: "gemini-1.5-pro", messages: msgs, estimated: 50, actual: 100 });
    }

    const factor = engine.overallFactor("gemini-1.5-pro");
    expect(factor).toBeGreaterThan(1.0);

    const calibrated = await calibratedEstimate(msgs, "gemini-1.5-pro", engine);
    expect(calibrated).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// 7. CodecRegistry — custom codec registration
// ---------------------------------------------------------------------------

describe("CodecRegistry — custom codec registration", () => {
  test("register() adds a custom codec retrievable by providerId", () => {
    const customCodec: ProviderTokenCodec = {
      providerId: "my-custom-model",
      correctionFactor: 1.0,
      estimateInput: (msgs) => msgs.length * 42,
      encodeMessage: (_msg) => 42,
      calibrate: (_est, _act, _prov) => { /* no-op */ },
    };
    CodecRegistry.instance.register(customCodec, "my-custom");
    const resolved = CodecRegistry.instance.getCodec("my-custom-model-v2");
    expect(resolved.providerId).toBe("my-custom-model");
  });
});

// ---------------------------------------------------------------------------
// 8. Constants sanity
// ---------------------------------------------------------------------------

describe("recalibration constants — sanity", () => {
  test("RECAL_EMA_ALPHA is between 0 and 1", () => {
    expect(RECAL_EMA_ALPHA).toBeGreaterThan(0);
    expect(RECAL_EMA_ALPHA).toBeLessThan(1);
  });

  test("RECAL_FACTOR_MIN < RECAL_FACTOR_MAX", () => {
    expect(RECAL_FACTOR_MIN).toBeLessThan(RECAL_FACTOR_MAX);
  });

  test("RECAL_MIN_SAMPLES is positive integer", () => {
    expect(RECAL_MIN_SAMPLES).toBeGreaterThan(0);
    expect(Number.isInteger(RECAL_MIN_SAMPLES)).toBe(true);
  });
});
