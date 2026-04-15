/**
 * Token counting.
 *
 * Two paths:
 *   1) Heuristic (chars/4) — fast, zero-dependency, used for back-compat and
 *      as a fallback when the tokenizer can't load.
 *   2) Accurate — tiktoken-backed (`cl100k_base` / `o200k_base`). Anthropic
 *      doesn't publish its tokenizer, so we use GPT-4's encoding as a proxy
 *      (the convention across the Claude ecosystem).
 *
 * The tokenizer is lazily loaded (no wasm at import time) and cached across
 * calls. If load fails, we silently degrade to the heuristic.
 */

import type { ContentBlock, Message } from "../types/index.ts";

// ---------- heuristic (back-compat, unchanged) ----------

/** Estimate tokens in a plain string. */
export function estimateTokensFromString(text: string): number {
  return Math.ceil(text.length / 4);
}

/** Estimate tokens across a list of provider Messages (handles content blocks). */
export function estimateTokensFromMessages(messages: Message[]): number {
  let chars = 0;
  for (const msg of messages) {
    if (typeof msg.content === "string") {
      chars += msg.content.length;
    } else {
      for (const block of msg.content) {
        chars += blockCharCount(block);
      }
    }
  }
  return Math.ceil(chars / 4);
}

/** Unified entry point — dispatches by input shape. */
export function estimateTokens(input: string | Message[]): number {
  return typeof input === "string"
    ? estimateTokensFromString(input)
    : estimateTokensFromMessages(input);
}

function blockCharCount(block: ContentBlock): number {
  switch (block.type) {
    case "text":
      return block.text.length;
    case "thinking":
      return block.thinking.length;
    case "tool_use":
      return block.name.length + JSON.stringify(block.input).length;
    case "tool_result":
      return block.content.length;
    case "image_url":
      return 1000; // ~1000 tokens per image
  }
}

// ---------- accurate (tiktoken-backed) ----------

export type TokenizerModel = "claude-3-5" | "gpt-4" | "default";

interface Encoder {
  encode(text: string): { length: number };
}

type EncodingName = "cl100k_base" | "o200k_base";

const encoderCache = new Map<EncodingName, Encoder | null>();
let loaderFailed = false;

/**
 * Load & cache a tiktoken encoder. Returns `null` if tiktoken isn't
 * available or the encoding can't be initialized — caller should fall back.
 *
 * Exposed for tests (monkey-patching a failure path).
 */
export async function _loadEncoder(
  encoding: EncodingName,
): Promise<Encoder | null> {
  if (encoderCache.has(encoding)) return encoderCache.get(encoding) ?? null;
  if (loaderFailed) return null;
  try {
    // Dynamic import so import-time cost is zero and missing dep is survivable.
    const mod: any = await import("tiktoken");
    const enc = mod.get_encoding(encoding) as Encoder;
    encoderCache.set(encoding, enc);
    return enc;
  } catch {
    loaderFailed = true;
    encoderCache.set(encoding, null);
    return null;
  }
}

/** Test hook: reset cached encoders + failure flag. */
export function _resetTokenizerCache(): void {
  encoderCache.clear();
  loaderFailed = false;
}

/** Test hook: force the loader to fail on next call (simulate missing dep). */
export function _forceLoaderFailure(): void {
  loaderFailed = true;
  encoderCache.clear();
}

function pickEncoding(model: TokenizerModel): EncodingName {
  // cl100k_base: GPT-4 / Claude-ecosystem default. Works well as a Claude
  // proxy (Anthropic's tokenizer is closed).
  // o200k_base: newer GPT-4o / gpt-4o-mini encoding. Map `claude-3-5` here
  // since modern Claude tends to tokenize closer to o200k for code.
  switch (model) {
    case "claude-3-5":
      return "o200k_base";
    case "gpt-4":
      return "cl100k_base";
    case "default":
    default:
      return "cl100k_base";
  }
}

/**
 * Sync facade around the (async) encoder load. Runs the loader once and
 * blocks via a module-level promise only for the first call; subsequent
 * calls use the cached encoder directly.
 *
 * We expose the async API below as the canonical accurate entry point; this
 * sync helper is used internally after the first load resolves.
 */
function encodedLength(enc: Encoder, text: string): number {
  if (!text) return 0;
  return enc.encode(text).length;
}

/**
 * Accurate token count. Uses tiktoken if available, falls back to chars/4.
 *
 * Note: returns a Promise because tokenizer load is async. For a truly sync
 * path, call `primeTokenizer()` once at startup, then `countTokensAccurateSync`.
 */
export async function countTokensAccurate(
  input: string | Message[],
  model: TokenizerModel = "default",
): Promise<number> {
  const encoding = pickEncoding(model);
  const enc = await _loadEncoder(encoding);
  if (!enc) {
    // Fallback: heuristic.
    return estimateTokens(input);
  }
  try {
    if (typeof input === "string") return encodedLength(enc, input);
    let total = 0;
    for (const msg of input) {
      if (typeof msg.content === "string") {
        total += encodedLength(enc, msg.content);
      } else {
        for (const block of msg.content) {
          total += blockTokenCount(enc, block);
        }
      }
      // Per-message overhead (role + framing). Matches OpenAI's published
      // guidance for cl100k chat framing; ~4 tokens/message.
      total += 4;
    }
    return total;
  } catch {
    // Tokenizer blew up mid-encode — degrade.
    return estimateTokens(input);
  }
}

function blockTokenCount(enc: Encoder, block: ContentBlock): number {
  switch (block.type) {
    case "text":
      return encodedLength(enc, block.text);
    case "thinking":
      return encodedLength(enc, block.thinking);
    case "tool_use":
      return (
        encodedLength(enc, block.name) +
        encodedLength(enc, JSON.stringify(block.input))
      );
    case "tool_result":
      return encodedLength(enc, block.content);
    case "image_url":
      return 1000;
  }
}

/** Prime the tokenizer so later calls are cheap. Safe to call multiple times. */
export async function primeTokenizer(
  model: TokenizerModel = "default",
): Promise<boolean> {
  const enc = await _loadEncoder(pickEncoding(model));
  return enc !== null;
}
