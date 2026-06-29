/**
 * Multi-Provider Token Estimation Codec Registry
 *
 * Provides a `ProviderTokenCodec` interface for per-provider token estimation,
 * a `CodecRegistry` that maps provider/model slugs to their codec, and
 * auto-detection of provider from chat-response metadata.
 *
 * Design goals:
 *   - Zero circular imports: does not import from index.ts.
 *   - Stateless codecs: all state lives in RecalibrationEngine (recalibration-engine.ts).
 *   - Matches the existing MODEL_ENCODING_MAP heuristics from index.ts so the
 *     two paths converge on the same baseline estimate.
 */

import type { Message, ContentBlock } from "../types/index.ts";

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/**
 * A single provider's token-counting codec.
 *
 * Implementations may be purely heuristic (chars/4) or backed by tiktoken /
 * a model-specific BPE encoder. Callers should always go through
 * `CodecRegistry.getCodec()` rather than instantiating codecs directly.
 */
export interface ProviderTokenCodec {
  /** Canonical provider key, e.g. "claude-sonnet", "gpt-4", "llama". */
  readonly providerId: string;

  /**
   * Estimate input tokens for a list of messages.
   * Must be synchronous — async token counting is handled at a higher layer
   * (countTokensAccurate in index.ts) where tiktoken is available.
   */
  estimateInput(messages: Message[]): number;

  /**
   * Estimate tokens for a single message (used for per-message binning in
   * the RecalibrationEngine).
   */
  encodeMessage(message: Message): number;

  /**
   * Incorporate a real (estimated, actual) feedback pair to tune the codec's
   * own internal correction factor.  Called by RecalibrationEngine after
   * receiving a live API response.
   *
   * @param estimated  The count this codec produced before the call.
   * @param actual     The count billed by the provider (billing-equivalent).
   * @param provider   Full or partial model slug (for context).
   */
  calibrate(estimated: number, actual: number, provider: string): void;

  /**
   * Current correction factor applied on top of the base estimate.
   * Starts at 1.0; RecalibrationEngine drives it via `calibrate()`.
   */
  readonly correctionFactor: number;
}

// ---------------------------------------------------------------------------
// Bin type — mirrors content-block categories the RecalibrationEngine tracks
// ---------------------------------------------------------------------------

export type MessageBin = "text" | "tool_result" | "tool_use" | "thinking" | "image" | "mixed";

/** Classify a message into a bin for per-bin error tracking. */
export function classifyMessage(message: Message): MessageBin {
  if (typeof message.content === "string") return "text";
  const types = new Set(message.content.map((b) => b.type));
  if (types.has("image_url")) return "image";
  if (types.has("thinking")) return "thinking";
  if (types.has("tool_use") && types.has("tool_result")) return "mixed";
  if (types.has("tool_use")) return "tool_use";
  if (types.has("tool_result")) return "tool_result";
  if (types.size === 1 && types.has("text")) return "text";
  return "mixed";
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function blockCharCount(block: ContentBlock): number {
  switch (block.type) {
    case "text":       return block.text.length;
    case "thinking":   return block.thinking.length;
    case "tool_use":   return block.name.length + JSON.stringify(block.input).length;
    case "tool_result":return block.content.length;
    case "image_url":  return 4000; // ~1000 tokens × 4 chars/token proxy
    default:           return 0;
  }
}

function charEstimate(messages: Message[], perMessageOverhead = 16): number {
  let chars = 0;
  for (const msg of messages) {
    chars += perMessageOverhead; // role + framing
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

function charEstimateMsg(message: Message, perMessageOverhead = 16): number {
  let chars = perMessageOverhead;
  if (typeof message.content === "string") {
    chars += message.content.length;
  } else {
    for (const block of message.content) {
      chars += blockCharCount(block);
    }
  }
  return Math.ceil(chars / 4);
}

// ---------------------------------------------------------------------------
// Base codec (shared calibrate logic; subclasses override baseMultiplier)
// ---------------------------------------------------------------------------

abstract class BaseCodec implements ProviderTokenCodec {
  abstract readonly providerId: string;

  /** Static nominal multiplier (mirrors MODEL_ENCODING_MAP). */
  protected abstract readonly nominalMultiplier: number;

  /** Live correction factor maintained by calibrate(). */
  private _correctionFactor = 1.0;
  get correctionFactor(): number { return this._correctionFactor; }

  /** EMA smoothing factor for calibrate() updates. */
  private static readonly EMA_ALPHA = 0.3;
  private static readonly FACTOR_MIN = 0.80;
  private static readonly FACTOR_MAX = 2.00;

  estimateInput(messages: Message[]): number {
    const base = charEstimate(messages);
    const effective = this.nominalMultiplier * this._correctionFactor;
    return effective === 1.0 ? base : Math.ceil(base * effective);
  }

  encodeMessage(message: Message): number {
    const base = charEstimateMsg(message);
    const effective = this.nominalMultiplier * this._correctionFactor;
    return effective === 1.0 ? base : Math.ceil(base * effective);
  }

  calibrate(estimated: number, actual: number, _provider: string): void {
    if (estimated <= 0 || actual <= 0) return;
    // Target factor: ratio of actual to estimated, normalised by nominal.
    const rawRatio = actual / estimated;
    const target = rawRatio / this.nominalMultiplier;
    // EMA blend toward target.
    const updated =
      this._correctionFactor +
      BaseCodec.EMA_ALPHA * (target - this._correctionFactor);
    // Clamp.
    this._correctionFactor = Math.max(
      BaseCodec.FACTOR_MIN,
      Math.min(BaseCodec.FACTOR_MAX, updated),
    );
  }

  /** Reset correction factor (for tests). */
  resetFactor(): void {
    this._correctionFactor = 1.0;
  }
}

// ---------------------------------------------------------------------------
// Provider-specific codec implementations
// ---------------------------------------------------------------------------

class ClaudeSonnetCodec extends BaseCodec {
  readonly providerId = "claude-sonnet";
  protected readonly nominalMultiplier = 1.0;
}

class ClaudeHaikuCodec extends BaseCodec {
  readonly providerId = "claude-haiku";
  protected readonly nominalMultiplier = 1.0;
}

class Gpt4Codec extends BaseCodec {
  readonly providerId = "gpt-4";
  protected readonly nominalMultiplier = 1.0;
}

class Gpt4oCodec extends BaseCodec {
  readonly providerId = "gpt-4o";
  protected readonly nominalMultiplier = 1.0;
}

class LlamaCodec extends BaseCodec {
  readonly providerId = "llama";
  protected readonly nominalMultiplier = 1.05;
}

class QwenCodec extends BaseCodec {
  readonly providerId = "qwen";
  protected readonly nominalMultiplier = 1.08;
}

class MistralCodec extends BaseCodec {
  readonly providerId = "mistral";
  protected readonly nominalMultiplier = 1.02;
}

class GeminiCodec extends BaseCodec {
  readonly providerId = "gemini";
  protected readonly nominalMultiplier = 1.0;
}

class GroqCodec extends BaseCodec {
  // Groq is an inference provider, not a model family.
  // Default to 1.0× (most Groq models are Llama / Mistral-based; callers
  // should detect the underlying model name separately).
  readonly providerId = "groq";
  protected readonly nominalMultiplier = 1.0;
}

class DefaultCodec extends BaseCodec {
  readonly providerId = "default";
  protected readonly nominalMultiplier = 1.1; // conservative safety margin
}

// ---------------------------------------------------------------------------
// CodecRegistry
// ---------------------------------------------------------------------------

/**
 * Registry that maps provider/model slugs to `ProviderTokenCodec` instances.
 *
 * All codec instances are singletons within a registry instance; the registry
 * is also a singleton (use `CodecRegistry.instance`).
 *
 * Resolution order:
 *   1. Exact `providerId` key (e.g. "gpt-4").
 *   2. Prefix scan: model slug starts with a known prefix.
 *   3. Substring scan: model slug contains a provider keyword.
 *   4. "default" fallback (cl100k_base + 1.1× margin).
 */
export class CodecRegistry {
  private static _instance: CodecRegistry | null = null;

  /** Singleton accessor. */
  static get instance(): CodecRegistry {
    if (!CodecRegistry._instance) {
      CodecRegistry._instance = new CodecRegistry();
    }
    return CodecRegistry._instance;
  }

  /** Replace the singleton (used in tests). */
  static resetInstance(): void {
    CodecRegistry._instance = null;
  }

  private readonly _codecs = new Map<string, ProviderTokenCodec>();

  /** Ordered prefix → codec mapping for model-slug resolution. */
  private readonly _prefixMap: Array<[string, ProviderTokenCodec]>;

  /** Ordered keyword → codec mapping for substring resolution. */
  private readonly _keywordMap: Array<[string, ProviderTokenCodec]>;

  private constructor() {
    // Instantiate built-in codecs.
    const claudeSonnet = new ClaudeSonnetCodec();
    const claudeHaiku  = new ClaudeHaikuCodec();
    const gpt4         = new Gpt4Codec();
    const gpt4o        = new Gpt4oCodec();
    const llama        = new LlamaCodec();
    const qwen         = new QwenCodec();
    const mistral      = new MistralCodec();
    const gemini       = new GeminiCodec();
    const groq         = new GroqCodec();
    const def          = new DefaultCodec();

    // Register by providerId for direct lookup.
    for (const codec of [claudeSonnet, claudeHaiku, gpt4, gpt4o, llama, qwen, mistral, gemini, groq, def]) {
      this._codecs.set(codec.providerId, codec);
    }

    // Prefix map (longest / most-specific first).
    this._prefixMap = [
      ["claude-3-5",      claudeSonnet],
      ["claude-3-opus",   claudeSonnet],
      ["claude-3-sonnet", claudeSonnet],
      ["claude-3-haiku",  claudeHaiku],
      ["claude-3",        claudeSonnet],
      ["claude-sonnet",   claudeSonnet],
      ["claude-haiku",    claudeHaiku],
      ["claude-2",        claudeSonnet],
      ["claude-1",        claudeSonnet],
      ["claude-instant",  claudeSonnet],
      ["gpt-4o",          gpt4o],
      ["gpt-4",           gpt4],
      ["gpt-3.5",         gpt4],
      ["o1",              gpt4o],
      ["o3",              gpt4o],
      ["llama3",          llama],
      ["llama2",          llama],
      ["llama-",          llama],
      ["meta-llama",      llama],
      ["codellama",       llama],
      ["qwen",            qwen],
      ["mistral",         mistral],
      ["mixtral",         mistral],
      ["gemini",          gemini],
      ["groq",            groq],
    ];

    // Keyword map for substring matching (fallback).
    this._keywordMap = [
      ["claude",   claudeSonnet],
      ["gpt",      gpt4],
      ["llama",    llama],
      ["qwen",     qwen],
      ["gemini",   gemini],
      ["mistral",  mistral],
      ["mixtral",  mistral],
      ["groq",     groq],
    ];
  }

  /**
   * Resolve the best codec for a model/provider slug.
   * Never returns null — falls back to "default".
   */
  getCodec(modelOrProvider: string): ProviderTokenCodec {
    // 1. Direct providerId lookup.
    const direct = this._codecs.get(modelOrProvider);
    if (direct) return direct;

    const lower = modelOrProvider.toLowerCase();

    // Also try direct lookup on lowercase.
    const lowerDirect = this._codecs.get(lower);
    if (lowerDirect) return lowerDirect;

    // 2. Prefix scan.
    for (const [prefix, codec] of this._prefixMap) {
      if (lower.startsWith(prefix)) return codec;
    }

    // 3. Substring scan.
    for (const [keyword, codec] of this._keywordMap) {
      if (lower.includes(keyword)) return codec;
    }

    // 4. Default.
    return this._codecs.get("default")!;
  }

  /**
   * Register a custom codec (for extension / testing).
   * Also inserts it at the front of the prefix map if `prefix` is provided.
   */
  register(codec: ProviderTokenCodec, prefix?: string): void {
    this._codecs.set(codec.providerId, codec);
    if (prefix) {
      this._prefixMap.unshift([prefix.toLowerCase(), codec]);
    }
  }

  /**
   * Auto-detect the provider from a raw API response object.
   *
   * Inspects common fields that providers include in their response metadata:
   *   - `model` field (present in OpenAI, Anthropic, Groq responses)
   *   - `object` field (OpenAI uses "chat.completion")
   *   - `provider` field (some proxies like OpenRouter)
   *
   * Returns the resolved codec (never null — falls back to default).
   */
  detectFromResponse(response: Record<string, unknown>): ProviderTokenCodec {
    const model =
      typeof response["model"] === "string" ? response["model"] :
      typeof response["provider"] === "string" ? response["provider"] :
      undefined;

    if (model) return this.getCodec(model);
    return this._codecs.get("default")!;
  }

  /** List all registered provider IDs. */
  registeredProviders(): string[] {
    return Array.from(this._codecs.keys());
  }
}
