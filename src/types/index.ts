/**
 * Shared types — mirror ashlrcode/src/providers/types.ts so consumers of
 * @ashlr/core-efficiency don't need to import the concrete ProviderRouter.
 *
 * Kept deliberately minimal. Extend sparingly; anything more exotic
 * belongs in ashlrcode's provider layer, not here.
 */

export interface Message {
  role: "user" | "assistant" | "tool";
  content: string | ContentBlock[];
}

export type ContentBlock =
  | { type: "text"; text: string }
  | { type: "thinking"; thinking: string; signature?: string }
  | { type: "image_url"; image_url: { url: string } }
  | { type: "tool_use"; id: string; name: string; input: Record<string, unknown> }
  | { type: "tool_result"; tool_use_id: string; content: string; is_error?: boolean };

export interface ToolDefinition {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

export type StopReason = "end_turn" | "tool_use" | "max_tokens";

export interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
  costTicks?: number;
  reasoningTokens?: number;
}

/**
 * Metadata describing a cache block attached to a prompt.
 * `cache_type: 'ephemeral'` — cleared at session end (default Anthropic behavior).
 * `cache_type: 'static'`    — persists across sessions (longer TTL, higher write cost).
 * `created_token_count`     — token count at cache-write time (optional, from API response).
 */
export interface CacheBlockMetadata {
  cache_type: "ephemeral" | "static";
  created_token_count?: number;
}

/**
 * Subset of the Anthropic `Message` response shape relevant to token accounting.
 * Mirrors `Anthropic.Message` so callers can pass the raw SDK response without
 * importing the SDK here.
 */
export interface MessageResponse {
  /** Normal (non-cached) input tokens billed at 1× rate. */
  input_tokens: number;
  /** Tokens written into the cache (billed at 1.25× rate). */
  cache_creation_input_tokens?: number;
  /** Tokens read from the cache (billed at 0.1× rate). */
  cache_read_input_tokens?: number;
  /** Output tokens generated. */
  output_tokens: number;
  /** Stop reason from the model. */
  stop_reason?: string | null;
}

export interface StreamEvent {
  type:
    | "text_delta"
    | "thinking_delta"
    | "tool_call_start"
    | "tool_call_delta"
    | "tool_call_end"
    | "message_end"
    | "usage";
  text?: string;
  signature?: string;
  toolCall?: { id?: string; name?: string; input?: Record<string, unknown> };
  stopReason?: StopReason;
  usage?: TokenUsage;
}

export interface ProviderRequest {
  systemPrompt: string;
  messages: Message[];
  tools: ToolDefinition[];
  maxTokens?: number;
}

/**
 * Minimal streaming LLM contract. ProviderRouter in ashlrcode structurally
 * satisfies this via matching `stream(ProviderRequest): AsyncGenerator<StreamEvent>`.
 * Lets compression/autoCompact and genome/scribe consolidate without
 * pulling the concrete router into core-efficiency.
 */
export interface LLMSummarizer {
  stream(request: ProviderRequest): AsyncGenerator<StreamEvent>;
}
