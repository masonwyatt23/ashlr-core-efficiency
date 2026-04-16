/**
 * cacheBreakpoints — add `cache_control: { type: 'ephemeral' }` at the
 * logical static/dynamic boundaries of an Anthropic Messages request.
 *
 * Reference: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
 *
 * Anthropic allows up to 4 cache breakpoints per request. This helper places
 * them on the high-leverage boundaries in order:
 *   1. Tool definitions (tools are static across a session).
 *   2. System prompt (genome + instructions rarely change mid-session).
 *   3. The last block of static user content (if a message is marked static).
 *   4. The boundary between the last static and first dynamic message.
 *
 * Callers pass a plain object with whichever of `system`, `tools`, `messages`
 * apply to their request. The function returns a new object with cache
 * markers added — input is not mutated.
 */

// ---------------------------------------------------------------------------
// Structural types
//
// We deliberately avoid importing from @anthropic-ai/sdk so this module stays
// peer-dep-free. The shapes below mirror the SDK's `MessageCreateParams`
// surface area we actually touch.
// ---------------------------------------------------------------------------

interface EphemeralCacheControl {
  type: "ephemeral";
}

interface Cacheable {
  cache_control?: EphemeralCacheControl | null;
}

/** A single content block inside a user/assistant message. */
export type CacheableContentBlock = Cacheable & {
  type: string;
  [key: string]: unknown;
};

export interface CacheableMessage {
  role: "user" | "assistant";
  content: string | CacheableContentBlock[];
  /**
   * Optional hint: when true, this message is treated as part of the
   * static prefix and a cache marker is placed on its final content block.
   * The last `static: true` message in the array gets the boundary marker.
   */
  cache?: boolean;
}

export interface CacheableTool extends Cacheable {
  name: string;
  description?: string;
  input_schema: unknown;
}

export type CacheableSystem =
  | string
  | Array<
      Cacheable & {
        type: "text";
        text: string;
      }
    >;

export interface CacheableRequest {
  system?: CacheableSystem;
  tools?: CacheableTool[];
  messages: CacheableMessage[];
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

const EPHEMERAL: EphemeralCacheControl = { type: "ephemeral" };

/** Mark the last text block of a system prompt as cached. */
function cacheSystem(system: CacheableSystem | undefined): CacheableSystem | undefined {
  if (system === undefined) return undefined;
  if (typeof system === "string") {
    // Promote string → blocks so we can attach cache_control.
    return [{ type: "text", text: system, cache_control: EPHEMERAL }];
  }
  if (system.length === 0) return system;
  return system.map((block, i) =>
    i === system.length - 1 ? { ...block, cache_control: EPHEMERAL } : { ...block },
  );
}

/** Mark the last tool as cached (covers the full tool-definition list). */
function cacheTools(tools: CacheableTool[] | undefined): CacheableTool[] | undefined {
  if (!tools || tools.length === 0) return tools;
  return tools.map((tool, i) =>
    i === tools.length - 1 ? { ...tool, cache_control: EPHEMERAL } : { ...tool },
  );
}

/**
 * Attach `cache_control` to the last content block of the given message,
 * promoting a string body to a `text` block when needed.
 */
function cacheMessageTail(msg: CacheableMessage): CacheableMessage {
  if (typeof msg.content === "string") {
    return {
      ...msg,
      content: [{ type: "text", text: msg.content, cache_control: EPHEMERAL }],
    };
  }
  if (msg.content.length === 0) return msg;
  const content = msg.content.map((b, i) =>
    i === msg.content.length - 1 ? { ...b, cache_control: EPHEMERAL } : { ...b },
  );
  return { ...msg, content };
}

/**
 * Apply the static-message boundary cache marker.
 *
 * Strategy: if any message has `cache: true`, mark the *last* such message
 * (explicit opt-in wins). Otherwise, do nothing — callers without static
 * content should lean on system+tools caching.
 */
function cacheMessages(messages: CacheableMessage[]): CacheableMessage[] {
  if (messages.length === 0) return messages;
  let lastStaticIdx = -1;
  for (let i = 0; i < messages.length; i++) {
    if (messages[i]?.cache) lastStaticIdx = i;
  }
  if (lastStaticIdx === -1) return messages.map((m) => ({ ...m }));
  return messages.map((m, i) => {
    if (i !== lastStaticIdx) return { ...m };
    const cached = cacheMessageTail(m);
    // Strip the meta `cache` flag before it leaves our hands — the SDK
    // doesn't know about it.
    const { cache: _cache, ...clean } = cached;
    return clean;
  });
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Add prompt-caching breakpoints at the static→dynamic boundaries of a
 * request. Input is not mutated.
 *
 * - Strings passed as `system` are promoted to a text-block array so a
 *   cache marker can be attached.
 * - Messages flagged with `cache: true` get a marker on their last content
 *   block; the last such flagged message forms the static prefix boundary.
 *
 * @example
 * ```ts
 * const req = cacheBreakpoints({
 *   system: systemPrompt,
 *   tools: mcpTools,
 *   messages: [
 *     { role: 'user', content: longProjectContext, cache: true },
 *     { role: 'user', content: 'actual question here' },
 *   ],
 * });
 * await client.messages.create({ ...req, model: 'claude-opus-4-5', max_tokens: 1024 });
 * ```
 */
export function cacheBreakpoints<T extends CacheableRequest>(req: T): T {
  return {
    ...req,
    system: cacheSystem(req.system),
    tools: cacheTools(req.tools),
    messages: cacheMessages(req.messages),
  };
}

/**
 * Variant that accepts just a message array (the common case when `system`
 * and `tools` are configured elsewhere). Returns a new array.
 */
export function cacheMessagesBreakpoints(messages: CacheableMessage[]): CacheableMessage[] {
  return cacheMessages(messages);
}
