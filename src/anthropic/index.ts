/**
 * @ashlr/core-efficiency/anthropic — Anthropic SDK integration surface.
 *
 * Three helpers that let any caller using `@anthropic-ai/sdk` (Messages API)
 * or `@anthropic-ai/claude-agent-sdk` (stdio MCP) get ashlr's tools, genome
 * RAG, and prompt caching in 2-3 lines:
 *
 *   - `ashlrMcpConfig()`  — auto-detect the installed ashlr-plugin and
 *                           return a list of stdio MCP server configs.
 *   - `withGenome()`      — prepend project genome context to a system prompt.
 *   - `cacheBreakpoints()` — insert ephemeral cache markers at static/dynamic
 *                            boundaries of a Messages request.
 *
 * All helpers are runtime-agnostic — `@anthropic-ai/sdk` is a peer dependency
 * and is never imported by this module directly.
 */

export {
  type AshlrMcpConfigOptions,
  type AshlrMcpStdioServer,
  type AshlrPlugin,
  AshlrPluginNotFoundError,
  ashlrMcpConfig,
  ashlrMcpConfigRecord,
  resolvePluginRoot,
} from "./mcp-config.ts";

export { type WithGenomeOptions, withGenome } from "./genome-tools.ts";

export {
  type CacheableContentBlock,
  type CacheableMessage,
  type CacheableRequest,
  type CacheableSystem,
  type CacheableTool,
  cacheBreakpoints,
  cacheMessagesBreakpoints,
} from "./prompt-cache.ts";
