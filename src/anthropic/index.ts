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
  type CacheBreakpointsOptions,
  cacheBreakpoints,
  cacheMessagesBreakpoints,
} from "./prompt-cache.ts";

export {
  CACHE_AUDIT_DIR,
  CACHE_AUDIT_FILE,
  type CacheStrategy,
  type CacheAuditEntry,
  type CacheAuditSummary,
  cacheAuditPath,
  computeEffectiveCostMicroUsd,
  computeBaselineCostMicroUsd,
  appendCacheAudit,
  loadCacheAudit,
  summarizeAuditWindow,
} from "./cache-audit.ts";

export {
  CacheOptimizer,
  resolveModelPricing,
  type CacheOptimizerOptions,
  type StrategyROI,
  type ProviderSwitchRecommendation,
  type CacheOptimizationReport,
} from "./cache-optimizer.ts";

export {
  CacheBreakpointOptimizer,
  recommendBreakpoints,
  MAX_BREAKPOINTS,
  type Message,
  type OptimizerOptions,
  type BreakpointROI,
  type BreakpointRecommendation,
  type BreakpointEvolutionRecord,
} from "./cache-breakpoint-optimizer.ts";

export {
  MultiBreakpointOrchestrator,
  recommendCacheStrategy,
  type TranscriptTurn,
  type TurnTier,
  type MultiBreakpointStrategy,
  type CacheStrategy_MultiBreakpoint,
  type OrchestratorOptions,
  type OrchestratorAuditRecord,
} from "./multi-breakpoint-orchestrator.ts";

export {
  MultiProviderAdapter,
  recommendProviderCacheStrategy,
  resolveProviderStrategy,
  normalizeProviderKey,
  PROVIDER_CACHE_STRATEGIES,
  type ProviderCacheStrategy,
  type MultiProviderAdapterOptions,
  type CacheAdapterResult,
  type CacheDegradation,
  type AdapterOutcome,
} from "./multi-provider-adapter.ts";
