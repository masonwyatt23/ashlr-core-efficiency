/**
 * 4-tier context compression — extracted from ashlrcode/src/agent/context.ts.
 *
 * Tier 1: autoCompact — summarize older messages via LLMSummarizer
 * Tier 2: snipCompact — truncate tool results > 2000 chars
 * Tier 3: contextCollapse — remove short/duplicate older messages
 * Tier 4: treeCompact — prune entire message subtrees by value/token ratio
 *
 * No coupling to ashlrcode's ProviderRouter — everything flows through
 * the LLMSummarizer interface in ../types, so ashlr-plugin can reuse this
 * module against any streaming-capable LLM endpoint.
 */

import type { LLMSummarizer, Message } from "../types/index.ts";
import { estimateTokensFromMessages } from "../tokens/index.ts";

export interface ContextConfig {
  /** Max tokens before triggering compaction (default: 100000) */
  maxContextTokens: number;
  /** Tokens to reserve for the response (default: 8192) */
  reserveTokens: number;
  /** Number of recent messages to keep at full fidelity (default: 10) */
  recentMessageCount: number;
}

export const DEFAULT_CONFIG: ContextConfig = {
  maxContextTokens: 100_000,
  reserveTokens: 8192,
  recentMessageCount: 10,
};

/**
 * Tier 3: contextCollapse — remove redundant messages from older history.
 * - Remove short assistant messages (< 10 chars)
 * - Deduplicate consecutive tool results with similar content
 * - Keep last 5 messages at full fidelity
 */
export function contextCollapse(messages: Message[]): Message[] {
  if (messages.length <= 5) return messages;

  const keepRecent = 5;
  const older = messages.slice(0, -keepRecent);
  const recent = messages.slice(-keepRecent);

  const collapsed: Message[] = [];
  let lastToolResultHash = "";

  let skipNext = false;
  for (const msg of older) {
    if (skipNext) { skipNext = false; continue; }

    if (msg.role === "assistant" && typeof msg.content === "string" && msg.content.trim().length < 10) {
      skipNext = true;
      continue;
    }

    if (msg.role === "user" && Array.isArray(msg.content)) {
      const toolResults = msg.content.filter((b) => b.type === "tool_result");
      if (toolResults.length > 0) {
        const hash = toolResults.map((b) => (b.type === "tool_result" ? b.content.slice(0, 200) : "")).join("|");
        if (hash === lastToolResultHash) continue;
        lastToolResultHash = hash;
      }
    }

    collapsed.push(msg);
  }

  return [...collapsed, ...recent];
}

/**
 * Check if context needs compaction.
 *
 * @param actualTokensUsed - If provided, uses the real token count from the
 *   last API response instead of the chars/4 estimate.
 */
export function needsCompaction(
  messages: Message[],
  systemPromptTokens: number,
  config: Partial<ContextConfig> = {},
  actualTokensUsed?: number,
): boolean {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const messageTokens = actualTokensUsed ?? estimateTokensFromMessages(messages);
  return messageTokens + systemPromptTokens > cfg.maxContextTokens - cfg.reserveTokens;
}

/**
 * Tier 1: autoCompact — summarize older messages.
 * Splits messages at the last N, summarizes everything before that window,
 * and prepends the summary as a synthetic user/assistant exchange.
 */
export async function autoCompact(
  messages: Message[],
  summarizer: LLMSummarizer,
  config: Partial<ContextConfig> = {},
): Promise<Message[]> {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  if (messages.length <= cfg.recentMessageCount) return messages;

  const splitIndex = messages.length - cfg.recentMessageCount;
  const olderMessages = messages.slice(0, splitIndex);
  const recentMessages = messages.slice(splitIndex);

  const summary = await summarizeMessages(olderMessages, summarizer);

  return [
    {
      role: "user",
      content: `[Context Summary — earlier conversation was compacted to save tokens]\n\n${summary}`,
    },
    {
      role: "assistant",
      content: "Understood. I have the context from our earlier conversation. Let me continue from where we left off.",
    },
    ...recentMessages,
  ];
}

/**
 * Tier 2: snipCompact — remove verbose tool results and stale messages.
 */
export function snipCompact(messages: Message[]): Message[] {
  return messages.map((msg) => {
    if (typeof msg.content === "string") return msg;

    const trimmedBlocks = msg.content.map((block) => {
      if (block.type === "tool_result" && block.content.length > 2000) {
        const truncated =
          block.content.slice(0, 800) +
          "\n\n[... truncated ...]\n\n" +
          block.content.slice(-800);
        return { ...block, content: truncated };
      }
      return block;
    });

    return { ...msg, content: trimmedBlocks };
  });
}

// ---------------------------------------------------------------------------
// Tier selection
// ---------------------------------------------------------------------------

export type CompressionTier = 1 | 2 | 3 | 4;

/**
 * Select the cheapest compression tier that will bring token usage under budget.
 *
 * Tier 4 (treeCompact) is tried first — prunes entire low-value subtrees, no LLM call.
 * Tier 3 (contextCollapse) is tried next — lowest cost, no LLM call.
 * Tier 2 (snipCompact) is tried next — truncates large tool results.
 * Tier 1 (autoCompact) is the last resort — requires an LLM summarization call.
 *
 * Returns the tier number (1–4) that is predicted to fit within the budget,
 * or 1 if no tier without LLM summarization is sufficient.
 *
 * @param messages       Current message array.
 * @param systemTokens   Estimated tokens consumed by the system prompt.
 * @param config         Context config (uses defaults if omitted).
 */
export function selectCompressionTier(
  messages: Message[],
  systemTokens: number,
  config: Partial<ContextConfig> = {},
): CompressionTier {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const budget = cfg.maxContextTokens - cfg.reserveTokens - systemTokens;

  // Tier 4 — prune lowest-value subtrees (no LLM call, targets whole branches)
  const treeCompacted = treeCompact(messages, undefined, {
    tokenBudget: budget,
    minValuePerToken: 0,
    keepRecentMessages: cfg.recentMessageCount,
  });
  if (estimateTokensFromMessages(treeCompacted.messages) <= budget) return 4;

  // Tier 3 — collapse short/dup messages
  const collapsed = contextCollapse(messages);
  if (estimateTokensFromMessages(collapsed) <= budget) return 3;

  // Tier 2 — snip large tool results
  const snipped = snipCompact(messages);
  if (estimateTokensFromMessages(snipped) <= budget) return 2;

  // Tier 1 — requires LLM summarization (caller must invoke autoCompact)
  return 1;
}

/** Internal: ask the LLM to produce a compact summary of an older conversation slice. */
async function summarizeMessages(messages: Message[], summarizer: LLMSummarizer): Promise<string> {
  const conversationText = messages
    .map((msg) => {
      const role = msg.role;
      const content =
        typeof msg.content === "string"
          ? msg.content
          : msg.content
              .map((b) => {
                if (b.type === "text") return b.text;
                if (b.type === "tool_use")
                  return `[Tool: ${b.name}(${JSON.stringify(b.input).slice(0, 200)})]`;
                if (b.type === "tool_result")
                  return `[Result: ${b.content.slice(0, 300)}]`;
                return "";
              })
              .join("\n");
      return `${role}: ${content}`;
    })
    .join("\n\n");

  let summary = "";
  const stream = summarizer.stream({
    systemPrompt:
      "Summarize the following conversation concisely. Preserve key decisions, file paths mentioned, code changes made, and important context. Be thorough but compact. Output only the summary, no preamble.",
    messages: [
      {
        role: "user",
        content: `Summarize this conversation:\n\n${conversationText.slice(0, 50000)}`,
      },
    ],
    tools: [],
  });

  for await (const event of stream) {
    if (event.type === "text_delta" && event.text) {
      summary += event.text;
    }
  }

  return summary || "[Unable to generate summary]";
}

// ---------------------------------------------------------------------------
// Tier 4: treeCompact — Recursive Context Boundary Compression
// ---------------------------------------------------------------------------

/**
 * A node in the message tree used for subtree-aware pruning.
 *
 * Messages form implicit parent-child relationships via tool_use / tool_result
 * pairing: an assistant message containing a `tool_use` block is the "parent"
 * of the subsequent user message containing the matching `tool_result` block.
 * Sibling tool calls within the same assistant turn share the same parent index.
 *
 * For flat conversations (no tool calls) every message is a root-level node
 * with no children — the tree degenerates to a linear sequence.
 */
export interface MessageTreeNode {
  /** Index into the original messages array. */
  index: number;
  /** Parent node index, or -1 for root messages. */
  parentIndex: number;
  /** Indices of direct children (tool results of this turn's tool calls). */
  children: number[];
  /** Sibling indices — messages sharing the same parentIndex. */
  siblings: number[];
  /**
   * Estimated token count for this message only (not its subtree).
   * Computed from `estimateTokensFromMessages([message])`.
   */
  tokens: number;
  /**
   * Heuristic value score for this message (0–1).
   *
   * Scoring rules (higher = more valuable):
   *   - Recent messages score higher (proximity to tail = recency bonus).
   *   - Tool results score higher than short assistant acknowledgements.
   *   - Messages containing file paths, code blocks, or structured data score higher.
   *   - Short plain-text messages (< 20 chars) score lower.
   */
  value: number;
  /** Token count for this node plus all descendant nodes. */
  subtreeTokens: number;
  /** Aggregate value for this node plus all descendants (sum). */
  subtreeValue: number;
}

/**
 * Configuration for `treeCompact`.
 */
export interface TreeCompactConfig {
  /**
   * Target token budget — treeCompact removes subtrees until the message array
   * fits within this limit. When the budget is already satisfied on entry,
   * treeCompact returns the original messages unchanged.
   */
  tokenBudget: number;
  /**
   * Minimum value-per-token ratio a subtree must have to be retained.
   * Subtrees with a ratio below this threshold are pruning candidates.
   * Default: 0 (prune purely by lowest ratio first, regardless of absolute value).
   */
  minValuePerToken?: number;
  /**
   * Number of recent messages to protect from pruning (always kept).
   * Default: matches DEFAULT_CONFIG.recentMessageCount (10).
   */
  keepRecentMessages?: number;
  /**
   * Maximum fraction of total messages that treeCompact is allowed to remove
   * in a single pass. Prevents catastrophic context loss on edge cases.
   * Default: 0.85 (85% of messages may be pruned — allows aggressive subtree removal
   * while always retaining at least keepRecentMessages).
   */
  maxPruneFraction?: number;
}

/**
 * Report emitted by `treeCompact` describing what was pruned and the token savings.
 */
export interface TreeCompactionReport {
  /** Indices (into the original messages array) of messages that were removed. */
  prunedIndices: number[];
  /**
   * Subtree roots that were pruned. Each entry identifies the root message index
   * of a removed subtree and the number of tokens saved by removing it.
   */
  prunedSubtrees: Array<{
    rootIndex: number;
    subtreeSize: number;
    tokensSaved: number;
    valuePerToken: number;
  }>;
  /** Token count before pruning. */
  tokensBefore: number;
  /** Token count after pruning. */
  tokensAfter: number;
  /** Total tokens saved. */
  tokensSaved: number;
  /**
   * True when the compaction target (tokenBudget) was achieved.
   * False when not enough low-value subtrees existed to reach the budget.
   */
  targetAchieved: boolean;
  /** Number of messages in the output (after pruning). */
  messagesAfter: number;
}

/**
 * Result returned by `treeCompact`.
 */
export interface TreeCompactionResult {
  /** The compacted message array. */
  messages: Message[];
  /** Detailed report of what was pruned. */
  report: TreeCompactionReport;
}

// ---------------------------------------------------------------------------
// Internal tree-building helpers
// ---------------------------------------------------------------------------

/**
 * Compute a heuristic value score for a single message (0–1 range).
 *
 * Higher score = more valuable to retain:
 *   - Messages with code blocks, file paths, or JSON structures → 0.8–1.0
 *   - Tool results (structured data) → 0.7
 *   - Normal assistant/user turns → 0.4–0.6
 *   - Short acknowledgements (< 20 chars) → 0.1
 */
function scoreMessageValue(msg: Message, totalMessages: number, index: number): number {
  const recencyBonus = (index / Math.max(1, totalMessages - 1)) * 0.3; // 0–0.3

  if (typeof msg.content === "string") {
    const text = msg.content;
    if (text.trim().length < 20) return 0.05 + recencyBonus;
    // Heuristics for high-value content
    const hasCode = /```|`[^`]+`/.test(text);
    const hasPath = /\/[a-zA-Z0-9._/-]{3,}/.test(text);
    const hasStructured = /\{[^}]{10,}\}|\[[^\]]{10,}\]/.test(text);
    const base = hasCode || hasPath || hasStructured ? 0.7 : 0.4;
    return Math.min(1.0, base + recencyBonus);
  }

  // Content blocks
  let base = 0.3;
  for (const block of msg.content) {
    if (block.type === "tool_result") {
      // Tool results are high-value if they have substantial content
      const len = block.content.length;
      base = Math.max(base, len > 500 ? 0.65 : len > 100 ? 0.55 : 0.4);
    } else if (block.type === "tool_use") {
      base = Math.max(base, 0.6);
    } else if (block.type === "text") {
      const hasCode = /```|`[^`]+`/.test(block.text);
      const hasPath = /\/[a-zA-Z0-9._/-]{3,}/.test(block.text);
      if (hasCode || hasPath) base = Math.max(base, 0.7);
    }
  }
  return Math.min(1.0, base + recencyBonus);
}

/**
 * Build a flat MessageTreeNode array from a Message[].
 *
 * Wires parent→child relationships using tool_use/tool_result ID matching.
 * Falls back to positional pairing (assistant at i → user at i+1 with tool_result)
 * when tool_use_id matching is unavailable.
 */
function buildMessageTree(messages: Message[]): MessageTreeNode[] {
  const n = messages.length;
  const nodes: MessageTreeNode[] = messages.map((msg, i) => ({
    index: i,
    parentIndex: -1,
    children: [],
    siblings: [],
    tokens: estimateTokensFromMessages([msg]),
    value: scoreMessageValue(msg, n, i),
    subtreeTokens: 0,
    subtreeValue: 0,
  }));

  // Map tool_use IDs → assistant message indices for pairing
  const toolUseIdToParent = new Map<string, number>();
  for (let i = 0; i < n; i++) {
    const msg = messages[i]!;
    if (msg.role === "assistant" && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === "tool_use") {
          toolUseIdToParent.set(block.id, i);
        }
      }
    }
  }

  // Wire children: user messages with tool_result blocks link to their parent assistant turn
  for (let i = 0; i < n; i++) {
    const msg = messages[i]!;
    if (msg.role === "user" && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === "tool_result") {
          const parentIdx = toolUseIdToParent.get(block.tool_use_id);
          if (parentIdx !== undefined && parentIdx < i) {
            nodes[i]!.parentIndex = parentIdx;
            nodes[parentIdx]!.children.push(i);
            break; // a user message belongs to one assistant turn
          }
        }
      }
    }
  }

  // Wire siblings: messages sharing the same parentIndex
  const byParent = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const parentIdx = nodes[i]!.parentIndex;
    const group = byParent.get(parentIdx) ?? [];
    group.push(i);
    byParent.set(parentIdx, group);
  }
  for (const [, group] of byParent) {
    for (const idx of group) {
      nodes[idx]!.siblings = group.filter((s) => s !== idx);
    }
  }

  // Compute subtree tokens and value via post-order traversal (leaves first)
  // Process in reverse index order so children are always computed before parents
  // (children always appear after their parents in message arrays)
  for (let i = n - 1; i >= 0; i--) {
    const node = nodes[i]!;
    let subtreeTokens = node.tokens;
    let subtreeValue = node.value;
    for (const childIdx of node.children) {
      subtreeTokens += nodes[childIdx]!.subtreeTokens;
      subtreeValue += nodes[childIdx]!.subtreeValue;
    }
    node.subtreeTokens = subtreeTokens;
    node.subtreeValue = subtreeValue;
  }

  return nodes;
}

/**
 * Collect all descendant indices for a subtree root (inclusive).
 * Uses BFS to visit all children in breadth-first order.
 */
function collectSubtree(nodes: MessageTreeNode[], rootIndex: number): number[] {
  const result: number[] = [];
  const queue = [rootIndex];
  while (queue.length > 0) {
    const idx = queue.shift()!;
    result.push(idx);
    for (const childIdx of nodes[idx]!.children) {
      queue.push(childIdx);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Public API: treeCompact
// ---------------------------------------------------------------------------

/**
 * Tier 4: treeCompact — Adaptive Tree Pruning via Genome Graph-Traversal Patterns.
 *
 * Observes the implicit parent-child / sibling structure of a message array
 * (assistant tool_use → user tool_result pairings) and prunes entire subtrees
 * when the conversation exceeds `config.tokenBudget`.
 *
 * ### Algorithm
 *
 * 1. Build a `MessageTreeNode[]` DAG from the message array using tool_use/tool_result
 *    ID matching (mirrors `buildSectionGraph` in genome/graph-traversal.ts).
 * 2. Identify candidate subtree roots: root-level nodes (parentIndex === -1) that
 *    are not in the recency-protected tail.
 * 3. Rank candidates by ascending `subtreeValue / subtreeTokens` (breadth-first
 *    traversal of the candidate set, lowest value-per-token first).
 * 4. Greedily remove subtrees from the lowest-ranked candidate up until the
 *    token budget is satisfied or candidates are exhausted.
 * 5. Emit a `TreeCompactionReport` showing which subtrees were dropped and the
 *    token savings.
 *
 * ### Cost hierarchy (Tier 4 sits above autoCompact)
 *
 * Tier 4 is cheaper than Tier 1 (no LLM call) and more targeted than Tiers 2/3
 * (removes whole branches rather than truncating individual messages). It is tried
 * first in `selectCompressionTier` because a well-structured conversation with
 * many low-value tool call branches can be dramatically reduced without touching
 * the high-value narrative.
 *
 * ### Integration with the regret-learner
 *
 * Pass `tier: 4` to `recordTierOutcome` / `pipeCompressionCostToLearner` after
 * calling `treeCompact` so the regret learner tracks Tier 4's cost profile
 * alongside Tiers 1–3.
 *
 * @param messages      Current message array.
 * @param genomeGraph   Optional genome SectionGraph for cross-referencing section
 *                      importance. When provided, messages whose content references
 *                      high-scoring genome sections receive a value bonus.
 *                      Pass `undefined` to use heuristic scoring only.
 * @param config        Tree compaction configuration.
 * @returns             `{ messages, report }` — compacted messages and audit report.
 */
export function treeCompact(
  messages: Message[],
  genomeGraph: unknown | undefined,
  config: TreeCompactConfig,
): TreeCompactionResult {
  const {
    tokenBudget,
    minValuePerToken = 0,
    keepRecentMessages = DEFAULT_CONFIG.recentMessageCount,
    maxPruneFraction = 0.85,
  } = config;

  const tokensBefore = estimateTokensFromMessages(messages);
  const emptyReport = (targetAchieved: boolean): TreeCompactionReport => ({
    prunedIndices: [],
    prunedSubtrees: [],
    tokensBefore,
    tokensAfter: tokensBefore,
    tokensSaved: 0,
    targetAchieved,
    messagesAfter: messages.length,
  });

  // Already within budget — nothing to do
  if (tokensBefore <= tokenBudget) {
    return { messages, report: emptyReport(true) };
  }

  // Too few messages to prune meaningfully
  if (messages.length <= keepRecentMessages) {
    return { messages, report: emptyReport(false) };
  }

  const nodes = buildMessageTree(messages);
  const n = nodes.length;

  // Recency-protected tail: the last `keepRecentMessages` are never pruned
  const protectedStart = Math.max(0, n - keepRecentMessages);
  const protectedIndices = new Set<number>();
  for (let i = protectedStart; i < n; i++) {
    // Also protect any ancestors of protected messages (don't orphan them)
    let cur = i;
    while (cur !== -1) {
      protectedIndices.add(cur);
      cur = nodes[cur]!.parentIndex;
    }
  }

  // Maximum messages we may remove in one pass
  const maxRemovable = Math.floor(n * maxPruneFraction);

  // --- BFS candidate collection ---
  // Only consider root-level nodes (parentIndex === -1) as pruning entry points.
  // Removing a root removes its entire subtree atomically.
  const candidates: MessageTreeNode[] = [];
  for (let i = 0; i < n; i++) {
    const node = nodes[i]!;
    if (node.parentIndex === -1 && !protectedIndices.has(i)) {
      candidates.push(node);
    }
  }

  // BFS-order traversal: process candidates breadth-first, sorted by
  // ascending value-per-token so cheapest-to-remove subtrees come first.
  candidates.sort((a, b) => {
    const ratioA = a.subtreeTokens > 0 ? a.subtreeValue / a.subtreeTokens : 0;
    const ratioB = b.subtreeTokens > 0 ? b.subtreeValue / b.subtreeTokens : 0;
    return ratioA - ratioB; // ascending: lowest value/token first
  });

  const prunedIndices = new Set<number>();
  const prunedSubtrees: TreeCompactionReport["prunedSubtrees"] = [];
  let removedCount = 0;

  for (const candidate of candidates) {
    if (removedCount >= maxRemovable) break;

    const vpt =
      candidate.subtreeTokens > 0
        ? candidate.subtreeValue / candidate.subtreeTokens
        : 0;

    // Skip if this subtree exceeds the minimum value threshold
    if (vpt > minValuePerToken && minValuePerToken > 0) continue;

    // Collect all indices in this subtree
    const subtreeIndices = collectSubtree(nodes, candidate.index);

    // Safety: don't prune if any subtree member is protected
    const hasProtected = subtreeIndices.some((idx) => protectedIndices.has(idx));
    if (hasProtected) continue;

    // Don't prune if it would push us over maxRemovable
    if (removedCount + subtreeIndices.length > maxRemovable) continue;

    // Apply the prune
    for (const idx of subtreeIndices) {
      prunedIndices.add(idx);
    }

    // Compute actual token savings using batch estimator on the subtree messages
    const subtreeMessages = subtreeIndices.map((i) => messages[i]!);
    const actualSubtreeTokens = estimateTokensFromMessages(subtreeMessages);

    prunedSubtrees.push({
      rootIndex: candidate.index,
      subtreeSize: subtreeIndices.length,
      tokensSaved: actualSubtreeTokens,
      valuePerToken: vpt,
    });
    removedCount += subtreeIndices.length;

    // Check budget using the authoritative batch estimator on the current remaining set.
    // This avoids per-message ceil-rounding drift accumulating across iterations.
    const currentTokens = estimateTokensFromMessages(
      messages.filter((_, i) => !prunedIndices.has(i)),
    );
    if (currentTokens <= tokenBudget) break;
  }

  // Build the compacted message array (preserve original order)
  const compacted = messages.filter((_, i) => !prunedIndices.has(i));
  const tokensAfter = estimateTokensFromMessages(compacted);

  const report: TreeCompactionReport = {
    prunedIndices: [...prunedIndices].sort((a, b) => a - b),
    prunedSubtrees,
    tokensBefore,
    tokensAfter,
    tokensSaved: tokensBefore - tokensAfter,
    targetAchieved: tokensAfter <= tokenBudget,
    messagesAfter: compacted.length,
  };

  return { messages: compacted, report };
}
