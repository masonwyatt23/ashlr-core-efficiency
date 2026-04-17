/**
 * Context window manager for small-context local models (e.g., Qwen3-Coder 32K).
 *
 * The key insight: a 32K model has ~6x less context than Claude. You MUST be
 * aggressive — compact at 75% (not 90%), always snip tool results, and inject
 * genome sections at startup (highest-value context per token).
 *
 * Compaction tiers (cheapest first):
 *  1. snipCompact  — truncate tool results >2KB (zero LLM cost)
 *  2. contextCollapse — drop short/duplicate older messages (zero LLM cost)
 *  3. autoCompact  — LLM-summarize old messages (requires summarizer)
 */

import type { LLMSummarizer, Message } from "../types/index.ts";
import { estimateTokens } from "../tokens/index.ts";
import {
  snipCompact,
  contextCollapse,
  autoCompact,
} from "../compression/context.ts";
import {
  retrieveSectionsV2,
  formatGenomeForPrompt,
} from "../genome/retriever.ts";

export interface LocalContextConfig {
  /** Total context window of the model (e.g., 28000 for a 32K model with 4K response reserve). */
  maxTokens: number;
  /** Fraction (0-1) at which compaction triggers. Default: 0.75. */
  compactThreshold: number;
  /** Token budget reserved for genome sections in the system prompt. Default: 2000. */
  genomeTokenBudget: number;
  /** Token budget reserved for the system prompt (excluding genome). Default: 1000. */
  systemPromptBudget: number;
}

const DEFAULT_LOCAL_CONFIG: LocalContextConfig = {
  maxTokens: 28_000,
  compactThreshold: 0.75,
  genomeTokenBudget: 2_000,
  systemPromptBudget: 1_000,
};

export class LocalContextWindow {
  private config: LocalContextConfig;
  private messages: Message[] = [];
  private systemPrompt = "";
  private genomeContent = "";
  private compactionCount = 0;

  constructor(config: Partial<LocalContextConfig> = {}) {
    this.config = { ...DEFAULT_LOCAL_CONFIG, ...config };
  }

  /** Current token usage as a fraction (0-1). */
  utilization(): number {
    const used = this.usedTokens();
    return used / this.config.maxTokens;
  }

  /**
   * Add a message to the window.
   * Always snipCompacts incoming messages (tool results are truncated on ingestion).
   * Returns true if it fit, false if compaction is needed before the next LLM call.
   */
  add(message: Message): boolean {
    // Eagerly snip tool results on ingestion — cheapest possible savings
    const snipped = snipCompact([message]);
    this.messages.push(snipped[0] ?? message);

    return this.utilization() < this.config.compactThreshold;
  }

  /**
   * Compact the context window to free space.
   *
   * Applies tiers in order (cheapest first):
   *  1. snipCompact  — re-snip all messages (catches any that grew)
   *  2. contextCollapse — drop short/duplicate older messages
   *  3. autoCompact  — LLM-summarize (only if summarizer provided AND still over threshold)
   */
  async compact(summarizer?: LLMSummarizer): Promise<void> {
    this.compactionCount++;

    // Tier 1: snip tool results
    this.messages = snipCompact(this.messages);

    // Tier 2: collapse redundant messages
    this.messages = contextCollapse(this.messages);

    // Check if we're under threshold now
    if (this.utilization() < this.config.compactThreshold) return;

    // Tier 3: LLM summarize (expensive — only if caller provided a summarizer)
    if (summarizer) {
      this.messages = await autoCompact(this.messages, summarizer, {
        maxContextTokens: this.config.maxTokens,
        reserveTokens: Math.ceil(this.config.maxTokens * (1 - this.config.compactThreshold)),
        recentMessageCount: 6, // Keep fewer recent messages on small context
      });
    }
  }

  /**
   * Get all messages ready for an LLM call.
   * If genome content has been injected, it's prepended to the system prompt
   * as a synthetic first user message (for models that don't support system prompts)
   * or returned separately via the system prompt getter.
   */
  getMessages(): Message[] {
    return [...this.messages];
  }

  /**
   * Get the full system prompt including genome context.
   */
  getSystemPrompt(): string {
    if (!this.genomeContent) return this.systemPrompt;
    return `${this.genomeContent}\n\n---\n\n${this.systemPrompt}`;
  }

  /**
   * Set the base system prompt (without genome content).
   */
  setSystemPrompt(prompt: string): void {
    this.systemPrompt = prompt;
  }

  /**
   * Inject genome context into the system prompt.
   * Retrieves the most relevant genome sections for this project and optional query,
   * fitting within the configured genomeTokenBudget.
   */
  async injectGenome(cwd: string, query?: string): Promise<void> {
    const sections = await retrieveSectionsV2(
      cwd,
      query ?? "",
      this.config.genomeTokenBudget,
    );
    this.genomeContent = formatGenomeForPrompt(sections);
  }

  /** Get stats for display / debugging. */
  stats(): {
    totalTokens: number;
    used: number;
    utilization: number;
    messageCount: number;
    compactions: number;
  } {
    const used = this.usedTokens();
    return {
      totalTokens: this.config.maxTokens,
      used,
      utilization: used / this.config.maxTokens,
      messageCount: this.messages.length,
      compactions: this.compactionCount,
    };
  }

  /** Internal: compute total tokens used across system prompt + genome + messages. */
  private usedTokens(): number {
    const systemTokens = estimateTokens(this.getSystemPrompt());
    const messageTokens = estimateTokens(this.messages);
    return systemTokens + messageTokens;
  }
}
