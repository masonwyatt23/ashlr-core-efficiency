// Shared interfaces so consumers don't pull ashlrcode's ProviderRouter directly.

export interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content: string | ContentBlock[];
}

export interface ContentBlock {
  type: string;
  text?: string;
  [key: string]: unknown;
}

/**
 * Minimal streaming LLM interface. Genome/scribe and compression/autoCompact
 * depend on this instead of ashlrcode's ProviderRouter, so they can be
 * extracted cleanly and reused from ashlr-plugin.
 */
export interface LLMSummarizer {
  stream(messages: Message[]): AsyncIterable<string>;
}
