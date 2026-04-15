// Phase B: port from ashlrcode/src/agent/
// Files to move: context.ts (compression tiers), system-prompt.ts (builder)
// estimateTokens() moves to ../tokens/ (consolidated from 3 duplicates)
// Add: priority.ts — PromptPriority enum replacing 11 magic numbers

export const COMPRESSION_MODULE_STATUS = "placeholder" as const;
