/**
 * Compression module — 3-tier context compression + prompt-priority slots.
 *
 * system-prompt.ts (the builder) stays in ashlrcode because it couples to
 * ashlrcode-specific subsystems (tool registry, model patches, settings,
 * undercover mode). Only the reusable efficiency primitives live here.
 */

export * from "./context.ts";
export * from "./priority.ts";
