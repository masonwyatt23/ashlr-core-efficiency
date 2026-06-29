/**
 * Genome module — genetic AI development loop.
 *
 * Self-evolving project specs via RAG + scribe protocol.
 * commands.ts (the CLI layer) stays in ashlrcode; everything else lives here.
 *
 * Uses per-file star re-exports so every symbol defined in the module is
 * exposed through `@ashlr/core-efficiency/genome` without a hand-maintained list.
 * Manifest's `estimateTokens` conflicts with the tokens module barrel — handled
 * in the root ../index.ts, not here.
 */

export * from "./embeddings.ts";
export * from "./embedding-codec.ts";
export * from "./embedding-router.ts";
export * from "./fitness.ts";
export * from "./fitness-hooks.ts";
export * from "./generations.ts";
export * from "./init.ts";
export * from "./jsonl.ts";
export * from "./manifest.ts";
export * from "./manifest-versioning.ts";
export * from "./retriever.ts";
export * from "./scribe.ts";
export * from "./strategies.ts";
export * from "./retrieval-adapter.ts";
export * from "./graph-traversal.ts";
export * from "./quantized-ann.ts";
export * from "./quantization-strategy.ts";
