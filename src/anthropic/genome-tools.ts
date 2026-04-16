/**
 * withGenome — prepend genome RAG context to an Anthropic system prompt.
 *
 * Reads `.ashlrcode/genome/` at `cwd` (walking up if absent), retrieves the
 * most relevant sections for the given query via `retrieveSectionsV2`
 * (Ollama-backed semantic search when available, TF-IDF fallback), and
 * returns a new system-prompt string with the retrieved sections stitched
 * above the caller's original instructions.
 *
 * If no genome is found, the original system prompt is returned unchanged.
 */

import { existsSync } from "fs";
import { dirname, resolve } from "path";
import { formatGenomeForPrompt, retrieveSectionsV2 } from "../genome/retriever.ts";

const DEFAULT_MAX_TOKENS = 2000;

export interface WithGenomeOptions {
  /**
   * Retrieval token budget. Defaults to 2000 — matches the conservative
   * injection budget used by ashlrcode's orchestrator.
   */
  maxTokens?: number;
  /**
   * Query used to rank sections. Empty string selects the default "core"
   * sections (north-star, current milestone, active strategies).
   */
  query?: string;
  /**
   * If true (default) and no `.ashlrcode/genome/` is found at `cwd`,
   * walk up the directory tree looking for one. Handles the common case
   * where the SDK is invoked from a subdirectory of the repo.
   */
  walkUp?: boolean;
}

/**
 * Walk from `start` up the filesystem until a `.ashlrcode/genome/` directory
 * is found. Returns the directory containing `.ashlrcode/` (what
 * `retrieveSectionsV2` expects), or `undefined` if nothing is found.
 */
function findGenomeRoot(start: string): string | undefined {
  let dir = resolve(start);
  // Cap the walk at 20 hops to guard against pathological setups; in
  // practice genome should be at most a handful of levels up.
  for (let i = 0; i < 20; i++) {
    if (existsSync(`${dir}/.ashlrcode/genome`)) return dir;
    const parent = dirname(dir);
    if (parent === dir) return undefined;
    dir = parent;
  }
  return undefined;
}

/**
 * Prepend genome RAG context to a system prompt.
 *
 * @param systemPrompt The caller's base system prompt.
 * @param cwd Working directory from which to locate the genome.
 * @param opts Retrieval options.
 * @returns A new system prompt with genome sections stitched above the original.
 *
 * @example
 * ```ts
 * const system = await withGenome(
 *   "You are a coding assistant.",
 *   process.cwd(),
 *   { query: "authentication flow", maxTokens: 1500 }
 * );
 * ```
 */
export async function withGenome(
  systemPrompt: string,
  cwd: string,
  opts: WithGenomeOptions = {},
): Promise<string> {
  const maxTokens = opts.maxTokens ?? DEFAULT_MAX_TOKENS;
  const query = opts.query ?? "";
  const walkUp = opts.walkUp ?? true;

  const root = walkUp ? findGenomeRoot(cwd) : existsSync(`${cwd}/.ashlrcode/genome`) ? cwd : undefined;
  if (!root) return systemPrompt;

  let sections;
  try {
    sections = await retrieveSectionsV2(root, query, maxTokens);
  } catch {
    // Retrieval failed (missing manifest, corrupted genome, etc.) — fail
    // open: return the original prompt rather than blocking the caller.
    return systemPrompt;
  }

  if (sections.length === 0) return systemPrompt;

  const formatted = formatGenomeForPrompt(sections);
  return `${formatted}\n\n---\n\n${systemPrompt}`;
}
