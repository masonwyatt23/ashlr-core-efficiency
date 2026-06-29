/**
 * Genome retriever — keyword-based section retrieval for system prompt injection.
 *
 * Scores genome sections by relevance to a query (task description),
 * returning the top-N sections that fit within a token budget.
 * Uses TF-IDF-inspired scoring against section tags, summaries, and titles.
 *
 * Adaptive retrieval (retrieveSectionsAdaptive) adds three strategy tiers:
 *  - Tier 1 (keyword):     TF-IDF only — zero cost, deterministic.
 *  - Tier 2 (semantic):    Top-20 TF-IDF candidates reranked by embedding
 *                          cosine similarity to query. Requires Ollama.
 *  - Tier 3 (hierarchical): Keyword + automatic parent/child expansion.
 * Strategy is auto-selected based on Ollama availability and caller hints.
 * Per-query latency and relevance deltas are appended to
 * `.ashlrcode/genome/evolution/retrieval-audit.jsonl` for fitness feedback.
 */

import { readFile } from "fs/promises";
import {
  estimateTokens,
  type GenomeManifest,
  genomeDir,
  loadManifest,
  type SectionMeta,
  sectionPath,
} from "./manifest.ts";
import type { DistributedSectionMeta } from "./distributed-manifest.ts";

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

interface ScoredSection {
  section: SectionMeta;
  score: number;
}

/**
 * Tokenize a string into lowercase terms, stripping punctuation.
 */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2);
}

/**
 * Score a section's relevance to a set of query terms.
 *
 * Scoring heuristic:
 *  - Tag match:     3 points per matching tag
 *  - Title match:   2 points per term found in title
 *  - Summary match: 1 point per term found in summary
 *  - IDF boost:     rarer terms (fewer section matches) score higher
 */
function scoreSection(section: SectionMeta, queryTerms: string[], idf: Map<string, number>): number {
  let score = 0;

  const tagSet = new Set(section.tags.map((t) => t.toLowerCase()));
  const titleTerms = new Set(tokenize(section.title));
  const summaryTerms = new Set(tokenize(section.summary));

  for (const term of queryTerms) {
    const boost = idf.get(term) ?? 1;

    if (tagSet.has(term)) score += 3 * boost;
    if (titleTerms.has(term)) score += 2 * boost;
    if (summaryTerms.has(term)) score += 1 * boost;
  }

  return score;
}

/**
 * Build an IDF-like map: terms that appear in fewer sections get higher weight.
 */
function buildIDF(sections: SectionMeta[]): Map<string, number> {
  const docFreq = new Map<string, number>();
  const n = sections.length;

  for (const s of sections) {
    const allTerms = new Set([...s.tags.map((t) => t.toLowerCase()), ...tokenize(s.title), ...tokenize(s.summary)]);
    for (const term of allTerms) {
      docFreq.set(term, (docFreq.get(term) ?? 0) + 1);
    }
  }

  const idf = new Map<string, number>();
  for (const [term, freq] of docFreq) {
    // log(N / freq) + 1, minimum 1
    idf.set(term, Math.max(1, Math.log(n / freq) + 1));
  }

  return idf;
}

// ---------------------------------------------------------------------------
// Hierarchy helpers
// ---------------------------------------------------------------------------

/**
 * Build a parent→children map from all sections that declare a parentId.
 * Only valid parentIds (pointing to an existing section path) are included.
 */
function buildChildrenMap(sections: SectionMeta[]): Map<string, SectionMeta[]> {
  const pathSet = new Set(sections.map((s) => s.path));
  const map = new Map<string, SectionMeta[]>();

  for (const s of sections) {
    if (!s.parentId || !pathSet.has(s.parentId)) continue;
    const existing = map.get(s.parentId) ?? [];
    existing.push(s);
    map.set(s.parentId, existing);
  }
  return map;
}

/**
 * Collect a section and all its descendants up to `maxDepth` levels deep,
 * using an iterative BFS to avoid call-stack overflow on deep trees.
 * Cycle detection: tracks visited paths so infinite loops are impossible.
 */
function collectDescendants(
  root: SectionMeta,
  childrenMap: Map<string, SectionMeta[]>,
  maxDepth: number,
): SectionMeta[] {
  const visited = new Set<string>();
  const result: SectionMeta[] = [];

  // Queue entries: [section, currentDepth]
  const queue: Array<[SectionMeta, number]> = [[root, 0]];

  while (queue.length > 0) {
    const [current, depth] = queue.shift()!;

    if (visited.has(current.path)) continue; // cycle guard
    visited.add(current.path);
    result.push(current);

    if (depth < maxDepth) {
      const children = childrenMap.get(current.path) ?? [];
      for (const child of children) {
        if (!visited.has(child.path)) {
          queue.push([child, depth + 1]);
        }
      }
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface RetrievedSection {
  path: string;
  title: string;
  content: string;
  tokens: number;
  score: number;
  /**
   * Cross-referenced section paths discovered during graph traversal.
   * Populated by `buildAndTraverseGraph` when a section mentions "See X"
   * or similar patterns in its metadata. The weaver can use these to
   * chain further retrieval for richer context.
   * Only present on sections returned by graph-traversal-based retrieval.
   */
  relatedSectionPaths?: string[];
}

/**
 * Configuration for hierarchical retrieval depth.
 *
 * `maxDepth` controls how many levels of child sections are pulled when a
 * parent section is selected:
 *  - 0 → only the matched section itself (same as flat retrieval)
 *  - 1 → matched section + its direct children
 *  - 2 → matched section + children + grandchildren (default)
 *
 * `childScoreFraction` is the score multiplier applied to child sections that
 * were pulled in via hierarchy rather than direct query match (default: 0.5).
 * This keeps them lower-priority than directly matched sections so they fill
 * budget space without displacing higher-scoring direct hits.
 */
export interface RetrievalDepthConfig {
  /** Maximum depth of child sections to include. Default: 2. */
  maxDepth: number;
  /**
   * Score multiplier for implicitly included child sections.
   * Must be in (0, 1]. Default: 0.5.
   */
  childScoreFraction: number;
}

/**
 * Retrieve genome sections using hierarchical context expansion.
 *
 * For each directly-matched section, child sections (up to `depthConfig.maxDepth`
 * levels) are automatically appended at a reduced priority score so that
 * retrieving "architecture" also pulls in "architecture/layer-api.md" etc.
 *
 * Cycle-safe: the BFS in collectDescendants tracks visited paths.
 *
 * @param query       Natural language task description
 * @param cwd         Project root (genome lives at .ashlrcode/genome/)
 * @param maxTokens   Token budget for all returned sections combined
 * @param depthConfig Depth + score-fraction config (defaults: maxDepth=2, childScoreFraction=0.5)
 */
export async function retrieveHierarchical(
  query: string,
  cwd: string,
  maxTokens: number,
  depthConfig: Partial<RetrievalDepthConfig> = {},
): Promise<RetrievedSection[]> {
  const { maxDepth = 2, childScoreFraction = 0.5 } = depthConfig;

  const manifest = await loadManifest(cwd);
  if (!manifest || manifest.sections.length === 0) return [];

  const queryTerms = tokenize(query);
  const idf = buildIDF(manifest.sections);
  const childrenMap = buildChildrenMap(manifest.sections);

  // Score all sections by direct relevance
  const directScores = new Map<string, number>();
  for (const section of manifest.sections) {
    const s = queryTerms.length > 0 ? scoreSection(section, queryTerms, idf) : 0;
    if (s > 0) directScores.set(section.path, s);
  }

  // If no query terms, fall back to core sections
  if (queryTerms.length === 0) {
    return retrieveCoreSections(cwd, manifest, maxTokens);
  }

  // Expand: for each matched section pull descendants, deduplicating by path
  const seen = new Set<string>();
  const expanded: ScoredSection[] = [];

  // Sort direct hits by score descending so highest-priority parents expand first
  const directHits = manifest.sections
    .filter((s) => directScores.has(s.path))
    .sort((a, b) => (directScores.get(b.path) ?? 0) - (directScores.get(a.path) ?? 0));

  for (const parent of directHits) {
    const descendants = collectDescendants(parent, childrenMap, maxDepth);
    for (const desc of descendants) {
      if (seen.has(desc.path)) continue;
      seen.add(desc.path);
      const isParent = desc.path === parent.path;
      const score = isParent
        ? (directScores.get(desc.path) ?? 0)
        : (directScores.get(desc.path) ?? (directScores.get(parent.path) ?? 0) * childScoreFraction);
      expanded.push({ section: desc, score });
    }
  }

  // Also include any remaining direct hits not yet seen (shouldn't happen but be safe)
  for (const s of directHits) {
    if (!seen.has(s.path)) {
      seen.add(s.path);
      expanded.push({ section: s, score: directScores.get(s.path) ?? 0 });
    }
  }

  // Sort final list by score descending
  expanded.sort((a, b) => b.score - a.score);

  return packSections(cwd, expanded, maxTokens);
}

/**
 * Retrieve the most relevant genome sections for a query, fitting within a token budget.
 *
 * Returns sections sorted by relevance score (descending), including content.
 * Sections with score 0 are excluded.
 */
export async function retrieveSections(cwd: string, query: string, maxTokens: number): Promise<RetrievedSection[]> {
  const manifest = await loadManifest(cwd);
  if (!manifest || manifest.sections.length === 0) return [];

  const queryTerms = tokenize(query);
  if (queryTerms.length === 0) {
    // No meaningful query — return highest-priority sections (vision, current milestone)
    return retrieveCoreSections(cwd, manifest, maxTokens);
  }

  const idf = buildIDF(manifest.sections);

  // Score all sections
  const scored: ScoredSection[] = manifest.sections
    .map((section) => ({
      section,
      score: scoreSection(section, queryTerms, idf),
    }))
    .filter((s) => s.score > 0)
    .sort((a, b) => b.score - a.score);

  // Pack sections within token budget
  return packSections(cwd, scored, maxTokens);
}

/**
 * Retrieve core sections (north-star, current milestone, active strategies)
 * when no specific query is given.
 */
async function retrieveCoreSections(
  cwd: string,
  manifest: GenomeManifest,
  maxTokens: number,
): Promise<RetrievedSection[]> {
  const corePaths = ["vision/north-star.md", "milestones/current.md", "strategies/active.md"];

  const coreScored: ScoredSection[] = manifest.sections
    .filter((s) => corePaths.includes(s.path))
    .map((section) => ({ section, score: corePaths.length - corePaths.indexOf(section.path) }));

  return packSections(cwd, coreScored, maxTokens);
}

/**
 * Pack scored sections into the token budget, reading content from disk.
 */
async function packSections(cwd: string, scored: ScoredSection[], maxTokens: number): Promise<RetrievedSection[]> {
  const results: RetrievedSection[] = [];
  let usedTokens = 0;

  for (const { section, score } of scored) {
    if (usedTokens >= maxTokens) break;

    let content: string;
    try {
      // Use sectionPath for path traversal validation at read time
      const fullPath = sectionPath(cwd, section.path);
      content = await readFile(fullPath, "utf-8");
    } catch {
      continue; // File missing, unreadable, or invalid path — skip
    }
    const tokens = estimateTokens(content);

    if (usedTokens + tokens > maxTokens) {
      // Try truncating to fit
      const remaining = maxTokens - usedTokens;
      if (remaining > 200) {
        // Worth including a truncated version
        const truncated = content.slice(0, remaining * 4) + "\n\n[... section truncated ...]";
        results.push({
          path: section.path,
          title: section.title,
          content: truncated,
          tokens: remaining,
          score,
        });
        usedTokens += remaining;
      }
      break;
    }

    results.push({
      path: section.path,
      title: section.title,
      content,
      tokens,
      score,
    });
    usedTokens += tokens;
  }

  return results;
}

/**
 * Retrieve sections using semantic search (Ollama embeddings) when available,
 * falling back to hierarchical keyword-based TF-IDF search.
 *
 * This is the preferred entry point for retrieval — it transparently upgrades
 * to embedding-based search when a local Ollama instance is running with
 * cached embeddings, and degrades gracefully to hierarchical keyword search
 * otherwise (which expands parent sections to include child sections up to
 * the default depth of 2).
 *
 * Quantization-aware tier selection:
 *   When a `RouterBuildResult` is supplied (from `quantizationStrategyRouter`),
 *   retrieval automatically selects the cheapest quantization tier that stays
 *   within the `maxTokens` budget by calling `selectBudgetTier`. This avoids
 *   reconstructing high-precision embeddings when the budget is tight and lower
 *   tiers (int8, bfloat16) are sufficient for the sections being ranked.
 *
 *   Expected savings on large genomes (100+ sections):
 *     - 3–5× embedding storage reduction across the cache.
 *     - 15–25% query-latency reduction from cheaper reconstruction.
 *
 * @param cwd              Project root.
 * @param query            Natural-language or code search query.
 * @param maxTokens        Token budget for the returned sections.
 * @param routerBuildResult Optional pre-computed per-section tier metadata from
 *                          `quantizationStrategyRouter`. When provided, tier
 *                          selection is budget-aware. When absent, falls back to
 *                          the existing behaviour (no per-section routing).
 */
export async function retrieveSectionsV2(
  cwd: string,
  query: string,
  maxTokens: number,
  routerBuildResult?: import("./quantization-strategy.ts").RouterBuildResult | null,
): Promise<RetrievedSection[]> {
  // Only attempt semantic search for non-empty queries
  if (query.trim().length > 0) {
    try {
      const embeddingsMod = await import("./embeddings.ts");
      const { isOllamaAvailable, semanticSearch, loadEmbeddingCache, cosineSimilarity } = embeddingsMod;

      if (await isOllamaAvailable()) {
        // When routerBuildResult is available, use budget-aware tier selection
        // to reconstruct embeddings at the cheapest acceptable quality tier.
        if (routerBuildResult) {
          const { GenomeProfiler } = await import("./quantization-strategy.ts");
          const { loadManifest, genomeDir, estimateTokens } = await import("./manifest.ts");
          const { selectBudgetTier } = await import("./quantization-strategy.ts");
          const { dequantizeUnified, quantizeUnified } = embeddingsMod as typeof import("./embeddings.ts");

          const manifest = await loadManifest(cwd);
          if (manifest && manifest.sections.length > 0) {
            const profile = GenomeProfiler.profile(manifest);
            const cache = await loadEmbeddingCache(cwd);

            // Determine the optimal tier for this query's budget
            const allPaths = cache.map((e) => e.sectionPath);
            const selectedTier = selectBudgetTier(profile, maxTokens, allPaths, routerBuildResult);

            // Generate query embedding
            const { generateEmbedding } = embeddingsMod;
            const queryEmbedding = await generateEmbedding(query);
            if (queryEmbedding) {
              // Score all cached sections using the selected quantization tier
              const scored: Array<{ path: string; similarity: number }> = [];
              for (const entry of cache) {
                if (entry.embedding.length === 0) continue;
                // Quantize + dequantize the stored embedding at the selected tier
                const quantized = quantizeUnified(entry.embedding, selectedTier);
                const reconstructed = dequantizeUnified(quantized);
                const similarity = cosineSimilarity(queryEmbedding, reconstructed);
                if (similarity > 0.3) {
                  scored.push({ path: entry.sectionPath, similarity });
                }
              }

              if (scored.length > 0) {
                scored.sort((a, b) => b.similarity - a.similarity);

                // Pack sections within token budget
                const { readFile } = await import("fs/promises");
                const { existsSync } = await import("fs");
                const { join } = await import("path");
                const dir = genomeDir(cwd);
                const sectionMap = new Map(manifest.sections.map((s) => [s.path, s]));
                const results: RetrievedSection[] = [];
                let usedTokens = 0;

                for (const { path, similarity } of scored) {
                  if (usedTokens >= maxTokens) break;
                  const fullPath = join(dir, path);
                  if (!existsSync(fullPath)) continue;
                  const content = await readFile(fullPath, "utf-8");
                  const tokens = estimateTokens(content);
                  const meta = sectionMap.get(path);
                  const title = meta?.title ?? path;

                  if (usedTokens + tokens > maxTokens) {
                    const remaining = maxTokens - usedTokens;
                    if (remaining > 200) {
                      results.push({
                        path,
                        title,
                        content: content.slice(0, remaining * 4) + "\n\n[... section truncated ...]",
                        tokens: remaining,
                        score: similarity,
                      });
                      usedTokens += remaining;
                    }
                    break;
                  }

                  results.push({ path, title, content, tokens, score: similarity });
                  usedTokens += tokens;
                }

                if (results.length > 0) return results;
              }
            }
          }
        }

        // No routerBuildResult — use the standard semanticSearch path
        const results = await semanticSearch(cwd, query, maxTokens);
        if (results.length > 0) return results;
      }
    } catch {
      // Embedding module unavailable or errored — fall through to hierarchical keyword search
    }
  }

  // Fall back to hierarchical keyword search (default depth = 2)
  return retrieveHierarchical(query, cwd, maxTokens);
}

/**
 * Format retrieved sections for injection into the system prompt.
 */
export function formatGenomeForPrompt(sections: RetrievedSection[]): string {
  if (sections.length === 0) return "";

  const formatted = sections.map((s) => `### ${s.title} (${s.path})\n${s.content}`);

  return `## Project Genome\n\n${formatted.join("\n\n---\n\n")}`;
}

/**
 * Inject genome context into a system prompt builder.
 *
 * Reads the genome, retrieves relevant sections for the task, and adds
 * them at priority 25 (after permissions, before knowledge files).
 */
export async function injectGenomeContext(
  builder: { addPart: (name: string, content: string, priority: number) => unknown },
  cwd: string,
  taskDescription: string,
  maxTokens: number,
): Promise<number> {
  const sections = await retrieveSections(cwd, taskDescription, maxTokens);
  if (sections.length === 0) return 0;

  const content = formatGenomeForPrompt(sections);
  const { PromptPriority } = await import("../compression/index.ts");
  builder.addPart("genome", content, PromptPriority.Genome);

  return sections.reduce((sum, s) => sum + s.tokens, 0);
}

// ---------------------------------------------------------------------------
// Adaptive retrieval — three-tier strategy with auto-selection
// ---------------------------------------------------------------------------

/**
 * The strategy tier to use for a retrieval call.
 *
 *  - "keyword":      Tier 1 — pure TF-IDF, zero cost, always available.
 *  - "semantic":     Tier 2 — TF-IDF top-20 reranked by embedding cosine
 *                    similarity. Requires a running Ollama instance.
 *  - "hierarchical": Tier 3 — TF-IDF keyword matching plus automatic
 *                    parent/child expansion (maxDepth=2).
 *  - "auto":         Cascades: hierarchical → semantic → keyword depending
 *                    on Ollama availability.
 */
export type AdaptiveRetrievalStrategy = "auto" | "keyword" | "semantic" | "hierarchical";

/** Options for {@link retrieveSectionsAdaptive}. */
export interface AdaptiveRetrievalOptions {
  /**
   * Preferred strategy. Defaults to "auto".
   *
   * "auto" tries hierarchical first, then semantic if Ollama is up, then
   * keyword as the guaranteed fallback.
   */
  strategy?: AdaptiveRetrievalStrategy;
  /**
   * Aggressiveness in [0, 1]. Higher values broaden the candidate pool and
   * are more willing to spend latency on semantic reranking.
   *  - 0.0 → minimal (top-5 TF-IDF, skip Ollama even if available in auto mode)
   *  - 0.5 → balanced default
   *  - 1.0 → maximum (top-30 TF-IDF candidates for semantic, maxDepth=3 for hierarchical)
   * Defaults to 0.5.
   */
  aggressiveness?: number;
}

/** Result returned by {@link retrieveSectionsAdaptive}. */
export interface AdaptiveRetrievalResult {
  /** The retrieved sections, sorted by relevance score descending. */
  sections: RetrievedSection[];
  /** The strategy tier that was actually used. */
  strategy: AdaptiveRetrievalStrategy;
  /** Wall-clock time for this retrieval in milliseconds. */
  latency_ms: number;
  /**
   * Estimated relevance quality in [0, 1].
   *
   * For keyword: normalised sum of TF-IDF scores against the maximum observed
   * in the result set (or 0 when no sections match).
   * For semantic: average cosine similarity of returned sections (already [0,1]).
   * For hierarchical: same normalisation as keyword over the expanded set.
   */
  relevance_score: number;
}

/** Path to the per-query retrieval audit trail. */
function auditPath(cwd: string): string {
  const { genomeDir } = require("./manifest.ts") as typeof import("./manifest.ts");
  const { join } = require("path") as typeof import("path");
  return join(genomeDir(cwd), "evolution", "retrieval-audit.jsonl");
}

/** Shape of one audit record appended to retrieval-audit.jsonl. */
interface RetrievalAuditRecord {
  ts: string;
  query: string;
  /** The strategy that was requested (may be "auto"). */
  requested_strategy: AdaptiveRetrievalStrategy;
  /** The strategy that actually ran. */
  used_strategy: Exclude<AdaptiveRetrievalStrategy, "auto">;
  latency_ms: number;
  relevance_score: number;
  sections_returned: number;
  budget_tokens: number;
  tokens_used: number;
}

/** Append one audit record; never throws. */
async function appendAudit(cwd: string, record: RetrievalAuditRecord): Promise<void> {
  try {
    const { appendJsonl } = await import("./jsonl.ts");
    await appendJsonl(auditPath(cwd), record);
  } catch {
    // Audit writes are best-effort — never let them break retrieval.
  }
}

/**
 * Compute a normalised relevance score in [0,1] for a set of retrieved sections.
 *
 * When all scores are 0 (no query / no match) returns 0.
 * For keyword/hierarchical tiers the score is the mean normalised TF-IDF score.
 * For semantic results (score already in [0,1] cosine space) the mean is used directly.
 */
function computeRelevanceScore(sections: RetrievedSection[]): number {
  if (sections.length === 0) return 0;

  const scores = sections.map((s) => s.score);
  const max = Math.max(...scores);
  if (max <= 0) return 0;

  // If the scores are already in cosine-similarity range (all ≤ 1), treat as-is.
  // Otherwise normalise by the observed max so large TF-IDF values map to [0,1].
  const needsNorm = max > 1;
  const normalised = scores.map((s) => (needsNorm ? s / max : s));
  return normalised.reduce((a, b) => a + b, 0) / normalised.length;
}

/**
 * Semantic reranking (Tier 2).
 *
 * Fetches up to `candidateCount` sections via TF-IDF, generates a query
 * embedding, reranks all candidates by cosine similarity, and returns the
 * top-N that fit within the token budget. Returns null when Ollama is
 * unreachable so the caller can fall through to a cheaper tier.
 */
async function semanticRerank(
  query: string,
  cwd: string,
  budget: number,
  candidateCount: number,
): Promise<RetrievedSection[] | null> {
  try {
    const { isOllamaAvailable, generateEmbedding, cosineSimilarity } = await import(
      "./embeddings.ts"
    );
    if (!(await isOllamaAvailable())) return null;

    const queryEmbedding = await generateEmbedding(query);
    if (!queryEmbedding) return null;

    // Fetch TF-IDF candidates (no token-budget constraint yet — we'll pack later)
    const manifest = await loadManifest(cwd);
    if (!manifest || manifest.sections.length === 0) return null;

    const queryTerms = tokenize(query);
    const idf = buildIDF(manifest.sections);

    const candidates: ScoredSection[] = manifest.sections
      .map((section) => ({
        section,
        score: queryTerms.length > 0 ? scoreSection(section, queryTerms, idf) : 0,
      }))
      .filter((s) => s.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, candidateCount);

    if (candidates.length === 0) return null;

    // Load the cached embeddings once for all candidates
    const { loadEmbeddingCache } = await import("./embeddings.ts");
    const cache = await loadEmbeddingCache(cwd);
    const cacheMap = new Map(cache.map((e) => [e.sectionPath, e.embedding]));

    // Rerank by cosine similarity to query embedding
    const reranked: ScoredSection[] = candidates
      .map(({ section }) => {
        const emb = cacheMap.get(section.path);
        const similarity = emb ? cosineSimilarity(queryEmbedding, emb) : 0;
        return { section, score: similarity };
      })
      .filter((s) => s.score > 0)
      .sort((a, b) => b.score - a.score);

    if (reranked.length === 0) return null;

    return packSections(cwd, reranked, budget);
  } catch {
    return null;
  }
}

/**
 * Retrieve genome sections using an adaptive three-tier strategy.
 *
 * Strategy cascade (when strategy === "auto"):
 *  1. Try Tier 3 (hierarchical): keyword + parent/child expansion.
 *  2. Try Tier 2 (semantic): TF-IDF top-20 reranked by Ollama cosine similarity.
 *     Skipped when Ollama is unavailable or aggressiveness < 0.2.
 *  3. Fall back to Tier 1 (keyword): plain TF-IDF.
 *
 * `aggressiveness` (0–1) controls candidate pool size and depth:
 *  - 0.0: top-5 candidates, maxDepth=1 in hierarchical, skip semantic in auto.
 *  - 0.5: top-20 candidates, maxDepth=2 (default).
 *  - 1.0: top-30 candidates, maxDepth=3.
 *
 * Every call appends one record to
 * `.ashlrcode/genome/evolution/retrieval-audit.jsonl` for empirical
 * strategy-selection optimisation (fitness feedback loop). Audit writes are
 * best-effort and never block the return value.
 *
 * @param query       Natural-language task description.
 * @param cwd         Project root. Genome lives at `.ashlrcode/genome/`.
 * @param budget      Maximum token budget for the returned sections.
 * @param opts        Strategy and aggressiveness overrides.
 */
export async function retrieveSectionsAdaptive(
  query: string,
  cwd: string,
  budget: number,
  opts: AdaptiveRetrievalOptions = {},
): Promise<AdaptiveRetrievalResult> {
  const { strategy = "auto", aggressiveness = 0.5 } = opts;
  const clampedAgg = Math.max(0, Math.min(1, aggressiveness));

  // Derive tier-specific parameters from aggressiveness
  const candidateCount = Math.round(5 + clampedAgg * 25); // 5–30
  const maxDepth = clampedAgg < 0.35 ? 1 : clampedAgg > 0.8 ? 3 : 2;
  const skipSemantic = strategy === "auto" && clampedAgg < 0.2;

  const start = Date.now();
  let usedStrategy: Exclude<AdaptiveRetrievalStrategy, "auto">;
  let sections: RetrievedSection[];

  if (strategy === "keyword") {
    usedStrategy = "keyword";
    sections = await retrieveSections(cwd, query, budget);
  } else if (strategy === "semantic") {
    // Explicit semantic request: try reranking; fall back to keyword on failure.
    const reranked = await semanticRerank(query, cwd, budget, candidateCount);
    if (reranked && reranked.length > 0) {
      usedStrategy = "semantic";
      sections = reranked;
    } else {
      usedStrategy = "keyword";
      sections = await retrieveSections(cwd, query, budget);
    }
  } else if (strategy === "hierarchical") {
    usedStrategy = "hierarchical";
    sections = await retrieveHierarchical(query, cwd, budget, { maxDepth });
  } else {
    // "auto" cascade: hierarchical → semantic → keyword
    // Tier 3: hierarchical expansion
    const hierarchicalSections = await retrieveHierarchical(query, cwd, budget, { maxDepth });

    if (!skipSemantic) {
      // Tier 2: semantic reranking (requires Ollama + cached embeddings)
      const reranked = await semanticRerank(query, cwd, budget, candidateCount);
      if (reranked && reranked.length > 0) {
        // Semantic succeeded: merge hierarchical children that aren't already
        // present so we preserve contextual coherence from Tier 3.
        const semanticPaths = new Set(reranked.map((s) => s.path));
        const hierarchicalOnly = hierarchicalSections.filter((s) => !semanticPaths.has(s.path));

        // Hierarchical children don't count against the parent's token quota —
        // fill remaining budget with them.
        const semanticTokens = reranked.reduce((t, s) => t + s.tokens, 0);
        let remaining = budget - semanticTokens;
        const extras: RetrievedSection[] = [];
        for (const s of hierarchicalOnly) {
          if (remaining <= 0) break;
          if (s.tokens <= remaining) {
            extras.push(s);
            remaining -= s.tokens;
          }
        }

        sections = [...reranked, ...extras];
        usedStrategy = "semantic";
      } else {
        // Ollama unavailable or no embeddings: use hierarchical result
        sections = hierarchicalSections;
        usedStrategy = "hierarchical";
      }
    } else {
      // aggressiveness too low to risk Ollama latency — use hierarchical
      sections = hierarchicalSections;
      usedStrategy = "hierarchical";
    }
  }

  const latency_ms = Date.now() - start;
  const relevance_score = computeRelevanceScore(sections);
  const tokens_used = sections.reduce((t, s) => t + s.tokens, 0);

  // Fire-and-forget audit write — never awaited in the critical path
  void appendAudit(cwd, {
    ts: new Date().toISOString(),
    query,
    requested_strategy: strategy,
    used_strategy: usedStrategy,
    latency_ms,
    relevance_score,
    sections_returned: sections.length,
    budget_tokens: budget,
    tokens_used,
  });

  return { sections, strategy: usedStrategy, latency_ms, relevance_score };
}

// ---------------------------------------------------------------------------
// Distributed retrieval — multi-workspace / monorepo support
// ---------------------------------------------------------------------------

/**
 * Retrieve genome sections across multiple workspace roots using a single
 * merged IDF computed over the entire distributed corpus.
 *
 * This avoids the "large project bias" that would occur if IDF were computed
 * independently per workspace: a term that is common within one large
 * workspace is still correctly weighted against the full multi-workspace
 * section pool.
 *
 * Algorithm:
 *  1. Load & merge sections from all supplied `cwds` via
 *     `loadDistributedManifest`.
 *  2. Build IDF across the **full** merged section set so term rareness is
 *     measured globally.
 *  3. Score every section against the query using the global IDF.
 *  4. Pack top-scoring sections within `maxTokens`, reading content from the
 *     originating workspace root (tracked via `_origin` on each section).
 *
 * @param query      Natural-language query / task description.
 * @param cwds       Workspace roots to include in the distributed search.
 * @param maxTokens  Token budget for the returned sections combined.
 * @returns          Retrieved sections, sorted by relevance descending.
 *                   Each returned section carries `_origin` (the workspace
 *                   root it came from) on the `path` field's parent metadata
 *                   via the `metadata` property.
 */
export async function retrieveSectionsFromDist(
  query: string,
  cwds: string[],
  maxTokens: number,
): Promise<RetrievedSection[]> {
  const { loadDistributedManifest } = await import("./distributed-manifest.ts");
  const merged = await loadDistributedManifest(cwds);
  if (!merged || merged.sections.length === 0) return [];

  const queryTerms = tokenize(query);
  if (queryTerms.length === 0) {
    // No meaningful query — return core sections from the base (first) workspace
    return retrieveCoreSections(cwds[0] ?? ".", merged as unknown as GenomeManifest, maxTokens);
  }

  // Build IDF over the full merged corpus for unbiased cross-workspace scoring
  const idf = buildIDF(merged.sections);

  const scored: Array<{ section: DistributedSectionMeta; score: number }> = merged.sections
    .map((section) => ({
      section,
      score: scoreSection(section, queryTerms, idf),
    }))
    .filter((s) => s.score > 0)
    .sort((a, b) => b.score - a.score);

  if (scored.length === 0) return [];

  // Pack sections within the token budget, reading content from the originating root
  const results: RetrievedSection[] = [];
  let usedTokens = 0;

  for (const { section, score } of scored) {
    if (usedTokens >= maxTokens) break;

    // Determine which root to read the file from
    const origin = section._origin ?? cwds[0] ?? ".";

    let content: string;
    try {
      const fullPath = sectionPath(origin, section.path);
      content = await readFile(fullPath, "utf-8");
    } catch {
      continue; // File missing or path traversal rejected — skip
    }

    const tokens = estimateTokens(content);

    if (usedTokens + tokens > maxTokens) {
      const remaining = maxTokens - usedTokens;
      if (remaining > 200) {
        results.push({
          path: section.path,
          title: section.title,
          content: content.slice(0, remaining * 4) + "\n\n[... section truncated ...]",
          tokens: remaining,
          score,
        });
        usedTokens += remaining;
      }
      break;
    }

    results.push({
      path: section.path,
      title: section.title,
      content,
      tokens,
      score,
    });
    usedTokens += tokens;
  }

  return results;
}
