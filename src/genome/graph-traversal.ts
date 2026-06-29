/**
 * Genome Section Hierarchical Traversal + Cross-Reference Weaving
 *
 * Builds a bidirectional DAG from section metadata and implements:
 *  - Cycle-safe depth-limited hierarchical expansion (parent backlinks, sibling pulls)
 *  - Cross-reference resolution: sections mentioning "See X/Y.md" auto-pull the target
 *  - Sibling scoring: relevance-overlap ranking avoids fetching unrelated branches
 *  - Aggressive token-budget enforcement to prevent context bloat
 *
 * The main entry point is `buildSectionGraph` + `traverseSectionGraph`.
 * The weaver uses `RetrievedSection.relatedSectionPaths` to discover additional
 * cross-referenced sections discovered during traversal.
 */

import { readFile } from "fs/promises";
import {
  estimateTokens,
  type GenomeManifest,
  type SectionMeta,
  sectionPath,
} from "./manifest.ts";
import type { RetrievedSection } from "./retriever.ts";

// ---------------------------------------------------------------------------
// Graph node & edge types
// ---------------------------------------------------------------------------

/** A node in the section DAG. Holds the original metadata plus adjacency info. */
export interface SectionNode {
  /** Original section metadata from the manifest. */
  meta: SectionMeta;
  /** Paths of direct children (meta.parentId === this.meta.path). */
  children: string[];
  /** Path of direct parent (meta.parentId), or undefined for roots. */
  parent: string | undefined;
  /**
   * Sibling paths: sections sharing the same parent (excluding self).
   * Populated during `buildSectionGraph`.
   */
  siblings: string[];
  /**
   * Cross-reference targets extracted from the section's summary / tags.
   * These are candidate paths mentioned via "See X", "→ X/Y.md" etc.
   * Validated against the manifest before being stored.
   */
  crossRefs: string[];
}

/** The bidirectional DAG: path → SectionNode. */
export type SectionGraph = Map<string, SectionNode>;

// ---------------------------------------------------------------------------
// Cross-reference extraction
// ---------------------------------------------------------------------------

/**
 * Patterns that identify an explicit cross-reference in section metadata.
 *
 * Matches forms like:
 *  - "See architecture/patterns.md"
 *  - "→ vision/north-star"
 *  - "see also: strategies/active"
 *  - "ref: milestones/current.md"
 *  - "#architecture/patterns"
 */
const CROSS_REF_PATTERNS: RegExp[] = [
  // Longer alternatives first to prevent "see " swallowing "also" as the path.
  // Matches: "see also: X", "see X", "→ X", "ref: X", "reference: X", "#X"
  /(?:see\s+also\s*:\s*|see\s+|→\s*|ref(?:erence)?:\s*|#)([a-z0-9_\-./]+(?:\.md)?)/gi,
];

/**
 * Extract candidate cross-reference paths from a section's text fields.
 * Returns raw candidate strings; the caller is responsible for validation.
 */
export function extractCrossRefCandidates(meta: SectionMeta): string[] {
  const text = [meta.summary, ...meta.tags].join(" ");
  const candidates = new Set<string>();

  for (const pattern of CROSS_REF_PATTERNS) {
    // Reset lastIndex before each exec loop to handle the /g flag correctly
    pattern.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(text)) !== null) {
      const raw = match[1];
      if (!raw) continue;
      // Normalise: strip leading slash, ensure .md suffix when the path has a /
      let candidate = raw.replace(/^\//, "");
      if (candidate.includes("/") && !candidate.endsWith(".md")) {
        candidate = candidate + ".md";
      }
      candidates.add(candidate);
    }
  }

  return [...candidates];
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

/**
 * Build a bidirectional DAG from manifest sections.
 *
 * For every section:
 *  - `parent` links up to the section's declared parentId (if it exists in the manifest)
 *  - `children` links down to all sections that declare this section as their parentId
 *  - `siblings` lists peer sections sharing the same parent
 *  - `crossRefs` lists validated cross-reference targets found in summary/tags
 *
 * The construction is O(n) using a single pass for parent/child setup plus a
 * second pass for sibling resolution and cross-reference validation.
 *
 * @param manifest  The loaded genome manifest.
 * @returns         A map from section path to its SectionNode.
 */
export function buildSectionGraph(manifest: GenomeManifest): SectionGraph {
  const graph: SectionGraph = new Map();
  const pathSet = new Set(manifest.sections.map((s) => s.path));

  // First pass: create all nodes with empty adjacency sets
  for (const meta of manifest.sections) {
    // Only link parent if it actually exists in the manifest
    const parent =
      meta.parentId && pathSet.has(meta.parentId) ? meta.parentId : undefined;

    graph.set(meta.path, {
      meta,
      children: [],
      parent,
      siblings: [],
      crossRefs: [],
    });
  }

  // Second pass: wire children, siblings, and cross-refs
  // Group by parent for efficient sibling resolution
  const byParent = new Map<string, string[]>(); // parentPath → childPaths

  for (const [path, node] of graph) {
    if (node.parent) {
      // Wire child into the parent node
      const parentNode = graph.get(node.parent);
      if (parentNode && !parentNode.children.includes(path)) {
        parentNode.children.push(path);
      }
      // Track for sibling resolution
      const peers = byParent.get(node.parent) ?? [];
      peers.push(path);
      byParent.set(node.parent, peers);
    }

    // Validate and store cross-refs
    const candidates = extractCrossRefCandidates(node.meta);
    node.crossRefs = candidates.filter(
      (c) => pathSet.has(c) && c !== path, // must exist and not self-reference
    );
  }

  // Wire siblings: for each parent group, give every child the list of its peers
  for (const [, children] of byParent) {
    for (const childPath of children) {
      const node = graph.get(childPath);
      if (!node) continue;
      node.siblings = children.filter((p) => p !== childPath);
    }
  }

  return graph;
}

// ---------------------------------------------------------------------------
// Relevance scoring helpers
// ---------------------------------------------------------------------------

/**
 * Tokenise text into lowercase terms (mirrors retriever.ts tokenize).
 */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2);
}

/**
 * Score a section node against a set of query terms.
 *
 * Scoring weights (matches retriever.ts conventions):
 *  - Tag match:     3 pts
 *  - Title match:   2 pts
 *  - Summary match: 1 pt
 */
function scoreNode(node: SectionNode, queryTerms: string[]): number {
  if (queryTerms.length === 0) return 0;
  const { meta } = node;
  const tagSet = new Set(meta.tags.map((t) => t.toLowerCase()));
  const titleTerms = new Set(tokenize(meta.title));
  const summaryTerms = new Set(tokenize(meta.summary));
  let score = 0;
  for (const term of queryTerms) {
    if (tagSet.has(term)) score += 3;
    if (titleTerms.has(term)) score += 2;
    if (summaryTerms.has(term)) score += 1;
  }
  return score;
}

/**
 * Jaccard-like overlap between two term sets used to rank siblings.
 *
 * Returns a value in [0, 1]:
 *  - 1.0 → identical term sets
 *  - 0.0 → fully disjoint sets
 *
 * Two empty sets return 0 to avoid division-by-zero.
 */
export function termOverlap(setA: Set<string>, setB: Set<string>): number {
  if (setA.size === 0 || setB.size === 0) return 0;
  let intersection = 0;
  for (const t of setA) {
    if (setB.has(t)) intersection++;
  }
  const union = setA.size + setB.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

/**
 * Score a sibling section against the query terms of the anchor section.
 *
 * Sibling pull is justified only when there is meaningful query-term overlap.
 * Returns a score ≥ 0; caller applies a multiplier to keep siblings
 * lower-priority than direct hits.
 */
function scoreSiblingRelevance(
  sibling: SectionNode,
  anchorTerms: Set<string>,
  queryTerms: string[],
): number {
  // How much do the sibling's text fields share with the anchor's?
  const siblingTerms = new Set([
    ...sibling.meta.tags.map((t) => t.toLowerCase()),
    ...tokenize(sibling.meta.title),
    ...tokenize(sibling.meta.summary),
  ]);
  const overlap = termOverlap(anchorTerms, siblingTerms);

  // Also score the sibling directly against the query
  const directScore = scoreNode(sibling, queryTerms);

  // Composite: direct score matters more; overlap tiebreaks
  return directScore + overlap * 2;
}

// ---------------------------------------------------------------------------
// Traversal options
// ---------------------------------------------------------------------------

/** Configuration for `traverseSectionGraph`. */
export interface GraphTraversalOptions {
  /**
   * Maximum depth for hierarchical expansion (parent→child direction).
   * 0 = seed nodes only; 1 = direct children; 2 = grandchildren (default).
   * Must be ≥ 0; values > 10 are clamped to 10.
   */
  maxDepth?: number;
  /**
   * Whether to also traverse upward to transitive parents (backlinks).
   * Default: true. Pulls direct parent and grandparent for anchoring context.
   * Parent traversal is limited to `maxParentDepth` levels.
   */
  includeParents?: boolean;
  /**
   * Maximum number of parent levels to traverse upward.
   * Default: 2 (direct parent + grandparent).
   */
  maxParentDepth?: number;
  /**
   * Whether to resolve cross-references found in section metadata.
   * Default: true.
   */
  resolveCrossRefs?: boolean;
  /**
   * Minimum overlap score for sibling sections to be included.
   * Range [0, 1]; default 0.2. Set higher to exclude loosely related siblings.
   */
  siblingOverlapThreshold?: number;
  /**
   * Score multiplier applied to child sections pulled in via hierarchy.
   * Must be in (0, 1]. Default: 0.5.
   */
  childScoreFraction?: number;
  /**
   * Score multiplier applied to parent sections pulled in via backlinks.
   * Must be in (0, 1]. Default: 0.6.
   */
  parentScoreFraction?: number;
  /**
   * Score multiplier for cross-reference targets.
   * Must be in (0, 1]. Default: 0.4.
   */
  crossRefScoreFraction?: number;
  /**
   * Score multiplier for sibling sections.
   * Must be in (0, 1]. Default: 0.3.
   */
  siblingScoreFraction?: number;
  /**
   * Maximum number of sections to return (before token budget enforcement).
   * Prevents runaway expansion on large graphs. Default: 30.
   */
  maxSections?: number;
  /**
   * Token budget: total tokens across all returned sections.
   * Required — no default (must be passed by the caller).
   */
  maxTokens: number;
}

// ---------------------------------------------------------------------------
// Scored traversal result (internal)
// ---------------------------------------------------------------------------

interface TraversalEntry {
  path: string;
  score: number;
  /** How this entry was discovered: seed hit, child, parent, sibling, xref. */
  source: "direct" | "child" | "parent" | "sibling" | "xref";
  /** Cross-ref paths discovered from this section during traversal. */
  discoveredRefs: string[];
}

// ---------------------------------------------------------------------------
// Core traversal
// ---------------------------------------------------------------------------

/**
 * Traverse the section graph starting from seed sections, expanding outward
 * using depth-limited BFS with cycle detection.
 *
 * Expansion order (priority-first):
 *  1. Seed nodes (direct query matches, highest score)
 *  2. Child expansion (parent→child, depth-limited by maxDepth)
 *  3. Parent backlinks (child→parent, limited by maxParentDepth)
 *  4. Cross-reference resolution (sections mentioned via "See X")
 *  5. Sibling pulls (filtered by overlap threshold)
 *
 * The result is sorted by score descending and packed within the token budget.
 *
 * @param graph       Bidirectional section graph from `buildSectionGraph`.
 * @param seeds       Directly matched sections with their initial scores.
 * @param queryTerms  Tokenised query terms for scoring expansion candidates.
 * @param opts        Traversal configuration.
 * @returns           Ordered list of traversal entries (before content loading).
 */
export function traverseSectionGraph(
  graph: SectionGraph,
  seeds: Array<{ path: string; score: number }>,
  queryTerms: string[],
  opts: GraphTraversalOptions,
): TraversalEntry[] {
  const {
    maxDepth = 2,
    includeParents = true,
    maxParentDepth = 2,
    resolveCrossRefs = true,
    siblingOverlapThreshold = 0.2,
    childScoreFraction = 0.5,
    parentScoreFraction = 0.6,
    crossRefScoreFraction = 0.4,
    siblingScoreFraction = 0.3,
    maxSections = 30,
  } = opts;

  const clampedMaxDepth = Math.min(Math.max(0, maxDepth), 10);
  const clampedMaxParentDepth = Math.min(Math.max(0, maxParentDepth), 5);

  // visited tracks all paths that have been enqueued (not just processed)
  // to prevent duplicate entries from multiple expansion paths
  const visited = new Set<string>();
  // results map: path → best TraversalEntry (highest score wins)
  const resultMap = new Map<string, TraversalEntry>();

  // BFS queue entries: [path, score, source, depth, parentDepth]
  type QueueEntry = [
    path: string,
    score: number,
    source: TraversalEntry["source"],
    depth: number,
    parentDepth: number,
  ];
  const queue: QueueEntry[] = [];

  // Seed the queue with direct hits
  for (const seed of seeds) {
    if (!graph.has(seed.path)) continue;
    if (!visited.has(seed.path)) {
      visited.add(seed.path);
      queue.push([seed.path, seed.score, "direct", 0, 0]);
    }
  }

  while (queue.length > 0 && resultMap.size < maxSections) {
    const entry = queue.shift();
    if (!entry) break;
    const [path, score, source, depth, parentDepth] = entry;

    const node = graph.get(path);
    if (!node) continue;

    // Build anchor terms for sibling overlap scoring
    const anchorTerms = new Set<string>([
      ...node.meta.tags.map((t) => t.toLowerCase()),
      ...tokenize(node.meta.title),
      ...tokenize(node.meta.summary),
    ]);

    // Collect cross-refs discovered from this node
    const discoveredRefs = resolveCrossRefs ? [...node.crossRefs] : [];

    // Store in result map; keep highest score if revisited via multiple paths
    const existing = resultMap.get(path);
    if (!existing || score > existing.score) {
      resultMap.set(path, { path, score, source, discoveredRefs });
    }

    // --- Child expansion (downward) ---
    if (depth < clampedMaxDepth) {
      for (const childPath of node.children) {
        if (visited.has(childPath)) continue;
        const childNode = graph.get(childPath);
        if (!childNode) continue;

        // Score child: direct query match or inherited fraction of parent score
        const directChildScore = scoreNode(childNode, queryTerms);
        const childScore =
          directChildScore > 0
            ? directChildScore
            : score * childScoreFraction;

        visited.add(childPath);
        queue.push([childPath, childScore, "child", depth + 1, parentDepth]);
      }
    }

    // --- Parent backlinks (upward) ---
    if (includeParents && node.parent && parentDepth < clampedMaxParentDepth) {
      const parentPath = node.parent;
      if (!visited.has(parentPath)) {
        const parentNode = graph.get(parentPath);
        if (parentNode) {
          const directParentScore = scoreNode(parentNode, queryTerms);
          const parentScore =
            directParentScore > 0
              ? directParentScore
              : score * parentScoreFraction;

          visited.add(parentPath);
          queue.push([
            parentPath,
            parentScore,
            "parent",
            depth,
            parentDepth + 1,
          ]);
        }
      }
    }

    // --- Cross-reference resolution ---
    if (resolveCrossRefs) {
      for (const refPath of node.crossRefs) {
        if (visited.has(refPath)) continue;
        const refNode = graph.get(refPath);
        if (!refNode) continue;

        // Only pull the cross-ref if it has non-trivial query relevance
        const refScore = scoreNode(refNode, queryTerms);
        if (refScore > 0 || queryTerms.length === 0) {
          const xrefScore =
            refScore > 0 ? refScore * crossRefScoreFraction : score * crossRefScoreFraction;
          visited.add(refPath);
          queue.push([refPath, xrefScore, "xref", depth, parentDepth]);
        }
      }
    }

    // --- Sibling pulls ---
    // Only pull siblings that share meaningful overlap with anchor terms
    // and with the query — avoids fetching unrelated branches
    for (const sibPath of node.siblings) {
      if (visited.has(sibPath)) continue;
      const sibNode = graph.get(sibPath);
      if (!sibNode) continue;

      const sibRelevance = scoreSiblingRelevance(sibNode, anchorTerms, queryTerms);
      if (sibRelevance <= 0) continue; // score=0 → no overlap, skip

      // Only include if overlap exceeds threshold
      const sibTerms = new Set<string>([
        ...sibNode.meta.tags.map((t) => t.toLowerCase()),
        ...tokenize(sibNode.meta.title),
        ...tokenize(sibNode.meta.summary),
      ]);
      const overlap = termOverlap(anchorTerms, sibTerms);
      if (overlap < siblingOverlapThreshold) continue;

      const sibScore = sibRelevance * siblingScoreFraction;
      visited.add(sibPath);
      queue.push([sibPath, sibScore, "sibling", depth, parentDepth]);
    }
  }

  // Sort by score descending
  const results = [...resultMap.values()].sort((a, b) => b.score - a.score);
  return results.slice(0, maxSections);
}

// ---------------------------------------------------------------------------
// Token-budget packing + content loading
// ---------------------------------------------------------------------------

/**
 * Load content for traversal entries and pack into the token budget.
 *
 * Sections are read from disk in score order. If a section exceeds the
 * remaining budget, a truncated version is included when >200 tokens remain.
 * Populates `relatedSectionPaths` on each returned section from the
 * `discoveredRefs` collected during traversal.
 *
 * @param cwd      Project root.
 * @param entries  Ordered traversal entries from `traverseSectionGraph`.
 * @param maxTokens Total token budget.
 * @returns        Packed sections with content and relatedSectionPaths.
 */
export async function packGraphSections(
  cwd: string,
  entries: TraversalEntry[],
  maxTokens: number,
): Promise<RetrievedSection[]> {
  const results: RetrievedSection[] = [];
  let usedTokens = 0;

  for (const entry of entries) {
    if (usedTokens >= maxTokens) break;

    let content: string;
    try {
      const fullPath = sectionPath(cwd, entry.path);
      content = await readFile(fullPath, "utf-8");
    } catch {
      continue; // missing or unreadable — skip
    }

    const tokens = estimateTokens(content);

    if (usedTokens + tokens > maxTokens) {
      const remaining = maxTokens - usedTokens;
      if (remaining > 200) {
        const truncated =
          content.slice(0, remaining * 4) + "\n\n[... section truncated ...]";
        const node: RetrievedSection = {
          path: entry.path,
          title: entry.path, // title loaded from graph below
          content: truncated,
          tokens: remaining,
          score: entry.score,
        };
        results.push(node);
        usedTokens += remaining;
      }
      break;
    }

    const node: RetrievedSection = {
      path: entry.path,
      title: entry.path,
      content,
      tokens,
      score: entry.score,
    };
    results.push(node);
    usedTokens += tokens;
  }

  return results;
}

// ---------------------------------------------------------------------------
// Main public entry point
// ---------------------------------------------------------------------------

/**
 * Build a section graph and run hierarchical traversal + cross-reference weaving.
 *
 * This is the primary API for graph-aware section retrieval:
 *  1. Builds the bidirectional DAG from manifest metadata.
 *  2. Scores all sections against the query using TF-IDF heuristics.
 *  3. Seeds traversal from the top direct matches.
 *  4. Expands children (depth-limited), parent backlinks, cross-refs, siblings.
 *  5. Packs the result within the token budget.
 *
 * The returned sections include `relatedSectionPaths` populated from
 * cross-references discovered during traversal, enabling the weaver to
 * chain further retrieval if needed.
 *
 * @param manifest  Loaded genome manifest.
 * @param query     Natural-language task description.
 * @param cwd       Project root (genome at `.ashlrcode/genome/`).
 * @param opts      Traversal configuration (must include `maxTokens`).
 * @returns         Retrieved sections sorted by score descending.
 */
export async function buildAndTraverseGraph(
  manifest: GenomeManifest,
  query: string,
  cwd: string,
  opts: GraphTraversalOptions,
): Promise<RetrievedSection[]> {
  if (!manifest || manifest.sections.length === 0) return [];

  const graph = buildSectionGraph(manifest);

  // Tokenise query
  const queryTerms = tokenize(query);

  // Score all sections directly to find seeds
  const seeds: Array<{ path: string; score: number }> = [];
  for (const [path, node] of graph) {
    const score =
      queryTerms.length > 0 ? scoreNode(node, queryTerms) : 0;
    if (score > 0) seeds.push({ path, score });
  }

  // Sort seeds by score descending; cap seed set to top-25 to bound complexity
  seeds.sort((a, b) => b.score - a.score);
  const topSeeds = seeds.slice(0, 25);

  if (topSeeds.length === 0) {
    // No query match — return empty (caller falls back to core sections)
    return [];
  }

  // Traverse the graph
  const entries = traverseSectionGraph(graph, topSeeds, queryTerms, opts);

  // Load content and pack within token budget
  const sections = await packGraphSections(cwd, entries, opts.maxTokens);

  // Populate titles from graph metadata (packGraphSections uses path as placeholder)
  for (const section of sections) {
    const node = graph.get(section.path);
    if (node) section.title = node.meta.title;
  }

  // Attach relatedSectionPaths from cross-references discovered during traversal
  const entryMap = new Map(entries.map((e) => [e.path, e]));
  for (const section of sections) {
    const entry = entryMap.get(section.path);
    if (entry && entry.discoveredRefs.length > 0) {
      section.relatedSectionPaths = entry.discoveredRefs;
    }
  }

  return sections;
}

// ---------------------------------------------------------------------------
// Exports for testing
// ---------------------------------------------------------------------------

export type { TraversalEntry };
