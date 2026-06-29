/**
 * Embedding Cache Coherence — cross-request stale index detection and incremental updates.
 *
 * Builds a cache-awareness layer for genome embeddings that:
 *  1. Stores metadata alongside the index (builtAt, manifestHash, sectionCount, quantDepth, modelTier).
 *  2. Detects stale indices by comparing manifest hash + section count vs stored metadata.
 *  3. Applies incremental updates for small section additions rather than full rebuilds.
 *  4. Validates index freshness on LocalContextWindow.injectGenome() calls.
 *  5. Enables cross-agent cache coordination via the session-log JSONL.
 *
 * Stale detection rules:
 *  - AGE_STALE:        builtAt is older than maxAgeMs (default: 1 hour)
 *  - HASH_MISMATCH:    manifest content hash changed
 *  - SECTIONS_ADDED:   manifest has more sections than index knows about
 *  - SECTIONS_REMOVED: manifest has fewer sections → full rebuild required
 *  - FRESH:            none of the above apply
 *
 * Incremental update strategy:
 *  - If N newly added sections <= incrementalThreshold (default 5): call addEntry() per section
 *  - If N > incrementalThreshold OR sections were removed: full rebuild
 */

import { createHash } from "crypto";
import { existsSync } from "fs";
import { mkdir, readFile, writeFile } from "fs/promises";
import { join } from "path";
import type { GenomeManifest, SectionMeta } from "./manifest.ts";
import { genomeDir } from "./manifest.ts";
import { appendJsonl, readJsonl } from "./jsonl.ts";
import type { ModelTier } from "./embedding-router.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Metadata stored alongside the embedding index.
 * Persisted to ~/.ashlr/genome/embedding-cache-meta.json (or .ashlrcode/genome/).
 */
export interface EmbeddingCacheMetadata {
  /** ISO timestamp when the index was last built */
  builtAt: string;
  /** SHA-256 hash of the manifest JSON (stable across re-reads of same data) */
  manifestHash: string;
  /** Number of sections in the manifest when the index was built */
  sectionCount: number;
  /** Bit-depth used for quantization (8 or 16) */
  quantDepth: 8 | 16;
  /** Embedding model tier used to build the index */
  modelTier: ModelTier;
  /** Paths of sections indexed (for incremental diff) */
  indexedSectionPaths: string[];
}

/** Staleness classification result. */
export type StalenessReason =
  | "FRESH"
  | "AGE_STALE"
  | "HASH_MISMATCH"
  | "SECTIONS_ADDED"
  | "SECTIONS_REMOVED"
  | "NO_METADATA";

export interface FreshnessResult {
  reason: StalenessReason;
  /** True when the index can be used as-is */
  isFresh: boolean;
  /** Sections present in manifest but absent from the index */
  newSections: SectionMeta[];
  /** Section paths present in index but absent from manifest */
  removedSectionPaths: string[];
  /** Age of the index in milliseconds (0 when no metadata exists) */
  ageMs: number;
  /** Stored metadata (null when no metadata file exists) */
  metadata: EmbeddingCacheMetadata | null;
}

/** Audit record emitted when incremental updates are applied. */
export interface IncrementalUpdateAuditRecord {
  id: string;
  type: "incremental_add" | "full_rebuild";
  timestamp: string;
  cwd: string;
  sectionsAdded: number;
  sectionsRemoved: number;
  previousSectionCount: number;
  newSectionCount: number;
  reason: StalenessReason;
  elapsedMs: number;
}

/** Cross-agent embedding operation record written to the session log. */
export interface AgentEmbeddingOpRecord {
  id: string;
  type: "embed_sections";
  timestamp: string;
  cwd: string;
  sectionPaths: string[];
  modelTier: ModelTier;
  quantDepth: 8 | 16;
  agentId: string;
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const CACHE_META_FILE = "embedding-cache-meta.json";
const INCREMENTAL_AUDIT_FILE = "embedding-incremental-audit.jsonl";
const AGENT_SESSION_LOG_FILE = "agent-embedding-session-log.jsonl";
const MAX_AGE_MS_DEFAULT = 60 * 60 * 1000; // 1 hour
const INCREMENTAL_THRESHOLD_DEFAULT = 5;

function cacheMetaPath(cwd: string): string {
  return join(genomeDir(cwd), CACHE_META_FILE);
}

function incrementalAuditPath(cwd: string): string {
  return join(genomeDir(cwd), "evolution", INCREMENTAL_AUDIT_FILE);
}

function agentSessionLogPath(cwd: string): string {
  return join(genomeDir(cwd), "evolution", AGENT_SESSION_LOG_FILE);
}

// ---------------------------------------------------------------------------
// Manifest hashing
// ---------------------------------------------------------------------------

/**
 * Compute a stable SHA-256 hash of a manifest's content.
 *
 * The hash is computed over a canonical JSON representation that includes
 * section paths, summaries, tags, and updatedAt timestamps — fields that
 * change when content changes but are stable across re-serialization.
 * We explicitly exclude the top-level `updatedAt` field (which changes on
 * every save) to avoid false positives from metadata-only writes.
 */
export function hashManifest(manifest: GenomeManifest): string {
  const canonical = {
    project: manifest.project,
    sections: manifest.sections.map((s) => ({
      path: s.path,
      title: s.title,
      summary: s.summary,
      tags: [...s.tags].sort(),
      tokens: s.tokens,
      updatedAt: s.updatedAt,
    })).sort((a, b) => a.path.localeCompare(b.path)),
    schemaVersion: manifest.schemaVersion ?? 1,
  };
  return createHash("sha256").update(JSON.stringify(canonical)).digest("hex");
}

// ---------------------------------------------------------------------------
// Metadata persistence
// ---------------------------------------------------------------------------

/**
 * Load stored embedding cache metadata.
 * Returns null when no metadata file exists.
 */
export async function loadCacheMetadata(cwd: string): Promise<EmbeddingCacheMetadata | null> {
  const path = cacheMetaPath(cwd);
  if (!existsSync(path)) return null;
  try {
    const raw = await readFile(path, "utf-8");
    return JSON.parse(raw) as EmbeddingCacheMetadata;
  } catch {
    return null;
  }
}

/**
 * Persist embedding cache metadata alongside the index.
 */
export async function saveCacheMetadata(
  cwd: string,
  metadata: EmbeddingCacheMetadata,
): Promise<void> {
  const dir = genomeDir(cwd);
  if (!existsSync(dir)) {
    await mkdir(dir, { recursive: true });
  }
  await writeFile(cacheMetaPath(cwd), JSON.stringify(metadata, null, 2), "utf-8");
}

/**
 * Create a fresh EmbeddingCacheMetadata from a manifest snapshot.
 */
export function createCacheMetadata(
  manifest: GenomeManifest,
  quantDepth: 8 | 16,
  modelTier: ModelTier,
): EmbeddingCacheMetadata {
  return {
    builtAt: new Date().toISOString(),
    manifestHash: hashManifest(manifest),
    sectionCount: manifest.sections.length,
    quantDepth,
    modelTier,
    indexedSectionPaths: manifest.sections.map((s) => s.path),
  };
}

// ---------------------------------------------------------------------------
// IndexFreshnessValidator
// ---------------------------------------------------------------------------

/**
 * Validates embedding index freshness against the current manifest state.
 *
 * Detection logic:
 *  1. No metadata → NO_METADATA (stale, full rebuild needed)
 *  2. Sections removed → SECTIONS_REMOVED (full rebuild required)
 *  3. Age > maxAgeMs → AGE_STALE (rebuild or incremental depending on diffs)
 *  4. Manifest hash changed → HASH_MISMATCH
 *  5. New sections added (within incremental threshold) → SECTIONS_ADDED
 *  6. All checks pass → FRESH
 *
 * Note: SECTIONS_REMOVED is checked before AGE_STALE because a removal
 * always requires a full rebuild regardless of age.
 */
export class IndexFreshnessValidator {
  private readonly maxAgeMs: number;

  constructor(maxAgeMs = MAX_AGE_MS_DEFAULT) {
    this.maxAgeMs = maxAgeMs;
  }

  /**
   * Validate the freshness of the embedding index for a given manifest.
   *
   * @param manifest  Current manifest (freshly loaded from disk)
   * @param metadata  Stored cache metadata (from loadCacheMetadata)
   * @param nowMs     Current timestamp in milliseconds (injectable for tests)
   */
  validate(
    manifest: GenomeManifest,
    metadata: EmbeddingCacheMetadata | null,
    nowMs = Date.now(),
  ): FreshnessResult {
    if (metadata === null) {
      return {
        reason: "NO_METADATA",
        isFresh: false,
        newSections: manifest.sections,
        removedSectionPaths: [],
        ageMs: 0,
        metadata: null,
      };
    }

    const builtAtMs = new Date(metadata.builtAt).getTime();
    const ageMs = nowMs - builtAtMs;

    const indexedSet = new Set(metadata.indexedSectionPaths);
    const manifestPaths = new Set(manifest.sections.map((s) => s.path));

    // Sections present in index but missing from manifest → deletion occurred
    const removedSectionPaths = metadata.indexedSectionPaths.filter(
      (p) => !manifestPaths.has(p),
    );

    // Sections in manifest but not in index → additions since last build
    const newSections = manifest.sections.filter((s) => !indexedSet.has(s.path));

    // Check section deletions first — always requires full rebuild
    if (removedSectionPaths.length > 0) {
      return {
        reason: "SECTIONS_REMOVED",
        isFresh: false,
        newSections,
        removedSectionPaths,
        ageMs,
        metadata,
      };
    }

    // Check for new sections before hash comparison — adding sections changes the hash,
    // so we classify this specifically as SECTIONS_ADDED (incremental add path).
    if (newSections.length > 0) {
      return {
        reason: "SECTIONS_ADDED",
        isFresh: false,
        newSections,
        removedSectionPaths: [],
        ageMs,
        metadata,
      };
    }

    // Check age staleness
    if (ageMs > this.maxAgeMs) {
      return {
        reason: "AGE_STALE",
        isFresh: false,
        newSections: [],
        removedSectionPaths: [],
        ageMs,
        metadata,
      };
    }

    // Check manifest hash (content changed in existing sections)
    const currentHash = hashManifest(manifest);
    if (currentHash !== metadata.manifestHash) {
      return {
        reason: "HASH_MISMATCH",
        isFresh: false,
        newSections: [],
        removedSectionPaths: [],
        ageMs,
        metadata,
      };
    }

    return {
      reason: "FRESH",
      isFresh: true,
      newSections: [],
      removedSectionPaths: [],
      ageMs,
      metadata,
    };
  }
}

// ---------------------------------------------------------------------------
// IncrementalEmbeddingUpdater
// ---------------------------------------------------------------------------

/**
 * Interface for the minimum API surface of QuantizedANNSearcher needed for
 * incremental updates. Using a structural interface avoids a circular import.
 */
export interface ANNSearcherLike {
  addEntry(id: string, embedding: number[]): void;
  buildIndex(entries: Array<{ id: string; embedding: number[] }>): void;
}

/**
 * Embedding provider callback — called per section to produce an embedding.
 * Matches the signature used by EmbeddingRouter.embed() results.
 */
export type EmbedFn = (sectionPath: string, content: string) => Promise<number[] | null>;

/**
 * Applies incremental embedding updates to an existing ANN index.
 *
 * For small additions (N <= threshold), calls addEntry() per section.
 * For large additions or deletions, triggers a full rebuild via the
 * provided embedAll callback.
 */
export class IncrementalEmbeddingUpdater {
  private readonly incrementalThreshold: number;

  constructor(incrementalThreshold = INCREMENTAL_THRESHOLD_DEFAULT) {
    this.incrementalThreshold = incrementalThreshold;
  }

  /**
   * Apply incremental updates based on a freshness result.
   *
   * @param cwd           Project root
   * @param searcher      The ANN searcher to update in-place
   * @param freshness     Result from IndexFreshnessValidator.validate()
   * @param manifest      Current manifest
   * @param embedFn       Async function: (sectionPath, content) → embedding
   * @param readSection   Async function: (cwd, relativePath) → content | null
   * @param embedAll      Async function for full rebuild: returns all (id, embedding) pairs
   * @param quantDepth    Bit depth for new metadata
   * @param modelTier     Model tier for new metadata
   * @returns             Audit record for the operation
   */
  async applyUpdates(
    cwd: string,
    searcher: ANNSearcherLike,
    freshness: FreshnessResult,
    manifest: GenomeManifest,
    embedFn: EmbedFn,
    readSection: (cwd: string, path: string) => Promise<string | null>,
    embedAll: () => Promise<Array<{ id: string; embedding: number[] }>>,
    quantDepth: 8 | 16,
    modelTier: ModelTier,
  ): Promise<IncrementalUpdateAuditRecord> {
    const t0 = Date.now();
    const previousSectionCount = freshness.metadata?.sectionCount ?? 0;
    const newSectionCount = manifest.sections.length;
    const sectionsAdded = freshness.newSections.length;
    const sectionsRemoved = freshness.removedSectionPaths.length;

    let updateType: "incremental_add" | "full_rebuild";

    if (
      freshness.reason !== "SECTIONS_REMOVED" &&
      sectionsAdded <= this.incrementalThreshold &&
      sectionsAdded > 0
    ) {
      // Incremental path: embed and add each new section
      updateType = "incremental_add";
      for (const section of freshness.newSections) {
        const content = await readSection(cwd, section.path);
        if (content === null) continue;
        const embedding = await embedFn(section.path, content);
        if (embedding === null) continue;
        searcher.addEntry(section.path, embedding);
      }
    } else {
      // Full rebuild path
      updateType = "full_rebuild";
      const allEntries = await embedAll();
      searcher.buildIndex(allEntries);
    }

    // Update stored metadata
    const newMetadata = createCacheMetadata(manifest, quantDepth, modelTier);
    await saveCacheMetadata(cwd, newMetadata);

    const elapsedMs = Date.now() - t0;

    // Emit audit record
    const auditRecord: IncrementalUpdateAuditRecord = {
      id: `icu-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      type: updateType,
      timestamp: new Date().toISOString(),
      cwd,
      sectionsAdded,
      sectionsRemoved,
      previousSectionCount,
      newSectionCount,
      reason: freshness.reason,
      elapsedMs,
    };

    await appendJsonl(incrementalAuditPath(cwd), auditRecord);

    return auditRecord;
  }
}

// ---------------------------------------------------------------------------
// CacheCoherence — single-call freshness check + auto-rebuild
// ---------------------------------------------------------------------------

export interface CacheCoherenceOptions {
  /** Maximum index age in milliseconds before rebuild. Default: 1 hour. */
  maxAgeMs?: number;
  /** Max new sections before switching from incremental to full rebuild. Default: 5. */
  incrementalThreshold?: number;
  /** When true, always rebuild even if the index is fresh. Default: false. */
  force?: boolean;
}

/**
 * Validates index freshness and auto-rebuilds/updates when stale.
 *
 * Called from LocalContextWindow.injectGenome() before semantic search.
 */
export class CacheCoherence {
  private readonly validator: IndexFreshnessValidator;
  private readonly updater: IncrementalEmbeddingUpdater;
  private readonly force: boolean;

  constructor(options: CacheCoherenceOptions = {}) {
    this.validator = new IndexFreshnessValidator(options.maxAgeMs ?? MAX_AGE_MS_DEFAULT);
    this.updater = new IncrementalEmbeddingUpdater(
      options.incrementalThreshold ?? INCREMENTAL_THRESHOLD_DEFAULT,
    );
    this.force = options.force ?? false;
  }

  /**
   * Ensure the embedding index is fresh before a genome search.
   *
   * If stale, applies incremental updates or triggers a full rebuild
   * depending on the staleness reason.
   *
   * @returns The freshness result (FRESH when no update was needed,
   *          or the pre-update staleness reason when an update ran).
   */
  async ensureFresh(
    cwd: string,
    manifest: GenomeManifest,
    searcher: ANNSearcherLike,
    embedFn: EmbedFn,
    readSection: (cwd: string, path: string) => Promise<string | null>,
    embedAll: () => Promise<Array<{ id: string; embedding: number[] }>>,
    quantDepth: 8 | 16 = 8,
    modelTier: ModelTier = "balanced",
    nowMs?: number,
  ): Promise<{ freshness: FreshnessResult; updated: boolean; auditRecord?: IncrementalUpdateAuditRecord }> {
    const metadata = await loadCacheMetadata(cwd);
    const freshness = this.validator.validate(manifest, metadata, nowMs);

    if (freshness.isFresh && !this.force) {
      return { freshness, updated: false };
    }

    // Stale — apply updates
    const auditRecord = await this.updater.applyUpdates(
      cwd,
      searcher,
      freshness,
      manifest,
      embedFn,
      readSection,
      embedAll,
      quantDepth,
      modelTier,
    );

    return { freshness, updated: true, auditRecord };
  }
}

// ---------------------------------------------------------------------------
// CrossAgentCacheSync
// ---------------------------------------------------------------------------

/**
 * Enables multiple agents working on the same project to share embedding
 * knowledge without re-embedding the same sections.
 *
 * When Agent A embeds sections 1–10, it writes an AgentEmbeddingOpRecord.
 * Agent B can read the session log and see which sections were recently
 * embedded, avoiding redundant work.
 */
export class CrossAgentCacheSync {
  private readonly agentId: string;

  constructor(agentId?: string) {
    this.agentId = agentId ?? `agent-${Date.now()}-${Math.random().toString(36).slice(2, 5)}`;
  }

  /**
   * Record that this agent embedded a set of sections.
   * Written to the shared session log so other agents can read it.
   */
  async recordEmbeddingOp(
    cwd: string,
    sectionPaths: string[],
    modelTier: ModelTier,
    quantDepth: 8 | 16,
  ): Promise<string> {
    const id = `aeo-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    const record: AgentEmbeddingOpRecord = {
      id,
      type: "embed_sections",
      timestamp: new Date().toISOString(),
      cwd,
      sectionPaths,
      modelTier,
      quantDepth,
      agentId: this.agentId,
    };
    await appendJsonl(agentSessionLogPath(cwd), record);
    return id;
  }

  /**
   * Read all embedding operations from other agents within the given time window.
   *
   * @param cwd         Project root
   * @param maxAgeMs    How far back to look (default: 10 minutes)
   * @param nowMs       Current timestamp (injectable for tests)
   */
  async readRecentOps(
    cwd: string,
    maxAgeMs = 10 * 60 * 1000,
    nowMs = Date.now(),
  ): Promise<AgentEmbeddingOpRecord[]> {
    const all = await readJsonl<AgentEmbeddingOpRecord>(agentSessionLogPath(cwd));
    const cutoff = nowMs - maxAgeMs;
    return all.filter((r) => {
      const ts = new Date(r.timestamp).getTime();
      return ts >= cutoff;
    });
  }

  /**
   * Check if a given set of section paths were recently embedded by any agent.
   * Returns the subset of paths that were recently embedded (i.e. no re-embed needed).
   *
   * @param cwd          Project root
   * @param sectionPaths Paths to check
   * @param maxAgeMs     Freshness window (default: 10 minutes)
   * @param nowMs        Timestamp override for tests
   */
  async getRecentlyEmbeddedPaths(
    cwd: string,
    sectionPaths: string[],
    maxAgeMs = 10 * 60 * 1000,
    nowMs = Date.now(),
  ): Promise<string[]> {
    const recentOps = await this.readRecentOps(cwd, maxAgeMs, nowMs);
    const recentlyEmbedded = new Set<string>();
    for (const op of recentOps) {
      for (const path of op.sectionPaths) {
        recentlyEmbedded.add(path);
      }
    }
    return sectionPaths.filter((p) => recentlyEmbedded.has(p));
  }

  /**
   * Check whether the index can be considered fresh based on recent agent activity.
   * If another agent embedded all sections within maxAgeMs, our index is likely fresh.
   *
   * @param cwd           Project root
   * @param sectionPaths  All paths that should be in the index
   * @param maxAgeMs      Freshness window (default: 10 minutes)
   * @param nowMs         Timestamp override for tests
   */
  async isIndexFreshFromAgentActivity(
    cwd: string,
    sectionPaths: string[],
    maxAgeMs = 10 * 60 * 1000,
    nowMs = Date.now(),
  ): Promise<boolean> {
    if (sectionPaths.length === 0) return true;
    const recentlyEmbedded = await this.getRecentlyEmbeddedPaths(
      cwd,
      sectionPaths,
      maxAgeMs,
      nowMs,
    );
    return recentlyEmbedded.length === sectionPaths.length;
  }

  /** Return this agent's ID. */
  get id(): string {
    return this.agentId;
  }
}

// ---------------------------------------------------------------------------
// Public convenience API
// ---------------------------------------------------------------------------

/**
 * Validate the freshness of the embedding index for a project.
 *
 * @param cwd       Project root
 * @param maxAgeMs  Maximum acceptable index age in milliseconds (default: 1 hour)
 */
export async function validateIndexFreshness(
  cwd: string,
  maxAgeMs = MAX_AGE_MS_DEFAULT,
  manifest?: GenomeManifest,
  nowMs?: number,
): Promise<FreshnessResult> {
  const metadata = await loadCacheMetadata(cwd);

  if (manifest === undefined) {
    // No manifest provided — can only check age and presence of metadata
    if (metadata === null) {
      return {
        reason: "NO_METADATA",
        isFresh: false,
        newSections: [],
        removedSectionPaths: [],
        ageMs: 0,
        metadata: null,
      };
    }
    const builtAtMs = new Date(metadata.builtAt).getTime();
    const ageMs = (nowMs ?? Date.now()) - builtAtMs;
    if (ageMs > maxAgeMs) {
      return {
        reason: "AGE_STALE",
        isFresh: false,
        newSections: [],
        removedSectionPaths: [],
        ageMs,
        metadata,
      };
    }
    return {
      reason: "FRESH",
      isFresh: true,
      newSections: [],
      removedSectionPaths: [],
      ageMs,
      metadata,
    };
  }

  const validator = new IndexFreshnessValidator(maxAgeMs);
  return validator.validate(manifest, metadata, nowMs);
}

/**
 * Rebuild the embedding index if it is stale.
 *
 * @param cwd       Project root
 * @param force     When true, rebuild even if index is fresh
 */
export async function rebuildIfStale(
  cwd: string,
  force = false,
  manifest?: GenomeManifest,
  searcher?: ANNSearcherLike,
  embedFn?: EmbedFn,
  readSection?: (cwd: string, path: string) => Promise<string | null>,
  embedAll?: () => Promise<Array<{ id: string; embedding: number[] }>>,
  quantDepth: 8 | 16 = 8,
  modelTier: ModelTier = "balanced",
  maxAgeMs = MAX_AGE_MS_DEFAULT,
): Promise<{ rebuilt: boolean; reason: StalenessReason }> {
  const metadata = await loadCacheMetadata(cwd);

  if (!force && metadata !== null && manifest !== undefined) {
    const validator = new IndexFreshnessValidator(maxAgeMs);
    const freshness = validator.validate(manifest, metadata);
    if (freshness.isFresh) {
      return { rebuilt: false, reason: "FRESH" };
    }

    if (searcher && embedFn && readSection && embedAll && manifest) {
      const coherence = new CacheCoherence({ maxAgeMs, force });
      const result = await coherence.ensureFresh(
        cwd,
        manifest,
        searcher,
        embedFn,
        readSection,
        embedAll,
        quantDepth,
        modelTier,
      );
      return { rebuilt: result.updated, reason: result.freshness.reason };
    }
  }

  if (force && manifest && searcher && embedAll) {
    const allEntries = await embedAll();
    searcher.buildIndex(allEntries);
    const newMetadata = createCacheMetadata(manifest, quantDepth, modelTier);
    await saveCacheMetadata(cwd, newMetadata);
    return { rebuilt: true, reason: "FRESH" };
  }

  return { rebuilt: false, reason: metadata === null ? "NO_METADATA" : "FRESH" };
}

/**
 * Return diagnostic statistics for the embedding index.
 *
 * @param cwd  Project root
 */
export async function getIndexStats(cwd: string): Promise<{
  hasMetadata: boolean;
  builtAt: string | null;
  sectionCount: number;
  quantDepth: 8 | 16 | null;
  modelTier: ModelTier | null;
  ageMs: number | null;
  indexedSectionPaths: string[];
  auditRecordCount: number;
}> {
  const metadata = await loadCacheMetadata(cwd);
  const auditRecords = await readJsonl<IncrementalUpdateAuditRecord>(incrementalAuditPath(cwd));

  if (metadata === null) {
    return {
      hasMetadata: false,
      builtAt: null,
      sectionCount: 0,
      quantDepth: null,
      modelTier: null,
      ageMs: null,
      indexedSectionPaths: [],
      auditRecordCount: auditRecords.length,
    };
  }

  const ageMs = Date.now() - new Date(metadata.builtAt).getTime();

  return {
    hasMetadata: true,
    builtAt: metadata.builtAt,
    sectionCount: metadata.sectionCount,
    quantDepth: metadata.quantDepth,
    modelTier: metadata.modelTier,
    ageMs,
    indexedSectionPaths: metadata.indexedSectionPaths,
    auditRecordCount: auditRecords.length,
  };
}
