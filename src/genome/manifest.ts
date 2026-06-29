/**
 * Genome manifest — index and metadata for all genome sections.
 *
 * The manifest tracks every section file, its tags, summary, token count,
 * and the current generation number. Stored at `.ashlrcode/genome/manifest.json`.
 */

import { existsSync } from "fs";
import { mkdir, readFile, rename, writeFile } from "fs/promises";
import { dirname, join, sep } from "path";
import { estimateTokensFromString as estimateTokens } from "../tokens/index.ts";
import { CURRENT_SCHEMA_VERSION, upgradeManifestOnLoad } from "./manifest-versioning.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SectionMeta {
  /** Relative path within genome dir, e.g. "vision/north-star.md" */
  path: string;
  /** Human-readable title */
  title: string;
  /** One-line summary used for retrieval matching */
  summary: string;
  /** Keywords for retrieval scoring */
  tags: string[];
  /** Estimated token count (chars / 4) */
  tokens: number;
  /** ISO timestamp of last modification */
  updatedAt: string;
  /**
   * Optional parent section path for hierarchical organisation.
   * E.g. a section at "architecture/layer-api.md" may set
   * parentId = "architecture/overview.md" so that retrieving the
   * parent automatically pulls this child section too.
   */
  parentId?: string;
}

export interface GenerationMeta {
  number: number;
  milestone: string;
  startedAt: string;
  endedAt?: string;
}

export interface QuantizationStats {
  /** Total size of all quantized embeddings in bytes */
  totalSize: number;
  /** Total size of all original float32 embeddings in bytes */
  originalSize: number;
  /** Ratio of quantized to original size (e.g. 0.25 for 4× compression) */
  compressionRatio: number;
  /** Mean cosine-similarity delta between original and reconstructed embeddings */
  avgSimilarityDelta: number;
}

export interface GenomeManifest {
  /** Schema version for forward compat */
  version: 1;
  /**
   * Versioning-system schema version.  Added by the manifest-versioning layer;
   * absent in legacy files (treated as 1 on load).
   */
  schemaVersion?: number;
  /** ISO timestamp of last migration run (set by manifest-versioning layer). */
  migratedAt?: string;
  /** Human-readable migration audit trail (set by manifest-versioning layer). */
  migrationAudit?: string[];
  /** Project name (informational) */
  project: string;
  /** All genome sections */
  sections: SectionMeta[];
  /** Current generation info */
  generation: GenerationMeta;
  /** Fitness scores per generation */
  fitnessHistory: Array<{ generation: number; scores: Record<string, number> }>;
  /** ISO timestamp */
  createdAt: string;
  updatedAt: string;
  /**
   * Statistics from the most recent quantized embedding pass.
   * Only present when updateEmbeddings was called with { quantize: true }.
   */
  quantizationStats?: QuantizationStats;
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const GENOME_DIR = ".ashlrcode/genome";
const MANIFEST_FILE = "manifest.json";

export function genomeDir(cwd: string): string {
  return join(cwd, GENOME_DIR);
}

export function manifestPath(cwd: string): string {
  return join(genomeDir(cwd), MANIFEST_FILE);
}

export function sectionPath(cwd: string, relativePath: string): string {
  const resolved = join(genomeDir(cwd), relativePath);
  // Prevent path traversal — resolved path must stay within genome dir.
  // Append platform separator to prevent "genome-evil" matching "genome" prefix.
  // Must use `path.sep` (not hardcoded "/") because `join` on Windows produces
  // "\"-separated paths, so appending "/" would mismatch every startsWith check.
  const gDir = genomeDir(cwd) + sep;
  if (!resolved.startsWith(gDir)) {
    throw new Error(`Invalid section path: ${relativePath} escapes genome directory`);
  }
  return resolved;
}

// ---------------------------------------------------------------------------
// Token estimation — imported from the single source of truth above.
// Re-exported so existing consumers keep working without an import change.
// ---------------------------------------------------------------------------

export { estimateTokens };

// ---------------------------------------------------------------------------
// Load / Save
// ---------------------------------------------------------------------------

export function genomeExists(cwd: string): boolean {
  return existsSync(manifestPath(cwd));
}

export async function loadManifest(cwd: string): Promise<GenomeManifest | null> {
  const path = manifestPath(cwd);
  if (!existsSync(path)) return null;

  try {
    const raw = await readFile(path, "utf-8");
    const parsed = JSON.parse(raw) as GenomeManifest;

    // Auto-upgrade if the stored schema version is behind the current version.
    const storedVersion = typeof parsed.schemaVersion === "number" ? parsed.schemaVersion : 1;
    if (storedVersion < CURRENT_SCHEMA_VERSION) {
      const upgraded = await upgradeManifestOnLoad(cwd);
      return upgraded ?? parsed;
    }

    // Backfill schemaVersion on legacy manifests that lack it
    if (parsed.schemaVersion === undefined) {
      parsed.schemaVersion = 1;
    }

    return parsed;
  } catch {
    // Corrupt or partially written manifest — treat as missing
    return null;
  }
}

export async function saveManifest(cwd: string, manifest: GenomeManifest): Promise<void> {
  const dir = genomeDir(cwd);
  if (!existsSync(dir)) {
    await mkdir(dir, { recursive: true });
  }
  manifest.updatedAt = new Date().toISOString();
  // Atomic write: write to temp file then rename (safe on POSIX)
  const target = manifestPath(cwd);
  const tmp = target + ".tmp";
  try {
    await writeFile(tmp, JSON.stringify(manifest, null, 2), "utf-8");
    await rename(tmp, target);
  } catch (e) {
    // Clean up orphaned temp file on failure (e.g., disk full)
    const { unlink } = await import("fs/promises");
    await unlink(tmp).catch(() => {});
    throw e;
  }
}

// Serialize manifest writes to prevent concurrent read-modify-write races.
// Keyed by cwd so different projects don't block each other.
const manifestLocks = new Map<string, Promise<unknown>>();

/**
 * Update specific fields of the manifest atomically (read-modify-write).
 * Serialized: concurrent calls for the same cwd wait in queue.
 */
export async function updateManifest(
  cwd: string,
  updater: (manifest: GenomeManifest) => void,
): Promise<GenomeManifest> {
  const key = genomeDir(cwd);
  const prev = manifestLocks.get(key) ?? Promise.resolve();

  const current = prev.then(async () => {
    const manifest = await loadManifest(cwd);
    if (!manifest) throw new Error("No genome found. Run /genome init first.");
    updater(manifest);
    await saveManifest(cwd, manifest);
    return manifest;
  });

  const settled = current.catch(() => {});
  manifestLocks.set(key, settled);
  // Clean up lock entry when chain settles to prevent unbounded growth
  settled.then(() => {
    if (manifestLocks.get(key) === settled) manifestLocks.delete(key);
  });
  return current;
}

// ---------------------------------------------------------------------------
// Section CRUD
// ---------------------------------------------------------------------------

/**
 * Read a genome section's content by its relative path.
 */
export async function readSection(cwd: string, relativePath: string): Promise<string | null> {
  const path = sectionPath(cwd, relativePath);
  if (!existsSync(path)) return null;
  return readFile(path, "utf-8");
}

/**
 * Write a genome section and update its manifest entry.
 */
export async function writeSection(
  cwd: string,
  relativePath: string,
  content: string,
  meta: Omit<SectionMeta, "path" | "tokens" | "updatedAt">,
): Promise<void> {
  const fullPath = sectionPath(cwd, relativePath);
  const dir = dirname(fullPath);
  if (!existsSync(dir)) {
    await mkdir(dir, { recursive: true });
  }

  await writeFile(fullPath, content, "utf-8");

  // Update manifest
  await updateManifest(cwd, (m) => {
    const now = new Date().toISOString();
    const existing = m.sections.findIndex((s) => s.path === relativePath);
    const entry: SectionMeta = {
      path: relativePath,
      title: meta.title,
      summary: meta.summary,
      tags: meta.tags,
      tokens: estimateTokens(content),
      updatedAt: now,
    };

    if (existing >= 0) {
      m.sections[existing] = entry;
    } else {
      m.sections.push(entry);
    }
  });
}

/**
 * Get total token count across all genome sections.
 */
export function totalGenomeTokens(manifest: GenomeManifest): number {
  return manifest.sections.reduce((sum, s) => sum + s.tokens, 0);
}

/**
 * Create a fresh empty manifest.
 */
export function createEmptyManifest(project: string): GenomeManifest {
  const now = new Date().toISOString();
  return {
    version: 1,
    schemaVersion: CURRENT_SCHEMA_VERSION,
    project,
    sections: [],
    generation: {
      number: 1,
      milestone: "",
      startedAt: now,
    },
    fitnessHistory: [],
    createdAt: now,
    updatedAt: now,
  };
}
