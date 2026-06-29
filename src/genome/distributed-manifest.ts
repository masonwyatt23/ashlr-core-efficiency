/**
 * Distributed Genome Manifest — multi-workspace / monorepo support.
 *
 * Enables multiple workspaces (services, packages, apps) to each host their
 * own `.ashlrcode/genome/` and have their sections merged at retrieval time
 * into a single unified `GenomeManifest`.
 *
 * Key capabilities:
 *  - `resolveGenomeDirs`       — walk a directory tree (up to `depth` levels)
 *                                finding every `.ashlrcode/genome` directory.
 *  - `loadDistributedManifest` — load & merge manifests from multiple roots,
 *                                deduplicating on (project, section_id / path),
 *                                tracking origin cwd per section in metadata.
 *  - Version bookkeeping       — the merged manifest's `version` field is
 *                                incremented once per unique root merged in
 *                                (beyond the first), giving callers a stable
 *                                monotonic indicator that the manifest is a
 *                                composite.
 *
 * Section metadata extension:
 *  Each `SectionMeta` in the merged manifest may carry an `_origin` string
 *  field (the absolute cwd from which it was loaded).  This is stored on the
 *  object as a non-enumerable-compatible plain field so it travels through
 *  JSON.stringify if callers round-trip the manifest.
 */

import { existsSync } from "fs";
import { readdir } from "fs/promises";
import { join, resolve } from "path";
import { loadManifest } from "./manifest.ts";
import type { GenomeManifest, SectionMeta } from "./manifest.ts";

// ---------------------------------------------------------------------------
// Extended section metadata
// ---------------------------------------------------------------------------

/**
 * A `SectionMeta` augmented with origin provenance.
 * The `_origin` field records the absolute `cwd` from which this section
 * was loaded.  It is set by `loadDistributedManifest` and is transparently
 * round-tripped through JSON.
 */
export interface DistributedSectionMeta extends SectionMeta {
  /** Absolute path of the workspace root that contributed this section. */
  _origin?: string;
}

// ---------------------------------------------------------------------------
// DistributedManifest interface
// ---------------------------------------------------------------------------

/**
 * A `GenomeManifest` produced by merging sections from multiple workspace
 * roots.  Carries additional provenance fields to track which roots were
 * merged.
 *
 * Note: `version` is widened to `number` here because merging increments it
 * beyond the literal `1` that `GenomeManifest` declares.
 */
export interface DistributedManifest extends Omit<GenomeManifest, "sections" | "version"> {
  /** Widened from the base literal `1` — incremented once per contributing additional root. */
  version: number;
  sections: DistributedSectionMeta[];
  /**
   * The list of workspace roots that were merged to produce this manifest.
   * Ordered by the order in which they were supplied to
   * `loadDistributedManifest`.
   */
  _mergedRoots: string[];
}

// ---------------------------------------------------------------------------
// Directory discovery
// ---------------------------------------------------------------------------

/**
 * Walk the immediate children of `cwd` up to `depth` levels looking for
 * subdirectories that contain a `.ashlrcode/genome` directory (i.e. that
 * have a genome).  Also always checks `cwd` itself.
 *
 * This is intentionally a breadth-first scan of direct children (packages,
 * apps, services) rather than a deep recursive descent, to keep it fast on
 * large monorepos.  Pass `depth=1` for a flat packages/ layout; `depth=2`
 * or more for nested workspaces.
 *
 * @param cwd    Monorepo root (or any directory to start from).
 * @param depth  How many directory levels to descend.  Defaults to 3.
 * @returns Absolute paths of directories that contain `.ashlrcode/genome`.
 */
export async function resolveGenomeDirs(cwd: string, depth: number = 3): Promise<string[]> {
  const abs = resolve(cwd);
  const found: string[] = [];

  async function walk(dir: string, remaining: number): Promise<void> {
    // Check if this directory itself has a genome
    const genomeDir = join(dir, ".ashlrcode", "genome");
    if (existsSync(genomeDir)) {
      found.push(dir);
    }

    if (remaining <= 0) return;

    // Scan children
    let entries: string[];
    try {
      const dirents = await readdir(dir, { withFileTypes: true });
      entries = dirents.filter((d) => d.isDirectory() && !d.name.startsWith(".") && d.name !== "node_modules").map((d) => join(dir, d.name));
    } catch {
      return;
    }

    await Promise.all(entries.map((child) => walk(child, remaining - 1)));
  }

  await walk(abs, depth);

  // Deduplicate (resolve handles symlinks consistently) and return stable order
  const seen = new Set<string>();
  const result: string[] = [];
  for (const p of found) {
    const r = resolve(p);
    if (!seen.has(r)) {
      seen.add(r);
      result.push(r);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Merge helpers
// ---------------------------------------------------------------------------

/**
 * Produce a dedup key for a section that is stable across manifests.
 * We prefer a compound key of (project, sectionPath) so that two workspaces
 * with sections at the same relative path (e.g. "vision/north-star.md") are
 * kept distinct.
 */
function sectionKey(project: string, section: SectionMeta): string {
  return `${project}::${section.path}`;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Load genome manifests from multiple workspace roots and merge them into a
 * single `DistributedManifest`.
 *
 * Merge semantics:
 *  - Sections from the **first** root with a valid manifest are used as the
 *    base (its `project`, `generation`, `fitnessHistory`, etc. become the
 *    base manifest fields).
 *  - Sections from subsequent roots are appended, with deduplication on the
 *    `(project, path)` key — if both workspaces define a section at the same
 *    path within the same project name the first one wins.
 *  - Each section receives an `_origin` field set to the absolute cwd that
 *    contributed it.
 *  - The merged manifest's `version` field is incremented by the number of
 *    additional (non-base) roots that contributed at least one section.
 *
 * @param cwds  Array of workspace roots to load manifests from.  Roots where
 *              no manifest is found are silently skipped.
 * @returns     The merged `DistributedManifest`, or `null` if no root
 *              produced a readable manifest.
 */
export async function loadDistributedManifest(cwds: string[]): Promise<DistributedManifest | null> {
  if (cwds.length === 0) return null;

  // Load all manifests in parallel, but preserve the *supplied* order.
  // `Promise.all` resolves elements positionally, so indexing back into the
  // input order keeps the "first root with a valid manifest" semantics and
  // the documented `_mergedRoots` ordering deterministic regardless of which
  // filesystem read happens to finish first.
  const settled = await Promise.all(
    cwds.map(async (cwd) => {
      const r = resolve(cwd);
      const m = await loadManifest(r);
      return m ? { cwd: r, manifest: m } : null;
    }),
  );
  const loaded: Array<{ cwd: string; manifest: GenomeManifest }> = settled.filter(
    (x): x is { cwd: string; manifest: GenomeManifest } => x !== null,
  );

  if (loaded.length === 0) return null;

  // Base manifest comes from the first root that loaded successfully
  const [base, ...rest] = loaded;
  const baseCwd = base!.cwd;
  const baseManifest = base!.manifest;

  // Start with a copy of the base sections, annotated with origin
  const seen = new Map<string, true>();
  const mergedSections: DistributedSectionMeta[] = [];

  for (const s of baseManifest.sections) {
    const key = sectionKey(baseManifest.project, s);
    if (!seen.has(key)) {
      seen.set(key, true);
      mergedSections.push({ ...s, _origin: baseCwd });
    }
  }

  // Track how many additional roots actually contributed sections
  let additionalRootsContributed = 0;
  const mergedRoots: string[] = [baseCwd];

  for (const { cwd, manifest } of rest) {
    let contributed = false;
    for (const s of manifest.sections) {
      const key = sectionKey(manifest.project, s);
      if (!seen.has(key)) {
        seen.set(key, true);
        mergedSections.push({ ...s, _origin: cwd });
        contributed = true;
      }
    }
    if (contributed) {
      additionalRootsContributed++;
    }
    mergedRoots.push(cwd);
  }

  // Build the merged manifest
  const now = new Date().toISOString();
  const merged: DistributedManifest = {
    // Spread base manifest fields
    ...baseManifest,
    // Bump version once per additional root that contributed sections
    version: baseManifest.version + additionalRootsContributed,
    sections: mergedSections,
    updatedAt: now,
    _mergedRoots: mergedRoots,
  };

  return merged;
}
