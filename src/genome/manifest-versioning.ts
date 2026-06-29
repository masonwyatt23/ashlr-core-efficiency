/**
 * Genome Manifest Schema Versioning & Migration System
 *
 * Provides backward-compatible manifest format evolution with an audit trail.
 * When a manifest is loaded its schemaVersion is checked; if it is older than
 * CURRENT_SCHEMA_VERSION every registered migration step is run in sequence
 * before the caller receives the manifest.
 *
 * Adding a new field to GenomeManifest in the future:
 *   1. Bump CURRENT_SCHEMA_VERSION.
 *   2. Call registerMigration(prev, next, (m) => { /* back-fill field *\/ return m; }).
 *   3. The load path handles everything automatically.
 */

import { createHash } from "crypto";
import { existsSync } from "fs";
import { readFile } from "fs/promises";
import type { GenomeManifest } from "./manifest.ts";

// ---------------------------------------------------------------------------
// Version constants
// ---------------------------------------------------------------------------

/** Lowest schema version this code can read. */
export const MIN_SCHEMA_VERSION = 1;

/**
 * The schema version produced by this build.  Increment when a structural
 * (potentially breaking) change lands in GenomeManifest.
 */
export const CURRENT_SCHEMA_VERSION = 1;

// ---------------------------------------------------------------------------
// Enum (extensible — add entries here as versions land)
// ---------------------------------------------------------------------------

export enum ManifestSchemaVersion {
  /** Original format — all fields present in GenomeManifest as of v0.3.0. */
  v1 = 1,
  /**
   * Reserved for next breaking change.
   * Candidate: add SectionMeta.deprecated: boolean for archiving sections.
   */
  v2 = 2,
}

// ---------------------------------------------------------------------------
// Wrapper type
// ---------------------------------------------------------------------------

/**
 * Transport wrapper that carries schema provenance alongside the payload.
 * Stored in manifest.json as top-level fields (schemaVersion, migratedAt,
 * migrationAudit) rather than as a nested object — this lets existing readers
 * that don't know about versioning still parse the manifest without error.
 */
export interface ManifestVersioned<T> {
  schemaVersion: number;
  data: T;
  /** ISO timestamp of the last migration run (undefined if never migrated). */
  migratedAt?: string;
  /**
   * Human-readable log of each migration step that has run on this file,
   * e.g. ["v1→v2: added metadata.tags (2026-06-29T00:00:00.000Z)"].
   */
  migrationAudit?: string[];
}

// ---------------------------------------------------------------------------
// Migration registry
// ---------------------------------------------------------------------------

type MigrationFn = (manifest: GenomeManifest) => GenomeManifest;

interface MigrationEntry {
  from: number;
  to: number;
  fn: MigrationFn;
  description: string;
}

const registry: MigrationEntry[] = [];

/**
 * Register a migration function that transforms a manifest from one schema
 * version to the next.  Migrations must be idempotent and must not drop
 * fields that earlier versions rely on (additive only for non-breaking).
 *
 * @param from  Source schema version
 * @param to    Target schema version (must equal from + 1)
 * @param fn    Pure transform — receives a copy and should return the mutated copy
 * @param description  Short human-readable label for the audit trail
 */
export function registerMigration(
  from: number,
  to: number,
  fn: MigrationFn,
  description = `v${from}→v${to}`,
): void {
  if (to !== from + 1) {
    throw new Error(
      `registerMigration: migrations must be incremental (from=${from} to=${to}).  ` +
        `Only from+1 target is allowed.`,
    );
  }
  // Prevent duplicate registrations for the same step
  const duplicate = registry.find((e) => e.from === from && e.to === to);
  if (duplicate) {
    throw new Error(
      `registerMigration: a migration from v${from}→v${to} is already registered.`,
    );
  }
  registry.push({ from, to, fn, description });
}

// ---------------------------------------------------------------------------
// v1 → v2 placeholder migration (non-breaking)
// ---------------------------------------------------------------------------
// When ManifestSchemaVersion.v2 is finalised this registration should be
// updated with the real transform.  For now it is registered so the
// migration-chain code path is exercised and tests can verify wiring.

registerMigration(
  ManifestSchemaVersion.v1,
  ManifestSchemaVersion.v2,
  (manifest) => {
    // Placeholder: copy manifest unchanged.  The real v2 migration will add
    // SectionMeta.deprecated: boolean with a default of false on every section.
    return { ...manifest };
  },
  "v1→v2: reserved (no structural change yet)",
);

// ---------------------------------------------------------------------------
// Apply migration chain
// ---------------------------------------------------------------------------

/**
 * SHA-256 hex digest of the JSON-serialised manifest (deterministic key fields
 * only — excludes migrationAudit / migratedAt to avoid hash churn).
 */
function hashManifest(manifest: GenomeManifest): string {
  // Exclude audit fields so the hash reflects actual data content
  const { ...copy } = manifest as GenomeManifest & {
    migratedAt?: string;
    migrationAudit?: string[];
  };
  delete (copy as Record<string, unknown>).migratedAt;
  delete (copy as Record<string, unknown>).migrationAudit;
  return createHash("sha256").update(JSON.stringify(copy)).digest("hex").slice(0, 16);
}

/**
 * Run the registered migration chain from `from` up to (but not exceeding)
 * `to`.  Returns a new manifest object; the original is not mutated.
 *
 * Any schemaVersion / migrationAudit fields present on the input manifest
 * are preserved and extended.
 */
export function applyMigration(
  manifest: GenomeManifest,
  from: number,
  to: number,
): GenomeManifest {
  if (from === to) return manifest;
  if (from > to) {
    throw new Error(
      `applyMigration: downgrade path (from=${from} to=${to}) is not supported via applyMigration — use downgradeManifestOnSave instead.`,
    );
  }

  let current: GenomeManifest & {
    schemaVersion?: number;
    migratedAt?: string;
    migrationAudit?: string[];
  } = { ...manifest };

  // Carry forward any existing audit trail
  const audit: string[] = Array.isArray(current.migrationAudit)
    ? [...current.migrationAudit]
    : [];

  for (let v = from; v < to; v++) {
    const entry = registry.find((e) => e.from === v && e.to === v + 1);
    if (!entry) {
      throw new Error(
        `applyMigration: no migration registered for v${v}→v${v + 1}.  ` +
          `Register one with registerMigration() before loading manifests at this version.`,
      );
    }

    const beforeHash = hashManifest(current as GenomeManifest);
    current = entry.fn(current as GenomeManifest) as typeof current;
    const afterHash = hashManifest(current as GenomeManifest);
    const ts = new Date().toISOString();
    audit.push(`${entry.description} | before=${beforeHash} after=${afterHash} at=${ts}`);
  }

  current.schemaVersion = to;
  current.migratedAt = new Date().toISOString();
  current.migrationAudit = audit;

  return current as GenomeManifest;
}

// ---------------------------------------------------------------------------
// Version detection
// ---------------------------------------------------------------------------

/**
 * Read just the `schemaVersion` header from a manifest JSON file on disk
 * without parsing the entire document.
 *
 * @returns The schemaVersion number, or 1 if the field is absent (legacy
 *          manifests predating the versioning system default to v1).
 */
export async function getManifestVersion(file: string): Promise<number> {
  if (!existsSync(file)) {
    throw new Error(`getManifestVersion: file not found: ${file}`);
  }
  try {
    const raw = await readFile(file, "utf-8");
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const sv = parsed.schemaVersion;
    if (typeof sv === "number" && Number.isInteger(sv) && sv >= 1) {
      return sv;
    }
    // Legacy manifest: no schemaVersion → treat as v1
    return 1;
  } catch {
    // Unreadable / corrupt — treat as v1 so callers can surface a better error
    return 1;
  }
}

// ---------------------------------------------------------------------------
// Load-time upgrade
// ---------------------------------------------------------------------------

/**
 * Load a manifest from disk, automatically migrating it to CURRENT_SCHEMA_VERSION
 * if its stored schemaVersion is older.
 *
 * Called by `loadManifest()` in manifest.ts so all consumers get an up-to-date
 * manifest without any per-callsite migration logic.
 *
 * @returns The (possibly migrated) manifest, or null if the file does not exist.
 */
export async function upgradeManifestOnLoad(
  cwd: string,
): Promise<
  | (GenomeManifest & {
      schemaVersion?: number;
      migratedAt?: string;
      migrationAudit?: string[];
    })
  | null
> {
  const { manifestPath } = await import("./manifest.ts");
  const file = manifestPath(cwd);

  if (!existsSync(file)) return null;

  let raw: string;
  try {
    raw = await readFile(file, "utf-8");
  } catch {
    return null;
  }

  let manifest: GenomeManifest & {
    schemaVersion?: number;
    migratedAt?: string;
    migrationAudit?: string[];
  };

  try {
    manifest = JSON.parse(raw);
  } catch {
    return null;
  }

  const storedVersion: number =
    typeof manifest.schemaVersion === "number" ? manifest.schemaVersion : 1;

  if (storedVersion < CURRENT_SCHEMA_VERSION) {
    return applyMigration(
      manifest as GenomeManifest,
      storedVersion,
      CURRENT_SCHEMA_VERSION,
    ) as GenomeManifest & {
      schemaVersion?: number;
      migratedAt?: string;
      migrationAudit?: string[];
    };
  }

  // Ensure schemaVersion is always present on returned object
  if (manifest.schemaVersion === undefined) {
    manifest.schemaVersion = storedVersion;
  }

  return manifest;
}

// ---------------------------------------------------------------------------
// Save-time downgrade
// ---------------------------------------------------------------------------

/**
 * Produce a copy of the manifest serialised to an older schema version.
 * Used when writing to a location consumed by an older tool that does not
 * understand the current format.
 *
 * Currently the only supported downgrade target is v1 (identity — all fields
 * present in v1 are a strict subset of later versions).  Future downgrade
 * paths should strip fields that were added in later versions.
 *
 * @param manifest      Source manifest (at CURRENT_SCHEMA_VERSION or higher)
 * @param targetVersion Schema version to downgrade to
 */
export function downgradeManifestOnSave(
  manifest: GenomeManifest,
  targetVersion: number,
): GenomeManifest {
  if (targetVersion > CURRENT_SCHEMA_VERSION) {
    throw new Error(
      `downgradeManifestOnSave: targetVersion (${targetVersion}) is higher than CURRENT_SCHEMA_VERSION (${CURRENT_SCHEMA_VERSION}).`,
    );
  }

  // Clone deeply so the original is never mutated
  const copy = JSON.parse(JSON.stringify(manifest)) as GenomeManifest & {
    schemaVersion?: number;
    migratedAt?: string;
    migrationAudit?: string[];
  };

  // Always stamp the requested target version on the clone.
  // v1 format: strip fields that did not exist in v1.
  // As of this writing v1 IS the current version so nothing to strip yet.
  // When v2 lands, add stripping logic here guarded by targetVersion < 2.
  copy.schemaVersion = targetVersion;

  return copy as GenomeManifest;
}
