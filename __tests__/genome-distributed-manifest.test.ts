/**
 * Tests for the Distributed Genome Manifest system.
 *
 * Covers:
 *  - Merging sections from 2–3 tmp genome dirs with deduplication.
 *  - IDF re-computation across merged sections (no large-project bias).
 *  - Section origin tracking (_origin cwd in metadata).
 *  - Version bumping on merge.
 *  - resolveGenomeDirs directory discovery.
 *  - retrieveSectionsFromDist end-to-end.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { existsSync } from "fs";
import { mkdir, rm, writeFile } from "fs/promises";
import { join, resolve } from "path";
import {
  loadDistributedManifest,
  resolveGenomeDirs,
} from "../src/genome/distributed-manifest.ts";
import {
  createEmptyManifest,
  saveManifest,
  writeSection,
} from "../src/genome/manifest.ts";
import { retrieveSectionsFromDist } from "../src/genome/retriever.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TMP_BASE = join(import.meta.dir, ".tmp-dist-manifest");

async function makeWorkspace(label: string): Promise<string> {
  const dir = join(TMP_BASE, label + "-" + Date.now() + "-" + Math.random().toString(36).slice(2));
  await mkdir(join(dir, ".ashlrcode", "genome"), { recursive: true });
  return dir;
}

/** Write a minimal manifest + one section to a tmp workspace. */
async function seedWorkspace(
  cwd: string,
  project: string,
  sections: Array<{ path: string; title: string; summary: string; tags: string[]; content: string }>,
): Promise<void> {
  const manifest = createEmptyManifest(project);
  await saveManifest(cwd, manifest);

  for (const s of sections) {
    await writeSection(cwd, s.path, s.content, {
      title: s.title,
      summary: s.summary,
      tags: s.tags,
    });
  }
}

let workspaceDirs: string[] = [];

beforeEach(() => {
  workspaceDirs = [];
});

afterEach(async () => {
  for (const dir of workspaceDirs) {
    if (existsSync(dir)) {
      await rm(dir, { recursive: true, force: true });
    }
  }
  // Also clean up TMP_BASE if it exists
  if (existsSync(TMP_BASE)) {
    await rm(TMP_BASE, { recursive: true, force: true });
  }
});

async function ws(label: string): Promise<string> {
  const dir = await makeWorkspace(label);
  workspaceDirs.push(dir);
  return dir;
}

// ---------------------------------------------------------------------------
// loadDistributedManifest — basic merge
// ---------------------------------------------------------------------------

describe("loadDistributedManifest — basic merge", () => {
  test("returns null when cwds is empty", async () => {
    const result = await loadDistributedManifest([]);
    expect(result).toBeNull();
  });

  test("returns null when no workspace has a manifest", async () => {
    const dir = await ws("no-manifest");
    // No saveManifest call — no manifest file
    const result = await loadDistributedManifest([dir]);
    expect(result).toBeNull();
  });

  test("returns single-workspace manifest when only one root supplied", async () => {
    const dir = await ws("single");
    await seedWorkspace(dir, "project-a", [
      { path: "vision/north-star.md", title: "North Star", summary: "The vision", tags: ["vision"], content: "# Vision\nBuild great things." },
    ]);
    const result = await loadDistributedManifest([dir]);
    expect(result).not.toBeNull();
    expect(result!.sections).toHaveLength(1);
    expect(result!.sections[0]!.path).toBe("vision/north-star.md");
    expect(result!._mergedRoots).toHaveLength(1);
  });

  test("merges sections from two workspaces", async () => {
    const dir1 = await ws("ws1");
    const dir2 = await ws("ws2");

    await seedWorkspace(dir1, "service-a", [
      { path: "arch/overview.md", title: "Architecture", summary: "System architecture", tags: ["architecture"], content: "# Arch\nOverview." },
    ]);
    await seedWorkspace(dir2, "service-b", [
      { path: "api/endpoints.md", title: "API Endpoints", summary: "REST endpoints", tags: ["api", "rest"], content: "# API\nEndpoints." },
    ]);

    const result = await loadDistributedManifest([dir1, dir2]);
    expect(result).not.toBeNull();
    expect(result!.sections).toHaveLength(2);

    const paths = result!.sections.map((s) => s.path);
    expect(paths).toContain("arch/overview.md");
    expect(paths).toContain("api/endpoints.md");
  });

  test("merges sections from three workspaces", async () => {
    const dir1 = await ws("ws-three-1");
    const dir2 = await ws("ws-three-2");
    const dir3 = await ws("ws-three-3");

    await seedWorkspace(dir1, "svc-a", [
      { path: "vision.md", title: "Vision", summary: "Product vision", tags: ["vision"], content: "Vision content." },
    ]);
    await seedWorkspace(dir2, "svc-b", [
      { path: "roadmap.md", title: "Roadmap", summary: "Development roadmap", tags: ["roadmap"], content: "Roadmap content." },
    ]);
    await seedWorkspace(dir3, "svc-c", [
      { path: "security.md", title: "Security", summary: "Security guidelines", tags: ["security"], content: "Security content." },
    ]);

    const result = await loadDistributedManifest([dir1, dir2, dir3]);
    expect(result).not.toBeNull();
    expect(result!.sections).toHaveLength(3);
    expect(result!._mergedRoots).toHaveLength(3);
  });
});

// ---------------------------------------------------------------------------
// Deduplication
// ---------------------------------------------------------------------------

describe("loadDistributedManifest — deduplication", () => {
  test("deduplicates sections with same (project, path) — first workspace wins", async () => {
    const dir1 = await ws("dedup-1");
    const dir2 = await ws("dedup-2");

    // Both workspaces contribute a section with the same project name and path
    await seedWorkspace(dir1, "shared-project", [
      { path: "config.md", title: "Config v1", summary: "Original config", tags: ["config"], content: "Config v1 content." },
    ]);
    await seedWorkspace(dir2, "shared-project", [
      { path: "config.md", title: "Config v2", summary: "Updated config", tags: ["config"], content: "Config v2 content." },
    ]);

    const result = await loadDistributedManifest([dir1, dir2]);
    expect(result).not.toBeNull();

    // Only one section with that path should exist
    const configs = result!.sections.filter((s) => s.path === "config.md");
    expect(configs).toHaveLength(1);
    // First workspace wins
    expect(configs[0]!.title).toBe("Config v1");
  });

  test("keeps sections with same path but different project names (not deduped)", async () => {
    const dir1 = await ws("no-dedup-1");
    const dir2 = await ws("no-dedup-2");

    await seedWorkspace(dir1, "project-alpha", [
      { path: "overview.md", title: "Alpha Overview", summary: "Alpha service overview", tags: ["alpha"], content: "Alpha." },
    ]);
    await seedWorkspace(dir2, "project-beta", [
      { path: "overview.md", title: "Beta Overview", summary: "Beta service overview", tags: ["beta"], content: "Beta." },
    ]);

    const result = await loadDistributedManifest([dir1, dir2]);
    expect(result).not.toBeNull();

    // Different projects — both sections should be present
    const overviews = result!.sections.filter((s) => s.path === "overview.md");
    expect(overviews).toHaveLength(2);
  });

  test("skips workspace with no manifest without error", async () => {
    const dir1 = await ws("skip-1");
    const dir2 = await ws("skip-2"); // no manifest written

    await seedWorkspace(dir1, "valid-project", [
      { path: "readme.md", title: "Readme", summary: "Project readme", tags: ["docs"], content: "README." },
    ]);

    const result = await loadDistributedManifest([dir1, dir2]);
    expect(result).not.toBeNull();
    expect(result!.sections).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// Origin tracking
// ---------------------------------------------------------------------------

describe("loadDistributedManifest — origin tracking", () => {
  test("sections carry _origin set to the contributing workspace root", async () => {
    const dir1 = await ws("origin-1");
    const dir2 = await ws("origin-2");

    await seedWorkspace(dir1, "svc-x", [
      { path: "api.md", title: "API", summary: "API spec", tags: ["api"], content: "API spec." },
    ]);
    await seedWorkspace(dir2, "svc-y", [
      { path: "db.md", title: "Database", summary: "Database schema", tags: ["database"], content: "DB schema." },
    ]);

    const result = await loadDistributedManifest([dir1, dir2]);
    expect(result).not.toBeNull();

    const apiSection = result!.sections.find((s) => s.path === "api.md");
    const dbSection = result!.sections.find((s) => s.path === "db.md");

    expect(apiSection?._origin).toBe(resolve(dir1));
    expect(dbSection?._origin).toBe(resolve(dir2));
  });

  test("base workspace sections also have _origin set", async () => {
    const dir = await ws("origin-base");
    await seedWorkspace(dir, "my-project", [
      { path: "vision.md", title: "Vision", summary: "The vision", tags: ["vision"], content: "The vision." },
    ]);

    const result = await loadDistributedManifest([dir]);
    expect(result).not.toBeNull();
    expect(result!.sections[0]!._origin).toBe(resolve(dir));
  });

  test("_mergedRoots lists all supplied roots in order", async () => {
    const dir1 = await ws("roots-1");
    const dir2 = await ws("roots-2");
    const dir3 = await ws("roots-3");

    await seedWorkspace(dir1, "a", [{ path: "a.md", title: "A", summary: "A summary", tags: ["a"], content: "A." }]);
    await seedWorkspace(dir2, "b", [{ path: "b.md", title: "B", summary: "B summary", tags: ["b"], content: "B." }]);
    await seedWorkspace(dir3, "c", [{ path: "c.md", title: "C", summary: "C summary", tags: ["c"], content: "C." }]);

    const result = await loadDistributedManifest([dir1, dir2, dir3]);
    expect(result).not.toBeNull();
    expect(result!._mergedRoots[0]).toBe(resolve(dir1));
    expect(result!._mergedRoots[1]).toBe(resolve(dir2));
    expect(result!._mergedRoots[2]).toBe(resolve(dir3));
  });
});

// ---------------------------------------------------------------------------
// Version bumping
// ---------------------------------------------------------------------------

describe("loadDistributedManifest — version bumping", () => {
  test("version stays at base value when only one workspace", async () => {
    const dir = await ws("ver-single");
    await seedWorkspace(dir, "proj", [
      { path: "x.md", title: "X", summary: "X summary", tags: ["x"], content: "X content." },
    ]);

    const result = await loadDistributedManifest([dir]);
    expect(result).not.toBeNull();
    // No additional roots contributed, version should equal the base manifest version
    expect(result!.version).toBe(1);
  });

  test("version increments when second workspace contributes sections", async () => {
    const dir1 = await ws("ver-two-1");
    const dir2 = await ws("ver-two-2");

    await seedWorkspace(dir1, "proj-a", [
      { path: "a.md", title: "A", summary: "A", tags: ["a"], content: "A." },
    ]);
    await seedWorkspace(dir2, "proj-b", [
      { path: "b.md", title: "B", summary: "B", tags: ["b"], content: "B." },
    ]);

    const result = await loadDistributedManifest([dir1, dir2]);
    expect(result).not.toBeNull();
    // One additional root contributed → version bumped by 1
    expect(result!.version).toBeGreaterThan(1);
  });

  test("version increments once per contributing additional root", async () => {
    const dir1 = await ws("ver-three-1");
    const dir2 = await ws("ver-three-2");
    const dir3 = await ws("ver-three-3");

    await seedWorkspace(dir1, "p1", [{ path: "s1.md", title: "S1", summary: "S1", tags: ["s1"], content: "S1." }]);
    await seedWorkspace(dir2, "p2", [{ path: "s2.md", title: "S2", summary: "S2", tags: ["s2"], content: "S2." }]);
    await seedWorkspace(dir3, "p3", [{ path: "s3.md", title: "S3", summary: "S3", tags: ["s3"], content: "S3." }]);

    const result = await loadDistributedManifest([dir1, dir2, dir3]);
    expect(result).not.toBeNull();
    // Base version=1 + 2 additional contributors = 3
    expect(result!.version).toBe(3);
  });

  test("non-contributing root (no new sections after dedup) does not bump version", async () => {
    const dir1 = await ws("ver-nocontrib-1");
    const dir2 = await ws("ver-nocontrib-2");

    // Both share the same project+path — dir2 contributes nothing new
    await seedWorkspace(dir1, "shared", [{ path: "shared.md", title: "Shared", summary: "Shared", tags: ["shared"], content: "Shared." }]);
    await seedWorkspace(dir2, "shared", [{ path: "shared.md", title: "Shared dup", summary: "Shared dup", tags: ["shared"], content: "Dup." }]);

    const result = await loadDistributedManifest([dir1, dir2]);
    expect(result).not.toBeNull();
    // dir2 didn't contribute any new sections → version unchanged
    expect(result!.version).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// IDF re-computation — no large-project bias
// ---------------------------------------------------------------------------

describe("IDF re-computation — no large-project bias", () => {
  test("retrieveSectionsFromDist returns results from both workspaces for a global query", async () => {
    const dir1 = await ws("idf-large");
    const dir2 = await ws("idf-small");

    // dir1: a larger workspace with many sections sharing common terms
    await seedWorkspace(dir1, "large-project", [
      { path: "a.md", title: "A", summary: "authentication security token", tags: ["auth"], content: "# Auth\nAuthentication and security tokens." },
      { path: "b.md", title: "B", summary: "authentication security layer", tags: ["auth"], content: "# Auth B\nAnother auth module." },
      { path: "c.md", title: "C", summary: "authentication security validation", tags: ["auth"], content: "# Auth C\nValidation." },
    ]);

    // dir2: a smaller workspace with a unique rare term
    await seedWorkspace(dir2, "small-project", [
      { path: "exotic.md", title: "Exotic Feature", summary: "exotic quantum distributed cache", tags: ["exotic", "quantum"], content: "# Exotic\nQuantum distributed cache." },
    ]);

    const sections = await retrieveSectionsFromDist("exotic quantum", [dir1, dir2], 4000);
    expect(sections.length).toBeGreaterThan(0);

    // The exotic section should score highly because "exotic" and "quantum"
    // are rare across the full merged corpus — IDF properly weights them up
    const exoticSection = sections.find((s) => s.path === "exotic.md");
    expect(exoticSection).toBeDefined();

    // The exotic section should rank first (highest score) since its terms
    // are globally rare
    expect(sections[0]!.path).toBe("exotic.md");
  });

  test("common terms in large workspace don't dominate over rare terms in small workspace", async () => {
    const dir1 = await ws("idf-bias-large");
    const dir2 = await ws("idf-bias-small");

    // dir1: 5 sections all mentioning "api"
    await seedWorkspace(dir1, "big-svc", [
      { path: "api1.md", title: "API 1", summary: "api endpoint routes", tags: ["api"], content: "API 1." },
      { path: "api2.md", title: "API 2", summary: "api handler logic", tags: ["api"], content: "API 2." },
      { path: "api3.md", title: "API 3", summary: "api middleware chain", tags: ["api"], content: "API 3." },
      { path: "api4.md", title: "API 4", summary: "api response format", tags: ["api"], content: "API 4." },
      { path: "api5.md", title: "API 5", summary: "api error handling", tags: ["api"], content: "API 5." },
    ]);

    // dir2: 1 section mentioning "observatory" (unique)
    await seedWorkspace(dir2, "small-svc", [
      { path: "obs.md", title: "Observatory", summary: "observatory telemetry metrics", tags: ["observatory"], content: "Observatory." },
    ]);

    // Query for the rare term — should not be lost due to large-project IDF domination
    const sections = await retrieveSectionsFromDist("observatory telemetry", [dir1, dir2], 4000);
    const obs = sections.find((s) => s.path === "obs.md");
    expect(obs).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// resolveGenomeDirs
// ---------------------------------------------------------------------------

describe("resolveGenomeDirs", () => {
  test("finds genome in the root directory itself", async () => {
    const dir = await ws("rgd-root");
    // The workspace already has .ashlrcode/genome created by makeWorkspace
    const found = await resolveGenomeDirs(dir, 0);
    expect(found).toContain(resolve(dir));
  });

  test("finds genome in a child subdirectory", async () => {
    const dir = await ws("rgd-child");
    // Create a child subdirectory with a genome
    const child = join(dir, "packages", "service-a");
    await mkdir(join(child, ".ashlrcode", "genome"), { recursive: true });

    const found = await resolveGenomeDirs(dir, 2);
    expect(found).toContain(resolve(child));
  });

  test("does not include directories without .ashlrcode/genome", async () => {
    const dir = await ws("rgd-no-genome");
    // dir has genome from makeWorkspace; create a sibling child that does NOT
    const plain = join(dir, "plain-package");
    await mkdir(plain, { recursive: true });

    const found = await resolveGenomeDirs(dir, 2);
    expect(found).not.toContain(resolve(plain));
    expect(found).toContain(resolve(dir)); // root still found
  });

  test("returns deduplicated results (no duplicates from symlinks/traversal)", async () => {
    const dir = await ws("rgd-dedup");
    const found = await resolveGenomeDirs(dir, 2);
    const unique = new Set(found);
    expect(found.length).toBe(unique.size);
  });

  test("handles non-existent directory gracefully", async () => {
    // Should not throw — just return empty array
    const nonExistent = join(TMP_BASE, "does-not-exist-" + Date.now());
    const found = await resolveGenomeDirs(nonExistent, 2);
    expect(Array.isArray(found)).toBe(true);
    expect(found.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// retrieveSectionsFromDist — end-to-end
// ---------------------------------------------------------------------------

describe("retrieveSectionsFromDist — end-to-end", () => {
  test("returns empty array when no workspaces have manifests", async () => {
    const dir = await ws("ret-empty");
    const result = await retrieveSectionsFromDist("any query", [dir], 4000);
    expect(result).toEqual([]);
  });

  test("retrieves relevant sections from merged corpus", async () => {
    const dir1 = await ws("ret-e2e-1");
    const dir2 = await ws("ret-e2e-2");

    await seedWorkspace(dir1, "frontend", [
      { path: "components.md", title: "Components", summary: "React component architecture", tags: ["react", "components"], content: "# Components\nReact component architecture guide." },
    ]);
    await seedWorkspace(dir2, "backend", [
      { path: "database.md", title: "Database", summary: "PostgreSQL schema and migrations", tags: ["database", "postgres"], content: "# Database\nPostgreSQL schema and migrations." },
    ]);

    const sections = await retrieveSectionsFromDist("React components frontend", [dir1, dir2], 4000);
    expect(sections.length).toBeGreaterThan(0);
    // The React/components section should score higher for this query
    expect(sections[0]!.path).toBe("components.md");
  });

  test("respects maxTokens budget", async () => {
    const dir = await ws("ret-budget");
    await seedWorkspace(dir, "budget-project", [
      { path: "s1.md", title: "S1", summary: "database schema design patterns", tags: ["database"], content: "A".repeat(1000) },
      { path: "s2.md", title: "S2", summary: "database index optimization", tags: ["database"], content: "B".repeat(1000) },
      { path: "s3.md", title: "S3", summary: "database query performance", tags: ["database"], content: "C".repeat(1000) },
    ]);

    const sections = await retrieveSectionsFromDist("database", [dir], 200);
    const totalTokens = sections.reduce((sum, s) => sum + s.tokens, 0);
    expect(totalTokens).toBeLessThanOrEqual(200);
  });

  test("content is readable from the correct origin workspace", async () => {
    const dir1 = await ws("ret-content-1");
    const dir2 = await ws("ret-content-2");

    await seedWorkspace(dir1, "ws1", [
      { path: "feature.md", title: "Feature", summary: "feature flag management", tags: ["feature", "flags"], content: "Feature flags from workspace 1." },
    ]);
    await seedWorkspace(dir2, "ws2", [
      { path: "deploy.md", title: "Deploy", summary: "deployment pipeline configuration", tags: ["deploy", "pipeline"], content: "Deployment config from workspace 2." },
    ]);

    const sections = await retrieveSectionsFromDist("feature flags", [dir1, dir2], 4000);
    const featureSection = sections.find((s) => s.path === "feature.md");
    expect(featureSection).toBeDefined();
    expect(featureSection!.content).toContain("workspace 1");
  });
});
