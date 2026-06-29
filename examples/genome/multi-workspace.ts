/**
 * multi-workspace.ts — distributed genome retrieval in a monorepo scenario.
 *
 * Demonstrates how to use `resolveGenomeDirs` and `loadDistributedManifest`
 * to find and merge genome sections from multiple workspaces (services,
 * packages, apps) within a monorepo, then run a single query across the
 * unified corpus using `retrieveSectionsFromDist`.
 *
 * Typical monorepo layout this supports:
 *
 *   my-monorepo/
 *   ├── .ashlrcode/genome/          ← root workspace genome
 *   ├── apps/
 *   │   └── web/
 *   │       └── .ashlrcode/genome/  ← web app genome
 *   └── services/
 *       ├── api/
 *       │   └── .ashlrcode/genome/  ← API service genome
 *       └── worker/
 *           └── .ashlrcode/genome/  ← worker service genome
 *
 * Run (from any directory inside or above a monorepo):
 *   bun run examples/genome/multi-workspace.ts
 *   # or point at a specific monorepo root:
 *   MONOREPO_ROOT=/path/to/repo bun run examples/genome/multi-workspace.ts
 */

import {
  loadDistributedManifest,
  resolveGenomeDirs,
} from "../../src/genome/distributed-manifest.ts";
import { retrieveSectionsFromDist } from "../../src/genome/retriever.ts";
import { formatGenomeForPrompt } from "../../src/genome/retriever.ts";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/** Root of the monorepo to scan. Override with MONOREPO_ROOT env var. */
const MONOREPO_ROOT = process.env["MONOREPO_ROOT"] ?? process.cwd();

/**
 * How many directory levels beneath MONOREPO_ROOT to search for genomes.
 * Increase for deeper nesting (e.g. apps/web/packages/ui/).
 */
const SCAN_DEPTH = 3;

/** Token budget for retrieved sections. */
const MAX_TOKENS = 3000;

/** The query to run across the distributed corpus. */
const QUERY = process.argv[2] ?? "architecture overview and current milestone";

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

console.log("=== Ashlr Distributed Genome — Multi-Workspace Retrieval ===\n");
console.log(`Monorepo root : ${MONOREPO_ROOT}`);
console.log(`Scan depth    : ${SCAN_DEPTH}`);
console.log(`Token budget  : ${MAX_TOKENS}`);
console.log(`Query         : "${QUERY}"\n`);

// Step 1: Discover all genome directories in the monorepo tree
console.log("Step 1 — Discovering genome directories...");
const genomeDirs = await resolveGenomeDirs(MONOREPO_ROOT, SCAN_DEPTH);

if (genomeDirs.length === 0) {
  console.log(
    `No .ashlrcode/genome/ directories found under ${MONOREPO_ROOT}.\n` +
      "Run `ashlr genome init` in one or more workspaces first, then retry.",
  );
  process.exit(0);
}

console.log(`Found ${genomeDirs.length} workspace(s) with genomes:`);
for (const dir of genomeDirs) {
  console.log(`  • ${dir}`);
}
console.log();

// Step 2: Load and merge all genome manifests
console.log("Step 2 — Loading and merging manifests...");
const distributed = await loadDistributedManifest(genomeDirs);

if (!distributed) {
  console.log("No readable manifests found. Ensure genomes are initialized properly.");
  process.exit(0);
}

const totalSections = distributed.sections.length;
const mergedRoots = distributed._mergedRoots.length;
console.log(`Merged ${totalSections} section(s) from ${mergedRoots} workspace(s).`);
console.log(`Manifest version after merge: ${distributed.version}`);

// Show a breakdown of sections by origin workspace
const originCounts = new Map<string, number>();
for (const section of distributed.sections) {
  const origin = section._origin ?? "(unknown)";
  originCounts.set(origin, (originCounts.get(origin) ?? 0) + 1);
}
console.log("\nSection counts by workspace:");
for (const [origin, count] of originCounts) {
  const label = origin.startsWith(MONOREPO_ROOT)
    ? origin.slice(MONOREPO_ROOT.length) || "/"
    : origin;
  console.log(`  ${count.toString().padStart(3)} section(s) — ${label}`);
}
console.log();

// Step 3: Retrieve relevant sections for the query
console.log(`Step 3 — Retrieving sections for query: "${QUERY}"...`);
const sections = await retrieveSectionsFromDist(QUERY, genomeDirs, MAX_TOKENS);

if (sections.length === 0) {
  console.log("No sections matched the query within the token budget.");
  process.exit(0);
}

console.log(`Retrieved ${sections.length} section(s), ${sections.reduce((t, s) => t + s.tokens, 0)} tokens total.\n`);

// Show the ranked results
console.log("Ranked results:");
for (let i = 0; i < sections.length; i++) {
  const s = sections[i]!;
  console.log(`  ${i + 1}. [score=${s.score.toFixed(3)}] ${s.title} (${s.path}) — ${s.tokens} tokens`);
}
console.log();

// Step 4: Format into a system prompt snippet
const prompt = formatGenomeForPrompt(sections);
const preview = prompt.slice(0, 500);
console.log("--- System prompt preview (first 500 chars) ---");
console.log(preview);
if (prompt.length > 500) {
  console.log(`... [${prompt.length - 500} more characters]`);
}
