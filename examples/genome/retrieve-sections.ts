/**
 * retrieve-sections.ts — demonstrates @ashlr/core-efficiency/genome
 *
 * Subpath: @ashlr/core-efficiency/genome
 *
 * Shows how to check for a genome, retrieve relevant sections for a query
 * (TF-IDF or Ollama depending on what is available), and inject the result
 * into a system prompt string.
 *
 * Requires a project with a populated .ashlrcode/genome/ directory.
 * If none is found, the script reports it gracefully and exits.
 *
 * Run:
 *   bun run examples/genome/retrieve-sections.ts
 *   # or from a project that has a genome:
 *   cd /path/to/project && bun run /path/to/examples/genome/retrieve-sections.ts
 *
 * Windows note: genome paths are normalized with path.sep internally —
 * runs identically on macOS / Linux / Windows.
 */
import {
  genomeExists,
  retrieveSectionsV2,
  injectGenomeContext,
} from "../../src/genome/index.ts";

const cwd = process.cwd();

if (!(await genomeExists(cwd))) {
  console.log(
    `No .ashlrcode/genome/ found under ${cwd}.\n` +
      "Run `ashlrcode genome init` in a project to create one, then retry.",
  );
  process.exit(0);
}

const query = "architecture overview and current milestone";
console.log(`Retrieving sections for: "${query}"`);

const sections = await retrieveSectionsV2(query, cwd, { maxTokens: 2000 });
console.log(`Retrieved ${sections.length} section(s).`);

const baseSystem = "You are a senior engineer. Answer using the project genome.";
const enriched = injectGenomeContext(baseSystem, sections);

console.log(`\nSystem prompt length: ${baseSystem.length} → ${enriched.length} chars`);
console.log("\n--- System prompt preview (first 400 chars) ---");
console.log(enriched.slice(0, 400));
