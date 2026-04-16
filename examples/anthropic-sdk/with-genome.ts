/**
 * with-genome.ts — prepend project genome context to the system prompt.
 *
 * Scenario: the current project has a `.ashlrcode/genome/` populated with
 * RAG-able spec sections. `withGenome` reads the manifest, retrieves the
 * top sections for the user's query (semantic via Ollama when available,
 * TF-IDF fallback), and returns a new system prompt with those sections
 * stitched above the caller's instructions.
 *
 * If no genome exists, the original prompt is returned unchanged — safe
 * to use unconditionally.
 *
 * Run:
 *   ANTHROPIC_API_KEY=sk-... bun run examples/anthropic-sdk/with-genome.ts
 */
import Anthropic from "@anthropic-ai/sdk";
import { withGenome } from "../../src/anthropic/index.ts";

if (!process.env.ANTHROPIC_API_KEY) {
  console.error("ANTHROPIC_API_KEY is required. export it and retry.");
  process.exit(1);
}

const client = new Anthropic();

const baseSystem =
  "You are a senior engineer. Use the provided project genome to answer " +
  "concretely; if the genome is empty, say so.";

// Walks up from cwd looking for .ashlrcode/genome/; returns baseSystem
// unchanged if nothing is found.
const system = await withGenome(baseSystem, process.cwd(), {
  query: "architecture overview and current milestone",
  maxTokens: 1500,
});

console.error(`[genome] system prompt length: ${system.length} chars`);

const stream = client.messages.stream({
  model: "claude-sonnet-4-5",
  max_tokens: 512,
  system,
  messages: [
    {
      role: "user",
      content:
        "Based on the genome above, summarize this project's architecture " +
        "in 3 bullets. If no genome is present, say so explicitly.",
    },
  ],
});

for await (const event of stream) {
  if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
    process.stdout.write(event.delta.text);
  }
}
process.stdout.write("\n");

const final = await stream.finalMessage();
console.error(
  `[done] in=${final.usage.input_tokens} out=${final.usage.output_tokens}`,
);
