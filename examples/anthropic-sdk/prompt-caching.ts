/**
 * prompt-caching.ts — measurable cache hits with `cacheBreakpoints`.
 *
 * Scenario: two back-to-back calls that share a long static prefix
 * (system prompt + a big "project context" user block). The second call
 * should report cache_read_input_tokens > 0 — proof the prefix hit cache.
 *
 * `cacheBreakpoints` adds ephemeral cache_control markers at the
 * static→dynamic boundary for you.
 *
 * Run:
 *   ANTHROPIC_API_KEY=sk-... bun run examples/anthropic-sdk/prompt-caching.ts
 */
import Anthropic from "@anthropic-ai/sdk";
import { cacheBreakpoints } from "../../src/anthropic/index.ts";

if (!process.env.ANTHROPIC_API_KEY) {
  console.error("ANTHROPIC_API_KEY is required. export it and retry.");
  process.exit(1);
}

const client = new Anthropic();

// A "big" static prefix — repeat a paragraph so we exceed the minimum
// cacheable prefix (~1024 tokens for sonnet).
const prefix = ("ASHLR project context. Stack: TypeScript + Bun. " +
  "Ships @ashlr/core-efficiency and the ashlr-plugin for Claude Code. " +
  "Token-saving primitives: genome RAG, 3-tier compression, provider " +
  "budgeting. ").repeat(120);

const system =
  "You are an assistant for the ASHLR project. Answer using the context " +
  "below precisely; do not invent facts.";

function buildRequest(userQuestion: string) {
  return cacheBreakpoints({
    system,
    messages: [
      { role: "user" as const, content: prefix, cache: true },
      { role: "user" as const, content: userQuestion },
    ],
  });
}

async function ask(q: string, label: string) {
  const req = buildRequest(q);
  const res = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 128,
    ...req,
  });
  const u = res.usage;
  console.log(
    `[${label}] in=${u.input_tokens} out=${u.output_tokens} ` +
      `cache_create=${u.cache_creation_input_tokens ?? 0} ` +
      `cache_read=${u.cache_read_input_tokens ?? 0}`,
  );
  const text = res.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("");
  console.log(`  ↳ ${text.slice(0, 160).replace(/\s+/g, " ")}...\n`);
}

// First call writes the cache, second call should read it.
await ask("What stack does this project use?", "call 1 (cold)");
await ask("Name one token-saving primitive.", "call 2 (warm)");
