/**
 * basic.ts — smallest runnable integration: ashlr MCP tools streamed via
 * the Claude Agent SDK.
 *
 * Scenario: ask Claude to Read a file. ashlr-efficiency's `read` tool
 * returns a snipCompact-truncated view (head + tail, elided middle)
 * instead of the full payload — the token-saving win is automatic.
 *
 * Run:
 *   ANTHROPIC_API_KEY=sk-... bun run examples/anthropic-sdk/basic.ts
 *
 * Peer deps:
 *   bun add @anthropic-ai/claude-agent-sdk
 */
import { query } from "@anthropic-ai/claude-agent-sdk";
import { ashlrMcpConfigRecord } from "../../src/anthropic/index.ts";

if (!process.env.ANTHROPIC_API_KEY) {
  console.error("ANTHROPIC_API_KEY is required. export it and retry.");
  process.exit(1);
}

// Auto-detects the ashlr-plugin root (Claude Code cache, env, or ~/Desktop).
const mcpServers = ashlrMcpConfigRecord({ plugins: ["efficiency"] });

const iter = query({
  prompt:
    "Use the ashlr__read tool to read package.json in the current directory " +
    "and summarize the package's purpose in one sentence.",
  options: {
    model: "claude-sonnet-4-6",
    mcpServers,
    allowedTools: ["mcp__ashlr-efficiency__ashlr__read"],
    permissionMode: "bypassPermissions",
    cwd: process.cwd(),
  },
});

for await (const msg of iter) {
  if (msg.type === "assistant") {
    for (const block of msg.message.content) {
      if (block.type === "text") process.stdout.write(block.text);
    }
  }
  if (msg.type === "result") {
    process.stdout.write("\n");
    console.error(
      `[done] turns=${msg.num_turns} cost=$${msg.total_cost_usd.toFixed(4)} ` +
        `in=${msg.usage.input_tokens} out=${msg.usage.output_tokens}`,
    );
  }
}
