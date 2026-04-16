/**
 * ashlrMcpConfig — auto-detect the ashlr-plugin install and emit MCP stdio
 * server configs that can be dropped into the Claude Agent SDK's
 * `mcpServers` option (structurally compatible with
 * `McpStdioServerConfig` from `@anthropic-ai/claude-agent-sdk`).
 *
 * The shape is intentionally the `{ type: 'stdio', name, command, args }`
 * array form so callers can either spread into an object-keyed record
 * (`Object.fromEntries(cfg.map(s => [s.name, s]))`) or pass through to any
 * consumer that accepts a stdio MCP definition list.
 */
import { existsSync, readdirSync } from "fs";
import { homedir } from "os";
import { join } from "path";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Plugins bundled with ashlr-plugin — each maps to one MCP stdio server. */
export type AshlrPlugin =
  | "efficiency"
  | "genome"
  | "bash"
  | "diff"
  | "github"
  | "http"
  | "logs"
  | "orient"
  | "sql"
  | "tree";

/**
 * Stdio MCP server definition. Matches the shape expected by the Claude
 * Agent SDK's `McpStdioServerConfig` with an added `name` field so the
 * array can be consumed directly or converted to a name-keyed record.
 */
export interface AshlrMcpStdioServer {
  type: "stdio";
  name: string;
  command: string;
  args: string[];
}

export interface AshlrMcpConfigOptions {
  /**
   * Which plugins to include. Default: all 10.
   */
  plugins?: AshlrPlugin[];
  /**
   * Absolute path to the ashlr-plugin checkout/install. If omitted,
   * the resolver walks the auto-detect chain (env → plugin cache → ~/Desktop).
   */
  pluginRoot?: string;
  /**
   * Override the runtime used to exec MCP servers. Default: "bun".
   * The plugin's servers are `.ts` files, so a TypeScript-capable runtime
   * (bun, tsx, ts-node) is required.
   */
  runtime?: string;
}

/**
 * Thrown when `ashlrMcpConfig` cannot locate the ashlr-plugin root.
 * The message includes actionable remediation steps.
 */
export class AshlrPluginNotFoundError extends Error {
  constructor(checked: string[]) {
    const list = checked.map((p) => `  - ${p}`).join("\n");
    super(
      [
        "Could not locate the ashlr-plugin install.",
        "Checked the following paths:",
        list,
        "",
        "Fix one of the following:",
        "  1. Install the plugin in Claude Code (it populates ~/.claude/plugins/cache/ashlr-marketplace/ashlr/).",
        "  2. Clone https://github.com/ashlrai/ashlr-plugin to ~/Desktop/ashlr-plugin.",
        "  3. Set ASHLR_PLUGIN_ROOT to an absolute path.",
        "  4. Pass `pluginRoot` to ashlrMcpConfig({ pluginRoot: '/abs/path' }).",
      ].join("\n"),
    );
    this.name = "AshlrPluginNotFoundError";
  }
}

// ---------------------------------------------------------------------------
// Plugin-root resolution
// ---------------------------------------------------------------------------

/** Map of plugin → relative path to its MCP server script under the plugin root. */
const PLUGIN_SERVER_FILES: Record<AshlrPlugin, string> = {
  efficiency: "servers/efficiency-server.ts",
  genome: "servers/genome-server.ts",
  bash: "servers/bash-server.ts",
  diff: "servers/diff-server.ts",
  github: "servers/github-server.ts",
  http: "servers/http-server.ts",
  logs: "servers/logs-server.ts",
  orient: "servers/orient-server.ts",
  sql: "servers/sql-server.ts",
  tree: "servers/tree-server.ts",
};

const ALL_PLUGINS: AshlrPlugin[] = Object.keys(PLUGIN_SERVER_FILES) as AshlrPlugin[];

/**
 * Parse a semver-ish folder name (e.g. "0.7.0", "1.2.3-beta.1") into a
 * comparable tuple. Returns [0,0,0] on unparseable input so it sorts last.
 */
function parseSemverLoose(name: string): [number, number, number] {
  const m = /^(\d+)\.(\d+)\.(\d+)/.exec(name);
  if (!m) return [0, 0, 0];
  return [Number(m[1]), Number(m[2]), Number(m[3])];
}

/** Compare two semver tuples: returns >0 if a<b, <0 if a>b (descending sort). */
function semverDesc(a: string, b: string): number {
  const [a1, a2, a3] = parseSemverLoose(a);
  const [b1, b2, b3] = parseSemverLoose(b);
  if (b1 !== a1) return b1 - a1;
  if (b2 !== a2) return b2 - a2;
  return b3 - a3;
}

/** Return true if `dir` looks like an ashlr-plugin checkout. */
function looksLikePluginRoot(dir: string): boolean {
  return existsSync(join(dir, "servers", "efficiency-server.ts"));
}

/**
 * Walk the auto-detect chain and return the first plugin root that exists,
 * together with the list of paths checked (for error messages).
 */
export function resolvePluginRoot(explicit?: string): { root: string; checked: string[] } {
  const checked: string[] = [];

  // 1. Explicit option wins
  if (explicit) {
    checked.push(`${explicit} (pluginRoot option)`);
    if (looksLikePluginRoot(explicit)) return { root: explicit, checked };
  }

  // 2. Env var
  const env = process.env.ASHLR_PLUGIN_ROOT;
  if (env) {
    checked.push(`${env} (ASHLR_PLUGIN_ROOT env)`);
    if (looksLikePluginRoot(env)) return { root: env, checked };
  }

  // 3. Claude Code plugin cache — pick latest semver
  const cacheDir = join(homedir(), ".claude", "plugins", "cache", "ashlr-marketplace", "ashlr");
  checked.push(`${cacheDir}/<latest> (Claude Code plugin cache)`);
  if (existsSync(cacheDir)) {
    try {
      const versions = readdirSync(cacheDir, { withFileTypes: true })
        .filter((e) => e.isDirectory())
        .map((e) => e.name)
        .sort(semverDesc);
      for (const v of versions) {
        const candidate = join(cacheDir, v);
        if (looksLikePluginRoot(candidate)) return { root: candidate, checked };
      }
    } catch {
      // ignore — fall through
    }
  }

  // 4. ~/Desktop/ashlr-plugin (developer default)
  const desktop = join(homedir(), "Desktop", "ashlr-plugin");
  checked.push(`${desktop} (developer default)`);
  if (looksLikePluginRoot(desktop)) return { root: desktop, checked };

  throw new AshlrPluginNotFoundError(checked);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Build the MCP stdio server list for the requested ashlr plugins.
 *
 * @example
 * ```ts
 * import Anthropic from "@anthropic-ai/claude-agent-sdk";
 * import { ashlrMcpConfig } from "@ashlr/core-efficiency/anthropic";
 *
 * const servers = ashlrMcpConfig();
 * const mcpServers = Object.fromEntries(servers.map(s => [s.name, s]));
 * ```
 */
export function ashlrMcpConfig(options: AshlrMcpConfigOptions = {}): AshlrMcpStdioServer[] {
  const plugins = options.plugins ?? ALL_PLUGINS;
  const runtime = options.runtime ?? "bun";
  const { root } = resolvePluginRoot(options.pluginRoot);

  return plugins.map((plugin) => {
    const rel = PLUGIN_SERVER_FILES[plugin];
    const abs = join(root, rel);
    return {
      type: "stdio" as const,
      name: `ashlr-${plugin}`,
      command: runtime,
      args: ["run", abs],
    };
  });
}

/**
 * Convenience: same result as `ashlrMcpConfig` but keyed by server name.
 * Handy for drop-in use with the Claude Agent SDK which expects
 * `mcpServers: Record<string, McpServerConfig>`.
 */
export function ashlrMcpConfigRecord(
  options: AshlrMcpConfigOptions = {},
): Record<string, Omit<AshlrMcpStdioServer, "name">> {
  const list = ashlrMcpConfig(options);
  return Object.fromEntries(list.map(({ name, ...rest }) => [name, rest]));
}
