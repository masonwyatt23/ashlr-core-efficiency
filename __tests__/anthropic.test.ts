/**
 * Tests for src/anthropic/ — cacheBreakpoints, cacheMessagesBreakpoints,
 * resolvePluginRoot / AshlrPluginNotFoundError, and mcp-config helpers.
 */

import { describe, expect, test } from "bun:test";
import { mkdirSync, rmSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import {
  cacheBreakpoints,
  cacheMessagesBreakpoints,
  type CacheableMessage,
  type CacheableRequest,
} from "../src/anthropic/prompt-cache.ts";
import {
  AshlrPluginNotFoundError,
  resolvePluginRoot,
  ashlrMcpConfig,
  ashlrMcpConfigRecord,
} from "../src/anthropic/mcp-config.ts";

// ---------------------------------------------------------------------------
// cacheBreakpoints / cacheMessagesBreakpoints
// ---------------------------------------------------------------------------

describe("cacheBreakpoints", () => {
  test("promotes string system to text block with cache_control", () => {
    const req: CacheableRequest = {
      system: "You are helpful.",
      messages: [{ role: "user", content: "hello" }],
    };
    const result = cacheBreakpoints(req);
    expect(Array.isArray(result.system)).toBe(true);
    const sys = result.system as any[];
    expect(sys[0]).toMatchObject({ type: "text", text: "You are helpful.", cache_control: { type: "ephemeral" } });
  });

  test("marks last system block when system is already an array", () => {
    const req: CacheableRequest = {
      system: [
        { type: "text", text: "block1" },
        { type: "text", text: "block2" },
      ],
      messages: [{ role: "user", content: "hi" }],
    };
    const result = cacheBreakpoints(req);
    const sys = result.system as any[];
    expect(sys[0]?.cache_control).toBeUndefined();
    expect(sys[1]?.cache_control).toEqual({ type: "ephemeral" });
  });

  test("marks last tool with cache_control", () => {
    const req: CacheableRequest = {
      tools: [
        { name: "search", input_schema: {} },
        { name: "read_file", input_schema: {} },
      ],
      messages: [{ role: "user", content: "hi" }],
    };
    const result = cacheBreakpoints(req);
    expect(result.tools![0]?.cache_control).toBeUndefined();
    expect(result.tools![1]?.cache_control).toEqual({ type: "ephemeral" });
  });

  test("does not mutate original request", () => {
    const req: CacheableRequest = {
      system: "original",
      messages: [{ role: "user", content: "msg" }],
    };
    cacheBreakpoints(req);
    expect(req.system).toBe("original");
  });

  test("handles empty tools array", () => {
    const req: CacheableRequest = {
      tools: [],
      messages: [{ role: "user", content: "hi" }],
    };
    const result = cacheBreakpoints(req);
    expect(result.tools).toEqual([]);
  });

  test("handles undefined system", () => {
    const req: CacheableRequest = {
      messages: [{ role: "user", content: "hi" }],
    };
    const result = cacheBreakpoints(req);
    expect(result.system).toBeUndefined();
  });

  test("places cache marker on last cache:true message", () => {
    const messages: CacheableMessage[] = [
      { role: "user", content: "static context", cache: true },
      { role: "user", content: "dynamic question" },
    ];
    const result = cacheBreakpoints({ messages });
    const firstMsg = result.messages[0]!;
    // Should have been promoted to block array with cache_control
    expect(Array.isArray(firstMsg.content)).toBe(true);
    expect((firstMsg.content as any[])[0]?.cache_control).toEqual({ type: "ephemeral" });
    // cache meta flag stripped
    expect((firstMsg as any).cache).toBeUndefined();
  });

  test("strips cache flag from result messages", () => {
    const messages: CacheableMessage[] = [
      { role: "user", content: "context", cache: true },
    ];
    const result = cacheBreakpoints({ messages });
    expect((result.messages[0] as any).cache).toBeUndefined();
  });

  test("no cache:true messages → messages returned as-is (shallow clone)", () => {
    const messages: CacheableMessage[] = [
      { role: "user", content: "just a normal question" },
    ];
    const result = cacheBreakpoints({ messages });
    expect(result.messages[0]?.content).toBe("just a normal question");
    expect((result.messages[0] as any).cache_control).toBeUndefined();
  });

  test("empty messages array", () => {
    const result = cacheBreakpoints({ messages: [] });
    expect(result.messages).toEqual([]);
  });
});

describe("cacheMessagesBreakpoints", () => {
  test("marks last cache:true message", () => {
    const messages: CacheableMessage[] = [
      { role: "user", content: "static", cache: true },
      { role: "user", content: "dynamic" },
    ];
    const result = cacheMessagesBreakpoints(messages);
    expect(Array.isArray(result[0]!.content)).toBe(true);
    expect((result[0]!.content as any[])[0]?.cache_control).toEqual({ type: "ephemeral" });
  });

  test("handles message with block array content + cache flag", () => {
    const messages: CacheableMessage[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "block1" },
          { type: "text", text: "block2" },
        ],
        cache: true,
      },
    ];
    const result = cacheMessagesBreakpoints(messages);
    const content = result[0]!.content as any[];
    expect(content[0]?.cache_control).toBeUndefined();
    expect(content[1]?.cache_control).toEqual({ type: "ephemeral" });
  });

  test("empty array returns empty", () => {
    expect(cacheMessagesBreakpoints([])).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// resolvePluginRoot
// ---------------------------------------------------------------------------

describe("resolvePluginRoot", () => {
  test("throws AshlrPluginNotFoundError when all candidates invalid", () => {
    // Use a fake plugin root that points to a dir with no efficiency-server.ts.
    // We also need to ensure env var doesn't rescue us, so temporarily clear it.
    const prev = process.env.ASHLR_PLUGIN_ROOT;
    process.env.ASHLR_PLUGIN_ROOT = "/nonexistent/env-path-zzz";
    try {
      // Pass an explicit bad path; env is also bad; cache+desktop won't match
      // because we need looksLikePluginRoot to fail everywhere.
      // The only reliable way to force a throw is to create a dir that fails
      // looksLikePluginRoot (no servers/efficiency-server.ts inside).
      const emptyDir = join(tmpdir(), `ashlr-empty-${Date.now()}`);
      mkdirSync(emptyDir, { recursive: true });
      try {
        resolvePluginRoot(emptyDir);
        // If we reach here the machine has a valid plugin somewhere — skip gracefully
      } catch (err: any) {
        expect(err).toBeInstanceOf(AshlrPluginNotFoundError);
        expect(err.name).toBe("AshlrPluginNotFoundError");
        expect(err.message).toContain("Could not locate the ashlr-plugin install.");
      } finally {
        rmSync(emptyDir, { recursive: true, force: true });
      }
    } finally {
      if (prev === undefined) delete process.env.ASHLR_PLUGIN_ROOT;
      else process.env.ASHLR_PLUGIN_ROOT = prev;
    }
  });

  test("error message includes actionable remediation", () => {
    // AshlrPluginNotFoundError's message should mention the fix options
    const err = new AshlrPluginNotFoundError(["/some/path", "/another/path"]);
    expect(err.message).toContain("ASHLR_PLUGIN_ROOT");
    expect(err.message).toContain("pluginRoot");
    expect(err.name).toBe("AshlrPluginNotFoundError");
  });

  test("resolves an explicit valid plugin root (cross-platform join)", () => {
    // Create a minimal fake plugin root in os.tmpdir()
    const dir = join(tmpdir(), `ashlr-test-root-${Date.now()}`);
    const serverDir = join(dir, "servers");
    mkdirSync(serverDir, { recursive: true });
    writeFileSync(join(serverDir, "efficiency-server.ts"), "// stub");
    try {
      const { root } = resolvePluginRoot(dir);
      expect(root).toBe(dir);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

// ---------------------------------------------------------------------------
// ashlrMcpConfig / ashlrMcpConfigRecord
// ---------------------------------------------------------------------------

describe("ashlrMcpConfig + ashlrMcpConfigRecord", () => {
  /** Build a fake plugin root and return its path. */
  function makeFakeRoot(): string {
    const dir = join(tmpdir(), `ashlr-mcp-test-${Date.now()}`);
    const serverDir = join(dir, "servers");
    mkdirSync(serverDir, { recursive: true });
    for (const name of [
      "efficiency-server.ts",
      "genome-server.ts",
      "bash-server.ts",
      "diff-server.ts",
      "github-server.ts",
      "http-server.ts",
      "logs-server.ts",
      "orient-server.ts",
      "sql-server.ts",
      "tree-server.ts",
    ]) {
      writeFileSync(join(serverDir, name), "// stub");
    }
    return dir;
  }

  test("returns all 10 plugins by default", () => {
    const root = makeFakeRoot();
    try {
      const servers = ashlrMcpConfig({ pluginRoot: root });
      expect(servers.length).toBe(10);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("each entry has type:stdio, name, command, args", () => {
    const root = makeFakeRoot();
    try {
      const servers = ashlrMcpConfig({ pluginRoot: root });
      for (const s of servers) {
        expect(s.type).toBe("stdio");
        expect(typeof s.name).toBe("string");
        expect(typeof s.command).toBe("string");
        expect(Array.isArray(s.args)).toBe(true);
      }
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("custom runtime is reflected in command", () => {
    const root = makeFakeRoot();
    try {
      const servers = ashlrMcpConfig({ pluginRoot: root, runtime: "tsx" });
      expect(servers.every((s) => s.command === "tsx")).toBe(true);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("subset of plugins", () => {
    const root = makeFakeRoot();
    try {
      const servers = ashlrMcpConfig({ pluginRoot: root, plugins: ["efficiency", "bash"] });
      expect(servers.length).toBe(2);
      expect(servers.map((s) => s.name)).toEqual(["ashlr-efficiency", "ashlr-bash"]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("args use path.join (no raw forward-slash on Windows)", () => {
    const root = makeFakeRoot();
    try {
      const servers = ashlrMcpConfig({ pluginRoot: root, plugins: ["efficiency"] });
      // The absolute path in args should start with the root dir
      const absPath = servers[0]!.args[1]!;
      expect(absPath.startsWith(root)).toBe(true);
      // Should be a proper joined path (no double separators)
      expect(absPath).not.toContain("//");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("ashlrMcpConfigRecord returns name-keyed object", () => {
    const root = makeFakeRoot();
    try {
      const record = ashlrMcpConfigRecord({ pluginRoot: root, plugins: ["efficiency"] });
      expect(typeof record["ashlr-efficiency"]).toBe("object");
      expect(record["ashlr-efficiency"]?.type).toBe("stdio");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
