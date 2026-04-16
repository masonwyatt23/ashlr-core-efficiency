/**
 * session-log — behavior tests.
 *
 * Each test points ASHLR_SESSION_LOG_PATH at a unique temp file so tests
 * are isolated and can run in parallel. We never touch the real
 * ~/.ashlr/session-log.jsonl from the test suite.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  append,
  read,
  rotate,
  SESSION_LOG_MAX_BYTES,
  tail,
} from "../index.ts";

let tmpDir = "";
let logPath = "";
const prevEnvPath = process.env.ASHLR_SESSION_LOG_PATH;
const prevEnvFlag = process.env.ASHLR_SESSION_LOG;

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), "ashlr-session-log-"));
  logPath = join(tmpDir, "session-log.jsonl");
  process.env.ASHLR_SESSION_LOG_PATH = logPath;
  delete process.env.ASHLR_SESSION_LOG;
});

afterEach(() => {
  if (prevEnvPath !== undefined) process.env.ASHLR_SESSION_LOG_PATH = prevEnvPath;
  else delete process.env.ASHLR_SESSION_LOG_PATH;
  if (prevEnvFlag !== undefined) process.env.ASHLR_SESSION_LOG = prevEnvFlag;
  else delete process.env.ASHLR_SESSION_LOG;
  try {
    rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* ignore */
  }
});

describe("append + read round trip", () => {
  test("appends one entry and reads it back", async () => {
    await append({
      agent: "claude-code",
      event: "tool_call",
      tool: "Read",
      summary: "read foo.ts",
    });

    const entries = read();
    expect(entries.length).toBe(1);
    const [e] = entries;
    expect(e?.agent).toBe("claude-code");
    expect(e?.event).toBe("tool_call");
    expect(e?.tool).toBe("Read");
    expect(e?.summary).toBe("read foo.ts");
    expect(typeof e?.ts).toBe("string");
    // ISO-8601 UTC with millis
    expect(e?.ts).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
  });

  test("preserves order (newest first on read)", async () => {
    await append({ agent: "a", event: "message", summary: "first" });
    await append({ agent: "b", event: "message", summary: "second" });
    await append({ agent: "c", event: "message", summary: "third" });

    const entries = read();
    expect(entries.map((e) => e.summary)).toEqual(["third", "second", "first"]);
  });

  test("respects caller-supplied ts", async () => {
    const ts = "2026-04-16T05:58:00.123Z";
    await append({ ts, agent: "claude-code", event: "session_start" });
    const entries = read();
    expect(entries[0]?.ts).toBe(ts);
  });

  test("filters by agent", async () => {
    await append({ agent: "claude-code", event: "message", summary: "c1" });
    await append({ agent: "goose", event: "message", summary: "g1" });
    await append({ agent: "claude-code", event: "message", summary: "c2" });

    const cc = read({ agent: "claude-code" });
    expect(cc.map((e) => e.summary)).toEqual(["c2", "c1"]);
  });

  test("filters by since", async () => {
    const old = "2020-01-01T00:00:00.000Z";
    const recent = new Date().toISOString();
    await append({ ts: old, agent: "a", event: "message", summary: "old" });
    await append({ ts: recent, agent: "a", event: "message", summary: "new" });

    const entries = read({ since: new Date("2024-01-01T00:00:00.000Z") });
    expect(entries.map((e) => e.summary)).toEqual(["new"]);
  });

  test("limit caps result count", async () => {
    for (let i = 0; i < 10; i++) {
      await append({ agent: "a", event: "message", summary: `m${i}` });
    }
    const entries = read({ limit: 3 });
    expect(entries.length).toBe(3);
    // Most-recent-first.
    expect(entries.map((e) => e.summary)).toEqual(["m9", "m8", "m7"]);
  });

  test("missing file reads as empty", () => {
    expect(read()).toEqual([]);
  });

  test("skips corrupted lines defensively", async () => {
    await append({ agent: "a", event: "message", summary: "valid1" });
    // Inject garbage directly.
    const raw = readFileSync(logPath, "utf8");
    writeFileSync(logPath, raw + "this-is-not-json\n{\"half\":\n");
    await append({ agent: "a", event: "message", summary: "valid2" });

    const entries = read();
    expect(entries.map((e) => e.summary)).toEqual(["valid2", "valid1"]);
  });

  test("ASHLR_SESSION_LOG=0 disables writes but reads still work", async () => {
    await append({ agent: "a", event: "message", summary: "before" });

    process.env.ASHLR_SESSION_LOG = "0";
    await append({ agent: "a", event: "message", summary: "disabled" });

    const entries = read();
    expect(entries.map((e) => e.summary)).toEqual(["before"]);
  });
});

describe("rotate", () => {
  test("renames current → .1 and starts fresh", async () => {
    await append({ agent: "a", event: "message", summary: "pre" });
    expect(existsSync(logPath)).toBe(true);

    rotate();

    expect(existsSync(logPath)).toBe(true);
    expect(existsSync(logPath + ".1")).toBe(true);
    expect(readFileSync(logPath, "utf8")).toBe("");

    await append({ agent: "a", event: "message", summary: "post" });
    const entries = read();
    expect(entries.map((e) => e.summary)).toEqual(["post"]);
  });

  test("rotate on missing file is a no-op", () => {
    expect(() => rotate()).not.toThrow();
    expect(existsSync(logPath)).toBe(false);
  });

  test("auto-rotates when file exceeds SESSION_LOG_MAX_BYTES", async () => {
    // Seed the file just over the cap without doing 10M tiny appends.
    const filler = "x".repeat(SESSION_LOG_MAX_BYTES + 1024);
    writeFileSync(logPath, filler);

    await append({ agent: "a", event: "message", summary: "after-rotate" });

    expect(existsSync(logPath + ".1")).toBe(true);
    // Post-rotation file should contain ONLY the new entry.
    const entries = read();
    expect(entries.length).toBe(1);
    expect(entries[0]?.summary).toBe("after-rotate");
  });
});

describe("concurrent writes", () => {
  test("many parallel appends all land without byte corruption", async () => {
    const N = 200;
    const ops: Promise<void>[] = [];
    for (let i = 0; i < N; i++) {
      ops.push(
        append({
          agent: i % 2 === 0 ? "claude-code" : "goose",
          event: "tool_call",
          tool: "Read",
          summary: `op-${i}`,
          meta: { i },
        }),
      );
    }
    await Promise.all(ops);

    const entries = read();
    expect(entries.length).toBe(N);

    // Every entry must round-trip — no partial/interleaved lines.
    const seen = new Set<number>();
    for (const e of entries) {
      const meta = e.meta as { i?: number } | undefined;
      expect(typeof meta?.i).toBe("number");
      seen.add(meta!.i!);
    }
    expect(seen.size).toBe(N);
  });
});

describe("tail", () => {
  test("yields new entries as they are appended, stops on abort", async () => {
    // Prime the file so the watcher has something to bind to.
    await append({ agent: "a", event: "message", summary: "seed" });

    const ac = new AbortController();
    const collected: string[] = [];

    const consumer = (async () => {
      for await (const entry of tail({ signal: ac.signal })) {
        if (typeof entry.summary === "string") collected.push(entry.summary);
        if (collected.length >= 2) ac.abort();
      }
    })();

    // Give the watcher a tick to attach, then write two entries.
    await new Promise((r) => setTimeout(r, 50));
    await append({ agent: "a", event: "message", summary: "live1" });
    await new Promise((r) => setTimeout(r, 20));
    await append({ agent: "a", event: "message", summary: "live2" });

    // Bound the wait so a broken watcher doesn't hang CI forever.
    await Promise.race([
      consumer,
      new Promise((_, reject) =>
        setTimeout(() => {
          ac.abort();
          reject(new Error("tail timed out"));
        }, 5000),
      ),
    ]).catch(() => {
      /* swallow — we assert via `collected` below */
    });

    expect(collected).toContain("live1");
    expect(collected).toContain("live2");
  });
});
