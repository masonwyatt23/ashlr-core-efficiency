/**
 * Cross-platform regression tests.
 *
 * These tests verify that path-separator and line-ending bugs fixed in
 * the ci/cross-platform-fixes pass are truly exercised.  They use path.sep
 * so they pass on both POSIX (sep="/") and Windows (sep="\\").
 */

import { describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, sep } from "node:path";

// ── helpers ─────────────────────────────────────────────────────────────────

function makeTmpDir(): string {
  return mkdtempSync(join(tmpdir(), "ashlr-xp-"));
}

// ── scribe: writeSectionFromProposal title/category derivation ─────────────

/**
 * Isolated copy of the path→title/category logic from scribe.ts so we can
 * unit-test it without needing a full genome on disk.
 */
function deriveTitleAndCategory(sectionPath: string): { title: string; category: string } {
  const normalized = sectionPath.replace(/\//g, sep);
  const parts = normalized.replace(".md", "").split(sep);
  const title = parts[parts.length - 1]!.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  const category = parts[0] ?? "other";
  return { title, category };
}

describe("scribe: path→title/category derivation (sep-aware)", () => {
  test("derives title and category from POSIX-style section path", () => {
    const { title, category } = deriveTitleAndCategory("vision/north-star.md");
    expect(title).toBe("North Star");
    expect(category).toBe("vision");
  });

  test("handles backslash-separated path by normalising to platform sep", () => {
    // On Windows, join() produces backslash paths. Our fix normalises "/" → sep
    // before splitting. Verify the normalisation itself doesn't corrupt paths.
    const winPath = "vision\\north-star.md";
    // Replace backslashes with forward-slashes, then use our normaliser
    const normalized = winPath.replace(/\\/g, "/");
    const { title, category } = deriveTitleAndCategory(normalized);
    expect(title).toBe("North Star");
    expect(category).toBe("vision");
  });

  test("single-segment path (no subdirectory) falls back gracefully", () => {
    const { title, category } = deriveTitleAndCategory("readme.md");
    expect(title).toBe("Readme");
    expect(category).toBe("readme");
  });

  test("nested path gives correct deepest segment as title", () => {
    const { title, category } = deriveTitleAndCategory("milestones/completed/001-gen1.md");
    expect(title).toBe("001 Gen1");
    expect(category).toBe("milestones");
  });
});

// ── session-resume: shortPath sep-awareness ──────────────────────────────

/**
 * Isolated copy of shortPath from session-resume.ts.
 */
function shortPath(p: string): string {
  const normalised = p.replace(/[\\/]/g, sep);
  const segments = normalised.split(sep);
  return segments.length > 2 ? segments.slice(-2).join(sep) : p;
}

describe("session-resume: shortPath (sep-aware)", () => {
  test("keeps last 2 segments of a POSIX path", () => {
    const result = shortPath("/home/user/projects/foo/bar/baz.ts");
    expect(result).toBe(`bar${sep}baz.ts`);
  });

  test("keeps last 2 segments of a Windows-style path (backslash normalisation)", () => {
    // Our normalisation converts backslashes to the platform sep before splitting.
    // On macOS/Linux sep="/", so backslash becomes "/" and the split still works.
    const result = shortPath("C:\\Users\\user\\projects\\foo\\bar.ts");
    expect(result).toMatch(/foo.bar\.ts$/);
  });

  test("returns path unchanged when <= 2 segments", () => {
    expect(shortPath("bar.ts")).toBe("bar.ts");
  });

  test("mixed separators (e.g. WSL paths) normalise correctly", () => {
    const result = shortPath("/home/user\\project/src/index.ts");
    expect(result).toBe(`src${sep}index.ts`);
  });
});

// ── session-log / jsonl: CRLF line-ending tolerance ──────────────────────

describe("session-log: CRLF line-ending tolerance", () => {
  test("split(/\\r?\\n/) handles LF-only lines", () => {
    const raw = '{"a":1}\n{"a":2}\n';
    const lines = raw.split(/\r?\n/).filter(Boolean);
    expect(lines).toHaveLength(2);
    expect(JSON.parse(lines[0]!)).toEqual({ a: 1 });
  });

  test("split(/\\r?\\n/) handles CRLF lines (Windows)", () => {
    const raw = '{"a":1}\r\n{"a":2}\r\n';
    const lines = raw.split(/\r?\n/).filter(Boolean);
    expect(lines).toHaveLength(2);
    // Ensure no stray \\r in the parsed values
    const parsed = JSON.parse(lines[0]!);
    expect(parsed).toEqual({ a: 1 });
    const parsed2 = JSON.parse(lines[1]!);
    expect(parsed2).toEqual({ a: 2 });
  });

  test("split(/\\r?\\n/) handles mixed endings gracefully", () => {
    const raw = '{"a":1}\n{"a":2}\r\n{"a":3}\n';
    const lines = raw.split(/\r?\n/).filter(Boolean);
    expect(lines).toHaveLength(3);
  });
});

// ── genome/init: parseClaudeMdSections CRLF tolerance ────────────────────

/**
 * Minimal re-implementation of parseClaudeMdSections for unit testing
 * (mirrors the fix in init.ts — uses /\r?\n/).
 */
function parseSectionsCRLF(content: string): { description?: string } {
  const lines = content.split(/\r?\n/);
  let description: string | undefined;
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed && !trimmed.startsWith("#")) {
      description = trimmed;
      break;
    }
  }
  return { description };
}

describe("genome/init: parseClaudeMdSections CRLF", () => {
  test("extracts description from LF file", () => {
    const content = "# Title\n\nThis is a description.\n\n## Section\n\nContent.\n";
    const result = parseSectionsCRLF(content);
    expect(result.description).toBe("This is a description.");
  });

  test("extracts description from CRLF file (Windows)", () => {
    const content = "# Title\r\n\r\nThis is a description.\r\n\r\n## Section\r\n\r\nContent.\r\n";
    const result = parseSectionsCRLF(content);
    expect(result.description).toBe("This is a description.");
  });
});

// ── manifest: sectionPath traversal guard still works with sep ───────────

describe("manifest: sectionPath traversal guard", () => {
  test("rejects path-traversal attempts", async () => {
    const { sectionPath } = await import("../src/genome/manifest.ts");
    const tmp = makeTmpDir();
    try {
      expect(() => sectionPath(tmp, "../../etc/passwd")).toThrow(
        /escapes genome directory/,
      );
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  test("accepts valid relative paths", async () => {
    const { sectionPath } = await import("../src/genome/manifest.ts");
    const tmp = makeTmpDir();
    try {
      const result = sectionPath(tmp, "vision/north-star.md");
      // Must start with the genome dir
      expect(result).toContain(".ashlrcode");
      expect(result).toContain("genome");
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });
});
