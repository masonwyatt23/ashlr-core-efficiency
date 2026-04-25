/**
 * Session resume for local models — rebuild context from session-log + genome.
 *
 * When a local model session starts (or resumes after a context reset), this
 * module builds a compact (~500-1000 token) context block from:
 *  - Last 10 session-log entries (what happened recently)
 *  - Top genome sections for the project (what matters here)
 *  - A 1-paragraph summary of the last session
 *
 * This gives the local model enough orientation to continue productively
 * without needing to re-read files that would blow the 32K context.
 */

import { sep } from "node:path";
import { read } from "../session-log/index.ts";
import type { SessionLogEntry } from "../session-log/types.ts";
import { retrieveSectionsV2, formatGenomeForPrompt } from "../genome/retriever.ts";

export interface SessionResumeContext {
  /** Last 10 session-log entries formatted as compact text. */
  recentActivity: string;
  /** Top genome sections for this project, formatted for prompt injection. */
  genomeContext: string;
  /** 1-paragraph summary of the last session's activity. */
  lastSessionSummary: string;
}

/**
 * Build a resume context block for a local model session.
 *
 * Reads the session log and genome, then formats everything into a compact
 * system prompt addition. Target: 500-1000 tokens total.
 *
 * @param cwd - Project working directory (used for genome retrieval and log filtering)
 */
export async function buildResumeContext(cwd: string): Promise<SessionResumeContext> {
  // Read last 10 session-log entries
  const entries = read({ limit: 10 });
  const recentActivity = formatEntries(entries);

  // Retrieve top genome sections (budget: 600 tokens — leaves room for activity + summary)
  const sections = await retrieveSectionsV2(cwd, "", 600);
  const genomeContext = formatGenomeForPrompt(sections);

  // Derive a 1-paragraph summary from the session-log entries
  const lastSessionSummary = deriveSessionSummary(entries, cwd);

  return { recentActivity, genomeContext, lastSessionSummary };
}

/**
 * Format session-log entries into compact, readable text.
 * Target: ~200-300 tokens for 10 entries.
 */
function formatEntries(entries: SessionLogEntry[]): string {
  if (entries.length === 0) return "(no recent activity)";

  return entries
    .map((e) => {
      const time = e.ts.slice(11, 19); // HH:MM:SS
      const parts = [time, e.agent, e.event];
      if (e.tool) parts.push(e.tool);
      if (e.path) parts.push(shortPath(e.path));
      if (e.summary) parts.push(`- ${e.summary}`);
      return parts.join(" ");
    })
    .join("\n");
}

/**
 * Derive a brief summary paragraph from session-log entries.
 * Pure heuristic — no LLM call (we can't afford the tokens on a local model).
 */
function deriveSessionSummary(entries: SessionLogEntry[], cwd: string): string {
  if (entries.length === 0) return "No prior session activity found.";

  // Filter to entries from this project
  const projectEntries = entries.filter(
    (e) => e.cwd && e.cwd.startsWith(cwd),
  );
  const source = projectEntries.length > 0 ? projectEntries : entries;

  // Collect unique files edited, tools used, and agents involved
  const files = new Set<string>();
  const tools = new Set<string>();
  const agents = new Set<string>();
  const summaries: string[] = [];

  for (const e of source) {
    if (e.path) files.add(shortPath(e.path));
    if (e.tool) tools.add(e.tool);
    agents.add(e.agent);
    if (e.summary) summaries.push(e.summary);
  }

  const parts: string[] = [];

  if (agents.size > 0) parts.push(`Agent(s): ${[...agents].join(", ")}`);
  if (files.size > 0) parts.push(`touched ${[...files].slice(0, 5).join(", ")}`);
  if (tools.size > 0) parts.push(`used ${[...tools].slice(0, 5).join(", ")}`);

  // Use the most recent summary entry as the headline
  const headline = summaries.length > 0
    ? summaries[0]
    : "Session activity recorded";

  return `${headline}. ${parts.join("; ")}.`;
}

/**
 * Shorten an absolute path for display — keep last 2 segments.
 * Normalises both forward-slash (POSIX) and backslash (Windows) separators
 * so the split works correctly on all platforms.
 */
function shortPath(p: string): string {
  // Normalise to platform sep, then split.
  const normalised = p.replace(/[\\/]/g, sep);
  const segments = normalised.split(sep);
  return segments.length > 2
    ? segments.slice(-2).join(sep)
    : p;
}
