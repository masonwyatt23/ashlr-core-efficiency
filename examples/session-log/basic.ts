/**
 * basic.ts — demonstrates @ashlr/core-efficiency/session-log
 *
 * Subpath: @ashlr/core-efficiency/session-log
 *
 * Records tool-call events during an agent session and prints a cost/savings
 * summary at the end.
 *
 * Run:
 *   bun run examples/session-log/basic.ts
 *
 * Windows note: no shell-outs; runs identically on macOS / Linux / Windows.
 */
import { SessionLog } from "../../src/session-log/index.ts";

const log = new SessionLog();

// Simulate recording events from a short agent session.
log.record({ type: "tool_call", tool: "ashlr__read",  inputTokens: 120, savedTokens: 840 });
log.record({ type: "tool_call", tool: "ashlr__grep",  inputTokens:  64, savedTokens: 192 });
log.record({ type: "tool_call", tool: "ashlr__edit",  inputTokens:  88, savedTokens:   0 });
log.record({ type: "tool_call", tool: "ashlr__read",  inputTokens:  97, savedTokens: 603 });

const summary = log.summary();
console.log("Session summary:");
console.log(JSON.stringify(summary, null, 2));
