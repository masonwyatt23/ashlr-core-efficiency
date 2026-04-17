/**
 * Local model session management — context window + session resume for
 * small-context models (32K and below).
 *
 * Subpath import: `@ashlr/core-efficiency/local`
 */

export { LocalContextWindow, type LocalContextConfig } from "./context-window.ts";
export { buildResumeContext, type SessionResumeContext } from "./session-resume.ts";
