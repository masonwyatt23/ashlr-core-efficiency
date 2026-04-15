// Consolidates the 3 duplicate estimateTokens() heuristics found in ashlrcode
// (src/agent/context.ts, src/agent/system-prompt.ts, src/genome/manifest.ts).

export function estimateTokens(input: string | { content: unknown }[]): number {
  if (typeof input === "string") {
    return Math.ceil(input.length / 4);
  }
  // Message array — sum approximate chars from content
  let total = 0;
  for (const msg of input) {
    const c = msg.content;
    if (typeof c === "string") total += c.length;
    else if (Array.isArray(c)) {
      for (const block of c) {
        if (block && typeof block === "object" && "text" in block && typeof block.text === "string") {
          total += block.text.length;
        }
      }
    }
  }
  return Math.ceil(total / 4);
}
