/**
 * Priority slots for prompt-part ordering.
 *
 * Lower values sort earlier in the assembled prompt and are more likely to
 * survive token-budget pruning. Numeric values are stable so callers passing
 * raw ints still work.
 */
export enum PromptPriority {
  Core = 0,
  PlanMode = 5,
  Tools = 10,
  Permissions = 20,
  Genome = 25,
  Knowledge = 30,
  Git = 35,
  Memory = 40,
  Default = 50,
  BuddyInfluence = 85,
  ModelPatches = 90,
  Undercover = 95,
}
