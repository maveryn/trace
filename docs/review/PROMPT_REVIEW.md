# Prompt Review

Review prompt text against `docs/contracts/PROMPT_SYSTEM.md` and the applicable
task contract. Classify a finding as `good`, `bad`, or `borderline`, and explain
borderline design tradeoffs instead of turning preferences into failures.

## Scope

Resolve the prompt bundle and keys from the task's domain/scene config, then
enumerate every task that consumes the same bundle. Prompt metadata records the
hash of the complete bundle file, so editing one task's template or default slot
rotates semantic identity for all consumers of that bundle. Use this expanded
consumer set for recipe capture and verification, while keeping the wording
finding attached to the task or layer whose rendered text actually changed.

A wording change that alters the program, answer schema, or witness roles is not
prompt-only; review the task contract before continuing.

## Checks

- The prompt asks exactly for the generated answer and names visible scope
  precisely enough to avoid multiple interpretations.
- Scene, task, optional query, and output-mode text compose without duplicated
  instructions or contradictory examples.
- Every required template list contains five useful semantic variants, not
  superficial punctuation changes.
- Sampled values enter through recorded prompt slots; task modules contain no
  user-facing prompt constants.
- Both output modes state contract-valid JSON shapes.
- Examples match the active answer support, annotation type, and query branch.
- Labels are quoted when needed, option markers agree with the image, and the
  prompt does not expose task ids, query ids, trace fields, or generator state.
- Wording uses `annotation` consistently and does not ask models to infer
  information that is absent or illegible in the image.
- Prompt metadata identifies the bundle version, selected variants, slots, and
  output mode needed to replay the prompt.

## Verification

1. Add or update a focused regression for the required rule and scope language,
   rendered prompt composition, both output modes, and recorded slot metadata.
2. Run the task/scene prompt tests and the prompt-bundle schema tests before
   generating review artifacts.
3. Commit the exact change, start from a clean frozen checkout, and follow
   `docs/review/REVIEW_RECIPE.md` to capture the expanded draft slice twice.
4. Verify the new draft rather than waiving expected failures against the older
   recipe. For prompt-only changes, require stable seeds, answers, annotations,
   execution semantics, raw pixels, and PNGs while prompt-related semantic
   fields rotate as expected.
5. Materialize the expanded slice and inspect every represented query and both
   output modes in `trace-review serve` before complete canonical promotion.

Inspect all query variants and output modes represented by the expanded recipe
slice. File a task-level issue for systematic wording and a sample-level issue
for a specific slot/render interaction.
