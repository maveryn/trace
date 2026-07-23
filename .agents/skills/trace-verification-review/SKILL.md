---
name: trace-verification-review
description: Validate Trace changes with focused tests, deterministic review recipes, materialized samples, distribution audits, and reviewer issue follow-up. Use after task, prompt, renderer, verifier, or shared-infrastructure changes.
---

# Trace Verification Review

## Read first

1. `docs/workflows/BUILD_VALIDATION.md`
2. `docs/contracts/VALIDATION_ERROR_CODES.md`
3. `docs/workflows/CODE_REVIEW_GUIDELINES.md`
4. `docs/review/REVIEW_RECIPE.md`
5. `docs/workflows/TASK_REVIEW_WEB_APP.md`

For prompt wording, bundles, slots, examples, or output modes, also read
`docs/review/PROMPT_REVIEW.md` and `docs/contracts/PROMPT_SYSTEM.md`.

## Verify

1. Run focused tests for the changed contract before generating review artifacts.
2. Expand the affected slice through shared dependencies. A prompt-bundle edit,
   for example, affects every task whose semantic metadata records that bundle's
   content hash, even when only one task's rendered wording changes.
3. Run integration checks when shared infrastructure or packaged resources move.
4. Materialize an unchanged-contract slice from the accepted canonical recipe.
   For an intentional semantic or source/resource change, commit the change and
   capture the affected draft slice twice from the same clean, frozen checkout
   under ignored `review/`. Never waive an old-recipe semantic mismatch.
5. Treat a draft slice only as review evidence. Promote accepted behavior through
   two matching complete `--all` captures. Keep the fixed recipe schema and
   canonical path; identify the replacement by its new manifest digest and
   producer revision in a reviewed commit. Do not splice or hand-edit rows.
6. Run filtered `trace-review verify`, then `trace-review audit`, and inspect
   prompt, image, answer, annotation, and verifier/trace surfaces in
   `trace-review serve`; inspect distribution diagnostics when the focused
   workflow produced them.
7. Treat semantic hash mismatches against the recipe under test as failures.
   Report host-native pixel or PNG drift as an environment warning unless strict
   rendering verification was explicitly requested.
8. Diagnose support skew through constructive sampling and the recorded seed
   stream before adding rejection loops.
9. Add a concise repair note to an existing issue after validation, but leave
   human resolution to the reviewer unless explicitly asked.

Keep materialized images, workbooks, databases, endpoint responses, and local
review state under ignored `review/`; never commit them.

Use `$trace-code-review` when reviewing a patch authored by someone else. Never substitute optional endpoint calibration for deterministic or manual review gates.
