---
name: trace-code-review
description: Review Trace changes for contract drift, helper placement, documentation synchronization, duplication, and validation gaps. Use for implementation reviews, refactor reviews, and pre-merge checks in the Trace repository.
---

# Trace Code Review

## Read first

1. `docs/workflows/CODE_REVIEW_GUIDELINES.md`
2. `docs/contracts/SYSTEM_ARCHITECTURE.md`
3. `docs/contracts/SOURCE_LAYOUT.md`
4. `docs/contracts/BLUEPRINT.md`

## Review

1. Inspect the diff and identify every changed contract surface before judging individual lines.
2. Confirm helpers live at the narrowest reusable ownership layer and deterministic logic is not duplicated.
3. Confirm prompt text stays external, prompt metadata remains complete, and answer, annotation, witness, and trace projections share one execution path.
4. Check public task ids, filenames, imports, and module layout against the source-layout contract.
5. Require focused regression tests and synchronized docs for code, config, resource, or boundary changes.
6. Flag dead shims, stale exports, hidden randomness, silent constraint relaxation, credentials, and machine-specific paths.
7. Report findings by severity with a concrete file reference and rationale. Do not edit unless the user requested implementation.

## Route related work

- Use `$trace-task-unit-audit` when task or query boundaries are unclear.
- Use `$trace-prompt-design` when prompt bundles or output examples change.
- Use `$trace-verification-review` when behavior or generated review artifacts change.

Update the source-of-truth documentation when a finding reveals a reusable rule; keep this skill as workflow routing rather than duplicated policy.
