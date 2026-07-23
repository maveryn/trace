---
name: trace-prompt-design
description: Design or revise Trace prompt bundles, prompt slots, JSON examples, metadata, and output-mode wiring. Use when a contribution changes user-facing task wording or prompt-facing answer and annotation contracts.
---

# Trace Prompt Design

## Read first

1. `docs/contracts/PROMPT_SYSTEM.md`
2. `docs/workflows/TASK_AUTHORING.md`
3. `docs/contracts/ANNOTATION_AND_REWARD_CONTRACTS.md` when examples include annotation
4. `docs/review/PROMPT_REVIEW.md`

## Design and implement

1. Preserve the scene, task, optional query, and output-mode composition layers.
2. Keep user-facing wording in versioned bundles under `src/trace_tasks/resources/prompts/`, never task-module constants.
3. Keep exactly five strong variants for each required template list unless the prompt-system contract changes first.
4. Make examples valid for the active answer and annotation schemas, including query-dependent shapes.
5. Use prompt slots for sampled values and shared formatters for public value representations.
6. Record the selected bundle, variant, slots, and mode in prompt metadata.
7. Run focused prompt-contract tests and inspect materialized samples with `$trace-verification-review`.

If wording changes the program, task boundary, or output schema, stop and use `$trace-task-unit-audit` before modifying the bundle.
