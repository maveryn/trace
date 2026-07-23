---
name: trace-task-implementation
description: Implement or refactor Trace tasks, configs, registration, shared helpers, and source layout. Use after a task contract is stable and code must be added or changed in the public Trace package.
---

# Trace Task Implementation

## Read first

1. `docs/contracts/SYSTEM_ARCHITECTURE.md`
2. `docs/contracts/SOURCE_LAYOUT.md`
3. `docs/workflows/TASK_AUTHORING.md`
4. `docs/workflows/DOC_STRUCTURE.md`
5. `docs/workflows/DOCS_AND_SKILLS_MAINTENANCE.md`

## Implement

1. Place task modules under `src/trace_tasks/tasks/<domain>/<scene_id>/` and reusable helpers at the narrowest shared package that has multiple real callers.
2. Search `src/trace_tasks/core/`, task shared packages, and domain shared packages before adding utilities.
3. Keep prompts in packaged prompt bundles and defaults in packaged YAML resources.
4. Register through the active public registry without compatibility aliases or broad eager imports.
5. Build the scene, execution trace, answer, projected annotation, and verifier payload through one deterministic path.
6. Reject ambiguous samples without relaxing semantic constraints or introducing unrecorded randomness.
7. Remove obsolete wrappers and exports, update affected docs, and run focused tests before broader validation.

Stop and use `$trace-task-unit-audit` if the task/query boundary changes. Use `$trace-prompt-design` for prompt changes and finish with `$trace-verification-review`.
