---
name: trace-task-design
description: Design or reshape a Trace task contract, including task-versus-query placement, answer and annotation schemas, sampling, balancing, and documentation. Use before implementing a new task or changing an existing task's public behavior.
---

# Trace Task Design

## Read first

1. `docs/contracts/BLUEPRINT.md`
2. `docs/workflows/TASK_AUTHORING.md`
3. `docs/contracts/TASK_UNIT_POLICY.md`
4. `docs/ACTIVE_TASK_INVENTORY.md`
5. The applicable document under `docs/domains/`

## Freeze the contract

1. Confirm `domain`, `scene_id`, `task_id`, and whether each semantic branch is a public task or an internal `query_id`.
2. Search the active inventory and neighboring task docs to avoid near-duplicate scenes and objectives.
3. Specify the scene and query structure, program schema, answer schema, annotation schema, uniqueness constraints, and trace fields.
4. Derive answer and annotation from one execution trace and select the most direct visual witness.
5. Design constructive or target-first sampling when feasible answer support depends on layout, board size, or other generated state.
6. Specify scene, task, optional query, and output-mode prompt layers.
7. List the tests, task documentation, inventory, domain contract, and prompt resources that must change together.

Use `$trace-task-unit-audit` for an ambiguous split or merge, `$trace-prompt-design` for prompt-facing changes, and `$trace-task-implementation` only after the contract is stable.
