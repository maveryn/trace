# Domain Docs

Use this folder for domain-specific contracts only, including domain scene
boundaries, annotation conventions, prompt constraints, and rendering/style
rules that are not shared across every domain.

Active contract docs should stay narrow and current. Planning notes, transition
notes, external benchmark inventories, task lists, and generated coverage
reports do not belong in this folder.

Workflow skills may route agents to these files, but canonical domain
contracts live here.

## Domain Contract Docs
- `charts.md` — chart, map, table, and data-display contracts.
- `games.md` — game-state, board, card, arcade, and rule contracts.
- `geometry.md` — geometric diagram, coordinate, and measurement contracts.
- `graph.md` — graph, tree, network, and route contracts.
- `icons.md` — icon-field, icon-relation, and icon-asset contracts.
- `illustrations.md` — synthetic object illustration and object-part contracts.
- `symbolic.md` — miscellaneous notation and instrument-like renderer contracts.
- `pages.md` — page-like form, document, diagram, map, GUI, and web contracts.
- `physics.md` — diagram-grounded physics contracts.
- `puzzles.md` — puzzle-state, rule-grid, word, and visual puzzle contracts.
- `three_d.md` — synthetic 3D scene, camera, projection, and object contracts.

## What Belongs Elsewhere
- Active scene/task counts: `docs/ACTIVE_TASK_INVENTORY.md`.
- Cross-domain task-boundary policy: `docs/contracts/TASK_UNIT_POLICY.md`.
- Source layout and shared-code ownership: `docs/contracts/SOURCE_LAYOUT.md`.
- Task authoring workflow: `docs/workflows/TASK_AUTHORING.md`.
