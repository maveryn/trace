# `task_icons__overlap_grid__occlusion_order_count`

## Program Contract

Program: `count.relation_attribute(scene=overlap_grid, scope=curated_icon_overlap_cells, candidates=labeled_scene_cells, relation=front_to_back_order_equals_reference, output=integer)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `curated_icon_overlap_cells` objective scope.
Operands: visible scene state and prompt-bound operands named by `overlap_grid`, `curated_icon_overlap_cells`, `candidates`, `labeled_scene_cells`, `relation`, `front_to_back_order_equals_reference`.
Operation: evaluate `count.relation_attribute` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `spatial_relations`

## Identity

- Domain: `icons`
- Scene id: `overlap_grid`
- Task id: `task_icons__overlap_grid__occlusion_order_count`
- Objective contract: count labeled Scene cells whose front-to-back overlap order matches the Reference cell.
- Module: `src/trace_tasks/tasks/icons/overlap_grid/occlusion_order_count.py`
- Prompt bundle: `src/trace_tasks/resources/prompts/icons/overlap_grid/icons_overlap_grid_v1.json`

## Contract

- Supported `query_id` values: `single`.
- Answer schema: `integer`.
- Annotation schema: `bbox_set`.
- The image contains one Reference overlap cell and a labeled Scene grid of overlap cells.
- The Reference and all Scene cells reuse one sampled pair of curated icon shapes; only front/back order decides whether a Scene cell matches.
- The fixed internal relation is `same_front_to_back_order`; it is recorded in trace metadata, not exposed as a public query branch.
- Target count defaults to `0..4`, distractor count defaults to `1..5`, and total Scene cells default to `2..8`.

## Prompt

- Prompt bundle: `icons_overlap_grid_v1`.
- `scene_key`: `overlap_grid_occlusion_order`.
- `task_key`: `occlusion_order_count`.
- Answer JSON shape: `{"answer":3}`.
- Answer+annotation JSON shape: `{"annotation":[[336,104,506,274],[532,104,702,274],[728,104,898,274]],"answer":3}`.

## Annotation

- `bbox_set` marks the matching Scene cells, sorted top-to-bottom then left-to-right.
- The task is not scalar-annotation eligible because the witness count can be zero, one, or many.
- The Reference cell and non-matching Scene cells are supporting context, not annotation witnesses.

## Trace

- `scene_ir.entities` contains the Reference pair and every labeled Scene cell.
- `scene_ir.relations.reference_order_id` records the Reference front/back order.
- `execution_trace.fixed_relation_id` records `same_front_to_back_order`.
- `execution_trace.matching_cell_labels` records the symbolic labels counted by the integer answer.
- `projected_annotation.bbox_set` is derived from the same matching Scene cells as the answer.

## Tests

- Behavior and trace tests: `tests/test_icons_overlap_grid_occlusion_order_tasks.py`.
- Determinism/build tests: `tests/test_icons_overlap_grid_occlusion_order_contracts.py`.
- Config tests: `tests/test_icons_scene_config.py`.
- Prompt bundle tests: `tests/test_prompt_system.py`.
- Source-layout contract checks: `tests/test_public_source_layout_contracts.py`.
