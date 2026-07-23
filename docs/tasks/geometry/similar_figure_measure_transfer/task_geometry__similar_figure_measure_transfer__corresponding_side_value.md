# `task_geometry__similar_figure_measure_transfer__corresponding_side_value`

## Contract
1. Domain: `geometry`
2. Scene id: `similar_figure_measure_transfer`
5. Query id: `single`
6. Answer schema: `integer`
7. Annotation schema: `point_map`

## Program Contract
- `solve_formula(visible_similar_figure_side_relation, unknown_role=target_corresponding_side_length, formula_schema=linear_scale_transfer); scene=similar_figure_measure_transfer; scope=corresponding_side_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_similar_figure_measure_transfer_v0`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Query and Construction Axes
- `single` is the only public query id.
- `construction_family` is replay metadata and may be `direct_side_transfer`, `two_pair_side_transfer`, or `nested_side_transfer`.

## Annotation
Prompt-facing annotation uses a `point_map` keyed by the visible point labels needed for the requested side and support side pairs, such as `A`, `B`, `A'`, and `B'`.

Side labels, tick marks, figure labels, scale factors, and derived values remain visible scene content plus private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/similar_figure_measure_transfer.yaml`
- Task module: `src/trace_tasks/tasks/geometry/similar_figure_measure_transfer/corresponding_side_value.py`
