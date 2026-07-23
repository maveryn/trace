# `task_geometry__similar_figure_measure_transfer__variable_value`

## Contract
1. Domain: `geometry`
2. Scene id: `similar_figure_measure_transfer`
3. Query id: `single`
4. Answer schema: `number`
5. Answer precision: `one_decimal`
6. Annotation schema: `point_map`

## Program Contract
- `solve_formula(visible_similar_figure_marked_side_equation, unknown_role=variable_value, formula_schema=similar_side_ratio); scene=similar_figure_measure_transfer; scope=variable_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_geo3k_marked_equations_v0`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Query and Construction Axes
- `single` is the only public query id.
- `construction_family` is replay metadata and may be `triangle_ratio`, `polygon_ratio`, or `two_expression_ratio`.

## Annotation
Prompt-facing annotation uses a `point_map` keyed by visible point labels for the side pair containing the variable and the supporting corresponding side pair.

Expression labels, tick marks, vertex labels, and solved variable values remain visible scene content plus private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/similar_figure_measure_transfer.yaml`
- Task module: `src/trace_tasks/tasks/geometry/similar_figure_measure_transfer/variable_value.py`
