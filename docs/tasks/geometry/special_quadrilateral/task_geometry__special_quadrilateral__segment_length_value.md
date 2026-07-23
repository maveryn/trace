# `task_geometry__special_quadrilateral__segment_length_value`

## Contract
1. Domain: `geometry`
2. Scene id: `special_quadrilateral`
3. Task id: `task_geometry__special_quadrilateral__segment_length_value`
4. Supported `query_id`: `parallelogram_opposite_side_expression`, `rhombus_all_sides_expression`, `kite_adjacent_equal_side_expression`, `parallelogram_diagonal_bisection_expression`
5. Answer schema: `integer`
6. Annotation schema: `point_map`

## Program Contract
- `formula.solve_unknown(visible_quadrilateral=parallelogram|rhombus|kite, visible_expressions=two_marked_side_or_diagonal_segment_expressions, relation=opposite_sides_equal|all_sides_equal|adjacent_kite_sides_equal|diagonals_bisect, target=marked_segment_length); scene=special_quadrilateral; scope=segment_length_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_special_quadrilateral_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation maps each visible witness point label to its pixel point `[x,y]`.
The keys are the visible point labels required by the active construction: `A`, `B`, `C`, `D`, and `O` when a diagonal intersection is shown.
The task keeps `point_map` rather than scalar segment annotation because the same scene contract also binds support points used by the algebraic relation.

## Query Semantics
- `parallelogram_opposite_side_expression`: opposite side lengths in a parallelogram are equal.
- `rhombus_all_sides_expression`: all side lengths in a rhombus are equal.
- `kite_adjacent_equal_side_expression`: marked adjacent side lengths in a kite are equal.
- `parallelogram_diagonal_bisection_expression`: diagonals of a parallelogram bisect each other.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/special_quadrilateral.yaml`
- Task module: `src/trace_tasks/tasks/geometry/special_quadrilateral/segment_length_value.py`
