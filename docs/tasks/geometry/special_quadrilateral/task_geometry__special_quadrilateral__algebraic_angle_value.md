# `task_geometry__special_quadrilateral__algebraic_angle_value`

## Contract
1. Domain: `geometry`
2. Scene id: `special_quadrilateral`
3. Task id: `task_geometry__special_quadrilateral__algebraic_angle_value`
4. Supported `query_id`: `parallelogram_opposite_angle_expression`, `parallelogram_consecutive_angle_expression`, `rhombus_diagonal_half_angle_expression`, `kite_opposite_angle_expression`
5. Answer schema: `integer`
6. Annotation schema: `point_map`

## Program Contract
- `formula.solve_unknown(visible_quadrilateral=parallelogram|rhombus|kite, visible_expressions=two_marked_angle_expressions, relation=opposite_angles_equal|consecutive_angles_supplementary|diagonal_bisects_angle, target=marked_angle_measure); scene=special_quadrilateral; scope=algebraic_angle_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_special_quadrilateral_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation maps each visible witness point label to its pixel point `[x,y]`.
The keys are the visible point labels required by the active construction: `A`, `B`, `C`, `D`, and `O` when a diagonal intersection is shown.
The task keeps `point_map` rather than scalar annotation because multiple role-bound labeled points are required.

## Query Semantics
- `parallelogram_opposite_angle_expression`: equal opposite angles in a parallelogram.
- `parallelogram_consecutive_angle_expression`: supplementary consecutive angles in a parallelogram.
- `rhombus_diagonal_half_angle_expression`: rhombus diagonal bisects a vertex angle.
- `kite_opposite_angle_expression`: equal marked non-vertex opposite angles in a kite.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/special_quadrilateral.yaml`
- Task module: `src/trace_tasks/tasks/geometry/special_quadrilateral/algebraic_angle_value.py`
