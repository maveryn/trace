# `task_geometry__triangle_relations__split_triangle_angle_value`

## Contract
1. Domain: `geometry`
2. Scene id: `triangle_relations`
3. Query id: `single`
4. Answer schema: `integer`
5. Annotation schema: `point_map`
6. Scalar annotation checked: true

## Program Contract
- `solve_formula(split_triangle_angle_sum, unknown_role=target_angle_measure, formula_schema=triangle_angle_sum_or_adjacent_straight_angle); scene=triangle_relations; scope=split_triangle_angle_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `triangle_relations`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses keyed pixel points for the labeled construction points `A`, `B`, `C`, and `D`. Angle labels and internal construction family remain visible diagram content plus private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/triangle_relations.yaml`
- Task module: `src/trace_tasks/tasks/geometry/triangle_relations/split_triangle_angle_value.py`
