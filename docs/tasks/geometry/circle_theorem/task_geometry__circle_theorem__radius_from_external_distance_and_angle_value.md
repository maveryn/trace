# `task_geometry__circle_theorem__radius_from_external_distance_and_angle_value`

## Contract
1. Domain: `geometry`
2. Scene id: `circle_theorem`
3. Query id: `single`
4. Answer schema: `number`
5. Answer precision: `one_decimal`
6. Annotation schema: `point_map`

## Program Contract
- `solve_formula(visible_tangent_radius_right_triangle, unknown_role=radius_length, formula_schema=radius_from_external_distance_and_angle); scene=circle_theorem; scope=radius_from_external_distance_and_angle_value`

## Reasoning Operations

Families: `formula_evaluation`

## Query Branches
- `single`: given the center-to-exterior distance and the exterior angle to the tangent point, solve the circle radius.

## Prompt Bundle
- Prompt text is loaded from the geometry circle prompt bundle configured for this scene/task override.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses role-keyed pixel points for the circle center, tangent point, and exterior point. Length labels, angle labels, target cues, and the right-angle marker are visible diagram annotations plus verifier metadata, not separate public annotation objects.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/circle_theorem.yaml`
- Task module: `src/trace_tasks/tasks/geometry/circle_theorem/radius_from_external_distance_and_angle_value.py`
