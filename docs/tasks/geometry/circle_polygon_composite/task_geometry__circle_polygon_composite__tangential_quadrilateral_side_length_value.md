# `task_geometry__circle_polygon_composite__tangential_quadrilateral_side_length_value`

## Contract
1. Domain: `geometry`
2. Scene id: `circle_polygon_composite`
4. Query ids: `missing_side_from_tangent_quadrilateral`
5. Answer schema: `integer_value`
6. Annotation schema: `point_map`

## Program Contract
- `solve_formula(tangential_quadrilateral, unknown_role=missing_side_length, formula_schema=pitot_theorem); scene=circle_polygon_composite; scope=tangential_quadrilateral_side_length_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_circle_polygon_composite_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space keyed points for the four labeled quadrilateral vertices. Required keys are `A`, `B`, `C`, and `D`. Numeric side-length labels, tangency points, and the incircle are visible construction context plus verifier metadata, not annotation.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/circle_polygon_composite.yaml`
- Task module: `src/trace_tasks/tasks/geometry/circle_polygon_composite/tangential_quadrilateral_side_length_value.py`
