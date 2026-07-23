# `task_geometry__circle_polygon_composite__tangent_angle_value`

## Contract
1. Domain: `geometry`
2. Scene id: `circle_polygon_composite`
4. Query id: `single`
5. Answer schema: `integer_value`
6. Annotation schema: `point_map`

## Program Contract
- `solve_formula(circle_polygon_tangent_angle_construction, unknown_role=target_angle, formula_schema=tangent_radius_perpendicular_angle_transfer); scene=circle_polygon_composite; scope=tangent_angle_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_circle_polygon_composite_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space keyed points for the visible construction labels. Required keys are `A`, `B`, `C`, `D`, `O`, and `T`. Numeric angle labels and angle arcs are visible diagram marks plus verifier metadata, not annotation.

Construction kind (`incircle` vs. `semicircle`) is sampled as trace metadata, not as a public query branch. The internal formula tag is `tangent_angle_from_radius_perpendicular`; public query metadata uses `single`. The prompt still describes the visible construction specifically for each generated instance.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/circle_polygon_composite.yaml`
- Task module: `src/trace_tasks/tasks/geometry/circle_polygon_composite/tangent_angle_value.py`
