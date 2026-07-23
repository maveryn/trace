# `task_geometry__cuboid_views__cuboid_projection_surface_area_value`

## Contract
1. Domain: `geometry`
2. Scene id: `cuboid_views`
5. Query id: `single`
6. Answer schema: `integer`
7. Annotation schema: `bbox_map`

## Program Contract
- `solve_formula(visible_cuboid_views_measurements, unknown_role=surface_area, formula_schema=surface_area_from_orthographic_views); scene=cuboid_views; scope=cuboid_projection_surface_area_value`

## Reasoning Operations

Families: `transformation`, `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `cuboid_views`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space bboxes for the three role-bound orthographic rectangles: `top_view`, `front_view`, and `right_view`. The role binding matters because each rectangle contributes a different pair of cuboid dimensions. Dimensions, formulas, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/cuboid_views.yaml`
- Task module: `src/trace_tasks/tasks/geometry/cuboid_views/cuboid_projection_surface_area_value.py`
