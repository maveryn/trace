# `task_geometry__composite_shape__l_profile_area`

## Contract
1. Domain: `geometry`
2. Scene id: `composite_shape`
5. Query id: `single`
6. Answer schema: `integer_value`
7. Annotation schema: `point_map`

## Program Contract
- `solve_formula(visible_composite_shape_measurements, unknown_role=area_measure, formula_schema=outer_rectangle_minus_corner_rectangle); scene=composite_shape; scope=l_profile_area`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `composite_shape`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space witnesses only. Map annotation is used where witness roles matter; graph coordinates, formulas, labels, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/composite_shape.yaml`
- Task module: `src/trace_tasks/tasks/geometry/composite_shape/l_profile_area.py`
