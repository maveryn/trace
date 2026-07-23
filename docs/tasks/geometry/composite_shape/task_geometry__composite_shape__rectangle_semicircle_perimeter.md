# `task_geometry__composite_shape__rectangle_semicircle_perimeter`

## Contract
1. Domain: `geometry`
2. Scene id: `composite_shape`
3. Query id: `cap_perimeter`, `cutout_perimeter`
4. Answer schema: `number`
5. Answer precision: `one_decimal`
6. Annotation schema: `point_map`

## Program Contract
- `solve_formula(visible_composite_shape_measurements, unknown_role=perimeter_measure, formula_schema=rectangle_semicircle_boundary_with_side_remainders); scene=composite_shape; scope=rectangle_semicircle_perimeter`

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
- Task module: `src/trace_tasks/tasks/geometry/composite_shape/rectangle_semicircle_perimeter.py`
