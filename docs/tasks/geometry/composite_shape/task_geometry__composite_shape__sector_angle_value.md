# `task_geometry__composite_shape__sector_angle_value`

## Contract
1. Domain: `geometry`
2. Scene id: `composite_shape`
3. Query id: `from_arc_length`, `from_sector_area`
4. Answer schema: `number`
5. Answer precision: `one_decimal`
6. Annotation schema: `point_map`

## Program Contract
- `derive_geometry_metric(visible_composite_shape_measurements, derivation_rule=sector_angle_from_visible_measure, output_role=central_angle); scene=composite_shape; scope=sector_angle_value`

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
- Task module: `src/trace_tasks/tasks/geometry/composite_shape/sector_angle_value.py`
