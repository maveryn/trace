# `task_geometry__sector__related_angle_from_sector_measure_value`

## Contract
1. Domain: `geometry`
2. Scene id: `sector`
3. Task id: `task_geometry__sector__related_angle_from_sector_measure_value`
4. Supported `query_id`: `complement_from_arc_length`, `supplement_from_sector_area`, `remaining_from_arc_length`
5. Answer schema: `number`
6. Answer precision: `one_decimal`
7. Annotation schema: `bbox`

## Program Contract
- `solve_formula(circular_sector, target=related_angle, formula_schema=derive_sector_angle_from_visible_measure_then_apply_visible_angle_relation); scene=sector; scope=related_angle_from_sector_measure_value`
- Branch `complement_from_arc_length`: derive the sector angle from visible radius and arc length, then compute the marked complementary angle.
- Branch `supplement_from_sector_area`: derive the sector angle from visible radius and sector area, then compute the marked supplementary angle.
- Branch `remaining_from_arc_length`: derive the sector angle from visible radius and arc length, then compute the marked remaining angle around the circle.

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_sector_formula_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
The diagram labels center `O` and endpoints `A`/`B`, plus `C` when an extra relation ray is drawn. Prompt-facing annotation is one scalar pixel bbox around the marked related-angle arc at `O`, excluding point labels and numeric readout text.

Numeric readout text and point labels are visible in the image but are not part of the public annotation. Formula metadata remains private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/sector.yaml`
- Task module: `src/trace_tasks/tasks/geometry/sector/related_angle_from_sector_measure_value.py`
