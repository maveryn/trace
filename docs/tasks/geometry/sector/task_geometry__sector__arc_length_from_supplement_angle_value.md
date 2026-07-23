# `task_geometry__sector__arc_length_from_supplement_angle_value`

## Contract
1. Domain: `geometry`
2. Scene id: `sector`
3. Task id: `task_geometry__sector__arc_length_from_supplement_angle_value`
4. Supported `query_id`: `single`
5. Answer schema: `number`
6. Answer precision: `one_decimal`
7. Annotation schema: `bbox`

## Program Contract
- `solve_formula(circular_sector, target=arc_length, formula_schema=arc_length_from_radius_and_supplement_angle); scene=sector; scope=arc_length_from_supplement_angle_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_sector_formula_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
The diagram labels center `O` and endpoints `A`/`B`. Prompt-facing annotation is one scalar pixel bbox around arc `AB`, excluding point labels and numeric readout text.

Numeric readout text and point labels are visible in the image but are not part of the public annotation. Formula metadata remains private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/sector.yaml`
- Task module: `src/trace_tasks/tasks/geometry/sector/arc_length_from_supplement_angle_value.py`
