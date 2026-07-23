# `task_geometry__sector__central_angle_from_sector_measure_value`

## Contract
1. Domain: `geometry`
2. Scene id: `sector`
3. Task id: `task_geometry__sector__central_angle_from_sector_measure_value`
4. Supported `query_id`: `from_arc_length`, `from_sector_area`
5. Answer schema: `number`
6. Answer precision: `one_decimal`
7. Annotation schema: `bbox`

## Program Contract
- `solve_formula(circular_sector, target=central_angle, formula_schema=derive_central_angle_from_visible_sector_measure_and_radius); scene=sector; scope=central_angle_from_sector_measure_value`
- Branch `from_arc_length`: derive the sector central angle from visible radius and arc length.
- Branch `from_sector_area`: derive the sector central angle from visible radius and sector area.

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_sector_formula_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
The diagram labels center `O` and endpoints `A`/`B`. Prompt-facing annotation is one scalar pixel bbox around the marked angle `AOB`, excluding point labels and numeric readout text.

Numeric readout text and point labels are visible in the image but are not part of the public annotation. Formula metadata remains private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/sector.yaml`
- Task module: `src/trace_tasks/tasks/geometry/sector/central_angle_from_sector_measure_value.py`
