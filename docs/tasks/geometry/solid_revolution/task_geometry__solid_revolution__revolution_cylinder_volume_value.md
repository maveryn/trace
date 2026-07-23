# `task_geometry__solid_revolution__revolution_cylinder_volume_value`

## Contract
1. Domain: `geometry`
2. Scene id: `solid_revolution`
3. Task id: `task_geometry__solid_revolution__revolution_cylinder_volume_value`
4. Supported `query_id` values: `single`
5. Answer schema: `number`
6. Answer precision: `one_decimal`
7. Annotation schema: `bbox_map`
8. Scalar annotation checked: `true` (not scalar-eligible; the task binds source and result shape witnesses)

## Program Contract
- `solve_formula(visible_solid_revolution_measurements, formula_schema=cylinder_volume_from_rectangle, target=volume); scene=solid_revolution; scope=revolution_cylinder_volume_value`

## Reasoning Operations

Families: `transformation`, `formula_evaluation`

## Query IDs
- `single`: a rectangle is rotated 360 degrees around the marked axis; solve the resulting cylinder volume from the visible height and diameter.

## Prompt Bundle
- Prompt text is loaded from `src/trace_tasks/resources/prompts/geometry/solid_revolution/geometry_solid_revolution_v1.json`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
The annotation is a `bbox_map` with role-bound pixel boxes for the source generating diagram and the resulting solid. Numeric labels, individual dimension marks, the rotation arrow, and formula cues remain visible scene content plus private verifier metadata, not separate annotation targets:

- `source_diagram_bbox`
- `resulting_solid_bbox`

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/solid_revolution.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/geometry/solid_revolution/geometry_solid_revolution_v1.json`
- Task module: `src/trace_tasks/tasks/geometry/solid_revolution/revolution_cylinder_volume_value.py`
