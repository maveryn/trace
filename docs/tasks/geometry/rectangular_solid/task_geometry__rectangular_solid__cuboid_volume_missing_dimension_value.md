# `task_geometry__rectangular_solid__cuboid_volume_missing_dimension_value`

## Contract
1. Domain: `geometry`
2. Scene id: `rectangular_solid`
3. Task id: `task_geometry__rectangular_solid__cuboid_volume_missing_dimension_value`
4. Supported `query_id` values: `missing_length_from_volume`, `missing_width_from_volume`, `missing_height_from_volume`
5. Answer schema: `integer_value`
6. Annotation schema: `segment`
7. Scalar annotation checked: `true` (exactly one target dimension-guide segment witness)

## Program Contract
- `solve_formula(cuboid_volume_measurements, unknown_role=length|width|height, formula_schema=cuboid_volume_missing_dimension); scene=rectangular_solid; scope=cuboid_volume_missing_dimension_value`

## Reasoning Operations

Families: `formula_evaluation`

## Query Semantics
- `missing_length_from_volume` asks for the cuboid length `L` from visible volume, width, and height labels.
- `missing_width_from_volume` asks for the cuboid width `W` from visible volume, length, and height labels.
- `missing_height_from_volume` asks for the cuboid height `H` from visible volume, length, and width labels.
- The sampled cuboid dimensions, view jitter, face palette, style, font, and render retry index are internal replay metadata.

## Prompt Bundle
- Prompt text is loaded from `src/trace_tasks/resources/prompts/geometry/rectangular_solid/geometry_rectangular_solid_v1.json`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is the scalar pixel-space segment for the target dimension guide marked with `?`. Numeric labels, the volume readout, and the `?` marker remain visible diagram content and private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/rectangular_solid.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/geometry/rectangular_solid/geometry_rectangular_solid_v1.json`
- Task module: `src/trace_tasks/tasks/geometry/rectangular_solid/cuboid_volume_missing_dimension_value.py`
