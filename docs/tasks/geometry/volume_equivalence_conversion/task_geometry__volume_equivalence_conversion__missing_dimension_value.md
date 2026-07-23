# `task_geometry__volume_equivalence_conversion__missing_dimension_value`

## Contract
1. Domain: `geometry`
2. Scene id: `volume_equivalence_conversion`
3. Task id: `task_geometry__volume_equivalence_conversion__missing_dimension_value`
4. Supported `query_id` values: `cuboid_to_cylinder_length`, `cylinder_to_cone_height`, `cone_to_cuboid_height`
5. Answer schema: `integer`
6. Annotation schema: `bbox_map`
7. Scalar annotation checked: `true` (not scalar-eligible; the task requires role-bound source and target solid boxes)

## Program Contract
- `solve_formula(equal_volume_solid_conversion, target=missing_dimension, formula_schema=volume_equivalence_missing_dimension); scene=volume_equivalence_conversion; scope=missing_dimension_value`

## Reasoning Operations

Families: `formula_evaluation`

## Query Semantics
- `cuboid_to_cylinder_length` asks for the target cylinder length that gives the same volume as the source cuboid.
- `cylinder_to_cone_height` asks for the target cone height that gives the same volume as the source cylinder.
- `cone_to_cuboid_height` asks for the target cuboid height that gives the same volume as the source cone.
- Numeric conversion cases, diagram style, labels, palette, and render retry index are internal replay metadata.

## Prompt Bundle
- Prompt text is loaded from `src/trace_tasks/resources/prompts/geometry/volume_equivalence_conversion/geometry_volume_equivalence_conversion_v1.json`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses a pixel-space bbox map with `source_solid_bbox` and `target_solid_bbox`. These boxes mark the source and target solid shapes as the canonical visual primitives for the equal-volume diagram. Numeric labels and the `?` marker remain visible diagram content and trace metadata, but are not public annotation targets.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/volume_equivalence_conversion.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/geometry/volume_equivalence_conversion/geometry_volume_equivalence_conversion_v1.json`
- Task module: `src/trace_tasks/tasks/geometry/volume_equivalence_conversion/missing_dimension_value.py`
