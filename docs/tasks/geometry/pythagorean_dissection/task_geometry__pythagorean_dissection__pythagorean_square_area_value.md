# `task_geometry__pythagorean_dissection__pythagorean_square_area_value`

## Contract
1. Domain: `geometry`
2. Scene id: `pythagorean_dissection`
3. Task id: `task_geometry__pythagorean_dissection__pythagorean_square_area_value`
4. Query id: `single`
5. Answer schema: `integer_value`
6. Annotation schema: `point_map`
7. Scalar annotation checked: `true` (not scalar-eligible; the task always asks for multiple role-bound visual witnesses)

## Program Contract
- `derive_geometry_metric(visible_pythagorean_square_dissection, derivation_rule=leg_a_squared_plus_leg_b_squared, output_role=square_EFGH_area); scene=pythagorean_dissection; scope=pythagorean_square_area_value`

## Reasoning Operations

Families: `formula_evaluation`

## Query Semantics
- `single` is the public no-branch query. Segment values, answer support, fill palette, orientation, font, layout, and whole-scene rotation are internal replay metadata.

## Prompt Bundle
- Prompt text is loaded from `src/trace_tasks/resources/prompts/geometry/pythagorean_dissection/geometry_pythagorean_dissection_v1.json`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is a `point_map` with keys `E`, `F`, `G`, and `H`. Each value is the pixel point for that labeled vertex of the target square after final layout and whole-scene rotation. The outer-square labels `A`, `B`, `C`, and `D` plus the visible segment labels are context but are not part of the annotation contract.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/pythagorean_dissection.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/geometry/pythagorean_dissection/geometry_pythagorean_dissection_v1.json`
- Task module: `src/trace_tasks/tasks/geometry/pythagorean_dissection/pythagorean_square_area_value.py`
