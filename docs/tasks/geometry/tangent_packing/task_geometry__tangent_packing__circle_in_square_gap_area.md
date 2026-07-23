# `task_geometry__tangent_packing__circle_in_square_gap_area`

## Contract
1. Domain: `geometry`
2. Scene id: `tangent_packing`
3. Task id: `task_geometry__tangent_packing__circle_in_square_gap_area`
4. Supported `query_id`: `single`
5. Answer schema: `number`
6. Answer precision: `one_decimal`
7. Annotation schema: `bbox`

## Program Contract
- `curvilinear_gap_area(container=square, packed_shape=circle, given=square_side, target=shaded_area); scene=tangent_packing; scope=circle_in_square_gap_area`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the v1 scene prompt bundle configured for `tangent_packing`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is one scalar pixel bbox around the marked target geometric region or shape, excluding numeric labels and measurement text.

The answer placeholder label is not a separate annotation witness. Formula metadata remains private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/tangent_packing.yaml`
- Task module: `src/trace_tasks/tasks/geometry/tangent_packing/circle_in_square_gap_area.py`
