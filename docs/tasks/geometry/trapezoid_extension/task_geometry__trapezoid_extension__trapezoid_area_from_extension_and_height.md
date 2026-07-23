# `task_geometry__trapezoid_extension__trapezoid_area_from_extension_and_height`

## Contract
1. Domain: `geometry`
2. Scene id: `trapezoid_extension`
3. Task id: `task_geometry__trapezoid_extension__trapezoid_area_from_extension_and_height`
4. Supported `query_id` values: `single`
5. Answer schema: `number`
6. Answer precision: `one_decimal`
7. Annotation schema: `bbox`
8. Scalar annotation checked: `true`

## Program Contract
- `solve_formula(visible_trapezoid_extension_measurements, unknown_role=area_measure, formula_schema=trapezoid_area_from_extension_and_height); scene=trapezoid_extension; scope=trapezoid_area_from_extension_and_height`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `src/trace_tasks/resources/prompts/geometry/trapezoid_extension/geometry_trapezoid_extension_v1.json`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is one pixel-space bbox `[x0,y0,x1,y1]` around the original trapezoid `ABCD`. Numeric labels and construction metadata remain private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/trapezoid_extension.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/geometry/trapezoid_extension/geometry_trapezoid_extension_v1.json`
- Task module: `src/trace_tasks/tasks/geometry/trapezoid_extension/trapezoid_area_from_extension_and_height.py`
