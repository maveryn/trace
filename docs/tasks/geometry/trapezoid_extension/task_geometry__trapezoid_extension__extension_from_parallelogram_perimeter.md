# `task_geometry__trapezoid_extension__extension_from_parallelogram_perimeter`

## Contract
1. Domain: `geometry`
2. Scene id: `trapezoid_extension`
3. Task id: `task_geometry__trapezoid_extension__extension_from_parallelogram_perimeter`
4. Supported `query_id` values: `single`
5. Answer schema: `number`
6. Answer precision: `one_decimal`
7. Annotation schema: `segment`
8. Scalar annotation checked: `true`

## Program Contract
- `solve_formula(visible_trapezoid_extension_measurements, unknown_role=length_measure, formula_schema=extension_from_parallelogram_perimeter); scene=trapezoid_extension; scope=extension_from_parallelogram_perimeter`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `src/trace_tasks/resources/prompts/geometry/trapezoid_extension/geometry_trapezoid_extension_v1.json`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is one pixel-space segment `[[x0,y0],[x1,y1]]` for the requested extension segment `BE`. Numeric labels and construction metadata remain private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/trapezoid_extension.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/geometry/trapezoid_extension/geometry_trapezoid_extension_v1.json`
- Task module: `src/trace_tasks/tasks/geometry/trapezoid_extension/extension_from_parallelogram_perimeter.py`
