# `task_geometry__solid_cross_section__cone_parallel_slice_area`

## Contract
1. Domain: `geometry`
2. Scene id: `solid_cross_section`
3. Task id: `task_geometry__solid_cross_section__cone_parallel_slice_area`
4. Supported `query_id` values: `single`
5. Answer schema: `number`
6. Answer precision: `one_decimal`
7. Annotation schema: `bbox`
8. Scalar annotation checked: `true`

## Program Contract
- `solve_formula(visible_solid_cross_section_measurements, unknown_role=area_measure, formula_schema=cone_parallel_slice_area); scene=solid_cross_section; scope=cone_parallel_slice_area`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `src/trace_tasks/resources/prompts/geometry/solid_cross_section/geometry_solid_cross_section_v1.json`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is a scalar pixel-space `bbox` around the marked cross-section. Numeric measurement labels are visible readouts used for solving, but they are not annotation targets. Formula values, scale factors, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/solid_cross_section.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/geometry/solid_cross_section/geometry_solid_cross_section_v1.json`
- Task module: `src/trace_tasks/tasks/geometry/solid_cross_section/cone_parallel_slice_area.py`
