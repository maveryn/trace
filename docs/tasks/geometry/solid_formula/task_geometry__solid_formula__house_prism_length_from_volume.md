# `task_geometry__solid_formula__house_prism_length_from_volume`

## Contract
1. Domain: `geometry`
2. Scene id: `solid_formula`
3. Task id: `task_geometry__solid_formula__house_prism_length_from_volume`
4. Supported `query_id` values: `single`
5. Answer schema: `number`
6. Answer precision: `one_decimal`
7. Annotation schema: `bbox`
8. Scalar annotation checked: `true` (one bbox around the visible compound solid shape)

## Program Contract
- `solve_formula(visible_solid_formula_measurements, unknown_role=prism_length, formula_schema=house_prism_length_from_volume); scene=solid_formula; scope=house_prism_length_from_volume`
- The image shows a house-shaped prism with rectangular wall and triangular roof cross-section. The prism length is unknown, while base width, wall height, roof height, and volume are labeled.
- Annotation witness: one bbox around the house-prism solid.

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `src/trace_tasks/resources/prompts/geometry/solid_formula/geometry_solid_formula_v1.json`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses one pixel-space bbox around the visible compound solid shape. Numeric readout text, dimension labels, and dimension-line witnesses remain visible in the image and recorded in trace metadata, but they are not public annotation witnesses. Graph coordinates, formulas, labels, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/solid_formula.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/geometry/solid_formula/geometry_solid_formula_v1.json`
- Task module: `src/trace_tasks/tasks/geometry/solid_formula/house_prism_length_from_volume.py`
