# `task_geometry__regular_polygon_decomposition__wedge_area_from_side_apothem_value`

## Contract
1. Domain: `geometry`
2. Scene id: `regular_polygon_decomposition`
3. Task id: `task_geometry__regular_polygon_decomposition__wedge_area_from_side_apothem_value`
4. Supported `query_id` values: `single`
5. Answer schema: `number`
6. Answer precision: `one_decimal`
7. Annotation schema: `point_map`
8. Scalar annotation checked: `true` (not scalar-eligible; the task binds triangular wedge and apothem-foot roles)

## Program Contract
- `solve_formula(regular_polygon_equal_wedge_decomposition, target=single_wedge_area, formula_schema=side_length_times_apothem_divided_by_two); scene=regular_polygon_decomposition; scope=wedge_area_from_side_apothem_value`

## Reasoning Operations

Families: `formula_evaluation`

## Query Semantics
- `single` asks for the area of one regular-polygon triangular center wedge from its visible side-length and apothem labels.
- The semantic prompt branch uses `wedge_area_from_side_and_apothem`; the public query id remains `single`.
- The number of polygon sides, selected wedge start, style, font, layout jitter, and rotation are internal replay metadata.

## Prompt Bundle
- Prompt text is loaded from `src/trace_tasks/resources/prompts/geometry/regular_polygon_decomposition/geometry_regular_polygon_decomposition_v1.json`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space point map keys `O`, `A`, `B`, and `M`, marking the polygon center, the side endpoints of the target wedge base, and the apothem foot. Measurement labels and readout panels remain visible diagram content and private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/regular_polygon_decomposition.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/geometry/regular_polygon_decomposition/geometry_regular_polygon_decomposition_v1.json`
- Task module: `src/trace_tasks/tasks/geometry/regular_polygon_decomposition/wedge_area_from_side_apothem_value.py`
