# `task_geometry__regular_polygon_decomposition__marked_piece_area_value`

## Contract
1. Domain: `geometry`
2. Scene id: `regular_polygon_decomposition`
3. Task id: `task_geometry__regular_polygon_decomposition__marked_piece_area_value`
4. Supported `query_id` values: `single`
5. Answer schema: `number`
6. Answer precision: `one_decimal`
7. Annotation schema: `point_map`
8. Scalar annotation checked: `true` (not scalar-eligible; the task binds center and target wedge-boundary vertex roles)

## Program Contract
- `solve_formula(regular_polygon_equal_wedge_decomposition, target=marked_wedge_group_area, formula_schema=total_area_divided_by_side_count_times_marked_wedge_count); scene=regular_polygon_decomposition; scope=marked_piece_area_value`

## Reasoning Operations

Families: `filtering`, `counting`, `formula_evaluation`

## Query Semantics
- `single` asks for the area of the shaded wedge or adjacent shaded wedge group from the total polygon area by counting all equal center wedges and the shaded wedges.
- The semantic prompt branch uses `marked_wedges_area_from_total`; the public query id remains `single`.
- The number of polygon sides, selected wedge start, marked wedge count, style, font, layout jitter, and rotation are internal replay metadata.

## Prompt Bundle
- Prompt text is loaded from `src/trace_tasks/resources/prompts/geometry/regular_polygon_decomposition/geometry_regular_polygon_decomposition_v1.json`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space point map keys `O`, `A`, and `B`, marking the polygon center and the two visible rays that bound the requested shaded wedge or wedge group. Measurement labels and readout panels remain visible diagram content and private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/regular_polygon_decomposition.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/geometry/regular_polygon_decomposition/geometry_regular_polygon_decomposition_v1.json`
- Task module: `src/trace_tasks/tasks/geometry/regular_polygon_decomposition/marked_piece_area_value.py`
