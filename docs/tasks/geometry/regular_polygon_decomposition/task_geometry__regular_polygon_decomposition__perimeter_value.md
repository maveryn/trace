# `task_geometry__regular_polygon_decomposition__perimeter_value`

## Contract
1. Domain: `geometry`
2. Scene id: `regular_polygon_decomposition`
3. Task id: `task_geometry__regular_polygon_decomposition__perimeter_value`
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Annotation schema: `point_map`
7. Scalar annotation checked: `true` (not scalar-eligible; the task binds polygon center and apothem-foot roles)

## Program Contract
- `solve_formula(regular_polygon_equal_wedge_decomposition, target=regular_polygon_perimeter, formula_schema=two_area_divided_by_apothem); scene=regular_polygon_decomposition; scope=perimeter_value`

## Reasoning Operations

Families: `filtering`, `formula_evaluation`

## Query Semantics
- `single` asks for regular-polygon perimeter from visible total-area and apothem labels.
- The semantic prompt branch uses `perimeter_from_total_area_and_apothem`; the public query id remains `single`.
- The number of polygon sides, selected side, style, font, layout jitter, and rotation are internal replay metadata.

## Prompt Bundle
- Prompt text is loaded from `src/trace_tasks/resources/prompts/geometry/regular_polygon_decomposition/geometry_regular_polygon_decomposition_v1.json`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space point map keys `O` and `M`. These mark the polygon center and the apothem foot. Measurement labels and readout panels remain visible diagram content and private verifier metadata. This task does not shade a wedge region or label side endpoints; the decomposition lines remain visible only to support the area-apothem perimeter relation.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/regular_polygon_decomposition.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/geometry/regular_polygon_decomposition/geometry_regular_polygon_decomposition_v1.json`
- Task module: `src/trace_tasks/tasks/geometry/regular_polygon_decomposition/perimeter_value.py`
