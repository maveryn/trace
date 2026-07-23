# `task_geometry__regular_polygon_decomposition__central_angle_value`

## Contract
1. Domain: `geometry`
2. Scene id: `regular_polygon_decomposition`
3. Task id: `task_geometry__regular_polygon_decomposition__central_angle_value`
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Annotation schema: `point_map`
7. Scalar annotation checked: `true` (not scalar-eligible; the task binds center and two angle-ray endpoint roles)

## Program Contract
- `solve_formula(regular_polygon_equal_wedge_decomposition, target=marked_central_angle, formula_schema=marked_wedge_count_times_360_degrees_divided_by_side_count); scene=regular_polygon_decomposition; scope=central_angle_value`

## Reasoning Operations

Families: `counting`, `formula_evaluation`

## Query Semantics
- `single` asks for the measure of marked central angle `AOB` by counting the equal center wedges in the polygon and the one or two marked wedges between `OA` and `OB`.
- The number of polygon sides, selected wedge start, marked wedge count, style, font, layout jitter, and rotation are internal replay metadata; central-angle samples use one or two marked wedges.

## Prompt Bundle
- Prompt text is loaded from `src/trace_tasks/resources/prompts/geometry/regular_polygon_decomposition/geometry_regular_polygon_decomposition_v1.json`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space point map keys `O`, `A`, and `B`, marking the polygon center and the two visible rays that bound the requested center angle. Angle arcs and `?` markers remain visible diagram content and private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/regular_polygon_decomposition.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/geometry/regular_polygon_decomposition/geometry_regular_polygon_decomposition_v1.json`
- Task module: `src/trace_tasks/tasks/geometry/regular_polygon_decomposition/central_angle_value.py`
