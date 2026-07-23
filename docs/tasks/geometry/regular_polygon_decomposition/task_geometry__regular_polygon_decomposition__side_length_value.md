# `task_geometry__regular_polygon_decomposition__side_length_value`

## Contract
1. Domain: `geometry`
2. Scene id: `regular_polygon_decomposition`
3. Task id: `task_geometry__regular_polygon_decomposition__side_length_value`
4. Supported `query_id` values: `side_length_from_perimeter`, `side_length_from_total_area_and_apothem`, `side_length_from_wedge_area_and_apothem`
5. Answer schema: `integer`
6. Annotation schema: `point_map`
7. Scalar annotation checked: `true` (not scalar-eligible; the task binds polygon center, target side endpoints, and sometimes apothem-foot or wedge-region roles)

## Program Contract
- `solve_formula(regular_polygon_equal_wedge_decomposition, target=regular_polygon_side_length, formula_schema=perimeter_divided_by_side_count_or_area_apothem_relation); scene=regular_polygon_decomposition; scope=side_length_value`

## Reasoning Operations

Families: `counting`, `formula_evaluation`

## Query Semantics
- `side_length_from_perimeter` asks for one regular-polygon side length from the visible total perimeter.
- `side_length_from_total_area_and_apothem` asks for one regular-polygon side length from total area and apothem labels.
- `side_length_from_wedge_area_and_apothem` asks for one regular-polygon side length from shaded wedge area and apothem labels.
- The number of polygon sides, selected side, style, font, layout jitter, and rotation are internal replay metadata.

## Prompt Bundle
- Prompt text is loaded from `src/trace_tasks/resources/prompts/geometry/regular_polygon_decomposition/geometry_regular_polygon_decomposition_v1.json`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space point map keys `O`, `A`, `B`, and, when needed, `M` and `W`. These mark the polygon center, target side endpoints, apothem foot, and target wedge-region point. Measurement labels and readout panels remain visible diagram content and private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/regular_polygon_decomposition.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/geometry/regular_polygon_decomposition/geometry_regular_polygon_decomposition_v1.json`
- Task module: `src/trace_tasks/tasks/geometry/regular_polygon_decomposition/side_length_value.py`
