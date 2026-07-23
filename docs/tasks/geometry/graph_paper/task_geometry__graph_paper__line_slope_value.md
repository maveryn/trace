# `task_geometry__graph_paper__line_slope_value`

## Contract
1. Domain: `geometry`
2. Scene id: `graph_paper`
3. Task id: `task_geometry__graph_paper__line_slope_value`
4. Supported `query_id`: `single`
5. Answer schema: `number`
6. Answer precision: `one_decimal`
7. Annotation schema: `segment`

## Program Contract
- `compute_segment_slope_from_grid_rise_run(target=line_segment, output_role=slope_number); scene=graph_paper; scope=single_segment`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_graph_paper_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
The annotation is the selected visual line segment as two pixel endpoints.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.
