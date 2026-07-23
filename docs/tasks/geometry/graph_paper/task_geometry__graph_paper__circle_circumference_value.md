# `task_geometry__graph_paper__circle_circumference_value`

## Contract
1. Domain: `geometry`
2. Scene id: `graph_paper`
3. Task id: `task_geometry__graph_paper__circle_circumference_value`
4. Supported `query_id`: `single`
5. Answer schema: `string`
6. Annotation schema: `bbox`

## Program Contract
- `compute_circle_circumference_from_grid_radius(target=circle, output_role=exact_pi_expression); scene=graph_paper; scope=single_circle`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_graph_paper_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
The annotation is the circle bounding box in pixel coordinates.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.
