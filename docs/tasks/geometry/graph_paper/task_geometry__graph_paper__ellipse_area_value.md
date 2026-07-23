# `task_geometry__graph_paper__ellipse_area_value`

## Contract
1. Domain: `geometry`
2. Scene id: `graph_paper`
3. Task id: `task_geometry__graph_paper__ellipse_area_value`
4. Supported `query_id`: `single`
5. Answer schema: `string`
6. Annotation schema: `bbox`

## Program Contract
- `compute_ellipse_area_from_grid_radii(target=ellipse, output_role=exact_pi_expression); scene=graph_paper; scope=single_ellipse`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_graph_paper_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
The annotation is the ellipse bounding box in pixel coordinates.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.
