# `task_geometry__graph_paper__right_angle_vertex_count`

## Contract
1. Domain: `geometry`
2. Scene id: `graph_paper`
3. Task id: `task_geometry__graph_paper__right_angle_vertex_count`
4. Supported `query_id`: `single`
5. Answer schema: `integer`
6. Annotation schema: `point_set`

## Program Contract
- `count_polygon_right_angle_vertices(target=single_polygon, output_role=count); scene=graph_paper; scope=single_lattice_polygon`
- The scene renders one polygon on graph paper with 6 to 12 vertices.
- The answer is the number of polygon vertices where the two adjacent sides form a right angle.
- The supported answer range is 1 to 5.

## Reasoning Operations

Families: `counting`, `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_graph_paper_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
The annotation is the unordered set of pixel points at every right-angle vertex.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.
