# `task_geometry__graph_paper__shape_type_count`

## Contract
1. Domain: `geometry`
2. Scene id: `graph_paper`
3. Task id: `task_geometry__graph_paper__shape_type_count`
4. Supported `query_id`: `triangle_count`, `quadrilateral_count`, `pentagon_count`, `hexagon_count`, `circle_count`, `ellipse_count`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`

## Program Contract
- `count_shape_class(query_id={triangle_count|quadrilateral_count|pentagon_count|hexagon_count|circle_count|ellipse_count}, target_class={triangle|quadrilateral|pentagon|hexagon|circle|ellipse}, output_role=count); scene=graph_paper; scope=mixed_shape_set`
- The `ellipse_count` prompt-facing target is `non-circular ellipses`; circles are counted only for `circle_count`.
- Shape families use class-preserving graph-paper variation: polygon vertices sit on graph intersections, circles and ellipses use integer centers/radii so their diameter or major/minor-axis witnesses align to graph points, and pentagons/hexagons are not always regular.
- Mixed-shape triangle variants avoid exact equilateral triangles so every polygonal shape in this task can keep lattice vertices.
- Count objects are placed using their actual graph-unit bounds so independent shapes do not overlap.

## Reasoning Operations

Families: `filtering`, `counting`

## Prompt Bundle
- Prompt text is loaded from `geometry_graph_paper_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
The annotation is the unordered set of bounding boxes for every matching shape.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.
