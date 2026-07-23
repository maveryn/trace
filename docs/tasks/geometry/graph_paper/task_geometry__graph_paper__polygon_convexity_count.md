# `task_geometry__graph_paper__polygon_convexity_count`

## Contract
1. Domain: `geometry`
2. Scene id: `graph_paper`
3. Task id: `task_geometry__graph_paper__polygon_convexity_count`
4. Supported `query_id`: `convex_polygon_count`, `concave_polygon_count`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`

## Program Contract
- `count_polygon_convexity_class(query_id={convex_polygon_count|concave_polygon_count}, target_class={convex|concave}, output_role=count); scene=graph_paper; scope=polygon_set`
- Convex and concave polygons use integer graph-paper vertices and vary their side count from 5 to 7. Convex polygons are not required to be regular, and concave polygons use a visible inward notch.
- Count objects are placed using their actual graph-unit bounds so independent polygons do not overlap.

## Reasoning Operations

Families: `counting`

## Prompt Bundle
- Prompt text is loaded from `geometry_graph_paper_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
The annotation is the unordered set of bounding boxes for every matching polygon.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.
