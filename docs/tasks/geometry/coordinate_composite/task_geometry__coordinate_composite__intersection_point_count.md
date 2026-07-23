# `task_geometry__coordinate_composite__intersection_point_count`

## Contract
1. Domain: `geometry`
2. Scene id: `coordinate_composite`
5. Query id: `line_circle_intersection_count`, `circle_circle_intersection_count`, `line_polygon_intersection_count`, `circle_polygon_intersection_count`, or `mixed_object_intersection_count`
6. Answer schema: `integer_count`
7. Annotation schema: `point_set`

## Program Contract
- `count(intersection_points(filter_pairs(objects, pair_filter))); scene=coordinate_composite; scope=intersection_point_count`

## Reasoning Operations

Families: `counting`, `spatial_relations`

## Prompt Bundle
- Prompt text is loaded from `geometry_coordinate_composite_v0`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space witnesses only. Annotation is the unordered set of pixel points where drawn objects visibly intersect. When the answer is zero, annotation is an empty array. Object outlines, colors, grid coordinates, and symbolic object definitions remain visible annotations plus private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/coordinate_composite.yaml`
- Task module: `src/trace_tasks/tasks/geometry/coordinate_composite/intersection_point_count.py`
