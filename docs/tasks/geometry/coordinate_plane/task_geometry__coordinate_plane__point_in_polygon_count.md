# `task_geometry__coordinate_plane__point_in_polygon_count`

## Contract
1. Domain: `geometry`
2. Scene id: `coordinate_plane`
5. Query id: `single`
6. Answer schema: `integer_count`
7. Annotation schema: `point_set`

## Program Contract
- `count(filter(candidate_points, inside_polygon(point, target_polygon))); scene=coordinate_plane; scope=point_in_polygon_count`

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `coordinate_plane`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is the unordered `point_set` of candidate point centers inside the target polygon. If the answer is `0`, annotation is an empty array. Graph coordinates, labels, polygon definitions, and construction metadata remain private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/coordinate_plane.yaml`
- Task module: `src/trace_tasks/tasks/geometry/coordinate_plane/point_in_polygon_count.py`
