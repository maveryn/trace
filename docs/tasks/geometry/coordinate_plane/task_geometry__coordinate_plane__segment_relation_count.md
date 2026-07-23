# `task_geometry__coordinate_plane__segment_relation_count`

## Contract
1. Domain: `geometry`
2. Scene id: `coordinate_plane`
5. Query id: `parallel_count`, `perpendicular_count`
6. Answer schema: `integer_count`
7. Annotation schema: `segment_set`

## Program Contract
- `count(filter(coordinate_plane_segment_pairs, segment_relation(pair)=target_segment_relation)); scene=coordinate_plane; scope=segment_relation_count`

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `coordinate_plane`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is the unordered `segment_set` of counted non-AB segments. Each witness segment is `[[x0,y0],[x1,y1]]` in pixel coordinates; endpoint order is not semantically meaningful. If the answer is `0`, annotation is an empty array. Graph coordinates, labels, and relation metadata remain private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/coordinate_plane.yaml`
- Task module: `src/trace_tasks/tasks/geometry/coordinate_plane/segment_relation_count.py`
