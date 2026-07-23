# `task_three_d__conveyor__between_object_type_anchors_count`

## Summary
- Domain: `three_d`
- Scene id: `conveyor`
- Query ids: `single`
- Answer type: `integer`
- Annotation type: `bbox_set`

## Program Contract

Program: `count(objects_between(anchor_a, anchor_b, filter(conveyor_objects, lane_key=target_lane_key))); scene=conveyor; scope=between_object_type_anchors_count`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `between_object_type_anchors_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `objects_between`, `anchor_a`, `anchor_b`, `filter`, `conveyor_objects`, `lane_key`, `target_lane_key`, `conveyor`, `between_object_type_anchors_count`.
Operation: evaluate `count` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## Annotation Contract
Annotation is an unordered array of `[x0, y0, x1, y1]` pixel boxes around the counted objects between the marked anchors. Anchor objects themselves are not annotation.

## Prompt And Trace
The prompt bundle is `three_d_conveyor_v1`. Trace metadata records `query_id="single"` and `internal_query_id="between_object_anchors_count"`.
