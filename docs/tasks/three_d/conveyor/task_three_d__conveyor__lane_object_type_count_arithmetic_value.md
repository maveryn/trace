# `task_three_d__conveyor__lane_object_type_count_arithmetic_value`

## Summary
- Domain: `three_d`
- Scene id: `conveyor`
- Query ids: `total_count`, `difference_count`
- Answer type: `integer`
- Annotation type: `bbox_set_map`

## Program Contract

Program: `arithmetic(count(filter(conveyor_objects, lane_key=left_lane_key, shape_type=target_shape_type)), count(filter(conveyor_objects, lane_key=right_lane_key, shape_type=target_shape_type)), operator=query_id); scene=conveyor; scope=lane_object_type_count_arithmetic_value`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `lane_object_type_count_arithmetic_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `conveyor_objects`, `lane_key`, `left_lane_key`, `shape_type`, `target_shape_type`, `right_lane_key`, `operator`, `query_id`, `conveyor`, `lane_object_type_count_arithmetic_value` plus the active `query_id` branch.
Operation: evaluate `arithmetic` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `total_count`, `difference_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `formula_evaluation`

## Contract
The scene has three straight conveyor lanes. The prompt selects two lanes and one object type, then asks for either the total or absolute difference across the two lanes. Object type is a sampled operand; arithmetic operator is the query id.

## Annotation Contract
Annotation is a `bbox_set_map` keyed by selected lane roles, such as `top_objects` and `middle_objects`. Each value contains `[x0, y0, x1, y1]` boxes around counted objects for that lane.

## Prompt And Trace
The prompt bundle is `three_d_conveyor_v1`. Trace metadata records `internal_query_id` as `object_count_sum` or `object_count_difference`.
