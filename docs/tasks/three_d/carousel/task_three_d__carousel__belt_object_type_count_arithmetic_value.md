# `task_three_d__carousel__belt_object_type_count_arithmetic_value`

## Summary
- Domain: `three_d`
- Scene id: `carousel`
- Query ids: `total_count`, `difference_count`
- Answer type: `integer`
- Annotation type: `bbox_set_map`

## Program Contract

Program: `arithmetic(count(filter(carousel_objects, belt_key=left_belt_key, shape_type=target_shape_type)), count(filter(carousel_objects, belt_key=right_belt_key, shape_type=target_shape_type)), operator=query_id); scene=carousel; scope=belt_object_type_count_arithmetic_value`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `belt_object_type_count_arithmetic_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `carousel_objects`, `belt_key`, `left_belt_key`, `shape_type`, `target_shape_type`, `right_belt_key`, `operator`, `query_id`, `carousel`, `belt_object_type_count_arithmetic_value` plus the active `query_id` branch.
Operation: evaluate `arithmetic` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `total_count`, `difference_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `formula_evaluation`

## Contract
The scene has inner and outer carousel belts. The prompt names one object type and asks for either the total or absolute difference across the two belts. Object type is a sampled operand; arithmetic operator is the query id.

## Annotation Contract
Annotation is a `bbox_set_map` with keys `inner_objects` and `outer_objects`, each containing `[x0, y0, x1, y1]` boxes around counted objects for that belt.

## Prompt And Trace
The prompt bundle is `three_d_carousel_v1`. Trace metadata records `internal_query_id` as `object_count_sum` or `object_count_difference`.
