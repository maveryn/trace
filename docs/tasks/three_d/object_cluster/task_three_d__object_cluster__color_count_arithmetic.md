# `task_three_d__object_cluster__color_count_arithmetic`

## Summary
- Domain: `three_d`
- Scene id: `object_cluster`
- Query ids: `total_count`, `difference_count`
- Answer type: `integer`
- Annotation type: `bbox_set_map`

## Program Contract

Program: `arithmetic(count(filter(object_cluster_objects, color_name=left_color_name)), count(filter(object_cluster_objects, color_name=right_color_name)), operator=query_id); scene=object_cluster; scope=color_count_arithmetic`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `color_count_arithmetic` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `object_cluster_objects`, `color_name`, `left_color_name`, `right_color_name`, `operator`, `query_id`, `object_cluster`, `color_count_arithmetic` plus the active `query_id` branch.
Operation: evaluate `arithmetic` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `total_count`, `difference_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `formula_evaluation`

## Contract
The image shows many small 3D objects arranged in one cluster. The prompt names two canonical semantic colors and asks for either the combined count or the absolute difference between their counts. Color names are sampled operands inside this one task; the visual reasoning channel is fixed to color.

## Annotation Contract
Annotation is a `bbox_set_map` with keys `left_operand` and `right_operand`. Each value is an array of `[x0, y0, x1, y1]` pixel boxes around the objects in that operand group.

## Prompt And Trace
The prompt bundle is `three_d_object_cluster_v1`. The trace records `query_id`, `internal_query_id`, operand color groups, object specs, projected boxes, camera, and solver count metadata.
