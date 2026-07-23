# `task_three_d__carousel__color_ordered_adjacent_pair_count`

## Summary
- Domain: `three_d`
- Scene id: `carousel`
- Query ids: `single`
- Answer type: `integer`
- Annotation type: `segment_set`

## Program Contract

Program: `count(ordered_adjacent_pairs(filter(carousel_objects, belt_key=target_belt_key), first_color_name=left_color_name, second_color_name=right_color_name)); scene=carousel; scope=color_ordered_adjacent_pair_count`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `color_ordered_adjacent_pair_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `ordered_adjacent_pairs`, `filter`, `carousel_objects`, `belt_key`, `target_belt_key`, `first_color_name`, `left_color_name`, `second_color_name`, `right_color_name`, `carousel`, `color_ordered_adjacent_pair_count`.
Operation: evaluate `count` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `segment_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## Annotation Contract
Annotation is an array of segments `[[x0, y0], [x1, y1]]`, one per counted ordered pair, from the first object center to the second object center.

## Prompt And Trace
The prompt bundle is `three_d_carousel_v1`. Trace metadata records `query_id="single"` and `internal_query_id="color_ordered_pair_count"`.
