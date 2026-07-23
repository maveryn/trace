# `task_three_d__carousel__color_transfer_total_count`

## Summary
- Domain: `three_d`
- Scene id: `carousel`
- Query ids: `single`
- Answer type: `integer`
- Annotation type: `bbox_set_map`

## Program Contract

Program: `count(filter(carousel_objects, belt_key=destination_belt_key)) + count(filter(carousel_objects, belt_key=source_belt_key, color_name=target_color_name)); scene=carousel; scope=color_transfer_total_count`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `color_transfer_total_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `carousel_objects`, `belt_key`, `destination_belt_key`, `source_belt_key`, `color_name`, `target_color_name`, `carousel`, `color_transfer_total_count`.
Operation: evaluate `count` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `state_update`, `formula_evaluation`

## Annotation Contract
Annotation is a `bbox_set_map` with keys `source_moved_objects` and `destination_existing_objects`, each containing `[x0, y0, x1, y1]` boxes before the move.

## Prompt And Trace
The prompt bundle is `three_d_carousel_v1`. Trace metadata records `query_id="single"` and `internal_query_id="color_transfer_total_count"`.
