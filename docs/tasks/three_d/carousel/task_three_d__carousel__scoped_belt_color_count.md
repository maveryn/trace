# `task_three_d__carousel__scoped_belt_color_count`

## Summary
- Domain: `three_d`
- Scene id: `carousel`
- Query ids: `single`
- Answer type: `integer`
- Annotation type: `bbox_set`

## Program Contract

Program: `count(filter(carousel_objects, belt_key=target_belt_key, color_name=target_color_name)); scene=carousel; scope=scoped_belt_color_count`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `scoped_belt_color_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `carousel_objects`, `belt_key`, `target_belt_key`, `color_name`, `target_color_name`, `carousel`, `scoped_belt_color_count`.
Operation: evaluate `count` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Contract
The scene has inner and outer elliptical conveyor belts. The prompt selects one belt and one canonical named color. Target color is a sampled operand, not a query id.

## Annotation Contract
Annotation is an unordered array of `[x0, y0, x1, y1]` pixel boxes around counted colored objects on the requested belt. Empty arrays are valid for answer `0`.

## Prompt And Trace
The prompt bundle is `three_d_carousel_v1`. Trace metadata records `query_id="single"` and `internal_query_id="color_belt_count"`.
