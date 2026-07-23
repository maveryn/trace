# `task_three_d__carousel__belt_total_object_count`

## Summary
- Domain: `three_d`
- Scene id: `carousel`
- Scene package: `carousel`
- Query ids: `single` publicly; internal query id `belt_total_count`
- Answer type: `integer`
- Annotation type: unordered `bbox_set`
- Annotation schema: `bbox_set`

## Program Contract

Program: `count(filter(conveyor_objects, belt_key=target_belt_key)); scene=carousel; scope=belt_total_object_count`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `belt_total_object_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `conveyor_objects`, `belt_key`, `target_belt_key`, `carousel`, `belt_total_object_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`, `belt_total_count`.

## Reasoning Operations

Families: `counting`

## Contract
The image shows one 3D conveyor carousel with two visible concentric
elliptical belts: an inner belt and an outer belt. The belts are distinguished
by position, not by text written on the image. Small 3D objects sit on the belt
surfaces.

Each generated instance uses one sampled object type across both belts. Object
colors may vary, but color is not part of the task predicate. The task asks for
the total number of visible objects on the requested belt.

The target belt is sampled as a generation axis. If the target belt is `inner`,
the answer support is `1..8`. If the target belt is `outer`, the answer support
is `1..10`.

## Annotation Contract
Annotation is a `bbox_set` containing one `[x0, y0, x1, y1]` pixel box around
each counted object on the requested belt. Objects on the other belt, belt
surfaces, arrows, and decorative station context are not annotation.

## Prompt And Trace
The prompt bundle is `three_d_carousel_v1` under
`src/trace_tasks/resources/prompts/three_d/carousel/`. The prompt asks for total objects on the
`INNER` or `OUTER` belt and does not mention the sampled object type or colors.

The trace records scene variant, belt records, target belt, sampled object type,
object specs, projected object boxes and centers, camera, projection frame, and
solver count predicate.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config
defaults, prompt bundle, and code versions. Answers and annotation come from the
same finalized conveyor trace.
