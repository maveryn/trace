# `task_three_d__conveyor__belt_total_object_count`

## Summary
- Domain: `three_d`
- Scene id: `conveyor`
- Scene package: `conveyor`
- Query ids: `single` publicly; internal query id `belt_total_count`
- Answer type: `integer`
- Annotation type: unordered `bbox_set`
- Annotation schema: `bbox_set`

## Program Contract

Program: `count(filter(conveyor_objects, lane_key=target_lane_key)); scene=conveyor; scope=belt_total_object_count`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `belt_total_object_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `conveyor_objects`, `lane_key`, `target_lane_key`, `conveyor`, `belt_total_object_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`, `belt_total_count`.

## Reasoning Operations

Families: `counting`

## Contract
The image shows one 3D straight-conveyor scene with three visible parallel
belts. Landscape canvases use three horizontal lanes labeled by prompt position
as `TOP`, `MIDDLE`, or `BOTTOM`. Portrait canvases use three vertical lanes
labeled by prompt position as `LEFT`, `MIDDLE`, or `RIGHT`. Square canvases may
use either orientation. The lane positions are not written as text on the image.

Each generated instance uses one sampled object type across all lanes. Object
colors may vary, but color is not part of the task predicate. The task asks for
the total number of visible objects on the requested belt/lane.

The target lane is sampled as a generation axis. Every lane count, including
the target lane, has support `1..8`.

## Annotation Contract
Annotation is a `bbox_set` containing one `[x0, y0, x1, y1]` pixel box around
each counted object on the requested lane. Objects on other lanes and belt
surfaces are not annotation.

## Prompt And Trace
The prompt bundle is `three_d_conveyor_v1` under
`src/trace_tasks/resources/prompts/three_d/conveyor/`. The prompt asks for total objects on the requested
positioned belt and does not mention the sampled object type or colors.

The trace records scene variant, layout orientation, lane records, target lane,
sampled object type, object specs, projected object boxes and centers, camera,
projection frame, and solver count predicate.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config
defaults, prompt bundle, and code versions. Answers and annotation come from the
same finalized straight conveyor trace.
