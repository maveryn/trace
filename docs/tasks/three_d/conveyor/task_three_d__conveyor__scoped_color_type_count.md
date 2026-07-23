# `task_three_d__conveyor__scoped_color_type_count`

## Summary
- Domain: `three_d`
- Scene id: `conveyor`
- Scene package: `conveyor`
- Query ids: `single`
- Answer type: `integer`
- Annotation type: unordered `bbox_set`
- Annotation schema: `bbox_set`

## Program Contract

Program: `count(filter(conveyor_objects, lane_key=target_lane_key, shape_type=target_shape_type, color_name=target_color_name)); scene=conveyor; scope=scoped_color_type_count`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `scoped_color_type_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `conveyor_objects`, `lane_key`, `target_lane_key`, `shape_type`, `target_shape_type`, `color_name`, `target_color_name`, `conveyor`, `scoped_color_type_count`.
Operation: evaluate `count` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`

## Contract
The image shows one 3D straight-conveyor scene with three visible parallel
belts. Landscape canvases use three horizontal lanes labeled by prompt position
as `TOP`, `MIDDLE`, or `BOTTOM`. Portrait canvases use three vertical lanes
labeled by prompt position as `LEFT`, `MIDDLE`, or `RIGHT`. Square canvases may
use either orientation. The lane positions are not written as text on the image.

The prompt asks for the number of objects on one requested belt that match both
a canonical named color and one sampled object type. The answer is the integer
count of finalized objects whose `lane_key`, `shape_type`, and `color_name` all
match the sampled target predicate. The answer support is `0..5`.

Color-readout generation avoids target-confusable named color distractors, such
as red with maroon or blue with cyan. Generation includes partial-match
distractors on the requested belt, including same-color wrong-type objects and
same-type wrong-color objects. Other lanes may contain objects with the target
color and target type, but those are outside the requested scope and do not
count. Wrong-type distractors avoid target-confusable object families such as
card/envelope/book, sphere/button, cup/bowl/tray, and lantern/candle. The
straight-conveyor object pool excludes long thin pen/ruler style objects that
do not fit lane grammar clearly.

## Annotation Contract
Annotation is a `bbox_set` containing one `[x0, y0, x1, y1]` pixel box around
each counted target object. Objects on other lanes, same-lane partial-match
distractor objects, and belt surfaces are not annotation. If the answer is `0`,
the annotation is an empty `bbox_set`.

## Prompt And Trace
The prompt bundle is `three_d_conveyor_v1` under
`src/trace_tasks/resources/prompts/three_d/conveyor/`. Named color prompts use the canonical repo-wide
color format, such as `blue [#2D75E6]`.

The trace records scene variant, layout orientation, lane records, target lane,
target object type, target color, object specs, projected object boxes and
centers, camera, projection frame, and solver count predicate.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config
defaults, prompt bundle, and code versions. Answers and annotation come from the
same finalized straight conveyor trace.
