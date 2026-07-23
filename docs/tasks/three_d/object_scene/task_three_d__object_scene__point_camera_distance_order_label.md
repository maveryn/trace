# `task_three_d__object_scene__point_camera_distance_order_label`

## Summary
- Domain: `three_d`
- Scene id: `object_scene`
- Package: `src/trace_tasks/tasks/three_d/object_scene/`
- Supported `query_id`: `single`
- Answer type: `option_letter`
- Annotation type: `point_map`
- Annotation schema: `point_map`

## Program Contract

Program: `select(option_label(permutation(marked_points), order_by(camera_distance, near_to_far))); scene=object_scene; scope=point_camera_distance_order_label`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `point_camera_distance_order_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `option_label`, `permutation`, `marked_points`, `order_by`, `camera_distance`, `near_to_far`, `object_scene`, `point_camera_distance_order_label`.
Operation: evaluate `select` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`

## Contract
The image uses the `object_scene` renderer: a perspective 3D floor, table, or platform scene with context objects and exactly three marked floor points. The image includes a visual option panel with all six possible orders of the three point labels.

The verifier computes the true order from finalized 3D camera-distance metadata, not from pixels. Generation enforces visible point separation and a unique camera-distance order that is consistent with projected depth cues. Render style, camera, canvas preset, context objects, labels, colors, and prompt wording variants are generation metadata, not public task axes.

## Annotation Contract
Annotation is a `point_map` keyed by the visible point labels `P`, `Q`, and `R`; each value is the center of that marked point.
The option panel is not an annotation witness; it only maps the computed order to an answer label.

## Prompt And Trace
The trace records selected prompt keys, camera/projection data, point records, option descriptors, rendered pixel witnesses, answer-support metadata, and solver fields needed to recompute the answer and annotation from the same finalized scene.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config defaults, prompt bundle, and code versions. Answer and annotation come from the same finalized 3D scene trace.
