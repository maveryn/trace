# `task_three_d__object_scene__reference_nearest_label`

## Summary
- Domain: `three_d`
- Scene id: `object_scene`
- Package: `src/trace_tasks/tasks/three_d/object_scene/`
- Supported `query_id`: `closest_to_reference`, `farthest_from_reference`
- Answer type: `option_letter`
- Annotation type: `bbox`
- Annotation schema: `bbox`

## Program Contract

Program: `select(label(candidate_objects, extremum(projected_screen_center_distance_to_reference))); scene=object_scene; scope=reference_nearest_label`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `reference_nearest_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `candidate_objects`, `extremum`, `projected_screen_center_distance_to_reference`, `object_scene`, `reference_nearest_label` plus the active `query_id` branch.
Operation: evaluate `select` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; intentionally based on projected visual distance because this is what.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `closest_to_reference`, `farthest_from_reference`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`

## Contract
The image uses the `object_scene` renderer: a perspective 3D floor, table, or platform scene with projected objects, markers, references, or paired views depending on the task. The public task id defines the stable objective contract; query ids are used only for genuine semantic operations within that contract. Render style, camera, canvas preset, object placement, labels, colors, and prompt wording variants are generation metadata, not public task axes.

The verifier computes the answer from finalized scene metadata and projection records, not from pixels. The prompt bundle is `three_d_object_scene_v1` under `src/trace_tasks/resources/prompts/three_d/object_scene/`.

Each scene contains exactly one large named reference prop and six small named
candidate objects. Candidate objects are placed in front of or beside the
reference from the camera view, with render-time rejection if a candidate is
hidden behind the reference prop. Query ids choose closest vs farthest by
projected screen-space center distance to the reference.

The trace also records 3D surface-gap ordering as diagnostic metadata, but the
answer is intentionally based on projected visual distance because this is what
the rendered question asks the model to judge. The selected answer must have a
clear screen-space margin. The task excludes `heart`, `sword`, and
`remote_control` candidates because their silhouettes made the visual relation
ambiguous during generation checks.

## Annotation Contract
Annotation is a scalar `bbox` around the selected visible object.
The selected object is the only visual witness; option text is not annotation.

## Prompt And Trace
The trace records selected prompt keys, camera/projection data, object or marker records, rendered pixel witnesses, answer-support metadata, and the solver fields needed to recompute the answer and annotation from the same finalized scene.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config defaults, prompt bundle, and code versions. Answers and annotation come from the same finalized 3D scene trace.
