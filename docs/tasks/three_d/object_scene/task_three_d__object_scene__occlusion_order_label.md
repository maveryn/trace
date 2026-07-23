# `task_three_d__object_scene__occlusion_order_label`

## Summary
- Domain: `three_d`
- Scene id: `object_scene`
- Package: `src/trace_tasks/tasks/three_d/object_scene/`
- Supported `query_id`: `single`
- Answer type: `option_letter`
- Annotation type: `bbox`
- Annotation schema: `bbox`

## Program Contract

Program: `select(label(candidate_objects, visibly_occludes(reference_object))); scene=object_scene; scope=occlusion_order_label`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `occlusion_order_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `candidate_objects`, `visibly_occludes`, `reference_object`, `object_scene`, `occlusion_order_label`.
Operation: evaluate `select` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`

## Contract
The image uses the `object_scene` renderer: a perspective 3D floor, table, or platform scene with projected objects, markers, references, or paired views depending on the task. The public task id defines the stable objective contract; query ids are used only for genuine semantic operations within that contract. Render style, camera, canvas preset, object placement, labels, colors, and prompt wording variants are generation metadata, not public task axes.

The verifier computes the answer from finalized scene metadata and projection records, not from pixels. The prompt bundle is `three_d_object_scene_v1` under `src/trace_tasks/resources/prompts/three_d/object_scene/`.

## Annotation Contract
Annotation is a scalar `bbox` around the selected visible object.
The selected object is the only visual witness; option text is not annotation.

The reference object is a solid rectangular platform so the requested visual
witness is visible occlusion, not an inferred camera-depth comparison. The
answer object is placed just in front of the camera-facing platform face with a
small world-space gap, then accepted only if its projected bbox visibly covers
part of the platform and no other candidate does. Candidate acceptance uses
finalized projection/depth metadata to guarantee that exactly one candidate
visibly blocks part of the reference.

Candidate objects use a task-local occlusion-safe pool with enough projected
height or thickness to read as blockers. Thin or floor-flat objects from the
broader object-scene named pool are excluded from this task.

## Prompt And Trace
The trace records selected prompt keys, camera/projection data, object or marker records, rendered pixel witnesses, answer-support metadata, and the solver fields needed to recompute the answer and annotation from the same finalized scene.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config defaults, prompt bundle, and code versions. Answers and annotation come from the same finalized 3D scene trace.
