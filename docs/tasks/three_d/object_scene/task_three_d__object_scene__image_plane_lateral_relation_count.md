# `task_three_d__object_scene__image_plane_lateral_relation_count`

## Summary
- Domain: `three_d`
- Scene id: `object_scene`
- Package: `src/trace_tasks/tasks/three_d/object_scene/`
- Supported `query_id`: `left_of_reference_in_view_count`, `right_of_reference_in_view_count`
- Answer type: `integer`
- Annotation type: `bbox_set`
- Annotation schema: `bbox_set`

## Program Contract

Program: `count(filter(candidate_objects, rendered_bbox_side_of_reference = requested_side)); scene=object_scene; scope=image_plane_lateral_relation_count`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `image_plane_lateral_relation_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `candidate_objects`, `rendered_bbox_side_of_reference`, `requested_side`, `object_scene`, `image_plane_lateral_relation_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `left_of_reference_in_view_count`, `right_of_reference_in_view_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Contract
The image uses the `object_scene` renderer: a perspective 3D floor, table, or platform scene with projected objects, markers, references, or paired views depending on the task. The public task id defines the stable objective contract; query ids are used only for genuine semantic operations within that contract. Render style, camera, canvas preset, object placement, labels, colors, and prompt wording variants are generation metadata, not public task axes.

The verifier computes the answer from finalized scene metadata and projection records, not from pixels. The prompt bundle is `three_d_object_scene_v1` under `src/trace_tasks/resources/prompts/three_d/object_scene/`.

The lateral relation layout uses nine objects spread along the camera's screen-right floor axis so every non-reference object is visually on one side of the red-boxed reference object. The supported answer count range is `1..6`.

## Annotation Contract
Annotation is an unordered `bbox_set` containing one box around each counted object.
All witnesses have the same counted-object role, so ordering is not meaningful.
The red reference box identifies the comparison object but is not part of the annotation.
For left/right membership, the counted object's rendered bbox must be fully on the requested side of the red-boxed reference object's rendered bbox with a minimum horizontal gap.

## Prompt And Trace
The trace records selected prompt keys, camera/projection data, object or marker records, rendered pixel witnesses, answer-support metadata, and the solver fields needed to recompute the answer and annotation from the same finalized scene.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config defaults, prompt bundle, and code versions. Answers and annotation come from the same finalized 3D scene trace.
