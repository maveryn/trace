# `task_three_d__room__wall_object_same_wall_reference_label`

## Summary
- Domain: `three_d`
- Scene id: `room`
- Scene: `room`
- Supported `query_id`: `single`
- Answer type: `option_letter`
- Annotation schema: `bbox`

## Contract
The image shows a synthetic perspective 3D indoor room with a floor, back/side walls, furniture, floor context objects, one named wall reference object, wall-mounted candidate objects, and a below-scene text option panel. The prompt asks which option describes the wall-mounted object on the same wall as the named reference object.

Each instance renders exactly `6` wall-mounted answer candidates across the left, back, and right walls. These six candidates correspond to the MCQ options. Exactly one candidate is on the reference object's wall. Candidate object types exclude the sampled reference object type, and generation requires the reference prompt name to appear exactly once in finalized scene metadata.

The reference object is sampled from recognizable wall-mounted categories such as TV, clock, mirror, fan, air conditioner, and coat. Floor objects provide room context but are excluded from answer options.

The renderer uses a lower interior camera, extends the open/front floor toward the camera, keeps side-wall continuation capped to avoid cutaway wall panels, and includes foreground floor context so the scene reads from inside the room. Reference/candidate placement, wall assignments, same-wall flags, and verifier geometry still use the semantic room coordinates recorded in trace metadata.

## Program Contract

Program: `select(label(candidate_wall_objects, wall_id == reference_wall_object.wall_id)); scene=room; scope=wall_object_same_wall_reference_label`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `wall_object_same_wall_reference_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `candidate_wall_objects`, `wall_id`, `reference_wall_object`, `room`, `wall_object_same_wall_reference_label`.
Operation: evaluate `select` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `spatial_relations`

## Annotation Contract
Annotation is the bounding box of the selected wall-mounted object in the room scene. The option panel, option text, and named reference object are not annotation.

## Prompt And Trace
The prompt bundle is `three_d_room_v1` under `src/trace_tasks/resources/prompts/three_d/room/`. The trace records camera pose, projection frame, room scene variant, render-only floor front (`render_front_y`), render-only side-wall front (`render_side_wall_front_y`), bounded semantic room front (`semantic_front_y`), reference object id/type/name/wall, per-label same-wall flags, candidate wall assignments, selected object id/type/wall, projected object bboxes, and option-panel descriptors/bboxes.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config defaults, prompt bundle, and code versions. Answers and annotation come from the same finalized 3D room scene trace.
