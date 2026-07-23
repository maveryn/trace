# `task_three_d__room__wall_object_side_relation_label`

## Summary
- Domain: `three_d`
- Scene id: `room`
- Scene: `room`
- Supported `query_id` values: `left_of_reference_on_wall`, `right_of_reference_on_wall`
- Answer type: `option_letter`
- Annotation schema: `bbox`

## Contract
The image shows a synthetic perspective 3D indoor room with a floor, back/side walls, furniture, floor context objects, one TV mounted on a wall, wall-mounted candidate objects across room walls, and a below-scene text option panel. The prompt asks which option describes the wall-mounted object to the left or right of the TV along the TV's wall plane.

Each instance renders exactly `6` wall-mounted answer candidates corresponding to the MCQ options. Exactly two candidates are mounted on the TV's wall: one on the requested side and one on the opposite side. The remaining candidates are mounted on other room walls to reduce same-wall clutter. Exactly one candidate satisfies the sampled side relation on the TV wall: greater wall-plane left coordinate for `left_of_reference_on_wall`, or smaller wall-plane left coordinate for `right_of_reference_on_wall`. Candidate object types exclude TVs so the named reference remains unique.

The TV reference may appear on the back, left, or right wall. The verifier uses the finalized wall coordinate system rather than raw image x-position, so side-wall examples can require reasoning about the room wall plane instead of screen-left.

The renderer uses a lower interior camera, extends the open/front floor toward the camera, keeps side-wall continuation capped to avoid cutaway wall panels, and includes foreground floor context so the scene reads from inside the room. Reference/candidate placement, wall coordinates, side-relation flags, and verifier geometry still use the semantic room coordinates recorded in trace metadata.

## Program Contract

Program: `select(label(candidate_wall_objects, wall_id == reference.wall_id and wall_plane_side(candidate, reference) == requested_side)); scene=room; scope=wall_object_side_relation_label`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `wall_object_side_relation_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `candidate_wall_objects`, `wall_id`, `reference`, `wall_plane_side`, `candidate`, `requested_side`, `room`, `wall_object_side_relation_label` plus the active `query_id` branch.
Operation: evaluate `select` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `left_of_reference_on_wall`, `right_of_reference_on_wall`.

## Reasoning Operations

Families: `logical_composition`, `spatial_relations`

## Annotation Contract
Annotation is the bounding box of the selected wall-mounted object in the room scene. The option panel, option text, and unlettered TV reference are not annotation.

## Prompt And Trace
The prompt bundle is `three_d_room_v1` under `src/trace_tasks/resources/prompts/three_d/room/`. The trace records camera pose, projection frame, room scene variant, render-only floor front (`render_front_y`), render-only side-wall front (`render_side_wall_front_y`), bounded semantic room front (`semantic_front_y`), TV reference id/type/name/wall, reference wall coordinate, per-label wall-plane left coordinates, per-label left/right side-relation flags, candidate wall assignments, selected object id/type/wall, projected/visible object bboxes, and option-panel descriptors/bboxes.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config defaults, prompt bundle, and code versions. Answers and annotation come from the same finalized 3D room scene trace.
