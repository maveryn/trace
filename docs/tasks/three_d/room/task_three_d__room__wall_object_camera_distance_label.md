# `task_three_d__room__wall_object_camera_distance_label`

## Summary
- Domain: `three_d`
- Scene id: `room`
- Scene: `room`
- Supported `query_id`: `single`
- Answer type: `option_letter`
- Annotation schema: `bbox`

## Contract
The image shows a synthetic perspective 3D indoor room with a floor, back/side walls, furniture, floor context objects, wall-mounted candidate objects, and a below-scene text option panel. The prompt asks which option describes the wall-mounted object closest to the camera.

Each instance renders exactly `6` wall-mounted answer candidates spread across the left, back, and right walls. These are the only task-owned wall objects, and each corresponds to one MCQ option. Candidate object types come from TVs, clocks, picture frames, mirrors, wall fans, air conditioners, and hanging coats. Floor/furniture props provide room context but are excluded from the answer options.

The task uses a narrower front-oblique camera band and keeps side-wall candidates away from the open front edge of the room, so wall-mounted objects remain visibly hanging on the wall rather than collapsing into edge-on slivers. Generation rejects side-wall candidates whose projected wall face is too skinny.

The renderer uses a lower interior camera, extends the open/front floor toward the camera, keeps side-wall continuation capped to avoid cutaway wall panels, and includes foreground floor context so the scene reads from inside the room. Candidate placement, wall assignments, camera distances, and verifier geometry still use the semantic room coordinates recorded in trace metadata.

The answer is computed from finalized metadata using the minimum `camera_distance` among candidate wall-mounted objects. Generation also requires that the selected candidate is the front-most candidate by room depth with a visible depth margin, so lateral camera position cannot make a visually farther side-wall object win. The trace records the full near-to-far option-label order, front-to-back room-depth order, per-label camera distances, candidate walls, and nearest/depth margins.

## Program Contract

Program: `select(label(candidate_wall_objects, argmin(camera_distance))); scene=room; scope=wall_object_camera_distance_label`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `wall_object_camera_distance_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `candidate_wall_objects`, `argmin`, `camera_distance`, `room`, `wall_object_camera_distance_label`.
Operation: evaluate `select` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`

## Annotation Contract
Annotation is the bounding box of the selected wall-mounted object in the room scene. The option panel and option text are not annotation.

## Prompt And Trace
The prompt bundle is `three_d_room_v1` under `src/trace_tasks/resources/prompts/three_d/room/`. The trace records camera pose, projection frame, room scene variant, render-only floor front (`render_front_y`), render-only side-wall front (`render_side_wall_front_y`), bounded semantic room front (`semantic_front_y`), wall and floor object specs, per-label camera distances, candidate wall assignments, candidate projected bboxes, near-to-far order, room-depth order, selected object id/type/wall, projected object bboxes, and option-panel descriptors/bboxes.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config defaults, prompt bundle, and code versions. Answers and annotation come from the same finalized 3D room scene trace.
