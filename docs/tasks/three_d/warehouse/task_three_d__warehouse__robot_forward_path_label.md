# `task_three_d__warehouse__robot_forward_path_label`

## Summary
- Domain: `three_d`
- Scene id: `warehouse`
- Public task id: `task_three_d__warehouse__robot_forward_path_label`
- Supported `query_id`: `single`
- Answer schema: `option_letter`
- Annotation schema: `bbox`

## Program Contract

Program: `select(label(candidate_objects, argmin(positive_forward_distance_in_robot_path_corridor))); scene=warehouse; scope=robot_forward_path_label`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `robot_forward_path_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `candidate_objects`, `argmin`, `positive_forward_distance_in_robot_path_corridor`, `warehouse`, `robot_forward_path_label`.
Operation: evaluate `select` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`, `topology`

## Prompt And Trace
The prompt bundle is `three_d_warehouse_v1` under `src/trace_tasks/resources/prompts/three_d/warehouse/`. The trace records camera pose, projection frame, scene variant, robot heading/design/color metadata, travel direction vector, path corridor polygon, candidate object types by label, forward/lateral path coordinates by label, first-reached flags by label, selected object id/type, projected bboxes, and option-panel descriptors/bboxes.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config defaults, prompt bundle, and code versions. Answer and annotation come from the same finalized 3D warehouse scene trace.
