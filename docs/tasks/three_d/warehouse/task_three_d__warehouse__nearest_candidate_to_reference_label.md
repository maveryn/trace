# `task_three_d__warehouse__nearest_candidate_to_reference_label`

## Summary
- Domain: `three_d`
- Scene id: `warehouse`
- Public task id: `task_three_d__warehouse__nearest_candidate_to_reference_label`
- Supported `query_id`: `closest_object_to_reference`, `closest_object_to_robot`
- Answer schema: `option_letter`
- Annotation schema: `bbox`

## Program Contract

Program: `select(label(candidate_items, argmin(ground_plane_surface_gap_to_reference))); scene=warehouse; scope=nearest_candidate_to_reference_label`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `nearest_candidate_to_reference_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `candidate_items`, `argmin`, `ground_plane_surface_gap_to_reference`, `warehouse`, `nearest_candidate_to_reference_label` plus the active `query_id` branch.
Operation: evaluate `select` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `closest_object_to_reference`, `closest_object_to_robot`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`

## Prompt And Trace
The prompt bundle is `three_d_warehouse_v1` under `src/trace_tasks/resources/prompts/three_d/warehouse/`. The trace records camera pose, projection frame, scene variant, aisle heading, reference metadata, candidate labels/types, robot designs/headings/colors where applicable, nearest-distance order, per-label distances to the reference, selected object id, projected bboxes, and option-panel descriptors/bboxes.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config defaults, prompt bundle, and code versions. Answer and annotation come from the same finalized 3D warehouse scene trace.
