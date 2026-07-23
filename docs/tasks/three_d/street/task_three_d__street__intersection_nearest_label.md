# `task_three_d__street__intersection_nearest_label`

## Summary
- Domain: `three_d`
- Scene id: `street`
- Scene package: `street`
- Supported `query_id`: `single`
- Answer type: `option_letter`
- Annotation schema: `bbox`

## Program Contract

Program: `select(label(candidate_street_objects, argmin(ground_distance_to_intersection_center))); scene=street; scope=intersection_nearest_label`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `intersection_nearest_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `candidate_street_objects`, `argmin`, `ground_distance_to_intersection_center`, `street`, `intersection_nearest_label`.
Operation: evaluate `select` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`

## Contract
The image shows a synthetic perspective 3D street intersection or T intersection with roads, sidewalks, crosswalk markings, unlettered street context, unlettered street-object candidates, and a below-scene text option panel. The street surface renders full-bleed: sidewalk ground fills the canvas and road strips continue to the visible image edges.

Each instance renders `4` unlettered candidate street objects. Exactly one candidate has the smallest finalized ground-plane distance from its object center to the intersection center, with a margin from the next-nearest candidate.

Street context can include buildings, storefronts, trees, shrubs, traffic lights, street signs, benches, and other unlettered objects. Context objects are visual context only and are excluded from answer candidates.

## Annotation Contract
Annotation is the bounding box of the selected street object in the scene. Road markings, the intersection center, unlettered context objects, the option panel, and option text are not annotation.

## Prompt And Trace
The prompt bundle is `three_d_street_v1` under `src/trace_tasks/resources/prompts/three_d/street/`. The trace records camera pose, projection frame, scene variant, intersection layout, intersection center, full-bleed floor bounds, candidate ground positions, candidate distances to the intersection center, near-to-far label order, selected object id/type, projected object bboxes, and option-panel descriptors/bboxes.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config defaults, prompt bundle, and code versions. Answers and annotation come from the same finalized 3D street scene trace.
