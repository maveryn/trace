# `task_three_d__street__same_road_arm_reference_label`

## Summary
- Domain: `three_d`
- Scene id: `street`
- Scene package: `street`
- Supported `query_id`: `single`
- Answer type: `option_letter`
- Annotation schema: `bbox`

## Program Contract

Program: `select(label(candidate_street_objects, predicate=road_arm(candidate)==road_arm(reference_object))); scene=street; scope=same_road_arm_reference_label`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `same_road_arm_reference_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `candidate_street_objects`, `predicate`, `road_arm`, `candidate`, `reference_object`, `street`, `same_road_arm_reference_label`.
Operation: evaluate `select` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`

## Contract
The image shows a synthetic perspective 3D street intersection or T intersection with roads, sidewalks, crosswalk markings, unlettered street context, one red-boxed reference street object, unlettered street-object candidates, and a below-scene text option panel.

Each instance renders `4` unlettered candidate street objects. The reference object is unlettered and marked with a red bounding box. Candidate object types exclude the reference type. Exactly one candidate has the same finalized `road_arm` metadata as the reference. Distractor candidates are placed on other present road arms, and T-intersection layouts remove candidates from the missing road arm.

Street context can include buildings, storefronts, trees, shrubs, traffic lights, street signs, and benches. Context objects are visual context only and are excluded from answer candidates.

## Annotation Contract
Annotation is the bounding box of the selected street object in the scene. The red reference box, road markings, context objects, option panel, and option text are not annotation.

## Prompt And Trace
The prompt bundle is `three_d_street_v1` under `src/trace_tasks/resources/prompts/three_d/street/`. The trace records camera pose, projection frame, scene variant, intersection layout, full-bleed floor bounds, present/missing road arms, reference object id/type/name/road arm, candidate road arms by label, same-road-arm flags by label, selected object id/type, projected object bboxes, and option-panel descriptors/bboxes.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config defaults, prompt bundle, and code versions. Answers and annotation come from the same finalized 3D street scene trace.
