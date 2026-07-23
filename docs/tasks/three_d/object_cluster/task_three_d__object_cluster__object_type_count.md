# `task_three_d__object_cluster__object_type_count`

## Summary
- Domain: `three_d`
- Scene id: `object_cluster`
- Package: `src/trace_tasks/tasks/three_d/object_cluster/`
- Supported `query_id`: `single`
- Answer type: `integer`
- Annotation type: unordered `bbox_set`
- Annotation schema: `bbox_set`

## Program Contract

Program: `count(filter(object_cluster_objects, shape_type = target_shape_type)); scene=object_cluster; scope=object_type_count`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `object_type_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `object_cluster_objects`, `shape_type`, `target_shape_type`, `object_cluster`, `object_type_count`.
Operation: evaluate `count` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Contract
The image shows many small synthetic perspective 3D objects arranged on a plain surface. This scene is a bare clustered-counting surface: it does not use option labels, named reference objects, relation prompts, or grid-based spatial cues.

The prompt asks how many objects of one named type are present. The target type
is sampled from the full cluster object pool: the prompt-safe object-scene small
shapes plus CountQA-aligned loose objects such as writing tools, flat packets,
small tableware, hardware, containers, miniature furniture, plants, and game
pieces.

Generation samples a distractor-backed `cluster_composition_mode` axis:

- `near_homogeneous_cluster` (`0.7`): the scene is mostly the target type with `1-4` visually distinct non-target distractors; target answer is in `4-8`.
- `mixed_type_cluster` (`0.3`): the target type is counted among more varied distractor types, with target answer in `4-8` and total object count capped at `20`.

The default answer distribution is weighted over the capped `4-8` support. Visually confusable same-family distractors are excluded for the selected target in modes that use distractors.

Prompt-facing wording must not mention absent labels, missing letters, or other non-present features for this scene. The prompt should describe only the visible scaffold and ask the count question directly.

The answer is the integer count of finalized objects whose `shape_type` equals the sampled target shape. Pixels are render output, not verifier source of truth.

## Annotation Contract
Annotation is a `bbox_set` containing one `[x0, y0, x1, y1]` pixel box around each counted target object. The annotation set is unordered because all witnesses have the same semantic role and annotation cardinality matches the answer.

## Prompt And Trace
The prompt bundle is `three_d_object_cluster_v1` under `src/trace_tasks/resources/prompts/three_d/object_cluster/`. The trace records camera pose, projection frame, object world coordinates, sampled dimensions, prompt-facing object names, target shape, target object ids, per-shape counts, projected object boxes, and the solver count predicate.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config defaults, prompt bundle, and code versions. Answers and annotation come from the same finalized 3D scene trace.
