# `task_three_d__object_cluster__counterfactual_count`

## Summary
- Domain: `three_d`
- Scene id: `object_cluster`
- Package: `src/trace_tasks/tasks/three_d/object_cluster/`
- Supported `query_id`: `single`
- Answer type: `integer`
- Annotation type: unordered `bbox_set`
- Annotation schema: `bbox_set`

## Program Contract

Program: `count(filter(object_cluster_objects, predicate = target_predicate)) + sum(edit_delta_i(target_predicate, ordered_edit_sequence)); scene=object_cluster; scope=counterfactual_count`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `counterfactual_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `object_cluster_objects`, `predicate`, `target_predicate`, `sum`, `edit_delta_i`, `ordered_edit_sequence`, `object_cluster`, `counterfactual_count`.
Operation: evaluate `count` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `aggregation`, `state_update`, `formula_evaluation`

## Contract
The image shows many small synthetic perspective 3D objects arranged on a plain surface. This scene is a bare clustered-counting surface: it does not use option labels, named reference objects, relation prompts, or grid-based spatial cues.

The prompt asks for the final count of a queried property after applying an ordered sequence of textual add/remove edits to the visible starting cluster. The target predicate is one of:

- object type only, such as `cups`
- semantic color only, such as `blue [#1F77B4] objects`
- object type plus semantic color, such as `blue [#1F77B4] cups`

Each instance uses 2-3 ordered edits. At least one edit affects the queried property and at least one edit is a distractor whose predicate is disjoint from the queried property. Three-step instances may include a second target-affecting edit. Remove edits are generated only when enough starting visible objects matching that edit predicate are available, and generation keeps the final answer positive. Semantic color targets use the repo-wide canonical named color palette and prompt-facing hex labels.

The image includes non-target distractors. For color+object targets, generation includes structured partial-match distractors when possible, such as same-type/wrong-color and same-color/wrong-type objects. Wrong-type distractors avoid target-confusable object families such as card/envelope/book, sphere/button, cup/bowl/tray, lantern/candle, and pencil/ruler. The answer is computed from metadata as:

`initial_visible_target_count + sum(target_affecting_add_amounts) - sum(target_affecting_remove_amounts)`

Pixels are render output, not verifier source of truth.

## Annotation Contract
Annotation is a `bbox_set` containing one `[x0, y0, x1, y1]` pixel box around each starting visible object matching the queried property before the edits. The final answer can differ from the annotation cardinality because the prompt asks for the count after the hypothetical edit sequence.

The annotation set is unordered because all visible witnesses have the same semantic role.

## Prompt And Trace
The prompt bundle is `three_d_object_cluster_v1` under `src/trace_tasks/resources/prompts/three_d/object_cluster/`. The trace records camera pose, projection frame, object world coordinates, sampled dimensions, prompt-facing object names, semantic colors, target predicate kind, initial target count, ordered counterfactual edit steps, target-affecting deltas, final target count, target object ids, projected object boxes, and the solver count predicate.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config defaults, prompt bundle, and code versions. The starting visual annotation and final counterfactual answer are recorded in the same finalized 3D scene trace.
