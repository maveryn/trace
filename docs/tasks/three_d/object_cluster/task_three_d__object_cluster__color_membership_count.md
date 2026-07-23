# `task_three_d__object_cluster__color_membership_count`

## Summary
- Domain: `three_d`
- Scene id: `object_cluster`
- Package: `src/trace_tasks/tasks/three_d/object_cluster/`
- Supported `query_id`: `single`
- Answer type: `integer`
- Annotation type: unordered `bbox_set`
- Annotation schema: `bbox_set`

## Program Contract

Program: `count(filter(object_cluster_objects, color_name = target_color_name)); scene=object_cluster; scope=color_membership_count`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `color_membership_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `object_cluster_objects`, `color_name`, `target_color_name`, `object_cluster`, `color_membership_count`.
Operation: evaluate `count` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Contract
The image shows many small synthetic perspective 3D colored objects arranged on a plain surface. The prompt asks how many objects match one semantic color, such as blue objects or red objects. Prompt-facing semantic colors include the canonical color hex label, for example `blue [#2D75E6]`, while verifier metadata keeps the raw color name.

The answer is the integer count of finalized clustered objects whose recorded `color_name` equals the sampled target color. Semantic color is recorded in verifier metadata as `color_name`, `prompt_color_name`, and `fill_rgb`; pixels are render output, not verifier source of truth. Generated color distractors avoid near-color named pairs such as blue/cyan/purple and red/maroon/magenta.

## Annotation Contract
Annotation is a `bbox_set` containing one `[x0, y0, x1, y1]` pixel box around each counted object matching the requested color. The annotation set is unordered because all witnesses have the same role and annotation cardinality matches the answer.

## Prompt And Trace
The prompt bundle is `three_d_object_cluster_v1` under `src/trace_tasks/resources/prompts/three_d/object_cluster/`. The trace records camera pose, projection frame, object world coordinates, sampled dimensions, prompt-facing object names, semantic colors, target color, target object ids, per-color counts, projected object boxes, and the solver count predicate.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config defaults, prompt bundle, and code versions. Answers and annotation come from the same finalized 3D scene trace.
