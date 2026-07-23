# `task_three_d__object_scene__object_relation_label`

## Summary
- Domain: `three_d`
- Scene id: `object_scene`
- Package: `src/trace_tasks/tasks/three_d/object_scene/`
- Supported `query_id`: `on_top_of_prop`, `under_prop`, `inside_prop`
- Answer type: `option_letter`
- Annotation type: `bbox`
- Annotation schema: `bbox`

## Program Contract

Program: `select(label(candidate_objects, spatial_relation_to_reference = requested_relation)); scene=object_scene; scope=object_relation_label`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `object_relation_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `candidate_objects`, `spatial_relation_to_reference`, `requested_relation`, `object_scene`, `object_relation_label` plus the active `query_id` branch.
Operation: evaluate `select` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `on_top_of_prop`, `under_prop`, `inside_prop`.

## Reasoning Operations

Families: `spatial_relations`

## Contract
The image uses the `object_scene` renderer: a perspective 3D floor, table, or platform scene with projected objects, markers, references, or paired views depending on the task. The public task id defines the stable objective contract; query ids are used only for genuine semantic operations within that contract. Render style, camera, canvas preset, object placement, labels, colors, and prompt wording variants are generation metadata, not public task axes.

The verifier computes the answer from finalized scene metadata and projection records, not from pixels. The prompt bundle is `three_d_object_scene_v1` under `src/trace_tasks/resources/prompts/three_d/object_scene/`.

Candidate option objects are sampled from the curated object_scene-compatible
named-object pool derived from the domain-wide `THREE_D_NAMED_OBJECT_SHAPE_TYPES`
support. This excludes broad render-only small objects such as `drum` from named
MCQ candidates. Candidate objects are rendered with a relation-task scale
multiplier so they read as small props relative to the larger reference prop.
The same candidate object pool is used for every relation query id; query ids
change only the requested spatial relation and compatible reference prop.
Piano is excluded from this task's extra context props. For `under` relations,
the selected object is scaled smaller, placed in the open interior of a table or
arch support, and accepted only when its projected bbox does not collide heavily
with visible legs, posts, or the lintel/tabletop. Under objects use normal
depth ordering rather than a foreground overpaint bias.

## Annotation Contract
Annotation is a scalar `bbox` around the selected visible object.
The selected object is the only visual witness; option text is not annotation.

## Prompt And Trace
The trace records selected prompt keys, camera/projection data, object or marker records, rendered pixel witnesses, answer-support metadata, and the solver fields needed to recompute the answer and annotation from the same finalized scene.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config defaults, prompt bundle, and code versions. Answers and annotation come from the same finalized 3D scene trace.
