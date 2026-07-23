# `task_three_d__object_scene__reference_triangle_inside_label`

## Summary
- Domain: `three_d`
- Scene id: `object_scene`
- Package: `src/trace_tasks/tasks/three_d/object_scene/`
- Supported `query_id`: `single`
- Answer type: `option_letter`
- Annotation type: `point`
- Annotation schema: `point`

## Program Contract

Program: `select(label(marked_points, inside_triangle(reference_object_a, reference_object_b, reference_object_c))); scene=object_scene; scope=reference_triangle_inside_label`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `reference_triangle_inside_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `marked_points`, `inside_triangle`, `reference_object_a`, `reference_object_b`, `reference_object_c`, `object_scene`, `reference_triangle_inside_label`.
Operation: evaluate `select` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `spatial_relations`

## Contract
The image uses the `object_scene` renderer: a perspective 3D floor, table, or platform scene with one or more large context props, several uniquely named small objects, and six lettered point markers. Each marked point is rendered as a visible point glyph with a nearby letter label. The prompt names three unique reference objects and asks for the marked point inside the triangle formed by those reference objects.

The verifier computes the answer from finalized object projections and marker projections, not from pixels. The correct marker is the only marked point with a positive containment margin inside the projected reference triangle. Distractor markers are constrained outside the triangle with a required outside margin. Render style, camera, canvas preset, object placement, labels, colors, and prompt wording variants are generation metadata, not public task axes.

## Annotation Contract
Annotation is a scalar `point` at the selected marked point center.
The selected point-glyph center is the only visual witness; marker label text and the reference object names are used only to identify the answer option and triangle relation.

## Prompt And Trace
The trace records selected prompt keys, camera/projection data, object and marker records, rendered pixel witnesses, answer-support metadata, reference-object roles, triangle vertices, and solver fields needed to recompute the answer and annotation from the same finalized scene.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config defaults, prompt bundle, and code versions. Answers and annotation come from the same finalized 3D scene trace.
