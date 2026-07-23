# `task_three_d__object_scene__line_side_label`

## Summary
- Domain: `three_d`
- Scene id: `object_scene`
- Package: `src/trace_tasks/tasks/three_d/object_scene/`
- Supported `query_id`: `left_of_directed_line`, `right_of_directed_line`
- Answer type: `option_letter`
- Annotation type: `point`
- Annotation schema: `point`

## Program Contract

Program: `select(label(marked_points, side_of_directed_line(reference_object_a, reference_object_b) = requested_side)); scene=object_scene; scope=line_side_label`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `line_side_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `marked_points`, `side_of_directed_line`, `reference_object_a`, `reference_object_b`, `requested_side`, `object_scene`, `line_side_label` plus the active `query_id` branch.
Operation: evaluate `select` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `left_of_directed_line`, `right_of_directed_line`.

## Reasoning Operations

Families: `spatial_relations`

## Contract
The image uses the `object_scene` renderer: a perspective 3D floor, table, or platform scene with one or more large context props, several uniquely named small objects, and six lettered point markers. Each marked point is rendered as a visible point glyph with a nearby letter label. The prompt names two unique reference objects and asks for the marked point on the requested side of the directed line from the first named object to the second.

The verifier computes the answer from finalized object projections and marker projections, not from pixels. The correct marker is the only marked point on the requested side of the projected directed reference line with a required side-distance margin. Distractor markers are constrained to the opposite side of that line. Render style, camera, canvas preset, object placement, labels, colors, and prompt wording variants are generation metadata, not public task axes.

## Annotation Contract
Annotation is a scalar `point` at the selected marked point center.
The selected point-glyph center is the only visual witness; marker label text and the reference object names are used only to identify the answer option and directed-line relation.

## Prompt And Trace
The trace records selected prompt keys, camera/projection data, object and marker records, rendered pixel witnesses, answer-support metadata, reference-object roles, requested side, and solver fields needed to recompute the answer and annotation from the same finalized scene.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config defaults, prompt bundle, and code versions. Answers and annotation come from the same finalized 3D scene trace.
