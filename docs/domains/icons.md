# Icons Domain Contract

Use this document for icon-domain rules. Exact active scenes and tasks live in
`docs/ACTIVE_TASK_INVENTORY.md` and `docs/tasks/icons/`.

## Scope
Icons covers controlled 2D icon fields, reference-vs-scene panels, icon
relations, icon transformations, and icon pattern reasoning. The semantic unit
is an icon instance, icon cell, reference cell, or visible option image.

Use `icons` for abstract icon grids/fields. Use `illustrations` when the scene
is a richer object illustration with natural placement context, and `pages` when
the icon is part of a document/app/page layout.

## Scene Boundary
A scene is the stable icon scaffold: single field, reference-and-scene panel,
grid pattern, mirror grid, strip, option gallery, or transformation display.
Style, icon pool, color palette, rotation, scale, and mild noise are scene
variants when the same objective remains valid.

Create a new scene when the reference layout, option interface, spatial
relationship, or rule-application scaffold changes materially.

## Task And Query Boundary
Counting task contracts should distinguish:

- total object count;
- single-attribute membership count;
- multi-attribute AND count;
- multi-attribute OR/union count;
- multi-attribute exclusion count;
- exact-one/XOR count;
- arithmetic over counts;
- hypothetical/counterfactual count after visible/textual edits.

Literal icon type, color, shape, size, rotation, or named category values can
be sampled parameters inside one task when the visual scan/reasoning channel
stays the same. Split public tasks when the channel or predicate arity changes:
type match vs color match, color change vs rotation change, color-pattern
violation vs size-pattern violation, and shape-only lookup vs color+shape
lookup are separate objective contracts. Arithmetic operators such as
`total_count` and `difference_count` may remain query ids when the operand roles
and annotation witnesses are otherwise unchanged.

## Annotation Policy
Prompt-facing annotation should mark icon instances, cells, reference pairs,
missing/violating slots, or selected visual option images. Use `bbox_set` for
counted icons/cells and map annotation when reference vs scene, before vs
after, anchor vs candidate, or option roles matter.

For option-image tasks, annotation may mark the selected option panel only when
the option is a complete visual candidate being matched, completed, or
transformed.

Icon-object bbox annotations must have a minimum side of 24 px after final
projection. Expand icon-object annotation boxes around the rendered icon center
when needed, clipped to the local panel/cell/slot when available. Do not use
this expansion for full option panels, grid cells, line regions, or other
non-icon witnesses.

## Assets And Rendering
Use icon manifests through shared icon asset loaders. Use asymmetric icons when
orientation, mirror symmetry, transformation identity, or attribute binding
could collapse under symmetry.

Construct positives and distractors explicitly from recorded icon attributes;
do not rely on random icon placement or palette draws to realize the answer.
Semantic icon colors, sizes, rotations, and types must be recorded in trace
metadata when queried.

Procedural named icons use small non-semantic pose jitter by default in
counting/spatial scenes: sample within +/-15 degrees unless rotation is part of
the task contract. Dense stack layouts should keep icons unrotated so row/column
stack counting remains clean.

## Shared Code
Reusable icon asset loading, color/style sampling, grid layout, relation
evaluation, and transformation helpers belong under `src/trace_tasks/tasks/icons/shared/`.
Keep scene-local helpers under the scene source directory when they are tied to one
scaffold.
