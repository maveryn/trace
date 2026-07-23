# `task_icons__paired_canvas__panel_set_relation_count`

## Identity
- domain: `icons`
- scene_id: `paired_canvas`
- source scene: `paired_canvas`
- module: `src/trace_tasks/tasks/icons/paired_canvas/panel_set_relation_count.py`
- prompt bundle: `icons_paired_canvas_v0`

## Program Contract

Program: `count.set_relation(scene=paired_canvas, scope=left_right_icon_panels, relation=added_in_right|missing_from_right, output=count)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `left_right_icon_panels` objective scope.
Operands: visible scene state and prompt-bound operands named by `paired_canvas`, `left_right_icon_panels`, `relation`, `added_in_right`, `missing_from_right`.
Operation: evaluate `count.set_relation` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `added_in_right_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`

## Scene And Query
Renders two icon panels labeled `Left` and `Right`, then asks for a set relation
between the panels.

Query ids:
- `added_in_right_count`
- `missing_from_right_count`

Answer schema: integer.
Annotation schema: `bbox_set`; added-icon queries box counted Right-panel icons,
while missing-icon queries box counted Left-panel icons.
`projected_annotation` mirrors this as typed bbox-set annotation with `bbox_set`,
`pixel_bbox_set`, and bbox-center `pixel_point_set`.

Scalar annotation checked: not applicable. The count can have zero or multiple
visual witnesses, so `bbox_set` is the stable annotation schema.

## Notes
Added and missing identities are sampled distinctly so panel membership is
unambiguous.
Render metadata records panel-title text legibility.
