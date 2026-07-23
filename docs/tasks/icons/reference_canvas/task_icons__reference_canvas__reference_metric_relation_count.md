# `task_icons__reference_canvas__reference_metric_relation_count`

## Identity
- domain: `icons`
- scene_id: `reference_canvas`
- module: `src/trace_tasks/tasks/icons/reference_canvas/reference_metric_relation_count.py`
- prompt bundle: `icons_reference_canvas_v1`

## Program Contract

Program: `count.reference_icon_metric_predicate(scene=reference_canvas, scope=scene_panel_icons, reference=left_panel_reference_icon, metric=nominal_size, predicates=smaller_than_reference|larger_than_reference, output=count)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `scene_panel_icons` objective scope.
Operands: visible scene state and prompt-bound operands named by `reference_canvas`, `scene_panel_icons`, `reference`, `left_panel_reference_icon`, `metric`, `nominal_size`, `predicates`, `smaller_than_reference`, `larger_than_reference`.
Operation: evaluate `count.reference_icon_metric_predicate` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Annotation schema: `bbox_set`.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Notes
The concrete Reference size, per-icon nominal sizes, and minimum size delta are
retained in trace metadata. Annotation covers only the counted Scene icons.
