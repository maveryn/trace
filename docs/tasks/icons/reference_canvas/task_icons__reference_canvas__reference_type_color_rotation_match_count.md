# `task_icons__reference_canvas__reference_type_color_rotation_match_count`

## Identity
- domain: `icons`
- scene_id: `reference_canvas`
- module: `src/trace_tasks/tasks/icons/reference_canvas/reference_type_color_rotation_match_count.py`
- prompt bundle: `icons_reference_canvas_v1`

## Program Contract

Program: `count.reference_icon_predicate(scene=reference_canvas, scope=scene_panel_icons, predicate=same_type_and_color_and_rotation, output=count)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `scene_panel_icons` objective scope.
Operands: visible scene state and prompt-bound operands named by `reference_canvas`, `scene_panel_icons`, `predicate`, `same_type_and_color_and_rotation`.
Operation: evaluate `count.reference_icon_predicate` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Annotation schema: `bbox_set`.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `transformation`, `matching`
