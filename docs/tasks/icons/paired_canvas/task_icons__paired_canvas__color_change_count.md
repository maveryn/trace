# `task_icons__paired_canvas__color_change_count`

## Identity
- domain: `icons`
- scene_id: `paired_canvas`
- module: `src/trace_tasks/tasks/icons/paired_canvas/color_change_count.py`
- prompt bundle: `icons_paired_canvas_v0`

## Program Contract

Program: `count.pairwise_comparison(scene=paired_canvas, scope=aligned_left_right_icon_pairs, predicate=color_changed, output=count)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `aligned_left_right_icon_pairs` objective scope.
Operands: visible scene state and prompt-bound operands named by `paired_canvas`, `aligned_left_right_icon_pairs`, `predicate`, `color_changed`.
Operation: evaluate `count.pairwise_comparison` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`
