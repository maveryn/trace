# `task_icons__reference_canvas__anchor_position_count`

## Identity
- domain: `icons`
- scene_id: `reference_canvas`
- module: `src/trace_tasks/tasks/icons/reference_canvas/anchor_position_count.py`
- prompt bundle: `icons_reference_canvas_v1`

## Program Contract

Program: `count.reference_icon_anchor_spatial_predicate(scene=reference_canvas, scope=scene_panel_candidate_icons, reference=left_panel_reference_icon, anchor=marked_scene_icon, predicates=left_of_anchor|right_of_anchor|above_anchor|below_anchor, output=count)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `scene_panel_candidate_icons` objective scope.
Operands: visible scene state and prompt-bound operands named by `reference_canvas`, `scene_panel_candidate_icons`, `reference`, `left_panel_reference_icon`, `anchor`, `marked_scene_icon`, `predicates`, `left_of_anchor`, `right_of_anchor`, `above_anchor`, `below_anchor`.
Operation: evaluate `count.reference_icon_anchor_spatial_predicate` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Annotation schema: `bbox_set`.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Notes
The Reference and Anchor boxes are retained in trace metadata but are not part
of user-facing annotation. The Anchor icon itself is not part of the counted
candidate set.
