# `task_icons__icon_field__singleton_type_count`

## Identity
- domain: `icons`
- scene_id: `icon_field`
- module: `src/trace_tasks/tasks/icons/icon_field/singleton_type_count.py`
- prompt bundle: `icons_icon_field_v1`

## Program Contract

Program: `count.group_predicate(scene=icon_field, scope=single_panel_icon_types, groups=icon_type, predicate=singleton, output=count)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `single_panel_icon_types` objective scope.
Operands: visible scene state and prompt-bound operands named by `icon_field`, `single_panel_icon_types`, `groups`, `icon_type`, `predicate`, `singleton`.
Operation: evaluate `count.group_predicate` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation schema: `bbox_set`.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Notes
The scene enforces the requested singleton count by construction, while
repeated icon types serve as distractors.
