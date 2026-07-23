# `task_icons__single_transform_options__geometric_transform_result_label`

## Program Contract

Program: `selection.geometric_transform_result_label(scene=single_transform_options, scope=labeled_option_cells, source=reference_icon, operation=rotate_90_clockwise|rotate_90_counterclockwise|rotate_180|flip_horizontal|flip_vertical, output=option_letter)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `labeled_option_cells` objective scope.
Operands: visible scene state and prompt-bound operands named by `single_transform_options`, `labeled_option_cells`, `source`, `reference_icon`, `operation`, `rotate_90_clockwise`, `rotate_90_counterclockwise`, `rotate_180`, `flip_horizontal`, `flip_vertical` plus the active `query_id` branch.
Operation: evaluate `selection.geometric_transform_result_label` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `selected_option` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `rotate_90_clockwise_result_label`, `rotate_90_counterclockwise_result_label`, `rotate_180_result_label`, `flip_horizontal_result_label`, `flip_vertical_result_label`.

## Reasoning Operations

Families: `transformation`

## Identity

- Domain: `icons`
- Scene id: `single_transform_options`
- Task id: `task_icons__single_transform_options__geometric_transform_result_label`
- Objective: select the labeled option that shows a Reference icon after one geometric transform.

## Contract

- Supported `query_id` values: `rotate_90_clockwise_result_label`, `rotate_90_counterclockwise_result_label`, `rotate_180_result_label`, `flip_horizontal_result_label`, `flip_vertical_result_label`.
- Answer schema: `option_letter`.
- Annotation schema: `bbox_map`, keyed by `reference_icon` and `selected_option`.
- The rendered image has one Reference icon with a visible transform cue and six labeled option cells.
- Exactly one option cell shows the Reference icon after the queried transform.
- Distractors are the other supported transform results of the same Reference icon.

## Generation

- Uses the curated non-symmetric icon pool so identity, rotation, and flip signatures remain visually distinct.
- All six option icons share one tint within an instance, so the answer depends on geometric transform rather than color.
- The Reference icon and option icons use the same nominal icon size.
- Query selection is task-owned and uniform unless an explicit supported `query_id` is supplied.
- `object_count` is fixed at `6` because this task is a six-option visual MCQ.

## Prompt

- Prompt bundle: `icons_single_transform_options_v1`
- `scene_key`: `single_transform_options_transformation`
- `task_key`: `transformation_query`
- Query templates name the requested transform and ask for the option letter.
- Answer JSON shape: `{"answer":"C"}`
- Answer+annotation JSON shape: `{"annotation":{"reference_icon":[82,144,250,312],"selected_option":[532,104,702,274]},"answer":"C"}`

## Annotation

- `reference_icon` marks the visible Reference icon bbox.
- `selected_option` marks the full selected option cell bbox, not just the transformed icon.
- The annotation is keyed because the two boxes have different semantic roles.

## Tests

- Behavior and trace tests: `tests/test_icons_transformation_single_transform_options_tasks.py`
- Config and prompt bundle tests: `tests/test_icons_scene_config.py`
- Source-layout contract checks: `tests/test_public_source_layout_contracts.py`
