# `task_icons__single_transform_options__inverse_geometric_transform_source_label`

## Program Contract

Program: `selection.inverse_geometric_transform_source_label(scene=single_transform_options, scope=labeled_source_option_cells, reference=result_icon, operation=rotate_90_clockwise|rotate_90_counterclockwise|rotate_180|flip_horizontal|flip_vertical, output=option_letter)`

Candidate set: four labeled source-option cells.
Operands: one transformed Reference icon, the visible operation cue, and four
labeled candidate source icons.
Operation: choose the option icon that would become the Reference icon after
the shown operation is applied to that option.
Output binding: `answer` uses the `option_letter` schema; generation binds a
unique final option by construction.
Annotation witnesses: `annotation` uses the `bbox_map` schema with keys
`reference_icon` and `selected_option`.
Query ids: `rotate_90_clockwise_source_label`,
`rotate_90_counterclockwise_source_label`, `rotate_180_source_label`,
`flip_horizontal_source_label`, `flip_vertical_source_label`.

## Reasoning Operations

Families: `transformation`

## Identity

- Domain: `icons`
- Scene id: `single_transform_options`
- Task id: `task_icons__single_transform_options__inverse_geometric_transform_source_label`
- Objective: select the source option that produces the Reference after one
  geometric transform.

## Contract

- Supported query ids are the five inverse operation branches listed in the
  program contract.
- Answer schema: `option_letter`.
- Annotation schema: `bbox_map`, keyed by `reference_icon` and
  `selected_option`.
- The rendered image has one Reference icon with a visible transform cue and
  four labeled source-option cells.
- Exactly one option cell becomes the Reference icon after the prompted
  operation.
- The option that already looks identical to the Reference is excluded when it
  would be a distractor, so the answer cannot be obtained by direct visual
  matching.

## Generation

- Uses the curated non-symmetric icon pool so rotation and flip signatures
  remain visually distinct.
- All option icons and the Reference icon share one tint within an instance, so
  the answer depends on geometric transform rather than color.
- The Reference icon is sampled as the transformed result of the hidden correct
  source option.
- Query selection is task-owned and uniform unless an explicit supported
  `query_id` is supplied.
- `object_count` is fixed at `4` because this inverse task intentionally uses a
  four-option visual MCQ.

## Prompt

- Prompt bundle: `icons_single_transform_options_v1`
- `scene_key`: `single_transform_options_inverse_transformation`
- `task_key`: `transformation_query`
- Query templates ask which option would become the Reference icon after the
  named operation is applied.
- Answer JSON shape: `{"answer":"C"}`
- Answer+annotation JSON shape:
  `{"annotation":{"reference_icon":[82,144,250,312],"selected_option":[532,104,702,274]},"answer":"C"}`

## Annotation

- `reference_icon` marks the visible Reference icon bbox.
- `selected_option` marks the full selected option cell bbox, not just the
  source icon.
- The annotation is keyed because the two boxes have different semantic roles.

## Tests

- Behavior and trace tests:
  `tests/test_icons_transformation_single_transform_options_tasks.py`
- Config and prompt bundle tests: `tests/test_icons_scene_config.py`
- Source-layout contract checks:
  `tests/test_public_source_layout_contracts.py`
