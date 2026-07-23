# `task_icons__icon_field__frequency_extreme_type_label`

## Identity
- domain: `icons`
- scene_id: `icon_field`
- module: `src/trace_tasks/tasks/icons/icon_field/frequency_extreme_type_label.py`
- prompt bundle: `icons_icon_field_v1`

## Program Contract

Program: `selection.frequency_extreme_type(scene=icon_field, scope=marked_icon_types, extrema=most|least, output=option_letter)`

Candidate set: marked representative icons in the single icon-field panel. Each marked representative stands for one icon type, and all icon types present in the field are marked.
Operands: visible icon type identity and frequency for every icon in the panel.
Operation: choose the marked icon type with the unique maximum or unique minimum frequency, depending on the query id. Generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema.
Annotation schema: `bbox_set`.
Annotation witnesses: `annotation` contains one icon-object bbox for every icon of the selected marked type, including the marked representative.
Query ids: `most_frequent_type_label`, `least_frequent_type_label`.

## Reasoning Operations

Families: `filtering`, `ranking`

## Notes
All icons in a generated instance share one rendered color, so color is not a counting cue. The candidate letters are drawn directly on representative icons in the main canvas; there is no separate option panel.
