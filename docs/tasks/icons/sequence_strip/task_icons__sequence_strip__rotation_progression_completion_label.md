# `task_icons__sequence_strip__rotation_progression_completion_label`

## Program Contract

Program: `selection.sequence_completion(scene=sequence_strip, scope=four_cell_icon_sequence_with_visual_options, attribute=rotation, rule=constant_rotation_step, missing_role=question_mark_cell, output=option_label)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `four_cell_icon_sequence_with_visual_options` objective scope.
Operands: visible scene state and prompt-bound operands named by `sequence_strip`, `four_cell_icon_sequence_with_visual_options`, `attribute`, `rotation`, `constant_rotation_step`, `missing_role`, `question_mark_cell`, `option_label`.
Operation: evaluate `selection.sequence_completion` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `D` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `matching`

## Identity

- Domain: `icons`
- Scene id: `sequence_strip`
- Task id: `task_icons__sequence_strip__rotation_progression_completion_label`
- Module: `src/trace_tasks/tasks/icons/sequence_strip/rotation_progression_completion_label.py`
- Prompt bundle: `icons_sequence_strip_v1`

## Contract

- Supported `query_id` values: `single`.
- Answer schema: `string`, one of `A`, `B`, `C`, or `D`.
- Annotation schema: scalar `bbox` around the correct bottom-row option box.
- The top row has four boxed sequence cells and one question-mark cell.
- The bottom row has four fixed option boxes labeled `A` through `D`.
- Rotations follow a constant angular step over asymmetric icons.

## Trace

- `execution_trace.full_sequence_values` stores the complete rotation sequence in degrees.
- `execution_trace.missing_index` stores the hidden top-row position.
- `execution_trace.option_values_by_label` maps each option label to its rotation in degrees.
