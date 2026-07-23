# `task_puzzles__sheet_transform__fold_projection_result_label`

## Task
1. Domain: `puzzles`
2. Scene id: `sheet_transform`
3. Task id: `task_puzzles__sheet_transform__fold_projection_result_label`
4. Answer schema: `option_letter`
5. Annotation schema: `bbox`
6. Supported `query_id`: `single`

## Program Contract

Program: `select_label(sheet_transform.fold_projection.option, rule=single_axis_fold_mark_projection, options=4); scene=sheet_transform; scope=fold_projection_result_label`

Candidate set: the visible source sheet/fold/cut/overlay panels, marked cells or holes, and labeled result options inside the `fold_projection_result_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `sheet_transform`, `fold_projection`, `single_axis_fold_mark_projection`, `fold_projection_result_label`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; the selected folded-result option letter.
Annotation witnesses: `annotation` uses the `bbox` schema; exactly one image-pixel bbox around the selected option panel.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`

## Contract
1. The scene shows one fold-reference panel and exactly four labeled result options.
2. Parameter axis: `fold_axis=vertical|horizontal`.
3. Scene axis: `fold_strip|fold_card|fold_outline`.
4. Answer is the selected folded-result option letter.
5. Annotation is exactly one image-pixel bbox around the selected option panel.
6. The correct option is unique by construction.
7. `scalar_annotation_checked=true`
