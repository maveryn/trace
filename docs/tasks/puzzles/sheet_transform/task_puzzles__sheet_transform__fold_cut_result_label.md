# `task_puzzles__sheet_transform__fold_cut_result_label`

## Task
1. Domain: `puzzles`
2. Scene id: `sheet_transform`
3. Task id: `task_puzzles__sheet_transform__fold_cut_result_label`
4. Answer schema: `option_letter`
5. Annotation schema: `bbox`
6. Supported `query_id`: `single`

## Program Contract

Program: `select_label(sheet_transform.fold_cut.option, rule=fold_cut_unfolded_hole_pattern, options=4); scene=sheet_transform; scope=fold_cut_result_label`

Candidate set: the visible source sheet/fold/cut/overlay panels, marked cells or holes, and labeled result options inside the `fold_cut_result_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `sheet_transform`, `fold_cut`, `fold_cut_unfolded_hole_pattern`, `fold_cut_result_label`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; the selected unfolded-result option letter.
Annotation witnesses: `annotation` uses the `bbox` schema; exactly one image-pixel bbox around the selected option panel.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`

## Contract
1. The scene shows one fold-and-cut reference panel and exactly four labeled unfolded-result options.
2. Parameter axes: `fold_count=1|2`; `fold_axis=vertical|horizontal`.
3. Scene axis: `fold_strip|fold_card|fold_outline`.
4. Rendered cut holes use one sampled shape per instance: `circle|square|diamond|rounded_square`.
5. Answer is the selected unfolded-result option letter.
6. Annotation is exactly one image-pixel bbox around the selected option panel.
7. The correct option is unique by construction.
8. `scalar_annotation_checked=true`
