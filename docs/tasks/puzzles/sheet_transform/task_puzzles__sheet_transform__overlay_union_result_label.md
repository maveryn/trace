# `task_puzzles__sheet_transform__overlay_union_result_label`

## Task
1. Domain: `puzzles`
2. Scene id: `sheet_transform`
3. Task id: `task_puzzles__sheet_transform__overlay_union_result_label`
4. Answer schema: `option_letter`
5. Annotation schema: `bbox`
6. Supported `query_id`: `single`

## Program Contract

Program: `select_label(sheet_transform.overlay_union.option, rule=union_of_two_aligned_source_sheets, options=4); scene=sheet_transform; scope=overlay_union_result_label`

Candidate set: the visible source sheet/fold/cut/overlay panels, marked cells or holes, and labeled result options inside the `overlay_union_result_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `sheet_transform`, `overlay_union`, `union_of_two_aligned_source_sheets`, `overlay_union_result_label`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; the selected overlay-result option letter.
Annotation witnesses: `annotation` uses the `bbox` schema; exactly one image-pixel bbox around the selected option panel.
Query ids: `single`.

## Reasoning Operations

Families: `logical_composition`, `transformation`

## Contract
1. The scene shows two aligned source sheets and exactly four labeled result options.
2. Selection rule: the selected option's marked cells equal the union of the two source-sheet marked cell sets.
3. Scene axis: `overlay_strip|overlay_card|overlay_outline`.
4. Answer is the selected overlay-result option letter.
5. Annotation is exactly one image-pixel bbox around the selected option panel.
6. The correct option is unique by construction.
7. `scalar_annotation_checked=true`
