# `task_puzzles__color_gradient__color_gradient_violation_cell_label`

## Program Contract

Program: `find_gradient_violation(swatch_grid, row_column_progression_rule); scene=color_gradient; scope=color_gradient_violation_cell_label`

Candidate set: the visible swatch sequence or swatch grid, missing/violating swatches, and labeled options or cells inside the `color_gradient_violation_cell_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `swatch_grid`, `row_column_progression_rule`, `color_gradient`, `color_gradient_violation_cell_label`.
Operation: evaluate `find_gradient_violation` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; the violating swatch label.
Annotation witnesses: `annotation` uses the `bbox` schema; the image-pixel bounding box of the violating swatch cell.
Query ids: `single`.

## Reasoning Operations

Families: `matching`

## Answer And Annotation
1. `answer_gt.type = option_letter`
2. `answer_gt.value` is the violating swatch label.
3. `annotation_gt.type = bbox`
4. Annotation schema: `bbox`
5. Annotation is the image-pixel bounding box of the violating swatch cell.
6. `scalar_annotation_checked = true`; this task always has one visual witness.

## Query Contract
1. Public `query_id`: `single`
2. Internal trace metadata may vary grid size, progression rule, scene variant,
   font, unit-size jitter, and answer label.
3. These generation axes do not change the answer schema, annotation schema, or
   reasoning program.

## Prompt Contract
1. Bundle: `puzzles_color_gradient_v1`
2. Scene key: `color_gradient`
3. Task key: `color_gradient_violation_query`
4. Prompt query key: `color_gradient_violation_cell_label`
