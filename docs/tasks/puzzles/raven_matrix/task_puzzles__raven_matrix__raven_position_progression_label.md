# `task_puzzles__raven_matrix__raven_position_progression_label`

## Public Taxonomy
1. Domain: `puzzles`
2. Scene id: `raven_matrix`
3. Source scene: `raven_matrix`
4. Task id: `task_puzzles__raven_matrix__raven_position_progression_label`

## Query Contract
1. Supported `query_id`: `single`
2. Internal question format: `raven_position_progression_label`
3. Prompt asks for the option that completes the missing lower-right Raven matrix cell under a marker-position progression rule.
4. Internal variation: progression mode, row/column line assignment, option order, option label, scene treatment, and style are generation/render metadata.

## Program Contract

Program: `select_label(raven_option, rule=non_wrapping_position_progression_matrix_completion); scene=raven_matrix; scope=raven_position_progression_label`

Candidate set: the visible Raven matrix cells, visual features, missing-cell cue, rule-bearing rows/columns, and labeled candidate options inside the `raven_position_progression_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `raven_option`, `non_wrapping_position_progression_matrix_completion`, `raven_matrix`, `raven_position_progression_label`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; the capital-letter label on the correct option cell.
Annotation witnesses: `annotation` uses the `bbox` schema; one bbox around the correct option cell.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `matching`

## Answer And Annotation
1. `answer_gt.type = option_letter`
2. `answer_gt.value` is the capital-letter label on the correct option cell.
3. `annotation_gt.type = bbox`
4. Annotation schema: scalar `bbox`
5. Annotation target: one bbox around the correct option cell.
6. `scalar_annotation_checked = true`.

## Trace Contract
1. `execution_trace.raven_rule_code = position_progression_matrix`.
2. `execution_trace.option_specs` records all option cells, labels, and the unique correct option.
3. `render_map.option_cell_bboxes_px` contains the selected option cell bbox projected into `annotation_gt`.
4. `execution_trace.solver_trace.progression_mode` records the concrete non-wrapping mode.
5. `execution_trace.solver_trace.progression_axis` records whether the progression is row-wise or column-wise.
6. `execution_trace.solver_trace.progression_direction` records the within-mini-grid direction.
7. `execution_trace.solver_trace.progression_line_indices` records the mini-grid row or column assigned to each matrix row or column.
8. `execution_trace.solver_trace.position_table` records the generated marker coordinates for every matrix cell.

## Prompt Contract
1. Bundle: `puzzles_raven_matrix_v1`
2. Scene key: `raven_matrix`
3. Task key: `raven_position_progression_label_query`
4. Query key: `raven_position_progression_label`
