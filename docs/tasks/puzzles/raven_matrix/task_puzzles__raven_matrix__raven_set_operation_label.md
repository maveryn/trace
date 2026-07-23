# `task_puzzles__raven_matrix__raven_set_operation_label`

## Public Taxonomy
1. Domain: `puzzles`
2. Scene id: `raven_matrix`
3. Source scene: `raven_matrix`
4. Task id: `task_puzzles__raven_matrix__raven_set_operation_label`

## Query Contract
1. Supported `query_id`: `single`
2. Internal question format: `raven_set_operation_label`
3. Prompt asks for the option that completes the missing lower-right Raven matrix cell under a filled-cell combination rule.
4. Supported set operations are `union` and `intersection`.
5. Internal variation: set operation, option order, option label, scene treatment, and style are generation/render metadata.

## Program Contract

Program: `select_label(raven_option, rule=row_set_operation_matrix_completion); scene=raven_matrix; scope=raven_set_operation_label`

Candidate set: the visible Raven matrix cells, visual features, missing-cell cue, rule-bearing rows/columns, and labeled candidate options inside the `raven_set_operation_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `raven_option`, `row_set_operation_matrix_completion`, `raven_matrix`, `raven_set_operation_label`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; the capital-letter label on the correct option cell.
Annotation witnesses: `annotation` uses the `bbox` schema; one bbox around the correct option cell.
Query ids: `single`.

## Reasoning Operations

Families: `logical_composition`, `matching`

## Answer And Annotation
1. `answer_gt.type = option_letter`
2. `answer_gt.value` is the capital-letter label on the correct option cell.
3. `annotation_gt.type = bbox`
4. Annotation schema: scalar `bbox`
5. Annotation target: one bbox around the correct option cell.
6. `scalar_annotation_checked = true`.

## Trace Contract
1. `execution_trace.raven_rule_code = set_operation_matrix`.
2. `execution_trace.solver_trace.operation` records either `union` or `intersection`.
3. `execution_trace.option_specs` records all option cells, labels, and the unique correct option.
4. `render_map.option_cell_bboxes_px` contains the selected option cell bbox projected into `annotation_gt`.

## Prompt Contract
1. Bundle: `puzzles_raven_matrix_v1`
2. Scene key: `raven_matrix`
3. Task key: `raven_set_operation_label_query`
4. Query key: `raven_set_operation_label`
5. Prompt wording may identify the broad filled-cell combination family, but must not reveal the concrete set operation or row/operand structure.
