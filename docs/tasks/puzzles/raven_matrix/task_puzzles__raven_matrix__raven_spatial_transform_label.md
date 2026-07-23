# `task_puzzles__raven_matrix__raven_spatial_transform_label`

## Public Taxonomy
1. Domain: `puzzles`
2. Scene id: `raven_matrix`
3. Source scene: `raven_matrix`
4. Task id: `task_puzzles__raven_matrix__raven_spatial_transform_label`

## Query Contract
1. Supported `query_id`: `single`
2. Internal question format: `raven_spatial_transform_label`
3. Prompt asks for the option that completes the missing lower-right Raven matrix cell under a spatial arrangement rule.
4. Internal variation: rotation mode, row/column progression axis, option order, option label, scene treatment, and style are generation/render metadata.

## Program Contract

Program: `select_label(raven_option, rule=rotation_90_matrix_completion); scene=raven_matrix; scope=raven_spatial_transform_label`

Candidate set: the visible Raven matrix cells, visual features, missing-cell cue, rule-bearing rows/columns, and labeled candidate options inside the `raven_spatial_transform_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `raven_option`, `rotation_90_matrix_completion`, `raven_matrix`, `raven_spatial_transform_label`.
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
1. `execution_trace.raven_rule_code = spatial_transform_matrix`.
2. `execution_trace.solver_trace.rotation_mode` records the concrete row/column rotation mode.
3. `execution_trace.solver_trace.progression_axis` records whether the rule is row-wise or column-wise.
4. `execution_trace.solver_trace.rotation_sequence` records either identity-clockwise-counterclockwise or identity-counterclockwise-clockwise.
5. `execution_trace.option_specs` records all option cells, labels, and the unique correct option.
6. `render_map.option_cell_bboxes_px` contains the selected option cell bbox projected into `annotation_gt`.

## Prompt Contract
1. Bundle: `puzzles_raven_matrix_v1`
2. Scene key: `raven_matrix`
3. Task key: `raven_spatial_transform_label_query`
4. Query key: `raven_spatial_transform_label`
5. Prompt wording may identify the broad spatial-arrangement family, but must not reveal the concrete rotation direction or operator.
