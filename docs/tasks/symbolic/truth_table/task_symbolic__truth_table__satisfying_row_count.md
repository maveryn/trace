# `task_symbolic__truth_table__satisfying_row_count`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `truth_table`
3. Scene: `truth_table`
4. Task id: `task_symbolic__truth_table__satisfying_row_count`
5. Objective: count the rows where the displayed Boolean expression is true.

## Program Contract

Program: `truth_table.satisfying_row_count(scene=truth_table, scope=three_variable_input_rows_plus_expression, output=integer)`

Candidate set: the visible symbolic notation, tokens, rows, columns, cards, labels, components, and target markers inside the `three_variable_input_rows_plus_expression` objective scope.

Operands: the visible `A`, `B`, `C` input rows and the displayed expression `P`.
Operation: evaluate expression `P` for each row and count rows where the result is `1`.
Output binding: `answer` is the integer count.
Annotation witnesses: a `bbox_set` containing full table-row boxes for rows where expression `P` evaluates to `1`.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`, `formula_evaluation`

## 2) Scene + Task Contract
1. The table uses three variables `A`, `B`, and `C`.
2. Row order is fixed from `000` through `111`.
3. Values use `1` for true and `0` for false.
4. The `P` column cells are blank; the model must evaluate the expression from the row inputs rather than count visible output values.
5. Expression headers use `!`, `&`, `|`, and `^` for NOT, AND, OR, and XOR.
6. Count support is constrained to `1..7`, so the public answer is not binary-only.
7. `answer_gt.type`: `integer`
8. `annotation_gt.type`: `bbox_set`
9. Annotation schema: `bbox_set`

## 3) Prompt Contract
1. Bundle: `symbolic_truth_table_v1`
2. `scene_key`: `truth_table`
3. `task_key`: `satisfying_row_count`
4. Modes: `answer_only`, `answer_and_annotation`
5. Prompt text comes from the external prompt bundle.

## 4) Trace Contract
1. `execution_trace.truth_table_metadata.expression_id` records the sampled expression.
2. `execution_trace.truth_table_metadata.truth_pattern` records the top-to-bottom output pattern.
3. `render_map.row_bboxes_px` contains row-level projections used for annotation, and `render_map.cell_bboxes_px` contains the individual cell projections.
4. `projected_annotation` mirrors the prompt-facing `bbox_set`.

## 5) Determinism + Tests
1. Deterministic generation and rendering from `instance_seed`.
2. Answers and annotation come from the same finalized table.
3. Behavior tests: `tests/test_symbolic_truth_table_tasks.py`
