# `task_symbolic__truth_table__expression_from_rows_label`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `truth_table`
3. Scene: `truth_table`
4. Task id: `task_symbolic__truth_table__expression_from_rows_label`
5. Objective: select the Boolean expression that produces the completed output column.

## Program Contract
Program: `truth_table.expression_from_rows_label(scene=truth_table, scope=input_rows_plus_completed_output_column_and_expression_options, output=option_letter)`

Candidate set: the four visible expression option cards labeled `W..Z`.
Operands: the visible `A`, `B`, `C` input rows, the completed `P` output column, and candidate expressions.
Operation: compare each candidate expression's truth values over the shown rows to the completed `P` column.
Output binding: `answer` is the selected candidate label.
Annotation witnesses: a scalar `bbox` around the selected expression option card.
Query ids: `single`.

## Reasoning Operations

Families: `logical_composition`, `formula_evaluation`, `matching`

## 2) Scene + Task Contract
1. The table uses variables `A`, `B`, and `C` with fixed row order `000..111`.
2. The `P` column shows completed truth values for all rows.
3. Candidate labels `W..Z` do not collide with input variables `A`, `B`, and `C`.
4. Exactly one candidate expression has the same truth pattern as column `P`.
5. Values use `1` for true and `0` for false.
6. `answer_gt.type`: `string`
7. `annotation_gt.type`: `bbox`
8. Annotation schema: `bbox`

## 3) Prompt Contract
1. Bundle: `symbolic_truth_table_v1`
2. `scene_key`: `truth_table`
3. `task_key`: `expression_from_rows_label`
4. Modes: `answer_only`, `answer_and_annotation`
5. Prompt text comes from the external prompt bundle.

## 4) Trace Contract
1. `execution_trace.truth_table_metadata.truth_pattern` records the visible `P` column from top to bottom.
2. `execution_trace.truth_table_metadata.candidate_expressions` records each candidate expression and whether it matches the output column.
3. `render_map.cell_bboxes_px` contains table-cell projections.
4. `render_map.item_bboxes_px` contains expression-option card projections.
5. `projected_annotation` mirrors the selected option `bbox`.

## 5) Determinism + Tests
1. Deterministic generation and rendering from `instance_seed`.
2. Options are sampled with one unique final answer.
3. Behavior tests: `tests/test_symbolic_truth_table_tasks.py`
