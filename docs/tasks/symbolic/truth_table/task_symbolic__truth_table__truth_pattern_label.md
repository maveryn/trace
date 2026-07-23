# `task_symbolic__truth_table__truth_pattern_label`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `truth_table`
3. Scene: `truth_table`
4. Task id: `task_symbolic__truth_table__truth_pattern_label`
5. Objective: select the output pattern produced by a displayed Boolean expression.

## Program Contract
Program: `truth_table.truth_pattern_label(scene=truth_table, scope=target_expression_plus_pattern_options, output=option_letter)`

Candidate set: the six visible pattern option cards labeled `A..F`.
Operands: the displayed target expression `P`, the canonical input rows, and the candidate output patterns.
Operation: evaluate `P` on each row from top to bottom and select the matching pattern option.
Output binding: `answer` is the selected option letter.
Annotation witnesses: a scalar `bbox` around the selected pattern option card.
Query ids: `single`.

## Reasoning Operations

Families: `logical_composition`, `formula_evaluation`, `matching`

## 2) Scene + Task Contract
1. The input table shows variables `A`, `B`, and `C` in row order `000..111`.
2. Six option cards show eight-value patterns in top-to-bottom row order.
3. Exactly one option card matches the evaluated target expression.
4. Values use `1` for true and `0` for false.
5. `answer_gt.type`: `string`
6. `annotation_gt.type`: `bbox`
7. Annotation schema: `bbox`

## 3) Prompt Contract
1. Bundle: `symbolic_truth_table_v1`
2. `scene_key`: `truth_table`
3. `task_key`: `truth_pattern_label`
4. Modes: `answer_only`, `answer_and_annotation`
5. Prompt text comes from the external prompt bundle.

## 4) Trace Contract
1. `execution_trace.truth_table_metadata.truth_pattern` records the correct pattern.
2. `execution_trace.truth_table_metadata.option_patterns` records the visible option patterns.
3. `render_map.item_bboxes_px` contains option-card projections.
4. `projected_annotation` mirrors the selected option `bbox`.

## 5) Determinism + Tests
1. Deterministic generation and rendering from `instance_seed`.
2. Options are sampled with one unique final answer.
3. Behavior tests: `tests/test_symbolic_truth_table_tasks.py`
