# `task_charts__matrix__threshold_cell_count`

## Contract
1. Domain: `charts`
2. Scene id: `matrix`
3. Source implementation scene: `charts/matrix`
4. Supported `query_id` values: `row_at_least_threshold_cell_count`, `row_at_most_threshold_cell_count`, `column_at_least_threshold_cell_count`, `column_at_most_threshold_cell_count`
5. Query ids bind the prompt-visible row/column axis and at-least/at-most comparison. The threshold value and selected row/column label are sampled generation metadata.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.matrix.threshold_cell_count.ChartsMatrixThresholdCellCountTask`
2. Prompt lookup domain/scene: `charts/matrix`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `bbox_set`.
3. Annotation marks exactly the cells in the selected row or column that satisfy the threshold condition.
4. Matrix headers, legends, titles, and distractor text are context unless the task explicitly asks for them as annotation.

## Program Contract

Program: `count(filter(line_cells(axis={row,column}, axis_label), compare(value(cell), threshold, relation={at_least,at_most}))); output=integer_count; annotation=bbox_set(counted_cells); scene=matrix; scope=threshold_cell_count`

Candidate set: the visible matrix cells with row and column labels inside the `threshold_cell_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `bbox_set` witnesses bound by `bbox_set(counted_cells)`. Annotation marks exactly the cells in the selected row or column that satisfy the threshold condition. Matrix headers, legends, titles, and distractor text are context unless the task explicitly asks for them as annotation.
Query ids: `row_at_least_threshold_cell_count`, `row_at_most_threshold_cell_count`, `column_at_least_threshold_cell_count`, `column_at_most_threshold_cell_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `row_at_least_threshold_cell_count` | `count.matrix_axis_threshold(axis=row,relation=at_least)` | `integer_count` | `bbox_set` |
| `row_at_most_threshold_cell_count` | `count.matrix_axis_threshold(axis=row,relation=at_most)` | `integer_count` | `bbox_set` |
| `column_at_least_threshold_cell_count` | `count.matrix_axis_threshold(axis=column,relation=at_least)` | `integer_count` | `bbox_set` |
| `column_at_most_threshold_cell_count` | `count.matrix_axis_threshold(axis=column,relation=at_most)` | `integer_count` | `bbox_set` |
