# `task_charts__matrix__axis_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `matrix`
3. Source implementation scene: `charts/matrix`
4. Supported `query_id` values: `row_highest_axis_extremum_label`, `row_lowest_axis_extremum_label`, `column_highest_axis_extremum_label`, `column_lowest_axis_extremum_label`
5. Query ids bind the prompt-visible row/column axis and highest/lowest direction. The selected row/column label and numeric matrix values are sampled generation metadata.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.matrix.axis_extremum_label.ChartsMatrixAxisExtremumLabelTask`
2. Prompt lookup domain/scene: `charts/matrix`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `bbox_set`.
3. Annotation marks all active candidate cells in the selected row or column.
4. If the answer is `unanswerable`, annotation is an empty `bbox_set`.
5. Matrix headers, legends, titles, and distractor text are context unless the task explicitly asks for them as annotation.

## Program Contract

Program: `select_label(arg_ranked_extreme(line_cells(axis={row,column}, axis_label), value(cell), rank=2, direction={highest,lowest})); output=string_label|unanswerable; annotation=bbox_set(candidate_line_cells); scene=matrix; scope=axis_extremum_label`

Candidate set: the visible matrix cells with row and column labels inside the `axis_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `select_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label|unanswerable`.
Annotation witnesses: `bbox_set` witnesses bound by `bbox_set(candidate_line_cells)`. Annotation marks all active candidate cells in the selected row or column. If the answer is `unanswerable`, annotation is an empty `bbox_set`. Matrix headers, legends, titles, and distractor text are context unless the task explicitly asks for them as annotation.
Query ids: `row_highest_axis_extremum_label`, `row_lowest_axis_extremum_label`, `column_highest_axis_extremum_label`, `column_lowest_axis_extremum_label`.

## Reasoning Operations

Families: `ranking`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `row_highest_axis_extremum_label` | `selection.matrix_axis_ranked_extreme(axis=row,direction=highest)` | `string_label` | `bbox_set` |
| `row_lowest_axis_extremum_label` | `selection.matrix_axis_ranked_extreme(axis=row,direction=lowest)` | `string_label` | `bbox_set` |
| `column_highest_axis_extremum_label` | `selection.matrix_axis_ranked_extreme(axis=column,direction=highest)` | `string_label` | `bbox_set` |
| `column_lowest_axis_extremum_label` | `selection.matrix_axis_ranked_extreme(axis=column,direction=lowest)` | `string_label` | `bbox_set` |
