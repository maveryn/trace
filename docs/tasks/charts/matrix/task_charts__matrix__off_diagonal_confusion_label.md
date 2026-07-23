# `task_charts__matrix__off_diagonal_confusion_label`

## Contract
1. Domain: `charts`
2. Scene id: `matrix`
3. Source implementation scene: `charts/matrix`
4. Supported `query_id` values: `single`
5. The task always asks for the largest off-diagonal predicted column for one actual-class row.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.matrix.off_diagonal_confusion_label.ChartsMatrixOffDiagonalConfusionLabelTask`
2. Prompt lookup domain/scene: `charts/matrix`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `bbox_set`.
3. Annotation marks the off-diagonal candidate cells in the selected actual-class row.
4. Matrix headers, diagonal cells, legends, titles, and distractor text are context unless the task explicitly asks for them as annotation.

## Program Contract

Program: `select_label(argmax(filter(row_cells(actual_label), column_label != actual_label), value(cell))); output=string_label; annotation=bbox_set(off_diagonal_candidate_cells); scene=matrix; scope=off_diagonal_confusion_label`

Candidate set: the visible matrix cells with row and column labels inside the `off_diagonal_confusion_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `select_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `bbox_set` witnesses bound by `bbox_set(off_diagonal_candidate_cells)`. Annotation marks the off-diagonal candidate cells in the selected actual-class row. Matrix headers, diagonal cells, legends, titles, and distractor text are context unless the task explicitly asks for them as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `ranking`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `selection.matrix_off_diagonal_argmax` | `string_label` | `bbox_set` |
