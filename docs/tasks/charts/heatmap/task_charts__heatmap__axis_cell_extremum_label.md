# `task_charts__heatmap__axis_cell_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `heatmap`
3. Source implementation: `src/trace_tasks/tasks/charts/heatmap/axis_cell_extremum_label.py`
4. Query ids: `row_hottest_column_label`, `row_coolest_column_label`, `column_hottest_row_label`, `column_coolest_row_label`
5. Query ids bind prompt-visible axis and extremum direction; sampled label names and heatmap style are generation metadata.
6. Controlled-unanswerable sampling is disabled because the task has a scalar `bbox` annotation contract.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.heatmap.axis_cell_extremum_label.ChartsHeatmapAxisCellExtremumLabelTask`
2. Prompt lookup: `src/trace_tasks/resources/prompts/charts/heatmap/charts_heatmap_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Program Contract

Program: `arg_extreme(opposite_axis_label, heat_value(cell(named_axis_label, opposite_axis_label)), direction={hottest,coolest}); output=string_label; annotation=bbox(selected_target_cell); scene=heatmap; scope=axis_cell_extremum_label`

Candidate set: the visible heatmap cells with row and column labels inside the `axis_cell_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `bbox` witnesses bound by `bbox(selected_target_cell)`. Annotation marks the rendered bbox of the selected target cell only. Axes, legend, title, and distractor text are context unless the task explicitly asks for them as annotation. The selected target cell is guaranteed answerable and visible.
Query ids: `row_hottest_column_label`, `row_coolest_column_label`, `column_hottest_row_label`, `column_coolest_row_label`.

## Reasoning Operations

Families: `ranking`

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `bbox`.
3. Annotation marks the rendered bbox of the selected target cell only.
4. Axes, legend, title, and distractor text are context unless the task explicitly asks for them as annotation.
5. The selected target cell is guaranteed answerable and visible.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `row_hottest_column_label` | `arg_extreme.cell_value_on_named_axis` | `string_label` | `bbox` |
| `row_coolest_column_label` | `arg_extreme.cell_value_on_named_axis` | `string_label` | `bbox` |
| `column_hottest_row_label` | `arg_extreme.cell_value_on_named_axis` | `string_label` | `bbox` |
| `column_coolest_row_label` | `arg_extreme.cell_value_on_named_axis` | `string_label` | `bbox` |
