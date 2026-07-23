# `task_charts__heatmap__axis_condition_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `heatmap`
3. Source implementation: `src/trace_tasks/tasks/charts/heatmap/axis_condition_extremum_label.py`
4. Query ids: `row_condition_extremum_label`, `column_condition_extremum_label`
5. Query ids bind the prompt-visible axis being selected; sampled condition kind and heatmap style are generation metadata.
6. Controlled-unanswerable sampling is disabled for this task.
7. Condition kinds match only the single extremal color level: highest/lowest intensity, highest/lowest activity, or strongest increase/decrease.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.heatmap.axis_condition_extremum_label.ChartsHeatmapAxisConditionExtremumLabelTask`
2. Prompt lookup: `src/trace_tasks/resources/prompts/charts/heatmap/charts_heatmap_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Program Contract

Program: `arg_extreme(axis_label, count(filter(cells_on_axis, exact_extremal_condition_kind)), direction=largest); output=string_label; annotation=bbox_set(matching_cells_in_winning_axis); scene=heatmap; scope=axis_condition_extremum_label`

Candidate set: the visible heatmap cells with row and column labels inside the `axis_condition_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `bbox_set` witnesses bound by `bbox_set(matching_cells_in_winning_axis)`. Annotation marks rendered bboxes for matching cells in the winning row or column. Axes, legend, title, and distractor text are context unless the task explicitly asks for them as annotation.
Query ids: `row_condition_extremum_label`, `column_condition_extremum_label`.

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `bbox_set`.
3. Annotation marks rendered bboxes for matching cells in the winning row or column.
4. Axes, legend, title, and distractor text are context unless the task explicitly asks for them as annotation.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `row_condition_extremum_label` | `arg_extreme.axis_exact_condition_count` | `string_label` | `bbox_set` |
| `column_condition_extremum_label` | `arg_extreme.axis_exact_condition_count` | `string_label` | `bbox_set` |
