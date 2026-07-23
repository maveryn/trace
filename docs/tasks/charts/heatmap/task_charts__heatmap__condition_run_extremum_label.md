# `task_charts__heatmap__condition_run_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `heatmap`
3. Source implementation: `src/trace_tasks/tasks/charts/heatmap/condition_run_extremum_label.py`
4. Query id: `single`
5. Condition kind is a sampled generation parameter; it does not change the public query branch.
6. Condition kinds match only the single extremal color level: highest/lowest intensity, highest/lowest activity, or strongest increase/decrease.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.heatmap.condition_run_extremum_label.ChartsHeatmapConditionRunExtremumLabelTask`
2. Prompt lookup: `src/trace_tasks/resources/prompts/charts/heatmap/charts_heatmap_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Program Contract

Program: `arg_extreme(row_label, longest_run(filter(row_cells, exact_extremal_condition_kind)), direction=largest); output=string_label; annotation=bbox_set(winning_consecutive_run_cells); scene=heatmap; scope=condition_run_extremum_label`

Candidate set: the visible heatmap cells with row and column labels inside the `condition_run_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `bbox_set` witnesses bound by `bbox_set(winning_consecutive_run_cells)`. Annotation marks rendered bboxes for the winning consecutive run cells from left to right. Axes, legend, title, and distractor text are context unless the task explicitly asks for them as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `ranking`

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `bbox_set`.
3. Annotation marks rendered bboxes for the winning consecutive run cells from left to right.
4. Axes, legend, title, and distractor text are context unless the task explicitly asks for them as annotation.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `arg_extreme.longest_exact_condition_run` | `string_label` | `bbox_set` |
