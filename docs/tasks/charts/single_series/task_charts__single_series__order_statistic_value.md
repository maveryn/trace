# `task_charts__single_series__order_statistic_value`

## Contract
1. Domain: `charts`
2. Scene id: `single_series`
3. Source implementation: `src/trace_tasks/tasks/charts/single_series/order_statistic_value.py`
4. Public task id: `task_charts__single_series__order_statistic_value`
5. Supported `query_id` values: `median_order_statistic_value`, `nth_highest_order_statistic_value`, `nth_lowest_order_statistic_value`
6. Query ids are internal replay metadata; scene style, label pool, mark count, and context mode are generation metadata.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.single_series.order_statistic_value.ChartsSingleSeriesOrderStatisticValueTask`
2. Prompt lookup: `src/trace_tasks/resources/prompts/charts/single_series/charts_statistics_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Program Contract

Program: `value(select_ranked_mark(marks, statistic_kind)); output=integer_value; annotation=point(selected_mark); scene=single_series; scope=order_statistic_value`

Candidate set: the visible marks in the ordered single-series chart inside the `order_statistic_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `value` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point` witnesses bound by `point(selected_mark)`. Annotation marks the selected ranked-statistic mark point only. Axes, legend, titles, captions, decorative context, and distractor text are context unless the task explicitly asks for them as annotation.
Query ids: `median_order_statistic_value`, `nth_highest_order_statistic_value`, `nth_lowest_order_statistic_value`.

## Reasoning Operations

Families: `ranking`

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `point`.
3. Annotation marks the selected ranked-statistic mark point only.
4. Axes, legend, titles, captions, decorative context, and distractor text are context unless the task explicitly asks for them as annotation.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `median_order_statistic_value` | `value.selected_median_mark` | `integer_value` | `point` |
| `nth_highest_order_statistic_value` | `value.selected_nth_highest_mark` | `integer_value` | `point` |
| `nth_lowest_order_statistic_value` | `value.selected_nth_lowest_mark` | `integer_value` | `point` |
