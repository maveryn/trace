# `task_charts__single_series__interval_rate_value`

## Contract
1. Domain: `charts`
2. Scene id: `single_series`
3. Source implementation: `src/trace_tasks/tasks/charts/single_series/interval_rate_value.py`
4. Public task id: `task_charts__single_series__interval_rate_value`
5. Supported `query_id` values: `single`
6. Query ids are internal replay metadata; scene style, label pool, mark count, and context mode are generation metadata.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.single_series.interval_rate_value.ChartsTrendIntervalRateValueTask`
2. Prompt lookup: `src/trace_tasks/resources/prompts/charts/single_series/charts_trend_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Program Contract

Program: `abs(quotient(value(end_mark)-value(start_mark), step_count(start_mark,end_mark))); output=integer_value; annotation=point_map(start_mark,end_mark); scene=single_series; scope=interval_rate_value`

Candidate set: the visible marks in the ordered single-series chart inside the `interval_rate_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `abs` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point_map` witnesses bound by `point_map(start_mark,end_mark)`. Annotation maps `start_mark` and `end_mark` to the two endpoint mark points named by the prompt. Axes, legend, titles, captions, decorative context, and distractor text are context unless the task explicitly asks for them as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `formula_evaluation`

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `point_map`.
3. Annotation maps `start_mark` and `end_mark` to the two endpoint mark points named by the prompt.
4. Axes, legend, titles, captions, decorative context, and distractor text are context unless the task explicitly asks for them as annotation.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `absolute_quotient.endpoint_delta_by_step_count` | `integer_value` | `point_map` |
