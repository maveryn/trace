# `task_charts__single_series__interval_value_count`

## Contract
1. Domain: `charts`
2. Scene id: `single_series`
3. Source implementation: `src/trace_tasks/tasks/charts/single_series/interval_value_count.py`
4. Public task id: `task_charts__single_series__interval_value_count`
5. Supported `query_id` values: `single`
6. Query ids are internal replay metadata; scene style, label pool, mark count, and context mode are generation metadata.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.single_series.interval_value_count.ChartsCountingIntervalValueCountTask`
2. Prompt lookup: `src/trace_tasks/resources/prompts/charts/single_series/charts_counting_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Program Contract

Program: `count(filter(marks, lower_bound <= value(mark) <= upper_bound)); output=integer_count; annotation=point_set(matching_marks); scene=single_series; scope=interval_value_count`

Candidate set: the visible marks in the ordered single-series chart inside the `interval_value_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `point_set` witnesses bound by `point_set(matching_marks)`. Annotation marks every visible mark whose value falls inside the inclusive interval. Axes, legend, titles, captions, decorative context, and distractor text are context unless the task explicitly asks for them as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `point_set`.
3. Annotation marks every visible mark whose value falls inside the inclusive interval.
4. Axes, legend, titles, captions, decorative context, and distractor text are context unless the task explicitly asks for them as annotation.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `count.inclusive_value_interval` | `integer_count` | `point_set` |
