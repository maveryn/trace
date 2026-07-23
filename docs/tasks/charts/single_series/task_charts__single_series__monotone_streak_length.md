# `task_charts__single_series__monotone_streak_length`

## Contract
1. Domain: `charts`
2. Scene id: `single_series`
3. Source implementation: `src/trace_tasks/tasks/charts/single_series/monotone_streak_length.py`
4. Public task id: `task_charts__single_series__monotone_streak_length`
5. Supported `query_id` values: `longest_increasing_streak_length`, `longest_decreasing_streak_length`
6. Query ids are internal replay metadata; scene style, label pool, mark count, and context mode are generation metadata.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.single_series.monotone_streak_length.ChartsTrendMonotoneStreakLengthTask`
2. Prompt lookup: `src/trace_tasks/resources/prompts/charts/single_series/charts_trend_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Program Contract

Program: `length(unique_longest_monotone_run(sequence(values), direction)); output=integer_value; annotation=point_set(run_marks); scene=single_series; scope=monotone_streak_length`

Candidate set: the visible marks in the ordered single-series chart inside the `monotone_streak_length` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `length` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point_set` witnesses bound by `point_set(run_marks)`. Annotation marks every visible mark in the unique longest monotone run. Axes, legend, titles, captions, decorative context, and distractor text are context unless the task explicitly asks for them as annotation.
Query ids: `longest_increasing_streak_length`, `longest_decreasing_streak_length`.

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `point_set`.
3. Annotation marks every visible mark in the unique longest monotone run.
4. Axes, legend, titles, captions, decorative context, and distractor text are context unless the task explicitly asks for them as annotation.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `longest_increasing_streak_length` | `length.unique_longest_increasing_run` | `integer_value` | `point_set` |
| `longest_decreasing_streak_length` | `length.unique_longest_decreasing_run` | `integer_value` | `point_set` |
