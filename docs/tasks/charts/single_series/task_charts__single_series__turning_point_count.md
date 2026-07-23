# `task_charts__single_series__turning_point_count`

## Contract
1. Domain: `charts`
2. Scene id: `single_series`
3. Source implementation: `src/trace_tasks/tasks/charts/single_series/turning_point_count.py`
4. Public task id: `task_charts__single_series__turning_point_count`
5. Supported `query_id` values: `peak_turning_point_count`, `trough_turning_point_count`
6. Query ids are internal replay metadata; scene style, label pool, mark count, and context mode are generation metadata.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.single_series.turning_point_count.ChartsTrendTurningPointCountTask`
2. Prompt lookup: `src/trace_tasks/resources/prompts/charts/single_series/charts_trend_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Program Contract

Program: `count(turning_points(sequence(values), turning_kind)); output=integer_count; annotation=point_set(turning_points); scene=single_series; scope=turning_point_count`

Candidate set: the visible marks in the ordered single-series chart inside the `turning_point_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `point_set` witnesses bound by `point_set(turning_points)`. Annotation marks every local peak or trough point counted by the selected query. Axes, legend, titles, captions, decorative context, and distractor text are context unless the task explicitly asks for them as annotation.
Query ids: `peak_turning_point_count`, `trough_turning_point_count`.

## Reasoning Operations

Families: `counting`

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `point_set`.
3. Annotation marks every local peak or trough point counted by the selected query.
4. Axes, legend, titles, captions, decorative context, and distractor text are context unless the task explicitly asks for them as annotation.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `peak_turning_point_count` | `count.local_peaks` | `integer_count` | `point_set` |
| `trough_turning_point_count` | `count.local_troughs` | `integer_count` | `point_set` |
