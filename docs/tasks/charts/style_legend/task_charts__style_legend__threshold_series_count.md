# `task_charts__style_legend__threshold_series_count`

## Contract
1. Domain: `charts`
2. Scene id: `style_legend`
3. Public task id: `task_charts__style_legend__threshold_series_count`
4. Supported `query_id`: `above_threshold_series_count`, `below_threshold_series_count`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.style_legend.threshold_series_count.ChartsStyleLegendThresholdSeriesCountTask`
2. Prompt bundle: `charts_style_legend_v1`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `point_set`.
3. Annotation marks each counted plotted marker at the queried x-axis label; use an empty array when the count is zero.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `count(filter(series, compare(value(series, x_position), threshold, comparator={above,below}))); output=integer_count; annotation=point_set; scene=style_legend; scope=threshold_series_count`

Candidate set: the visible plotted series/marks and legend entries inside the `threshold_series_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `point_set` witnesses bound by `point_set`. Annotation marks each counted plotted marker at the queried x-axis label; use an empty array when the count is zero. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `above_threshold_series_count`, `below_threshold_series_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `above_threshold_series_count` | `count(series where value(series, x_position) > threshold)` | `integer_count` | `point_set` |
| `below_threshold_series_count` | `count(series where value(series, x_position) < threshold)` | `integer_count` | `point_set` |
