# `task_charts__style_legend__x_position_extremum_series_label`

## Contract
1. Domain: `charts`
2. Scene id: `style_legend`
3. Public task id: `task_charts__style_legend__x_position_extremum_series_label`
4. Supported `query_id`: `x_position_highest_series_label`, `x_position_lowest_series_label`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.style_legend.x_position_extremum_series_label.ChartsStyleLegendXPositionExtremumSeriesLabelTask`
2. Prompt bundle: `charts_style_legend_v1`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point`.
3. Annotation marks the selected plotted marker at the queried x-axis label.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `arg_extremum(series, value(series, x_position), direction={highest,lowest}); output=string_label; annotation=point; scene=style_legend; scope=x_position_extremum_series_label`

Candidate set: the visible plotted series/marks and legend entries inside the `x_position_extremum_series_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extremum` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point` witnesses bound by `point`. Annotation marks the selected plotted marker at the queried x-axis label. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `x_position_highest_series_label`, `x_position_lowest_series_label`.

## Reasoning Operations

Families: `ranking`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `x_position_highest_series_label` | `argmax(series, value(series, x_position))` | `string_label` | `point` |
| `x_position_lowest_series_label` | `argmin(series, value(series, x_position))` | `string_label` | `point` |
