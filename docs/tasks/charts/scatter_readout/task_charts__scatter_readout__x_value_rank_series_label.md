# `task_charts__scatter_readout__x_value_rank_series_label`

## Contract
1. Domain: `charts`
2. Scene id: `scatter_readout`
3. Public task id: `task_charts__scatter_readout__x_value_rank_series_label`
4. Supported `query_id` values: `x_highest_series_label`, `x_lowest_series_label`
5. The query id changes the visible extremum direction in the prompt and the program argument.

## Program Contract

Program: `select_label(arg_extreme(series, value(series,x_label), direction)); output=string_label; annotation=point(target_mark); scene=scatter_readout; scope=x_value_rank_series_label`

Candidate set: the visible scatter points, readout markers, and axis labels inside the `x_value_rank_series_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `select_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point` witnesses bound by `point(target_mark)`. Annotation is one [x,y] pixel point at the center of the answer series scatter mark at the requested x-axis label. Axes, legends, titles, readout numbers, and distractor text are metadata.
Query ids: `x_highest_series_label`, `x_lowest_series_label`.

## Reasoning Operations

Families: `ranking`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.scatter_readout.x_value_rank_series_label.ChartsScatterXValueRankSeriesLabelTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/scatter_readout/charts_scatter_readout_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point`.
3. Annotation is one [x,y] pixel point at the center of the answer series scatter mark at the requested x-axis label.
4. Axes, legends, titles, readout numbers, and distractor text are metadata.

## Query Details

| Query id | Program arguments | Answer schema | Annotation schema |
|---|---|---|---|
| `x_highest_series_label` | `direction=highest` | `string_label` | `point` |
| `x_lowest_series_label` | `direction=lowest` | `string_label` | `point` |
