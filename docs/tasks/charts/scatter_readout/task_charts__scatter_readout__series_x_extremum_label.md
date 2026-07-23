# `task_charts__scatter_readout__series_x_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `scatter_readout`
3. Public task id: `task_charts__scatter_readout__series_x_extremum_label`
4. Supported `query_id` values: `series_highest_x_label`, `series_lowest_x_label`
5. The query id changes the visible extremum direction in the prompt and the program argument.

## Program Contract

Program: `select_label(x_label(arg_extreme(point in series, y_value(point), direction))); output=string_label_or_unanswerable; annotation=point(target_mark)|empty_map; scene=scatter_readout; scope=series_x_extremum_label`

Candidate set: the visible scatter points, readout markers, and axis labels inside the `series_x_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `select_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label_or_unanswerable` value bound by `string_label_or_unanswerable`.
Annotation witnesses: `scalar` witnesses bound by `point(target_mark)|empty_map`. Answerable annotation is one [x,y] pixel point at the center of the selected scatter mark. If the sampled branch is unanswerable because the requested series is absent from the legend, annotation is an empty object. Axes, legends, titles, readout numbers, and distractor text are metadata.
Query ids: `series_highest_x_label`, `series_lowest_x_label`.

## Reasoning Operations

Families: `ranking`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.scatter_readout.series_x_extremum_label.ChartsScatterSeriesExtremumXLabelTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/scatter_readout/charts_scatter_readout_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label_or_unanswerable`.
2. Annotation schema: scalar `point` for answerable samples; empty map for the controlled-unanswerable branch.
3. Answerable annotation is one [x,y] pixel point at the center of the selected scatter mark.
4. If the sampled branch is unanswerable because the requested series is absent from the legend, annotation is an empty object.
5. Axes, legends, titles, readout numbers, and distractor text are metadata.

## Query Details

| Query id | Program arguments | Answer schema | Annotation schema |
|---|---|---|---|
| `series_highest_x_label` | `direction=highest` | `string_label_or_unanswerable` | `point` or empty map |
| `series_lowest_x_label` | `direction=lowest` | `string_label_or_unanswerable` | `point` or empty map |
