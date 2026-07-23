# `task_charts__scatter_readout__series_value_at_x_value`

## Contract
1. Domain: `charts`
2. Scene id: `scatter_readout`
3. Public task id: `task_charts__scatter_readout__series_value_at_x_value`
4. Supported `query_id` values: `single`
5. The task reads one visible series value at one visible x-axis label.

## Program Contract

Program: `value(series, x_label); output=integer_value; annotation=point(target_mark); scene=scatter_readout; scope=series_value_at_x_value`

Candidate set: the visible scatter points, readout markers, and axis labels inside the `series_value_at_x_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `value` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point` witnesses bound by `point(target_mark)`. Annotation is one [x,y] pixel point at the center of the selected scatter mark. Axes, legends, titles, readout numbers, and distractor text are metadata.
Query ids: `single`.

## Reasoning Operations

Families: `direct_retrieval`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.scatter_readout.series_value_at_x_value.ChartsScatterSeriesValueAtXValueTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/scatter_readout/charts_scatter_readout_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `point`.
3. Annotation is one [x,y] pixel point at the center of the selected scatter mark.
4. Axes, legends, titles, readout numbers, and distractor text are metadata.

## Query Details

| Query id | Program arguments | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `operation=direct_value_readout` | `integer_value` | `point` |
