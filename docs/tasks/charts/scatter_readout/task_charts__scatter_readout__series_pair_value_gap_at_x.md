# `task_charts__scatter_readout__series_pair_value_gap_at_x`

## Contract
1. Domain: `charts`
2. Scene id: `scatter_readout`
3. Public task id: `task_charts__scatter_readout__series_pair_value_gap_at_x`
4. Supported `query_id` values: `single`
5. The task always computes an absolute y-value gap between two visible series at one resolved x-axis label.

## Program Contract

Program: `abs(value(series_a,x_label)-value(series_b,x_label)); output=integer_value; annotation=segment(series_a_mark,series_b_mark); scene=scatter_readout; scope=series_pair_value_gap_at_x`

Candidate set: the visible scatter points, readout markers, and axis labels inside the `series_pair_value_gap_at_x` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `abs` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `segment` witnesses bound by `segment(series_a_mark,series_b_mark)`. Annotation is one segment connecting the centers of the two compared scatter marks at the requested x-axis label. Axes, legends, titles, readout numbers, and distractor text are metadata.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.scatter_readout.series_pair_value_gap_at_x.ChartsScatterSeriesPairValueGapAtXTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/scatter_readout/charts_scatter_readout_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `segment`.
3. Annotation is one segment connecting the centers of the two compared scatter marks at the requested x-axis label.
4. Axes, legends, titles, readout numbers, and distractor text are metadata.

## Query Details

| Query id | Program arguments | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `operation=absolute_difference` | `integer_value` | `segment` |
