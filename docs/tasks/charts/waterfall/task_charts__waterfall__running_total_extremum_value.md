# `task_charts__waterfall__running_total_extremum_value`

## Contract
1. Domain: `charts`
2. Scene id: `waterfall`
3. Source implementation: `src/trace_tasks/tasks/charts/waterfall/running_total_extremum_value.py`
4. Public query ids: `maximum_running_total`, `minimum_running_total`
5. Prompt query key: `running_total_extremum_value`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.waterfall.running_total_extremum_value.ChartsWaterfallRunningTotalExtremumValueTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/waterfall/charts_waterfall_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `bbox`.
3. The annotation marks the full start or contribution bar where the requested cumulative extremum is reached.
4. The summary final bar is not a candidate because it duplicates the last contribution's running total.

## Program Contract

Program: `extremum(running_total(start_and_contribution_bars), direction); direction={maximum,minimum}; output=integer_value; annotation=bbox(extremum_bar); scene=waterfall; scope=running_total_extremum_value`

Candidate set: the visible waterfall bars, contribution steps, and total bars inside the `running_total_extremum_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `extremum` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `bbox` witnesses bound by `bbox(extremum_bar)`. The annotation marks the full start or contribution bar where the requested cumulative extremum is reached. The summary final bar is not a candidate because it duplicates the last contribution's running total.
Query ids: `maximum_running_total`, `minimum_running_total`.

## Reasoning Operations

Families: `ranking`, `aggregation`

## Query Details

| Query id | Program contract | Answer schema | Annotation schema |
|---|---|---|---|
| `maximum_running_total` | `max(running_total(start_and_contribution_bars)); output=integer_value; annotation=bbox(extremum_bar); scene=waterfall; scope=running_total_extremum_value` | `integer_value` | `bbox` |
| `minimum_running_total` | `min(running_total(start_and_contribution_bars)); output=integer_value; annotation=bbox(extremum_bar); scene=waterfall; scope=running_total_extremum_value` | `integer_value` | `bbox` |
