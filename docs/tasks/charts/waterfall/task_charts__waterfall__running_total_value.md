# `task_charts__waterfall__running_total_value`

## Contract
1. Domain: `charts`
2. Scene id: `waterfall`
3. Source implementation: `src/trace_tasks/tasks/charts/waterfall/running_total_value.py`
4. Public query id: `single`
5. Prompt query key: `running_total_after_step`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.waterfall.running_total_value.ChartsWaterfallRunningTotalValueTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/waterfall/charts_waterfall_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `bbox_set`.
3. The annotation contains full waterfall bar boxes from the start bar through the requested step, including the target contribution bar.

## Program Contract

Program: `sum(start_value, signed_deltas_through(target_step)); target_step=visible_step_label; output=integer_value; annotation=bbox_set(running_bars_through_target); scene=waterfall; scope=running_total_value`

Candidate set: the visible waterfall bars, contribution steps, and total bars inside the `running_total_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `sum` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `bbox_set` witnesses bound by `bbox_set(running_bars_through_target)`. The annotation contains full waterfall bar boxes from the start bar through the requested step, including the target contribution bar.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`

## Query Details

| Query id | Program contract | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `sum(start_value, signed_deltas_through(target_step)); target_step=visible_step_label; output=integer_value; annotation=bbox_set(running_bars_through_target); scene=waterfall; scope=running_total_value` | `integer_value` | `bbox_set` |
