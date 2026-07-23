# `task_charts__waterfall__reverse_step_final_total`

## Contract
1. Domain: `charts`
2. Scene id: `waterfall`
3. Source implementation: `src/trace_tasks/tasks/charts/waterfall/reverse_step_final_total.py`
4. Public query id: `single`
5. Prompt query key: `reverse_step_final_total`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.waterfall.reverse_step_final_total.ChartsWaterfallReverseStepFinalTotalTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/waterfall/charts_waterfall_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.
5. Sampling constrains the target contribution magnitude and counterfactual final total so the reversed total remains inside the visible chart scale.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `bbox_map`.
3. `final_total_bar` marks the full final-total bar.
4. `target_contribution_bar` marks the full contribution bar whose sign is reversed.

## Program Contract

Program: `final_total - 2 * delta(target_step); target_step=visible_step_label; output=integer_value; annotation=bbox_map(final_total_bar,target_contribution_bar); scene=waterfall; scope=reverse_step_final_total`

Candidate set: the visible waterfall bars, contribution steps, and total bars inside the `reverse_step_final_total` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `final_total - 2 * delta` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `bbox_map` witnesses bound by `bbox_map(final_total_bar,target_contribution_bar)`. `final_total_bar` marks the full final-total bar. `target_contribution_bar` marks the full contribution bar whose sign is reversed.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `state_update`, `formula_evaluation`

## Query Details

| Query id | Program contract | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `final_total - 2 * delta(target_step); target_step=visible_step_label; output=integer_value; annotation=bbox_map(final_total_bar,target_contribution_bar); scene=waterfall; scope=reverse_step_final_total` | `integer_value` | `bbox_map` |
