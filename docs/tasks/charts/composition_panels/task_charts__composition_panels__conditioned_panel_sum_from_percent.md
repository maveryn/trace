# `task_charts__composition_panels__conditioned_panel_sum_from_percent`

## Contract
1. Domain: `charts`
2. Scene id: `composition_panels`
3. Public task id: `task_charts__composition_panels__conditioned_panel_sum_from_percent`
4. Query id: `single`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.composition_panels.conditioned_panel_sum_from_percent.ChartsCompositionPanelsConditionedPanelSumFromPercentTask`
2. Prompt bundle: `charts_composition_panels_v1`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `bbox_set`.
3. Annotation marks the full panels included in the sum.

## Program Contract

Program: `sum(count(panel,target_segment) for panel in filter(panels, share(panel,condition_segment) > threshold)); output=integer_value; annotation=bbox_set(selected_panels); scene=composition_panels; scope=conditioned_panel_sum_from_percent`

Candidate set: the visible panel segments and panel/category labels inside the `conditioned_panel_sum_from_percent` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `sum` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `bbox_set` witnesses bound by `bbox_set(selected_panels)`. Annotation marks the full panels included in the sum.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`, `aggregation`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `sum(count(panel,target_segment) for panel in filter(panels, share(panel,condition_segment) > threshold))` | `integer_value` | `bbox_set` |
