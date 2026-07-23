# `task_charts__composition_panels__top_k_by_segment_then_sum_other_segment_count`

## Contract
1. Domain: `charts`
2. Scene id: `composition_panels`
3. Public task id: `task_charts__composition_panels__top_k_by_segment_then_sum_other_segment_count`
4. Query id: `single`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.composition_panels.top_k_by_segment_then_sum_other_segment_count.ChartsCompositionPanelsTopKBySegmentThenSumOtherSegmentCountTask`
2. Prompt bundle: `charts_composition_panels_v1`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `bbox_set`.
3. Annotation marks the full selected panels included in the sum.

## Program Contract

Program: `sum(count(panel,target_segment) for panel in top_k(panels, share(panel,rank_segment), k)); output=integer_count; annotation=bbox_set(selected_panels); scene=composition_panels; scope=top_k_by_segment_then_sum_other_segment_count`

Candidate set: the visible panel segments and panel/category labels inside the `top_k_by_segment_then_sum_other_segment_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `sum` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `bbox_set` witnesses bound by `bbox_set(selected_panels)`. Annotation marks the full selected panels included in the sum.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `ranking`, `aggregation`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `sum(count(panel,target_segment) for panel in top_k(panels, share(panel,rank_segment), k))` | `integer_count` | `bbox_set` |
