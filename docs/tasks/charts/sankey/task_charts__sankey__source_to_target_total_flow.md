# `task_charts__sankey__source_to_target_total_flow`

## Contract
1. Domain: `charts`
2. Scene id: `sankey`
3. Task id: `task_charts__sankey__source_to_target_total_flow`
4. Objective contract: `source_to_target_total_flow`
5. Supported `query_id` values: `single`

## Program Contract

Program: `sum(min(value(source_to_middle), value(middle_to_target)) for route in routes(source_label, target_label)); scene=sankey; scope=source_to_target_total_flow`

Candidate set: the visible flow nodes, links, and labels inside the `source_to_target_total_flow` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `sum` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point_set` witnesses bound by `see_annotation_contract`. Annotation marks the set of printed value-label centers for the selected route bottlenecks included in the sum. Node boxes, non-bottleneck labels on selected routes, unselected flow labels, flow curves, title, and panel frame are context unless explicitly referenced by the task.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `aggregation`, `topology`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.sankey.source_to_target_total_flow.ChartsFlowSankeySourceToTargetTotalFlowPublicTask`
2. Prompt bundle: `charts_sankey_v1`
3. Scene key: `sankey`
4. Task key: `sankey_path_query`
5. Query key: `source_to_target_total_flow`

## Annotation Contract
1. Answer schema: `integer_value`
2. Annotation schema: `point_set`
3. Annotation marks the set of printed value-label centers for the selected route bottlenecks included in the sum.
4. Node boxes, non-bottleneck labels on selected routes, unselected flow labels, flow curves, title, and panel frame are context unless explicitly referenced by the task.

## Query Details

| Query id | Program argument | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `source_label,target_label` | `integer_value` | `point_set` |
