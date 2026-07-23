# `task_charts__radial_sankey__transfer_total_value`

## Contract
1. Domain: `charts`
2. Scene id: `radial_sankey`
3. Task id: `task_charts__radial_sankey__transfer_total_value`
4. Objective contract: `transfer_total_value`
5. Supported `query_id` values: `source_to_targets_total`, `sources_to_target_total`

## Program Contract

Program: `sum(value(flow) for flow in grouped_flows sharing one endpoint); scene=radial_sankey; scope=transfer_total_value`

Candidate set: the visible radial flow nodes, links, and labels inside the `transfer_total_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `sum` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `bbox_set` witnesses bound by `see_annotation_contract`. Annotation marks the printed value-label boxes for every flow included in the sum. Source/target node boxes, unselected flow labels, the flow curves, title, panel frame, and ring are context unless explicitly referenced by the task.
Query ids: `source_to_targets_total`, `sources_to_target_total`.

## Reasoning Operations

Families: `aggregation`, `topology`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.radial_sankey.transfer_total_value.ChartsRadialSankeyTransferTotalValueTask`
2. Prompt bundle: `charts_radial_sankey_v1`
3. Scene key: `radial_sankey`
4. Task key: `radial_sankey_query`
5. Query keys: `source_to_targets_total`, `sources_to_target_total`

## Annotation Contract
1. Answer schema: `integer_value`
2. Annotation schema: `bbox_set`
3. Annotation marks the printed value-label boxes for every flow included in the sum.
4. Source/target node boxes, unselected flow labels, the flow curves, title, panel frame, and ring are context unless explicitly referenced by the task.

## Query Details

| Query id | Program argument | Answer schema | Annotation schema |
|---|---|---|---|
| `source_to_targets_total` | `fixed_endpoint=source; grouped_endpoint=targets` | `integer_value` | `bbox_set` |
| `sources_to_target_total` | `fixed_endpoint=target; grouped_endpoint=sources` | `integer_value` | `bbox_set` |
