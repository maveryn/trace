# `task_charts__sankey__node_side_total_value`

## Contract
1. Domain: `charts`
2. Scene id: `sankey`
3. Task id: `task_charts__sankey__node_side_total_value`
4. Objective contract: `node_side_total_value`
5. Supported `query_id` values: `source_outgoing_total_flow`, `target_incoming_total_flow`

## Program Contract

Program: `sum(value(link) for link in node_side_links(node, side)); scene=sankey; scope=node_side_total_value`

Candidate set: the visible flow nodes, links, and labels inside the `node_side_total_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `sum` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point_set` witnesses bound by `see_annotation_contract`. Annotation marks multiple printed value-label centers: every outgoing or incoming band included in the node-side total contributes one point. Node boxes, unselected flow labels, flow curves, title, and panel frame are context unless explicitly referenced by the task.
Query ids: `source_outgoing_total_flow`, `target_incoming_total_flow`.

## Reasoning Operations

Families: `aggregation`, `topology`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.sankey.node_side_total_value.ChartsFlowSankeyNodeSideTotalValuePublicTask`
2. Prompt bundle: `charts_sankey_v1`
3. Scene key: `sankey`
4. Task key: `sankey_path_query`
5. Query keys: `source_outgoing_total_flow`, `target_incoming_total_flow`

## Annotation Contract
1. Answer schema: `integer_value`
2. Annotation schema: `point_set`
3. Annotation marks multiple printed value-label centers: every outgoing or incoming band included in the node-side total contributes one point.
4. Node boxes, unselected flow labels, flow curves, title, and panel frame are context unless explicitly referenced by the task.

## Query Details

| Query id | Program argument | Answer schema | Annotation schema |
|---|---|---|---|
| `source_outgoing_total_flow` | `side=source_outgoing` | `integer_value` | `point_set` |
| `target_incoming_total_flow` | `side=target_incoming` | `integer_value` | `point_set` |
