# `task_charts__sankey__path_bottleneck_value`

## Contract
1. Domain: `charts`
2. Scene id: `sankey`
3. Task id: `task_charts__sankey__path_bottleneck_value`
4. Objective contract: `path_bottleneck_value`
5. Supported `query_id` values: `single`

## Program Contract

Program: `min(value(source_to_middle), value(middle_to_target)); scene=sankey; scope=path_bottleneck_value`

Candidate set: the visible flow nodes, links, and labels inside the `path_bottleneck_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `min` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point` witnesses bound by `see_annotation_contract`. Annotation marks one printed value-label center on the selected route: the uniquely lower-valued bottleneck band. Node boxes, the other flow label on the same route, unselected flow labels, flow curves, title, and panel frame are context unless explicitly referenced by the task.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `topology`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.sankey.path_bottleneck_value.ChartsFlowSankeyPathBottleneckValuePublicTask`
2. Prompt bundle: `charts_sankey_v1`
3. Scene key: `sankey`
4. Task key: `sankey_path_query`
5. Query key: `path_bottleneck_value`

## Annotation Contract
1. Answer schema: `integer_value`
2. Annotation schema: `point`
3. Annotation marks one printed value-label center on the selected route: the uniquely lower-valued bottleneck band.
4. Node boxes, the other flow label on the same route, unselected flow labels, flow curves, title, and panel frame are context unless explicitly referenced by the task.

## Query Details

| Query id | Program argument | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `selected_path` | `integer_value` | `point` |
