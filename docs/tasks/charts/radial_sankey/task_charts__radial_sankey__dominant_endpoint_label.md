# `task_charts__radial_sankey__dominant_endpoint_label`

## Contract
1. Domain: `charts`
2. Scene id: `radial_sankey`
3. Task id: `task_charts__radial_sankey__dominant_endpoint_label`
4. Objective contract: `dominant_endpoint_label`
5. Supported `query_id` values: `largest_target_for_source`, `largest_source_for_target`

## Program Contract

Program: `argmax_label(endpoint(flow), value(flow), fixed_opposite_endpoint); scene=radial_sankey; scope=dominant_endpoint_label`

Candidate set: the visible radial flow nodes, links, and labels inside the `dominant_endpoint_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `argmax_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `bbox` witnesses bound by `see_annotation_contract`. Annotation marks the selected source or target endpoint node box. The flow curves, title, panel frame, ring, and unrelated nodes are context unless explicitly referenced by the task.
Query ids: `largest_target_for_source`, `largest_source_for_target`.

## Reasoning Operations

Families: `ranking`, `topology`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.radial_sankey.dominant_endpoint_label.ChartsRadialSankeyDominantEndpointLabelTask`
2. Prompt bundle: `charts_radial_sankey_v1`
3. Scene key: `radial_sankey`
4. Task key: `radial_sankey_query`
5. Query keys: `largest_target_for_source`, `largest_source_for_target`

## Annotation Contract
1. Answer schema: `string_label`
2. Annotation schema: `bbox`
3. Annotation marks the selected source or target endpoint node box.
4. The flow curves, title, panel frame, ring, and unrelated nodes are context unless explicitly referenced by the task.

## Query Details

| Query id | Program argument | Answer schema | Annotation schema |
|---|---|---|---|
| `largest_target_for_source` | `endpoint=target; fixed_opposite_endpoint=source_label` | `string_label` | `bbox` |
| `largest_source_for_target` | `endpoint=source; fixed_opposite_endpoint=target_label` | `string_label` | `bbox` |
