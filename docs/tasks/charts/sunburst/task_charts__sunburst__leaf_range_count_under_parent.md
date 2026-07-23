# `task_charts__sunburst__leaf_range_count_under_parent`

## Contract
1. Domain: `charts`
2. Scene id: `sunburst`
3. Public task id: `task_charts__sunburst__leaf_range_count_under_parent`
4. Supported `query_id`: `single`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.sunburst.leaf_range_count_under_parent.ChartsSunburstLeafRangeCountUnderParentTask`
2. Prompt bundle: `charts_sunburst_v1`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `point_set`.
3. Annotation marks the centers of the printed outer leaf value labels inside the requested inclusive value range under the requested parent category.
4. Renderer context such as decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `count(leaf under parent where lower <= value(leaf) <= upper); output=integer_count; annotation=point_set(matching_leaf_value_labels); scene=sunburst; scope=leaf_range_count_under_parent`

Candidate set: the visible hierarchy wedges, rings, and node labels inside the `leaf_range_count_under_parent` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `point_set` witnesses bound by `point_set(matching_leaf_value_labels)`. Annotation marks the centers of the printed outer leaf value labels inside the requested inclusive value range under the requested parent category. Renderer context such as decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`, `topology`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `count(leaf under parent where lower <= value(leaf) <= upper)` | `integer_count` | `point_set` |
