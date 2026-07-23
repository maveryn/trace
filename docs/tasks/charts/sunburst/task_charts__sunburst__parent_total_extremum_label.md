# `task_charts__sunburst__parent_total_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `sunburst`
3. Public task id: `task_charts__sunburst__parent_total_extremum_label`
4. Supported `query_id`: `highest_parent_total_label`, `lowest_parent_total_label`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.sunburst.parent_total_extremum_label.ChartsSunburstParentTotalExtremumLabelTask`
2. Prompt bundle: `charts_sunburst_v1`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point_set`.
3. Annotation marks the centers of the printed outer leaf value labels under the answer parent category.
4. Renderer context such as decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `arg_extremum(parent, sum(value(leaf) for leaf under parent), direction={highest,lowest}); output=string_label; annotation=point_set(answer_parent_leaf_value_labels); scene=sunburst; scope=parent_total_extremum_label`

Candidate set: the visible hierarchy wedges, rings, and node labels inside the `parent_total_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extremum` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point_set` witnesses bound by `point_set(answer_parent_leaf_value_labels)`. Annotation marks the centers of the printed outer leaf value labels under the answer parent category. Renderer context such as decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `highest_parent_total_label`, `lowest_parent_total_label`.

## Reasoning Operations

Families: `ranking`, `aggregation`, `topology`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `highest_parent_total_label` | `argmax(parent, sum(value(leaf) for leaf under parent))` | `string_label` | `point_set` |
| `lowest_parent_total_label` | `argmin(parent, sum(value(leaf) for leaf under parent))` | `string_label` | `point_set` |
