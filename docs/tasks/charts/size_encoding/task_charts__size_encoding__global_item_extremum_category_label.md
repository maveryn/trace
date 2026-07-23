# `task_charts__size_encoding__global_item_extremum_category_label`

## Contract
1. Domain: `charts`
2. Scene id: `size_encoding`
3. Source implementation: `src/trace_tasks/tasks/charts/size_encoding/global_item_extremum_category_label.py`
4. Supported `query_id` values: `largest_overall_size_category_label`, `smallest_overall_size_category_label`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.size_encoding.global_item_extremum_category_label.ChartsSizeEncodingGlobalItemExtremumCategoryLabelTask`
2. Prompt lookup domain/group: `charts/size_encoding`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `bbox`.
3. Annotation marks the single displayed item with the global extremal size; the answer is that item's category label.
4. Renderer context such as legends, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `select_label(category(arg_extreme(items, encoded_value(item), direction))); output=string_label; annotation=bbox(answer_item); scene=size_encoding; scope=global_item_extremum_category_label`

Candidate set: the visible items whose size encodes value and their category labels inside the `global_item_extremum_category_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `select_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `bbox` witnesses bound by `bbox(answer_item)`. Annotation marks the single displayed item with the global extremal size; the answer is that item's category label. Renderer context such as legends, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `largest_overall_size_category_label`, `smallest_overall_size_category_label`.

## Reasoning Operations

Families: `ranking`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `largest_overall_size_category_label` | `selection.extreme_metric_label` | `string_label` | `bbox` |
| `smallest_overall_size_category_label` | `selection.extreme_metric_label` | `string_label` | `bbox` |
