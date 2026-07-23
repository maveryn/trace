# `task_charts__size_encoding__category_relative_size_count`

## Contract
1. Domain: `charts`
2. Scene id: `size_encoding`
3. Source implementation: `src/trace_tasks/tasks/charts/size_encoding/category_relative_size_count.py`
4. Supported `query_id` values: `larger_than_reference_in_category_count`, `smaller_than_reference_in_category_count`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.size_encoding.category_relative_size_count.ChartsSizeEncodingCategoryRelativeSizeCountTask`
2. Prompt lookup domain/group: `charts/size_encoding`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `bbox_set_map`.
3. Annotation maps `reference_item` to a one-box set for the reference item and `counted_items` to boxes for the counted same-category items.
4. Renderer context such as legends, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `count(filter(items, category=target_category and encoded_value(item) comparison reference_item)); comparison={larger,smaller}; output=integer_value; annotation=bbox_set_map(reference_item,counted_items); scene=size_encoding; scope=category_relative_size_count`

Candidate set: the visible items whose size encodes value and their category labels inside the `category_relative_size_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `bbox_set_map` witnesses bound by `bbox_set_map(reference_item,counted_items)`. Annotation maps `reference_item` to a one-box set for the reference item and `counted_items` to boxes for the counted same-category items. Renderer context such as legends, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `larger_than_reference_in_category_count`, `smaller_than_reference_in_category_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `larger_than_reference_in_category_count` | `count.relative_metric_threshold` | `integer_value` | `bbox_set_map` |
| `smaller_than_reference_in_category_count` | `count.relative_metric_threshold` | `integer_value` | `bbox_set_map` |
