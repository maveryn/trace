# `task_charts__size_encoding__panel_category_extremum_panel_label`

## Contract
1. Domain: `charts`
2. Scene id: `size_encoding`
3. Source implementation: `src/trace_tasks/tasks/charts/size_encoding/panel_category_extremum_panel_label.py`
4. Supported `query_id` values: `largest_category_item_panel_label`, `smallest_category_item_panel_label`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.size_encoding.panel_category_extremum_panel_label.ChartsSizeEncodingPanelCategoryExtremumPanelLabelTask`
2. Prompt lookup domain/group: `charts/size_encoding`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `bbox`.
3. Annotation marks the single displayed item whose panel is the answer.
4. Renderer context such as legends, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `select_label(panel(arg_extreme(filter(items, category=target_category), encoded_value(item), direction))); output=string_label; annotation=bbox(answer_item); scene=size_encoding; scope=panel_category_extremum_panel_label`

Candidate set: the visible items whose size encodes value and their category labels inside the `panel_category_extremum_panel_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `select_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `bbox` witnesses bound by `bbox(answer_item)`. Annotation marks the single displayed item whose panel is the answer. Renderer context such as legends, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `largest_category_item_panel_label`, `smallest_category_item_panel_label`.

## Reasoning Operations

Families: `filtering`, `ranking`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `largest_category_item_panel_label` | `selection.extreme_metric_label` | `string_label` | `bbox` |
| `smallest_category_item_panel_label` | `selection.extreme_metric_label` | `string_label` | `bbox` |
