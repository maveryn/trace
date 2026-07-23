# `task_charts__dashboard__category_total_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `dashboard`
3. Source implementation domain/group: `charts/dashboard`
4. Query ids: `largest_category_total_label`, `smallest_category_total_label`
5. Semantic query details are recorded in `query_id` and trace params.
6. Controlled-unanswerable sampling may omit one category from one dashboard panel.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.dashboard.category_total_extremum_label.ChartsDashboardCategoryTotalExtremumLabelTask`
2. Prompt lookup domain/group: `charts/dashboard`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label` or the literal string `unanswerable`.
2. Annotation schema: `point_set`.
3. Answerable instances annotate the answer category mark in every dashboard panel.
4. Unanswerable instances use an empty annotation array because at least one category is not shown in every dashboard panel.
5. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `if every category is shown in every dashboard panel: arg_extreme(category, sum(value(category,panel) for panel in dashboard_panels), direction={largest,smallest}); else unanswerable; output=string_label|unanswerable; annotation=point_set(answer_category_marks_across_panels)|empty_set; scene=dashboard; scope=category_total_extremum_label`

Candidate set: the visible dashboard panels, linked marks, and panel labels inside the `category_total_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `if every category is shown in every dashboard panel: arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label|unanswerable`.
Annotation witnesses: `point_set` witnesses bound by `point_set(answer_category_marks_across_panels)|empty_set`. Answerable instances annotate the answer category mark in every dashboard panel. Unanswerable instances use an empty annotation array because at least one category is not shown in every dashboard panel. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `largest_category_total_label`, `smallest_category_total_label`.

## Reasoning Operations

Families: `ranking`, `aggregation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `largest_category_total_label` | `selection.category_total_extremum_label` | `string_label_or_unanswerable` | `point_set` |
| `smallest_category_total_label` | `selection.category_total_extremum_label` | `string_label_or_unanswerable` | `point_set` |
