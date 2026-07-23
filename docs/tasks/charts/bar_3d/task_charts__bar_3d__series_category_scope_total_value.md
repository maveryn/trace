# `task_charts__bar_3d__series_category_scope_total_value`

## Contract
1. Domain: `charts`
2. Scene id: `bar_3d`
3. Source implementation scene: `charts/bar_3d`
4. Query ids: `series_total_value`, `series_interval_total_value`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.bar_3d.series_category_scope_total_value.ChartsThreeDBarSeriesCategoryScopeTotalValueTask`
2. Prompt lookup scene: `charts/bar_3d`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `point_set`.
3. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `sum(value(category,series_label) for category in selected_category_scope); output=integer_value; annotation=point_set(selected_bar_top_centers); scene=bar_3d; scope=series_category_scope_total_value; category_scope={all,contiguous_interval}`

Candidate set: the visible 3D bars grouped by category and series inside the `series_category_scope_total_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `sum` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point_set` witnesses bound by `point_set(selected_bar_top_centers)`. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `series_total_value`, `series_interval_total_value`.

## Reasoning Operations

Families: `aggregation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `series_total_value` | `numeric.series_category_scope_sum` | `integer_value` | `point_set` |
| `series_interval_total_value` | `numeric.series_category_scope_sum` | `integer_value` | `point_set` |
