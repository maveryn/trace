# `task_charts__dashboard__category_panel_condition_count`

## Contract
1. Domain: `charts`
2. Scene id: `dashboard`
3. Source implementation domain/group: `charts/dashboard`
4. Query ids: `category_panel_greater_than_threshold_count`, `category_panel_less_than_threshold_count`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.dashboard.category_panel_condition_count.ChartsDashboardCategoryPanelConditionCountTask`
2. Prompt lookup domain/group: `charts/dashboard`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `point_set`.
3. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `count(panel where compare(value(category_label,panel), threshold, relation={greater_than,less_than})); output=integer_count; annotation=point_set(matching_panel_marks); scene=dashboard; scope=category_panel_condition_count`

Candidate set: the visible dashboard panels, linked marks, and panel labels inside the `category_panel_condition_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `point_set` witnesses bound by `point_set(matching_panel_marks)`. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `category_panel_greater_than_threshold_count`, `category_panel_less_than_threshold_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `category_panel_greater_than_threshold_count` | `count.category_panel_condition` | `integer_count` | `point_set` |
| `category_panel_less_than_threshold_count` | `count.category_panel_condition` | `integer_count` | `point_set` |
