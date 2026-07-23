# `task_charts__dashboard__panel_total_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `dashboard`
3. Source implementation domain/group: `charts/dashboard`
4. Query ids: `largest_panel_total_label`, `smallest_panel_total_label`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.dashboard.panel_total_extremum_label.ChartsDashboardPanelTotalExtremumLabelTask`
2. Prompt lookup domain/group: `charts/dashboard`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point_set`.
3. Annotation points to every category mark in the answer panel.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `arg_extreme(panel, sum(value(category,panel) for category in shared_categories), direction={largest,smallest}); output=string_label; annotation=point_set(answer_panel_category_marks); scene=dashboard; scope=panel_total_extremum_label`

Candidate set: the visible dashboard panels, linked marks, and panel labels inside the `panel_total_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point_set` witnesses bound by `point_set(answer_panel_category_marks)`. Annotation points to every category mark in the answer panel. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `largest_panel_total_label`, `smallest_panel_total_label`.

## Reasoning Operations

Families: `ranking`, `aggregation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `largest_panel_total_label` | `selection.panel_total_extremum_label` | `string_label` | `point_set` |
| `smallest_panel_total_label` | `selection.panel_total_extremum_label` | `string_label` | `point_set` |
