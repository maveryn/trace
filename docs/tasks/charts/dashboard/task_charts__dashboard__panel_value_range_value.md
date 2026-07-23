# `task_charts__dashboard__panel_value_range_value`

## Contract
1. Domain: `charts`
2. Scene id: `dashboard`
3. Source implementation domain/group: `charts/dashboard`
4. Query ids: `single`
5. Semantic query details are recorded in trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.dashboard.panel_value_range_value.ChartsDashboardPanelValueRangeValueTask`
2. Prompt lookup domain/group: `charts/dashboard`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `point_map`.
3. Annotation keys are `largest_value` and `smallest_value`, each pointing to the corresponding category mark in the named dashboard panel.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `difference(max(value(category,selected_panel)), min(value(category,selected_panel))); output=integer_value; annotation=point_map(largest_value, smallest_value); scene=dashboard; scope=panel_value_range_value`

Candidate set: the visible dashboard panels, linked marks, and panel labels inside the `panel_value_range_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `difference` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point_map` witnesses bound by `point_map(largest_value, smallest_value)`. Annotation keys are `largest_value` and `smallest_value`, each pointing to the corresponding category mark in the named dashboard panel. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `numeric.panel_value_range` | `integer_value` | `point_map` |
