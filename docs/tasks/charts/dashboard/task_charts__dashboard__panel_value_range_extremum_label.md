# `task_charts__dashboard__panel_value_range_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `dashboard`
3. Source implementation domain/group: `charts/dashboard`
4. Query ids: `largest_panel_value_range_label`, `smallest_panel_value_range_label`
5. Semantic query details are recorded in trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.dashboard.panel_value_range_extremum_label.ChartsDashboardPanelValueRangeExtremumLabelTask`
2. Prompt lookup domain/group: `charts/dashboard`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point_map`.
3. Annotation keys are `largest_value` and `smallest_value`, each pointing to the category mark that defines the answer panel's range.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `arg_extreme(panel, difference(max(value(category,panel)), min(value(category,panel))), direction={largest,smallest}); output=string_label; annotation=point_map(largest_value, smallest_value); scene=dashboard; scope=panel_value_range_extremum_label`

Candidate set: the visible dashboard panels, linked marks, and panel labels inside the `panel_value_range_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point_map` witnesses bound by `point_map(largest_value, smallest_value)`. Annotation keys are `largest_value` and `smallest_value`, each pointing to the category mark that defines the answer panel's range. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `largest_panel_value_range_label`, `smallest_panel_value_range_label`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `largest_panel_value_range_label` | `selection.panel_value_range_extremum_label` | `string_label` | `point_map` |
| `smallest_panel_value_range_label` | `selection.panel_value_range_extremum_label` | `string_label` | `point_map` |
