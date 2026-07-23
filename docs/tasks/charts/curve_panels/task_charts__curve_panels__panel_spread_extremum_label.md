# `task_charts__curve_panels__panel_spread_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `curve_panels`
3. Source implementation domain/group: `charts/curve_panels`
4. Query ids: `largest_panel_spread_label`, `smallest_panel_spread_label`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.curve_panels.panel_spread_extremum_label.ChartsScientificPanelSpreadExtremumLabelTask`
2. Prompt lookup domain/group: `charts/curve_panels`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point_map`.
3. Annotation should map `min_point` and `max_point` to the answer subplot's minimum and maximum plotted markers.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `arg_extreme(panel_label, max(value(points_in_panel))-min(value(points_in_panel)), direction={largest,smallest}); output=string_label; annotation=point_map(min_point, max_point); scene=curve_panels; scope=panel_spread_extremum_label`

Candidate set: the visible curve panels, curve traces, points, and panel labels inside the `panel_spread_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point_map` witnesses bound by `point_map(min_point, max_point)`. Annotation should map `min_point` and `max_point` to the answer subplot's minimum and maximum plotted markers. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `largest_panel_spread_label`, `smallest_panel_spread_label`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `largest_panel_spread_label` | `selection.panel_spread_extremum_label` | `string_label` | `point_map` |
| `smallest_panel_spread_label` | `selection.panel_spread_extremum_label` | `string_label` | `point_map` |
