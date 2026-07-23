# `task_charts__curve_panels__global_value_extremum_panel_label`

## Contract
1. Domain: `charts`
2. Scene id: `curve_panels`
3. Source implementation domain/group: `charts/curve_panels`
4. Query ids: `overall_maximum_value_panel_label`, `overall_minimum_value_panel_label`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.curve_panels.global_value_extremum_panel_label.ChartsScientificGlobalValueExtremumPanelLabelTask`
2. Prompt lookup domain/group: `charts/curve_panels`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point`.
3. Annotation should mark the single global maximum/minimum marker that determines the answer.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `arg_extreme(panel_label, value(point), direction={maximum,minimum}, over=all_curve_points_all_panels); output=string_label; annotation=point(answer_extremum_mark); scene=curve_panels; scope=global_value_extremum_panel_label`

Candidate set: the visible curve panels, curve traces, points, and panel labels inside the `global_value_extremum_panel_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point` witnesses bound by `point(answer_extremum_mark)`. Annotation should mark the single global maximum/minimum marker that determines the answer. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `overall_maximum_value_panel_label`, `overall_minimum_value_panel_label`.

## Reasoning Operations

Families: `ranking`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `overall_maximum_value_panel_label` | `selection.global_value_extremum_panel_label` | `string_label` | `point` |
| `overall_minimum_value_panel_label` | `selection.global_value_extremum_panel_label` | `string_label` | `point` |
