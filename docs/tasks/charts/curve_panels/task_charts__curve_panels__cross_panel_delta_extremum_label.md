# `task_charts__curve_panels__cross_panel_delta_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `curve_panels`
3. Source implementation domain/group: `charts/curve_panels`
4. Query id: `single`
5. Semantic query details are recorded in `query_id` and trace params.
6. Controlled-unanswerable sampling may request a visible method label that is missing from one subplot.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.curve_panels.cross_panel_delta_extremum_label.ChartsScientificCrossPanelDeltaExtremumLabelTask`
2. Prompt lookup domain/group: `charts/curve_panels`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label` or the literal string `unanswerable`.
2. Annotation schema: `point_map`.
3. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy.
4. Answerable instances map `start_point` and `end_point` to points on the selected method markers in the answer subplot.
5. Unanswerable instances use an empty annotation object because the requested method is not plotted in every subplot.
6. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `if plotted_in_every_panel(method): argmax(panel_label, value(method,end_x,panel_label)-value(method,start_x,panel_label)); else unanswerable; output=string_label|unanswerable; annotation=point_map(start_point,end_point)|empty_map; scene=curve_panels; scope=cross_panel_delta_extremum_label`

Candidate set: the visible curve panels, curve traces, points, and panel labels inside the `cross_panel_delta_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `if plotted_in_every_panel` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label|unanswerable`.
Annotation witnesses: `point_map` witnesses bound by `point_map(start_point,end_point)|empty_map`. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy. Answerable instances map `start_point` and `end_point` to points on the selected method markers in the answer subplot. Unanswerable instances use an empty annotation object because the requested method is not plotted in every subplot.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `selection.cross_panel_delta_extremum_label` | `string_label_or_unanswerable` | `point_map` |
