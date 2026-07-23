# `task_charts__surface_3d__series_trend_label`

## Contract
1. Domain: `charts`
2. Scene id: `surface_3d`
3. Public task id: `task_charts__surface_3d__series_trend_label`
4. Query ids: `increase`, `decrease`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.surface_3d.series_trend_label.ChartsThreeDSeriesTrendLabelTask`
2. Source file: `src/trace_tasks/tasks/charts/surface_3d/series_trend_label.py`
3. Prompt bundle: `charts_surface_3d_v1`
4. Prompt keys: scene `surface_3d`, task `three_d_chart_query`, query `increase` or `decrease`
5. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
6. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `segment`.
3. Annotation is one segment connecting the first and last marker centers of the answer series.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `selection.extreme_label(difference(z_value(last_x_point(series)), z_value(first_x_point(series))), direction); scene=surface_3d; scope=series_trend_label`

Candidate set: the visible 3D surface samples, grid lines, and axis labels inside the `series_trend_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `selection.extreme_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `segment` witnesses bound by `see_annotation_contract`. Annotation is one segment connecting the first and last marker centers of the answer series. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `increase`, `decrease`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `increase` | `selection.extreme_label(difference(z_value(last_x_point(series)), z_value(first_x_point(series))), increase)` | `string_label` | `segment` |
| `decrease` | `selection.extreme_label(difference(z_value(last_x_point(series)), z_value(first_x_point(series))), decrease)` | `string_label` | `segment` |
