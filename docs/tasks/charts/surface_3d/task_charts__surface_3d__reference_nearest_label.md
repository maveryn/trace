# `task_charts__surface_3d__reference_nearest_label`

## Contract
1. Domain: `charts`
2. Scene id: `surface_3d`
3. Public task id: `task_charts__surface_3d__reference_nearest_label`
4. Query ids: `single`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.surface_3d.reference_nearest_label.ChartsThreeDReferenceNearestLabelTask`
2. Source file: `src/trace_tasks/tasks/charts/surface_3d/reference_nearest_label.py`
3. Prompt bundle: `charts_surface_3d_v1`
4. Prompt keys: scene `surface_3d`, task `three_d_chart_query`, query `reference_nearest_label`
5. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
6. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point`.
3. Annotation marks one [x,y] pixel point at the center of the selected point marker.
4. The renderer may draw floor-projection guides and a y-reference guide to make the 3D y-axis readout visible; these guide marks are not annotation targets.
5. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `selection.nearest_label(value(point, axis), reference_value); scene=surface_3d; scope=reference_nearest_label`

Candidate set: the visible 3D surface samples, grid lines, and axis labels inside the `reference_nearest_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `selection.nearest_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point` witnesses bound by `see_annotation_contract`. Annotation marks one [x,y] pixel point at the center of the selected point marker. The renderer may draw floor-projection guides and a y-reference guide to make the 3D y-axis readout visible; these guide marks are not annotation targets. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `selection.nearest_label(value(point, axis), reference_value)` | `string_label` | `point` |
