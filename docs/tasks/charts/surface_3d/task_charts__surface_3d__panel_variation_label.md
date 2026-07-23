# `task_charts__surface_3d__panel_variation_label`

## Contract
1. Domain: `charts`
2. Scene id: `surface_3d`
3. Public task id: `task_charts__surface_3d__panel_variation_label`
4. Query ids: `single`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.surface_3d.panel_variation_label.ChartsThreeDPanelVariationLabelTask`
2. Source file: `src/trace_tasks/tasks/charts/surface_3d/panel_variation_label.py`
3. Prompt bundle: `charts_surface_3d_v1`
4. Prompt keys: scene `surface_3d`, task `three_d_chart_query`, query `panel_variation_label`
5. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
6. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `bbox`.
3. Annotation marks one [x0,y0,x1,y1] pixel box around the selected panel.
4. The image shows either `4` or `6` chart panels by construction.
5. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `selection.extreme_label(range(z_values(panel)), largest); scene=surface_3d; scope=panel_variation_label`

Candidate set: the visible 3D surface samples, grid lines, and axis labels inside the `panel_variation_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `selection.extreme_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `bbox` witnesses bound by `see_annotation_contract`. Annotation marks one [x0,y0,x1,y1] pixel box around the selected panel. The image shows either `4` or `6` chart panels by construction. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `selection.extreme_label(range(z_values(panel)), largest)` | `string_label` | `bbox` |
