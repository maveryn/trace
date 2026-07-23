# `task_charts__scatter_cluster__centroid_option_selection_label`

## Contract
1. Domain: `charts`
2. Scene id: `scatter_cluster`
3. Task id: `task_charts__scatter_cluster__centroid_option_selection_label`
4. Supported `query_id`s: `single`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.scatter_cluster.centroid_option_selection_label.ChartsScatterClusterCentroidOptionSelectionLabelTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/scatter_cluster/charts_scatter_cluster_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.

## Program Contract

Program: `argmin_label(option, distance(point(option), centroid(target_cluster))); scene=scatter_cluster; scope=centroid_option_selection_label`

Candidate set: the visible scatter points, clusters, and cluster labels inside the `centroid_option_selection_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `argmin_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `option_letter` value bound by `option_letter`.
Annotation witnesses: `point` witnesses bound by `see_annotation_contract`. The rendered centroid-option markers use either `4` labels (`A..D`) or `6` labels (`A..F`) by construction. Annotation should mark the center point of the selected option marker. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `aggregation`, `spatial_relations`, `formula_evaluation`

## Annotation Contract
1. Answer schema: `option_letter`.
2. Annotation schema: `point`.
3. The rendered centroid-option markers use either `4` labels (`A..D`) or `6` labels (`A..F`) by construction.
4. Annotation should mark the center point of the selected option marker.
5. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Query Details

| Query id | Program arguments | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `target_cluster=sampled; option_set=4_or_6_markers` | `option_letter` | `point` |
