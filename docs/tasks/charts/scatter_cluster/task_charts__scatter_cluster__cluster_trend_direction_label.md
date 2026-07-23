# `task_charts__scatter_cluster__cluster_trend_direction_label`

## Contract
1. Domain: `charts`
2. Scene id: `scatter_cluster`
3. Task id: `task_charts__scatter_cluster__cluster_trend_direction_label`
4. Supported `query_id`s: `upward_trend_label`, `downward_trend_label`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.scatter_cluster.cluster_trend_direction_label.ChartsScatterClusterTrendDirectionLabelTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/scatter_cluster/charts_scatter_cluster_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.

## Program Contract

Program: `argmax_label(cluster, signed_trend_strength(cluster, direction)); scene=scatter_cluster; scope=cluster_trend_direction_label`

Candidate set: the visible scatter points, clusters, and cluster labels inside the `cluster_trend_direction_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `argmax_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string` value bound by `string`.
Annotation witnesses: `bbox` witnesses bound by `see_annotation_contract`. Annotation should mark the answer cluster hull, not the legend row or cluster label text. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `upward_trend_label`, `downward_trend_label`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Annotation Contract
1. Answer schema: `string`.
2. Annotation schema: `bbox`.
3. Annotation should mark the answer cluster hull, not the legend row or cluster label text.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Query Details

| Query id | Program arguments | Answer schema | Annotation schema |
|---|---|---|---|
| `upward_trend_label` | `direction=upward` | `string` | `bbox` |
| `downward_trend_label` | `direction=downward` | `string` | `bbox` |
