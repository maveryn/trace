# `task_charts__scatter_points__category_axis_mean_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `scatter_points`
3. Public task id: `task_charts__scatter_points__category_axis_mean_extremum_label`
4. Query ids encode the mean axis and extremum direction because both change prompt wording and program arguments.

## Program Contract

Program: `argextreme_label(category, mean(coord(points(category), axis)), direction); output=string_label; annotation=bbox(answer_category_point_cluster); scene=scatter_points; scope=category_axis_mean_extremum_label`

Candidate set: the visible scatter points and axis/value labels inside the `category_axis_mean_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `argextreme_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `bbox` witnesses bound by `bbox(answer_category_point_cluster)`. Annotation marks one bounding box around the answer category's point cluster. Axes, legends, category labels, titles, and distractor text are not annotation targets.
Query ids: `largest_mean_x_category_label`, `smallest_mean_x_category_label`, `largest_mean_y_category_label`, `smallest_mean_y_category_label`.

## Reasoning Operations

Families: `ranking`, `aggregation`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.scatter_points.category_axis_mean_extremum_label.ChartsScatterPointsCategoryAxisMeanExtremumLabelTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/scatter_points/charts_scatter_points_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `bbox`.
3. Annotation marks one bounding box around the answer category's point cluster.
4. Axes, legends, category labels, titles, and distractor text are not annotation targets.

## Query Details

| Query id | Program arguments | Answer schema | Annotation schema |
|---|---|---|---|
| `largest_mean_x_category_label` | `axis=x`, `direction=largest` | `string_label` | `bbox` |
| `smallest_mean_x_category_label` | `axis=x`, `direction=smallest` | `string_label` | `bbox` |
| `largest_mean_y_category_label` | `axis=y`, `direction=largest` | `string_label` | `bbox` |
| `smallest_mean_y_category_label` | `axis=y`, `direction=smallest` | `string_label` | `bbox` |
