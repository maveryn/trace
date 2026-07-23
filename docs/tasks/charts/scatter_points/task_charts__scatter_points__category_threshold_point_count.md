# `task_charts__scatter_points__category_threshold_point_count`

## Contract
1. Domain: `charts`
2. Scene id: `scatter_points`
3. Public task id: `task_charts__scatter_points__category_threshold_point_count`
4. Query ids encode the axis and threshold direction because both change prompt wording and program arguments.

## Program Contract

Program: `count(point, category(point)=target_category and compare(coord(point, axis), threshold, direction)); output=integer_count; annotation=point_set(counted_category_point_centers); scene=scatter_points; scope=category_threshold_point_count`

Candidate set: the visible scatter points and axis/value labels inside the `category_threshold_point_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `point_set` witnesses bound by `point_set(counted_category_point_centers)`. Annotation marks the centers of the counted points in the named category only. Axes, legends, threshold guides, titles, and distractor text are not annotation targets.
Query ids: `category_x_above_threshold_count`, `category_x_below_threshold_count`, `category_y_above_threshold_count`, `category_y_below_threshold_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`, `logical_composition`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.scatter_points.category_threshold_point_count.ChartsScatterPointsCategoryThresholdPointCountTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/scatter_points/charts_scatter_points_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `point_set`.
3. Annotation marks the centers of the counted points in the named category only.
4. Axes, legends, threshold guides, titles, and distractor text are not annotation targets.

## Query Details

| Query id | Program arguments | Answer schema | Annotation schema |
|---|---|---|---|
| `category_x_above_threshold_count` | `axis=x`, `direction=above` | `integer_count` | `point_set` |
| `category_x_below_threshold_count` | `axis=x`, `direction=below` | `integer_count` | `point_set` |
| `category_y_above_threshold_count` | `axis=y`, `direction=above` | `integer_count` | `point_set` |
| `category_y_below_threshold_count` | `axis=y`, `direction=below` | `integer_count` | `point_set` |
