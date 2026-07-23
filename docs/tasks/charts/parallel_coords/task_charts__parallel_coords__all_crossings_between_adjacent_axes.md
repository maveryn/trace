# `task_charts__parallel_coords__all_crossings_between_adjacent_axes`

## Contract
1. Domain: `charts`
2. Scene id: `parallel_coords`
3. Source implementation domain/scene: `charts/parallel_coords`
4. Supported `query_id`: `single`
5. The semantic prompt branch is `all_crossings_between_adjacent_axes`; public replay metadata uses `single`.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.parallel_coords.all_crossings_between_adjacent_axes.ChartsParallelCoordinatesAllCrossingsBetweenAdjacentAxesTask`
2. Prompt lookup domain/scene: `charts/parallel_coords`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Program Contract

Program: `count(intersections(profile_pair_lines, adjacent_axis_interval)); output=integer_count; annotation=point_set(crossing_points); scene=parallel_coords; scope=all_crossings_between_adjacent_axes`

Candidate set: the visible polylines, axes, and axis-value positions inside the `all_crossings_between_adjacent_axes` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `point_set` witnesses bound by `point_set(crossing_points)`. Annotation marks one point at each counted profile-line crossing between the named adjacent axes. Axes, labels, threshold text, and decorative context are renderer context unless explicitly requested.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `point_set`.
3. Annotation marks one point at each counted profile-line crossing between the named adjacent axes.
4. Axes, labels, threshold text, and decorative context are renderer context unless explicitly requested.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `count.intersection_or_crossing` | `integer_count` | `point_set` |
