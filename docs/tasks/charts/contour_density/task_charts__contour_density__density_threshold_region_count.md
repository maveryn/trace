# `task_charts__contour_density__density_threshold_region_count`

## Contract
1. Domain: `charts`
2. Scene id: `contour_density`
3. Source implementation scene: `charts/contour_density`
4. Query ids: `density_at_least_threshold_region_count`, `density_below_threshold_region_count`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.contour_density.density_threshold_region_count.ChartsContourDensityDensityThresholdRegionCountTask`
2. Prompt lookup domain/scene: `charts/contour_density`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `bbox_set`.
3. Annotation should mark every region matching the visible density-level threshold.
4. Renderer context such as axes, decorative labels, titles, and background treatments is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `count(region where compare(density_level(region), threshold_level, relation={at_least,below})); output=integer_count; annotation=bbox_set(matching_regions); scene=contour_density; scope=density_threshold_region_count`

Candidate set: the visible contour-density regions, guide labels, and marked areas inside the `density_threshold_region_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `bbox_set` witnesses bound by `bbox_set(matching_regions)`. Annotation should mark every region matching the visible density-level threshold. Renderer context such as axes, decorative labels, titles, and background treatments is metadata unless the task explicitly asks for it as annotation.
Query ids: `density_at_least_threshold_region_count`, `density_below_threshold_region_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `density_at_least_threshold_region_count` | `count.density_threshold_regions` | `integer_count` | `bbox_set` |
| `density_below_threshold_region_count` | `count.density_threshold_regions` | `integer_count` | `bbox_set` |
