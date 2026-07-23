# `task_charts__region_map__marker_region_threshold_count`

## Contract
1. Domain: `charts`
2. Scene id: `region_map`
3. Source implementation scene: `charts/region_map`
4. Query ids: `greater_than_marker_region_threshold_count`, `less_than_marker_region_threshold_count`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.region_map.marker_region_threshold_count.ChartsRegionMapMarkerRegionThresholdCountTask`
2. Prompt lookup domain/scene: `charts/region_map`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `point_set`.
3. Annotation marks one center point for every marker bubble whose value satisfies the requested threshold predicate.
4. Renderer context such as map outlines, legends, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `count(region where compare(marker_value(region), threshold_value, relation={greater_than,less_than})); output=integer_count; annotation=point_set(center(marker_bubble(matching_regions))); scene=region_map; scope=marker_region_threshold_count`

Candidate set: the visible map regions, region labels, and legend/value encodings inside the `marker_region_threshold_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `point_set` witnesses bound by `point_set(center(marker_bubble(matching_regions)))`. Annotation marks one center point for every marker bubble whose value satisfies the requested threshold predicate. Renderer context such as map outlines, legends, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `greater_than_marker_region_threshold_count`, `less_than_marker_region_threshold_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `greater_than_marker_region_threshold_count` | `count.one_bound_threshold(relation=greater_than)` | `integer_count` | `point_set` |
| `less_than_marker_region_threshold_count` | `count.one_bound_threshold(relation=less_than)` | `integer_count` | `point_set` |
