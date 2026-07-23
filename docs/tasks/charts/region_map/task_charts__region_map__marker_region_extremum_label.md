# `task_charts__region_map__marker_region_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `region_map`
3. Source implementation scene: `charts/region_map`
4. Query ids: `largest_marker_region_extremum_label`, `smallest_marker_region_extremum_label`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.region_map.marker_region_extremum_label.ChartsRegionMapMarkerRegionExtremumLabelTask`
2. Prompt lookup domain/scene: `charts/region_map`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point`.
3. Annotation marks the center point of the single marker bubble for the region whose visible label is the answer.
4. Renderer context such as map outlines, legends, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
5. The extremum marker layer shows at most ten labeled marker bubbles, using visible labels `A` through `J`; additional map regions may be present without marker bubbles.

## Program Contract

Program: `select_label(arg_extreme(marked_region, marker_value(marked_region), direction={largest,smallest})); output=string_label; annotation=point(center(marker_bubble(selected_region))); scene=region_map; scope=marker_region_extremum_label`

Candidate set: the visible map regions, region labels, and legend/value encodings inside the `marker_region_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `select_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point` witnesses bound by `point(center(marker_bubble(selected_region)))`. Annotation marks the center point of the single marker bubble for the region whose visible label is the answer. Renderer context such as map outlines, legends, titles, and distractor text is metadata unless the task explicitly asks for it as annotation. The extremum marker layer shows at most ten labeled marker bubbles, using visible labels `A` through `J`; additional map regions may be present without marker bubbles.
Query ids: `largest_marker_region_extremum_label`, `smallest_marker_region_extremum_label`.

## Reasoning Operations

Families: `ranking`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `largest_marker_region_extremum_label` | `selection.extreme_metric_label(direction=largest)` | `string_label` | `point` |
| `smallest_marker_region_extremum_label` | `selection.extreme_metric_label(direction=smallest)` | `string_label` | `point` |
