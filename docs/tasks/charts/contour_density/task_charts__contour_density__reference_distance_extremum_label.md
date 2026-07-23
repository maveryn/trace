# `task_charts__contour_density__reference_distance_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `contour_density`
3. Source implementation scene: `charts/contour_density`
4. Query ids: `point_nearest_region_label`, `point_farthest_region_label`, `vertical_line_nearest_region_label`, `vertical_line_farthest_region_label`, `horizontal_line_nearest_region_label`, `horizontal_line_farthest_region_label`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.contour_density.reference_distance_extremum_label.ChartsContourDensityReferenceDistanceExtremumLabelTask`
2. Prompt lookup domain/scene: `charts/contour_density`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `bbox`.
3. Annotation should contain one bbox around the selected answer contour region.
4. Renderer context such as axes, decorative labels, titles, and background treatments is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `arg_extreme(region_label, distance(region_label, reference_geometry={point,vertical_line,horizontal_line}), direction={nearest,farthest}); output=string_label; annotation=bbox(answer_region); scene=contour_density; scope=reference_distance_extremum_label`

Candidate set: the visible contour-density regions, guide labels, and marked areas inside the `reference_distance_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `bbox` witnesses bound by `bbox(answer_region)`. Annotation should contain one bbox around the selected answer contour region. Renderer context such as axes, decorative labels, titles, and background treatments is metadata unless the task explicitly asks for it as annotation.
Query ids: `point_nearest_region_label`, `point_farthest_region_label`, `vertical_line_nearest_region_label`, `vertical_line_farthest_region_label`, `horizontal_line_nearest_region_label`, `horizontal_line_farthest_region_label`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `point_nearest_region_label` | `selection.reference_distance_extremum_label` | `string_label` | `bbox` |
| `point_farthest_region_label` | `selection.reference_distance_extremum_label` | `string_label` | `bbox` |
| `vertical_line_nearest_region_label` | `selection.reference_distance_extremum_label` | `string_label` | `bbox` |
| `vertical_line_farthest_region_label` | `selection.reference_distance_extremum_label` | `string_label` | `bbox` |
| `horizontal_line_nearest_region_label` | `selection.reference_distance_extremum_label` | `string_label` | `bbox` |
| `horizontal_line_farthest_region_label` | `selection.reference_distance_extremum_label` | `string_label` | `bbox` |
