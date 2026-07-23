# `task_charts__curve_panels__threshold_series_count`

## Contract
1. Domain: `charts`
2. Scene id: `curve_panels`
3. Source implementation domain/group: `charts/curve_panels`
4. Query ids: `above_threshold_series_count`, `below_threshold_series_count`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.curve_panels.threshold_series_count.ChartsScientificThresholdSeriesCountTask`
2. Prompt lookup domain/group: `charts/curve_panels`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `point_set`.
3. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `count(curve_label in panel_label where compare(value(curve_label,x_ref),threshold,direction)); direction in {above,below}; output=integer_count; annotation=point_set(matching_curve_marks); scene=curve_panels; scope=threshold_series_count`

Candidate set: the visible curve panels, curve traces, points, and panel labels inside the `threshold_series_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `point_set` witnesses bound by `point_set(matching_curve_marks)`. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `above_threshold_series_count`, `below_threshold_series_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `above_threshold_series_count` | `count.thresholded_series_at_x(direction=above)` | `integer_count` | `point_set` |
| `below_threshold_series_count` | `count.thresholded_series_at_x(direction=below)` | `integer_count` | `point_set` |
