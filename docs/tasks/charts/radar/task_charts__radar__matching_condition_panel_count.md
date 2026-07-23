# `task_charts__radar__matching_condition_panel_count`

## Taxonomy

1. Domain: `charts`
2. Scene id: `radar`
3. Source implementation scene: `charts/radar`
4. Public task id: `task_charts__radar__matching_condition_panel_count`

## Implementation

1. Registered class: `trace_tasks.tasks.charts.radar.matching_condition_panel_count.ChartsRadarMatchingConditionPanelCountTask`
2. Prompt lookup domain/scene: `charts/radar`
3. Default dataset: enabled

## Contract

1. Supported `query_id` values: `single`
2. Answer schema: `integer_count`
3. Annotation schema: `bbox_set`
4. Annotation marks one bbox around each radar panel with at least the requested number of metrics above the sampled threshold.

## Program Contract

Program: `count(filter(radar_panels, count(filter(metrics, value(panel, metric) > threshold)) >= minimum_metric_count)); scene=radar; scope=matching_condition_panel_count`

Candidate set: the visible radar spokes, profile polygons, and profile labels inside the `matching_condition_panel_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `unspecified`.
Annotation witnesses: `unspecified` witnesses bound by `see_annotation_contract`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`
