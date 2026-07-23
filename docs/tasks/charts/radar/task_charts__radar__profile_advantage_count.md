# `task_charts__radar__profile_advantage_count`

## Taxonomy

1. Domain: `charts`
2. Scene id: `radar`
3. Source implementation scene: `charts/radar`
4. Public task id: `task_charts__radar__profile_advantage_count`

## Implementation

1. Registered class: `trace_tasks.tasks.charts.radar.profile_advantage_count.ChartsRadarProfileAdvantageCountTask`
2. Prompt lookup domain/scene: `charts/radar`
3. Default dataset: enabled

## Contract

1. Supported `query_id` values: `single`
2. Answer schema: `integer_count`
3. Annotation schema: `segment_set`
4. Annotation marks one segment per counted metric, connecting the two profile vertices on that metric spoke.

## Program Contract

Program: `count(filter(metrics, value(profile_a, metric) > value(profile_b, metric))); scene=radar; scope=profile_advantage_count`

Candidate set: the visible radar spokes, profile polygons, and profile labels inside the `profile_advantage_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `unspecified`.
Annotation witnesses: `unspecified` witnesses bound by `see_annotation_contract`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`
