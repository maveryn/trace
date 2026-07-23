# `task_charts__radial_progress__progress_threshold_count`

## Contract
1. Domain: `charts`
2. Scene id: `radial_progress`
3. Task id: `task_charts__radial_progress__progress_threshold_count`
4. Objective contract: `progress_threshold_count`
5. Supported `query_id` values: `at_least_threshold_count`, `below_threshold_count`

## Program Contract

Program: `count(filter(radial_progress_widgets, progress_value(widget) >= threshold | progress_value(widget) < threshold)); scene=radial_progress; scope=progress_threshold_count`

Candidate set: the visible radial progress rings, arcs, and labels inside the `progress_threshold_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `bbox_set` witnesses bound by `see_annotation_contract`. Annotation marks one widget card bbox for each counted widget. Titles, tick marks, card decorations, and uncounted widgets are context, not annotation.
Query ids: `at_least_threshold_count`, `below_threshold_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.radial_progress.progress_threshold_count.ChartsRadialProgressThresholdCountTask`
2. Prompt bundle: `charts_radial_progress_v1`
3. Scene key: `radial_progress_scene`
4. Task key: `radial_progress_condition_count_query`
5. Query keys: `at_least_threshold_count`, `below_threshold_count`

## Annotation Contract
1. Answer schema: `integer_count`
2. Annotation schema: `bbox_set`
3. Annotation marks one widget card bbox for each counted widget.
4. Titles, tick marks, card decorations, and uncounted widgets are context, not annotation.

## Query Details

| Query id | Program argument | Answer schema | Annotation schema |
|---|---|---|---|
| `at_least_threshold_count` | `direction=at_least` | `integer_count` | `bbox_set` |
| `below_threshold_count` | `direction=below` | `integer_count` | `bbox_set` |
