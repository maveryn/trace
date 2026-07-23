# `task_charts__population_pyramid__age_group_threshold_count`

## Taxonomy

1. Domain: `charts`
2. Scene id: `population_pyramid`
3. Source implementation scene: `charts/population_pyramid`
4. Public task id: `task_charts__population_pyramid__age_group_threshold_count`

## Implementation

1. Registered class: `trace_tasks.tasks.charts.population_pyramid.age_group_threshold_count.ChartsPopulationPyramidAgeGroupThresholdCountTask`
2. Prompt lookup domain/scene: `charts/population_pyramid`
3. Default dataset: enabled

## Contract

1. Query ids: `left_side_at_least_threshold_count`, `left_side_at_most_threshold_count`, `right_side_at_least_threshold_count`, `right_side_at_most_threshold_count`, `combined_total_at_least_threshold_count`, `combined_total_at_most_threshold_count`
2. Answer schema: `integer_count`.
3. Annotation schema: `bbox_set`
4. Annotation marks one bbox around the paired left/right bars for each counted age-group row.

## Program Contract

Program: `count(filter(age_group_rows, compare(metric(row, side), threshold, relation))); scene=population_pyramid; scope=age_group_threshold_count`

Candidate set: the visible left/right population bars and age-group labels inside the `age_group_threshold_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `unspecified`.
Annotation witnesses: `unspecified` witnesses bound by `see_annotation_contract`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `left_side_at_least_threshold_count`, `left_side_at_most_threshold_count`, `right_side_at_least_threshold_count`, `right_side_at_most_threshold_count`, `combined_total_at_least_threshold_count`, `combined_total_at_most_threshold_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`
