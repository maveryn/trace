# `task_charts__population_pyramid__dominant_side_count`

## Contract
1. Domain: `charts`
2. Scene id: `population_pyramid`
3. Public task id: `task_charts__population_pyramid__dominant_side_count`
4. Query ids: `left_side_greater_count`, `right_side_greater_count`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.population_pyramid.dominant_side_count.ChartsPopulationPyramidDominantSideCountTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/population_pyramid/charts_population_pyramid_v1.json`
3. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `bbox_set`.
3. Annotation marks one row bbox around the paired bars for each row where the queried side is greater than the other side.

## Program Contract

Program: `count(filter(age_group_rows, value(row, side) > value(row, other_side))); output=integer_count; annotation=bbox_set(counted_age_group_rows); scene=population_pyramid; scope=dominant_side_count`

Candidate set: the visible left/right population bars and age-group labels inside the `dominant_side_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `bbox_set` witnesses bound by `bbox_set(counted_age_group_rows)`. Annotation marks one row bbox around the paired bars for each row where the queried side is greater than the other side.
Query ids: `left_side_greater_count`, `right_side_greater_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`
