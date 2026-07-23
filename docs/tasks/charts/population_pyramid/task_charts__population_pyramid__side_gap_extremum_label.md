# `task_charts__population_pyramid__side_gap_extremum_label`

## Taxonomy

1. Domain: `charts`
2. Scene id: `population_pyramid`
3. Source implementation scene: `charts/population_pyramid`
4. Public task id: `task_charts__population_pyramid__side_gap_extremum_label`

## Implementation

1. Registered class: `trace_tasks.tasks.charts.population_pyramid.side_gap_extremum_label.ChartsPopulationPyramidSideGapExtremumLabelTask`
2. Prompt lookup domain/scene: `charts/population_pyramid`
3. Default dataset: enabled

## Contract

1. Query ids: `largest_side_gap_label`, `smallest_nonzero_side_gap_label`
2. Answer schema: `string_label`.
3. Annotation schema: `bbox`
4. Annotation marks one bbox around the paired left/right bars in the answer row.

## Program Contract

Program: `select_label(arg_extremum(age_group_rows, abs(left_value - right_value), rank)); scene=population_pyramid; scope=side_gap_extremum_label`

Candidate set: the visible left/right population bars and age-group labels inside the `side_gap_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `select_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `unspecified`.
Annotation witnesses: `unspecified` witnesses bound by `see_annotation_contract`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `largest_side_gap_label`, `smallest_nonzero_side_gap_label`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`
