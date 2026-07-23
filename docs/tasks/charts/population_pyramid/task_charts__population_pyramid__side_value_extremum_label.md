# `task_charts__population_pyramid__side_value_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `population_pyramid`
3. Public task id: `task_charts__population_pyramid__side_value_extremum_label`
4. Query ids: `left_side_largest_value_label`, `left_side_smallest_value_label`, `right_side_largest_value_label`, `right_side_smallest_value_label`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.population_pyramid.side_value_extremum_label.ChartsPopulationPyramidSideValueExtremumLabelTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/population_pyramid/charts_population_pyramid_v1.json`
3. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `bbox`.
3. Annotation marks the selected side-specific bar as one `[x0, y0, x1, y1]` pixel box.

## Program Contract

Program: `select_label(arg_extremum(age_group_rows, value(row, side), direction)); output=string_label; annotation=bbox(answer_side_bar); scene=population_pyramid; scope=side_value_extremum_label`

Candidate set: the visible left/right population bars and age-group labels inside the `side_value_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `select_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `bbox` witnesses bound by `bbox(answer_side_bar)`. Annotation marks the selected side-specific bar as one `[x0, y0, x1, y1]` pixel box.
Query ids: `left_side_largest_value_label`, `left_side_smallest_value_label`, `right_side_largest_value_label`, `right_side_smallest_value_label`.

## Reasoning Operations

Families: `ranking`
