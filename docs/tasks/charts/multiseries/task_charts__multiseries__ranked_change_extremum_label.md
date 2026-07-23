# `task_charts__multiseries__ranked_change_extremum_label`

## Contract
- Domain: `charts`
- Scene id: `multiseries`
- Query ids: `largest_increase_label`, `largest_decrease_label`, `largest_absolute_gap_label`, `smallest_absolute_gap_label`
- Answer schema: `string_label`
- Annotation schema: `point_map`
- Program contract: `arg_extreme(categories, change(value(series_a), value(series_b), change_measure), direction)`

## Program Contract

Program: `select_label(arg_extreme(categories, change(value(category, series_a), value(category, series_b), measure), direction)); output=string_label; annotation=point_map(mark_center(answer_category, {series_a,series_b})); scene=multiseries; scope=ranked_change_extremum_label`

Candidate set: the visible series marks across shared x/category labels inside the `ranked_change_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `select_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `string_label`.
Annotation witnesses: `unspecified` witnesses bound by `point_map(mark_center(answer_category, {series_a,series_b}))`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `largest_increase_label`, `largest_decrease_label`, `largest_absolute_gap_label`, `smallest_absolute_gap_label`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Implementation
- Source: `src/trace_tasks/tasks/charts/multiseries/ranked_change_extremum_label.py`
- Class: `ChartsMultiseriesRankedChangeExtremumTask`
- Prompt bundle: `src/trace_tasks/resources/prompts/charts/multiseries/charts_multiseries_v1.json`

## Annotation
Annotate the two queried series marks in the answer category using keys of the form `<category>:<series>`.
