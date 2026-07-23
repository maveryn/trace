# `task_charts__multiseries__category_total_extremum_label`

## Contract
- Domain: `charts`
- Scene id: `multiseries`
- Query ids: `largest_category_total_label`, `smallest_category_total_label`
- Answer schema: `string_label`
- Annotation schema: `point_map`
- Program contract: `arg_extreme(categories, sum(series_values), extremum_direction)`

## Program Contract

Program: `select_label(arg_extreme(categories, sum(series_values(category)), direction={largest,smallest})); output=string_label; annotation=point_map(mark_center(answer_category, all_series)); scene=multiseries; scope=category_total_extremum_label`

Candidate set: the visible series marks across shared x/category labels inside the `category_total_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `select_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `string_label`.
Annotation witnesses: `unspecified` witnesses bound by `point_map(mark_center(answer_category, all_series))`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `largest_category_total_label`, `smallest_category_total_label`.

## Reasoning Operations

Families: `ranking`, `aggregation`

## Implementation
- Source: `src/trace_tasks/tasks/charts/multiseries/category_total_extremum_label.py`
- Class: `ChartsMultiseriesCategoryTotalExtremumLabelTask`
- Prompt bundle: `src/trace_tasks/resources/prompts/charts/multiseries/charts_multiseries_v1.json`

## Annotation
Annotate every series mark in the answer category using keys of the form `<category>:<series>`.
