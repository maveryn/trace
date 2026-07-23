# `task_charts__multiseries__series_rank_at_category_label`

## Contract
- Domain: `charts`
- Scene id: `multiseries`
- Query ids: `largest_series_at_category_label`, `smallest_series_at_category_label`
- Answer schema: `string_label`
- Annotation schema: `point_map`
- Program contract: `arg_extreme(series, value_at(category), extremum_direction)`

## Program Contract

Program: `select_label(arg_extreme(series, value(target_category, series), direction={largest,smallest})); output=string_label; annotation=point_map(mark_center(target_category, all_series)); scene=multiseries; scope=series_rank_at_category_label`

Candidate set: the visible series marks across shared x/category labels inside the `series_rank_at_category_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `select_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `string_label`.
Annotation witnesses: `unspecified` witnesses bound by `point_map(mark_center(target_category, all_series))`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `largest_series_at_category_label`, `smallest_series_at_category_label`.

## Reasoning Operations

Families: `ranking`

## Implementation
- Source: `src/trace_tasks/tasks/charts/multiseries/series_rank_at_category_label.py`
- Class: `ChartsMultiseriesSeriesRankAtCategoryLabelTask`
- Prompt bundle: `src/trace_tasks/resources/prompts/charts/multiseries/charts_multiseries_v1.json`

## Annotation
Annotate every series mark in the queried category using keys of the form `<category>:<series>`.
