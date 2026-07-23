# `task_charts__table__sum_absolute_differences_between_rows_over_year_interval`

## Public Contract

1. Domain: `charts`
2. Scene: `table`
3. Source file: `src/trace_tasks/tasks/charts/table/sum_absolute_differences_between_rows_over_year_interval.py`
4. Prompt assets: `src/trace_tasks/resources/prompts/charts/table/charts_table_counting_v1.json`, `src/trace_tasks/resources/prompts/charts/table/charts_table_ranking_v1.json`, `src/trace_tasks/resources/prompts/charts/table/charts_table_statistics_v1.json`, and `src/trace_tasks/resources/prompts/charts/table/charts_table_temporal_v1.json`
5. Query ids: `single`

## Program Contract

Program: `sum(abs(value(row_a, year) - value(row_b, year)) for year in interval); output=integer_value; annotation=bbox_set(two_row_interval_cells); scene=table; scope=sum_absolute_differences_between_rows_over_year_interval`

Candidate set: the visible table cells with row and column labels inside the `sum_absolute_differences_between_rows_over_year_interval` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `sum` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `integer_value`.
Annotation witnesses: `unspecified` witnesses bound by `bbox_set(two_row_interval_cells)`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `formula_evaluation`

## Contract Notes

This task uses the current source layout. Scene-local reusable code lives under `src/trace_tasks/tasks/charts/table/shared/`; public task files own objective logic, query selection, answer binding, and annotation binding.
