# `task_charts__table__filtered_column_mean`

## Public Contract

1. Domain: `charts`
2. Scene: `table`
3. Source file: `src/trace_tasks/tasks/charts/table/filtered_column_mean.py`
4. Prompt assets: `src/trace_tasks/resources/prompts/charts/table/charts_table_counting_v1.json`, `src/trace_tasks/resources/prompts/charts/table/charts_table_ranking_v1.json`, `src/trace_tasks/resources/prompts/charts/table/charts_table_statistics_v1.json`, and `src/trace_tasks/resources/prompts/charts/table/charts_table_temporal_v1.json`
5. Query ids: `above_threshold_filtered_mean`, `below_threshold_filtered_mean`, `interval_filtered_mean`

## Program Contract

Program: `mean(value(target_column) for row where filter_column satisfies predicate); output=integer_value; annotation=bbox_set_map(filter_cells,target_cells); scene=table; scope=filtered_column_mean`

Candidate set: the visible table cells with row and column labels inside the `filtered_column_mean` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `mean` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `integer_value`.
Annotation witnesses: `unspecified` witnesses bound by `bbox_set_map(filter_cells,target_cells)`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `above_threshold_filtered_mean`, `below_threshold_filtered_mean`, `interval_filtered_mean`.

## Reasoning Operations

Families: `filtering`, `comparison`, `aggregation`

## Contract Notes

This task uses the current source layout. Scene-local reusable code lives under `src/trace_tasks/tasks/charts/table/shared/`; public task files own objective logic, query selection, answer binding, and annotation binding.
