# `task_charts__table__categorical_value_count`

## Public Contract

1. Domain: `charts`
2. Scene: `table`
3. Source file: `src/trace_tasks/tasks/charts/table/categorical_value_count.py`
4. Prompt assets: `src/trace_tasks/resources/prompts/charts/table/charts_table_counting_v1.json`, `src/trace_tasks/resources/prompts/charts/table/charts_table_ranking_v1.json`, `src/trace_tasks/resources/prompts/charts/table/charts_table_statistics_v1.json`, and `src/trace_tasks/resources/prompts/charts/table/charts_table_temporal_v1.json`
5. Query ids: `single`

## Program Contract

Program: `count(row where category(column) == target_category); output=integer_count; annotation=bbox_set(matching_category_cells); scene=table; scope=categorical_value_count`

Candidate set: the visible table cells with row and column labels inside the `categorical_value_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `integer_count`.
Annotation witnesses: `unspecified` witnesses bound by `bbox_set(matching_category_cells)`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Contract Notes

This task uses the current source layout. Scene-local reusable code lives under `src/trace_tasks/tasks/charts/table/shared/`; public task files own objective logic, query selection, answer binding, and annotation binding.
