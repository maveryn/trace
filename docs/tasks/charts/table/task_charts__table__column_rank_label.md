# `task_charts__table__column_rank_label`

## Public Contract

1. Domain: `charts`
2. Scene: `table`
3. Source file: `src/trace_tasks/tasks/charts/table/column_rank_label.py`
4. Prompt assets: `src/trace_tasks/resources/prompts/charts/table/charts_table_counting_v1.json`, `src/trace_tasks/resources/prompts/charts/table/charts_table_ranking_v1.json`, `src/trace_tasks/resources/prompts/charts/table/charts_table_statistics_v1.json`, and `src/trace_tasks/resources/prompts/charts/table/charts_table_temporal_v1.json`
5. Query ids: `highest_rank_in_column`, `lowest_rank_in_column`

## Program Contract

Program: `rank(rows by value(column), direction, rank_k); output=row_label; annotation=bbox(answer_value_cell); scene=table; scope=column_rank_label`

Candidate set: the visible table cells with row and column labels inside the `column_rank_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `rank` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `row_label`.
Annotation witnesses: `unspecified` witnesses bound by `bbox(answer_value_cell)`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `highest_rank_in_column`, `lowest_rank_in_column`.

## Reasoning Operations

Families: `ranking`

## Contract Notes

This task uses the current source layout. Scene-local reusable code lives under `src/trace_tasks/tasks/charts/table/shared/`; public task files own objective logic, query selection, answer binding, and annotation binding.
