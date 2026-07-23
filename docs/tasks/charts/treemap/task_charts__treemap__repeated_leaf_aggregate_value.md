# `task_charts__treemap__repeated_leaf_aggregate_value`

## Public Contract

1. Domain: `charts`
2. Scene: `treemap`
3. Source file: `src/trace_tasks/tasks/charts/treemap/repeated_leaf_aggregate_value.py`
4. Prompt assets: `src/trace_tasks/resources/prompts/charts/treemap/charts_treemap_v1.json`
5. Supported `query_id` values: `treemap_repeated_leaf_sum_value`, `treemap_repeated_leaf_average_value`

## Program Contract

Program: `aggregate(value(child_label across parents), operation=sum_or_average); output=integer_value; annotation=bbox_set(repeated_child_rectangles); scene=treemap; scope=repeated_leaf_aggregate_value`

Candidate set: the visible treemap rectangles and hierarchy labels inside the `repeated_leaf_aggregate_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `aggregate` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `integer_value`.
Annotation witnesses: `unspecified` witnesses bound by `bbox_set(repeated_child_rectangles)`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `treemap_repeated_leaf_sum_value`, `treemap_repeated_leaf_average_value`.

## Reasoning Operations

Families: `aggregation`, `topology`

## Contract Notes

This task uses the current source layout. Query ids select the aggregate operation; target child-label sampling remains objective-owned in the public task file and is recorded in trace params.
