# `task_charts__treemap__group_total_value`

## Public Contract

1. Domain: `charts`
2. Scene: `treemap`
3. Source file: `src/trace_tasks/tasks/charts/treemap/group_total_value.py`
4. Prompt assets: `src/trace_tasks/resources/prompts/charts/treemap/charts_treemap_v1.json`
5. Supported sampled `query_id`: `single`

## Program Contract

Program: `sum(value(child) for child in parent); output=integer_value; annotation=bbox_set(parent_child_rectangles); scene=treemap; scope=group_total_value`

Candidate set: the visible treemap rectangles and hierarchy labels inside the `group_total_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `sum` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `integer_value`.
Annotation witnesses: `unspecified` witnesses bound by `bbox_set(parent_child_rectangles)`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `topology`

## Contract Notes

This task uses the current source layout. The public task file owns target parent selection, answer binding, annotation binding, and prompt slots; scene-local shared code only provides treemap data, rendering, prompt, and projection primitives.
