# `task_charts__treemap__parent_total_extremum_label`

## Public Contract

1. Domain: `charts`
2. Scene: `treemap`
3. Source file: `src/trace_tasks/tasks/charts/treemap/parent_total_extremum_label.py`
4. Prompt assets: `src/trace_tasks/resources/prompts/charts/treemap/charts_treemap_v1.json`
5. Supported `query_id` values: `largest_parent_total`, `smallest_parent_total`

## Program Contract

Program: `select(parent where sum(value(child) for child in parent) is extremum(direction)); direction={largest,smallest}; output=string_label; annotation=bbox_set(parent_child_rectangles); scene=treemap; scope=parent_total_extremum_label`

Candidate set: the visible treemap rectangles and hierarchy labels inside the `parent_total_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `select` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `string_label`.
Annotation witnesses: `unspecified` witnesses bound by `bbox_set(parent_child_rectangles)`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `largest_parent_total`, `smallest_parent_total`.

## Reasoning Operations

Families: `ranking`, `aggregation`, `topology`

## Contract Notes

This task uses the current source layout. Query ids select the extremum direction; public task code owns the parent-total comparison, answer binding, annotation leaf ids, prompt slots, and task-specific trace fields.
