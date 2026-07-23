# `task_charts__violin__support_width_extremum_label`

## Public Contract

1. Domain: `charts`
2. Scene: `violin`
3. Source file: `src/trace_tasks/tasks/charts/violin/support_width_extremum_label.py`
4. Prompt assets: `src/trace_tasks/resources/prompts/charts/violin/charts_violin_v1.json`
5. Supported sampled `query_id`: `widest_support`, `narrowest_support`

## Program Contract

Program: `arg_extreme(label, support_span(distribution(label)), direction={widest,narrowest}); output=string_label; annotation=bbox(selected_violin); scene=violin; scope=support_width_extremum_label`

Candidate set: the visible violin glyphs, summary markers, and group labels inside the `support_width_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `string_label`.
Annotation witnesses: `unspecified` witnesses bound by `bbox(selected_violin)`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `widest_support`, `narrowest_support`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Contract Notes

This task uses the current source layout. The public task file owns support-width direction selection, answer binding, annotation binding, prompt branch selection, and task-specific trace fields; scene-local shared code only provides violin sampling, rendering, prompt, and annotation primitives.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `widest_support` | `argmax(label, support_span(distribution(label)))` | `string_label` | `bbox` |
| `narrowest_support` | `argmin(label, support_span(distribution(label)))` | `string_label` | `bbox` |
