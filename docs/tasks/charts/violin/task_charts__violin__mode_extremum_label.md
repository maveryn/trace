# `task_charts__violin__mode_extremum_label`

## Public Contract

1. Domain: `charts`
2. Scene: `violin`
3. Source file: `src/trace_tasks/tasks/charts/violin/mode_extremum_label.py`
4. Prompt assets: `src/trace_tasks/resources/prompts/charts/violin/charts_violin_v1.json`
5. Supported sampled `query_id`: `highest_mode`, `lowest_mode`

## Program Contract

Program: `arg_extreme(label, mode_location(distribution(label)), direction={highest,lowest}); output=string_label; annotation=bbox(selected_violin); scene=violin; scope=mode_extremum_label`

Candidate set: the visible violin glyphs, summary markers, and group labels inside the `mode_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `string_label`.
Annotation witnesses: `unspecified` witnesses bound by `bbox(selected_violin)`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `highest_mode`, `lowest_mode`.

## Reasoning Operations

Families: `ranking`

## Contract Notes

This task uses the current source layout. The public task file owns extremum direction selection, answer binding, annotation binding, prompt branch selection, and task-specific trace fields; scene-local shared code only provides violin sampling, rendering, prompt, and annotation primitives.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `highest_mode` | `argmax(label, mode_location(distribution(label)))` | `string_label` | `bbox` |
| `lowest_mode` | `argmin(label, mode_location(distribution(label)))` | `string_label` | `bbox` |
