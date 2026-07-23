# `task_charts__violin__modality_label`

## Public Contract

1. Domain: `charts`
2. Scene: `violin`
3. Source file: `src/trace_tasks/tasks/charts/violin/modality_label.py`
4. Prompt assets: `src/trace_tasks/resources/prompts/charts/violin/charts_violin_v1.json`
5. Supported sampled `query_id`: `single`

## Program Contract

Program: `select_unique(label where modality(distribution(label)) == bimodal); output=string_label; annotation=bbox(selected_violin); scene=violin; scope=modality_label`

Candidate set: the visible violin glyphs, summary markers, and group labels inside the `modality_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `select_unique` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `string_label`.
Annotation witnesses: `unspecified` witnesses bound by `bbox(selected_violin)`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `matching`

## Contract Notes

This task uses the current source layout. The public task file owns the bimodal objective, answer binding, annotation binding, prompt branch selection, and task-specific trace fields; scene-local shared code only provides violin sampling, rendering, prompt, and annotation primitives.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `select_unique(label where modality(distribution(label)) == bimodal)` | `string_label` | `bbox` |
