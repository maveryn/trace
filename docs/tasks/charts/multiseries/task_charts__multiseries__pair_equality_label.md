# `task_charts__multiseries__pair_equality_label`

## Contract
- Domain: `charts`
- Scene id: `multiseries`
- Query id: `single`
- Answer schema: `string_label`
- Annotation schema: `point_map`
- Program contract: `select_unique(categories, value(series_a) == value(series_b))`

## Program Contract

Program: `select_label(unique(category where value(category, series_a) == value(category, series_b))); output=string_label; annotation=point_map(mark_center(answer_category, {series_a,series_b})); scene=multiseries; scope=pair_equality_label`

Candidate set: the visible series marks across shared x/category labels inside the `pair_equality_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `select_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `string_label`.
Annotation witnesses: `unspecified` witnesses bound by `point_map(mark_center(answer_category, {series_a,series_b}))`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `comparison`, `matching`

## Implementation
- Source: `src/trace_tasks/tasks/charts/multiseries/pair_equality_label.py`
- Class: `ChartsMultiseriesPairEqualityLabelTask`
- Prompt bundle: `src/trace_tasks/resources/prompts/charts/multiseries/charts_multiseries_v1.json`

## Annotation
Annotate the two queried series marks in the answer category using keys of the form `<category>:<series>`.
