# `task_charts__pictogram__group_difference_value`

## Contract
1. Domain: `charts`
2. Scene id: `pictogram`
3. Public task id: `task_charts__pictogram__group_difference_value`
4. Supported `query_id`: `single`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.pictogram.group_difference_value.ChartsPictogramGroupDifferenceValueTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/pictogram/charts_pictogram_v1.json`
3. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `bbox_map`.
3. Annotation maps each compared category label to the bbox around that category row.

## Program Contract

Program: `difference(value(category_total(category_a, unit_scale)), value(category_total(category_b, unit_scale)), mode=absolute); output=integer_value; annotation=bbox_map(category_a_row,category_b_row); scene=pictogram; scope=group_difference_value`

Candidate set: the visible pictogram rows, repeated icons, and category labels inside the `group_difference_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `difference` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `bbox_map` witnesses bound by `bbox_map(category_a_row,category_b_row)`. Annotation maps each compared category label to the bbox around that category row.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `formula_evaluation`
