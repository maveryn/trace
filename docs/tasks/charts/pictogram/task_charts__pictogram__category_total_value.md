# `task_charts__pictogram__category_total_value`

## Contract
1. Domain: `charts`
2. Scene id: `pictogram`
3. Public task id: `task_charts__pictogram__category_total_value`
4. Supported `query_id`: `single`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.pictogram.category_total_value.ChartsPictogramCategoryTotalValueTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/pictogram/charts_pictogram_v1.json`
3. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `bbox`.
3. Annotation marks the requested category row as one `[x0, y0, x1, y1]` pixel box.

## Program Contract

Program: `value(category_total(target_category, unit_scale)); output=integer_value; annotation=bbox(target_category_row); scene=pictogram; scope=category_total_value`

Candidate set: the visible pictogram rows, repeated icons, and category labels inside the `category_total_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `value` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `bbox` witnesses bound by `bbox(target_category_row)`. Annotation marks the requested category row as one `[x0, y0, x1, y1]` pixel box.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `formula_evaluation`
