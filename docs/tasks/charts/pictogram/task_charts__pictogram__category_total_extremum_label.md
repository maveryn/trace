# `task_charts__pictogram__category_total_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `pictogram`
3. Public task id: `task_charts__pictogram__category_total_extremum_label`
4. Query ids: `largest_total_category_label`, `smallest_total_category_label`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.pictogram.category_total_extremum_label.ChartsPictogramCategoryTotalExtremumLabelTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/pictogram/charts_pictogram_v1.json`
3. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `bbox`.
3. Annotation marks the answer category row as one `[x0, y0, x1, y1]` pixel box.

## Program Contract

Program: `select_label(arg_extremum(categories, category_total(category, unit_scale), direction)); output=string_label; annotation=bbox(answer_category_row); scene=pictogram; scope=category_total_extremum_label`

Candidate set: the visible pictogram rows, repeated icons, and category labels inside the `category_total_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `select_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `bbox` witnesses bound by `bbox(answer_category_row)`. Annotation marks the answer category row as one `[x0, y0, x1, y1]` pixel box.
Query ids: `largest_total_category_label`, `smallest_total_category_label`.

## Reasoning Operations

Families: `filtering`, `ranking`, `formula_evaluation`
