# `task_charts__pictogram__threshold_count`

## Contract
1. Domain: `charts`
2. Scene id: `pictogram`
3. Public task id: `task_charts__pictogram__threshold_count`
4. Supported `query_id`: `greater_than_threshold`, `less_than_threshold`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.pictogram.threshold_count.ChartsPictogramThresholdCountTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/pictogram/charts_pictogram_v1.json`
3. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `bbox_set`.
3. Annotation marks every matching category row as an unordered set of row bboxes.

## Program Contract

Program: `count(filter(categories, compare(category_total(category, unit_scale), threshold, comparator={greater_than,less_than}))); output=integer_count; annotation=bbox_set(matching_category_rows); scene=pictogram; scope=threshold_count`

Candidate set: the visible pictogram rows, repeated icons, and category labels inside the `threshold_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `bbox_set` witnesses bound by `bbox_set(matching_category_rows)`. Annotation marks every matching category row as an unordered set of row bboxes.
Query ids: `greater_than_threshold`, `less_than_threshold`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`, `formula_evaluation`

## Query Details

| Query id | Comparator | Answer schema | Annotation schema |
|---|---|---|---|
| `greater_than_threshold` | `greater_than` | `integer_count` | `bbox_set` |
| `less_than_threshold` | `less_than` | `integer_count` | `bbox_set` |
