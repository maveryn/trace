# `task_charts__dashboard__global_value_extremum_category_label`

## Contract
1. Domain: `charts`
2. Scene id: `dashboard`
3. Source implementation domain/group: `charts/dashboard`
4. Query ids: `global_maximum_value_category_label`, `global_minimum_value_category_label`
5. Semantic query details are recorded in trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.dashboard.global_value_extremum_category_label.ChartsDashboardGlobalValueExtremumCategoryLabelTask`
2. Prompt lookup domain/group: `charts/dashboard`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point`.
3. The annotation point marks the single global maximum or minimum value mark whose category label is the answer.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `label(category_at(arg_extreme(mark, value(mark), direction={maximum,minimum}, over=all_dashboard_category_marks))); output=string_label; annotation=point(answer_mark); scene=dashboard; scope=global_value_extremum_category_label`

Candidate set: the visible dashboard panels, linked marks, and panel labels inside the `global_value_extremum_category_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point` witnesses bound by `point(answer_mark)`. The annotation point marks the single global maximum or minimum value mark whose category label is the answer. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `global_maximum_value_category_label`, `global_minimum_value_category_label`.

## Reasoning Operations

Families: `ranking`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `global_maximum_value_category_label` | `selection.global_value_extremum_category_label` | `string_label` | `point` |
| `global_minimum_value_category_label` | `selection.global_value_extremum_category_label` | `string_label` | `point` |
