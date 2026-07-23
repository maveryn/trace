# `task_charts__errorbar_series__bound_extremum_x_label`

## Contract
1. Domain: `charts`
2. Scene id: `errorbar_series`
3. Source implementation domain/group: `charts/errorbar_series`
4. Supported `query_id` values: `highest_upper_bound_x_label`, `lowest_lower_bound_x_label`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.errorbar_series.bound_extremum_x_label.ChartsErrorbarSeriesBoundExtremumXLabelTask`
2. Prompt lookup domain/group: `charts/errorbar_series`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point`.
3. Annotation is the pixel point on the selected error-bar bound endpoint.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `arg_extreme(x_label, bound_value(errorbar_interval(target_series,x), bound={upper,lower}), direction={highest,lowest}); output=string_label; annotation=point(selected_bound_endpoint); scene=errorbar_series; scope=bound_extremum_x_label`

Candidate set: the visible error bars, series markers, and category labels inside the `bound_extremum_x_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point` witnesses bound by `point(selected_bound_endpoint)`. Annotation is the pixel point on the selected error-bar bound endpoint. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `highest_upper_bound_x_label`, `lowest_lower_bound_x_label`.

## Reasoning Operations

Families: `ranking`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `highest_upper_bound_x_label` | `argmax_x_label(upper_bound(errorbar(target_series, x_position)))` | `string_label` | `point` |
| `lowest_lower_bound_x_label` | `argmin_x_label(lower_bound(errorbar(target_series, x_position)))` | `string_label` | `point` |
