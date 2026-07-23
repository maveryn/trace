# `task_charts__area__stacked_band_dominance_label`

## Contract
1. Domain: `charts`
2. Scene id: `area`
3. Source scene: `charts/area`
4. Query id: `single`
5. Semantic query details are recorded in trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.area.stacked_band_dominance_label.ChartsAreaStackedDominanceLabelTask`
2. Prompt lookup scene: `charts/area`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point_set`.
3. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `argmax(category_label, sum(value(category_label,x) for x in selected_x_interval)); output=string_label; annotation=point_set(winning_category_interval_marks); scene=area; scope=stacked_band_dominance_label`

Candidate set: the visible area-series bands, category labels, and interval endpoints inside the `stacked_band_dominance_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `argmax` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point_set` witnesses bound by `point_set(winning_category_interval_marks)`. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `aggregation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `selection.interval_category_sum_extremum_label` | `string_label` | `point_set` |
