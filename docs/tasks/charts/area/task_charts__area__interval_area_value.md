# `task_charts__area__interval_area_value`

## Contract
1. Domain: `charts`
2. Scene id: `area`
3. Source scene: `charts/area`
4. Query id: `single`
5. Semantic query details are recorded in trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.area.interval_area_value.ChartsAreaIntervalAreaValueTask`
2. Prompt lookup scene: `charts/area`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `point_set`.
3. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `sum(trapezoid_area(value(series,x_i), value(series,x_{i+1})) for adjacent intervals in selected_x_interval); output=integer_value; annotation=point_set(selected_interval_marks); scene=area; scope=interval_area_value`

Candidate set: the visible area-series bands, category labels, and interval endpoints inside the `interval_area_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `sum` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point_set` witnesses bound by `point_set(selected_interval_marks)`. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `numeric.interval_trapezoid_sum` | `integer_value` | `point_set` |
