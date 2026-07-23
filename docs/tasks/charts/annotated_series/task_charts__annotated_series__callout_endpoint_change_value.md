# `task_charts__annotated_series__callout_endpoint_change_value`

## Contract
1. Domain: `charts`
2. Scene id: `annotated_series`
3. Source package: `charts/annotated_series`
4. Query id: `single`
5. Semantic query details are recorded in `query_id` and trace params.
6. Answer schema: `integer_value`
7. Annotation schema: `point_map`
8. Endpoint side is an internal generation axis recorded as `endpoint_side=first|last`, not a query id.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.annotated_series.callout_endpoint_change_value.ChartsAnnotatedSeriesCalloutEndpointChangeValueTask`
2. Prompt lookup: `src/trace_tasks/resources/prompts/charts/annotated_series/charts_annotated_series_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `point_map`.
3. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `absolute_difference(value(mark(role=callout_mark)), value(mark(role=endpoint_mark))); output=integer_value; annotation=point_map(callout_mark, endpoint_mark); generation_metadata=endpoint_side:{first,last}; scene=annotated_series; scope=callout_endpoint_change_value`

Candidate set: the visible series marks and callout annotations inside the `callout_endpoint_change_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `absolute_difference` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point_map` witnesses bound by `point_map(callout_mark, endpoint_mark)`. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `numeric.callout_endpoint_difference` | `integer_value` | `point_map` |
