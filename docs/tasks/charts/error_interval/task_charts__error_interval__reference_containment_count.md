# `task_charts__error_interval__reference_containment_count`

## Contract
1. Domain: `charts`
2. Scene id: `error_interval`
3. Source implementation domain/group: `charts/error_interval`
4. Query id: `single`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.error_interval.reference_containment_count.ChartsErrorIntervalReferenceContainmentCountTask`
2. Prompt lookup domain/group: `charts/error_interval`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `segment_set`.
3. Annotation marks each matching interval's lower-to-upper visual span as one segment, following the cross-domain annotation policy.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `count(filter(intervals, lower <= reference_value <= upper)); output=integer_count; annotation=segment_set(matching_interval_lower_upper_spans); scene=error_interval; scope=reference_containment_count`

Candidate set: the visible interval marks, reference lines, and category labels inside the `reference_containment_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `segment_set` witnesses bound by `segment_set(matching_interval_lower_upper_spans)`. Annotation marks each matching interval's lower-to-upper visual span as one segment, following the cross-domain annotation policy. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `count.reference_containment_count` | `integer_count` | `segment_set` |
