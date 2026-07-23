# `task_charts__error_interval__reference_exclusion_side_count`

## Contract
1. Domain: `charts`
2. Scene id: `error_interval`
3. Source implementation domain/group: `charts/error_interval`
4. Query ids: `entirely_above_reference_count`, `entirely_below_reference_count`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.error_interval.reference_exclusion_side_count.ChartsErrorIntervalReferenceExclusionSideCountTask`
2. Prompt lookup domain/group: `charts/error_interval`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `segment_set`.
3. Annotation marks each matching interval's lower-to-upper visual span as one segment, following the cross-domain annotation policy.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `count(filter(intervals, interval_side(reference_value) == side)); output=integer_count; annotation=segment_set(matching_interval_lower_upper_spans); scene=error_interval; scope=reference_exclusion_side_count`

Candidate set: the visible interval marks, reference lines, and category labels inside the `reference_exclusion_side_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `segment_set` witnesses bound by `segment_set(matching_interval_lower_upper_spans)`. Annotation marks each matching interval's lower-to-upper visual span as one segment, following the cross-domain annotation policy. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `entirely_above_reference_count`, `entirely_below_reference_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `entirely_above_reference_count` | `count.reference_exclusion_side_count(side=above)` | `integer_count` | `segment_set` |
| `entirely_below_reference_count` | `count.reference_exclusion_side_count(side=below)` | `integer_count` | `segment_set` |
