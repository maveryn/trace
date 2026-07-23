# `task_charts__error_interval__interval_width_rank_label`

## Contract
1. Domain: `charts`
2. Scene id: `error_interval`
3. Source implementation domain/group: `charts/error_interval`
4. Query id: sampled from `narrowest_interval_label`, `second_narrowest_interval_label`, `second_widest_interval_label`, `widest_interval_label`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.error_interval.interval_width_rank_label.ChartsErrorIntervalRelationLabelTask`
2. Prompt lookup domain/group: `charts/error_interval`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `segment`.
3. Annotation marks the selected interval's lower-to-upper visual span as one segment, following the cross-domain annotation policy.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `label(select_rank(intervals, key=upper-lower, order={widest,narrowest}, rank={1,2})); output=string_label; annotation=segment(selected_interval_lower_upper_span); scene=error_interval; scope=interval_width_rank_label`

Candidate set: the visible interval marks, reference lines, and category labels inside the `interval_width_rank_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `segment` witnesses bound by `segment(selected_interval_lower_upper_span)`. Annotation marks the selected interval's lower-to-upper visual span as one segment, following the cross-domain annotation policy. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `narrowest_interval_label`, `second_narrowest_interval_label`, `second_widest_interval_label`, `widest_interval_label`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `narrowest_interval_label` | `selection.interval_width_rank_label(order=narrowest, rank=1)` | `string_label` | `segment` |
| `second_narrowest_interval_label` | `selection.interval_width_rank_label(order=narrowest, rank=2)` | `string_label` | `segment` |
| `second_widest_interval_label` | `selection.interval_width_rank_label(order=widest, rank=2)` | `string_label` | `segment` |
| `widest_interval_label` | `selection.interval_width_rank_label(order=widest, rank=1)` | `string_label` | `segment` |
