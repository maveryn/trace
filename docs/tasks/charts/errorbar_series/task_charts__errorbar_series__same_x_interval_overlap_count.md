# `task_charts__errorbar_series__same_x_interval_overlap_count`

## Contract
1. Domain: `charts`
2. Scene id: `errorbar_series`
3. Source implementation domain/group: `charts/errorbar_series`
4. Query id: `single`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.errorbar_series.same_x_interval_overlap_count.ChartsErrorbarSeriesSameXIntervalOverlapCountTask`
2. Prompt lookup domain/group: `charts/errorbar_series`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `segment_set`.
3. Annotation is an unordered array of lower-to-upper error-bar interval segments for the counted overlapping marks at the named x-axis label.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `count(filter(series != target_series, overlaps(interval(errorbar(series,target_x)), interval(errorbar(target_series,target_x))))); output=integer_count; annotation=segment_set(matching_errorbar_interval_spans); scene=errorbar_series; scope=same_x_interval_overlap_count`

Candidate set: the visible error bars, series markers, and category labels inside the `same_x_interval_overlap_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `segment_set` witnesses bound by `segment_set(matching_errorbar_interval_spans)`. Annotation is an unordered array of lower-to-upper error-bar interval segments for the counted overlapping marks at the named x-axis label. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`, `spatial_relations`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `count(series where series != target_series and overlaps(interval(errorbar(series, target_x)), interval(errorbar(target_series, target_x))))` | `integer_count` | `segment_set` |
