# `task_charts__parallel_coords__axis_condition_count`

## Contract
1. Domain: `charts`
2. Scene id: `parallel_coords`
3. Source implementation domain/scene: `charts/parallel_coords`
4. Supported `query_id`: `above_on_both_axes`, `below_on_both_axes`, `above_on_one_below_on_other`
5. Query ids select the threshold-comparator pair used in the prompt and program.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.parallel_coords.axis_condition_count.ChartsParallelCoordinatesAxisConditionCountTask`
2. Prompt lookup domain/scene: `charts/parallel_coords`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Program Contract

Program: `count(filter(profiles, compare(value(axis_i), threshold, comparator_i) and compare(value(axis_j), threshold, comparator_j))); comparator_pair={above_above,below_below,above_below}; axes=adjacent_pair; output=integer_count; annotation=segment_set(matching_profile_segments); scene=parallel_coords; scope=axis_condition_count`

Candidate set: the visible polylines, axes, and axis-value positions inside the `axis_condition_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `segment_set` witnesses bound by `segment_set(matching_profile_segments)`. Annotation marks each counted profile segment between the adjacent named axes as `[[x0,y0],[x1,y1]]`. Axes, labels, threshold text, and decorative context are renderer context unless explicitly requested.
Query ids: `above_on_both_axes`, `below_on_both_axes`, `above_on_one_below_on_other`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`, `logical_composition`

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `segment_set`.
3. Annotation marks each counted profile segment between the adjacent named axes as `[[x0,y0],[x1,y1]]`.
4. Axes, labels, threshold text, and decorative context are renderer context unless explicitly requested.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `above_on_both_axes` | `count.two_axis_threshold_predicate` | `integer_count` | `segment_set` |
| `below_on_both_axes` | `count.two_axis_threshold_predicate` | `integer_count` | `segment_set` |
| `above_on_one_below_on_other` | `count.two_axis_threshold_predicate` | `integer_count` | `segment_set` |
