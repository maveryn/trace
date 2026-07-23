# `task_charts__parallel_coords__axis_delta_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `parallel_coords`
3. Source implementation domain/scene: `charts/parallel_coords`
4. Supported `query_id`: `largest_increase_between_axes`, `largest_decrease_between_axes`, `largest_absolute_change_between_axes`
5. Query ids select the change mode used in the prompt and program.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.parallel_coords.axis_delta_extremum_label.ChartsParallelCoordinatesAxisDeltaExtremumLabelTask`
2. Prompt lookup domain/scene: `charts/parallel_coords`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Program Contract

Program: `select_label(arg_extreme(profiles, change(value(axis_i), value(axis_j), mode={increase,decrease,absolute}), direction=largest)); axes=adjacent_pair; output=string_label; annotation=segment(answer_profile_segment); scene=parallel_coords; scope=axis_delta_extremum_label`

Candidate set: the visible polylines, axes, and axis-value positions inside the `axis_delta_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `select_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `segment` witnesses bound by `segment(answer_profile_segment)`. Annotation marks the answer profile segment between the adjacent named axes as `[[x0,y0],[x1,y1]]`. Axes, labels, threshold text, and decorative context are renderer context unless explicitly requested.
Query ids: `largest_increase_between_axes`, `largest_decrease_between_axes`, `largest_absolute_change_between_axes`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `segment`.
3. Annotation marks the answer profile segment between the adjacent named axes as `[[x0,y0],[x1,y1]]`.
4. Axes, labels, threshold text, and decorative context are renderer context unless explicitly requested.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `largest_increase_between_axes` | `select.extreme_axis_change_label` | `string_label` | `segment` |
| `largest_decrease_between_axes` | `select.extreme_axis_change_label` | `string_label` | `segment` |
| `largest_absolute_change_between_axes` | `select.extreme_axis_change_label` | `string_label` | `segment` |
