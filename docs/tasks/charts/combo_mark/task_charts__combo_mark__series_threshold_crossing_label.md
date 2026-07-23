# `task_charts__combo_mark__series_threshold_crossing_label`

## Contract
1. Domain: `charts`
2. Scene id: `combo_mark`
3. Source implementation domain/scene: `charts/combo_mark`
4. Query ids: `primary_first_above_threshold_label`, `primary_first_below_threshold_label`, `line_first_above_threshold_label`, `line_first_below_threshold_label`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.combo_mark.series_threshold_crossing_label.ChartsComboSeriesThresholdCrossingLabelTask`
2. Prompt lookup domain/scene: `charts/combo_mark`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point`.
3. Annotation marks the target-series mark at the answer category only.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `first_x_label(series, threshold, crossing_direction={above,below}); output=string_label; annotation=point(answer_mark); scene=combo_mark; scope=series_threshold_crossing_label`

Candidate set: the visible primary marks, secondary-line marks, and shared category labels inside the `series_threshold_crossing_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `first_x_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point` witnesses bound by `point(answer_mark)`. Annotation marks the target-series mark at the answer category only. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `primary_first_above_threshold_label`, `primary_first_below_threshold_label`, `line_first_above_threshold_label`, `line_first_below_threshold_label`.

## Reasoning Operations

Families: `filtering`, `comparison`, `ranking`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `primary_first_above_threshold_label` | `selection.threshold_crossing_x_label` | `string_label` | `point` |
| `primary_first_below_threshold_label` | `selection.threshold_crossing_x_label` | `string_label` | `point` |
| `line_first_above_threshold_label` | `selection.threshold_crossing_x_label` | `string_label` | `point` |
| `line_first_below_threshold_label` | `selection.threshold_crossing_x_label` | `string_label` | `point` |
