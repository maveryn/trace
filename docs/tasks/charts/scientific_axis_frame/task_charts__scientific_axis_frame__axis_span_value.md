# `task_charts__scientific_axis_frame__axis_span_value`

## Contract
1. Domain: `charts`
2. Scene id: `scientific_axis_frame`
3. Source implementation domain/scene: `charts/scientific_axis_frame`
4. Supported `query_id` values: `x_axis_span_value`, `y_axis_span_value`
5. Semantic query details are recorded in `query_id` and trace params.

## Program Contract

Program: `difference(max_visible_tick(axis), min_visible_tick(axis)); output=integer_value; annotation=segment(axis_visible_span); scene=scientific_axis_frame; scope=axis_span_value`

Candidate set: the visible scientific-axis ticks, marked points, and scale labels inside the `axis_span_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `difference` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `segment` witnesses bound by `segment(axis_visible_span)`. Annotation is the segment from the smallest visible tick mark to the largest visible tick mark on the requested axis. Decorative plotted data, axis labels, and distractor text are metadata unless explicitly queried. Annotation marks the visible span on the requested axis as one segment between the minimum and maximum tick marks.
Query ids: `x_axis_span_value`, `y_axis_span_value`.

## Reasoning Operations

Families: `formula_evaluation`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.scientific_axis_frame.axis_span_value.ChartsScientificAxisFrameAxisSpanValueTask`
2. Prompt lookup domain/scene: `charts/scientific_axis_frame`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `segment`.
3. Annotation is the segment from the smallest visible tick mark to the largest visible tick mark on the requested axis.
4. Decorative plotted data, axis labels, and distractor text are metadata unless explicitly queried.
5. Annotation marks the visible span on the requested axis as one segment between the minimum and maximum tick marks.

## Query Details

| Query id | Program arguments | Answer schema | Annotation schema |
|---|---|---|---|
| `x_axis_span_value` | `axis=x` | `integer_value` | `segment` |
| `y_axis_span_value` | `axis=y` | `integer_value` | `segment` |
