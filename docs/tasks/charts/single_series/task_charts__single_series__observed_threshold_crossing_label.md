# `task_charts__single_series__observed_threshold_crossing_label`

## Contract
1. Domain: `charts`
2. Scene id: `single_series`
3. Source implementation: `src/trace_tasks/tasks/charts/single_series/observed_threshold_crossing_label.py`
4. Public task id: `task_charts__single_series__observed_threshold_crossing_label`
5. Supported `query_id` values: `observed_above_threshold_crossing_label`, `observed_below_threshold_crossing_label`
6. Query ids are internal replay metadata; scene style, label pool, mark count, and context mode are generation metadata.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.single_series.observed_threshold_crossing_label.ChartsTrendObservedThresholdCrossingLabelTask`
2. Prompt lookup: `src/trace_tasks/resources/prompts/charts/single_series/charts_trend_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Program Contract

Program: `first_label(filter_prefix(sequence(values), value(label) comparison threshold)); output=string_label; annotation=point(crossing_mark); scene=single_series; scope=observed_threshold_crossing_label`

Candidate set: the visible marks in the ordered single-series chart inside the `observed_threshold_crossing_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `first_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point` witnesses bound by `point(crossing_mark)`. Annotation marks the first observed threshold-crossing mark. Axes, legend, titles, captions, decorative context, and distractor text are context unless the task explicitly asks for them as annotation.
Query ids: `observed_above_threshold_crossing_label`, `observed_below_threshold_crossing_label`.

## Reasoning Operations

Families: `filtering`, `comparison`, `ranking`

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point`.
3. Annotation marks the first observed threshold-crossing mark.
4. Axes, legend, titles, captions, decorative context, and distractor text are context unless the task explicitly asks for them as annotation.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `observed_above_threshold_crossing_label` | `first_label.observed_crosses_above_threshold` | `string_label` | `point` |
| `observed_below_threshold_crossing_label` | `first_label.observed_crosses_below_threshold` | `string_label` | `point` |
