# `task_charts__scientific_axis_frame__tick_spacing_value`

## Contract
1. Domain: `charts`
2. Scene id: `scientific_axis_frame`
3. Source implementation domain/scene: `charts/scientific_axis_frame`
4. Supported `query_id` values: `x_first_tick_spacing_value`, `x_last_tick_spacing_value`, `y_first_tick_spacing_value`, `y_last_tick_spacing_value`
5. Semantic query details are recorded in `query_id` and trace params.

## Program Contract

Program: `difference(second_tick(axis,pair_position), first_tick(axis,pair_position)); output=integer_value; annotation=segment(tick_interval); scene=scientific_axis_frame; scope=tick_spacing_value`

Candidate set: the visible scientific-axis ticks, marked points, and scale labels inside the `tick_spacing_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `difference` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `segment` witnesses bound by `segment(tick_interval)`. Annotation is the segment between the two tick marks in the requested first or last visible adjacent interval. Decorative plotted data, axis labels, and distractor text are metadata unless explicitly queried. Annotation marks the requested first or last visible adjacent tick interval as one axis segment.
Query ids: `x_first_tick_spacing_value`, `x_last_tick_spacing_value`, `y_first_tick_spacing_value`, `y_last_tick_spacing_value`.

## Reasoning Operations

Families: `formula_evaluation`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.scientific_axis_frame.tick_spacing_value.ChartsScientificAxisFrameTickSpacingValueTask`
2. Prompt lookup domain/scene: `charts/scientific_axis_frame`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `segment`.
3. Annotation is the segment between the two tick marks in the requested first or last visible adjacent interval.
4. Decorative plotted data, axis labels, and distractor text are metadata unless explicitly queried.
5. Annotation marks the requested first or last visible adjacent tick interval as one axis segment.

## Query Details

| Query id | Program arguments | Answer schema | Annotation schema |
|---|---|---|---|
| `x_first_tick_spacing_value` | `axis=x; pair_position=first` | `integer_value` | `segment` |
| `x_last_tick_spacing_value` | `axis=x; pair_position=last` | `integer_value` | `segment` |
| `y_first_tick_spacing_value` | `axis=y; pair_position=first` | `integer_value` | `segment` |
| `y_last_tick_spacing_value` | `axis=y; pair_position=last` | `integer_value` | `segment` |
