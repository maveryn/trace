# `task_charts__combo_mark__conditioned_primary_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `combo_mark`
3. Source implementation domain/scene: `charts/combo_mark`
4. Query ids: `max_primary_where_line_below_threshold`, `min_primary_where_line_below_threshold`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.combo_mark.conditioned_primary_extremum_label.ChartsComboConditionedPrimaryExtremumLabelTask`
2. Prompt lookup domain/scene: `charts/combo_mark`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point_map`.
3. Annotation uses fixed keys `primary_mark` and `line_mark` for the answer category; dynamic category-label keys are not used.
4. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy.
5. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `arg_extreme(x_label, value(primary,x_label), filter=value(line,x_label)<threshold, direction={max,min}); output=string_label; annotation=point_map(primary_mark, line_mark); scene=combo_mark; scope=conditioned_primary_extremum_label`

Candidate set: the visible primary marks, secondary-line marks, and shared category labels inside the `conditioned_primary_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point_map` witnesses bound by `point_map(primary_mark, line_mark)`. Annotation uses fixed keys `primary_mark` and `line_mark` for the answer category; dynamic category-label keys are not used. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `max_primary_where_line_below_threshold`, `min_primary_where_line_below_threshold`.

## Reasoning Operations

Families: `filtering`, `comparison`, `ranking`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `max_primary_where_line_below_threshold` | `selection.filtered_primary_extremum_label` | `string_label` | `point_map` |
| `min_primary_where_line_below_threshold` | `selection.filtered_primary_extremum_label` | `string_label` | `point_map` |
