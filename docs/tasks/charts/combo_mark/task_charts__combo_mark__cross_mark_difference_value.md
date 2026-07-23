# `task_charts__combo_mark__cross_mark_difference_value`

## Contract
1. Domain: `charts`
2. Scene id: `combo_mark`
3. Source implementation domain/scene: `charts/combo_mark`
4. Query ids: `primary_minus_line_at_label`, `line_minus_primary_at_label`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.combo_mark.cross_mark_difference_value.ChartsComboCrossMarkDifferenceValueTask`
2. Prompt lookup domain/scene: `charts/combo_mark`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `point_map`.
3. Annotation uses fixed keys `primary_mark` and `line_mark` for the queried category; dynamic category-label keys are not used.
4. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy.
5. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `difference(value(primary,x_label), value(line,x_label), mode={primary_minus_line,line_minus_primary}); output=integer_value; annotation=point_map(primary_mark, line_mark); scene=combo_mark; scope=cross_mark_difference_value`

Candidate set: the visible primary marks, secondary-line marks, and shared category labels inside the `cross_mark_difference_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `difference` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point_map` witnesses bound by `point_map(primary_mark, line_mark)`. Annotation uses fixed keys `primary_mark` and `line_mark` for the queried category; dynamic category-label keys are not used. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `primary_minus_line_at_label`, `line_minus_primary_at_label`.

## Reasoning Operations

Families: `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `primary_minus_line_at_label` | `numeric.cross_mark_difference` | `integer_value` | `point_map` |
| `line_minus_primary_at_label` | `numeric.cross_mark_difference` | `integer_value` | `point_map` |
