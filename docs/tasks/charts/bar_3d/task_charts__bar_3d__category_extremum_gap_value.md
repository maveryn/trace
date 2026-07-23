# `task_charts__bar_3d__category_extremum_gap_value`

## Contract
1. Domain: `charts`
2. Scene id: `bar_3d`
3. Source implementation scene: `charts/bar_3d`
4. Query id: `single`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.bar_3d.category_extremum_gap_value.ChartsThreeDBarCategoryExtremumGapValueTask`
2. Prompt lookup scene: `charts/bar_3d`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `point_map`.
3. Annotation marks the top-center points of the highest and lowest bars in the requested category using keys `highest` and `lowest`.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `difference(max(value(category_label,series) for series in visible_series), min(value(category_label,series) for series in visible_series)); output=integer_value; annotation=point_map(highest, lowest); scene=bar_3d; scope=category_extremum_gap_value`

Candidate set: the visible 3D bars grouped by category and series inside the `category_extremum_gap_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `difference` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point_map` witnesses bound by `point_map(highest, lowest)`. Annotation marks the top-center points of the highest and lowest bars in the requested category using keys `highest` and `lowest`. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `numeric.category_extremum_gap` | `integer_value` | `point_map` |
