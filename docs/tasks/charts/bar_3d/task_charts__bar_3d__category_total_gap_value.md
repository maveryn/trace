# `task_charts__bar_3d__category_total_gap_value`

## Contract
1. Domain: `charts`
2. Scene id: `bar_3d`
3. Source implementation scene: `charts/bar_3d`
4. Query id: `single`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.bar_3d.category_total_gap_value.ChartsThreeDBarCategoryTotalGapValueTask`
2. Prompt lookup scene: `charts/bar_3d`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `point_set_map`.
3. Annotation is keyed by the two compared category labels; each key maps to the top-center points for all series bars in that category.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `difference(sum(value(category_a,series) for series in visible_series), sum(value(category_b,series) for series in visible_series)); output=integer_value; annotation=point_set_map(category_a_total_bars, category_b_total_bars); scene=bar_3d; scope=category_total_gap_value`

Candidate set: the visible 3D bars grouped by category and series inside the `category_total_gap_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `difference` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point_set_map` witnesses bound by `point_set_map(category_a_total_bars, category_b_total_bars)`. Annotation is keyed by the two compared category labels; each key maps to the top-center points for all series bars in that category. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `numeric.category_total_difference` | `integer_value` | `point_set_map` |
